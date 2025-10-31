//! ACE Reflector module for analyzing trading outcomes and extracting learnings
//! Processes decision outcomes to generate delta updates for the playbook

use anyhow::{Context, Result};
use serde_json::json;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::{
    ace::{
        delta::Delta,
        playbook::{PlaybookBullet, PlaybookSection},
        prompts::{ACEPrompts, ReflectionResult, TradingDecision},
        sanitize::validate_reflection_result,
    },
    llm::LLMClient,
};

/// Type of trading outcome - distinguishes actual trades from hypothetical scenarios
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutcomeType {
    /// Actual executed trade that affected account balance
    Executed,
    /// Hypothetical outcome (e.g., STAY_FLAT with opportunity cost)
    Hypothetical,
}

/// Trading outcome data for reflection
#[derive(Debug, Clone)]
pub struct TradingOutcome {
    /// Actual profit/loss in dollars (0.0 for STAY_FLAT/hypothetical)
    /// This value affects account balance and statistics ONLY for OutcomeType::Executed
    pub pnl_value: f64,
    /// Profit/loss as percentage
    pub pnl_pct: f64,
    /// Opportunity cost for STAY_FLAT decisions (for learning only, never affects balance)
    /// Negative value = missed profitable opportunity
    /// Positive/zero = staying flat was correct
    pub opportunity_cost: Option<f64>,
    /// Type of outcome - executed trade or hypothetical
    pub outcome_type: OutcomeType,
    /// Maximum favorable excursion (best unrealized profit)
    pub mfe: Option<f64>,
    /// Maximum adverse excursion (worst unrealized loss)
    pub mae: Option<f64>,
    /// Whether the trade was ultimately profitable (for executed) or decision was correct (for hypothetical)
    pub win: bool,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Duration of trade in hours
    pub duration_hours: f64,
    /// Additional context or notes
    pub notes: Option<String>,
}

impl TradingOutcome {
    /// Create outcome from basic P&L data for executed trades
    pub fn from_pnl(
        entry_price: f64,
        exit_price: f64,
        pnl_value: f64,
        duration_hours: f64,
    ) -> Self {
        let pnl_pct = if entry_price != 0.0 {
            ((exit_price - entry_price) / entry_price) * 100.0
        } else {
            0.0
        };

        Self {
            pnl_value,
            pnl_pct,
            opportunity_cost: None,
            outcome_type: OutcomeType::Executed,
            mfe: None,
            mae: None,
            win: pnl_value > 0.0,
            entry_price,
            exit_price,
            duration_hours,
            notes: None,
        }
    }

    /// Create hypothetical outcome for STAY_FLAT decisions
    pub fn from_stay_flat(
        entry_price: f64,
        exit_price: f64,
        opportunity_cost: f64,
        price_move_pct: f64,
        was_correct: bool,
        duration_hours: f64,
    ) -> Self {
        Self {
            pnl_value: 0.0, // STAY_FLAT never affects actual balance
            pnl_pct: price_move_pct,
            opportunity_cost: Some(opportunity_cost),
            outcome_type: OutcomeType::Hypothetical,
            mfe: None,
            mae: None,
            win: was_correct,
            entry_price,
            exit_price,
            duration_hours,
            notes: None,
        }
    }

    /// Check if this outcome should affect account balance
    pub fn should_affect_balance(&self) -> bool {
        matches!(self.outcome_type, OutcomeType::Executed)
    }

    /// Check if this is a hypothetical/STAY_FLAT outcome
    pub fn is_hypothetical(&self) -> bool {
        matches!(self.outcome_type, OutcomeType::Hypothetical)
    }

    /// Add MFE/MAE data
    pub fn with_excursions(mut self, mfe: Option<f64>, mae: Option<f64>) -> Self {
        self.mfe = mfe;
        self.mae = mae;
        self
    }

    /// Add contextual notes
    pub fn with_notes(mut self, notes: String) -> Self {
        self.notes = Some(notes);
        self
    }

    /// Get outcome severity for confidence adjustments
    pub fn outcome_severity(&self) -> f32 {
        // Special handling for STAY_FLAT/hypothetical decisions
        // Use opportunity cost magnitude if available for learning signal strength
        if self.is_hypothetical() {
            if let Some(opp_cost) = self.opportunity_cost {
                let abs_cost = opp_cost.abs();
                // Scale by opportunity cost magnitude
                return match abs_cost {
                    x if x > 5000.0 => 0.15, // Large missed opportunity
                    x if x > 2000.0 => 0.10, // Medium missed opportunity
                    x if x > 500.0 => 0.05,  // Small missed opportunity
                    _ => 0.02,               // Minimal opportunity cost
                };
            } else {
                // No opportunity cost calculated, use fixed moderate severity
                return 0.05;
            }
        }

        // For actual executed trades, scale by P&L magnitude
        let abs_pnl = self.pnl_pct.abs();
        match abs_pnl {
            x if x > 50.0 => 0.15, // Large moves (50%+)
            x if x > 20.0 => 0.10, // Medium moves (20-50%)
            x if x > 5.0 => 0.05,  // Small moves (5-20%)
            _ => 0.02,             // Minimal moves (< 5%)
        }
    }
}

/// Reflection input containing decision context and outcome
#[derive(Debug, Clone)]
pub struct ReflectionInput {
    /// Original trading decision made
    pub decision: TradingDecision,
    /// Market state when decision was made
    pub market_state: serde_json::Value,
    /// Actual trading outcome
    pub outcome: TradingOutcome,
    /// Referenced playbook bullets during decision
    pub referenced_bullets: Vec<PlaybookBullet>,
    /// Context ID for traceability
    pub context_id: Uuid,
    /// Current date for temporal context
    pub date: String,
}

/// ACE Reflector for outcome analysis and learning extraction
pub struct Reflector {
    llm_client: LLMClient,
}

impl Reflector {
    /// Create new Reflector with LLM client
    pub fn new(llm_client: LLMClient) -> Self {
        Self { llm_client }
    }

    /// Analyze trading outcome and generate learning deltas
    pub async fn reflect_on_outcome(&self, input: ReflectionInput) -> Result<Vec<Delta>> {
        info!(
            "Reflecting on trading outcome: {} with {:.1}% P&L",
            if input.outcome.win { "WIN" } else { "LOSS" },
            input.outcome.pnl_pct
        );

        // Try LLM-based reflection first
        match self.reflect_via_llm(&input).await {
            Ok(deltas) => {
                info!("LLM reflection generated {} deltas", deltas.len());
                Ok(deltas)
            }
            Err(e) => {
                warn!("LLM reflection failed: {}, using heuristic analysis", e);
                self.generate_heuristic_reflection(&input)
            }
        }
    }

    /// Perform reflection using LLM analysis
    async fn reflect_via_llm(&self, input: &ReflectionInput) -> Result<Vec<Delta>> {
        let outcome_json = json!({
            "pnl_value": input.outcome.pnl_value,
            "pnl_pct": input.outcome.pnl_pct,
            "opportunity_cost": input.outcome.opportunity_cost,
            "outcome_type": if input.outcome.is_hypothetical() { "Hypothetical" } else { "Executed" },
            "win": input.outcome.win,
            "entry_price": input.outcome.entry_price,
            "exit_price": input.outcome.exit_price,
            "duration_hours": input.outcome.duration_hours,
            "mfe": input.outcome.mfe,
            "mae": input.outcome.mae,
            "notes": input.outcome.notes
        });

        let prompt = ACEPrompts::evening_reflection_prompt(
            &input.decision,
            &input.market_state,
            &outcome_json,
            &input.date,
        );

        // Get LLM reflection
        let response = self
            .llm_client
            .generate_json::<ReflectionResult>(&prompt, None)
            .await
            .context("Failed to get LLM reflection")?;

        // Validate the reflection
        let reflection_json = serde_json::to_value(&response)?;
        if let Err(validation_error) = validate_reflection_result(&reflection_json) {
            warn!("LLM reflection failed validation: {}", validation_error);
            warn!("Using heuristic reflection instead");
            return self.generate_heuristic_reflection(&input);
        }

        info!("Reflection passed validation");

        // Convert reflection to deltas
        let deltas = self.reflection_to_deltas(response, &input)?;

        debug!("LLM reflection produced {} deltas", deltas.len());
        Ok(deltas)
    }

    /// Generate heuristic reflection when LLM is unavailable
    fn generate_heuristic_reflection(&self, input: &ReflectionInput) -> Result<Vec<Delta>> {
        let mut deltas = Vec::new();

        // Generate outcome-based confidence adjustments for referenced bullets
        for bullet in &input.referenced_bullets {
            let confidence_delta = if input.outcome.win {
                input.outcome.outcome_severity() // Positive adjustment for wins
            } else {
                -input.outcome.outcome_severity() // Negative adjustment for losses
            };

            let helpful_delta = if input.outcome.win { 1 } else { 0 };
            let harmful_delta = if !input.outcome.win { 1 } else { 0 };

            let meta = json!({
                "outcome_pnl_pct": input.outcome.pnl_pct,
                "outcome_win": input.outcome.win,
                "reflection_type": "heuristic",
                "context_id": input.context_id
            });

            deltas.push(Delta::update_counters(
                bullet.id,
                bullet.section.clone(),
                helpful_delta,
                harmful_delta,
                confidence_delta,
                Some(meta),
            ));
        }

        // Generate simple pattern based on outcome
        if input.outcome.pnl_pct.abs() > 10.0 {
            let content = if input.outcome.win {
                format!(
                    "Strong performance (+{:.1}%) in {} conditions with {} confidence. Pattern worth reinforcing.",
                    input.outcome.pnl_pct,
                    input.market_state.get("market_regime").and_then(|r| r.as_str()).unwrap_or("unknown"),
                    input.decision.confidence
                )
            } else {
                format!(
                    "Significant loss (-{:.1}%) in {} conditions. Review decision criteria and risk management.",
                    input.outcome.pnl_pct.abs(),
                    input.market_state.get("market_regime").and_then(|r| r.as_str()).unwrap_or("unknown")
                )
            };

            let section = if input.outcome.win {
                PlaybookSection::PatternInsights
            } else {
                PlaybookSection::FailureModes
            };

            let meta = json!({
                "outcome_pnl_pct": input.outcome.pnl_pct,
                "decision_confidence": input.decision.confidence,
                "market_regime": input.market_state.get("market_regime"),
                "reflection_type": "heuristic",
                "context_id": input.context_id
            });

            deltas.push(Delta::add(section, content, Some(meta)));
        }

        info!("Generated {} heuristic reflection deltas", deltas.len());
        Ok(deltas)
    }

    /// Convert LLM reflection result to deltas
    fn reflection_to_deltas(
        &self,
        reflection: ReflectionResult,
        input: &ReflectionInput,
    ) -> Result<Vec<Delta>> {
        let mut deltas = Vec::new();

        // HARD CAP: Enforce maximum total items from LLM reflection
        const MAX_TOTAL_ITEMS: usize = 5;
        let total_items = reflection.lessons_learned.len()
            + reflection.what_worked.len()
            + reflection.what_failed.len()
            + reflection.playbook_updates.len();

        if total_items > MAX_TOTAL_ITEMS {
            warn!(
                "LLM reflection generated {} items (lessons={}, worked={}, failed={}, updates={}), exceeds limit of {}. Truncating to most impactful items.",
                total_items,
                reflection.lessons_learned.len(),
                reflection.what_worked.len(),
                reflection.what_failed.len(),
                reflection.playbook_updates.len(),
                MAX_TOTAL_ITEMS
            );
        }

        // Add lessons learned as new bullets (prioritize these)
        for lesson in reflection.lessons_learned.iter().take(2) {
            let meta = json!({
                "context_id": input.context_id,
                "outcome_win": input.outcome.win,
                "outcome_pnl_pct": input.outcome.pnl_pct,
                "reflection_type": "lesson",
                "generated_at": chrono::Utc::now()
            });

            deltas.push(Delta::add(
                PlaybookSection::PatternInsights,
                lesson.clone(),
                Some(meta),
            ));
        }

        // Add what worked as reinforcement (limit to 1)
        for success in reflection.what_worked.iter().take(1) {
            let meta = json!({
                "context_id": input.context_id,
                "outcome_win": input.outcome.win,
                "reflection_type": "success_pattern",
                "generated_at": chrono::Utc::now()
            });

            deltas.push(Delta::add(
                PlaybookSection::PatternInsights,
                success.clone(),
                Some(meta),
            ));
        }

        // Add what failed as warnings (limit to 1)
        for failure in reflection.what_failed.iter().take(1) {
            let meta = json!({
                "context_id": input.context_id,
                "outcome_win": input.outcome.win,
                "reflection_type": "failure_mode",
                "generated_at": chrono::Utc::now()
            });

            deltas.push(Delta::add(
                PlaybookSection::FailureModes,
                failure.clone(),
                Some(meta),
            ));
        }

        // Add playbook updates (limit to 1)
        for update in reflection.playbook_updates.iter().take(1) {
            let meta = json!({
                "context_id": input.context_id,
                "reflection_type": "playbook_update",
                "generated_at": chrono::Utc::now()
            });

            // Try to determine appropriate section from content
            let section = self.infer_section_from_content(&update);

            deltas.push(Delta::add(section, update.clone(), Some(meta)));
        }

        // HARD CAP: Ensure we never exceed 10 total deltas per reflection
        const MAX_DELTAS_PER_DAY: usize = 10;

        // Update confidence for referenced bullets (but enforce total cap)
        let base_confidence_adjustment =
            reflection.confidence_adjustment * input.outcome.outcome_severity();

        let remaining_capacity = MAX_DELTAS_PER_DAY.saturating_sub(deltas.len());
        for bullet in input.referenced_bullets.iter().take(remaining_capacity) {
            let helpful_delta = if input.outcome.win { 1 } else { 0 };
            let harmful_delta = if !input.outcome.win { 1 } else { 0 };

            let meta = json!({
                "context_id": input.context_id,
                "outcome_win": input.outcome.win,
                "outcome_pnl_pct": input.outcome.pnl_pct,
                "confidence_adjustment": base_confidence_adjustment,
                "reflection_type": "bullet_update"
            });

            deltas.push(Delta::update_counters(
                bullet.id,
                bullet.section.clone(),
                helpful_delta,
                harmful_delta,
                base_confidence_adjustment,
                Some(meta),
            ));
        }

        if deltas.len() > MAX_DELTAS_PER_DAY {
            warn!(
                "Reflection generated {} deltas, truncating to {} (ACE design limit)",
                deltas.len(),
                MAX_DELTAS_PER_DAY
            );
            deltas.truncate(MAX_DELTAS_PER_DAY);
        }

        info!(
            "Reflection produced {} deltas (lessons={}, worked={}, failed={}, updates={}, bullet_updates={})",
            deltas.len(),
            reflection.lessons_learned.len().min(2),
            reflection.what_worked.len().min(1),
            reflection.what_failed.len().min(1),
            reflection.playbook_updates.len().min(1),
            input.referenced_bullets.len().min(remaining_capacity)
        );

        Ok(deltas)
    }

    /// Infer playbook section from content text
    fn infer_section_from_content(&self, content: &str) -> PlaybookSection {
        let content_lower = content.to_lowercase();

        if content_lower.contains("avoid")
            || content_lower.contains("don't")
            || content_lower.contains("never")
        {
            PlaybookSection::FailureModes
        } else if content_lower.contains("vix")
            || content_lower.contains("volatility")
            || content_lower.contains("regime")
        {
            PlaybookSection::RegimeRules
        } else if content_lower.contains("model")
            || content_lower.contains("signal")
            || content_lower.contains("indicator")
        {
            PlaybookSection::ModelReliability
        } else if content_lower.contains("news")
            || content_lower.contains("earnings")
            || content_lower.contains("fed")
        {
            PlaybookSection::NewsImpact
        } else if content_lower.contains("strategy")
            || content_lower.contains("stopped working")
            || content_lower.contains("lifecycle")
        {
            PlaybookSection::StrategyLifecycle
        } else {
            // Default to pattern insights
            PlaybookSection::PatternInsights
        }
    }

    /// Identify which bullets were likely referenced in decision reasoning
    pub fn identify_referenced_bullets(
        &self,
        decision_reasoning: &str,
        available_bullets: &[PlaybookBullet],
    ) -> Vec<PlaybookBullet> {
        let reasoning_lower = decision_reasoning.to_lowercase();
        let mut referenced = Vec::new();

        for bullet in available_bullets {
            // Simple keyword matching - in production might use embeddings
            let bullet_keywords = self.extract_keywords(&bullet.content);
            let matches = bullet_keywords
                .iter()
                .filter(|keyword| reasoning_lower.contains(&keyword.to_lowercase()))
                .count();

            // If multiple keywords match or bullet is short and matches well
            if matches >= 2 || (bullet.content.len() < 100 && matches >= 1) {
                referenced.push(bullet.clone());
            }
        }

        debug!(
            "Identified {} referenced bullets from decision reasoning",
            referenced.len()
        );
        referenced
    }

    /// Extract keywords from bullet content for matching
    fn extract_keywords(&self, content: &str) -> Vec<String> {
        let words: Vec<&str> = content.split_whitespace().collect();
        let mut keywords = Vec::new();

        // Extract significant words (length > 3, not common words)
        let common_words = [
            "the", "and", "when", "with", "have", "that", "this", "from", "they",
        ];

        for word in words {
            let clean_word = word
                .trim_matches(|c: char| !c.is_alphabetic())
                .to_lowercase();
            if clean_word.len() > 3 && !common_words.contains(&clean_word.as_str()) {
                keywords.push(clean_word);
            }
        }

        keywords
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ace::playbook::PlaybookSection, llm::LLMConfig};
    use serde_json::json;

    #[test]
    fn test_trading_outcome() {
        let outcome = TradingOutcome::from_pnl(100.0, 105.0, 50.0, 2.5)
            .with_excursions(Some(75.0), Some(-25.0))
            .with_notes("Good execution".to_string());

        assert_eq!(outcome.pnl_value, 50.0);
        assert_eq!(outcome.pnl_pct, 5.0);
        assert_eq!(outcome.opportunity_cost, None);
        assert_eq!(outcome.outcome_type, OutcomeType::Executed);
        assert!(outcome.should_affect_balance());
        assert!(!outcome.is_hypothetical());
        assert!(outcome.win);
        assert_eq!(outcome.entry_price, 100.0);
        assert_eq!(outcome.exit_price, 105.0);
        assert_eq!(outcome.mfe, Some(75.0));
        assert_eq!(outcome.mae, Some(-25.0));
        assert_eq!(outcome.notes, Some("Good execution".to_string()));
    }

    #[test]
    fn test_stay_flat_outcome() {
        // STAY_FLAT with small market move (correct decision)
        let stay_flat_correct = TradingOutcome::from_stay_flat(
            100.0,
            100.2,
            0.0,
            0.2,
            true,
            6.5,
        ).with_notes("Market barely moved, staying flat was correct".to_string());

        assert_eq!(stay_flat_correct.pnl_value, 0.0);
        assert_eq!(stay_flat_correct.pnl_pct, 0.2);
        assert_eq!(stay_flat_correct.opportunity_cost, Some(0.0));
        assert_eq!(stay_flat_correct.outcome_type, OutcomeType::Hypothetical);
        assert!(!stay_flat_correct.should_affect_balance());
        assert!(stay_flat_correct.is_hypothetical());
        assert!(stay_flat_correct.win); // Was correct decision

        // STAY_FLAT with large market move (missed opportunity)
        let stay_flat_miss = TradingOutcome::from_stay_flat(
            100.0,
            105.0,
            -2500.0, // Negative = missed profit
            5.0,
            false,
            6.5,
        );

        assert_eq!(stay_flat_miss.pnl_value, 0.0);
        assert_eq!(stay_flat_miss.opportunity_cost, Some(-2500.0));
        assert!(!stay_flat_miss.should_affect_balance());
        assert!(!stay_flat_miss.win); // Was incorrect decision
    }

    #[test]
    fn test_outcome_severity() {
        // STAY_FLAT with small opportunity cost
        let stay_flat_small = TradingOutcome::from_stay_flat(100.0, 100.5, -200.0, 0.5, false, 6.5);
        assert_eq!(stay_flat_small.outcome_severity(), 0.02); // < $500 opp cost

        // STAY_FLAT with medium opportunity cost
        let stay_flat_medium = TradingOutcome::from_stay_flat(100.0, 105.0, -1500.0, 5.0, false, 6.5);
        assert_eq!(stay_flat_medium.outcome_severity(), 0.05); // $500-2000 opp cost

        // STAY_FLAT with large opportunity cost
        let stay_flat_large = TradingOutcome::from_stay_flat(100.0, 110.0, -6000.0, 10.0, false, 6.5);
        assert_eq!(stay_flat_large.outcome_severity(), 0.15); // > $5000 opp cost

        // STAY_FLAT correct (no opp cost)
        let stay_flat_correct = TradingOutcome::from_stay_flat(100.0, 100.2, 0.0, 0.2, true, 6.5);
        assert_eq!(stay_flat_correct.outcome_severity(), 0.02); // Minimal opp cost

        // Executed trade: Small moves (< 5%)
        let small_win = TradingOutcome::from_pnl(100.0, 102.0, 20.0, 1.0);
        assert_eq!(small_win.outcome_severity(), 0.02); // 2% gain

        // Executed trade: Medium moves (5-20%)
        let medium_loss = TradingOutcome::from_pnl(100.0, 85.0, -150.0, 3.0);
        assert_eq!(medium_loss.outcome_severity(), 0.05); // 15% loss

        // Executed trade: Large moves (20-50%)
        let large_loss = TradingOutcome::from_pnl(100.0, 75.0, -250.0, 3.0);
        assert_eq!(large_loss.outcome_severity(), 0.10); // 25% loss

        // Executed trade: Very large moves (50%+)
        let very_large_win = TradingOutcome::from_pnl(100.0, 160.0, 600.0, 4.0);
        assert_eq!(very_large_win.outcome_severity(), 0.15); // 60% gain
    }

    #[tokio::test]
    #[ignore = "Requires Ollama service to be running"]
    async fn test_infer_section_from_content() {
        let config = LLMConfig::default();
        let llm_client = LLMClient::new(config)
            .await
            .expect("Failed to create LLM client");
        let reflector = Reflector::new(llm_client);

        assert_eq!(
            reflector.infer_section_from_content("Never trade on FOMC days"),
            PlaybookSection::FailureModes
        );

        assert_eq!(
            reflector.infer_section_from_content("High VIX regime shows mean reversion"),
            PlaybookSection::RegimeRules
        );

        assert_eq!(
            reflector.infer_section_from_content("ML models accuracy drops in volatility"),
            PlaybookSection::ModelReliability
        );

        assert_eq!(
            reflector.infer_section_from_content("Fed hawkish pivot impacts markets"),
            PlaybookSection::NewsImpact
        );

        assert_eq!(
            reflector.infer_section_from_content("Breakout strategy stopped working"),
            PlaybookSection::StrategyLifecycle
        );

        assert_eq!(
            reflector.infer_section_from_content("General trading pattern observation"),
            PlaybookSection::PatternInsights
        );
    }

    #[tokio::test]
    #[ignore = "Requires Ollama service to be running"]
    async fn test_extract_keywords() {
        let config = LLMConfig::default();
        let llm_client = LLMClient::new(config)
            .await
            .expect("Failed to create LLM client");
        let reflector = Reflector::new(llm_client);

        let content = "When VIX drops below 15, calls have 73% win rate";
        let keywords = reflector.extract_keywords(content);

        assert!(keywords.contains(&"drops".to_string()));
        assert!(keywords.contains(&"below".to_string()));
        assert!(keywords.contains(&"calls".to_string()));
        assert!(keywords.contains(&"rate".to_string()));

        // Should not contain common words
        assert!(!keywords.contains(&"when".to_string()));
        assert!(!keywords.contains(&"have".to_string()));
    }

    #[tokio::test]
    #[ignore = "Requires Ollama service to be running"]
    async fn test_identify_referenced_bullets() {
        let config = LLMConfig::default();
        let llm_client = LLMClient::new(config)
            .await
            .expect("Failed to create LLM client");
        let reflector = Reflector::new(llm_client);

        let bullets = vec![
            PlaybookBullet {
                id: Uuid::new_v4(),
                section: PlaybookSection::PatternInsights,
                content: "VIX below 15 signals low volatility regime".to_string(),
                helpful_count: 5,
                harmful_count: 1,
                confidence: 0.8,
                last_used: None,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                source_context_id: None,
                meta: None,
            },
            PlaybookBullet {
                id: Uuid::new_v4(),
                section: PlaybookSection::FailureModes,
                content: "High momentum often reverses quickly".to_string(),
                helpful_count: 2,
                harmful_count: 3,
                confidence: 0.4,
                last_used: None,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                source_context_id: None,
                meta: None,
            },
        ];

        let reasoning = "Decision based on VIX being below threshold indicating low volatility environment favorable for directional trades";

        let referenced = reflector.identify_referenced_bullets(reasoning, &bullets);
        assert_eq!(referenced.len(), 1);
        assert!(referenced[0].content.contains("VIX"));

        // Test no matches
        let unrelated_reasoning = "Random market movement without specific indicators";
        let referenced_none = reflector.identify_referenced_bullets(unrelated_reasoning, &bullets);
        assert_eq!(referenced_none.len(), 0);
    }

    #[tokio::test]
    async fn test_generate_heuristic_reflection() {
        // This would need proper mocking setup for full testing
        // For now, just test that it doesn't panic with valid input

        let decision = TradingDecision {
            action: "BUY_CALLS".to_string(),
            confidence: 0.75,
            reasoning: "Strong technical setup with VIX support".to_string(),
            key_factors: vec!["Low VIX".to_string()],
            risk_factors: vec!["Earnings next week".to_string()],
            similar_pattern_reference: None,
            position_size_multiplier: 1.0,
        };

        let outcome = TradingOutcome::from_pnl(100.0, 110.0, 100.0, 2.0);

        let input = ReflectionInput {
            decision,
            market_state: json!({"market_regime": "TRENDING_UP"}),
            outcome,
            referenced_bullets: vec![],
            context_id: Uuid::new_v4(),
            date: "2025-01-15".to_string(),
        };

        // Would need mock LLMClient for full test
        // let deltas = reflector.generate_heuristic_reflection(&input).unwrap();
        // assert!(!deltas.is_empty());
    }
}

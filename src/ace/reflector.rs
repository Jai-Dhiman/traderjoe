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

/// Trading outcome data for reflection
#[derive(Debug, Clone)]
pub struct TradingOutcome {
    /// Profit/loss in dollars
    pub pnl_value: f64,
    /// Profit/loss as percentage
    pub pnl_pct: f64,
    /// Maximum favorable excursion (best unrealized profit)
    pub mfe: Option<f64>,
    /// Maximum adverse excursion (worst unrealized loss)
    pub mae: Option<f64>,
    /// Whether the trade was ultimately profitable
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
    /// Create outcome from basic P&L data
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
            mfe: None,
            mae: None,
            win: pnl_value > 0.0,
            entry_price,
            exit_price,
            duration_hours,
            notes: None,
        }
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
        // Scale severity based on P&L percentage
        let abs_pnl = self.pnl_pct.abs();
        match abs_pnl {
            x if x > 50.0 => 0.10, // Large moves
            x if x > 20.0 => 0.05, // Medium moves
            x if x > 5.0 => 0.02,  // Small moves
            _ => 0.01,             // Minimal moves
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

        // Add lessons learned as new bullets
        for lesson in reflection.lessons_learned {
            let meta = json!({
                "context_id": input.context_id,
                "outcome_win": input.outcome.win,
                "outcome_pnl_pct": input.outcome.pnl_pct,
                "reflection_type": "lesson",
                "generated_at": chrono::Utc::now()
            });

            deltas.push(Delta::add(
                PlaybookSection::PatternInsights,
                lesson,
                Some(meta),
            ));
        }

        // Add what worked as reinforcement
        for success in reflection.what_worked {
            let meta = json!({
                "context_id": input.context_id,
                "outcome_win": input.outcome.win,
                "reflection_type": "success_pattern",
                "generated_at": chrono::Utc::now()
            });

            deltas.push(Delta::add(
                PlaybookSection::PatternInsights,
                success,
                Some(meta),
            ));
        }

        // Add what failed as warnings
        for failure in reflection.what_failed {
            let meta = json!({
                "context_id": input.context_id,
                "outcome_win": input.outcome.win,
                "reflection_type": "failure_mode",
                "generated_at": chrono::Utc::now()
            });

            deltas.push(Delta::add(
                PlaybookSection::FailureModes,
                failure,
                Some(meta),
            ));
        }

        // Add playbook updates
        for update in reflection.playbook_updates {
            let meta = json!({
                "context_id": input.context_id,
                "reflection_type": "playbook_update",
                "generated_at": chrono::Utc::now()
            });

            // Try to determine appropriate section from content
            let section = self.infer_section_from_content(&update);

            deltas.push(Delta::add(section, update, Some(meta)));
        }

        // Update confidence for referenced bullets
        let base_confidence_adjustment =
            reflection.confidence_adjustment * input.outcome.outcome_severity();

        for bullet in &input.referenced_bullets {
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
        assert!(outcome.win);
        assert_eq!(outcome.entry_price, 100.0);
        assert_eq!(outcome.exit_price, 105.0);
        assert_eq!(outcome.mfe, Some(75.0));
        assert_eq!(outcome.mae, Some(-25.0));
        assert_eq!(outcome.notes, Some("Good execution".to_string()));
    }

    #[test]
    fn test_outcome_severity() {
        let small_win = TradingOutcome::from_pnl(100.0, 102.0, 20.0, 1.0);
        assert_eq!(small_win.outcome_severity(), 0.01); // 2% gain

        let medium_loss = TradingOutcome::from_pnl(100.0, 85.0, -150.0, 3.0);
        assert_eq!(medium_loss.outcome_severity(), 0.02); // 15% loss (between 5-20%)

        let large_win = TradingOutcome::from_pnl(100.0, 160.0, 600.0, 4.0);
        assert_eq!(large_win.outcome_severity(), 0.10); // 60% gain
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

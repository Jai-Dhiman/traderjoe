//! ACE Generator module for producing new playbook bullets from market analysis
//! Extracts patterns and insights from trading contexts using LLM reasoning

use anyhow::{Context, Result};
use serde_json::json;
use tracing::{info, warn, debug};
use uuid::Uuid;

use crate::{
    ace::{
        delta::{Delta, DeltaOp},
        playbook::{PlaybookSection, PlaybookBullet},
        prompts::ACEPrompts,
    },
    llm::LLMClient,
    vector::ContextEntry,
};

/// Generator input containing market context and analysis state
#[derive(Debug, Clone)]
pub struct GeneratorInput {
    /// Current market state as JSON
    pub market_state: serde_json::Value,
    /// ML signals and technical indicators
    pub ml_signals: serde_json::Value,
    /// Similar historical contexts from vector search
    pub similar_contexts: Vec<ContextEntry>,
    /// Existing playbook bullets for context
    pub existing_playbook: Vec<PlaybookBullet>,
    /// Source context ID for traceability
    pub source_context_id: Option<Uuid>,
}

/// Pattern candidate extracted by the Generator
#[derive(Debug, Clone)]
pub struct PatternCandidate {
    /// Target section for the pattern
    pub section: PlaybookSection,
    /// Pattern description content
    pub content: String,
    /// Initial confidence score
    pub confidence: f32,
    /// Pattern type or category
    pub pattern_type: String,
    /// Supporting evidence or reasoning
    pub evidence: Vec<String>,
}

/// ACE Generator for extracting patterns from trading contexts
pub struct Generator {
    llm_client: LLMClient,
}

impl Generator {
    /// Create new Generator with LLM client
    pub fn new(llm_client: LLMClient) -> Self {
        Self { llm_client }
    }

    /// Generate new pattern candidates from market analysis
    pub async fn generate_patterns(&self, input: GeneratorInput) -> Result<Vec<Delta>> {
        info!("Generating patterns from market context with {} similar contexts", 
              input.similar_contexts.len());

        // Try LLM-based pattern extraction first
        match self.extract_patterns_via_llm(&input).await {
            Ok(deltas) => {
                info!("LLM generated {} pattern deltas", deltas.len());
                Ok(deltas)
            }
            Err(e) => {
                warn!("LLM pattern extraction failed: {}, using heuristic fallback", e);
                self.generate_heuristic_patterns(&input).await
            }
        }
    }

    /// Extract patterns using LLM analysis
    async fn extract_patterns_via_llm(&self, input: &GeneratorInput) -> Result<Vec<Delta>> {
        // Build context summary for LLM prompt
        let context_summary = self.build_context_summary(input);
        
        let prompt = ACEPrompts::pattern_extraction_prompt(&input.similar_contexts);
        
        // Enhance prompt with current market context
        let enhanced_prompt = format!(
            "{}\n\nCURRENT MARKET CONTEXT:\n{}\n\nML SIGNALS:\n{}\n\nTASK:\nBased on the historical contexts above and current market conditions, identify new patterns that should be added to the ACE playbook. Focus on:\n\n1. Market regime patterns (when certain conditions lead to predictable outcomes)\n2. Failure modes (what to avoid or watch for)\n3. Technical signal reliability (when indicators work vs don't work)\n4. Sentiment-price divergences\n5. Time-based patterns (day of week, time of day effects)\n\nFor each pattern, provide:\n- Section: pattern_insights, failure_modes, regime_rules, model_reliability, news_impact, or strategy_lifecycle\n- Content: Clear, specific description with quantifiable details where possible\n- Confidence: 0.0-1.0 based on evidence strength\n- Evidence: Supporting observations\n\nRespond with valid JSON array:\n[\n  {{\n    \"section\": \"pattern_insights\",\n    \"content\": \"When VIX drops below 15 while SPY is within 2% of ATH, calls have 73% win rate over next 3 days\",\n    \"confidence\": 0.75,\n    \"pattern_type\": \"low_volatility_momentum\",\n    \"evidence\": [\"Historical win rate\", \"Recent validation\"]\n  }}\n]",
            prompt,
            serde_json::to_string_pretty(&input.market_state).unwrap_or_default(),
            serde_json::to_string_pretty(&input.ml_signals).unwrap_or_default()
        );

        // Get LLM response
        let response = self.llm_client.generate(&enhanced_prompt, None).await
            .context("Failed to get LLM response for pattern extraction")?;

        // Parse response into pattern candidates
        let candidates = self.parse_llm_response(&response.content)
            .context("Failed to parse LLM response into patterns")?;

        // Convert candidates to deltas
        let deltas = self.candidates_to_deltas(candidates, input.source_context_id);

        debug!("Converted {} candidates to {} deltas", candidates.len(), deltas.len());
        Ok(deltas)
    }

    /// Generate heuristic patterns when LLM is unavailable
    async fn generate_heuristic_patterns(&self, input: &GeneratorInput) -> Result<Vec<Delta>> {
        let mut deltas = Vec::new();

        // Extract basic patterns from market state and signals
        if let Some(patterns) = self.extract_regime_patterns(&input.market_state, &input.ml_signals) {
            deltas.extend(patterns);
        }

        if let Some(patterns) = self.extract_reliability_patterns(&input.ml_signals) {
            deltas.extend(patterns);
        }

        // Generate pattern from similar contexts if available
        if !input.similar_contexts.is_empty() {
            if let Some(pattern) = self.synthesize_from_similar_contexts(&input.similar_contexts, input.source_context_id) {
                deltas.push(pattern);
            }
        }

        info!("Generated {} heuristic patterns as fallback", deltas.len());
        Ok(deltas)
    }

    /// Build context summary for LLM prompt
    fn build_context_summary(&self, input: &GeneratorInput) -> String {
        let mut summary = String::new();
        
        // Market regime assessment
        if let Some(regime) = input.market_state.get("market_regime").and_then(|r| r.as_str()) {
            summary.push_str(&format!("Market Regime: {}\n", regime));
        }

        // Key signals
        if let Some(momentum) = input.ml_signals.get("momentum_score").and_then(|s| s.as_f64()) {
            summary.push_str(&format!("Momentum Score: {:.2}\n", momentum));
        }

        // Similar context count
        summary.push_str(&format!("Similar Historical Contexts: {}\n", input.similar_contexts.len()));
        
        // Playbook size
        summary.push_str(&format!("Existing Playbook Bullets: {}\n", input.existing_playbook.len()));

        summary
    }

    /// Parse LLM response into pattern candidates
    fn parse_llm_response(&self, response: &str) -> Result<Vec<PatternCandidate>> {
        // Try to extract JSON from response
        let json_str = self.extract_json_from_response(response)
            .ok_or_else(|| anyhow::anyhow!("No valid JSON found in LLM response"))?;

        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .context("Failed to parse JSON response")?;

        let array = parsed.as_array()
            .ok_or_else(|| anyhow::anyhow!("Expected JSON array of patterns"))?;

        let mut candidates = Vec::new();
        for item in array {
            if let Some(candidate) = self.parse_pattern_item(item) {
                candidates.push(candidate);
            } else {
                warn!("Failed to parse pattern item: {}", item);
            }
        }

        Ok(candidates)
    }

    /// Extract JSON from potentially formatted LLM response
    fn extract_json_from_response(&self, response: &str) -> Option<String> {
        // Try to find JSON array in response
        if let Some(start) = response.find('[') {
            if let Some(end) = response.rfind(']') {
                if end > start {
                    return Some(response[start..=end].to_string());
                }
            }
        }

        // Try to find single JSON object
        if let Some(start) = response.find('{') {
            if let Some(end) = response.rfind('}') {
                if end > start {
                    // Wrap single object in array
                    return Some(format!("[{}]", &response[start..=end]));
                }
            }
        }

        None
    }

    /// Parse individual pattern item from JSON
    fn parse_pattern_item(&self, item: &serde_json::Value) -> Option<PatternCandidate> {
        let section_str = item.get("section")?.as_str()?;
        let section = PlaybookSection::from_str(section_str).ok()?;
        
        let content = item.get("content")?.as_str()?.to_string();
        let confidence = item.get("confidence")?.as_f64()? as f32;
        let pattern_type = item.get("pattern_type")
            .and_then(|t| t.as_str())
            .unwrap_or("unknown")
            .to_string();
        
        let evidence = item.get("evidence")
            .and_then(|e| e.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default();

        Some(PatternCandidate {
            section,
            content,
            confidence: confidence.max(0.0).min(1.0),
            pattern_type,
            evidence,
        })
    }

    /// Convert pattern candidates to deltas
    fn candidates_to_deltas(&self, candidates: Vec<PatternCandidate>, source_context_id: Option<Uuid>) -> Vec<Delta> {
        candidates.into_iter()
            .map(|candidate| {
                let meta = json!({
                    "source_context_id": source_context_id,
                    "pattern_type": candidate.pattern_type,
                    "confidence": candidate.confidence,
                    "evidence": candidate.evidence,
                    "generated_by": "llm",
                    "generated_at": chrono::Utc::now()
                });

                Delta {
                    op: DeltaOp::Add,
                    section: candidate.section,
                    content: Some(candidate.content),
                    bullet_id: None,
                    helpful_delta: None,
                    harmful_delta: None,
                    confidence_adjustment: None,
                    meta: Some(meta),
                }
            })
            .collect()
    }

    /// Extract regime-based patterns from market state
    fn extract_regime_patterns(&self, market_state: &serde_json::Value, ml_signals: &serde_json::Value) -> Option<Vec<Delta>> {
        let mut patterns = Vec::new();

        // High volatility regime pattern
        if let (Some(volatility), Some(regime)) = (
            market_state.get("market_data")
                .and_then(|md| md.get("volatility_20d"))
                .and_then(|v| v.as_f64()),
            market_state.get("market_regime").and_then(|r| r.as_str())
        ) {
            if volatility > 3.0 && regime == "HIGH_VOLATILITY" {
                let content = format!(
                    "High volatility regime detected ({}% volatility). Historical patterns show mean reversion bias increases.",
                    volatility
                );
                
                let meta = json!({
                    "pattern_type": "regime_volatility",
                    "volatility": volatility,
                    "generated_by": "heuristic"
                });

                patterns.push(Delta {
                    op: DeltaOp::Add,
                    section: PlaybookSection::RegimeRules,
                    content: Some(content),
                    bullet_id: None,
                    helpful_delta: None,
                    harmful_delta: None,
                    confidence_adjustment: None,
                    meta: Some(meta),
                });
            }
        }

        if patterns.is_empty() { None } else { Some(patterns) }
    }

    /// Extract model reliability patterns
    fn extract_reliability_patterns(&self, ml_signals: &serde_json::Value) -> Option<Vec<Delta>> {
        let mut patterns = Vec::new();

        // ML confidence pattern
        if let Some(ml_confidence) = ml_signals.get("ml_confidence").and_then(|c| c.as_f64()) {
            if ml_confidence < 0.6 {
                let content = format!(
                    "ML models showing low confidence ({:.1}%). Consider reducing position size or staying flat.",
                    ml_confidence * 100.0
                );

                let meta = json!({
                    "pattern_type": "model_confidence",
                    "confidence": ml_confidence,
                    "generated_by": "heuristic"
                });

                patterns.push(Delta {
                    op: DeltaOp::Add,
                    section: PlaybookSection::ModelReliability,
                    content: Some(content),
                    bullet_id: None,
                    helpful_delta: None,
                    harmful_delta: None,
                    confidence_adjustment: None,
                    meta: Some(meta),
                });
            }
        }

        if patterns.is_empty() { None } else { Some(patterns) }
    }

    /// Synthesize pattern from similar contexts
    fn synthesize_from_similar_contexts(&self, contexts: &[ContextEntry], source_context_id: Option<Uuid>) -> Option<Delta> {
        if contexts.is_empty() {
            return None;
        }

        // Find common themes in similar contexts
        let win_count = contexts.iter()
            .filter(|ctx| {
                ctx.outcome.as_ref()
                    .and_then(|o| o.get("win"))
                    .and_then(|w| w.as_bool())
                    .unwrap_or(false)
            })
            .count();

        let total_count = contexts.len();
        let win_rate = win_count as f64 / total_count as f64;

        if total_count >= 3 {
            let content = format!(
                "Pattern identified from {} similar contexts: {:.0}% success rate in comparable market conditions.",
                total_count, win_rate * 100.0
            );

            let meta = json!({
                "pattern_type": "historical_similarity",
                "win_rate": win_rate,
                "sample_size": total_count,
                "source_context_id": source_context_id,
                "generated_by": "heuristic"
            });

            Some(Delta {
                op: DeltaOp::Add,
                section: PlaybookSection::PatternInsights,
                content: Some(content),
                bullet_id: None,
                helpful_delta: None,
                harmful_delta: None,
                confidence_adjustment: None,
                meta: Some(meta),
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ace::playbook::PlaybookSection, llm::LLMConfig};
    use serde_json::json;

    fn create_test_input() -> GeneratorInput {
        GeneratorInput {
            market_state: json!({
                "market_regime": "TRENDING_UP",
                "market_data": {
                    "volatility_20d": 2.5,
                    "daily_change_pct": 1.2
                }
            }),
            ml_signals: json!({
                "momentum_score": 0.75,
                "ml_confidence": 0.68
            }),
            similar_contexts: vec![],
            existing_playbook: vec![],
            source_context_id: Some(Uuid::new_v4()),
        }
    }

    #[test]
    fn test_extract_json_from_response() {
        let config = LLMConfig::default();
        let llm_client = LLMClient::new(config).await.expect("Failed to create LLM client");
        let generator = Generator::new(llm_client);

        // Test JSON array extraction
        let response = "Here are the patterns:\n[{\"section\": \"pattern_insights\", \"content\": \"test\"}]\nEnd";
        let extracted = generator.extract_json_from_response(response).unwrap();
        assert!(extracted.starts_with('['));
        assert!(extracted.ends_with(']'));

        // Test single object wrapping
        let response = "Pattern: {\"section\": \"failure_modes\", \"content\": \"test\"}";
        let extracted = generator.extract_json_from_response(response).unwrap();
        assert!(extracted.starts_with('['));
        assert!(extracted.ends_with(']'));

        // Test no JSON
        let response = "No JSON here";
        assert!(generator.extract_json_from_response(response).is_none());
    }

    #[test]
    fn test_parse_pattern_item() {
        let config = LLMConfig::default();
        let llm_client = LLMClient::new(config).await.expect("Failed to create LLM client");
        let generator = Generator::new(llm_client);

        let item = json!({
            "section": "pattern_insights",
            "content": "When VIX < 15, calls outperform puts 3:1",
            "confidence": 0.8,
            "pattern_type": "volatility_regime",
            "evidence": ["Historical data", "Recent validation"]
        });

        let candidate = generator.parse_pattern_item(&item).unwrap();
        assert_eq!(candidate.section, PlaybookSection::PatternInsights);
        assert_eq!(candidate.content, "When VIX < 15, calls outperform puts 3:1");
        assert_eq!(candidate.confidence, 0.8);
        assert_eq!(candidate.pattern_type, "volatility_regime");
        assert_eq!(candidate.evidence.len(), 2);

        // Test invalid section
        let invalid_item = json!({
            "section": "invalid_section",
            "content": "test",
            "confidence": 0.5
        });
        assert!(generator.parse_pattern_item(&invalid_item).is_none());
    }

    #[tokio::test]
    async fn test_generate_heuristic_patterns() {
        // This test would require mocking LLMClient, so we'll focus on the heuristic parts
        // that don't require actual LLM calls
        
        let input = GeneratorInput {
            market_state: json!({
                "market_regime": "HIGH_VOLATILITY",
                "market_data": {
                    "volatility_20d": 4.5
                }
            }),
            ml_signals: json!({
                "ml_confidence": 0.4
            }),
            similar_contexts: vec![],
            existing_playbook: vec![],
            source_context_id: Some(Uuid::new_v4()),
        };

        // Test heuristic pattern generation without actual LLM
        // We would need to create a mock LLMClient for full testing
        let patterns = Generator::extract_regime_patterns(
            &Generator { llm_client: todo!() }, // This would need proper mocking
            &input.market_state,
            &input.ml_signals
        );

        if let Some(patterns) = patterns {
            assert!(!patterns.is_empty());
            assert_eq!(patterns[0].section, PlaybookSection::RegimeRules);
            assert!(patterns[0].content.as_ref().unwrap().contains("volatility"));
        }
    }

    #[test]
    fn test_candidates_to_deltas() {
        let config = LLMConfig::default();
        let llm_client = LLMClient::new(config).await.expect("Failed to create LLM client");
        let generator = Generator::new(llm_client);

        let candidates = vec![
            PatternCandidate {
                section: PlaybookSection::PatternInsights,
                content: "Test pattern".to_string(),
                confidence: 0.7,
                pattern_type: "test".to_string(),
                evidence: vec!["Evidence 1".to_string()],
            }
        ];

        let source_id = Uuid::new_v4();
        let deltas = generator.candidates_to_deltas(candidates, Some(source_id));

        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].op, DeltaOp::Add);
        assert_eq!(deltas[0].section, PlaybookSection::PatternInsights);
        assert_eq!(deltas[0].content, Some("Test pattern".to_string()));
        assert!(deltas[0].meta.is_some());

        let meta = deltas[0].meta.as_ref().unwrap();
        assert_eq!(meta["pattern_type"], "test");
        assert_eq!(meta["confidence"], 0.7);
        assert_eq!(meta["generated_by"], "llm");
    }
}
//! Prompt templates for ACE (Augmented Context Evolution) system
//! Structured prompts for daily decision making and context evolution

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Trading decision output format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingDecision {
    pub action: String,  // "BUY_CALLS", "BUY_PUTS", "STAY_FLAT"
    pub confidence: f32, // 0.0 to 1.0
    pub reasoning: String,
    pub key_factors: Vec<String>,
    pub risk_factors: Vec<String>,
    pub similar_pattern_reference: Option<String>,
    pub position_size_multiplier: f32, // 0.0 to 1.0, applied to base position size
}

/// Context analysis for ACE reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextAnalysis {
    pub market_regime: String, // "TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE"
    pub sentiment_summary: String,
    pub technical_summary: String,
    pub key_events: Vec<String>,
}

/// ACE reflection on previous decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionResult {
    pub what_worked: Vec<String>,
    pub what_failed: Vec<String>,
    pub lessons_learned: Vec<String>,
    pub playbook_updates: Vec<String>,
    pub confidence_adjustment: f32, // -1.0 to 1.0
}

/// Prompt template builder for ACE system
pub struct ACEPrompts;

impl ACEPrompts {
    /// Generate morning decision prompt with context and technical indicators
    pub fn morning_decision_prompt(
        market_state: &Value,
        ml_signals: &Value,  // Named ml_signals for backward compatibility, but contains technical indicators
        similar_contexts: &[crate::vector::ContextEntry],
        playbook_entries: &[String],
        current_date: &str,
    ) -> String {
        let similar_contexts_text = similar_contexts.iter()
            .enumerate()
            .map(|(i, ctx)| {
                format!(
                    "{}. Date: {}, Decision: {}, Reasoning: {}, Confidence: {:.2}, Outcome: {}, Similarity: {:.3}",
                    i + 1,
                    ctx.timestamp.format("%Y-%m-%d"),
                    serde_json::to_string_pretty(&ctx.decision).unwrap_or("Unknown".to_string()),
                    ctx.reasoning,
                    ctx.confidence,
                    ctx.outcome.as_ref()
                        .map(|_o| format!("Success"))
                        .unwrap_or("Pending".to_string()),
                    ctx.similarity.unwrap_or(0.0)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let playbook_text = playbook_entries
            .iter()
            .enumerate()
            .map(|(i, entry)| format!("{}. {}", i + 1, entry))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"You are an expert trading system using ACE (Augmented Context Evolution) to make daily trading decisions.

CURRENT DATE: {}

MARKET STATE:
{}

TECHNICAL INDICATORS:
{}

SIMILAR PAST CONTEXTS (Top {} most relevant):
{}

RELEVANT PLAYBOOK ENTRIES:
{}

TASK:
Analyze the current market situation and provide a trading recommendation for SPY options (0-1 DTE).

Your response must be valid JSON matching this exact format:
{{
    "action": "BUY_CALLS" | "BUY_PUTS" | "STAY_FLAT",
    "confidence": 0.75,
    "reasoning": "Clear explanation of your decision based on the evidence above",
    "key_factors": ["Factor 1", "Factor 2", "Factor 3"],
    "risk_factors": ["Risk 1", "Risk 2"],
    "similar_pattern_reference": "Reference to most relevant past pattern or null",
    "position_size_multiplier": 0.8
}}

REASONING PROCESS:
1. Analyze what the technical indicators suggest about momentum and trend
2. Compare to similar past situations - what worked before?
3. Consider current market regime and sentiment
4. Identify key risk factors that could invalidate the thesis
5. Determine confidence based on pattern strength and indicator alignment

RULES:
- Only trade if confidence > 0.6
- Reduce position size if confidence < 0.75
- Never risk more than intended with position_size_multiplier
- Be explicit about why this setup is attractive or unattractive
- Reference specific playbook entries when applicable

CRITICAL: Do NOT mention "fallback", "limited data", or "research availability" in your response.
Base decisions ONLY on the technical indicators, price movement, and sentiment shown above.

Focus on high-probability setups with favorable risk/reward rather than forcing trades."#,
            current_date,
            serde_json::to_string_pretty(market_state).unwrap_or("No market data".to_string()),
            serde_json::to_string_pretty(ml_signals).unwrap_or("No technical indicators".to_string()),
            similar_contexts.len(),
            similar_contexts_text,
            if playbook_entries.is_empty() {
                "No relevant playbook entries yet."
            } else {
                &playbook_text
            }
        )
    }

    /// Generate evening reflection prompt for learning from outcomes
    pub fn evening_reflection_prompt(
        original_decision: &TradingDecision,
        market_state: &Value,
        actual_outcome: &Value,
        current_date: &str,
    ) -> String {
        format!(
            r#"You are conducting an evening reflection for the ACE trading system to learn from today's outcome.

DATE: {}

ORIGINAL DECISION MADE THIS MORNING:
Action: {}
Confidence: {:.2}
Reasoning: {}
Key Factors: {}
Risk Factors: {}

MORNING MARKET STATE:
{}

ACTUAL OUTCOME:
{}

REFLECTION TASK:
Analyze what happened today and extract learnings for the playbook. Respond with valid JSON:

{{
    "what_worked": ["Aspect 1 that was correct", "Aspect 2 that helped"],
    "what_failed": ["Aspect 1 that was wrong", "Aspect 2 that hurt"],
    "lessons_learned": ["Lesson 1 for future", "Lesson 2 for future"],
    "playbook_updates": ["Update 1 to add/modify", "Update 2 to add/modify"],
    "confidence_adjustment": -0.1
}}

REFLECTION QUESTIONS:
1. Was the market regime assessment correct?
2. Did the technical indicators provide accurate guidance?
3. Were the identified risk factors the right ones to watch?
4. What unexpected factors emerged?
5. How should confidence be calibrated for similar setups?
6. What patterns should be added/updated in the playbook?

FOCUS ON:
- Specific, actionable learnings rather than generic advice
- Quantifiable patterns where possible
- Failure modes to avoid in the future
- Successful patterns to reinforce

Be honest about mistakes and clear about what would improve future decisions."#,
            current_date,
            original_decision.action,
            original_decision.confidence,
            original_decision.reasoning,
            original_decision.key_factors.join(", "),
            original_decision.risk_factors.join(", "),
            serde_json::to_string_pretty(market_state).unwrap_or("No market data".to_string()),
            serde_json::to_string_pretty(actual_outcome).unwrap_or("No outcome data".to_string())
        )
    }

    /// Generate context analysis prompt for understanding market regime
    pub fn context_analysis_prompt(
        market_data: &Value,
        news_data: &Value,
        sentiment_data: &Value,
        current_date: &str,
    ) -> String {
        format!(
            r#"You are analyzing the current market context to understand the trading environment.

DATE: {}

MARKET DATA:
{}

NEWS DATA:
{}

SENTIMENT DATA:
{}

TASK:
Analyze the current market context and provide a structured summary. Respond with valid JSON:

{{
    "market_regime": "TRENDING_UP" | "TRENDING_DOWN" | "RANGING" | "VOLATILE",
    "sentiment_summary": "Clear summary of market sentiment and key themes",
    "technical_summary": "Summary of technical patterns and indicators",
    "key_events": ["Event 1 affecting markets", "Event 2 affecting markets"]
}}

ANALYSIS FRAMEWORK:
1. Market Regime: Classify based on recent price action, volatility, and trend strength
2. Sentiment: Aggregate sentiment from news, social media, and options flow
3. Technical: Key support/resistance, momentum, volatility regime
4. Events: Catalysts, earnings, economic data, Fed events

Be concise but specific. Focus on actionable insights for options trading."#,
            current_date,
            serde_json::to_string_pretty(market_data).unwrap_or("No market data".to_string()),
            serde_json::to_string_pretty(news_data).unwrap_or("No news data".to_string()),
            serde_json::to_string_pretty(sentiment_data).unwrap_or("No sentiment data".to_string())
        )
    }

    /// Generate playbook query prompt for finding relevant patterns
    pub fn playbook_query_prompt(query: &str, context: &Value) -> String {
        format!(
            r#"You are searching the ACE trading playbook for patterns relevant to the current situation.

QUERY: {}

CURRENT CONTEXT:
{}

TASK:
Based on the query and current context, identify the most relevant trading patterns, rules, and insights that should be considered.

Respond with a list of relevant playbook entries that match this situation. Focus on:
1. Similar market conditions
2. Comparable setups or patterns  
3. Risk management rules that apply
4. Historical performance in similar contexts

Be specific and actionable."#,
            query,
            serde_json::to_string_pretty(context).unwrap_or("No context".to_string())
        )
    }

    /// Generate pattern extraction prompt for building playbook entries
    pub fn pattern_extraction_prompt(contexts: &[crate::vector::ContextEntry]) -> String {
        let contexts_text = contexts.iter()
            .enumerate()
            .map(|(i, ctx)| {
                format!(
                    "Context {}: Date: {}, Decision: {}, Confidence: {:.2}, Outcome: {}, Reasoning: {}",
                    i + 1,
                    ctx.timestamp.format("%Y-%m-%d"),
                    serde_json::to_string(&ctx.decision).unwrap_or("Unknown".to_string()),
                    ctx.confidence,
                    ctx.outcome.as_ref()
                        .map(|_| "Success")
                        .unwrap_or("Pending"),
                    ctx.reasoning
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        format!(
            r#"You are extracting trading patterns from historical ACE contexts to build playbook entries.

HISTORICAL CONTEXTS:
{}

TASK:
Analyze these contexts and extract actionable patterns for the trading playbook. Look for:

1. RECURRING SUCCESSFUL PATTERNS
   - What market conditions tend to produce successful trades?
   - What signals or combinations work well together?
   - What confidence levels correlate with success?

2. FAILURE MODES TO AVOID
   - What conditions lead to losses?
   - What overconfidence patterns exist?
   - What false signals to watch for?

3. MARKET REGIME RULES
   - How do strategies perform in different market conditions?
   - When do technical signals work best/worst?
   - What sentiment conditions favor different approaches?

Format each insight as: "Pattern: [Description] | Win Rate: [X/Y] | Avg Return: [Z%] | Key Conditions: [List]"

Focus on statistically meaningful patterns (at least 5 occurrences) and be specific about conditions."#,
            contexts_text
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_trading_decision_serialization() {
        let decision = TradingDecision {
            action: "BUY_CALLS".to_string(),
            confidence: 0.75,
            reasoning: "Strong bullish signals".to_string(),
            key_factors: vec!["Low VIX".to_string(), "Positive sentiment".to_string()],
            risk_factors: vec!["Market volatility".to_string()],
            similar_pattern_reference: Some("Pattern from 2024-01-15".to_string()),
            position_size_multiplier: 0.8,
        };

        let json = serde_json::to_string(&decision).expect("Failed to serialize");
        let deserialized: TradingDecision =
            serde_json::from_str(&json).expect("Failed to deserialize");

        assert_eq!(decision.action, deserialized.action);
        assert_eq!(decision.confidence, deserialized.confidence);
    }

    #[test]
    fn test_prompt_generation() {
        let market_state = json!({
            "spy_price": 450.0,
            "vix": 15.0
        });

        let ml_signals = json!({
            "technical_score": 0.7,
            "sentiment_score": 0.6
        });

        let prompt =
            ACEPrompts::morning_decision_prompt(&market_state, &ml_signals, &[], &[], "2025-01-15");

        assert!(prompt.contains("CURRENT DATE: 2025-01-15"));
        assert!(prompt.contains("spy_price"));
        assert!(prompt.contains("BUY_CALLS"));
        assert!(prompt.len() > 500); // Should be a substantial prompt
    }
}

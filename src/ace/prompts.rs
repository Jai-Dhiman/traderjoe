//! Prompt templates for ACE (Augmented Context Evolution) system
//! Structured prompts for daily decision making and context evolution

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Confidence score breakdown for structured scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceBreakdown {
    pub momentum_score: f32,        // 0-25: Trend strength and direction
    pub technical_score: f32,       // 0-25: Indicator alignment and signals
    pub regime_fit_score: f32,      // 0-20: Strategy-regime alignment
    pub volume_confirmation: f32,   // 0-15: Volume support for move
    pub risk_adjustment: f32,       // 0-15: Risk factors consideration
    pub total: f32,                 // Sum of above (0-100)
}

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
    #[serde(default)]
    pub confidence_breakdown: Option<ConfidenceBreakdown>, // Optional structured scoring
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

CRITICAL: Your response MUST be ONLY valid JSON with NO additional text, explanations, or markdown formatting.

Required JSON Schema:
{{
    "action": string,              // MUST be one of: "BUY_CALLS", "BUY_PUTS", or "STAY_FLAT"
    "confidence": number,          // MUST be between 0.0 and 1.0 (calculated from breakdown / 100)
    "reasoning": string,           // Clear explanation including score breakdown rationale
    "key_factors": array,          // Array of strings, at least 2 factors
    "risk_factors": array,         // Array of strings describing risks
    "similar_pattern_reference": string | null,  // Reference to past pattern or null
    "position_size_multiplier": number,  // MUST be between 0.0 and 1.0
    "confidence_breakdown": {{     // REQUIRED: Structured scoring components
        "momentum_score": number,       // 0-25 points
        "technical_score": number,      // 0-25 points
        "regime_fit_score": number,     // 0-20 points
        "volume_confirmation": number,  // 0-15 points
        "risk_adjustment": number,      // 0-15 points
        "total": number                 // Sum of above (must equal confidence * 100)
    }}
}}

Example valid response (using NEW scoring system with risk level 8-9/10):
{{
    "action": "BUY_CALLS",
    "confidence": 0.73,
    "reasoning": "Strong uptrend continuation setup with aggressive risk tolerance. Momentum: 25/28 (3-day uptrend with >0.7% daily gains). Technicals: 25/27 (MACD, RSI, MA aligned). Regime: 18/20 (trending market favors momentum). Volume: 13/15 (130% of 20d avg). Risk: 9/10 (0-1 DTE theta and near resistance are acceptable risks at 8-9/10 risk level). Total: 73/100.",
    "key_factors": ["3-day uptrend continuation", "Volume confirmation >130%", "All technical indicators aligned", "Trending regime favors momentum"],
    "risk_factors": ["0-1 DTE theta decay (manageable)", "Near resistance at $565 (breakout potential)"],
    "similar_pattern_reference": "2024-08-15: Similar momentum continuation resulted in +8% gain",
    "position_size_multiplier": 0.8,
    "confidence_breakdown": {{
        "momentum_score": 25,
        "technical_score": 25,
        "regime_fit_score": 18,
        "volume_confirmation": 13,
        "risk_adjustment": 9,
        "total": 73
    }}
}}

══════════════════════════════════════════════════════════
STRUCTURED CONFIDENCE SCORING SYSTEM
══════════════════════════════════════════════════════════

Your confidence MUST be calculated from these 5 components. Add up the scores to get your total (0-100), then divide by 100 for confidence (0.0-1.0).

NOTE: This system operates at RISK LEVEL 8-9/10 (aggressive). Weight opportunities higher than risks, but DO NOT ignore risks entirely. Risks should reduce confidence proportionally less than in conservative trading (e.g., -1 to -3 points per risk factor instead of -5 to -10). Volatility and momentum extremes create opportunities that may carry acceptable risks. The goal is to capture edge while acknowledging (not dismissing) risk.

1. MOMENTUM SCORE (0-28 points, increased from 25):
   Evaluate trend strength and direction:

   STRONG TRENDS (23-28 pts):
   • 3+ consecutive days same direction with >0.5% daily moves: 28 pts
   • 2-day trend with >0.7% total move + volume confirmation: 25 pts
   • Single day >1.0% move with strong volume: 23 pts

   MODERATE MOMENTUM (16-22 pts):
   • 2-day trend with 0.4-0.7% moves: 20 pts
   • Single day 0.5-1.0% move: 18 pts
   • Momentum building (acceleration visible): 16 pts

   WEAK/MIXED (6-15 pts):
   • Choppy action, no clear trend: 12 pts
   • Conflicting daily moves but range-bound: 8 pts
   • Consolidation after strong move: 6 pts

   NO MOMENTUM (0-5 pts):
   • Completely flat, no directional bias: 0 pts

   SCORING EXAMPLES:
   - SPY up 3 days (+1.2%, +0.8%, +0.9%) with volume: 28 pts
   - SPY down 2 days (-0.6%, -0.4%): 20 pts
   - SPY flat but near support/resistance: 10-12 pts

2. TECHNICAL SCORE (0-27 points, increased from 25):
   Evaluate indicator alignment:

   ALL ALIGNED (24-27 pts):
   • MACD, RSI, moving averages all point same direction: 27 pts
   • 4 of 5 indicators aligned: 25 pts
   • 3 indicators strongly aligned: 24 pts

   MOSTLY ALIGNED (19-23 pts):
   • 3 of 5 indicators aligned: 22 pts
   • 2 of 5 with strong signals: 20 pts
   • Divergences but interpretable as opportunity: 19 pts

   MIXED SIGNALS (11-18 pts):
   • Some bullish, some bearish but momentum building: 16 pts
   • Divergences present (can signal reversal): 13 pts
   • Consolidation pattern forming: 11 pts

   CONFLICTING (5-10 pts):
   • Major divergences without clear catalyst: 8 pts
   • Unclear technical picture: 5 pts

   NO SETUP (0-4 pts):
   • No technical edge visible: 0 pts

3. REGIME FIT SCORE (0-20 points, unchanged):
   How well does your strategy match the market regime?

   PERFECT FIT (18-20 pts):
   • TRENDING market + Momentum continuation strategy: 20 pts
   • RANGING market + Mean reversion strategy: 20 pts
   • VOLATILE market + Volatility expansion play: 18 pts

   GOOD FIT (13-17 pts):
   • TRENDING + Breakout strategy: 16 pts
   • RANGING + Premium selling: 15 pts
   • VOLATILE + Directional with tight stops: 13 pts

   NEUTRAL FIT (8-12 pts):
   • RANGING + Directional trade: 10 pts
   • TRENDING + Mean reversion: 8 pts

   POOR FIT (0-7 pts):
   • Fighting the regime: 5 pts
   • Unclear regime: 5 pts
   • High risk mismatch: 0 pts

4. VOLUME CONFIRMATION (0-15 points):
   Does volume support the move?

   STRONG CONFIRMATION (13-15 pts):
   • Volume >150% of 20-day average: 15 pts
   • Volume 120-150% of average: 13 pts

   MODERATE (8-12 pts):
   • Volume 100-120% of average: 10 pts
   • Volume 80-100% of average: 8 pts

   WEAK (0-7 pts):
   • Volume <80% of average: 5 pts
   • Very low volume: 0 pts

5. RISK ADJUSTMENT (0-10 points, reduced from 15):
   Account for SEVERE risk factors only (system operates at risk level 8-9/10):

   MINIMAL RISKS (9-10 pts):
   • Clean setup, no major concerns: 10 pts
   • 1-2 manageable risk factors (theta decay, near resistance): 9 pts

   MODERATE RISKS (6-8 pts):
   • 2-3 risk factors but edge still present: 8 pts
   • Volatility/uncertainty can create opportunities: 7 pts
   • One significant risk with mitigation plan: 6 pts

   ELEVATED RISKS (3-5 pts):
   • Multiple severe risks converging: 5 pts
   • Major structural headwinds: 3 pts

   EXTREME RISKS (0-2 pts):
   • Critical safety concerns (circuit breaker territory): 0 pts

NOTE: At risk level 8-9/10, common risks like "0-1 DTE theta", "near resistance", "no sentiment data" should reduce scores by 1-2 points each (not 5-10). Reserve larger penalties (3-5 points) for truly severe risks like market circuit breakers, extreme volatility, or systemic events. Acknowledge risks in reasoning but don't let them paralyze decision-making.

FINAL CONFIDENCE = (sum of 5 scores) / 100
New max total: 28 + 27 + 20 + 15 + 10 = 100 points

══════════════════════════════════════════════════════════
DIRECTIONAL STRATEGIES: MOMENTUM CONTINUATION
══════════════════════════════════════════════════════════

UPTREND CONTINUATION (BUY_CALLS):
Entry Conditions:
• SPY up >0.5% with volume >110% of 20-day average
• MACD bullish (histogram positive or MACD > signal)
• Price above key moving average (20-day or 50-day)
• 2-3 consecutive up days suggests established trend
• RSI 50-70 (strong but not overbought)

Expected Momentum Score: 20-25 pts
Example: Sept 2024 rally days - SPY +0.8% with volume spike = 22 pts

DOWNTREND CONTINUATION (BUY_PUTS):
Entry Conditions:
• SPY down >0.5% with volume >110% of 20-day average
• MACD bearish (histogram negative or MACD < signal)
• Price below key moving average
• 2-3 consecutive down days suggests established downtrend
• RSI 30-50 (weak but not oversold yet)

Expected Momentum Score: 20-25 pts
Example: Sept 6, 2024 selloff - SPY -1.74% = 25 pts for puts

TREND STRENGTH INDICATORS:
• Multi-day trends (3+) more reliable than single-day moves
• Volume confirmation REQUIRED - no volume, no conviction
• Larger daily moves (>0.7%) justify higher confidence
• Avoid late-stage trends (4+ days, RSI extremes)

REALISTIC CONFIDENCE TARGETS:
- Marginal Edge (45-55%): Weak trend, mixed signals, consider staying flat
- Decent Edge (55-65%): Clear trend but some risks, standard position
- Strong Edge (65-75%): Multi-day trend with volume, high conviction
- Exceptional Edge (75-85%): Perfect setup, rare (3-day trend + breakout + volume)

Professional 0DTE traders achieve 60-70% win rates. Don't overestimate edges.

══════════════════════════════════════════════════════════
ADDITIONAL RISK FACTORS (Include in Risk Adjustment Score)
══════════════════════════════════════════════════════════

TIME-OF-DAY RISK:
• 9:30-10:30 AM: Higher volatility, less predictable - reduce risk score by 2-3 pts
• 10:30 AM-3:30 PM: Optimal trading window - no adjustment
• 3:30-4:00 PM: Extreme gamma risk - reduce risk score by 5+ pts or avoid

DAY-OF-WEEK RISK:
• Monday-Thursday: Normal conditions - no adjustment
• Wednesday (FOMC/data): Event risk - reduce 2-3 pts
• Friday (OPEX): Weekend gap risk - reduce 2-3 pts

VIX ENVIRONMENT (factor into Risk Adjustment):
• VIX <15: Low vol, tighter ranges - neutral
• VIX 15-25: Normal environment - no adjustment
• VIX >25: High vol, wider swings - reduce 2-4 pts unless volatility play

══════════════════════════════════════════════════════════
SCORING EXAMPLES - Learn From These
══════════════════════════════════════════════════════════

EXAMPLE 1: Strong Uptrend Continuation (Sept 2, 2024)
SPY: +0.20%, 3-day uptrend, volume 110% avg, MACD bullish, near resistance

Momentum: 18 (2-day trend, moderate moves)
Technical: 20 (most indicators aligned)
Regime Fit: 16 (trending market, momentum strategy)
Volume: 10 (just above average)
Risk: 8 (near resistance, 0 DTE)
TOTAL: 72 = 0.72 confidence → BUY_CALLS

EXAMPLE 2: Bearish Breakdown (Sept 6, 2024)
SPY: -1.74%, volume 160% avg, MACD turning bearish, breaking support

Momentum: 25 (large single-day move)
Technical: 23 (breakdown confirmed)
Regime Fit: 18 (volatile, directional play)
Volume: 15 (strong confirmation)
Risk: 10 (clear setup, manageable risks)
TOTAL: 91 = 0.91 confidence → BUY_PUTS (exceptional setup)

EXAMPLE 3: Ranging/Choppy (Sept 4, 2024)
SPY: -0.20%, mixed signals, ranging regime, MACD/price divergence

Momentum: 8 (choppy, no clear trend)
Technical: 12 (mixed signals)
Regime Fit: 10 (ranging + trying directional)
Volume: 8 (below average)
Risk: 10 (multiple uncertainties)
TOTAL: 48 = 0.48 confidence → STAY_FLAT (below threshold)

══════════════════════════════════════════════════════════
TRADING EXECUTION RULES
══════════════════════════════════════════════════════════

CONFIDENCE THRESHOLDS (Regime-Based):
• TRENDING markets: Trade at 45%+ confidence (momentum reliable)
• RANGING markets: Trade at 55%+ confidence (harder setups)
• VOLATILE markets: Trade at 50%+ confidence (standard)

POSITION SIZING:
• 45-55%: Small position (fractional Kelly scales down automatically)
• 55-65%: Standard position
• 65-75%: Above-average position
• 75%+: High conviction (rare, don't oversize)

DIRECTIONAL BIAS:
• Be willing to trade BOTH directions (calls AND puts)
• Downtrends are tradeable - don't just wait for bullish setups
• Sept 6 -1.74% selloff = perfect put setup, not a "stay flat" day

CRITICAL RULES:
• Your reasoning MUST include the score breakdown
• Confidence must equal (total score / 100)
• Reference playbook entries when applicable
• Be explicit about momentum continuation vs other strategies
• Don't force trades - STAY_FLAT is valid when scores are low

Focus on high-probability momentum continuation setups with clear trends and volume confirmation."#,
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
    "what_worked": ["Aspect 1 that was correct"],
    "what_failed": ["Aspect 1 that was wrong"],
    "lessons_learned": ["Most important lesson"],
    "playbook_updates": ["Critical update to add"],
    "confidence_adjustment": -0.1
}}

CRITICAL CONSTRAINTS - ACE Grow-and-Refine Mechanism:
• MAXIMUM 3-5 total items across ALL arrays (not per array)
• Focus on ONE most impactful learning, not exhaustive list
• Only add insights that are genuinely novel or significantly refine existing knowledge
• Be highly selective - quality over quantity
• Most days should produce 1-3 deltas, not 10+

REFLECTION QUESTIONS:
1. What was THE MOST important factor that determined today's outcome?
2. Is there ONE critical lesson that would improve future decisions?
3. What single pattern (if any) should be reinforced or avoided?

GUIDELINES:
- If outcome was expected/routine, you may return EMPTY arrays (0 deltas)
- Only populate "what_worked" if something worked exceptionally well
- Only populate "what_failed" if there was a clear, avoidable mistake
- Only add "lessons_learned" if you learned something genuinely new
- Only add "playbook_updates" if a rule needs to change
- Specific, actionable, novel insights only - no generic advice
- Be brutally selective - the playbook is for critical learnings, not daily logs

EXAMPLE (Good - Selective):
{{
    "what_worked": [],
    "what_failed": ["Entered call position despite VIX >30, resulted in -15% loss"],
    "lessons_learned": ["VOLATILE regime with VIX >30 requires 70%+ confidence for calls, not 60%"],
    "playbook_updates": [],
    "confidence_adjustment": -0.1
}}

EXAMPLE (Bad - Too Verbose):
{{
    "what_worked": ["Good entry timing", "Volume confirmation", "Technical setup", "Risk management"],
    "what_failed": ["Didn't account for news", "Exit was late", "Position too large"],
    "lessons_learned": ["Entry timing matters", "Check news", "Use stops", "Size appropriately"],
    "playbook_updates": ["Update entry rules", "Add news checks", "Adjust sizing"],
    "confidence_adjustment": -0.1
}}

Be honest, selective, and focused. Most days produce 0-2 learnings, not 5-10."#,
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

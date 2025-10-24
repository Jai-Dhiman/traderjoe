//! Evening review orchestrator
//! Processes trading outcomes and updates ACE playbook through reflection

use anyhow::{Context as AnyhowContext, Result};
use chrono::Utc;
use serde_json::{json, Value};
use sqlx::PgPool;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::{
    ace::{
        playbook::PlaybookDAO, reflector::TradingOutcome, ContextDAO, CurationSummary, Curator,
        TradingDecision,
    },
    config::Config,
    data::MarketDataClient,
    llm::LLMClient,
    trading::PaperTradingEngine,
};

/// Result of evening review analysis
#[derive(Debug, Clone)]
pub struct EveningReviewResult {
    /// Context ID that was reviewed
    pub context_id: Uuid,
    /// Trading outcome data
    pub outcome: TradingOutcome,
    /// Curation summary showing playbook changes
    pub curation_summary: CurationSummary,
    /// Whether reflection was successful
    pub success: bool,
    /// Any error messages or notes
    pub notes: Option<String>,
}

impl EveningReviewResult {
    /// Display human-readable summary of the review
    pub fn display_summary(&self) {
        println!("\n╔════════════════════════════════════════════════════════════╗");
        println!(
            "║          EVENING REVIEW - {}           ║",
            Utc::now().format("%Y-%m-%d")
        );
        println!("╚════════════════════════════════════════════════════════════╝\n");

        println!("📊 TRADE OUTCOME:");
        println!("   Context ID: {}", self.context_id);
        println!(
            "   Result: {}",
            if self.outcome.win {
                "✅ WIN"
            } else {
                "❌ LOSS"
            }
        );
        println!(
            "   P&L: ${:.2} ({:+.2}%)",
            self.outcome.pnl_value, self.outcome.pnl_pct
        );
        println!(
            "   Entry: ${:.2} → Exit: ${:.2}",
            self.outcome.entry_price, self.outcome.exit_price
        );
        println!("   Duration: {:.1} hours", self.outcome.duration_hours);

        if let Some(mfe) = self.outcome.mfe {
            println!("   Max Favorable Excursion: ${:.2}", mfe);
        }
        if let Some(mae) = self.outcome.mae {
            println!("   Max Adverse Excursion: ${:.2}", mae);
        }

        if let Some(notes) = &self.outcome.notes {
            println!("   Notes: {}", notes);
        }

        println!("\n");
        self.curation_summary.display_summary();

        if let Some(notes) = &self.notes {
            println!("\n📝 Review Notes:");
            println!("   {}", notes);
        }

        println!("\n═══════════════════════════════════════════════════════════");
    }
}

/// Evening review orchestrator
pub struct EveningOrchestrator {
    _pool: PgPool,
    _config: Config,
    market_client: MarketDataClient,
    context_dao: ContextDAO,
    _playbook_dao: PlaybookDAO,
    curator: Curator,
    _paper_trading: PaperTradingEngine,
}

impl EveningOrchestrator {
    /// Create new evening orchestrator
    pub async fn new(pool: PgPool, config: Config) -> Result<Self> {
        info!("Initializing Evening Orchestrator");

        let market_client = MarketDataClient::new(pool.clone());
        let context_dao = ContextDAO::new(pool.clone());
        let playbook_dao = PlaybookDAO::new(pool.clone());
        let llm_client = LLMClient::from_config(&config).await?;

        // Create curator with default config
        let curator = Curator::new(
            playbook_dao.clone(),
            llm_client,
            None, // Use default curator config
            None, // Use default delta engine config
        )
        .await?;

        let paper_trading = PaperTradingEngine::new(pool.clone());

        info!("Evening Orchestrator initialized successfully");

        Ok(Self {
            _pool: pool,
            _config: config,
            market_client,
            context_dao,
            _playbook_dao: playbook_dao,
            curator,
            _paper_trading: paper_trading,
        })
    }

    /// Run evening review for the most recent context
    pub async fn review_latest(&self) -> Result<EveningReviewResult> {
        info!("🌙 Starting evening review for latest context");

        // Get the most recent context that hasn't been reviewed yet
        let context = self
            .context_dao
            .get_latest_without_outcome()
            .await
            .context("Failed to get latest context without outcome")?
            .ok_or_else(|| {
                warn!("No unreviewed contexts found");
                anyhow::anyhow!("No contexts available for review")
            })?;

        info!("Found context {} from {}", context.id, context.timestamp);

        self.review_context(context.id).await
    }

    /// Run evening review for a specific context
    pub async fn review_context(&self, context_id: Uuid) -> Result<EveningReviewResult> {
        info!("🌙 Starting evening review for context {}", context_id);

        // Step 1: Get the original context
        let context = self
            .context_dao
            .get_by_id(context_id)
            .await
            .context("Failed to get context")?
            .ok_or_else(|| anyhow::anyhow!("Context {} not found", context_id))?;

        info!("📋 Retrieved context from {}", context.timestamp);

        // Step 2: Parse the original decision
        let decision = self.parse_decision_from_context(&context)?;

        // Step 3: Compute actual outcome
        info!("📊 Computing trading outcome...");
        let outcome = self.compute_outcome(&context, &decision).await?;

        info!(
            "Outcome: {} with P&L ${:.2} ({:+.2}%)",
            if outcome.win { "WIN" } else { "LOSS" },
            outcome.pnl_value,
            outcome.pnl_pct
        );

        // Step 4: Run reflection and update playbook
        info!("🧠 Running ACE reflection...");
        let curation_summary = self
            .curator
            .reflect_and_update_playbook(
                decision.clone(),
                context.market_state.clone(),
                outcome.clone(),
                context_id,
            )
            .await
            .context("Failed to reflect and update playbook")?;

        info!(
            "Playbook updated: {} added, {} updated, {} removed",
            curation_summary.bullets_added,
            curation_summary.bullets_updated,
            curation_summary.bullets_removed
        );

        // Step 5: Update context with outcome
        let outcome_json = json!({
            "pnl_value": outcome.pnl_value,
            "pnl_pct": outcome.pnl_pct,
            "win": outcome.win,
            "entry_price": outcome.entry_price,
            "exit_price": outcome.exit_price,
            "duration_hours": outcome.duration_hours,
            "mfe": outcome.mfe,
            "mae": outcome.mae,
            "notes": outcome.notes,
            "reviewed_at": Utc::now().to_rfc3339(),
        });

        self.context_dao
            .update_outcome(context_id, &outcome_json)
            .await
            .context("Failed to update context outcome")?;

        info!("✅ Evening review completed successfully");

        Ok(EveningReviewResult {
            context_id,
            outcome,
            curation_summary,
            success: true,
            notes: None,
        })
    }

    /// Review all pending contexts (for batch processing)
    pub async fn review_all_pending(&self) -> Result<Vec<EveningReviewResult>> {
        info!("🌙 Starting batch evening review for all pending contexts");

        let pending_contexts = self
            .context_dao
            .get_all_without_outcome()
            .await
            .context("Failed to get pending contexts")?;

        info!(
            "Found {} pending contexts to review",
            pending_contexts.len()
        );

        let mut results = Vec::new();

        for context in pending_contexts {
            info!("Processing context {}", context.id);

            match self.review_context(context.id).await {
                Ok(result) => {
                    info!("✅ Context {} reviewed successfully", context.id);
                    results.push(result);
                }
                Err(e) => {
                    error!("❌ Failed to review context {}: {}", context.id, e);
                    // Continue with other contexts instead of failing completely
                    results.push(EveningReviewResult {
                        context_id: context.id,
                        outcome: TradingOutcome::from_pnl(0.0, 0.0, 0.0, 0.0),
                        curation_summary: crate::ace::CurationSummary::empty(),
                        success: false,
                        notes: Some(format!("Review failed: {}", e)),
                    });
                }
            }
        }

        info!(
            "✅ Batch review completed: {} contexts processed",
            results.len()
        );

        Ok(results)
    }

    /// Parse trading decision from context
    fn parse_decision_from_context(
        &self,
        context: &crate::ace::context::AceContext,
    ) -> Result<TradingDecision> {
        // The decision is stored in the context's decision field
        let decision = context
            .decision
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Context has no decision"))?;

        Ok(TradingDecision {
            action: decision
                .get("action")
                .and_then(|v| v.as_str())
                .unwrap_or("STAY_FLAT")
                .to_string(),
            confidence: decision
                .get("confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5) as f32,
            reasoning: decision
                .get("reasoning")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            key_factors: decision
                .get("key_factors")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .map(|s| s.to_string())
                        .collect()
                })
                .unwrap_or_default(),
            risk_factors: decision
                .get("risk_factors")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .map(|s| s.to_string())
                        .collect()
                })
                .unwrap_or_default(),
            similar_pattern_reference: decision
                .get("similar_pattern_reference")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            position_size_multiplier: decision
                .get("position_size_multiplier")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0) as f32,
        })
    }

    /// Compute actual trading outcome from executed paper trades or hypothetical analysis
    async fn compute_outcome(
        &self,
        context: &crate::ace::context::AceContext,
        decision: &TradingDecision,
    ) -> Result<TradingOutcome> {
        // First, try to get actual trade execution data from paper trading engine
        let paper_trade_outcome = self._paper_trading.get_context_outcome(context.id).await;

        if let Ok(Some(outcome)) = paper_trade_outcome {
            info!("Using actual paper trade outcome for context {}", context.id);
            return Ok(outcome);
        }

        info!("No executed paper trade found for context {}, computing hypothetical outcome", context.id);

        // Extract symbol from market state
        let symbol = context
            .market_state
            .get("symbol")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Missing symbol in market state - this indicates a data quality issue"
                )
            })?;

        // Get entry price (from context timestamp)
        let entry_price = context
            .market_state
            .get("market_data")
            .and_then(|md| md.get("latest_price"))
            .and_then(|c| c.as_f64())
            .ok_or_else(|| anyhow::anyhow!("Missing entry price in market data"))?;

        // Get current/exit price
        info!("Fetching current price for {} to compute outcome", symbol);
        let current_data = self
            .market_client
            .fetch_latest(symbol)
            .await
            .context("Failed to fetch current market data")?;

        let exit_price = current_data
            .get("close")
            .and_then(|c| c.as_f64())
            .ok_or_else(|| anyhow::anyhow!("Missing close price in current data"))?;

        // Calculate time duration
        let now = Utc::now();
        let duration_hours = (now - context.timestamp).num_seconds() as f64 / 3600.0;

        // Determine trade direction and calculate P&L
        let (pnl_value, pnl_pct) = match decision.action.as_str() {
            "BUY_CALLS" | "BULLISH" => {
                // Profit if price went up
                let pct_change = ((exit_price - entry_price) / entry_price) * 100.0;
                let value = pct_change * 10.0; // Simplified: assume $10 per 1% move
                (value, pct_change)
            }
            "BUY_PUTS" | "BEARISH" => {
                // Profit if price went down
                let pct_change = ((entry_price - exit_price) / entry_price) * 100.0;
                let value = pct_change * 10.0;
                (value, pct_change)
            }
            _ => {
                // STAY_FLAT or unknown action
                (0.0, 0.0)
            }
        };

        let win = pnl_value > 0.0;

        // Try to get MFE/MAE from paper trading engine if available
        let (mfe, mae) = self
            .get_excursions(context.id)
            .await
            .unwrap_or((None, None));

        let notes = if decision.action == "STAY_FLAT" {
            Some("No trade taken (stayed flat)".to_string())
        } else {
            None
        };

        Ok(TradingOutcome {
            pnl_value,
            pnl_pct,
            mfe,
            mae,
            win,
            entry_price,
            exit_price,
            duration_hours,
            notes,
        })
    }

    /// Get maximum favorable/adverse excursion if tracked
    async fn get_excursions(&self, context_id: Uuid) -> Result<(Option<f64>, Option<f64>)> {
        // This would query the paper trading engine for tracked excursions
        // For now, return None - can be enhanced later
        debug!(
            "Excursion tracking not yet implemented for context {}",
            context_id
        );
        Ok((None, None))
    }

    /// Get summary statistics for recent reviews
    pub async fn get_review_stats(&self, days: i64) -> Result<Value> {
        let cutoff = Utc::now() - chrono::Duration::days(days);

        let stats = self.context_dao.get_outcome_stats(cutoff).await?;

        Ok(json!({
            "period_days": days,
            "total_trades": stats.get("total_trades"),
            "wins": stats.get("wins"),
            "losses": stats.get("losses"),
            "win_rate": stats.get("win_rate"),
            "avg_pnl_pct": stats.get("avg_pnl_pct"),
            "total_pnl": stats.get("total_pnl"),
            "avg_duration_hours": stats.get("avg_duration_hours"),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evening_review_result_display() {
        let result = EveningReviewResult {
            context_id: Uuid::new_v4(),
            outcome: TradingOutcome::from_pnl(100.0, 105.0, 50.0, 2.5)
                .with_excursions(Some(75.0), Some(-25.0))
                .with_notes("Good execution".to_string()),
            curation_summary: crate::ace::CurationSummary::empty(),
            success: true,
            notes: Some("Test review".to_string()),
        };

        // Should not panic
        result.display_summary();
    }

    #[tokio::test]
    #[ignore = "Integration test - requires database setup"]
    async fn test_parse_decision_from_context() {
        // TODO: Implement integration test with database fixture
        // This would test parsing trading decisions from ACE contexts
    }
}

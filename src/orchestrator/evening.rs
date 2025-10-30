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
    trading::{black_scholes, AccountManager, PaperTradingEngine},
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
    account_manager: AccountManager,
}

impl EveningOrchestrator {
    /// Create new evening orchestrator
    pub async fn new(pool: PgPool, config: Config) -> Result<Self> {
        info!("Initializing Evening Orchestrator");

        let market_client = MarketDataClient::new(pool.clone());
        let context_dao = ContextDAO::new(pool.clone());
        let playbook_dao = PlaybookDAO::new(pool.clone());
        let llm_client = LLMClient::from_config(&config).await?;

        // Determine the timestamp for bullet creation (backtest vs live)
        let created_at = config.backtest_date.map(|date| {
            date.and_hms_opt(0, 0, 0)
                .expect("Invalid time")
                .and_utc()
        });

        // Create curator with backtest timestamp if in backtest mode
        let curator = Curator::new(
            playbook_dao.clone(),
            llm_client,
            None, // Use default curator config
            None, // Use default delta engine config
            created_at, // Use backtest date for bullet timestamps in backtest mode
        )
        .await?;

        let paper_trading = PaperTradingEngine::new(pool.clone());
        let account_manager = AccountManager::new(pool.clone());

        info!("Evening Orchestrator initialized successfully");

        Ok(Self {
            _pool: pool,
            _config: config,
            market_client,
            context_dao,
            _playbook_dao: playbook_dao,
            curator,
            _paper_trading: paper_trading,
            account_manager,
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
            "reviewed_at": self._config.get_effective_datetime().to_rfc3339(),
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
        // For backtest mode: close any open trades for this context before computing outcome
        if let Some(true) = self._config.backtest_mode {
            self.close_open_context_trades(context).await?;
        }

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
            .fetch_latest_with_date(symbol, self._config.backtest_date)
            .await
            .context("Failed to fetch current market data")?;

        let exit_price = current_data
            .get("close")
            .and_then(|c| c.as_f64())
            .ok_or_else(|| anyhow::anyhow!("Missing close price in current data"))?;

        // Calculate time duration
        let now = self._config.get_effective_datetime();
        let duration_hours = (now - context.timestamp).num_seconds() as f64 / 3600.0;

        // Extract volatility from market_state for Black-Scholes
        let volatility_pct = context
            .market_state
            .get("market_data")
            .and_then(|d| d.get("volatility_20d"))
            .and_then(|v| v.as_f64())
            .unwrap_or(2.0); // Default 2% daily volatility if not available

        // Convert to annualized volatility for Black-Scholes (daily_vol * sqrt(252))
        let annualized_vol = (volatility_pct / 100.0) * f64::sqrt(252.0);

        // Risk-free rate (approximate 5% annually, or daily rate)
        let risk_free_rate = 0.05;

        // Time to expiry in years (for 0-1 DTE, this is very small)
        let entry_time_to_expiry = 1.0 / 365.0; // 1 day at entry
        let exit_time_to_expiry = f64::max(0.0, entry_time_to_expiry - (duration_hours / 24.0 / 365.0));

        // Determine trade direction and calculate P&L using Black-Scholes
        let (pnl_value, pnl_pct, win, notes) = match decision.action.as_str() {
            "BUY_CALLS" | "BULLISH" => {
                // Calculate option prices using Black-Scholes
                // Use 30-delta call (typical for 0-1 DTE strategies)
                let strike = black_scholes::strike_from_delta(
                    entry_price,
                    0.30,
                    annualized_vol,
                    entry_time_to_expiry,
                    true,
                );

                let entry_option_price = black_scholes::call_price(
                    entry_price,
                    strike,
                    risk_free_rate,
                    annualized_vol,
                    entry_time_to_expiry,
                );

                let exit_option_price = black_scholes::call_price(
                    exit_price,
                    strike,
                    risk_free_rate,
                    annualized_vol,
                    exit_time_to_expiry,
                );

                // Calculate P&L based on option price movement
                // Assume 10 contracts * 100 shares/contract = 1000 shares equivalent
                let contracts = 10.0;
                let pnl_per_contract = (exit_option_price - entry_option_price) * 100.0; // Options are per 100 shares
                let total_pnl = pnl_per_contract * contracts;
                let pnl_pct = (total_pnl / (entry_option_price * 100.0 * contracts)) * 100.0;

                let win = total_pnl > 0.0;
                let notes_text = format!(
                    "Call option: strike=${:.2}, entry=${:.2}, exit=${:.2}, {} contracts",
                    strike, entry_option_price, exit_option_price, contracts as i32
                );

                (total_pnl, pnl_pct, win, Some(notes_text))
            }
            "BUY_PUTS" | "BEARISH" => {
                // Calculate put option prices using Black-Scholes
                // Use 30-delta put (typical for 0-1 DTE strategies)
                let strike = black_scholes::strike_from_delta(
                    entry_price,
                    -0.30,
                    annualized_vol,
                    entry_time_to_expiry,
                    false,
                );

                let entry_option_price = black_scholes::put_price(
                    entry_price,
                    strike,
                    risk_free_rate,
                    annualized_vol,
                    entry_time_to_expiry,
                );

                let exit_option_price = black_scholes::put_price(
                    exit_price,
                    strike,
                    risk_free_rate,
                    annualized_vol,
                    exit_time_to_expiry,
                );

                // Calculate P&L
                let contracts = 10.0;
                let pnl_per_contract = (exit_option_price - entry_option_price) * 100.0;
                let total_pnl = pnl_per_contract * contracts;
                let pnl_pct = (total_pnl / (entry_option_price * 100.0 * contracts)) * 100.0;

                let win = total_pnl > 0.0;
                let notes_text = format!(
                    "Put option: strike=${:.2}, entry=${:.2}, exit=${:.2}, {} contracts",
                    strike, entry_option_price, exit_option_price, contracts as i32
                );

                (total_pnl, pnl_pct, win, Some(notes_text))
            }
            "STAY_FLAT" | "FLAT" => {
                // Calculate opportunity cost of not trading
                let price_move_pct = ((exit_price - entry_price) / entry_price) * 100.0;
                let abs_move = price_move_pct.abs();

                // Evaluate if staying flat was the correct decision
                let (opportunity_cost, was_correct, reason) = if abs_move < 0.5 {
                    // Market barely moved - staying flat was correct
                    (0.0, true, format!(
                        "STAY_FLAT was correct: Market moved only {:+.2}% (< 0.5% threshold). No significant opportunity missed.",
                        price_move_pct
                    ))
                } else {
                    // Market moved significantly - calculate missed opportunity using Black-Scholes
                    // Determine what the best trade would have been based on price direction
                    let (is_call, strike) = if price_move_pct > 0.0 {
                        // Price went up - we should have bought calls
                        let strike = black_scholes::strike_from_delta(
                            entry_price,
                            0.30,
                            annualized_vol,
                            entry_time_to_expiry,
                            true,
                        );
                        (true, strike)
                    } else {
                        // Price went down - we should have bought puts
                        let strike = black_scholes::strike_from_delta(
                            entry_price,
                            -0.30,
                            annualized_vol,
                            entry_time_to_expiry,
                            false,
                        );
                        (false, strike)
                    };

                    // Calculate what we could have made
                    let (entry_price_opt, exit_price_opt) = if is_call {
                        (
                            black_scholes::call_price(entry_price, strike, risk_free_rate, annualized_vol, entry_time_to_expiry),
                            black_scholes::call_price(exit_price, strike, risk_free_rate, annualized_vol, exit_time_to_expiry),
                        )
                    } else {
                        (
                            black_scholes::put_price(entry_price, strike, risk_free_rate, annualized_vol, entry_time_to_expiry),
                            black_scholes::put_price(exit_price, strike, risk_free_rate, annualized_vol, exit_time_to_expiry),
                        )
                    };

                    let contracts = 10.0;
                    let missed_pnl = (exit_price_opt - entry_price_opt) * 100.0 * contracts;

                    (
                        -missed_pnl,
                        false,
                        format!(
                            "STAY_FLAT was suboptimal: Market moved {:+.2}% (>= 0.5%). Missed {} opportunity: ${:.2}",
                            price_move_pct,
                            if is_call { "CALL" } else { "PUT" },
                            missed_pnl
                        )
                    )
                };

                let detailed_notes = format!(
                    "{} Entry: ${:.2}, Exit: ${:.2}, Duration: {:.1}h, Vol: {:.1}%",
                    reason, entry_price, exit_price, duration_hours, volatility_pct
                );

                (opportunity_cost, price_move_pct, was_correct, Some(detailed_notes))
            }
            _ => {
                // Unknown action
                (0.0, 0.0, false, Some("Unknown action type".to_string()))
            }
        };

        // Try to get MFE/MAE from paper trading engine if available (only for actual trades)
        let (mfe, mae) = if decision.action != "STAY_FLAT" && decision.action != "FLAT" {
            self.get_excursions(context.id)
                .await
                .unwrap_or((None, None))
        } else {
            (None, None)
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

    /// Close any open trades associated with this context (for backtest mode)
    async fn close_open_context_trades(
        &self,
        context: &crate::ace::context::AceContext,
    ) -> Result<()> {
        use crate::trading::RiskManager;

        // Get all open positions
        let open_positions = self._paper_trading.get_open_positions().await?;

        // Find trades for this context
        let context_trades: Vec<_> = open_positions
            .iter()
            .filter(|trade| trade.context_id == Some(context.id))
            .collect();

        if context_trades.is_empty() {
            debug!("No open trades found for context {}", context.id);
            return Ok(());
        }

        info!("Found {} open trade(s) for context {}, checking for stop-loss/take-profit",
              context_trades.len(), context.id);

        // Initialize risk manager for stop-loss/take-profit checks
        let risk_manager = RiskManager::with_defaults();

        // Get symbol from context
        let symbol = context
            .market_state
            .get("symbol")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing symbol in market state"))?;

        // Fetch exit price and calculate exit time for backtest mode
        let (current_data, exit_time) = if let Some(backtest_date) = self._config.backtest_date {
            info!("Fetching exit price for backtest date: {}", backtest_date);
            let data = self.market_client
                .fetch_for_date(symbol, backtest_date)
                .await
                .context("Failed to fetch exit price for closing trade")?;

            // Use market close time (4:00 PM ET = 16:00) for backtest exit
            let historical_exit_time = backtest_date
                .and_hms_opt(16, 0, 0)
                .expect("Invalid time")
                .and_utc();

            info!("Using historical exit time for backtest: {}", historical_exit_time.format("%Y-%m-%d %H:%M"));
            (data, Some(historical_exit_time))
        } else {
            let data = self.market_client
                .fetch_latest(symbol)
                .await
                .context("Failed to fetch exit price for closing trade")?;
            (data, None)
        };

        let exit_price = current_data
            .get("close")
            .and_then(|c| c.as_f64())
            .ok_or_else(|| anyhow::anyhow!("Missing close price in market data"))?;

        // Close each trade with risk management checks
        for trade in context_trades {
            let is_option = true; // SPY options

            // Check for stop-loss trigger
            let stop_loss_hit = risk_manager
                .check_stop_loss(trade.entry_price, exit_price, &trade.trade_type, is_option)?;

            // Check for take-profit trigger
            let take_profit_hit = risk_manager
                .check_take_profit(trade.entry_price, exit_price, &trade.trade_type, is_option)?;

            // Check for trailing stop trigger
            let trailing_stop_hit = risk_manager
                .check_trailing_stop(
                    trade.entry_price,
                    exit_price,
                    trade.max_favorable_excursion,
                    &trade.trade_type
                )?;

            // Check for time-based exit (same-day trades at market close)
            // Use current time if exit_time is None (live trading)
            let current_time_for_check = exit_time.unwrap_or_else(|| chrono::Utc::now());
            let time_based_exit = risk_manager
                .check_time_based_exit(trade.entry_time, current_time_for_check)?;

            // Determine exit reason based on risk management checks
            // Priority: stop-loss > trailing stop > time-based > take-profit > auto
            let exit_reason = if stop_loss_hit {
                crate::trading::ExitReason::StopLoss
            } else if trailing_stop_hit {
                crate::trading::ExitReason::TrailingStop
            } else if time_based_exit {
                crate::trading::ExitReason::TimeBasedExit
            } else if take_profit_hit {
                crate::trading::ExitReason::TakeProfit
            } else {
                crate::trading::ExitReason::AutoExit
            };

            info!(
                "Closing trade {} at exit price ${:.2} (reason: {:?})",
                trade.id, exit_price, exit_reason
            );

            let closed_trade = self._paper_trading
                .exit_trade_with_time(
                    trade.id,
                    exit_price,
                    exit_reason,
                    exit_time,
                )
                .await?;

            // Update account balance with P&L from this trade
            self.account_manager.update_balance(&closed_trade).await?;
            info!("Updated account balance after trade {} closed", trade.id);
        }

        Ok(())
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

    /// Review a specific context at a historical date (used in backtesting)
    /// This computes the actual outcome using next day's market data
    pub async fn review_context_at_date(
        &self,
        context_id: Uuid,
        outcome_date: chrono::NaiveDate,
    ) -> Result<EveningReviewResult> {
        info!(
            "🌙 Starting historical review for context {} using outcome from {}",
            context_id, outcome_date
        );

        // For backtest mode, we run the regular review
        // The market_client will fetch data for the outcome_date from config
        self.review_context(context_id).await
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

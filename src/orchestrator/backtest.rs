//! ACE Backtest Orchestrator
//! Runs full ACE pipeline simulation on historical data with real learning

use anyhow::{Context as AnyhowContext, Result};
use chrono::{Datelike, Duration, NaiveDate, Weekday};
use sqlx::PgPool;
use tracing::{info, warn};
use uuid::Uuid;

use crate::{
    ace::{ContextDAO, TradingDecision},
    config::Config,
    trading::AccountManager,
};

/// Result of a single day's simulation
#[derive(Debug, Clone)]
pub struct DayResult {
    pub date: NaiveDate,
    pub decision: Option<TradingDecision>,
    pub executed: bool,
    pub pnl: Option<f64>,
    pub pnl_pct: Option<f64>,
    pub won: Option<bool>,
    pub playbook_bullets_added: usize,
    pub notes: String,
}

/// Complete backtest results
#[derive(Debug, Clone)]
pub struct BacktestResults {
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub symbol: String,
    pub total_days: usize,
    pub trading_days: usize,
    pub total_trades: usize,
    pub wins: usize,
    pub losses: usize,
    pub starting_capital: f64,
    pub ending_capital: f64,
    pub total_return: f64,
    pub total_return_pct: f64,
    pub max_drawdown_pct: f64,
    pub sharpe_ratio: f64,
    pub playbook_bullets_created: usize,
    pub day_results: Vec<DayResult>,
}

impl BacktestResults {
    /// Display comprehensive backtest summary
    pub fn display_summary(&self) {
        println!("\n╔════════════════════════════════════════════════════════════╗");
        println!("║          ACE BACKTEST RESULTS                              ║");
        println!("╚════════════════════════════════════════════════════════════╝\n");

        println!("📅 Period: {} to {} ({} trading days)",
                 self.start_date, self.end_date, self.trading_days);
        println!("💹 Symbol: {}", self.symbol);
        println!();

        println!("═══════════════════════════════════════════════════════════");
        println!("📊 TRADING PERFORMANCE");
        println!("═══════════════════════════════════════════════════════════\n");

        println!("  Total Trades: {}", self.total_trades);
        if self.total_trades > 0 {
            let win_rate = (self.wins as f64 / self.total_trades as f64) * 100.0;
            println!("  Wins: {} ({:.1}%)", self.wins, win_rate);
            println!("  Losses: {} ({:.1}%)", self.losses, 100.0 - win_rate);
        }

        println!();
        println!("═══════════════════════════════════════════════════════════");
        println!("💰 FINANCIAL METRICS");
        println!("═══════════════════════════════════════════════════════════\n");

        println!("  Starting Capital: ${:.2}", self.starting_capital);
        println!("  Ending Capital: ${:.2}", self.ending_capital);
        println!("  Total Return: ${:+.2} ({:+.2}%)",
                 self.total_return, self.total_return_pct);
        println!("  Max Drawdown: {:.2}%", self.max_drawdown_pct);
        println!("  Sharpe Ratio: {:.2}", self.sharpe_ratio);

        println!();
        println!("═══════════════════════════════════════════════════════════");
        println!("🧠 ACE LEARNING EVOLUTION");
        println!("═══════════════════════════════════════════════════════════\n");

        println!("  Playbook Bullets Created: {}", self.playbook_bullets_created);

        println!();
        println!("═══════════════════════════════════════════════════════════");
        println!("✅ PHASE 1 SUCCESS CRITERIA");
        println!("═══════════════════════════════════════════════════════════\n");

        let win_rate = if self.total_trades > 0 {
            (self.wins as f64 / self.total_trades as f64) * 100.0
        } else {
            0.0
        };

        println!("  Win Rate > 55%: {} ({:.1}%)",
                 if win_rate > 55.0 { "✅ PASS" } else { "❌ FAIL" }, win_rate);
        println!("  Sharpe > 1.5: {} ({:.2})",
                 if self.sharpe_ratio > 1.5 { "✅ PASS" } else { "❌ FAIL" }, self.sharpe_ratio);
        println!("  Max DD < 15%: {} ({:.2}%)",
                 if self.max_drawdown_pct < 15.0 { "✅ PASS" } else { "❌ FAIL" }, self.max_drawdown_pct);

        println!("\n═══════════════════════════════════════════════════════════\n");
    }
}

/// ACE Backtest Orchestrator
pub struct BacktestOrchestrator {
    pool: PgPool,
    config: Config,
    start_date: NaiveDate,
    end_date: NaiveDate,
    skip_sentiment: bool,
}

impl BacktestOrchestrator {
    /// Create new backtest orchestrator
    pub fn new(
        pool: PgPool,
        config: Config,
        start_date: NaiveDate,
        end_date: NaiveDate,
        skip_sentiment: bool,
    ) -> Self {
        Self {
            pool,
            config,
            start_date,
            end_date,
            skip_sentiment,
        }
    }

    /// Run complete backtest simulation
    pub async fn run(&self, symbol: &str) -> Result<BacktestResults> {
        info!(
            "🚀 Starting ACE backtest from {} to {}",
            self.start_date, self.end_date
        );

        println!("\n╔════════════════════════════════════════════════════════════╗");
        println!("║          ACE BACKTEST SIMULATION                           ║");
        println!("╚════════════════════════════════════════════════════════════╝\n");

        println!("📅 Period: {} to {}", self.start_date, self.end_date);
        println!("💹 Symbol: {}", symbol);
        println!("🧠 Strategy: Full ACE Pipeline with Learning");
        println!("💰 Starting Capital: $10,000");
        println!("🎭 Sentiment: {}", if self.skip_sentiment { "Disabled" } else { "Historical Reddit" });
        println!();
        println!("═══════════════════════════════════════════════════════════\n");

        // Get trading days (weekdays only)
        let trading_days = self.get_trading_days();
        info!("Found {} trading days in backtest period", trading_days.len());

        // Track results
        let mut day_results = Vec::new();
        let mut total_playbook_bullets = 0;

        // Get initial capital
        let account_mgr = AccountManager::new(self.pool.clone());
        let starting_capital = 10000.0; // Reset for backtest

        // Simulate each day
        for (day_num, current_date) in trading_days.iter().enumerate() {
            println!("📅 Day {}: {}", day_num + 1, current_date);

            match self.run_day_simulation(*current_date, symbol).await {
                Ok(result) => {
                    total_playbook_bullets += result.playbook_bullets_added;
                    println!("  {}", result.notes);
                    day_results.push(result);
                }
                Err(e) => {
                    warn!("Failed to simulate day {}: {}", current_date, e);
                    println!("  ⚠️  Skipped due to error: {}", e);
                    day_results.push(DayResult {
                        date: *current_date,
                        decision: None,
                        executed: false,
                        pnl: None,
                        pnl_pct: None,
                        won: None,
                        playbook_bullets_added: 0,
                        notes: format!("Error: {}", e),
                    });
                }
            }
            println!();
        }

        // Calculate final metrics
        let total_trades = day_results.iter().filter(|r| r.executed).count();
        let wins = day_results
            .iter()
            .filter(|r| r.won.unwrap_or(false))
            .count();
        let losses = total_trades - wins;

        // Get final capital
        let final_account = account_mgr.get_current_account().await?;
        let ending_capital = final_account.balance;
        let total_return = ending_capital - starting_capital;
        let total_return_pct = (total_return / starting_capital) * 100.0;

        // Calculate max drawdown
        let max_drawdown_pct = self.calculate_max_drawdown(&day_results, starting_capital);

        // Calculate Sharpe ratio
        let sharpe_ratio = self.calculate_sharpe_ratio(&day_results);

        let results = BacktestResults {
            start_date: self.start_date,
            end_date: self.end_date,
            symbol: symbol.to_string(),
            total_days: (self.end_date - self.start_date).num_days() as usize + 1,
            trading_days: trading_days.len(),
            total_trades,
            wins,
            losses,
            starting_capital,
            ending_capital,
            total_return,
            total_return_pct,
            max_drawdown_pct,
            sharpe_ratio,
            playbook_bullets_created: total_playbook_bullets,
            day_results,
        };

        results.display_summary();

        Ok(results)
    }

    /// Simulate a single trading day
    async fn run_day_simulation(
        &self,
        current_date: NaiveDate,
        symbol: &str,
    ) -> Result<DayResult> {
        // Import orchestrators here to avoid circular dependencies
        use crate::orchestrator::{EveningOrchestrator, MorningOrchestrator};

        // MORNING ROUTINE
        info!("Running morning analysis for {}", current_date);

        // Create morning orchestrator with backtest mode
        let mut morning_config = self.config.clone();
        morning_config.backtest_mode = Some(true);
        morning_config.backtest_date = Some(current_date);
        morning_config.skip_sentiment = Some(self.skip_sentiment);
        // Use OpenAI for backtest (configured in .env)
        morning_config.llm.provider = "openai".to_string();
        morning_config.llm.primary_model = "gpt-5-nano-2025-08-07".to_string();
        morning_config.llm.timeout_seconds = 120;

        let morning_orchestrator =
            MorningOrchestrator::new(self.pool.clone(), morning_config).await?;

        // Run analysis for this historical date
        let decision = morning_orchestrator
            .analyze_at_date(symbol, current_date)
            .await?;

        let confidence = decision.confidence;
        let action = &decision.action;

        // Get the context_id that was just created by analyze_at_date
        // We need to query the most recent context for this symbol
        let context_dao = ContextDAO::new(self.pool.clone());
        let recent_contexts = context_dao.get_recent_contexts(1).await?;
        let context_id = recent_contexts
            .first()
            .map(|c| c.id)
            .context("No context found after analysis")?;

        // Determine if we should execute a trade
        let should_execute = match action.as_str() {
            "STAY_FLAT" | "FLAT" => false,
            _ if confidence < 0.50 => false,
            _ => true,
        };

        // EXECUTE TRADE (if applicable)
        if should_execute {
            info!("Executing paper trade for context {}", context_id);
            let _trade_id = self.execute_backtest_trade(context_id, current_date).await?;
        } else {
            info!("Skipping trade execution: action={}, confidence={:.1}%", action, confidence * 100.0);
        }

        // EVENING ROUTINE - Compute outcome using next day's data
        let next_date = self.get_next_trading_day(current_date)?;

        info!("Running evening review for {} (outcome from {})", current_date, next_date);

        let mut evening_config = self.config.clone();
        evening_config.backtest_mode = Some(true);
        evening_config.backtest_date = Some(next_date); // Use next day for outcome
        // Use OpenAI for backtest (configured in .env)
        evening_config.llm.provider = "openai".to_string();
        evening_config.llm.primary_model = "gpt-5-nano-2025-08-07".to_string();
        evening_config.llm.timeout_seconds = 120;

        let evening_orchestrator =
            EveningOrchestrator::new(self.pool.clone(), evening_config).await?;

        let review_result = evening_orchestrator
            .review_context_at_date(context_id, next_date)
            .await?;

        // Format notes based on whether trade was executed
        let notes = if should_execute {
            format!(
                "✅ {}: {:.1}% conf → P&L ${:+.2} ({:+.2}%) {} | Bullets: +{}",
                decision.action,
                confidence * 100.0,
                review_result.outcome.pnl_value,
                review_result.outcome.pnl_pct,
                if review_result.outcome.win { "✅" } else { "❌" },
                review_result.curation_summary.bullets_added
            )
        } else {
            format!(
                "⏸️  {}: {:.1}% conf → P&L ${:+.2} (opp cost) {} | Bullets: +{}",
                decision.action,
                confidence * 100.0,
                review_result.outcome.pnl_value,
                if review_result.outcome.win { "✅" } else { "❌" },
                review_result.curation_summary.bullets_added
            )
        };

        Ok(DayResult {
            date: current_date,
            decision: Some(decision),
            executed: should_execute,
            pnl: Some(review_result.outcome.pnl_value),
            pnl_pct: Some(review_result.outcome.pnl_pct),
            won: Some(review_result.outcome.win),
            playbook_bullets_added: review_result.curation_summary.bullets_added,
            notes,
        })
    }

    /// Execute a paper trade in backtest mode
    async fn execute_backtest_trade(
        &self,
        context_id: Uuid,
        trade_date: NaiveDate,
    ) -> Result<Uuid> {
        use crate::trading::{PaperTradingEngine, PositionSizer, RiskManager, SlippageCalculator, TradeType};

        // Load context
        let context_dao = ContextDAO::new(self.pool.clone());
        let context = context_dao
            .get_context_by_id(context_id)
            .await?
            .context("Context not found")?;

        let action = context
            .decision
            .as_ref()
            .and_then(|d| d.get("action"))
            .and_then(|a| a.as_str())
            .unwrap_or("FLAT");

        let confidence = context.confidence.unwrap_or(0.0);

        // Extract VIX from market state for risk management
        let vix = context
            .market_state
            .get("market_data")
            .and_then(|d| d.get("vix"))
            .and_then(|v| v.as_f64());

        // Initialize risk manager
        let risk_manager = RiskManager::with_defaults();

        // Check VIX-based trading restrictions
        risk_manager.check_vix_restriction(vix, confidence as f64)?;

        // Get account for position sizing
        let account_mgr = AccountManager::new(self.pool.clone());
        let account = account_mgr.get_current_account().await?;

        // Calculate base position size
        let position_sizer = PositionSizer::default();
        let base_position_size =
            position_sizer.calculate_position_size_simple(account.balance, confidence as f64)?;

        // Apply volatility adjustment to position size
        let position_size = risk_manager.calculate_volatility_adjusted_position_size(
            base_position_size,
            vix,
        );

        // Get current price from market state
        let current_price = context
            .market_state
            .get("market_data")
            .and_then(|d| d.get("latest_price"))
            .and_then(|p| p.as_f64())
            .context("Could not get current price from market state")?;

        // Calculate realistic slippage based on market conditions
        let slippage_calc = SlippageCalculator::with_defaults();
        let slippage_pct = slippage_calc.calculate_slippage(vix, true);

        info!(
            "Calculated slippage: {:.3}% (VIX: {})",
            slippage_pct * 100.0,
            vix.map(|v| format!("{:.1}", v)).unwrap_or_else(|| "N/A".to_string())
        );

        // Determine trade type
        let trade_type = match action {
            "BUY_CALLS" => TradeType::Call,
            "BUY_PUTS" => TradeType::Put,
            _ => anyhow::bail!("Invalid trade action: {} (should have been caught earlier)", action),
        };

        // Calculate and log risk management levels
        let is_option = true; // Trading SPY options
        let stop_loss_price = risk_manager.get_stop_loss_price(current_price, &trade_type, is_option);
        let take_profit_price = risk_manager.get_take_profit_price(current_price, &trade_type, is_option);

        info!(
            "Risk levels for {} trade: Stop-Loss @ ${:.2} (-15%), Take-Profit @ ${:.2} (+50%)",
            trade_type,
            stop_loss_price,
            take_profit_price
        );

        // Calculate shares (contracts)
        let shares = (position_size / (current_price * 100.0)).floor().max(1.0);

        // Convert trade_date to DateTime for entry timestamp
        let entry_time = trade_date
            .and_hms_opt(9, 30, 0)
            .expect("Invalid time")
            .and_utc();

        info!(
            "Entering backtest trade at historical time: {} (not {})",
            entry_time.format("%Y-%m-%d %H:%M"),
            chrono::Utc::now().format("%Y-%m-%d %H:%M")
        );

        // Execute paper trade with historical entry time
        let paper_trading = PaperTradingEngine::new(self.pool.clone());
        let trade = paper_trading
            .enter_trade_with_time(
                Some(context_id),
                "SPY".to_string(),
                trade_type.clone(),
                current_price,
                shares,
                position_size,
                None,   // strike_price
                None,   // expiration_date
                slippage_pct,   // Use calculated realistic slippage
                0.65,   // commission
                None,   // notes
                Some(slippage_pct),   // estimated_slippage_pct (same as actual for now)
                vix,    // market_vix for tracking
                None,   // option_moneyness
                Some(entry_time), // Use historical date for entry
            )
            .await?;

        info!(
            "Trade {} entered: {} {} @ ${:.2} with {:.3}% slippage",
            trade.id,
            trade_type.to_string(),
            "SPY",
            current_price,
            slippage_pct * 100.0
        );

        Ok(trade.id)
    }

    /// Get list of trading days (weekdays only)
    fn get_trading_days(&self) -> Vec<NaiveDate> {
        let mut days = Vec::new();
        let mut current = self.start_date;

        while current <= self.end_date {
            // Only include weekdays
            if current.weekday() != Weekday::Sat && current.weekday() != Weekday::Sun {
                days.push(current);
            }
            current += Duration::days(1);
        }

        days
    }

    /// Get next trading day after given date
    fn get_next_trading_day(&self, date: NaiveDate) -> Result<NaiveDate> {
        let mut next = date + Duration::days(1);

        // Skip weekends
        while next.weekday() == Weekday::Sat || next.weekday() == Weekday::Sun {
            next += Duration::days(1);
        }

        if next > self.end_date {
            anyhow::bail!("Next trading day {} is beyond backtest range", next);
        }

        Ok(next)
    }

    /// Calculate maximum drawdown percentage
    fn calculate_max_drawdown(&self, results: &[DayResult], starting_capital: f64) -> f64 {
        let mut peak = starting_capital;
        let mut max_drawdown = 0.0;
        let mut current_capital = starting_capital;

        for result in results {
            if let Some(pnl) = result.pnl {
                current_capital += pnl;

                if current_capital > peak {
                    peak = current_capital;
                }

                let drawdown = ((peak - current_capital) / peak) * 100.0;
                if drawdown > max_drawdown {
                    max_drawdown = drawdown;
                }
            }
        }

        max_drawdown
    }

    /// Calculate Sharpe ratio (annualized)
    fn calculate_sharpe_ratio(&self, results: &[DayResult]) -> f64 {
        let returns: Vec<f64> = results
            .iter()
            .filter_map(|r| r.pnl_pct)
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        // Calculate mean and std dev
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0;
        }

        // Annualize (assuming ~252 trading days/year)
        let annualized_return = mean * 252.0;
        let annualized_std = std_dev * (252.0_f64).sqrt();

        // Sharpe ratio (assuming 0% risk-free rate for simplicity)
        annualized_return / annualized_std
    }
}

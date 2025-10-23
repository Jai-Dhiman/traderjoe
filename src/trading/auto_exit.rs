// Auto-Exit Logic
// Implements stop-loss, take-profit, and time-based exit rules

use anyhow::Result;
use chrono::{DateTime, NaiveTime, Utc};
use sqlx::PgPool;
use tracing::{info, warn};

use super::paper::{ExitReason, PaperTrade, PaperTradingEngine};

/// Configuration for auto-exit rules
#[derive(Debug, Clone)]
pub struct AutoExitConfig {
    /// Time to auto-exit positions (e.g., 3:00 PM ET)
    pub auto_exit_time: NaiveTime,

    /// Stop-loss percentage (e.g., -0.50 = -50%)
    pub stop_loss_pct: f64,

    /// Take-profit percentage (e.g., 0.30 = +30%)
    pub take_profit_pct: f64,

    /// Enable time-based exit
    pub enable_time_exit: bool,

    /// Enable stop-loss
    pub enable_stop_loss: bool,

    /// Enable take-profit
    pub enable_take_profit: bool,
}

impl Default for AutoExitConfig {
    fn default() -> Self {
        Self {
            auto_exit_time: NaiveTime::from_hms_opt(15, 0, 0)
                .expect("Invalid hardcoded time 15:00:00 - this is a bug"), // 3:00 PM ET
            stop_loss_pct: -0.50,  // -50% (options can go to zero)
            take_profit_pct: 0.30, // +30%
            enable_time_exit: true,
            enable_stop_loss: true,
            enable_take_profit: true,
        }
    }
}

pub struct AutoExitManager {
    engine: PaperTradingEngine,
    config: AutoExitConfig,
}

impl AutoExitManager {
    pub fn new(pool: PgPool, config: AutoExitConfig) -> Self {
        Self {
            engine: PaperTradingEngine::new(pool),
            config,
        }
    }

    /// Check all open positions and exit if necessary
    ///
    /// Returns: List of trades that were exited
    pub async fn check_and_exit_positions(
        &self,
        current_prices: &std::collections::HashMap<String, f64>,
    ) -> Result<Vec<PaperTrade>> {
        let mut exited_trades = Vec::new();
        let open_positions = self.engine.get_open_positions().await?;

        for trade in open_positions {
            // Get current price for this symbol
            let current_price = match current_prices.get(&trade.symbol) {
                Some(price) => *price,
                None => {
                    warn!("No current price available for {}", trade.symbol);
                    continue;
                }
            };

            // Check exit conditions
            if let Some((exit_reason, should_exit)) =
                self.should_exit(&trade, current_price, Utc::now())
            {
                if should_exit {
                    info!(
                        "Exiting trade {} ({}): {:?}",
                        trade.id,
                        trade.symbol,
                        exit_reason
                    );

                    // Update MFE/MAE before exiting
                    self.engine
                        .update_excursion(trade.id, current_price)
                        .await?;

                    // Exit the trade
                    let exited_trade = self.engine.exit_trade(trade.id, current_price, exit_reason).await?;
                    exited_trades.push(exited_trade);
                }
            }
        }

        Ok(exited_trades)
    }

    /// Check if a trade should be exited
    ///
    /// Returns: (ExitReason, should_exit)
    fn should_exit(
        &self,
        trade: &PaperTrade,
        current_price: f64,
        now: DateTime<Utc>,
    ) -> Option<(ExitReason, bool)> {
        // Calculate current P&L percentage
        let pnl_pct = (current_price - trade.entry_price) / trade.entry_price;

        // Check stop-loss
        if self.config.enable_stop_loss && pnl_pct <= self.config.stop_loss_pct {
            return Some((ExitReason::StopLoss, true));
        }

        // Check take-profit
        if self.config.enable_take_profit && pnl_pct >= self.config.take_profit_pct {
            return Some((ExitReason::TakeProfit, true));
        }

        // Check time-based exit
        if self.config.enable_time_exit && self.is_past_exit_time(now) {
            return Some((ExitReason::AutoExit, true));
        }

        None
    }

    /// Check if current time is past the auto-exit time
    ///
    /// NOTE: Simplified timezone handling - assumes ET offset
    fn is_past_exit_time(&self, now: DateTime<Utc>) -> bool {
        // Convert UTC to ET (simplified - doesn't handle DST)
        let et_offset = -5; // EST is UTC-5 (use -4 for EDT)
        let et_time = now + chrono::Duration::hours(et_offset);
        let current_time = et_time.time();

        current_time >= self.config.auto_exit_time
    }

    /// Update MFE/MAE for all open positions
    pub async fn update_excursions(
        &self,
        current_prices: &std::collections::HashMap<String, f64>,
    ) -> Result<()> {
        let open_positions = self.engine.get_open_positions().await?;

        for trade in open_positions {
            if let Some(current_price) = current_prices.get(&trade.symbol) {
                self.engine
                    .update_excursion(trade.id, *current_price)
                    .await?;
            }
        }

        Ok(())
    }

    /// Calculate trailing stop-loss level for a trade
    ///
    /// This is a more advanced exit strategy that can be added later
    pub fn calculate_trailing_stop(
        &self,
        entry_price: f64,
        current_price: f64,
        trailing_pct: f64,
    ) -> f64 {
        let max_price = current_price.max(entry_price);
        max_price * (1.0 - trailing_pct)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    #[test]
    fn test_should_exit_stop_loss() {
        let config = AutoExitConfig::default();
        let pool = sqlx::PgPool::connect_lazy("postgresql://localhost/test").unwrap();
        let manager = AutoExitManager::new(pool, config);

        let trade = PaperTrade {
            id: uuid::Uuid::new_v4(),
            context_id: None,
            symbol: "SPY".to_string(),
            trade_type: super::super::paper::TradeType::Call,
            entry_price: 2.0,
            entry_time: Utc::now(),
            exit_price: None,
            exit_time: None,
            shares: 1.0,
            status: super::super::paper::TradeStatus::Open,
            pnl: None,
            pnl_pct: None,
            notes: None,
            created_at: Utc::now(),
            strike_price: Some(587.0),
            expiration_date: Some(NaiveDate::from_ymd_opt(2025, 10, 22).unwrap()),
            position_size_usd: 200.0,
            commission: 0.65,
            slippage_pct: 0.03,
            max_favorable_excursion: None,
            max_adverse_excursion: None,
            exit_reason: None,
        };

        // Current price at 1.0 = -50% from entry
        let result = manager.should_exit(&trade, 1.0, Utc::now());

        assert!(result.is_some());
        let (reason, should_exit) = result.unwrap();
        assert_eq!(reason, ExitReason::StopLoss);
        assert!(should_exit);
    }

    #[test]
    fn test_should_exit_take_profit() {
        let config = AutoExitConfig::default();
        let pool = sqlx::PgPool::connect_lazy("postgresql://localhost/test").unwrap();
        let manager = AutoExitManager::new(pool, config);

        let trade = PaperTrade {
            id: uuid::Uuid::new_v4(),
            context_id: None,
            symbol: "SPY".to_string(),
            trade_type: super::super::paper::TradeType::Call,
            entry_price: 2.0,
            entry_time: Utc::now(),
            exit_price: None,
            exit_time: None,
            shares: 1.0,
            status: super::super::paper::TradeStatus::Open,
            pnl: None,
            pnl_pct: None,
            notes: None,
            created_at: Utc::now(),
            strike_price: Some(587.0),
            expiration_date: Some(NaiveDate::from_ymd_opt(2025, 10, 22).unwrap()),
            position_size_usd: 200.0,
            commission: 0.65,
            slippage_pct: 0.03,
            max_favorable_excursion: None,
            max_adverse_excursion: None,
            exit_reason: None,
        };

        // Current price at 2.6 = +30% from entry
        let result = manager.should_exit(&trade, 2.6, Utc::now());

        assert!(result.is_some());
        let (reason, should_exit) = result.unwrap();
        assert_eq!(reason, ExitReason::TakeProfit);
        assert!(should_exit);
    }

    #[test]
    fn test_calculate_trailing_stop() {
        let config = AutoExitConfig::default();
        let pool = sqlx::PgPool::connect_lazy("postgresql://localhost/test").unwrap();
        let manager = AutoExitManager::new(pool, config);

        let entry = 2.0;
        let current = 3.0;
        let trailing_pct = 0.15; // 15% trailing stop

        let stop = manager.calculate_trailing_stop(entry, current, trailing_pct);

        // Stop should be at 3.0 * 0.85 = 2.55
        assert_eq!(stop, 2.55);
    }

    #[test]
    fn test_no_exit_conditions_met() {
        let config = AutoExitConfig::default();
        let pool = sqlx::PgPool::connect_lazy("postgresql://localhost/test").unwrap();
        let manager = AutoExitManager::new(pool, config);

        let trade = PaperTrade {
            id: uuid::Uuid::new_v4(),
            context_id: None,
            symbol: "SPY".to_string(),
            trade_type: super::super::paper::TradeType::Call,
            entry_price: 2.0,
            entry_time: Utc::now(),
            exit_price: None,
            exit_time: None,
            shares: 1.0,
            status: super::super::paper::TradeStatus::Open,
            pnl: None,
            pnl_pct: None,
            notes: None,
            created_at: Utc::now(),
            strike_price: Some(587.0),
            expiration_date: Some(NaiveDate::from_ymd_opt(2025, 10, 22).unwrap()),
            position_size_usd: 200.0,
            commission: 0.65,
            slippage_pct: 0.03,
            max_favorable_excursion: None,
            max_adverse_excursion: None,
            exit_reason: None,
        };

        // Current price at 2.2 = +10% from entry (not at take-profit yet)
        // Also current time is early morning (not at exit time yet)
        let result = manager.should_exit(&trade, 2.2, Utc::now());

        assert!(result.is_none());
    }
}

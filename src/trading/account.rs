// Account Balance and Performance Tracking
// Maintains account state, equity, and performance metrics

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

use super::paper::{ExitReason, PaperTrade, TradeStatus, TradeType};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub id: Uuid,
    pub balance: f64,
    pub equity: f64, // balance + unrealized P&L
    pub timestamp: DateTime<Utc>,
    pub daily_pnl: Option<f64>,
    pub weekly_pnl: Option<f64>,
    pub monthly_pnl: Option<f64>,
    pub max_drawdown_pct: Option<f64>,
    pub sharpe_ratio: Option<f64>,
    pub win_rate: Option<f64>,
    pub total_trades: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub total_return: f64,
    pub total_return_pct: f64,
    pub daily_avg: f64,
    pub daily_avg_pct: f64,
    pub best_day: f64,
    pub worst_day: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_pct: f64,
    pub total_trades: i32,
    pub winning_trades: i32,
    pub losing_trades: i32,
}

pub struct AccountManager {
    pool: PgPool,
}

impl AccountManager {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Get current account state
    pub async fn get_current_account(&self) -> Result<Account> {
        let account = sqlx::query_as!(
            Account,
            r#"
            SELECT
                id,
                balance,
                equity,
                timestamp,
                daily_pnl,
                weekly_pnl,
                monthly_pnl,
                max_drawdown_pct,
                sharpe_ratio,
                win_rate,
                total_trades
            FROM account_balance
            ORDER BY timestamp DESC
            LIMIT 1
            "#
        )
        .fetch_one(&self.pool)
        .await
        .context("Failed to fetch current account")?;

        Ok(account)
    }

    /// Update account balance after a trade
    pub async fn update_balance(&self, trade: &PaperTrade) -> Result<Account> {
        let current = self.get_current_account().await?;

        // Update balance if trade is closed
        let new_balance = if let Some(pnl) = trade.pnl {
            current.balance + pnl
        } else {
            current.balance
        };

        // Calculate equity (balance + open position value)
        let equity = self.calculate_equity(new_balance).await?;

        // Insert new account snapshot
        let account = sqlx::query_as!(
            Account,
            r#"
            INSERT INTO account_balance (balance, equity, timestamp)
            VALUES ($1, $2, $3)
            RETURNING
                id,
                balance,
                equity,
                timestamp,
                daily_pnl,
                weekly_pnl,
                monthly_pnl,
                max_drawdown_pct,
                sharpe_ratio,
                win_rate,
                total_trades
            "#,
            new_balance,
            equity,
            Utc::now()
        )
        .fetch_one(&self.pool)
        .await
        .context("Failed to update account balance")?;

        Ok(account)
    }

    /// Calculate current equity (balance + unrealized P&L)
    pub async fn calculate_equity(&self, current_balance: f64) -> Result<f64> {
        // For now, just return balance
        // In a full implementation, you'd calculate unrealized P&L from open positions
        // by fetching current market prices
        Ok(current_balance)
    }

    /// Calculate performance statistics
    pub async fn get_performance_stats(&self, days: Option<i32>) -> Result<PerformanceStats> {
        let days = days.unwrap_or(30);

        // Get all closed trades in the period
        let trades = sqlx::query_as!(
            PaperTrade,
            r#"
            SELECT
                id,
                context_id,
                symbol,
                trade_type as "trade_type: TradeType",
                entry_price,
                entry_time,
                exit_price,
                exit_time,
                shares,
                status as "status: TradeStatus",
                pnl,
                pnl_pct,
                notes,
                created_at,
                strike_price,
                expiration_date,
                position_size_usd as "position_size_usd!",
                commission as "commission!",
                slippage_pct as "slippage_pct!",
                max_favorable_excursion,
                max_adverse_excursion,
                exit_reason as "exit_reason: ExitReason",
                estimated_slippage_pct,
                actual_entry_slippage_pct,
                actual_exit_slippage_pct,
                market_vix,
                option_moneyness
            FROM paper_trades
            WHERE
                status = 'CLOSED'
                AND exit_time >= NOW() - INTERVAL '1 day' * $1
            ORDER BY exit_time
            "#,
            days as f64
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to fetch trades for performance stats")?;

        // Calculate statistics
        let total_trades = trades.len() as i32;

        if total_trades == 0 {
            return Ok(PerformanceStats {
                total_return: 0.0,
                total_return_pct: 0.0,
                daily_avg: 0.0,
                daily_avg_pct: 0.0,
                best_day: 0.0,
                worst_day: 0.0,
                win_rate: 0.0,
                profit_factor: 0.0,
                avg_win: 0.0,
                avg_loss: 0.0,
                sharpe_ratio: 0.0,
                max_drawdown: 0.0,
                max_drawdown_pct: 0.0,
                total_trades: 0,
                winning_trades: 0,
                losing_trades: 0,
            });
        }

        let mut total_return = 0.0;
        let mut winning_trades = 0;
        let mut losing_trades = 0;
        let mut total_wins = 0.0;
        let mut total_losses = 0.0;
        let mut best_day = f64::MIN;
        let mut worst_day = f64::MAX;

        for trade in &trades {
            if let Some(pnl) = trade.pnl {
                total_return += pnl;

                if pnl > 0.0 {
                    winning_trades += 1;
                    total_wins += pnl;
                    best_day = best_day.max(pnl);
                } else if pnl < 0.0 {
                    losing_trades += 1;
                    total_losses += pnl.abs();
                    worst_day = worst_day.min(pnl);
                }
            }
        }

        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        let avg_win = if winning_trades > 0 {
            total_wins / winning_trades as f64
        } else {
            0.0
        };

        let avg_loss = if losing_trades > 0 {
            total_losses / losing_trades as f64
        } else {
            0.0
        };

        let profit_factor = if total_losses > 0.0 {
            total_wins / total_losses
        } else if total_wins > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Get initial balance to calculate percentage returns
        let initial_balance = sqlx::query_scalar!(
            "SELECT balance FROM account_balance ORDER BY timestamp ASC LIMIT 1"
        )
        .fetch_optional(&self.pool)
        .await?
        .unwrap_or(10000.0); // Fallback to default if no records exist

        let total_return_pct = total_return / initial_balance;
        let daily_avg = total_return / days as f64;
        let daily_avg_pct = total_return_pct / days as f64;

        // Calculate max drawdown
        let (max_drawdown, max_drawdown_pct) = self.calculate_max_drawdown(&trades, initial_balance)?;

        // Calculate Sharpe ratio from daily aggregated returns
        let sharpe_ratio = if days > 0 && !trades.is_empty() {
            use std::collections::HashMap;
            let mut daily_pnl: HashMap<String, f64> = HashMap::new();

            // Aggregate P&L by day (multiple trades can occur on same day)
            for trade in &trades {
                if let (Some(exit_time), Some(pnl)) = (trade.exit_time, trade.pnl) {
                    let date_key = exit_time.format("%Y-%m-%d").to_string();
                    *daily_pnl.entry(date_key).or_insert(0.0) += pnl;
                }
            }

            // Convert daily P&L to daily returns as percentage of initial balance
            let daily_returns: Vec<f64> = daily_pnl.values()
                .map(|&pnl| pnl / initial_balance)
                .collect();

            self.calculate_sharpe_ratio(&daily_returns)
        } else {
            0.0
        };

        Ok(PerformanceStats {
            total_return,
            total_return_pct,
            daily_avg,
            daily_avg_pct,
            best_day: if best_day == f64::MIN { 0.0 } else { best_day },
            worst_day: if worst_day == f64::MAX { 0.0 } else { worst_day },
            win_rate,
            profit_factor,
            avg_win,
            avg_loss,
            sharpe_ratio,
            max_drawdown,
            max_drawdown_pct,
            total_trades,
            winning_trades,
            losing_trades,
        })
    }

    /// Calculate maximum drawdown from a series of trades
    fn calculate_max_drawdown(&self, trades: &[PaperTrade], initial_balance: f64) -> Result<(f64, f64)> {
        let mut peak = initial_balance;
        let mut max_dd = 0.0;
        let mut running_balance = initial_balance;

        for trade in trades {
            if let Some(pnl) = trade.pnl {
                running_balance += pnl;

                if running_balance > peak {
                    peak = running_balance;
                }

                let dd = peak - running_balance;
                if dd > max_dd {
                    max_dd = dd;
                }
            }
        }

        let max_dd_pct = if peak > 0.0 {
            max_dd / peak
        } else {
            0.0
        };

        Ok((max_dd, max_dd_pct))
    }

    /// Calculate Sharpe ratio (simplified - assumes daily returns)
    fn calculate_sharpe_ratio(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

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

        // Annualized Sharpe ratio (assuming 252 trading days)
        // Risk-free rate assumed to be ~4% annual = 0.04/252 daily
        let risk_free_rate = 0.04 / 252.0;
        let sharpe = (mean - risk_free_rate) / std_dev * (252.0_f64).sqrt();

        sharpe
    }

    /// Get account balance history
    pub async fn get_balance_history(&self, days: Option<i32>) -> Result<Vec<Account>> {
        let days = days.unwrap_or(30);

        let history = sqlx::query_as!(
            Account,
            r#"
            SELECT
                id,
                balance,
                equity,
                timestamp,
                daily_pnl,
                weekly_pnl,
                monthly_pnl,
                max_drawdown_pct,
                sharpe_ratio,
                win_rate,
                total_trades
            FROM account_balance
            WHERE timestamp >= NOW() - INTERVAL '1 day' * $1
            ORDER BY timestamp
            "#,
            days as f64
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to fetch account balance history")?;

        Ok(history)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sharpe_ratio_calculation() {
        let manager = AccountManager::new(sqlx::PgPool::connect_lazy("postgresql://localhost/test").expect("Failed to create pool"));

        // Sample daily returns
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.01, -0.005, 0.02];

        let sharpe = manager.calculate_sharpe_ratio(&returns);

        // Should be a positive value
        assert!(sharpe > 0.0);
    }

    #[tokio::test]
    async fn test_sharpe_ratio_zero_returns() {
        let manager = AccountManager::new(sqlx::PgPool::connect_lazy("postgresql://localhost/test").expect("Failed to create pool"));

        let returns = vec![0.0, 0.0, 0.0];

        let sharpe = manager.calculate_sharpe_ratio(&returns);

        assert_eq!(sharpe, 0.0);
    }
}

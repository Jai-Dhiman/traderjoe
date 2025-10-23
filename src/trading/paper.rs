// Paper Trading Engine
// Implements realistic paper trading with slippage, commissions, and position tracking

use anyhow::{Context, Result};
use chrono::{DateTime, Utc, NaiveDate};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type, PartialEq)]
#[sqlx(type_name = "text", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TradeType {
    Call,
    Put,
    Flat,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type, PartialEq)]
#[sqlx(type_name = "text", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TradeStatus {
    Open,
    Closed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type, PartialEq)]
#[sqlx(type_name = "text", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ExitReason {
    AutoExit,
    StopLoss,
    TakeProfit,
    Manual,
    CircuitBreaker,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperTrade {
    pub id: Uuid,
    pub context_id: Option<Uuid>,
    pub symbol: String,
    pub trade_type: TradeType,
    pub entry_price: f64,
    pub entry_time: DateTime<Utc>,
    pub exit_price: Option<f64>,
    pub exit_time: Option<DateTime<Utc>>,
    pub shares: f64,
    pub status: TradeStatus,
    pub pnl: Option<f64>,
    pub pnl_pct: Option<f64>,
    pub notes: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,

    // Position tracking
    pub strike_price: Option<f64>,
    pub expiration_date: Option<NaiveDate>,
    pub position_size_usd: f64,
    pub commission: f64,
    pub slippage_pct: f64,

    // Risk metrics
    pub max_favorable_excursion: Option<f64>,
    pub max_adverse_excursion: Option<f64>,
    pub exit_reason: Option<ExitReason>,
}

pub struct PaperTradingEngine {
    pool: PgPool,
}

impl PaperTradingEngine {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Enter a new paper trade
    pub async fn enter_trade(
        &self,
        context_id: Option<Uuid>,
        symbol: String,
        trade_type: TradeType,
        entry_price: f64,
        shares: f64,
        position_size_usd: f64,
        strike_price: Option<f64>,
        expiration_date: Option<NaiveDate>,
        slippage_pct: f64,
        commission: f64,
        notes: Option<serde_json::Value>,
    ) -> Result<PaperTrade> {
        let id = Uuid::new_v4();
        let entry_time = Utc::now();
        let status = TradeStatus::Open;

        let trade = sqlx::query_as!(
            PaperTrade,
            r#"
            INSERT INTO paper_trades (
                id, context_id, symbol, trade_type, entry_price, entry_time,
                shares, status, position_size_usd, strike_price, expiration_date,
                slippage_pct, commission, notes, created_at
            )
            VALUES ($1, $2, $3, $4::text, $5, $6, $7, $8::text, $9, $10, $11, $12, $13, $14, $15)
            RETURNING
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
                exit_reason as "exit_reason?: ExitReason"
            "#,
            id,
            context_id,
            symbol,
            trade_type.to_string(),
            entry_price,
            entry_time,
            shares,
            status.to_string(),
            position_size_usd,
            strike_price,
            expiration_date,
            slippage_pct,
            commission,
            notes,
            entry_time
        )
        .fetch_one(&self.pool)
        .await
        .context("Failed to insert paper trade")?;

        Ok(trade)
    }

    /// Exit a trade with the given exit price
    pub async fn exit_trade(
        &self,
        trade_id: Uuid,
        exit_price: f64,
        exit_reason: ExitReason,
    ) -> Result<PaperTrade> {
        let exit_time = Utc::now();

        // First get the trade to calculate P&L
        let trade = self.get_trade(trade_id).await?;

        // Calculate P&L
        let gross_pnl = (exit_price - trade.entry_price) * trade.shares;
        let net_pnl = gross_pnl - (trade.commission * 2.0); // Entry + exit commission
        let pnl_pct = net_pnl / trade.position_size_usd;

        // Update the trade
        let updated_trade = sqlx::query_as!(
            PaperTrade,
            r#"
            UPDATE paper_trades
            SET
                exit_price = $1,
                exit_time = $2,
                status = $3::text,
                pnl = $4,
                pnl_pct = $5,
                exit_reason = $6::text
            WHERE id = $7
            RETURNING
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
                exit_reason as "exit_reason?: ExitReason"
            "#,
            exit_price,
            exit_time,
            TradeStatus::Closed.to_string(),
            net_pnl,
            pnl_pct,
            exit_reason.to_string(),
            trade_id
        )
        .fetch_one(&self.pool)
        .await
        .context("Failed to exit paper trade")?;

        Ok(updated_trade)
    }

    /// Get a specific trade by ID
    pub async fn get_trade(&self, trade_id: Uuid) -> Result<PaperTrade> {
        let trade = sqlx::query_as!(
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
                exit_reason as "exit_reason?: ExitReason"
            FROM paper_trades
            WHERE id = $1
            "#,
            trade_id
        )
        .fetch_one(&self.pool)
        .await
        .context("Failed to fetch paper trade")?;

        Ok(trade)
    }

    /// Get all open positions
    pub async fn get_open_positions(&self) -> Result<Vec<PaperTrade>> {
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
                exit_reason as "exit_reason?: ExitReason"
            FROM paper_trades
            WHERE status = 'OPEN'
            ORDER BY entry_time DESC
            "#
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to fetch open positions")?;

        Ok(trades)
    }

    /// Get all trades (for reporting)
    pub async fn get_all_trades(&self, limit: Option<i64>) -> Result<Vec<PaperTrade>> {
        let limit = limit.unwrap_or(100);

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
                exit_reason as "exit_reason?: ExitReason"
            FROM paper_trades
            ORDER BY entry_time DESC
            LIMIT $1
            "#,
            limit
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to fetch all trades")?;

        Ok(trades)
    }

    /// Update MFE (Max Favorable Excursion) and MAE (Max Adverse Excursion)
    pub async fn update_excursion(
        &self,
        trade_id: Uuid,
        current_price: f64,
    ) -> Result<()> {
        let trade = self.get_trade(trade_id).await?;

        let current_pnl = (current_price - trade.entry_price) * trade.shares - (trade.commission * 2.0);

        let mfe = trade.max_favorable_excursion.unwrap_or(0.0).max(current_pnl);
        let mae = trade.max_adverse_excursion.unwrap_or(0.0).min(current_pnl);

        sqlx::query!(
            r#"
            UPDATE paper_trades
            SET
                max_favorable_excursion = $1,
                max_adverse_excursion = $2
            WHERE id = $3
            "#,
            mfe,
            mae,
            trade_id
        )
        .execute(&self.pool)
        .await
        .context("Failed to update excursion metrics")?;

        Ok(())
    }

    /// Cancel a trade (e.g., if something goes wrong)
    pub async fn cancel_trade(&self, trade_id: Uuid) -> Result<PaperTrade> {
        let updated_trade = sqlx::query_as!(
            PaperTrade,
            r#"
            UPDATE paper_trades
            SET status = $1::text
            WHERE id = $2
            RETURNING
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
                exit_reason as "exit_reason?: ExitReason"
            "#,
            TradeStatus::Cancelled.to_string(),
            trade_id
        )
        .fetch_one(&self.pool)
        .await
        .context("Failed to cancel paper trade")?;

        Ok(updated_trade)
    }
}

impl std::fmt::Display for TradeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradeType::Call => write!(f, "CALL"),
            TradeType::Put => write!(f, "PUT"),
            TradeType::Flat => write!(f, "FLAT"),
        }
    }
}

impl std::fmt::Display for TradeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradeStatus::Open => write!(f, "OPEN"),
            TradeStatus::Closed => write!(f, "CLOSED"),
            TradeStatus::Cancelled => write!(f, "CANCELLED"),
        }
    }
}

impl std::fmt::Display for ExitReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExitReason::AutoExit => write!(f, "AUTO_EXIT"),
            ExitReason::StopLoss => write!(f, "STOP_LOSS"),
            ExitReason::TakeProfit => write!(f, "TAKE_PROFIT"),
            ExitReason::Manual => write!(f, "MANUAL"),
            ExitReason::CircuitBreaker => write!(f, "CIRCUIT_BREAKER"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trade_type_display() {
        assert_eq!(TradeType::Call.to_string(), "CALL");
        assert_eq!(TradeType::Put.to_string(), "PUT");
        assert_eq!(TradeType::Flat.to_string(), "FLAT");
    }

    #[test]
    fn test_trade_status_display() {
        assert_eq!(TradeStatus::Open.to_string(), "OPEN");
        assert_eq!(TradeStatus::Closed.to_string(), "CLOSED");
        assert_eq!(TradeStatus::Cancelled.to_string(), "CANCELLED");
    }
}

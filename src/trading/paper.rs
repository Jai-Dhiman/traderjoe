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

    // Slippage tracking (Migration 007)
    pub estimated_slippage_pct: Option<f64>,
    pub actual_entry_slippage_pct: Option<f64>,
    pub actual_exit_slippage_pct: Option<f64>,
    pub market_vix: Option<f64>,
    pub option_moneyness: Option<String>,
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
        self.enter_trade_with_tracking(
            context_id,
            symbol,
            trade_type,
            entry_price,
            shares,
            position_size_usd,
            strike_price,
            expiration_date,
            slippage_pct,
            commission,
            notes,
            None, // estimated_slippage
            None, // market_vix
            None, // option_moneyness
        )
        .await
    }

    /// Enter a new paper trade with slippage tracking
    pub async fn enter_trade_with_tracking(
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
        estimated_slippage_pct: Option<f64>,
        market_vix: Option<f64>,
        option_moneyness: Option<String>,
    ) -> Result<PaperTrade> {
        self.enter_trade_with_time(
            context_id,
            symbol,
            trade_type,
            entry_price,
            shares,
            position_size_usd,
            strike_price,
            expiration_date,
            slippage_pct,
            commission,
            notes,
            estimated_slippage_pct,
            market_vix,
            option_moneyness,
            None, // entry_time - defaults to Utc::now()
        )
        .await
    }

    /// Enter a new paper trade with custom entry time (for backtests)
    pub async fn enter_trade_with_time(
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
        estimated_slippage_pct: Option<f64>,
        market_vix: Option<f64>,
        option_moneyness: Option<String>,
        entry_time: Option<chrono::DateTime<Utc>>,
    ) -> Result<PaperTrade> {
        let id = Uuid::new_v4();
        let entry_time = entry_time.unwrap_or_else(|| Utc::now());
        let status = TradeStatus::Open;

        // Apply slippage to entry price (slippage worsens the entry)
        let adjusted_entry_price = match trade_type {
            TradeType::Call => entry_price * (1.0 + slippage_pct),
            TradeType::Put => entry_price * (1.0 + slippage_pct),
            TradeType::Flat => entry_price,
        };

        let trade = sqlx::query_as!(
            PaperTrade,
            r#"
            INSERT INTO paper_trades (
                id, context_id, symbol, trade_type, entry_price, entry_time,
                shares, status, position_size_usd, strike_price, expiration_date,
                slippage_pct, commission, notes, created_at,
                estimated_slippage_pct, market_vix, option_moneyness
            )
            VALUES ($1, $2, $3, $4::text, $5, $6, $7, $8::text, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
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
                exit_reason as "exit_reason?: ExitReason",
                estimated_slippage_pct,
                actual_entry_slippage_pct,
                actual_exit_slippage_pct,
                market_vix,
                option_moneyness
            "#,
            id,
            context_id,
            symbol,
            trade_type.to_string(),
            adjusted_entry_price,
            entry_time,
            shares,
            status.to_string(),
            position_size_usd,
            strike_price,
            expiration_date,
            slippage_pct,
            commission,
            notes,
            entry_time,
            estimated_slippage_pct,
            market_vix,
            option_moneyness.as_deref()
        )
        .fetch_one(&self.pool)
        .await
        .context("Failed to insert paper trade")?;

        // Log detailed entry information
        tracing::info!(
            trade_id = %trade.id,
            context_id = ?context_id,
            symbol = %symbol,
            trade_type = ?trade_type,
            entry_price = %entry_price,
            adjusted_entry = %adjusted_entry_price,
            shares = %shares,
            position_size_usd = %position_size_usd,
            slippage_pct = %slippage_pct,
            commission = %commission,
            estimated_slippage = ?estimated_slippage_pct,
            "📈 TRADE ENTRY: {} {} @ ${:.2} (adjusted: ${:.2}, {} shares, position: ${:.2})",
            trade_type.to_string(),
            symbol,
            entry_price,
            adjusted_entry_price,
            shares,
            position_size_usd
        );

        Ok(trade)
    }

    /// Exit a trade with the given exit price
    pub async fn exit_trade(
        &self,
        trade_id: Uuid,
        exit_price: f64,
        exit_reason: ExitReason,
    ) -> Result<PaperTrade> {
        self.exit_trade_with_time(trade_id, exit_price, exit_reason, None).await
    }

    /// Exit a trade with a custom exit time (for backtests)
    pub async fn exit_trade_with_time(
        &self,
        trade_id: Uuid,
        exit_price: f64,
        exit_reason: ExitReason,
        exit_time: Option<chrono::DateTime<Utc>>,
    ) -> Result<PaperTrade> {
        let exit_time = exit_time.unwrap_or_else(|| Utc::now());

        // First get the trade to calculate P&L
        let trade = self.get_trade(trade_id).await?;

        // Apply slippage to exit price (slippage worsens the exit)
        let adjusted_exit_price = match trade.trade_type {
            TradeType::Call => exit_price * (1.0 - trade.slippage_pct),
            TradeType::Put => exit_price * (1.0 - trade.slippage_pct),
            TradeType::Flat => exit_price,
        };

        // Calculate P&L based on trade direction
        let price_diff = match trade.trade_type {
            TradeType::Call => adjusted_exit_price - trade.entry_price,
            TradeType::Put => trade.entry_price - adjusted_exit_price,
            TradeType::Flat => 0.0,
        };
        let gross_pnl = price_diff * trade.shares;
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
                exit_reason as "exit_reason?: ExitReason",
                estimated_slippage_pct,
                actual_entry_slippage_pct,
                actual_exit_slippage_pct,
                market_vix,
                option_moneyness
            "#,
            adjusted_exit_price,
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

        // Calculate duration in hours
        let duration_hours = (exit_time - trade.entry_time).num_seconds() as f64 / 3600.0;

        // Log detailed exit information with P&L breakdown
        tracing::info!(
            trade_id = %trade_id,
            symbol = %trade.symbol,
            trade_type = ?trade.trade_type,
            exit_reason = ?exit_reason,
            entry_price = %trade.entry_price,
            exit_price = %exit_price,
            adjusted_exit = %adjusted_exit_price,
            gross_pnl = %gross_pnl,
            commission_total = %(trade.commission * 2.0),
            net_pnl = %net_pnl,
            pnl_pct = %(pnl_pct * 100.0),
            duration_hours = %duration_hours,
            mfe = ?updated_trade.max_favorable_excursion,
            mae = ?updated_trade.max_adverse_excursion,
            "📉 TRADE EXIT: {} {} - {:?} | Entry: ${:.2} → Exit: ${:.2} | P&L: ${:.2} ({:+.1}%) | Duration: {:.1}h | MFE: {:?} MAE: {:?}",
            trade.trade_type.to_string(),
            trade.symbol,
            exit_reason,
            trade.entry_price,
            adjusted_exit_price,
            net_pnl,
            pnl_pct * 100.0,
            duration_hours,
            updated_trade.max_favorable_excursion,
            updated_trade.max_adverse_excursion
        );

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
                exit_reason as "exit_reason?: ExitReason",
                estimated_slippage_pct,
                actual_entry_slippage_pct,
                actual_exit_slippage_pct,
                market_vix,
                option_moneyness
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
                exit_reason as "exit_reason?: ExitReason",
                estimated_slippage_pct,
                actual_entry_slippage_pct,
                actual_exit_slippage_pct,
                market_vix,
                option_moneyness
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
                exit_reason as "exit_reason?: ExitReason",
                estimated_slippage_pct,
                actual_entry_slippage_pct,
                actual_exit_slippage_pct,
                market_vix,
                option_moneyness
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

    /// Get trading outcome for a specific ACE context (for evening review)
    /// Returns None if no trade was executed for this context, or if the trade is still open
    pub async fn get_context_outcome(&self, context_id: Uuid) -> Result<Option<crate::ace::reflector::TradingOutcome>> {
        // Query for closed trades associated with this context
        // Use regular query instead of query_as! for offline compilation
        let row_opt = sqlx::query(
            r#"
            SELECT
                id, context_id, symbol, trade_type, entry_price, entry_time,
                exit_price, exit_time, shares, status, pnl, pnl_pct, notes,
                created_at, strike_price, expiration_date, position_size_usd,
                commission, slippage_pct, max_favorable_excursion, max_adverse_excursion,
                exit_reason, estimated_slippage_pct, actual_entry_slippage_pct,
                actual_exit_slippage_pct, market_vix, option_moneyness
            FROM paper_trades
            WHERE context_id = $1 AND status = 'CLOSED'
            ORDER BY exit_time DESC
            LIMIT 1
            "#
        )
        .bind(context_id)
        .fetch_optional(&self.pool)
        .await
        .context("Failed to fetch trade by context_id")?;

        let trade_opt = if let Some(row) = row_opt {
            use sqlx::Row;
            Some(PaperTrade {
                id: row.get("id"),
                context_id: row.get("context_id"),
                symbol: row.get("symbol"),
                trade_type: row.get("trade_type"),
                entry_price: row.get("entry_price"),
                entry_time: row.get("entry_time"),
                exit_price: row.get("exit_price"),
                exit_time: row.get("exit_time"),
                shares: row.get("shares"),
                status: row.get("status"),
                pnl: row.get("pnl"),
                pnl_pct: row.get("pnl_pct"),
                notes: row.get("notes"),
                created_at: row.get("created_at"),
                strike_price: row.get("strike_price"),
                expiration_date: row.get("expiration_date"),
                position_size_usd: row.get("position_size_usd"),
                commission: row.get("commission"),
                slippage_pct: row.get("slippage_pct"),
                max_favorable_excursion: row.get("max_favorable_excursion"),
                max_adverse_excursion: row.get("max_adverse_excursion"),
                exit_reason: row.get("exit_reason"),
                estimated_slippage_pct: row.get("estimated_slippage_pct"),
                actual_entry_slippage_pct: row.get("actual_entry_slippage_pct"),
                actual_exit_slippage_pct: row.get("actual_exit_slippage_pct"),
                market_vix: row.get("market_vix"),
                option_moneyness: row.get("option_moneyness"),
            })
        } else {
            None
        };

        if let Some(trade) = trade_opt {
            // Convert PaperTrade to TradingOutcome
            let exit_price = trade.exit_price.unwrap_or(trade.entry_price);
            let exit_time = trade.exit_time.unwrap_or(trade.entry_time);
            let duration_hours = (exit_time - trade.entry_time).num_seconds() as f64 / 3600.0;

            let pnl_value = trade.pnl.unwrap_or(0.0);
            let pnl_pct = trade.pnl_pct.unwrap_or(0.0);
            let win = pnl_value > 0.0;

            let notes = trade.notes.map(|n| {
                match trade.exit_reason {
                    Some(ref reason) => format!("Exit reason: {:?}. {}", reason, n),
                    None => n.to_string(),
                }
            });

            Ok(Some(crate::ace::reflector::TradingOutcome {
                pnl_value,
                pnl_pct,
                mfe: trade.max_favorable_excursion,
                mae: trade.max_adverse_excursion,
                win,
                entry_price: trade.entry_price,
                exit_price,
                duration_hours,
                notes,
            }))
        } else {
            // No closed trade found for this context
            Ok(None)
        }
    }

    /// Update MFE (Max Favorable Excursion) and MAE (Max Adverse Excursion)
    pub async fn update_excursion(
        &self,
        trade_id: Uuid,
        current_price: f64,
    ) -> Result<()> {
        let trade = self.get_trade(trade_id).await?;

        // Calculate P&L based on trade direction
        let price_diff = match trade.trade_type {
            TradeType::Call => current_price - trade.entry_price,
            TradeType::Put => trade.entry_price - current_price,
            TradeType::Flat => 0.0,
        };
        // Only entry commission paid so far (exit commission not yet incurred)
        let current_pnl = price_diff * trade.shares - trade.commission;

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
        // Validate trade status before canceling
        let trade = self.get_trade(trade_id).await?;

        if trade.status != TradeStatus::Open {
            anyhow::bail!("Cannot cancel trade with status {:?}", trade.status);
        }

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
                exit_reason as "exit_reason?: ExitReason",
                estimated_slippage_pct,
                actual_entry_slippage_pct,
                actual_exit_slippage_pct,
                market_vix,
                option_moneyness
            "#,
            TradeStatus::Cancelled.to_string(),
            trade_id
        )
        .fetch_one(&self.pool)
        .await
        .context("Failed to cancel paper trade")?;

        Ok(updated_trade)
    }

    /// Analyze slippage performance for closed trades
    pub async fn analyze_slippage(&self, days: Option<i32>) -> Result<SlippageAnalysis> {
        let days_unwrapped = days.unwrap_or(30);

        #[derive(Debug)]
        struct SlippageRow {
            vix_range: Option<String>,
            moneyness: Option<String>,
            total_trades: i64,
            avg_estimated: Option<f64>,
            avg_actual: Option<f64>,
        }

        let rows = sqlx::query_as!(
            SlippageRow,
            r#"
            SELECT
                vix_range,
                moneyness,
                COUNT(*) as "total_trades!",
                AVG(estimated_slippage_pct) as "avg_estimated!",
                AVG(actual_slippage) as "avg_actual!"
            FROM (
                SELECT
                    CASE
                        WHEN market_vix < 15 THEN 'VIX_LOW'
                        WHEN market_vix < 25 THEN 'VIX_NORMAL'
                        WHEN market_vix < 35 THEN 'VIX_ELEVATED'
                        ELSE 'VIX_HIGH'
                    END as vix_range,
                    COALESCE(option_moneyness, 'ATM') as moneyness,
                    estimated_slippage_pct,
                    COALESCE(actual_entry_slippage_pct, estimated_slippage_pct) as actual_slippage
                FROM paper_trades
                WHERE
                    status = 'CLOSED'
                    AND entry_time > NOW() - INTERVAL '1 day' * $1
                    AND estimated_slippage_pct IS NOT NULL
            ) as trades
            GROUP BY vix_range, moneyness
            ORDER BY vix_range, moneyness
            "#,
            days_unwrapped as f64
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to analyze slippage")?;

        let mut results = Vec::new();
        for row in rows {
            let avg_estimated = row.avg_estimated.unwrap_or_default();
            let avg_actual = row.avg_actual.unwrap_or_default();
            let error = avg_actual - avg_estimated;
            let error_pct = if avg_estimated != 0.0 {
                (error / avg_estimated) * 100.0
            } else {
                0.0
            };

            results.push(SlippageResult {
                vix_range: row.vix_range.unwrap_or_default(),
                moneyness: row.moneyness.unwrap_or_default(),
                total_trades: row.total_trades as i32,
                avg_estimated_slippage: avg_estimated,
                avg_actual_slippage: avg_actual,
                slippage_error: error,
                slippage_error_pct: error_pct,
            });
        }

        Ok(SlippageAnalysis { results })
    }

    /// Calibrate slippage model and save recommendations
    pub async fn calibrate_slippage(&self) -> Result<Vec<SlippageCalibration>> {
        let analysis = self.analyze_slippage(Some(30)).await?;

        let mut calibrations = Vec::new();
        let calibration_date = chrono::Utc::now().date_naive();

        for result in analysis.results {
            // Only calibrate if we have enough data
            if result.total_trades < 5 {
                continue;
            }

            // Calculate recommended slippage (blend of current estimate and actual)
            // Use 70% actual, 30% estimated to avoid overreacting
            let recommended = (result.avg_actual_slippage * 0.7)
                            + (result.avg_estimated_slippage * 0.3);

            let calibration_id = Uuid::new_v4();

            sqlx::query!(
                r#"
                INSERT INTO slippage_calibration (
                    id, calibration_date, vix_range, moneyness,
                    total_trades, avg_estimated_slippage, avg_actual_slippage,
                    slippage_error, slippage_error_pct, recommended_slippage
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                "#,
                calibration_id,
                calibration_date,
                result.vix_range,
                result.moneyness,
                result.total_trades as i32,
                result.avg_estimated_slippage,
                result.avg_actual_slippage,
                result.slippage_error,
                result.slippage_error_pct,
                recommended
            )
            .execute(&self.pool)
            .await
            .context("Failed to save calibration")?;

            calibrations.push(SlippageCalibration {
                vix_range: result.vix_range,
                moneyness: result.moneyness,
                recommended_slippage: recommended,
                total_trades: result.total_trades,
                slippage_error_pct: result.slippage_error_pct,
            });
        }

        Ok(calibrations)
    }
}

#[derive(Debug, Serialize)]
pub struct SlippageAnalysis {
    pub results: Vec<SlippageResult>,
}

#[derive(Debug, Serialize)]
pub struct SlippageResult {
    pub vix_range: String,
    pub moneyness: String,
    pub total_trades: i32,
    pub avg_estimated_slippage: f64,
    pub avg_actual_slippage: f64,
    pub slippage_error: f64,
    pub slippage_error_pct: f64,
}

#[derive(Debug, Serialize)]
pub struct SlippageCalibration {
    pub vix_range: String,
    pub moneyness: String,
    pub recommended_slippage: f64,
    pub total_trades: i32,
    pub slippage_error_pct: f64,
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

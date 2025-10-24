// Circuit Breaker System
// Implements trading halts based on loss limits and consecutive losses

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tracing::{info, warn};
use uuid::Uuid;

use super::account::AccountManager;
use super::paper::PaperTradingEngine;

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type, PartialEq)]
#[sqlx(type_name = "text", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CircuitBreakerReason {
    DailyLossLimit,
    WeeklyLossLimit,
    ConsecutiveLosses,
    SystemError,
    ManualHalt,
}

impl std::fmt::Display for CircuitBreakerReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitBreakerReason::DailyLossLimit => write!(f, "DAILY_LOSS_LIMIT"),
            CircuitBreakerReason::WeeklyLossLimit => write!(f, "WEEKLY_LOSS_LIMIT"),
            CircuitBreakerReason::ConsecutiveLosses => write!(f, "CONSECUTIVE_LOSSES"),
            CircuitBreakerReason::SystemError => write!(f, "SYSTEM_ERROR"),
            CircuitBreakerReason::ManualHalt => write!(f, "MANUAL_HALT"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerState {
    pub id: Uuid,
    pub is_halted: bool,
    pub reason: Option<CircuitBreakerReason>,
    pub halted_at: Option<DateTime<Utc>>,
    pub resumed_at: Option<DateTime<Utc>>,
    pub triggered_by: Option<String>,
    pub notes: Option<serde_json::Value>,
    pub version: Option<i32>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Maximum daily loss percentage (e.g., 0.03 = 3%)
    pub daily_loss_limit_pct: f64,

    /// Maximum weekly loss percentage (e.g., 0.10 = 10%)
    pub weekly_loss_limit_pct: f64,

    /// Maximum consecutive losses before halting
    pub max_consecutive_losses: i32,

    /// Enable daily loss limit check
    pub enable_daily_limit: bool,

    /// Enable weekly loss limit check
    pub enable_weekly_limit: bool,

    /// Enable consecutive loss check
    pub enable_consecutive_loss_check: bool,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            daily_loss_limit_pct: 0.03,  // 3% daily loss
            weekly_loss_limit_pct: 0.10, // 10% weekly loss
            max_consecutive_losses: 5,
            enable_daily_limit: true,
            enable_weekly_limit: true,
            enable_consecutive_loss_check: true,
        }
    }
}

pub struct CircuitBreaker {
    pool: PgPool,
    config: CircuitBreakerConfig,
    account_manager: AccountManager,
    trading_engine: PaperTradingEngine,
}

impl CircuitBreaker {
    pub fn new(pool: PgPool, config: CircuitBreakerConfig) -> Self {
        Self {
            account_manager: AccountManager::new(pool.clone()),
            trading_engine: PaperTradingEngine::new(pool.clone()),
            pool,
            config,
        }
    }

    /// Get current circuit breaker state
    pub async fn get_current_state(&self) -> Result<CircuitBreakerState> {
        let state = sqlx::query_as!(
            CircuitBreakerState,
            r#"
            SELECT
                id,
                is_halted,
                reason as "reason: CircuitBreakerReason",
                halted_at,
                resumed_at,
                triggered_by,
                notes,
                version,
                created_at
            FROM circuit_breakers
            ORDER BY created_at DESC
            LIMIT 1
            "#
        )
        .fetch_optional(&self.pool)
        .await
        .context("Failed to fetch circuit breaker state")?;

        // If no state exists, create a default one
        match state {
            Some(state) => Ok(state),
            None => {
                info!("No circuit breaker state found, creating default state");
                let now = Utc::now();
                let id = Uuid::new_v4();

                sqlx::query!(
                    r#"
                    INSERT INTO circuit_breakers (id, is_halted, created_at)
                    VALUES ($1, $2, $3)
                    "#,
                    id,
                    false,
                    now
                )
                .execute(&self.pool)
                .await
                .context("Failed to create default circuit breaker state")?;

                Ok(CircuitBreakerState {
                    id,
                    is_halted: false,
                    reason: None,
                    halted_at: None,
                    resumed_at: None,
                    triggered_by: None,
                    notes: None,
                    version: None,
                    created_at: now,
                })
            }
        }
    }

    /// Get current circuit breaker state with row-level lock (for update)
    /// This prevents race conditions during check-and-halt operations
    async fn get_current_state_for_update(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    ) -> Result<CircuitBreakerState> {
        let state = sqlx::query_as!(
            CircuitBreakerState,
            r#"
            SELECT
                id,
                is_halted,
                reason as "reason: CircuitBreakerReason",
                halted_at,
                resumed_at,
                triggered_by,
                notes,
                version,
                created_at
            FROM circuit_breakers
            ORDER BY created_at DESC
            LIMIT 1
            FOR UPDATE
            "#
        )
        .fetch_optional(&mut **tx)
        .await
        .context("Failed to fetch circuit breaker state with lock")?;

        // If no state exists, create a default one within the transaction
        match state {
            Some(state) => Ok(state),
            None => {
                info!("No circuit breaker state found in transaction, creating default state");
                let now = Utc::now();
                let id = Uuid::new_v4();

                sqlx::query!(
                    r#"
                    INSERT INTO circuit_breakers (id, is_halted, created_at)
                    VALUES ($1, $2, $3)
                    "#,
                    id,
                    false,
                    now
                )
                .execute(&mut **tx)
                .await
                .context("Failed to create default circuit breaker state in transaction")?;

                Ok(CircuitBreakerState {
                    id,
                    is_halted: false,
                    reason: None,
                    halted_at: None,
                    resumed_at: None,
                    triggered_by: None,
                    notes: None,
                    version: None,
                    created_at: now,
                })
            }
        }
    }

    /// Check if trading should be halted and update state if necessary
    ///
    /// Returns: (should_halt, reason)
    ///
    /// This method uses a database transaction with row-level locking to prevent
    /// race conditions when multiple processes check the circuit breaker simultaneously.
    pub async fn check_and_halt(&self) -> Result<(bool, Option<CircuitBreakerReason>)> {
        // Start a transaction for atomic check-and-halt
        let mut tx = self.pool.begin().await?;

        // Get current state with row-level lock (SELECT FOR UPDATE)
        let current_state = self.get_current_state_for_update(&mut tx).await?;

        // If already halted, commit transaction and return
        if current_state.is_halted {
            tx.commit().await?;
            return Ok((true, current_state.reason));
        }

        let account = self.account_manager.get_current_account().await?;

        // Get initial balance from first account record
        let initial_balance = sqlx::query_scalar!(
            "SELECT balance FROM account_balance ORDER BY timestamp ASC LIMIT 1"
        )
        .fetch_optional(&self.pool)
        .await?
        .unwrap_or(10000.0); // Fallback to default if no records exist

        // Determine if we should halt and the reason
        let mut should_halt = false;
        let mut halt_reason: Option<CircuitBreakerReason> = None;
        let mut triggered_by: Option<String> = None;

        // Check daily loss limit
        if self.config.enable_daily_limit {
            if let Some(daily_pnl) = account.daily_pnl {
                let daily_loss_pct = daily_pnl.abs() / initial_balance;
                if daily_pnl < 0.0 && daily_loss_pct > self.config.daily_loss_limit_pct {
                    should_halt = true;
                    halt_reason = Some(CircuitBreakerReason::DailyLossLimit);
                    triggered_by = Some(format!(
                        "Daily loss {:.2}% exceeds limit {:.2}%",
                        daily_loss_pct * 100.0,
                        self.config.daily_loss_limit_pct * 100.0
                    ));
                }
            }
        }

        // Check weekly loss limit (if not already halting)
        if !should_halt && self.config.enable_weekly_limit {
            if let Some(weekly_pnl) = account.weekly_pnl {
                let weekly_loss_pct = weekly_pnl.abs() / initial_balance;
                if weekly_pnl < 0.0 && weekly_loss_pct > self.config.weekly_loss_limit_pct {
                    should_halt = true;
                    halt_reason = Some(CircuitBreakerReason::WeeklyLossLimit);
                    triggered_by = Some(format!(
                        "Weekly loss {:.2}% exceeds limit {:.2}%",
                        weekly_loss_pct * 100.0,
                        self.config.weekly_loss_limit_pct * 100.0
                    ));
                }
            }
        }

        // Check consecutive losses (if not already halting)
        if !should_halt && self.config.enable_consecutive_loss_check {
            let consecutive_losses = self.count_consecutive_losses().await?;
            if consecutive_losses >= self.config.max_consecutive_losses {
                should_halt = true;
                halt_reason = Some(CircuitBreakerReason::ConsecutiveLosses);
                triggered_by = Some(format!(
                    "{} consecutive losses (limit: {})",
                    consecutive_losses, self.config.max_consecutive_losses
                ));
            }
        }

        // If we should halt, insert the halt record within the same transaction
        if should_halt {
            if let (Some(reason), Some(trigger_msg)) = (halt_reason.clone(), triggered_by) {
                let halted_at = Utc::now();

                sqlx::query!(
                    r#"
                    INSERT INTO circuit_breakers (is_halted, reason, halted_at, triggered_by, created_at)
                    VALUES ($1, $2::text, $3, $4, $5)
                    "#,
                    true,
                    reason.to_string(),
                    halted_at,
                    trigger_msg.clone(),
                    halted_at
                )
                .execute(&mut *tx)
                .await
                .context("Failed to insert circuit breaker halt")?;

                warn!("CIRCUIT BREAKER TRIGGERED: {:?} - {}", reason, trigger_msg);
            }
        }

        // Commit the transaction
        tx.commit().await?;

        Ok((should_halt, halt_reason))
    }

    /// Halt trading
    ///
    /// This method uses a transaction with row-level locking to ensure
    /// only one halt can be recorded at a time, preventing race conditions.
    async fn halt(&self, reason: CircuitBreakerReason, triggered_by: String) -> Result<()> {
        self.halt_with_notes(reason, triggered_by, None).await
    }

    /// Halt trading with optional notes
    ///
    /// This method uses a transaction with row-level locking to ensure
    /// only one halt can be recorded at a time, preventing race conditions.
    async fn halt_with_notes(
        &self,
        reason: CircuitBreakerReason,
        triggered_by: String,
        notes: Option<String>,
    ) -> Result<()> {
        let mut tx = self.pool.begin().await?;

        // Check current state with lock to prevent duplicate halts
        let current_state = self.get_current_state_for_update(&mut tx).await?;

        if current_state.is_halted {
            // Already halted, just commit and return
            tx.commit().await?;
            info!("Circuit breaker already halted, skipping duplicate halt");
            return Ok(());
        }

        let halted_at = Utc::now();

        sqlx::query!(
            r#"
            INSERT INTO circuit_breakers (is_halted, reason, halted_at, triggered_by, notes, created_at)
            VALUES ($1, $2::text, $3, $4, $5, $6)
            "#,
            true,
            reason.to_string(),
            halted_at,
            triggered_by.clone(),
            notes.map(|n| serde_json::Value::String(n)),
            halted_at
        )
        .execute(&mut *tx)
        .await
        .context("Failed to insert circuit breaker halt")?;

        tx.commit().await?;

        warn!("CIRCUIT BREAKER TRIGGERED: {:?} - {}", reason, triggered_by);

        Ok(())
    }

    /// Manually halt trading
    pub async fn manual_halt(&self, notes: Option<String>) -> Result<()> {
        let triggered_by = "Manual halt requested".to_string();
        self.halt_with_notes(CircuitBreakerReason::ManualHalt, triggered_by, notes)
            .await
    }

    /// Resume trading (requires manual override)
    pub async fn resume_trading(&self, manual_override: bool) -> Result<()> {
        if !manual_override {
            bail!("Circuit breaker resume requires manual override");
        }

        let current_state = self.get_current_state().await?;

        if !current_state.is_halted {
            bail!("Trading is not currently halted");
        }

        let resumed_at = Utc::now();

        sqlx::query!(
            r#"
            INSERT INTO circuit_breakers (is_halted, resumed_at, created_at)
            VALUES ($1, $2, $3)
            "#,
            false,
            resumed_at,
            resumed_at
        )
        .execute(&self.pool)
        .await
        .context("Failed to resume trading")?;

        info!("Trading resumed manually");

        Ok(())
    }

    /// Count consecutive losing trades
    async fn count_consecutive_losses(&self) -> Result<i32> {
        let recent_trades = self.trading_engine.get_all_trades(Some(20)).await?;

        let mut consecutive = 0;

        for trade in recent_trades {
            if let Some(pnl) = trade.pnl {
                if pnl < 0.0 {
                    consecutive += 1;
                } else {
                    break;
                }
            }
        }

        Ok(consecutive)
    }

    /// Check if trading is currently allowed
    pub async fn is_trading_allowed(&self) -> Result<bool> {
        let state = self.get_current_state().await?;
        Ok(!state.is_halted)
    }

    /// Halt trading due to system error
    pub async fn halt_on_error(&self, error_message: String) -> Result<()> {
        self.halt(CircuitBreakerReason::SystemError, error_message)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_reason_display() {
        assert_eq!(
            CircuitBreakerReason::DailyLossLimit.to_string(),
            "DAILY_LOSS_LIMIT"
        );
        assert_eq!(
            CircuitBreakerReason::WeeklyLossLimit.to_string(),
            "WEEKLY_LOSS_LIMIT"
        );
        assert_eq!(
            CircuitBreakerReason::ConsecutiveLosses.to_string(),
            "CONSECUTIVE_LOSSES"
        );
        assert_eq!(
            CircuitBreakerReason::SystemError.to_string(),
            "SYSTEM_ERROR"
        );
        assert_eq!(CircuitBreakerReason::ManualHalt.to_string(), "MANUAL_HALT");
    }

    #[test]
    fn test_default_config() {
        let config = CircuitBreakerConfig::default();

        assert_eq!(config.daily_loss_limit_pct, 0.03);
        assert_eq!(config.weekly_loss_limit_pct, 0.10);
        assert_eq!(config.max_consecutive_losses, 5);
        assert!(config.enable_daily_limit);
        assert!(config.enable_weekly_limit);
        assert!(config.enable_consecutive_loss_check);
    }
}

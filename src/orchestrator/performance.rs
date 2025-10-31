//! Performance tracking and confidence calibration
//! Tracks recent trading performance and adjusts confidence based on actual results

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tracing::{info, warn};

use crate::trading::PlattScaler;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradePerformance {
    pub trade_id: uuid::Uuid,
    pub entry_time: DateTime<Utc>,
    pub exit_time: Option<DateTime<Utc>>,
    pub confidence: f32,
    pub won: bool,
    pub pnl_pct: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub total_trades: usize,
    pub wins: usize,
    pub losses: usize,
    pub win_rate: f32,
    pub consecutive_losses: usize,
    pub consecutive_wins: usize,
    pub avg_confidence: f32,
}

pub struct PerformanceTracker {
    pool: PgPool,
    scaler: PlattScaler,
}

impl PerformanceTracker {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            scaler: PlattScaler::new(),
        }
    }

    /// Update Platt scaler with new trade outcome
    pub fn record_trade_outcome(&mut self, confidence: f64, won: bool) {
        self.scaler.record(confidence, won);
        self.scaler.fit(); // Refit after each new trade
    }

    /// Get current calibration report
    pub fn get_calibration_report(&self) -> String {
        self.scaler.calibration_report()
    }

    /// Get recent trade performance (last N closed trades)
    pub async fn get_recent_performance(&self, limit: i64) -> Result<Vec<TradePerformance>> {
        let trades = sqlx::query!(
            r#"
            SELECT
                pt.id,
                pt.entry_time,
                pt.exit_time,
                COALESCE(ac.confidence, 0.5) as "confidence!",
                pt.pnl_pct
            FROM paper_trades pt
            LEFT JOIN ace_contexts ac ON pt.context_id = ac.id
            WHERE pt.status = 'CLOSED'
            AND pt.exit_time IS NOT NULL
            ORDER BY pt.exit_time DESC
            LIMIT $1
            "#,
            limit
        )
        .fetch_all(&self.pool)
        .await?;

        let performance: Vec<TradePerformance> = trades
            .into_iter()
            .map(|t| {
                let pnl_pct = t.pnl_pct;
                let won = pnl_pct.unwrap_or(0.0) > 0.0;
                TradePerformance {
                    trade_id: t.id,
                    entry_time: t.entry_time,
                    exit_time: t.exit_time,
                    confidence: t.confidence,
                    won,
                    pnl_pct,
                }
            })
            .collect();

        Ok(performance)
    }

    /// Calculate performance statistics from recent trades
    pub fn calculate_stats(&self, trades: &[TradePerformance]) -> PerformanceStats {
        if trades.is_empty() {
            return PerformanceStats {
                total_trades: 0,
                wins: 0,
                losses: 0,
                win_rate: 0.0,
                consecutive_losses: 0,
                consecutive_wins: 0,
                avg_confidence: 0.5,
            };
        }

        let total_trades = trades.len();
        let wins = trades.iter().filter(|t| t.won).count();
        let losses = total_trades - wins;
        let win_rate = wins as f32 / total_trades as f32;
        let avg_confidence = trades.iter().map(|t| t.confidence).sum::<f32>() / total_trades as f32;

        // Calculate consecutive losses from most recent trades
        let mut consecutive_losses = 0;
        for trade in trades {
            if !trade.won {
                consecutive_losses += 1;
            } else {
                break;
            }
        }

        // Calculate consecutive wins
        let mut consecutive_wins = 0;
        for trade in trades {
            if trade.won {
                consecutive_wins += 1;
            } else {
                break;
            }
        }

        PerformanceStats {
            total_trades,
            wins,
            losses,
            win_rate,
            consecutive_losses,
            consecutive_wins,
            avg_confidence,
        }
    }

    /// Calibrate confidence using Platt scaling
    ///
    /// Maps raw LLM confidence to calibrated probability based on historical accuracy
    pub fn calibrate_confidence(
        &self,
        raw_confidence: f32,
        stats: &PerformanceStats,
    ) -> (f32, String) {
        // Use Platt scaling for professional calibration
        let calibrated = self.scaler.calibrate(raw_confidence as f64) as f32;

        let adjustment_summary = if stats.total_trades < 30 {
            format!("Platt scaling (identity - only {} samples)", stats.total_trades)
        } else {
            let brier = self.scaler.brier_score()
                .map(|b| format!("Brier: {:.4}", b))
                .unwrap_or_else(|| "Brier: N/A".to_string());
            format!("Platt scaled | {} | {} samples", brier, stats.total_trades)
        };

        // Add safety checks for consecutive losses
        // Increased cap to 75% (from 60%) for higher risk tolerance
        let final_calibrated = if stats.consecutive_losses >= 3 {
            // Cap at 75% after 3+ consecutive losses for safety
            calibrated.min(0.75)
        } else {
            calibrated
        };

        let full_summary = if final_calibrated < calibrated {
            format!("{} | capped at 75% ({}L streak)", adjustment_summary, stats.consecutive_losses)
        } else {
            adjustment_summary
        };

        (final_calibrated, full_summary)
    }

    /// Get calibrated confidence for decision making
    /// This is the main entry point called by the morning orchestrator
    pub async fn get_calibrated_confidence(
        &self,
        raw_confidence: f32,
        lookback_trades: i64,
    ) -> Result<(f32, PerformanceStats, String)> {
        // Get recent trade performance
        let recent_trades = self.get_recent_performance(lookback_trades).await?;

        // Calculate performance statistics
        let stats = self.calculate_stats(&recent_trades);

        // Calibrate confidence based on performance
        let (calibrated_confidence, adjustment_summary) = self.calibrate_confidence(raw_confidence, &stats);

        // Log calibration results
        if stats.total_trades > 0 {
            info!(
                "Confidence calibration: {:.1}% → {:.1}% | Win rate: {:.1}% ({}/{}) | Consecutive: {}L/{}W | {}",
                raw_confidence * 100.0,
                calibrated_confidence * 100.0,
                stats.win_rate * 100.0,
                stats.wins,
                stats.total_trades,
                stats.consecutive_losses,
                stats.consecutive_wins,
                adjustment_summary
            );

            // Warn if significant calibration occurred
            let adjustment_magnitude = (calibrated_confidence - raw_confidence).abs();
            if adjustment_magnitude > 0.1 {
                warn!(
                    "⚠️ Large confidence adjustment: {:.1}% → {:.1}% (Δ {:.1}%) | Reason: {}",
                    raw_confidence * 100.0,
                    calibrated_confidence * 100.0,
                    adjustment_magnitude * 100.0,
                    adjustment_summary
                );
            }
        } else {
            info!(
                "Confidence: {:.1}% (no calibration - no recent trades)",
                raw_confidence * 100.0
            );
        }

        Ok((calibrated_confidence, stats, adjustment_summary))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_dummy_tracker() -> PerformanceTracker {
        // Create a dummy tracker for testing calibration logic
        // Note: This won't have a valid connection, but calibrate_confidence doesn't need it
        PerformanceTracker {
            pool: unsafe { std::mem::zeroed() }, // Dummy pool for testing
            scaler: PlattScaler::new(),
        }
    }

    #[test]
    fn test_consecutive_loss_cap() {
        let tracker = create_dummy_tracker();

        let stats = PerformanceStats {
            total_trades: 5,
            wins: 1,
            losses: 4,
            win_rate: 0.2,
            consecutive_losses: 4,
            consecutive_wins: 0,
            avg_confidence: 0.8,
        };

        let (calibrated, summary) = tracker.calibrate_confidence(0.85, &stats);

        // Should be capped at 0.60 for 3+ consecutive losses
        assert_eq!(calibrated, 0.6);
        assert!(summary.contains("capped"));
    }

    #[test]
    fn test_platt_scaling_identity_with_few_samples() {
        let tracker = create_dummy_tracker();

        let stats = PerformanceStats {
            total_trades: 10,
            wins: 5,
            losses: 5,
            win_rate: 0.5,
            consecutive_losses: 0,
            consecutive_wins: 0,
            avg_confidence: 0.6,
        };

        let (calibrated, summary) = tracker.calibrate_confidence(0.7, &stats);

        // With < 30 samples, Platt scaling uses identity mapping
        assert_eq!(calibrated, 0.7);
        assert!(summary.contains("identity"));
    }

    #[test]
    fn test_record_trade_outcome() {
        let mut tracker = create_dummy_tracker();

        // Record some trades
        tracker.record_trade_outcome(0.7, true);
        tracker.record_trade_outcome(0.6, false);
        tracker.record_trade_outcome(0.8, true);

        // Scaler should have recorded 3 trades
        assert_eq!(tracker.scaler.total_samples(), 3);
    }
}

//! Performance tracking and confidence calibration
//! Tracks recent trading performance and adjusts confidence based on actual results

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tracing::{info, warn};

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
}

impl PerformanceTracker {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
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

    /// Calibrate confidence based on recent performance
    ///
    /// Adjusts raw LLM confidence using recent win rate and consecutive losses
    /// to prevent overconfidence after repeated failures
    pub fn calibrate_confidence(
        &self,
        raw_confidence: f32,
        stats: &PerformanceStats,
    ) -> (f32, String) {
        // Start with raw confidence
        let mut calibrated = raw_confidence;
        let mut adjustments = Vec::new();

        // If we have insufficient trade history, don't calibrate
        if stats.total_trades < 3 {
            return (
                calibrated,
                "Insufficient trade history for calibration (need 3+ trades)".to_string(),
            );
        }

        // 1. Apply consecutive loss penalty
        // Each consecutive loss reduces confidence by 5%, min 30%
        if stats.consecutive_losses > 0 {
            let loss_penalty = (stats.consecutive_losses as f32) * 0.05;
            calibrated = (calibrated - loss_penalty).max(0.3);
            adjustments.push(format!(
                "consecutive_losses:{} (-{:.1}%)",
                stats.consecutive_losses,
                loss_penalty * 100.0
            ));
        }

        // 2. Apply win rate calibration
        // If recent win rate is significantly below confidence, reduce confidence
        if stats.total_trades >= 5 {
            let confidence_error = raw_confidence - stats.win_rate;
            if confidence_error > 0.15 {
                // Overconfident by more than 15%
                let error_penalty = confidence_error * 0.3; // Reduce by 30% of error
                calibrated = (calibrated - error_penalty).max(0.3);
                adjustments.push(format!(
                    "win_rate:{:.1}% vs conf:{:.1}% (-{:.1}%)",
                    stats.win_rate * 100.0,
                    raw_confidence * 100.0,
                    error_penalty * 100.0
                ));
            }
        }

        // 3. Cap confidence at 60% after 3+ consecutive losses
        if stats.consecutive_losses >= 3 {
            if calibrated > 0.6 {
                calibrated = 0.6;
                adjustments.push("max_cap:60% (3+ losses)".to_string());
            }
        }

        // 4. Gradual recovery with wins
        // After consecutive wins, allow confidence to increase slightly
        if stats.consecutive_wins >= 2 && stats.total_trades >= 5 {
            let recovery_boost = (stats.consecutive_wins as f32 * 0.03).min(0.1);
            calibrated = (calibrated + recovery_boost).min(0.95);
            adjustments.push(format!(
                "win_recovery:+{:.1}% ({} wins)",
                recovery_boost * 100.0,
                stats.consecutive_wins
            ));
        }

        // Ensure confidence stays within valid bounds
        calibrated = calibrated.clamp(0.3, 0.95);

        let adjustment_summary = if adjustments.is_empty() {
            "No adjustments needed".to_string()
        } else {
            adjustments.join(", ")
        };

        (calibrated, adjustment_summary)
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
        }
    }

    #[test]
    fn test_consecutive_loss_penalty() {
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

        // Should apply 4 * 0.05 = 0.20 penalty: 0.85 - 0.20 = 0.65
        // But then capped at 0.60 for 3+ consecutive losses
        assert_eq!(calibrated, 0.6);
        assert!(summary.contains("consecutive_losses"));
    }

    #[test]
    fn test_win_rate_calibration() {
        let tracker = create_dummy_tracker();

        let stats = PerformanceStats {
            total_trades: 10,
            wins: 3,
            losses: 7,
            win_rate: 0.3,
            consecutive_losses: 1,
            consecutive_wins: 0,
            avg_confidence: 0.8,
        };

        let (calibrated, summary) = tracker.calibrate_confidence(0.8, &stats);

        // Confidence 0.8 vs win rate 0.3 = 0.5 error (> 0.15)
        // Should apply error penalty
        assert!(calibrated < 0.8);
        assert!(summary.contains("win_rate"));
    }

    #[test]
    fn test_recovery_boost() {
        let tracker = create_dummy_tracker();

        let stats = PerformanceStats {
            total_trades: 10,
            wins: 5,
            losses: 5,
            win_rate: 0.5,
            consecutive_losses: 0,
            consecutive_wins: 3,
            avg_confidence: 0.6,
        };

        let (calibrated, summary) = tracker.calibrate_confidence(0.6, &stats);

        // Should apply recovery boost for 3 consecutive wins
        assert!(calibrated > 0.6);
        assert!(summary.contains("win_recovery"));
    }

    #[test]
    fn test_insufficient_history() {
        let tracker = create_dummy_tracker();

        let stats = PerformanceStats {
            total_trades: 2,
            wins: 1,
            losses: 1,
            win_rate: 0.5,
            consecutive_losses: 0,
            consecutive_wins: 1,
            avg_confidence: 0.7,
        };

        let (calibrated, summary) = tracker.calibrate_confidence(0.75, &stats);

        // Should not calibrate with insufficient history
        assert_eq!(calibrated, 0.75);
        assert!(summary.contains("Insufficient"));
    }
}

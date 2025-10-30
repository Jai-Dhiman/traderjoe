//! Risk Management Module
//! Implements stop-loss, take-profit, VIX restrictions, and volatility-based position sizing

use anyhow::{bail, Result};
use tracing::{info, warn};

use super::TradeType;

/// Risk management configuration
#[derive(Debug, Clone)]
pub struct RiskConfig {
    /// Stop-loss percentage for options (e.g., 0.15 = 15%)
    pub stop_loss_options_pct: f64,

    /// Stop-loss percentage for shares (e.g., 0.02 = 2%)
    pub stop_loss_shares_pct: f64,

    /// Take-profit percentage for options (e.g., 0.50 = 50%)
    pub take_profit_options_pct: f64,

    /// Take-profit percentage for shares (e.g., 0.05 = 5%)
    pub take_profit_shares_pct: f64,

    /// VIX threshold above which trading is restricted
    pub vix_restriction_threshold: f64,

    /// Minimum confidence required to trade when VIX is high
    pub vix_high_min_confidence: f64,

    /// Volatility scaling factor for position sizing
    /// Higher values = more aggressive reduction in high volatility
    pub volatility_scaling_factor: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            stop_loss_options_pct: 0.15,      // 15% stop-loss for options
            stop_loss_shares_pct: 0.02,       // 2% stop-loss for shares
            take_profit_options_pct: 0.50,    // 50% take-profit for options
            take_profit_shares_pct: 0.05,     // 5% take-profit for shares
            vix_restriction_threshold: 40.0,  // VIX above 40 requires high confidence
            vix_high_min_confidence: 0.85,    // 85% minimum confidence when VIX > 40
            volatility_scaling_factor: 0.3,   // Reduce position size by 30% per 10 VIX points
        }
    }
}

/// Risk manager for trade execution and monitoring
pub struct RiskManager {
    config: RiskConfig,
}

impl RiskManager {
    pub fn new(config: RiskConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(RiskConfig::default())
    }

    /// Check if trading is allowed based on VIX and confidence
    ///
    /// Returns Ok(()) if trading is allowed, Err otherwise
    pub fn check_vix_restriction(&self, vix: Option<f64>, confidence: f64) -> Result<()> {
        if let Some(vix_level) = vix {
            if vix_level > self.config.vix_restriction_threshold {
                if confidence < self.config.vix_high_min_confidence {
                    warn!(
                        "Trading BLOCKED: VIX {:.1} > {:.1} requires confidence >= {:.1}%, got {:.1}%",
                        vix_level,
                        self.config.vix_restriction_threshold,
                        self.config.vix_high_min_confidence * 100.0,
                        confidence * 100.0
                    );
                    bail!(
                        "VIX {:.1} is too high. Requires {:.0}% confidence, got {:.0}%",
                        vix_level,
                        self.config.vix_high_min_confidence * 100.0,
                        confidence * 100.0
                    );
                } else {
                    info!(
                        "High VIX {:.1} but confidence {:.1}% >= {:.1}% threshold - trade allowed",
                        vix_level,
                        confidence * 100.0,
                        self.config.vix_high_min_confidence * 100.0
                    );
                }
            }
        }
        Ok(())
    }

    /// Calculate volatility-adjusted position size
    ///
    /// Reduces position size in high volatility environments to manage risk
    pub fn calculate_volatility_adjusted_position_size(
        &self,
        base_position_size: f64,
        vix: Option<f64>,
    ) -> f64 {
        let vix_level = match vix {
            Some(v) => v,
            None => {
                info!("No VIX data available, using base position size");
                return base_position_size;
            }
        };

        // Baseline VIX (normal market conditions)
        let baseline_vix = 20.0;

        if vix_level <= baseline_vix {
            // Low/normal volatility: no adjustment needed
            info!(
                "VIX {:.1} <= {:.1} (normal): using 100% of base position size ${:.2}",
                vix_level, baseline_vix, base_position_size
            );
            return base_position_size;
        }

        // Calculate reduction factor based on VIX level above baseline
        // Formula: reduction = (VIX - baseline) * scaling_factor / 10
        let vix_excess = vix_level - baseline_vix;
        let reduction_pct = (vix_excess * self.config.volatility_scaling_factor) / 10.0;

        // Cap reduction at 50% (never reduce more than half)
        let reduction_pct = reduction_pct.min(0.50);

        let adjustment_factor = 1.0 - reduction_pct;
        let adjusted_size = base_position_size * adjustment_factor;

        info!(
            "VIX {:.1} > {:.1}: reducing position by {:.1}% (${:.2} → ${:.2})",
            vix_level,
            baseline_vix,
            reduction_pct * 100.0,
            base_position_size,
            adjusted_size
        );

        adjusted_size
    }

    /// Check if stop-loss should be triggered
    ///
    /// Returns Ok(true) if stop-loss hit, Ok(false) if not
    pub fn check_stop_loss(
        &self,
        entry_price: f64,
        current_price: f64,
        trade_type: &TradeType,
        is_option: bool,
    ) -> Result<bool> {
        let stop_loss_pct = if is_option {
            self.config.stop_loss_options_pct
        } else {
            self.config.stop_loss_shares_pct
        };

        // Calculate P&L percentage based on trade direction
        let pnl_pct = match trade_type {
            TradeType::Call => (current_price - entry_price) / entry_price,
            TradeType::Put => (entry_price - current_price) / entry_price,
            TradeType::Flat => return Ok(false),
        };

        // Check if loss exceeds stop-loss threshold
        if pnl_pct <= -stop_loss_pct {
            warn!(
                "STOP-LOSS TRIGGERED: {} loss {:.1}% exceeds {:.1}% threshold (entry: ${:.2}, current: ${:.2})",
                trade_type,
                pnl_pct * 100.0,
                stop_loss_pct * 100.0,
                entry_price,
                current_price
            );
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Check if take-profit should be triggered
    ///
    /// Returns Ok(true) if take-profit hit, Ok(false) if not
    pub fn check_take_profit(
        &self,
        entry_price: f64,
        current_price: f64,
        trade_type: &TradeType,
        is_option: bool,
    ) -> Result<bool> {
        let take_profit_pct = if is_option {
            self.config.take_profit_options_pct
        } else {
            self.config.take_profit_shares_pct
        };

        // Calculate P&L percentage based on trade direction
        let pnl_pct = match trade_type {
            TradeType::Call => (current_price - entry_price) / entry_price,
            TradeType::Put => (entry_price - current_price) / entry_price,
            TradeType::Flat => return Ok(false),
        };

        // Check if profit exceeds take-profit threshold
        if pnl_pct >= take_profit_pct {
            info!(
                "TAKE-PROFIT TRIGGERED: {} profit {:.1}% exceeds {:.1}% threshold (entry: ${:.2}, current: ${:.2})",
                trade_type,
                pnl_pct * 100.0,
                take_profit_pct * 100.0,
                entry_price,
                current_price
            );
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get stop-loss price for a trade
    pub fn get_stop_loss_price(
        &self,
        entry_price: f64,
        trade_type: &TradeType,
        is_option: bool,
    ) -> f64 {
        let stop_loss_pct = if is_option {
            self.config.stop_loss_options_pct
        } else {
            self.config.stop_loss_shares_pct
        };

        match trade_type {
            TradeType::Call => entry_price * (1.0 - stop_loss_pct),
            TradeType::Put => entry_price * (1.0 + stop_loss_pct),
            TradeType::Flat => entry_price,
        }
    }

    /// Get take-profit price for a trade
    pub fn get_take_profit_price(
        &self,
        entry_price: f64,
        trade_type: &TradeType,
        is_option: bool,
    ) -> f64 {
        let take_profit_pct = if is_option {
            self.config.take_profit_options_pct
        } else {
            self.config.take_profit_shares_pct
        };

        match trade_type {
            TradeType::Call => entry_price * (1.0 + take_profit_pct),
            TradeType::Put => entry_price * (1.0 - take_profit_pct),
            TradeType::Flat => entry_price,
        }
    }

    /// Check if trailing stop should be triggered
    ///
    /// Trailing stop activates after +30% profit, then trails 10% behind peak
    ///
    /// # Arguments
    /// * `entry_price` - Entry price of the trade
    /// * `current_price` - Current market price
    /// * `max_favorable_excursion` - Peak favorable price movement (from MFE tracking)
    /// * `trade_type` - Call or Put
    ///
    /// Returns Ok(true) if trailing stop hit, Ok(false) if not
    pub fn check_trailing_stop(
        &self,
        entry_price: f64,
        current_price: f64,
        max_favorable_excursion: Option<f64>,
        trade_type: &TradeType,
    ) -> Result<bool> {
        // Trailing stop only applies to calls and puts
        if *trade_type == TradeType::Flat {
            return Ok(false);
        }

        // Need MFE to calculate trailing stop
        let mfe = match max_favorable_excursion {
            Some(m) => m,
            None => return Ok(false), // No peak yet, no trailing stop
        };

        // Calculate current P&L percentage
        let current_pnl_pct = match trade_type {
            TradeType::Call => (current_price - entry_price) / entry_price,
            TradeType::Put => (entry_price - current_price) / entry_price,
            TradeType::Flat => 0.0,
        };

        // Calculate peak P&L percentage from MFE
        let peak_pnl_pct = mfe;

        // Trailing stop activates after +30% profit
        const ACTIVATION_THRESHOLD: f64 = 0.30;
        if peak_pnl_pct < ACTIVATION_THRESHOLD {
            return Ok(false); // Not profitable enough to activate trailing stop
        }

        // Trail 10% behind peak
        const TRAIL_DISTANCE: f64 = 0.10;
        let trailing_stop_level = peak_pnl_pct - TRAIL_DISTANCE;

        // Check if current P&L has fallen below trailing stop level
        if current_pnl_pct < trailing_stop_level {
            warn!(
                "TRAILING STOP TRIGGERED: {} profit fell from peak {:.1}% to {:.1}% (trailing stop at {:.1}%)",
                trade_type,
                peak_pnl_pct * 100.0,
                current_pnl_pct * 100.0,
                trailing_stop_level * 100.0
            );
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Check if time-based exit should be triggered
    ///
    /// Closes same-day trades at 3:50 PM ET to avoid overnight risk
    ///
    /// # Arguments
    /// * `entry_time` - When the trade was entered
    /// * `current_time` - Current time
    ///
    /// Returns Ok(true) if should exit based on time, Ok(false) otherwise
    pub fn check_time_based_exit(
        &self,
        entry_time: chrono::DateTime<chrono::Utc>,
        current_time: chrono::DateTime<chrono::Utc>,
    ) -> Result<bool> {
        use chrono::Timelike;

        // Check if this is a same-day trade
        if entry_time.date_naive() != current_time.date_naive() {
            return Ok(false); // Not a same-day trade
        }

        // Convert to NY time (ET) for market hours check
        // Market closes at 4:00 PM ET, we want to exit at 3:50 PM ET
        // Simplified: Check if current hour is >= 15 (3 PM) and >= 50 minutes
        let hour = current_time.hour();
        let minute = current_time.minute();

        // Close at 3:50 PM ET (15:50) or later for same-day trades
        // Note: In UTC, this would be 19:50 or 20:50 depending on DST
        // For simplicity, we check if we're within last 10 minutes of trading day
        if hour >= 19 && minute >= 50 {
            info!(
                "TIME-BASED EXIT TRIGGERED: Same-day trade approaching market close ({}:{})",
                hour, minute
            );
            Ok(true)
        } else if hour >= 20 {
            // Definitely past market close
            info!(
                "TIME-BASED EXIT TRIGGERED: Past market close ({}:{})",
                hour, minute
            );
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vix_restriction_allowed() {
        let rm = RiskManager::with_defaults();

        // Low VIX, any confidence
        assert!(rm.check_vix_restriction(Some(25.0), 0.60).is_ok());

        // High VIX, high confidence
        assert!(rm.check_vix_restriction(Some(45.0), 0.90).is_ok());
    }

    #[test]
    fn test_vix_restriction_blocked() {
        let rm = RiskManager::with_defaults();

        // High VIX, low confidence should be blocked
        assert!(rm.check_vix_restriction(Some(45.0), 0.70).is_err());
    }

    #[test]
    fn test_volatility_adjusted_position_normal_vix() {
        let rm = RiskManager::with_defaults();
        let base_size = 1000.0;

        // VIX 20 or below: no adjustment
        let adjusted = rm.calculate_volatility_adjusted_position_size(base_size, Some(18.0));
        assert_eq!(adjusted, base_size);
    }

    #[test]
    fn test_volatility_adjusted_position_high_vix() {
        let rm = RiskManager::with_defaults();
        let base_size = 1000.0;

        // VIX 30 (10 points above baseline of 20)
        // Reduction: 10 * 0.3 / 10 = 0.3 = 30%
        // Adjusted: 1000 * 0.7 = 700
        let adjusted = rm.calculate_volatility_adjusted_position_size(base_size, Some(30.0));
        assert_eq!(adjusted, 700.0);
    }

    #[test]
    fn test_volatility_adjusted_position_extreme_vix() {
        let rm = RiskManager::with_defaults();
        let base_size = 1000.0;

        // VIX 60 (40 points above baseline)
        // Would be: 40 * 0.3 / 10 = 1.2 = 120% reduction
        // But capped at 50% reduction
        // Adjusted: 1000 * 0.5 = 500
        let adjusted = rm.calculate_volatility_adjusted_position_size(base_size, Some(60.0));
        assert_eq!(adjusted, 500.0);
    }

    #[test]
    fn test_stop_loss_call_triggered() {
        let rm = RiskManager::with_defaults();

        // Call option: entry $100, current $80 = -20% loss (exceeds 15% stop)
        let triggered = rm.check_stop_loss(100.0, 80.0, &TradeType::Call, true).unwrap();
        assert!(triggered);
    }

    #[test]
    fn test_stop_loss_call_not_triggered() {
        let rm = RiskManager::with_defaults();

        // Call option: entry $100, current $90 = -10% loss (below 15% stop)
        let triggered = rm.check_stop_loss(100.0, 90.0, &TradeType::Call, true).unwrap();
        assert!(!triggered);
    }

    #[test]
    fn test_stop_loss_put_triggered() {
        let rm = RiskManager::with_defaults();

        // Put option: entry $100, current $120 = -20% loss (exceeds 15% stop)
        let triggered = rm.check_stop_loss(100.0, 120.0, &TradeType::Put, true).unwrap();
        assert!(triggered);
    }

    #[test]
    fn test_take_profit_call_triggered() {
        let rm = RiskManager::with_defaults();

        // Call option: entry $100, current $160 = +60% profit (exceeds 50% target)
        let triggered = rm.check_take_profit(100.0, 160.0, &TradeType::Call, true).unwrap();
        assert!(triggered);
    }

    #[test]
    fn test_take_profit_call_not_triggered() {
        let rm = RiskManager::with_defaults();

        // Call option: entry $100, current $130 = +30% profit (below 50% target)
        let triggered = rm.check_take_profit(100.0, 130.0, &TradeType::Call, true).unwrap();
        assert!(!triggered);
    }

    #[test]
    fn test_stop_loss_price_call() {
        let rm = RiskManager::with_defaults();

        // Call: entry $100, stop-loss 15% = $85
        let stop_price = rm.get_stop_loss_price(100.0, &TradeType::Call, true);
        assert_eq!(stop_price, 85.0);
    }

    #[test]
    fn test_stop_loss_price_put() {
        let rm = RiskManager::with_defaults();

        // Put: entry $100, stop-loss 15% = $115
        let stop_price = rm.get_stop_loss_price(100.0, &TradeType::Put, true);
        assert_eq!(stop_price, 115.0);
    }

    #[test]
    fn test_take_profit_price_call() {
        let rm = RiskManager::with_defaults();

        // Call: entry $100, take-profit 50% = $150
        let tp_price = rm.get_take_profit_price(100.0, &TradeType::Call, true);
        assert_eq!(tp_price, 150.0);
    }

    #[test]
    fn test_shares_vs_options_stop_loss() {
        let rm = RiskManager::with_defaults();

        // Options: 15% stop-loss
        let options_stop = rm.get_stop_loss_price(100.0, &TradeType::Call, true);
        assert_eq!(options_stop, 85.0);

        // Shares: 2% stop-loss
        let shares_stop = rm.get_stop_loss_price(100.0, &TradeType::Call, false);
        assert_eq!(shares_stop, 98.0);
    }
}

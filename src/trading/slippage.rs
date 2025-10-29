//! Slippage modeling for realistic paper trading simulation
//!
//! SPY options are among the most liquid in the world. Realistic slippage models
//! are critical for accurate backtest results.

use tracing::debug;

/// Configuration for slippage calculation
#[derive(Debug, Clone)]
pub struct SlippageConfig {
    /// Base slippage for highly liquid SPY options (ATM, high OI)
    pub base_slippage_pct: f64,

    /// Multiplier for high volatility periods (VIX > 25)
    pub high_vix_multiplier: f64,

    /// Multiplier for extreme volatility (VIX > 40)
    pub extreme_vix_multiplier: f64,

    /// VIX threshold for high volatility adjustment
    pub high_vix_threshold: f64,

    /// VIX threshold for extreme volatility adjustment
    pub extreme_vix_threshold: f64,
}

impl Default for SlippageConfig {
    fn default() -> Self {
        Self {
            // SPY options typically have 0.3-0.5% bid-ask spread in normal conditions
            base_slippage_pct: 0.005, // 0.5%
            high_vix_multiplier: 1.5,
            extreme_vix_multiplier: 2.5,
            high_vix_threshold: 25.0,
            extreme_vix_threshold: 40.0,
        }
    }
}

/// Calculate realistic slippage based on market conditions
pub struct SlippageCalculator {
    config: SlippageConfig,
}

impl SlippageCalculator {
    pub fn new(config: SlippageConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(SlippageConfig::default())
    }

    /// Calculate slippage percentage based on market conditions
    ///
    /// # Arguments
    /// * `vix` - Current VIX level (optional)
    /// * `is_market_open` - Whether market is in regular trading hours (9:30-16:00 ET)
    ///
    /// # Returns
    /// Slippage as a decimal (e.g., 0.005 for 0.5%)
    pub fn calculate_slippage(
        &self,
        vix: Option<f64>,
        _is_market_open: bool, // Reserved for future time-of-day adjustments
    ) -> f64 {
        let mut slippage = self.config.base_slippage_pct;

        // Adjust for volatility
        if let Some(vix_level) = vix {
            if vix_level >= self.config.extreme_vix_threshold {
                slippage *= self.config.extreme_vix_multiplier;
                debug!(
                    "Extreme volatility (VIX {:.1}): slippage {:.3}% ({}x base)",
                    vix_level,
                    slippage * 100.0,
                    self.config.extreme_vix_multiplier
                );
            } else if vix_level >= self.config.high_vix_threshold {
                slippage *= self.config.high_vix_multiplier;
                debug!(
                    "High volatility (VIX {:.1}): slippage {:.3}% ({}x base)",
                    vix_level,
                    slippage * 100.0,
                    self.config.high_vix_multiplier
                );
            } else {
                debug!(
                    "Normal volatility (VIX {:.1}): slippage {:.3}% (base)",
                    vix_level,
                    slippage * 100.0
                );
            }
        } else {
            debug!(
                "No VIX data: using base slippage {:.3}%",
                slippage * 100.0
            );
        }

        // Future: could add time-of-day adjustments
        // - Wider spreads at open/close (9:30-9:45, 15:45-16:00)
        // - Tighter spreads mid-day (10:00-15:00)

        slippage
    }

    /// Calculate total round-trip slippage (entry + exit)
    pub fn calculate_total_slippage(
        &self,
        vix: Option<f64>,
        is_market_open: bool,
    ) -> f64 {
        // Entry and exit both experience slippage
        let one_way = self.calculate_slippage(vix, is_market_open);
        one_way * 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_slippage() {
        let calc = SlippageCalculator::with_defaults();
        let slippage = calc.calculate_slippage(None, true);
        assert_eq!(slippage, 0.005); // 0.5% base
    }

    #[test]
    fn test_normal_vix_slippage() {
        let calc = SlippageCalculator::with_defaults();
        let slippage = calc.calculate_slippage(Some(18.0), true);
        assert_eq!(slippage, 0.005); // Still base at VIX 18
    }

    #[test]
    fn test_high_vix_slippage() {
        let calc = SlippageCalculator::with_defaults();
        let slippage = calc.calculate_slippage(Some(30.0), true);
        assert_eq!(slippage, 0.005 * 1.5); // 0.75% at VIX 30
    }

    #[test]
    fn test_extreme_vix_slippage() {
        let calc = SlippageCalculator::with_defaults();
        let slippage = calc.calculate_slippage(Some(50.0), true);
        assert_eq!(slippage, 0.005 * 2.5); // 1.25% at VIX 50
    }

    #[test]
    fn test_total_slippage() {
        let calc = SlippageCalculator::with_defaults();
        let total = calc.calculate_total_slippage(Some(20.0), true);
        assert_eq!(total, 0.01); // 1% total (0.5% entry + 0.5% exit)
    }

    #[test]
    fn test_custom_config() {
        let config = SlippageConfig {
            base_slippage_pct: 0.003, // 0.3% for very liquid
            high_vix_multiplier: 2.0,
            extreme_vix_multiplier: 3.0,
            high_vix_threshold: 30.0,
            extreme_vix_threshold: 50.0,
        };
        let calc = SlippageCalculator::new(config);

        assert_eq!(calc.calculate_slippage(Some(20.0), true), 0.003);
        assert_eq!(calc.calculate_slippage(Some(35.0), true), 0.006);
        assert_eq!(calc.calculate_slippage(Some(55.0), true), 0.009);
    }
}

//! Market Regime Detection
//!
//! Identifies market regime (trending vs ranging vs volatile) to inform
//! strategy selection and confidence adjustments.

use serde::{Deserialize, Serialize};

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Strong directional momentum (ADX > 30)
    Trending,
    /// Choppy, no clear direction (ADX < 20)
    Ranging,
    /// Transitional or uncertain (ADX 20-30)
    Transitional,
}

impl MarketRegime {
    /// Detect regime from ADX value
    ///
    /// # Arguments
    /// * `adx` - Average Directional Index value (0-100)
    ///
    /// # Returns
    /// Market regime classification
    pub fn from_adx(adx: f64) -> Self {
        if adx > 30.0 {
            MarketRegime::Trending
        } else if adx < 20.0 {
            MarketRegime::Ranging
        } else {
            MarketRegime::Transitional
        }
    }

    /// Get suggested confidence adjustment for this regime
    ///
    /// Returns suggested adjustment range as (min, max) in percentage points
    pub fn confidence_adjustment_range(&self) -> (f64, f64) {
        match self {
            MarketRegime::Trending => {
                // Trending: momentum strategies favored
                // Slight boost for directional plays, penalty for mean-reversion
                (-0.05, 0.05)
            }
            MarketRegime::Ranging => {
                // Ranging: mean-reversion strategies favored
                // Boost for oscillator-based entries, penalty for breakouts
                (-0.05, 0.05)
            }
            MarketRegime::Transitional => {
                // Uncertain regime: be more selective
                (-0.10, 0.00)
            }
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            MarketRegime::Trending => {
                "Strong directional momentum - directional strategies favored, momentum signals reliable"
            }
            MarketRegime::Ranging => {
                "Choppy range-bound - mean-reversion strategies favored, oscillators more reliable"
            }
            MarketRegime::Transitional => {
                "Transitional/uncertain - use caution, wait for clearer regime or reduce position size"
            }
        }
    }
}

/// Volatility environment classification
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VolatilityRegime {
    /// VIX < 15 - Low volatility, compressed ranges
    Low,
    /// VIX 15-25 - Normal volatility
    Normal,
    /// VIX > 25 - Elevated volatility
    High,
}

impl VolatilityRegime {
    /// Detect volatility regime from VIX
    pub fn from_vix(vix: f64) -> Self {
        if vix < 15.0 {
            VolatilityRegime::Low
        } else if vix <= 25.0 {
            VolatilityRegime::Normal
        } else {
            VolatilityRegime::High
        }
    }

    /// Get suggested confidence adjustment for premium sellers
    ///
    /// Low VIX = lower premiums = harder to find edge
    /// High VIX = higher premiums BUT also higher risk
    pub fn premium_seller_adjustment(&self) -> f64 {
        match self {
            VolatilityRegime::Low => -0.05,  // Harder to find attractive premiums
            VolatilityRegime::Normal => 0.0,  // Goldilocks zone
            VolatilityRegime::High => 0.0,    // Higher premiums offset by higher risk
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            VolatilityRegime::Low => "Low volatility - compressed premiums, harder to find edge",
            VolatilityRegime::Normal => "Normal volatility - balanced risk/reward",
            VolatilityRegime::High => "High volatility - elevated premiums but increased risk",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adx_regime_detection() {
        assert_eq!(MarketRegime::from_adx(35.0), MarketRegime::Trending);
        assert_eq!(MarketRegime::from_adx(15.0), MarketRegime::Ranging);
        assert_eq!(MarketRegime::from_adx(25.0), MarketRegime::Transitional);
    }

    #[test]
    fn test_vix_regime_detection() {
        assert_eq!(VolatilityRegime::from_vix(12.0), VolatilityRegime::Low);
        assert_eq!(VolatilityRegime::from_vix(20.0), VolatilityRegime::Normal);
        assert_eq!(VolatilityRegime::from_vix(30.0), VolatilityRegime::High);
    }

    #[test]
    fn test_regime_adjustments() {
        let trending = MarketRegime::Trending;
        let (min, max) = trending.confidence_adjustment_range();
        assert!(min >= -0.10 && min <= 0.0);
        assert!(max >= 0.0 && max <= 0.10);
    }
}

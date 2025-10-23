// Position Sizing using Kelly Criterion
// Implements fractional Kelly for conservative position sizing

use anyhow::{bail, Result};

/// Position sizer with Kelly Criterion
#[derive(Debug, Clone)]
pub struct PositionSizer {
    /// Maximum position size as percentage of account (e.g., 0.05 = 5%)
    pub max_position_size_pct: f64,

    /// Kelly fraction to use (e.g., 0.25 = quarter Kelly for conservative sizing)
    pub kelly_fraction: f64,

    /// Minimum confidence threshold to trade (e.g., 0.5 = 50%)
    pub min_confidence: f64,
}

impl Default for PositionSizer {
    fn default() -> Self {
        Self {
            max_position_size_pct: 0.05, // 5% max
            kelly_fraction: 0.25,         // Quarter Kelly (conservative)
            min_confidence: 0.50,         // 50% minimum confidence
        }
    }
}

impl PositionSizer {
    pub fn new(max_position_size_pct: f64, kelly_fraction: f64, min_confidence: f64) -> Self {
        Self {
            max_position_size_pct,
            kelly_fraction,
            min_confidence,
        }
    }

    /// Calculate position size using Kelly Criterion
    ///
    /// Formula: f = (p*W - (1-p)*L) / W
    /// where:
    /// - f = fraction of bankroll to bet
    /// - p = probability of winning (win_rate)
    /// - W = average win amount
    /// - L = average loss amount
    ///
    /// We then apply:
    /// - Kelly fraction (typically 0.25 for quarter Kelly)
    /// - Confidence scaling from ACE
    /// - Hard cap at max_position_size_pct
    pub fn calculate_position_size(
        &self,
        account_balance: f64,
        win_rate: f64,
        avg_win: f64,
        avg_loss: f64,
        confidence: f64,
    ) -> Result<f64> {
        // Validate inputs
        if account_balance <= 0.0 {
            bail!("Account balance must be positive");
        }
        if win_rate < 0.0 || win_rate > 1.0 {
            bail!("Win rate must be between 0 and 1");
        }
        if avg_win <= 0.0 {
            bail!("Average win must be positive");
        }
        if avg_loss <= 0.0 {
            bail!("Average loss must be positive");
        }
        if confidence < 0.0 || confidence > 1.0 {
            bail!("Confidence must be between 0 and 1");
        }

        // Don't trade if confidence is below threshold
        if confidence < self.min_confidence {
            return Ok(0.0);
        }

        // Calculate Kelly fraction
        let p = win_rate;
        let w = avg_win;
        let l = avg_loss;

        let kelly = (p * w - (1.0 - p) * l) / w;

        // If Kelly is negative, don't trade
        if kelly <= 0.0 {
            return Ok(0.0);
        }

        // Apply fractional Kelly and confidence scaling
        let mut position_size = account_balance * kelly * self.kelly_fraction * confidence;

        // Cap at max position size
        let max_size = account_balance * self.max_position_size_pct;
        position_size = position_size.min(max_size);

        Ok(position_size)
    }

    /// Calculate position size with default historical statistics
    ///
    /// Uses conservative defaults if no historical data is available:
    /// - Win rate: 55%
    /// - Avg win: $100
    /// - Avg loss: $60
    pub fn calculate_position_size_simple(
        &self,
        account_balance: f64,
        confidence: f64,
    ) -> Result<f64> {
        self.calculate_position_size(
            account_balance,
            0.55, // 55% win rate (conservative default)
            100.0, // $100 avg win
            60.0,  // $60 avg loss
            confidence,
        )
    }

    /// Calculate number of contracts/shares for a given position size
    pub fn calculate_shares(
        &self,
        position_size_usd: f64,
        price_per_share: f64,
    ) -> Result<f64> {
        if price_per_share <= 0.0 {
            bail!("Price per share must be positive");
        }

        let shares = (position_size_usd / price_per_share).floor();
        Ok(shares.max(0.0))
    }

    /// Validate that a position size is acceptable
    pub fn validate_position_size(
        &self,
        position_size_usd: f64,
        account_balance: f64,
    ) -> Result<()> {
        if position_size_usd < 0.0 {
            bail!("Position size cannot be negative");
        }

        let position_pct = position_size_usd / account_balance;

        if position_pct > self.max_position_size_pct {
            bail!(
                "Position size {}% exceeds maximum {}%",
                position_pct * 100.0,
                self.max_position_size_pct * 100.0
            );
        }

        Ok(())
    }
}

/// Calculate optimal Kelly position size (utility function)
pub fn kelly_criterion(
    win_rate: f64,
    avg_win: f64,
    avg_loss: f64,
) -> f64 {
    let p = win_rate;
    let w = avg_win;
    let l = avg_loss;

    (p * w - (1.0 - p) * l) / w
}

/// Calculate fractional Kelly (utility function)
pub fn fractional_kelly(
    win_rate: f64,
    avg_win: f64,
    avg_loss: f64,
    fraction: f64,
) -> f64 {
    kelly_criterion(win_rate, avg_win, avg_loss) * fraction
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kelly_criterion() {
        // 60% win rate, avg win $100, avg loss $60
        let kelly = kelly_criterion(0.6, 100.0, 60.0);

        // Expected: (0.6 * 100 - 0.4 * 60) / 100 = (60 - 24) / 100 = 0.36
        assert!((kelly - 0.36).abs() < 0.001);
    }

    #[test]
    fn test_fractional_kelly() {
        let frac_kelly = fractional_kelly(0.6, 100.0, 60.0, 0.25);

        // Expected: 0.36 * 0.25 = 0.09
        assert!((frac_kelly - 0.09).abs() < 0.001);
    }

    #[test]
    fn test_position_sizing() {
        let sizer = PositionSizer::default();
        let account = 10000.0;
        let confidence = 0.75;

        let position = sizer
            .calculate_position_size(account, 0.6, 100.0, 60.0, confidence)
            .unwrap();

        // Expected: 10000 * 0.36 * 0.25 * 0.75 = 675
        // But capped at 5% = $500
        assert!((position - 500.0).abs() < 0.01);
    }

    #[test]
    fn test_position_sizing_low_confidence() {
        let sizer = PositionSizer::default();
        let account = 10000.0;
        let confidence = 0.40; // Below 50% threshold

        let position = sizer
            .calculate_position_size(account, 0.6, 100.0, 60.0, confidence)
            .unwrap();

        // Should return 0 because confidence is below threshold
        assert_eq!(position, 0.0);
    }

    #[test]
    fn test_position_sizing_negative_kelly() {
        let sizer = PositionSizer::default();
        let account = 10000.0;
        let confidence = 0.75;

        // Bad trade: 40% win rate, avg loss > avg win
        let position = sizer
            .calculate_position_size(account, 0.4, 50.0, 100.0, confidence)
            .unwrap();

        // Should return 0 because Kelly is negative
        assert_eq!(position, 0.0);
    }

    #[test]
    fn test_calculate_shares() {
        let sizer = PositionSizer::default();

        let shares = sizer.calculate_shares(500.0, 2.30).unwrap();

        // 500 / 2.30 = 217.39... floored = 217
        assert_eq!(shares, 217.0);
    }

    #[test]
    fn test_validate_position_size() {
        let sizer = PositionSizer::default();
        let account = 10000.0;

        // Valid position (4%)
        assert!(sizer.validate_position_size(400.0, account).is_ok());

        // Invalid position (6% > 5% max)
        assert!(sizer.validate_position_size(600.0, account).is_err());
    }

    #[test]
    fn test_simple_position_sizing() {
        let sizer = PositionSizer::default();
        let account = 10000.0;
        let confidence = 0.75;

        let position = sizer
            .calculate_position_size_simple(account, confidence)
            .unwrap();

        // Should calculate with defaults and cap at 5%
        assert!(position > 0.0);
        assert!(position <= 500.0); // 5% of 10000
    }
}

//! Confidence Calibration using Platt Scaling
//!
//! Ensures predicted confidence matches actual win rates through walk-forward calibration.
//! Based on Platt (1999) and modern ML calibration techniques.

use serde::{Deserialize, Serialize};

/// Calibration bin for tracking predictions vs outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationBin {
    pub min_confidence: f64,
    pub max_confidence: f64,
    pub predicted_count: i32,
    pub actual_wins: i32,
}

impl CalibrationBin {
    pub fn new(min_conf: f64, max_conf: f64) -> Self {
        Self {
            min_confidence: min_conf,
            max_confidence: max_conf,
            predicted_count: 0,
            actual_wins: 0,
        }
    }

    /// Check if this confidence value falls in this bin
    pub fn contains(&self, confidence: f64) -> bool {
        confidence >= self.min_confidence && confidence < self.max_confidence
    }

    /// Record a prediction in this bin
    pub fn record_prediction(&mut self) {
        self.predicted_count += 1;
    }

    /// Record an actual win in this bin
    pub fn record_win(&mut self) {
        self.actual_wins += 1;
    }

    /// Get actual win rate for this bin
    pub fn actual_win_rate(&self) -> Option<f64> {
        if self.predicted_count > 0 {
            Some(self.actual_wins as f64 / self.predicted_count as f64)
        } else {
            None
        }
    }

    /// Get midpoint confidence for this bin
    pub fn midpoint_confidence(&self) -> f64 {
        (self.min_confidence + self.max_confidence) / 2.0
    }

    /// Calculate calibration error (predicted - actual)
    pub fn calibration_error(&self) -> Option<f64> {
        self.actual_win_rate()
            .map(|actual| self.midpoint_confidence() - actual)
    }
}

/// Platt scaling calibrator
///
/// Maps raw confidence scores to calibrated probabilities using logistic regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlattScaler {
    /// Logistic regression parameters: P(win) = 1 / (1 + exp(A*conf + B))
    pub a: f64,
    pub b: f64,

    /// Calibration bins for tracking performance
    bins: Vec<CalibrationBin>,

    /// Minimum samples required before applying calibration
    min_samples: i32,
}

impl PlattScaler {
    /// Create new Platt scaler with default parameters (identity mapping initially)
    pub fn new() -> Self {
        Self {
            a: 0.0,  // Identity: P(win) = conf
            b: 0.0,
            bins: Self::create_bins(),
            min_samples: 30, // Require 30 samples before calibrating
        }
    }

    /// Create calibration bins (0-10%, 10-20%, ..., 90-100%)
    fn create_bins() -> Vec<CalibrationBin> {
        (0..10)
            .map(|i| {
                let min = i as f64 / 10.0;
                let max = (i + 1) as f64 / 10.0;
                CalibrationBin::new(min, max)
            })
            .collect()
    }

    /// Record a prediction and its eventual outcome
    pub fn record(&mut self, confidence: f64, won: bool) {
        for bin in &mut self.bins {
            if bin.contains(confidence) {
                bin.record_prediction();
                if won {
                    bin.record_win();
                }
                break;
            }
        }
    }

    /// Calibrate confidence using current Platt parameters
    ///
    /// Maps raw confidence to calibrated probability
    pub fn calibrate(&self, raw_confidence: f64) -> f64 {
        if self.total_samples() < self.min_samples {
            // Not enough data yet, return raw confidence
            return raw_confidence;
        }

        // Platt scaling: P(win) = 1 / (1 + exp(A*conf + B))
        let z = self.a * raw_confidence + self.b;
        1.0 / (1.0 + (-z).exp())
    }

    /// Fit Platt parameters using current calibration data
    ///
    /// Uses simplified maximum likelihood estimation
    pub fn fit(&mut self) {
        let total = self.total_samples();
        if total < self.min_samples {
            // Not enough data, keep identity mapping
            return;
        }

        // Collect bin statistics
        let mut sum_pred = 0.0;
        let mut sum_actual = 0.0;
        let mut valid_bins = 0;

        for bin in &self.bins {
            if let (Some(actual_rate), count) = (bin.actual_win_rate(), bin.predicted_count) {
                if count >= 3 {
                    // Only use bins with at least 3 samples
                    sum_pred += bin.midpoint_confidence() * count as f64;
                    sum_actual += actual_rate * count as f64;
                    valid_bins += count;
                }
            }
        }

        if valid_bins < 10 {
            // Not enough valid bins
            return;
        }

        // Calculate average predicted and actual rates
        let avg_pred = sum_pred / valid_bins as f64;
        let avg_actual = sum_actual / valid_bins as f64;

        // Simple Platt scaling estimation
        // If predicted > actual: we're overconfident, need to shift down
        // If predicted < actual: we're underconfident, need to shift up
        let shift = avg_actual - avg_pred;

        // Update parameters (simplified approach)
        // B controls the shift: positive B increases calibrated probability
        self.b = shift * 2.0; // Scale factor for sensitivity

        // A controls the slope: if we're consistently off, adjust slope
        let variance = self.calculate_variance();
        if variance > 0.1 {
            // High variance suggests poor calibration, flatten the curve
            self.a = -0.5;
        } else {
            self.a = 0.0; // Good calibration, keep linear
        }
    }

    /// Calculate variance of calibration errors
    fn calculate_variance(&self) -> f64 {
        let errors: Vec<f64> = self
            .bins
            .iter()
            .filter_map(|bin| bin.calibration_error())
            .collect();

        if errors.is_empty() {
            return 0.0;
        }

        let mean = errors.iter().sum::<f64>() / errors.len() as f64;
        let variance = errors.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / errors.len() as f64;

        variance
    }

    /// Get total number of samples across all bins
    pub fn total_samples(&self) -> i32 {
        self.bins.iter().map(|b| b.predicted_count).sum()
    }

    /// Calculate Brier score (lower is better, 0 = perfect)
    ///
    /// Brier score = mean((predicted - actual)^2)
    pub fn brier_score(&self) -> Option<f64> {
        let mut squared_errors = Vec::new();

        for bin in &self.bins {
            if let Some(actual_rate) = bin.actual_win_rate() {
                let predicted = bin.midpoint_confidence();
                squared_errors.push((predicted - actual_rate).powi(2));
            }
        }

        if squared_errors.is_empty() {
            None
        } else {
            Some(squared_errors.iter().sum::<f64>() / squared_errors.len() as f64)
        }
    }

    /// Get calibration report showing bins and their accuracy
    pub fn calibration_report(&self) -> String {
        let mut report = String::from("Confidence Calibration Report\n");
        report.push_str("================================\n\n");
        report.push_str(&format!(
            "Total samples: {}\n",
            self.total_samples()
        ));
        report.push_str(&format!(
            "Platt parameters: A={:.4}, B={:.4}\n\n",
            self.a, self.b
        ));

        if let Some(brier) = self.brier_score() {
            report.push_str(&format!("Brier Score: {:.4}\n\n", brier));
        }

        report.push_str("Bin Analysis:\n");
        report.push_str("Confidence Range | Predictions | Actual Win Rate | Error\n");
        report.push_str("-----------------|-------------|-----------------|-------\n");

        for bin in &self.bins {
            if bin.predicted_count > 0 {
                let actual = bin
                    .actual_win_rate()
                    .map(|r| format!("{:.1}%", r * 100.0))
                    .unwrap_or("N/A".to_string());

                let error = bin
                    .calibration_error()
                    .map(|e| format!("{:+.1}%", e * 100.0))
                    .unwrap_or("N/A".to_string());

                report.push_str(&format!(
                    "{:.0}-{:.0}%        | {:>11} | {:>15} | {:>5}\n",
                    bin.min_confidence * 100.0,
                    bin.max_confidence * 100.0,
                    bin.predicted_count,
                    actual,
                    error
                ));
            }
        }

        report
    }
}

impl Default for PlattScaler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_bin() {
        let mut bin = CalibrationBin::new(0.5, 0.6);

        assert!(bin.contains(0.55));
        assert!(!bin.contains(0.45));
        assert!(!bin.contains(0.65));

        bin.record_prediction();
        bin.record_prediction();
        bin.record_win();

        assert_eq!(bin.predicted_count, 2);
        assert_eq!(bin.actual_wins, 1);
        assert_eq!(bin.actual_win_rate(), Some(0.5));
    }

    #[test]
    fn test_platt_scaler_identity() {
        let scaler = PlattScaler::new();

        // With no data, should return identity mapping
        assert_eq!(scaler.calibrate(0.6), 0.6);
        assert_eq!(scaler.calibrate(0.8), 0.8);
    }

    #[test]
    fn test_platt_scaler_recording() {
        let mut scaler = PlattScaler::new();

        // Record some predictions
        scaler.record(0.65, true); // Win
        scaler.record(0.68, false); // Loss
        scaler.record(0.62, true); // Win

        assert_eq!(scaler.total_samples(), 3);
    }

    #[test]
    fn test_platt_scaler_calibration() {
        let mut scaler = PlattScaler::new();

        // Simulate overconfident predictions (70% confidence but only 50% win rate)
        for _ in 0..15 {
            scaler.record(0.70, true);
        }
        for _ in 0..15 {
            scaler.record(0.70, false);
        }

        scaler.fit();

        // After fitting, 70% confidence should be calibrated down closer to 50%
        let calibrated = scaler.calibrate(0.70);
        assert!(
            calibrated < 0.70,
            "Overconfident predictions should be calibrated down"
        );
    }

    #[test]
    fn test_brier_score_perfect_calibration() {
        let mut scaler = PlattScaler::new();

        // Perfect calibration: 60% confidence with 60% win rate
        for _ in 0..6 {
            scaler.record(0.60, true);
        }
        for _ in 0..4 {
            scaler.record(0.60, false);
        }

        let brier = scaler.brier_score().unwrap();
        assert!(
            brier < 0.01,
            "Perfect calibration should have Brier score near 0"
        );
    }

    #[test]
    fn test_calibration_report() {
        let mut scaler = PlattScaler::new();

        for i in 0..40 {
            let won = i % 2 == 0; // 50% win rate
            scaler.record(0.65, won);
        }

        scaler.fit();
        let report = scaler.calibration_report();

        assert!(report.contains("Total samples: 40"));
        assert!(report.contains("Brier Score"));
        assert!(report.contains("60-70%"));
    }
}

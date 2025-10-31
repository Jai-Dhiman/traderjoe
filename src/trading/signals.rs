//! Signal Normalization and Combination
//!
//! Professional quant trading requires proper signal normalization before combination.
//! This module provides z-score normalization, Winsorization, and weighted combination
//! based on Goldman Sachs research and industry best practices.

use std::collections::HashMap;

/// Signal statistics for normalization
#[derive(Debug, Clone)]
pub struct SignalStats {
    pub mean: f64,
    pub std_dev: f64,
    pub median: f64,
    pub mad: f64, // Median Absolute Deviation
}

impl SignalStats {
    /// Calculate statistics from a series of values
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                mean: 0.0,
                std_dev: 1.0,
                median: 0.0,
                mad: 1.0,
            };
        }

        // Calculate mean
        let mean = values.iter().sum::<f64>() / values.len() as f64;

        // Calculate standard deviation
        let variance = values
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;
        let std_dev = variance.sqrt().max(1e-8); // Prevent division by zero

        // Calculate median
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        // Calculate MAD (Median Absolute Deviation)
        let absolute_deviations: Vec<f64> = values.iter().map(|v| (v - median).abs()).collect();
        let mut sorted_deviations = absolute_deviations.clone();
        sorted_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = if sorted_deviations.len() % 2 == 0 {
            (sorted_deviations[sorted_deviations.len() / 2 - 1]
                + sorted_deviations[sorted_deviations.len() / 2])
                / 2.0
        } else {
            sorted_deviations[sorted_deviations.len() / 2]
        }
        .max(1e-8); // Prevent division by zero

        Self {
            mean,
            std_dev,
            median,
            mad,
        }
    }

    /// Normalize value to z-score
    pub fn normalize(&self, value: f64) -> f64 {
        (value - self.mean) / self.std_dev
    }

    /// Winsorize value at ±N MAD from median
    ///
    /// Goldman Sachs research shows Winsorization at ±3 MAD improves Sharpe by 0.06
    pub fn winsorize(&self, value: f64, n_mad: f64) -> f64 {
        let lower_bound = self.median - n_mad * self.mad;
        let upper_bound = self.median + n_mad * self.mad;
        value.clamp(lower_bound, upper_bound)
    }

    /// Normalize and winsorize in one step
    pub fn normalize_and_winsorize(&self, value: f64, n_mad: f64) -> f64 {
        let winsorized = self.winsorize(value, n_mad);
        self.normalize(winsorized)
    }
}

/// Weighted signal combination using risk parity
#[derive(Debug, Clone)]
pub struct SignalCombiner {
    /// Signal names and their weights
    weights: HashMap<String, f64>,
}

impl SignalCombiner {
    /// Create new combiner with equal weights
    pub fn new_equal_weighted(signal_names: Vec<String>) -> Self {
        let n = signal_names.len() as f64;
        let weight = 1.0 / n;

        let weights = signal_names
            .into_iter()
            .map(|name| (name, weight))
            .collect();

        Self { weights }
    }

    /// Create new combiner with risk parity (inverse volatility weighting)
    ///
    /// Allocates weight inversely proportional to signal volatility:
    /// w_i = (1/σ_i) / Σ(1/σ_j)
    pub fn new_risk_parity(signal_stats: Vec<(&str, &SignalStats)>) -> Self {
        let mut inverse_vols = Vec::new();
        let mut names = Vec::new();

        for (name, stats) in signal_stats.iter() {
            let inverse_vol = 1.0 / stats.std_dev;
            inverse_vols.push(inverse_vol);
            names.push(name.to_string());
        }

        let sum_inverse_vol: f64 = inverse_vols.iter().sum();

        let weights = names
            .into_iter()
            .zip(inverse_vols.iter())
            .map(|(name, &inv_vol)| (name, inv_vol / sum_inverse_vol))
            .collect();

        Self { weights }
    }

    /// Create combiner with custom weights
    pub fn new_custom(weights: HashMap<String, f64>) -> Self {
        // Normalize weights to sum to 1.0
        let sum: f64 = weights.values().sum();
        let normalized_weights = weights
            .into_iter()
            .map(|(k, v)| (k, v / sum))
            .collect();

        Self {
            weights: normalized_weights,
        }
    }

    /// Combine normalized signals into single score
    pub fn combine(&self, signals: &HashMap<String, f64>) -> f64 {
        let mut combined = 0.0;

        for (name, &weight) in &self.weights {
            if let Some(&value) = signals.get(name) {
                combined += weight * value;
            }
        }

        combined
    }

    /// Get weight for a signal
    pub fn get_weight(&self, signal_name: &str) -> Option<f64> {
        self.weights.get(signal_name).copied()
    }

    /// Regime-adaptive reweighting
    ///
    /// Adjusts weights based on market regime:
    /// - Trending: Boost momentum signals (MACD, MA), reduce mean-reversion (RSI, Stochastic)
    /// - Ranging: Boost oscillators (RSI, Stochastic), reduce momentum (MACD)
    pub fn adjust_for_regime(&mut self, regime: crate::trading::MarketRegime) {
        use crate::trading::MarketRegime;

        match regime {
            MarketRegime::Trending => {
                // Boost momentum signals
                if let Some(w) = self.weights.get_mut("macd") {
                    *w *= 1.3;
                }
                if let Some(w) = self.weights.get_mut("moving_average") {
                    *w *= 1.3;
                }
                // Reduce mean-reversion signals
                if let Some(w) = self.weights.get_mut("rsi") {
                    *w *= 0.7;
                }
                if let Some(w) = self.weights.get_mut("stochastic") {
                    *w *= 0.7;
                }
            }
            MarketRegime::Ranging => {
                // Boost oscillators
                if let Some(w) = self.weights.get_mut("rsi") {
                    *w *= 1.3;
                }
                if let Some(w) = self.weights.get_mut("stochastic") {
                    *w *= 1.3;
                }
                // Reduce momentum signals
                if let Some(w) = self.weights.get_mut("macd") {
                    *w *= 0.7;
                }
                if let Some(w) = self.weights.get_mut("moving_average") {
                    *w *= 0.7;
                }
            }
            MarketRegime::Transitional => {
                // Keep balanced weights in uncertain regime
                // No adjustment needed
            }
        }

        // Renormalize after adjustment
        let sum: f64 = self.weights.values().sum();
        for weight in self.weights.values_mut() {
            *weight /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_stats_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = SignalStats::from_values(&values);

        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert!((stats.median - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_z_score_normalization() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = SignalStats::from_values(&values);

        let z_score = stats.normalize(3.0);
        assert!(z_score.abs() < 1e-6); // Mean should normalize to 0
    }

    #[test]
    fn test_winsorization() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 100.0]; // Outlier
        let stats = SignalStats::from_values(&values);

        let winsorized = stats.winsorize(100.0, 3.0);
        assert!(winsorized < 100.0); // Outlier should be capped
    }

    #[test]
    fn test_equal_weighted_combination() {
        let combiner = SignalCombiner::new_equal_weighted(vec![
            "signal1".to_string(),
            "signal2".to_string(),
        ]);

        let mut signals = HashMap::new();
        signals.insert("signal1".to_string(), 0.5);
        signals.insert("signal2".to_string(), 1.0);

        let combined = combiner.combine(&signals);
        assert!((combined - 0.75).abs() < 1e-6); // (0.5 + 1.0) / 2
    }

    #[test]
    fn test_regime_adaptive_reweighting() {
        use crate::trading::MarketRegime;

        let mut weights = HashMap::new();
        weights.insert("macd".to_string(), 0.5);
        weights.insert("rsi".to_string(), 0.5);

        let mut combiner = SignalCombiner::new_custom(weights);
        combiner.adjust_for_regime(MarketRegime::Trending);

        // MACD should have higher weight in trending regime
        let macd_weight = combiner.get_weight("macd").unwrap();
        let rsi_weight = combiner.get_weight("rsi").unwrap();

        assert!(macd_weight > rsi_weight);
    }
}

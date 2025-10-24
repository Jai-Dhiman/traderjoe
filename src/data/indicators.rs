///! Technical indicators module
///! Implements RSI, MACD, SMA, and other common technical analysis indicators

use serde::{Deserialize, Serialize};
use super::OHLCV;

/// Technical indicator signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalSignals {
    pub rsi_14: Option<f64>,
    pub macd: Option<MACDSignal>,
    pub sma_20: Option<f64>,
    pub sma_50: Option<f64>,
    pub sma_200: Option<f64>,
    pub signal: TrendSignal,
    pub confidence: f64,
}

/// MACD indicator components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MACDSignal {
    pub macd_line: f64,
    pub signal_line: f64,
    pub histogram: f64,
    pub trend: String,
}

/// Trend signal enum
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrendSignal {
    StrongBuy,
    Buy,
    Neutral,
    Sell,
    StrongSell,
}

impl TrendSignal {
    pub fn as_str(&self) -> &'static str {
        match self {
            TrendSignal::StrongBuy => "STRONG_BUY",
            TrendSignal::Buy => "BUY",
            TrendSignal::Neutral => "NEUTRAL",
            TrendSignal::Sell => "SELL",
            TrendSignal::StrongSell => "STRONG_SELL",
        }
    }
}

/// Compute all technical indicators from OHLCV data
pub fn compute_indicators(data: &[OHLCV]) -> TechnicalSignals {
    if data.is_empty() {
        return TechnicalSignals {
            rsi_14: None,
            macd: None,
            sma_20: None,
            sma_50: None,
            sma_200: None,
            signal: TrendSignal::Neutral,
            confidence: 0.0,
        };
    }

    let closes: Vec<f64> = data.iter().map(|d| d.close).collect();

    let rsi_14 = if closes.len() >= 14 {
        Some(calculate_rsi(&closes, 14))
    } else {
        None
    };

    let macd = if closes.len() >= 26 {
        Some(calculate_macd(&closes, 12, 26, 9))
    } else {
        None
    };

    let sma_20 = if closes.len() >= 20 {
        Some(calculate_sma(&closes, 20))
    } else {
        None
    };

    let sma_50 = if closes.len() >= 50 {
        Some(calculate_sma(&closes, 50))
    } else {
        None
    };

    let sma_200 = if closes.len() >= 200 {
        Some(calculate_sma(&closes, 200))
    } else {
        None
    };

    // Determine overall signal
    let (signal, confidence) = determine_signal(rsi_14, &macd, sma_20, sma_50);

    TechnicalSignals {
        rsi_14,
        macd,
        sma_20,
        sma_50,
        sma_200,
        signal,
        confidence,
    }
}

/// Calculate RSI (Relative Strength Index)
pub fn calculate_rsi(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period + 1 {
        return 50.0; // Neutral
    }

    let mut gains = Vec::new();
    let mut losses = Vec::new();

    // Calculate price changes
    for i in 1..prices.len() {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(change.abs());
        }
    }

    if gains.is_empty() {
        return 50.0;
    }

    // Calculate average gain and loss using EMA
    let mut avg_gain = gains.iter().take(period).sum::<f64>() / period as f64;
    let mut avg_loss = losses.iter().take(period).sum::<f64>() / period as f64;

    // Apply smoothing for remaining periods
    for i in period..gains.len() {
        avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
    }

    if avg_loss == 0.0 {
        return 100.0;
    }

    let rs = avg_gain / avg_loss;
    100.0 - (100.0 / (1.0 + rs))
}

/// Calculate MACD (Moving Average Convergence Divergence)
/// Properly implements MACD with signal line calculation
pub fn calculate_macd(prices: &[f64], fast_period: usize, slow_period: usize, signal_period: usize) -> MACDSignal {
    if prices.len() < slow_period {
        return MACDSignal {
            macd_line: 0.0,
            signal_line: 0.0,
            histogram: 0.0,
            trend: "neutral".to_string(),
        };
    }

    // Calculate MACD values for all data points
    let mut macd_values = Vec::new();

    for i in slow_period..=prices.len() {
        let price_slice = &prices[..i];
        let ema_fast = calculate_ema(price_slice, fast_period);
        let ema_slow = calculate_ema(price_slice, slow_period);
        macd_values.push(ema_fast - ema_slow);
    }

    // Get the most recent MACD value
    let macd_line = *macd_values.last().unwrap_or(&0.0);

    // Calculate signal line (9-period EMA of MACD values)
    let signal_line = if macd_values.len() >= signal_period {
        calculate_ema(&macd_values, signal_period)
    } else {
        // If not enough MACD values, use simple average
        macd_values.iter().sum::<f64>() / macd_values.len() as f64
    };

    let histogram = macd_line - signal_line;

    // Determine trend based on MACD line position relative to signal line
    let trend = if macd_line > signal_line && histogram > 0.0 {
        "bullish"
    } else if macd_line < signal_line && histogram < 0.0 {
        "bearish"
    } else {
        "neutral"
    };

    MACDSignal {
        macd_line,
        signal_line,
        histogram,
        trend: trend.to_string(),
    }
}

/// Calculate SMA (Simple Moving Average)
pub fn calculate_sma(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period {
        return prices.iter().sum::<f64>() / prices.len() as f64;
    }

    let recent_prices = &prices[prices.len() - period..];
    recent_prices.iter().sum::<f64>() / period as f64
}

/// Calculate EMA (Exponential Moving Average)
pub fn calculate_ema(prices: &[f64], period: usize) -> f64 {
    if prices.is_empty() {
        return 0.0;
    }

    if prices.len() < period {
        return calculate_sma(prices, prices.len());
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema = calculate_sma(&prices[0..period], period);

    for &price in &prices[period..] {
        ema = (price - ema) * multiplier + ema;
    }

    ema
}

/// Determine overall trading signal from indicators
/// Production-ready implementation considering trend context
fn determine_signal(
    rsi: Option<f64>,
    macd: &Option<MACDSignal>,
    sma_20: Option<f64>,
    sma_50: Option<f64>,
) -> (TrendSignal, f64) {
    let mut bullish_signals = 0;
    let mut bearish_signals = 0;
    let mut total_signals = 0;

    // First, determine the primary trend from SMA crossover (most reliable for trend direction)
    let primary_trend = if let (Some(sma20), Some(sma50)) = (sma_20, sma_50) {
        if sma20 > sma50 {
            "bullish"
        } else if sma20 < sma50 {
            "bearish"
        } else {
            "neutral"
        }
    } else {
        "neutral"
    };

    // SMA crossover signal - weighted heavily as it defines the trend
    if let (Some(sma20), Some(sma50)) = (sma_20, sma_50) {
        total_signals += 1;
        let crossover_strength = ((sma20 - sma50) / sma50 * 100.0).abs();

        if sma20 > sma50 {
            bullish_signals += 2; // Strong weight for trend direction
            // Additional signal if crossover is significant (>1% separation)
            if crossover_strength > 1.0 {
                bullish_signals += 1;
            }
        } else if sma20 < sma50 {
            bearish_signals += 2;
            if crossover_strength > 1.0 {
                bearish_signals += 1;
            }
        }
    }

    // MACD signal - confirms trend momentum
    if let Some(macd_sig) = macd {
        total_signals += 1;

        if macd_sig.histogram > 0.0 {
            bullish_signals += 2; // MACD above signal = bullish momentum
        } else if macd_sig.histogram < 0.0 {
            bearish_signals += 2; // MACD below signal = bearish momentum
        }

        // Additional weight if MACD aligns with position (strong trend)
        if macd_sig.macd_line > 0.0 && macd_sig.histogram > 0.0 {
            bullish_signals += 1; // Strong bullish: positive MACD above signal
        } else if macd_sig.macd_line < 0.0 && macd_sig.histogram < 0.0 {
            bearish_signals += 1; // Strong bearish: negative MACD below signal
        }
    }

    // RSI signal - context-aware interpretation based on primary trend
    if let Some(rsi_val) = rsi {
        total_signals += 1;

        match primary_trend {
            "bullish" => {
                // In uptrend, only extreme oversold matters (buying opportunity)
                // High RSI is normal in strong uptrends
                if rsi_val < 30.0 {
                    bullish_signals += 2; // Pullback in uptrend = buy opportunity
                } else if rsi_val > 50.0 {
                    bullish_signals += 1; // Bullish momentum confirmed
                }
                // Don't penalize overbought RSI in an uptrend (normal behavior)
            },
            "bearish" => {
                // In downtrend, only extreme overbought matters (selling opportunity)
                // Low RSI is normal in strong downtrends
                if rsi_val > 70.0 {
                    bearish_signals += 2; // Rally in downtrend = sell opportunity
                } else if rsi_val < 50.0 {
                    bearish_signals += 1; // Bearish momentum confirmed
                }
                // Don't penalize oversold RSI in a downtrend (normal behavior)
            },
            _ => {
                // No clear trend - use traditional RSI signals
                if rsi_val < 30.0 {
                    bullish_signals += 1;
                } else if rsi_val > 70.0 {
                    bearish_signals += 1;
                }
            }
        }
    }

    if total_signals == 0 {
        return (TrendSignal::Neutral, 0.0);
    }

    let net_signal = bullish_signals as i32 - bearish_signals as i32;
    let max_possible = total_signals * 3; // Each signal can contribute up to 3 points
    let confidence = (net_signal.abs() as f64 / max_possible as f64).min(1.0);

    // Adjusted thresholds for more accurate signal classification
    let signal = match net_signal {
        s if s >= 5 => TrendSignal::StrongBuy,
        s if s >= 2 => TrendSignal::Buy,
        s if s <= -5 => TrendSignal::StrongSell,
        s if s <= -2 => TrendSignal::Sell,
        _ => TrendSignal::Neutral,
    };

    (signal, confidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma_calculation() {
        let prices = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let sma = calculate_sma(&prices, 3);
        assert!((sma - 13.0).abs() < 0.01); // Average of last 3: (12+13+14)/3 = 13
    }

    #[test]
    fn test_rsi_calculation() {
        let prices = vec![
            44.0, 44.25, 44.5, 43.75, 44.0, 44.25, 44.5, 44.75, 45.0,
            45.25, 45.5, 45.75, 46.0, 45.75, 45.5,
        ];
        let rsi = calculate_rsi(&prices, 14);
        assert!(rsi > 50.0 && rsi < 100.0); // Should be bullish
    }

    #[test]
    fn test_signal_determination() {
        let (signal, confidence) = determine_signal(
            Some(65.0),  // Above 50, confirms bullish momentum in uptrend
            &Some(MACDSignal {
                macd_line: 1.0,
                signal_line: 0.5,
                histogram: 0.5,  // MACD above signal = bullish
                trend: "bullish".to_string(),
            }),
            Some(100.0),
            Some(95.0),  // SMA20 > SMA50 = bullish trend (1.8% crossover strength)
        );
        // With production logic: SMA(+3 bullish), MACD(+3 bullish), RSI(+1 bullish) = 7 total
        // Net signal >= 5 = StrongBuy
        assert_eq!(signal, TrendSignal::StrongBuy);
        assert!(confidence > 0.0);
    }
}

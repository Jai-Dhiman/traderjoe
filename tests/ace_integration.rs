///! Integration tests for ACE pipeline
///! Tests the end-to-end flow from data → indicators → ACE context

use traderjoe::data::{compute_indicators, OHLCV, TrendSignal};
use chrono::NaiveDate;

#[test]
fn test_technical_indicators_computation() {
    // Create sample OHLCV data for testing
    let mut data = Vec::new();
    let base_date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();

    // Generate 50 days of sample data with upward trend
    for i in 0..50 {
        data.push(OHLCV {
            symbol: "SPY".to_string(),
            date: base_date + chrono::Duration::days(i),
            open: 400.0 + (i as f64 * 0.5),
            high: 405.0 + (i as f64 * 0.5),
            low: 398.0 + (i as f64 * 0.5),
            close: 402.0 + (i as f64 * 0.5),
            volume: 50_000_000,
            source: "test".to_string(),
        });
    }

    let signals = compute_indicators(&data);

    // Assertions
    assert!(signals.rsi_14.is_some(), "RSI should be computed");
    assert!(signals.sma_20.is_some(), "SMA 20 should be computed");
    assert!(signals.sma_50.is_some(), "SMA 50 should be computed");

    if let Some(rsi) = signals.rsi_14 {
        assert!(rsi >= 0.0 && rsi <= 100.0, "RSI should be between 0 and 100");
    }

    if let Some(sma_20) = signals.sma_20 {
        assert!(sma_20 > 0.0, "SMA 20 should be positive");
    }

    // With upward trend, should show bullish or buy signal
    assert!(
        signals.signal == TrendSignal::Buy || signals.signal == TrendSignal::StrongBuy,
        "Upward trend should produce buy signal, got: {:?}", signals.signal
    );

    println!("✅ Technical indicators test passed");
}

#[test]
fn test_technical_indicators_with_insufficient_data() {
    // Test with insufficient data (< 14 days for RSI)
    let data = vec![
        OHLCV {
            symbol: "SPY".to_string(),
            date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            open: 400.0,
            high: 405.0,
            low: 398.0,
            close: 402.0,
            volume: 50_000_000,
            source: "test".to_string(),
        },
    ];

    let signals = compute_indicators(&data);

    // Should handle gracefully
    assert!(signals.rsi_14.is_none(), "RSI should be None with insufficient data");
    assert!(signals.sma_20.is_none(), "SMA 20 should be None with insufficient data");

    println!("✅ Insufficient data handling test passed");
}

#[test]
fn test_trend_signal_classification() {
    use traderjoe::data::TrendSignal;

    // Test signal enum methods
    assert_eq!(TrendSignal::StrongBuy.as_str(), "STRONG_BUY");
    assert_eq!(TrendSignal::Buy.as_str(), "BUY");
    assert_eq!(TrendSignal::Neutral.as_str(), "NEUTRAL");
    assert_eq!(TrendSignal::Sell.as_str(), "SELL");
    assert_eq!(TrendSignal::StrongSell.as_str(), "STRONG_SELL");

    println!("✅ Trend signal classification test passed");
}

#[test]
fn test_bearish_trend_detection() {
    // Create sample data with downward trend
    let mut data = Vec::new();
    let base_date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();

    for i in 0..50 {
        data.push(OHLCV {
            symbol: "SPY".to_string(),
            date: base_date + chrono::Duration::days(i),
            open: 450.0 - (i as f64 * 0.8),
            high: 452.0 - (i as f64 * 0.8),
            low: 448.0 - (i as f64 * 0.8),
            close: 449.0 - (i as f64 * 0.8),
            volume: 50_000_000,
            source: "test".to_string(),
        });
    }

    let signals = compute_indicators(&data);

    // Downward trend should produce bearish signal
    assert!(
        signals.signal == TrendSignal::Sell || signals.signal == TrendSignal::StrongSell || signals.signal == TrendSignal::Neutral,
        "Downward trend should produce sell or neutral signal, got: {:?}", signals.signal
    );

    println!("✅ Bearish trend detection test passed");
}

// Note: Full ACE pipeline integration tests would require:
// - Database setup (omitted for simplicity)
// - LLM mocking
// - Vector store operations
// These would be added in a separate test suite with test containers

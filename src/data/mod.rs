//! Data pipeline module for fetching market data, news, and sentiment
//! Provides comprehensive error handling and data validation

pub mod errors;
pub mod indicators;
pub mod market;
pub mod news;
pub mod research;
pub mod sentiment;
pub mod retry;

// Re-export commonly used types
pub use errors::{DataError, DataResult};
pub use indicators::{TrendSignal, compute_indicators};
pub use market::{MarketDataClient, OHLCV};
pub use research::ResearchClient;
pub use sentiment::SentimentClient;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Market data point representing OHLCV data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub id: Option<i32>,
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: i64,
    pub source: String,
}

/// News article with sentiment analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsItem {
    pub id: Option<i32>,
    pub published_at: DateTime<Utc>,
    pub source: String,
    pub title: String,
    pub url: Option<String>,
    pub content: Option<String>,
    pub sentiment: Option<f32>, // -1.0 to 1.0
    pub symbols: Vec<String>,
}

/// Sentiment data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentData {
    pub id: Option<i32>,
    pub captured_at: DateTime<Utc>,
    pub source: String,
    pub symbol: Option<String>,
    pub score: f32, // -1.0 to 1.0
    pub meta: serde_json::Value,
}

/// Research query result from Exa API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchResult {
    pub id: Option<i32>,
    pub captured_at: DateTime<Utc>,
    pub query: String,
    pub result: serde_json::Value,
    pub embedding: Option<Vec<f32>>,
}

/// Data source configuration
#[derive(Debug, Clone)]
pub struct DataSourceConfig {
    pub name: String,
    pub enabled: bool,
    pub rate_limit_per_minute: Option<u32>,
    pub timeout_seconds: u64,
}

impl Default for DataSourceConfig {
    fn default() -> Self {
        Self {
            name: "unknown".to_string(),
            enabled: true,
            rate_limit_per_minute: Some(60), // Conservative default
            timeout_seconds: 30,
        }
    }
}

/// Validation helpers
pub mod validation {
    use super::*;
    
    /// Validate a stock symbol (basic US market symbols)
    pub fn validate_symbol(symbol: &str) -> DataResult<()> {
        if symbol.is_empty() {
            return Err(DataError::validation_error("symbol", "Symbol cannot be empty"));
        }
        
        if symbol.len() > 10 {
            return Err(DataError::validation_error("symbol", "Symbol too long (max 10 chars)"));
        }
        
        if !symbol.chars().all(|c| c.is_ascii_alphabetic()) {
            return Err(DataError::validation_error("symbol", "Symbol must contain only letters"));
        }
        
        Ok(())
    }
    
    /// Validate OHLCV data
    pub fn validate_ohlcv_data(data: &MarketData) -> DataResult<()> {
        validate_symbol(&data.symbol)?;
        
        if data.open <= 0.0 {
            return Err(DataError::validation_error("open", "Open price must be positive"));
        }
        
        if data.high <= 0.0 {
            return Err(DataError::validation_error("high", "High price must be positive"));
        }
        
        if data.low <= 0.0 {
            return Err(DataError::validation_error("low", "Low price must be positive"));
        }
        
        if data.close <= 0.0 {
            return Err(DataError::validation_error("close", "Close price must be positive"));
        }
        
        if data.volume < 0 {
            return Err(DataError::validation_error("volume", "Volume cannot be negative"));
        }
        
        // Basic OHLC relationship validation
        if data.high < data.low {
            return Err(DataError::validation_error("high_low", "High price cannot be less than low price"));
        }
        
        if data.high < data.open.max(data.close) {
            return Err(DataError::validation_error("high", "High price should be >= open and close"));
        }
        
        if data.low > data.open.min(data.close) {
            return Err(DataError::validation_error("low", "Low price should be <= open and close"));
        }
        
        Ok(())
    }
    
    /// Validate sentiment score
    pub fn validate_sentiment_score(score: f32) -> DataResult<()> {
        if !(-1.0..=1.0).contains(&score) {
            return Err(DataError::validation_error("sentiment_score", "Sentiment score must be between -1.0 and 1.0"));
        }
        Ok(())
    }
}

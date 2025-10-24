use chrono::{NaiveDate, Utc, DateTime};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::sync::Arc;
use tokio::sync::Mutex;
use super::{DataError, DataResult};

#[derive(Debug, Serialize, Deserialize)]
pub struct OHLCV {
    pub symbol: String,
    pub date: NaiveDate,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: i64,
    pub source: String,
}

/// Polygon.io API response structures
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct PolygonAggregatesResponse {
    ticker: Option<String>,
    #[serde(rename = "queryCount")]
    query_count: Option<i64>,
    #[serde(rename = "resultsCount")]
    results_count: Option<i64>,
    adjusted: Option<bool>,
    results: Option<Vec<PolygonAggregate>>,
    status: String,
    #[serde(rename = "next_url")]
    next_url: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct PolygonAggregate {
    #[serde(rename = "v")]
    volume: f64,  // Polygon sometimes returns volume as float
    #[serde(rename = "vw")]
    vwap: Option<f64>,
    #[serde(rename = "o")]
    open: f64,
    #[serde(rename = "c")]
    close: f64,
    #[serde(rename = "h")]
    high: f64,
    #[serde(rename = "l")]
    low: f64,
    #[serde(rename = "t")]
    timestamp: i64,  // Unix milliseconds
    #[serde(rename = "n")]
    transactions: Option<i64>,
}

/// Rate limiter for API calls (token bucket algorithm)
struct RateLimiter {
    tokens: f64,
    max_tokens: f64,
    refill_rate: f64,  // tokens per second
    last_refill: DateTime<Utc>,
}

impl RateLimiter {
    fn new(max_tokens: f64, refill_rate: f64) -> Self {
        Self {
            tokens: max_tokens,
            max_tokens,
            refill_rate,
            last_refill: Utc::now(),
        }
    }

    /// Try to consume a token, returns true if successful
    fn try_consume(&mut self) -> bool {
        self.refill();
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Wait until a token is available, then consume it
    async fn consume(&mut self) {
        loop {
            if self.try_consume() {
                return;
            }
            // Wait a bit before trying again
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    }

    fn refill(&mut self) {
        let now = Utc::now();
        let elapsed = (now - self.last_refill).num_milliseconds() as f64 / 1000.0;
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.max_tokens);
        self.last_refill = now;
    }

    /// Get time until next token is available (in seconds)
    fn time_until_token(&mut self) -> f64 {
        self.refill();
        if self.tokens >= 1.0 {
            0.0
        } else {
            (1.0 - self.tokens) / self.refill_rate
        }
    }
}

pub struct MarketDataClient {
    client: reqwest::Client,
    pool: PgPool,
    polygon_api_key: Option<String>,
    rate_limiter: Arc<Mutex<RateLimiter>>,
}

impl MarketDataClient {
    pub fn new(pool: PgPool) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("traderjoe/0.1.0")
            .build()
            .expect("Failed to build HTTP client");

        // Get Polygon API key from environment
        let polygon_api_key = std::env::var("POLYGON_API_KEY").ok();

        // Rate limiter: 5 calls per minute = 5/60 = 0.0833 tokens/second
        // Start with 5 tokens (can burst up to 5 calls immediately)
        let rate_limiter = Arc::new(Mutex::new(RateLimiter::new(5.0, 5.0 / 60.0)));

        Self {
            client,
            pool,
            polygon_api_key,
            rate_limiter,
        }
    }
    
    pub async fn fetch_ohlcv(&self, symbol: &str, days: u32) -> DataResult<Vec<OHLCV>> {
        tracing::info!("Fetching OHLCV data for {} (last {} days)", symbol, days);

        // Check if Polygon API key is configured
        let api_key = self.polygon_api_key.as_ref().ok_or_else(|| {
            DataError::Config(
                "POLYGON_API_KEY environment variable must be set. Get your free API key from https://polygon.io".to_string()
            )
        })?;

        // Calculate date range
        let end_date = Utc::now().date_naive();
        let start_date = end_date - chrono::Duration::days(days as i64);

        // Wait for rate limiter (respects 5 calls/min limit)
        {
            let mut limiter = self.rate_limiter.lock().await;
            let wait_time = limiter.time_until_token();
            if wait_time > 0.0 {
                tracing::info!(
                    "Rate limit: waiting {:.1}s before making API call",
                    wait_time
                );
                drop(limiter);  // Release lock while waiting
                tokio::time::sleep(tokio::time::Duration::from_secs_f64(wait_time)).await;
                limiter = self.rate_limiter.lock().await;
            }
            limiter.consume().await;
        }

        // Build Polygon API URL
        let url = format!(
            "https://api.polygon.io/v2/aggs/ticker/{}/range/1/day/{}/{}?adjusted=true&sort=asc&apiKey={}",
            symbol,
            start_date.format("%Y-%m-%d"),
            end_date.format("%Y-%m-%d"),
            api_key
        );

        tracing::debug!("Polygon API request: GET {}", url.replace(api_key, "***"));

        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            tracing::error!("Polygon.io API failed ({}): {}", status_code, error_text);

            return Err(DataError::Api {
                status_code,
                message: format!(
                    "Polygon.io: {}. Check your API key and subscription tier.",
                    error_text
                ),
            });
        }

        let polygon_response: PolygonAggregatesResponse = response.json().await?;

        // Check status - accept both OK and DELAYED
        // DELAYED is returned for free/basic tier subscriptions
        match polygon_response.status.as_str() {
            "OK" => {
                tracing::debug!("Polygon.io status: OK (real-time data)");
            }
            "DELAYED" => {
                tracing::warn!(
                    "Polygon.io status: DELAYED (delayed data - free/basic tier). \
                     Data is still valid but may not be real-time. \
                     Upgrade your subscription at https://polygon.io for real-time data."
                );
            }
            status => {
                return Err(DataError::Internal(format!(
                    "Polygon.io returned error status: {}. \
                     Check your API key and subscription tier at https://polygon.io",
                    status
                )));
            }
        }

        let results = polygon_response.results.ok_or_else(|| {
            DataError::NoData {
                symbol: symbol.to_string(),
                start: start_date.to_string(),
                end: end_date.to_string(),
            }
        })?;

        if results.is_empty() {
            return Err(DataError::NoData {
                symbol: symbol.to_string(),
                start: start_date.to_string(),
                end: end_date.to_string(),
            });
        }

        // Convert Polygon aggregates to our OHLCV format
        let mut data = Vec::new();
        for agg in results {
            // Convert Unix milliseconds to NaiveDate
            let datetime = DateTime::from_timestamp_millis(agg.timestamp)
                .ok_or_else(|| {
                    DataError::Parse {
                        message: format!("Invalid timestamp: {}", agg.timestamp),
                    }
                })?;
            let date = datetime.date_naive();

            data.push(OHLCV {
                symbol: symbol.to_string(),
                date,
                open: agg.open,
                high: agg.high,
                low: agg.low,
                close: agg.close,
                volume: agg.volume as i64,  // Convert float volume to integer
                source: "polygon".to_string(),
            });
        }

        tracing::info!(
            "Fetched {} daily bars from Polygon.io for {}",
            data.len(),
            symbol
        );

        Ok(data)
    }
    
    pub async fn persist_ohlcv(&self, data: &[OHLCV]) -> DataResult<usize> {
        let mut count = 0;
        
        for record in data {
            let result = sqlx::query!(
                r#"
                INSERT INTO ohlcv (symbol, date, open, high, low, close, volume, source)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (symbol, date, source) 
                DO UPDATE SET 
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
                "#,
                record.symbol,
                record.date,
                record.open as f32,
                record.high as f32,
                record.low as f32,
                record.close as f32,
                record.volume,
                record.source
            )
            .execute(&self.pool)
            .await?;
            
            if result.rows_affected() > 0 {
                count += 1;
            }
        }
        
        tracing::info!("Persisted {} OHLCV records", count);
        Ok(count)
    }

    /// Fetch latest market data as JSON (for evening review)
    pub async fn fetch_latest(&self, symbol: &str) -> DataResult<serde_json::Value> {
        use serde_json::json;

        // Fetch the most recent day's data
        let data = self.fetch_ohlcv(symbol, 1).await?;

        if data.is_empty() {
            return Err(DataError::NoData {
                symbol: symbol.to_string(),
                start: "today".to_string(),
                end: "today".to_string(),
            });
        }

        let latest = &data[data.len() - 1];

        Ok(json!({
            "symbol": latest.symbol,
            "date": latest.date.to_string(),
            "open": latest.open,
            "high": latest.high,
            "low": latest.low,
            "close": latest.close,
            "volume": latest.volume,
            "source": latest.source,
        }))
    }

    /// Fetch current VIX value (volatility index)
    pub async fn fetch_vix(&self) -> DataResult<f64> {
        use super::retry::retry_with_backoff;

        retry_with_backoff(|| async {
            tracing::info!("Fetching current VIX value");

            // Fetch VIX (^VIX) from Yahoo Finance
            let data = self.fetch_ohlcv("^VIX", 1).await?;

            if data.is_empty() {
                return Err(DataError::NoData {
                    symbol: "^VIX".to_string(),
                    start: "today".to_string(),
                    end: "today".to_string(),
                });
            }

            let latest = &data[data.len() - 1];
            tracing::info!("Current VIX: {:.2}", latest.close);

            Ok(latest.close)
        }, 3).await
    }

    /// Fetch VIX with market data context
    pub async fn fetch_vix_context(&self) -> DataResult<serde_json::Value> {
        use serde_json::json;

        let vix_value = self.fetch_vix().await?;
        let data = self.fetch_ohlcv("^VIX", 5).await?; // Last 5 days for context

        let avg_5d = if data.len() >= 5 {
            data.iter().take(5).map(|d| d.close).sum::<f64>() / 5.0
        } else {
            vix_value
        };

        // Classify volatility regime
        let regime = if vix_value < 15.0 {
            "low"
        } else if vix_value < 25.0 {
            "normal"
        } else if vix_value < 35.0 {
            "elevated"
        } else {
            "high"
        };

        Ok(json!({
            "current": vix_value,
            "avg_5d": avg_5d,
            "regime": regime,
            "interpretation": match regime {
                "low" => "Low volatility - markets relatively calm",
                "normal" => "Normal volatility - typical market conditions",
                "elevated" => "Elevated volatility - increased uncertainty",
                "high" => "High volatility - significant market stress",
                _ => "Unknown"
            }
        }))
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    
    #[test]
    fn test_ohlcv_serialization() {
        let ohlcv = OHLCV {
            symbol: "SPY".to_string(),
            date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            open: 100.0,
            high: 105.0,
            low: 99.0,
            close: 103.0,
            volume: 1000000,
            source: "yahoo".to_string(),
        };
        
        let json = serde_json::to_string(&ohlcv).unwrap();
        assert!(json.contains("SPY"));
        assert!(json.contains("2024-01-01"));
    }
}
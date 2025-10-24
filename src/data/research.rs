//! Exa API client for deep web research
//! Provides intelligent search capabilities for market analysis

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::PgPool;
use std::time::Duration;
use tokio::time::timeout;
use tracing::{error, info};
use super::{DataError, DataResult};

/// Exa API search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExaSearchResult {
    pub id: String,
    pub url: String,
    pub title: String,
    pub score: Option<f32>,
    pub published_date: Option<String>,
    pub author: Option<String>,
    pub text: Option<String>,
    pub highlights: Option<Vec<String>>,
    pub highlight_scores: Option<Vec<f32>>,
}

/// Exa API response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExaResponse {
    pub results: Vec<ExaSearchResult>,
    pub autoprompt_string: Option<String>,
    pub resolved_search_type: Option<String>,
}

/// Research client configuration
#[derive(Debug, Clone)]
pub struct ResearchConfig {
    pub exa_api_key: String,
    pub base_url: String,
    pub timeout_seconds: u64,
    pub rate_limit_per_minute: u32,
    pub default_num_results: usize,
    pub include_domains: Option<Vec<String>>,
    pub exclude_domains: Option<Vec<String>>,
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            exa_api_key: String::new(),
            base_url: "https://api.exa.ai".to_string(),
            timeout_seconds: 30,
            rate_limit_per_minute: 60,
            default_num_results: 10,
            // Strong allow list of premium financial sources for options trading
            // Note: Exa API only accepts ONE of include_domains or exclude_domains
            include_domains: Some(vec![
                // Major Financial News & Analysis
                "bloomberg.com".to_string(),
                "reuters.com".to_string(),
                "wsj.com".to_string(),
                "ft.com".to_string(),               // Financial Times
                "barrons.com".to_string(),
                "finance.yahoo.com".to_string(),    // Yahoo Finance news/articles
                "marketwatch.com".to_string(),
                "cnbc.com".to_string(),
                "investing.com".to_string(),

                // Options Trading Specific
                "cboe.com".to_string(),            // Chicago Board Options Exchange
                "optionsindustry.org".to_string(), // Options Industry Council
                "tastytrade.com".to_string(),
                "optionsplaybook.com".to_string(),
                "optionalpha.com".to_string(),
                "projectoption.com".to_string(),

                // Market Intelligence & Research
                "seekingalpha.com".to_string(),
                "fool.com".to_string(),            // Motley Fool
                "benzinga.com".to_string(),
                "thefly.com".to_string(),          // The Fly (breaking news)
                "briefing.com".to_string(),
                "zacks.com".to_string(),

                // Broker Research & Education
                "schwab.com".to_string(),
                "fidelity.com".to_string(),
                "thinkorswim.com".to_string(),     // TD Ameritrade platform
                "interactivebrokers.com".to_string(),
                "etrade.com".to_string(),

                // Economic Data & Fed
                "federalreserve.gov".to_string(),
                "stlouisfed.org".to_string(),      // FRED data
                "bls.gov".to_string(),             // Bureau of Labor Statistics

                // Technical Analysis & Trading
                "tradingview.com".to_string(),
                "finviz.com".to_string(),
                "stockcharts.com".to_string(),
                "barchart.com".to_string(),
            ]),
            exclude_domains: None,
        }
    }
}

/// Research client for Exa API integration
pub struct ResearchClient {
    client: reqwest::Client,
    pool: PgPool,
    config: ResearchConfig,
    last_request_time: std::sync::Arc<tokio::sync::Mutex<Option<DateTime<Utc>>>>,
}

impl ResearchClient {
    /// Create new research client
    pub fn new(pool: PgPool, api_key: Option<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("TraderJoe/0.1.0 (ACE Trading System)")
            .build()
            .expect("Failed to build HTTP client");
        
        let config = ResearchConfig {
            exa_api_key: api_key.unwrap_or_default(),
            ..ResearchConfig::default()
        };
        
        Self { 
            client, 
            pool, 
            config,
            last_request_time: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
        }
    }
    
    /// Create client with custom configuration
    pub fn with_config(pool: PgPool, config: ResearchConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .user_agent("TraderJoe/0.1.0 (ACE Trading System)")
            .build()
            .expect("Failed to build HTTP client");
        
        Self { 
            client, 
            pool, 
            config,
            last_request_time: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
        }
    }
    
    /// Perform intelligent search using Exa API
    pub async fn search(&self, query: &str) -> DataResult<serde_json::Value> {
        if self.config.exa_api_key.is_empty() {
            error!("EXA_API_KEY not configured - research functionality unavailable");
            return Err(DataError::Config(
                "EXA_API_KEY environment variable must be set. Get your API key from https://exa.ai".to_string()
            ));
        }
        
        // Rate limiting
        self.enforce_rate_limit().await;
        
        info!("Performing Exa research query: {}", query);
        
        let enhanced_query = self.enhance_query_for_trading(query);
        
        // Note: Exa API only accepts ONE of include_domains or exclude_domains, not both
        let mut search_request = json!({
            "query": enhanced_query,
            "num_results": self.config.default_num_results,
            "start_crawl_date": self.get_start_date_for_relevance(),
            "end_crawl_date": Utc::now().format("%Y-%m-%d").to_string(),
            "use_autoprompt": true,
            "type": "neural", // Use neural search for better semantic matching
            "contents": {
                "text": true,
                "highlights": true,
                "summary": true
            }
        });

        // Add domain filtering (prefer include_domains for quality control)
        if let Some(include) = &self.config.include_domains {
            search_request["include_domains"] = json!(include);
        } else if let Some(exclude) = &self.config.exclude_domains {
            search_request["exclude_domains"] = json!(exclude);
        }
        
        let url = format!("{}/search", self.config.base_url);
        
        let response = timeout(
            Duration::from_secs(self.config.timeout_seconds),
            self.client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.config.exa_api_key))
                .header("Content-Type", "application/json")
                .json(&search_request)
                .send()
        ).await;
        
        match response {
            Ok(Ok(resp)) => {
                if resp.status().is_success() {
                    match resp.json::<ExaResponse>().await {
                        Ok(exa_response) => {
                            let processed_result = self.process_exa_response(query, &exa_response).await;
                            self.persist_research_result(query, &processed_result).await?;
                            Ok(processed_result)
                        }
                        Err(e) => {
                            error!("Failed to parse Exa API response: {}", e);
                            Err(DataError::Internal(format!("Invalid Exa API response format: {}", e)))
                        }
                    }
                } else {
                    let status = resp.status();
                    let error_text = resp.text().await.unwrap_or("Unknown error".to_string());
                    error!("Exa API error {}: {}", status, error_text);

                    if status == 429 {
                        return Err(DataError::RateLimit { retry_after: 60 });
                    } else if status == 401 {
                        return Err(DataError::Authentication("Invalid Exa API key".to_string()));
                    }

                    Err(DataError::Api {
                        status_code: status.as_u16(),
                        message: format!("Exa API: {}", error_text)
                    })
                }
            }
            Ok(Err(e)) => {
                error!("HTTP request failed: {}", e);
                Err(DataError::Network(e))
            }
            Err(_) => {
                error!("Request timeout after {} seconds", self.config.timeout_seconds);
                Err(DataError::Timeout { timeout_seconds: self.config.timeout_seconds })
            }
        }
    }
    
    /// Enhanced search with embeddings for ACE context
    pub async fn search_with_embeddings(
        &self, 
        query: &str, 
        embedder: &crate::embeddings::EmbeddingGemma
    ) -> DataResult<(serde_json::Value, Vec<f32>)> {
        let result = self.search(query).await?;
        
        // Generate embedding for the research summary
        let summary_text = self.extract_summary_from_result(&result);
        let embedding = embedder.embed(&summary_text).await
            .map_err(|e| DataError::Internal(format!("Failed to generate embedding: {}", e)))?;
        
        Ok((result, embedding))
    }
    
    /// Enhance query for better trading-focused results
    fn enhance_query_for_trading(&self, query: &str) -> String {
        let trading_keywords = [
            "market analysis", "trading", "investment", "financial analysis",
            "technical analysis", "market sentiment", "options", "stocks"
        ];
        
        // If query already contains trading keywords, return as-is
        if trading_keywords.iter().any(|&keyword| 
            query.to_lowercase().contains(keyword)
        ) {
            return query.to_string();
        }
        
        // Otherwise, enhance with trading context
        format!("{} market analysis trading outlook", query)
    }
    
    /// Get appropriate start date for relevant financial content
    fn get_start_date_for_relevance(&self) -> String {
        // Look back 30 days for financial content relevance
        (Utc::now() - chrono::Duration::days(30))
            .format("%Y-%m-%d")
            .to_string()
    }
    
    /// Process Exa API response into our standard format
    async fn process_exa_response(&self, query: &str, exa_response: &ExaResponse) -> serde_json::Value {
        let results: Vec<serde_json::Value> = exa_response.results.iter().map(|result| {
            json!({
                "id": result.id,
                "title": result.title,
                "url": result.url,
                "text": result.text.as_ref().unwrap_or(&"No content available".to_string()),
                "score": result.score.unwrap_or(0.0),
                "published_date": result.published_date.as_ref().unwrap_or(&"Unknown".to_string()),
                "author": result.author.as_ref().unwrap_or(&"Unknown".to_string()),
                "highlights": result.highlights.as_ref().unwrap_or(&vec![]),
                "highlight_scores": result.highlight_scores.as_ref().unwrap_or(&vec![])
            })
        }).collect();
        
        json!({
            "query": query,
            "results": results,
            "source": "exa_api",
            "autoprompt_string": exa_response.autoprompt_string,
            "search_type": exa_response.resolved_search_type,
            "total_results": results.len(),
            "timestamp": Utc::now().to_rfc3339()
        })
    }
    
    /// Persist research result to database
    async fn persist_research_result(&self, query: &str, result: &serde_json::Value) -> DataResult<()> {
        sqlx::query!(
            r#"
            INSERT INTO research (captured_at, query, result)
            VALUES ($1, $2, $3)
            "#,
            Utc::now(),
            query,
            result.clone()
        )
        .execute(&self.pool)
        .await
        .map_err(DataError::Database)?;
        
        info!("Research result for '{}' persisted to database", query);
        Ok(())
    }
    
    /// Extract summary text for embedding generation
    fn extract_summary_from_result(&self, result: &serde_json::Value) -> String {
        let mut summary_parts = vec![];
        
        if let Some(query) = result.get("query").and_then(|q| q.as_str()) {
            summary_parts.push(format!("Research query: {}", query));
        }
        
        if let Some(results) = result.get("results").and_then(|r| r.as_array()) {
            for (i, result_item) in results.iter().take(3).enumerate() { // Top 3 results
                if let Some(title) = result_item.get("title").and_then(|t| t.as_str()) {
                    summary_parts.push(format!("Result {}: {}", i + 1, title));
                }
                
                if let Some(highlights) = result_item.get("highlights").and_then(|h| h.as_array()) {
                    let highlight_text: Vec<String> = highlights.iter()
                        .filter_map(|h| h.as_str())
                        .map(|h| h.to_string())
                        .collect();
                    if !highlight_text.is_empty() {
                        summary_parts.push(format!("Key points: {}", highlight_text.join(", ")));
                    }
                }
            }
        }
        
        summary_parts.join(". ")
    }
    
    /// Enforce rate limiting
    async fn enforce_rate_limit(&self) {
        let mut last_request = self.last_request_time.lock().await;
        
        if let Some(last_time) = *last_request {
            let min_interval = Duration::from_secs(60 / self.config.rate_limit_per_minute as u64);
            let elapsed = Utc::now().signed_duration_since(last_time).to_std().unwrap_or(Duration::ZERO);
            
            if elapsed < min_interval {
                let sleep_duration = min_interval - elapsed;
                info!("Rate limiting: sleeping for {:?}", sleep_duration);
                tokio::time::sleep(sleep_duration).await;
            }
        }
        
        *last_request = Some(Utc::now());
    }
    
    /// Test API connectivity
    pub async fn health_check(&self) -> DataResult<bool> {
        if self.config.exa_api_key.is_empty() {
            return Ok(false); // API key not configured
        }
        
        let test_query = "market news";
        match timeout(
            Duration::from_secs(10),
            self.search(test_query)
        ).await {
            Ok(Ok(_)) => Ok(true),
            _ => Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use sqlx::PgPool;
    
    fn create_test_pool() -> PgPool {
        // These unit tests don't actually use the pool, they test pure functions
        // Use lazy connection to avoid actual database requirement
        PgPool::connect_lazy("postgresql://localhost/test")
            .expect("Failed to create lazy pool for testing")
    }
    
    #[tokio::test]
    async fn test_query_enhancement() {
        let config = ResearchConfig::default();
        let pool = create_test_pool();
        let client = ResearchClient::with_config(pool, config);
        
        // Test query enhancement
        assert_eq!(
            client.enhance_query_for_trading("market analysis"),
            "market analysis" // Should not enhance if already has trading keywords
        );
        
        assert_eq!(
            client.enhance_query_for_trading("SPY outlook"),
            "SPY outlook market analysis trading outlook"
        );
    }
    
    #[tokio::test]
    async fn test_start_date_generation() {
        let config = ResearchConfig::default();
        let pool = create_test_pool();
        let client = ResearchClient::with_config(pool, config);
        
        let start_date = client.get_start_date_for_relevance();
        assert!(start_date.len() == 10); // YYYY-MM-DD format
        assert!(start_date.contains("-"));
    }
    
    #[tokio::test]
    async fn test_exa_response_processing() {
        let config = ResearchConfig::default();
        let pool = create_test_pool();
        let client = ResearchClient::with_config(pool, config);
        
        let test_exa_response = ExaResponse {
            results: vec![
                ExaSearchResult {
                    id: "test_1".to_string(),
                    url: "https://example.com".to_string(),
                    title: "Test Article".to_string(),
                    score: Some(0.95),
                    published_date: Some("2025-01-15".to_string()),
                    author: Some("Test Author".to_string()),
                    text: Some("Test content".to_string()),
                    highlights: Some(vec!["market".to_string(), "analysis".to_string()]),
                    highlight_scores: Some(vec![0.9, 0.8]),
                }
            ],
            autoprompt_string: Some("Enhanced query".to_string()),
            resolved_search_type: Some("neural".to_string()),
        };
        
        let processed = client.process_exa_response("test query", &test_exa_response).await;
        
        assert_eq!(processed["query"].as_str().unwrap(), "test query");
        assert_eq!(processed["source"].as_str().unwrap(), "exa_api");
        assert_eq!(processed["total_results"].as_u64().unwrap(), 1);
        
        let results = processed["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        
        let first_result = &results[0];
        assert_eq!(first_result["id"].as_str().unwrap(), "test_1");
        assert_eq!(first_result["title"].as_str().unwrap(), "Test Article");
        // Use approximate comparison for floating point
        assert!((first_result["score"].as_f64().unwrap() - 0.95).abs() < 0.001);
    }
    
    #[tokio::test]
    async fn test_summary_extraction() {
        let config = ResearchConfig::default();
        let pool = create_test_pool();
        let client = ResearchClient::with_config(pool, config);
        
        let test_result = json!({
            "query": "market outlook",
            "results": [
                {
                    "title": "Market Analysis Today",
                    "highlights": ["bullish sentiment", "technical breakout"]
                },
                {
                    "title": "Economic Report",
                    "highlights": ["GDP growth", "inflation concerns"]
                }
            ]
        });
        
        let summary = client.extract_summary_from_result(&test_result);
        assert!(summary.contains("Research query: market outlook"));
        assert!(summary.contains("Result 1: Market Analysis Today"));
        assert!(summary.contains("bullish sentiment"));
    }
    
    #[tokio::test]
    #[ignore] // Requires Exa API key
    async fn test_exa_api_integration() {
        // This test would require a real Exa API key
        // Skip for unit testing, include in integration tests with real API
    }
}

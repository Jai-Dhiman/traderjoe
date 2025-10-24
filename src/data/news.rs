use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::PgPool;
use super::{DataError, DataResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsArticle {
    pub title: String,
    pub description: Option<String>,
    pub url: String,
    pub source: String,
    pub published_at: String,
    pub sentiment: Option<f64>,
}

pub struct NewsClient {
    pool: PgPool,
    api_key: Option<String>,
    http_client: reqwest::Client,
}

impl NewsClient {
    pub fn new(pool: PgPool, api_key: Option<String>) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("traderjoe/0.1.0")
            .build()
            .expect("Failed to build HTTP client");

        Self {
            pool,
            api_key,
            http_client,
        }
    }

    /// Fetch news headlines for a symbol (or general market news)
    pub async fn fetch_news(&self, symbol: Option<&str>) -> DataResult<Vec<NewsArticle>> {
        let symbol_str = symbol.unwrap_or("stock market");
        tracing::info!("Fetching news for: {}", symbol_str);

        // If API key not configured, use fallback
        if self.api_key.is_none() {
            tracing::warn!("NewsAPI key not configured, using fallback");
            return Ok(self.generate_fallback_news(symbol_str));
        }

        // Try NewsAPI first
        match self.fetch_from_newsapi(symbol_str).await {
            Ok(articles) => {
                self.persist_news(&articles).await?;
                Ok(articles)
            }
            Err(e) => {
                tracing::warn!("NewsAPI failed: {}, using fallback", e);
                Ok(self.generate_fallback_news(symbol_str))
            }
        }
    }

    /// Fetch news from NewsAPI
    async fn fetch_from_newsapi(&self, query: &str) -> DataResult<Vec<NewsArticle>> {
        use super::retry::retry_with_backoff;

        let api_key = self.api_key.as_ref().ok_or_else(|| {
            DataError::Config("NewsAPI key not configured".to_string())
        })?;

        retry_with_backoff(|| async {
            let url = format!(
                "https://newsapi.org/v2/everything?q={}&sortBy=publishedAt&pageSize=10&apiKey={}",
                urlencoding::encode(query),
                api_key
            );

            let response = self.http_client
                .get(&url)
                .send()
                .await?;

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                return Err(DataError::Api {
                    status_code: status.as_u16(),
                    message: format!("NewsAPI error: {}", error_text),
                });
            }

            let response_json: serde_json::Value = response.json().await?;

            let articles = response_json["articles"]
                .as_array()
                .ok_or_else(|| DataError::Parse {
                    message: "No articles array in response".to_string(),
                })?;

            let mut news_articles = Vec::new();
            for article in articles {
                let title = article["title"].as_str().unwrap_or("No title").to_string();
                let description = article["description"].as_str().map(String::from);
                let url = article["url"].as_str().unwrap_or("").to_string();
                let source = article["source"]["name"].as_str().unwrap_or("Unknown").to_string();
                let published_at = article["publishedAt"].as_str().unwrap_or("").to_string();

                // Simple sentiment based on keywords
                let sentiment = self.estimate_sentiment(&title, description.as_deref());

                news_articles.push(NewsArticle {
                    title,
                    description,
                    url,
                    source,
                    published_at,
                    sentiment: Some(sentiment),
                });
            }

            tracing::info!("Fetched {} news articles from NewsAPI", news_articles.len());
            Ok(news_articles)
        }, 2).await
    }

    /// Estimate sentiment from title and description
    fn estimate_sentiment(&self, title: &str, description: Option<&str>) -> f64 {
        let text = format!(
            "{} {}",
            title.to_lowercase(),
            description.unwrap_or("").to_lowercase()
        );

        let positive_words = ["gain", "surge", "rally", "jump", "rise", "bull", "strong", "positive", "growth"];
        let negative_words = ["fall", "drop", "crash", "decline", "bear", "weak", "negative", "loss", "concern"];

        let mut positive_count = 0;
        let mut negative_count = 0;

        for word in positive_words {
            if text.contains(word) {
                positive_count += 1;
            }
        }

        for word in negative_words {
            if text.contains(word) {
                negative_count += 1;
            }
        }

        let total = positive_count + negative_count;
        if total == 0 {
            return 0.0;
        }

        (positive_count as f64 - negative_count as f64) / total as f64
    }

    /// Generate fallback news when API is unavailable
    fn generate_fallback_news(&self, _query: &str) -> Vec<NewsArticle> {
        vec![
            NewsArticle {
                title: "Market News (Fallback Mode)".to_string(),
                description: Some("NewsAPI credentials not configured. Using fallback mode.".to_string()),
                url: "https://example.com".to_string(),
                source: "fallback".to_string(),
                published_at: Utc::now().to_rfc3339(),
                sentiment: Some(0.0),
            }
        ]
    }

    /// Persist news articles to database
    async fn persist_news(&self, articles: &[NewsArticle]) -> DataResult<()> {
        for article in articles {
            // Store in research table (reusing existing table for news storage)
            sqlx::query(
                r#"
                INSERT INTO research (created_at, query, source, result_data)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT DO NOTHING
                "#,
            )
            .bind(Utc::now())
            .bind(&article.title)
            .bind("newsapi")
            .bind(json!({
                "title": &article.title,
                "description": &article.description,
                "url": &article.url,
                "source": &article.source,
                "published_at": &article.published_at,
                "sentiment": &article.sentiment,
            }))
            .execute(&self.pool)
            .await?;
        }

        tracing::info!("Persisted {} news articles", articles.len());
        Ok(())
    }

    /// Get summary of recent news sentiment
    pub async fn get_news_summary(&self, symbol: Option<&str>) -> DataResult<serde_json::Value> {
        let articles = self.fetch_news(symbol).await?;

        let avg_sentiment = if !articles.is_empty() {
            articles.iter()
                .filter_map(|a| a.sentiment)
                .sum::<f64>() / articles.len() as f64
        } else {
            0.0
        };

        let interpretation = if avg_sentiment > 0.3 {
            "Positive"
        } else if avg_sentiment < -0.3 {
            "Negative"
        } else {
            "Neutral"
        };

        Ok(json!({
            "count": articles.len(),
            "avg_sentiment": avg_sentiment,
            "interpretation": interpretation,
            "headlines": articles.iter().take(5).map(|a| &a.title).collect::<Vec<_>>(),
        }))
    }
}

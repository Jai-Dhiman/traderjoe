use chrono::Utc;
use serde_json::json;
use sqlx::PgPool;
use roux::Subreddit;
use super::{DataError, DataResult};

/// Lexicon-based sentiment keywords (simple approach)
const BULLISH_WORDS: &[&str] = &[
    "bullish", "bull", "calls", "moon", "rocket", "pump", "rally", "breakout",
    "buy", "long", "support", "green", "gains", "profit", "strong"
];

const BEARISH_WORDS: &[&str] = &[
    "bearish", "bear", "puts", "crash", "dump", "sell", "short", "resistance",
    "red", "loss", "weak", "drop", "fall", "decline", "correction"
];

pub struct SentimentClient {
    pool: PgPool,
    reddit_client_id: Option<String>,
    reddit_client_secret: Option<String>,
    _http_client: reqwest::Client,
}

impl SentimentClient {
    pub fn new(pool: PgPool, client_id: Option<String>, client_secret: Option<String>) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("traderjoe/0.1.0")
            .build()
            .expect("Failed to build HTTP client");

        Self {
            pool,
            reddit_client_id: client_id,
            reddit_client_secret: client_secret,
            _http_client: http_client,
        }
    }

    pub async fn analyze_reddit(&self, symbol: Option<&str>) -> DataResult<serde_json::Value> {
        let symbol_str = symbol.unwrap_or("SPY");
        tracing::info!("Analyzing Reddit sentiment for: {}", symbol_str);

        // If credentials are not configured, return a useful stub
        if self.reddit_client_id.is_none() || self.reddit_client_secret.is_none() {
            tracing::warn!("Reddit credentials not configured, using heuristic fallback");
            return self.fallback_sentiment(symbol_str).await;
        }

        // Try to fetch real Reddit data
        match self.fetch_reddit_sentiment(symbol_str).await {
            Ok(sentiment) => {
                self.persist_sentiment(&sentiment).await?;
                Ok(sentiment)
            }
            Err(e) => {
                tracing::warn!("Reddit API failed: {}, using fallback", e);
                self.fallback_sentiment(symbol_str).await
            }
        }
    }

    /// Fetch sentiment from Reddit using roux library
    async fn fetch_reddit_sentiment(&self, symbol: &str) -> DataResult<serde_json::Value> {
        use super::retry::retry_with_backoff;

        retry_with_backoff(|| async {
            // Target subreddits for trading sentiment
            let subreddits = vec!["wallstreetbets", "stocks", "options"];

            let mut all_posts = Vec::new();
            let mut total_score = 0.0;
            let mut post_count = 0;

            for sub in subreddits {
                // Fetch hot posts from subreddit
                let subreddit = Subreddit::new(sub);

                match subreddit.hot(25, None).await {
                    Ok(posts) => {
                        for post in posts.data.children {
                            let title = post.data.title.to_lowercase();
                            let selftext = post.data.selftext.to_lowercase();
                            let combined = format!("{} {}", title, selftext);

                            // Check if post mentions the symbol
                            if combined.contains(&symbol.to_lowercase())
                                || combined.contains(&format!("${}", symbol.to_lowercase())) {

                                // Calculate sentiment score for this post
                                let sentiment_score = self.calculate_sentiment(&combined);

                                // Weight by upvotes (more upvotes = more influence)
                                let weight = (post.data.ups as f64).ln().max(1.0);
                                total_score += sentiment_score * weight;
                                post_count += 1;

                                all_posts.push(json!({
                                    "title": post.data.title,
                                    "score": post.data.ups,
                                    "sentiment": sentiment_score,
                                    "subreddit": sub,
                                    "url": format!("https://reddit.com{}", post.data.permalink),
                                }));

                                if post_count >= 20 {
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to fetch from r/{}: {}", sub, e);
                        continue;
                    }
                }

                if post_count >= 20 {
                    break;
                }
            }

            if post_count == 0 {
                return Err(DataError::NoData {
                    symbol: symbol.to_string(),
                    start: "reddit".to_string(),
                    end: "no posts found".to_string(),
                });
            }

            // Normalize sentiment score to -1 to +1 range
            let normalized_score = (total_score / post_count as f64).tanh();

            Ok(json!({
                "symbol": symbol,
                "score": normalized_score,
                "source": "reddit",
                "sample_size": post_count,
                "timestamp": Utc::now().to_rfc3339(),
                "interpretation": self.interpret_sentiment(normalized_score),
                "sample_posts": all_posts.iter().take(5).collect::<Vec<_>>(),
            }))
        }, 2).await
    }

    /// Calculate sentiment score from text using lexicon-based approach
    fn calculate_sentiment(&self, text: &str) -> f64 {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut bullish_count = 0;
        let mut bearish_count = 0;

        for word in &words {
            let word_lower = word.to_lowercase();
            if BULLISH_WORDS.iter().any(|&w| word_lower.contains(w)) {
                bullish_count += 1;
            }
            if BEARISH_WORDS.iter().any(|&w| word_lower.contains(w)) {
                bearish_count += 1;
            }
        }

        let total = bullish_count + bearish_count;
        if total == 0 {
            return 0.0; // Neutral
        }

        // Return score from -1 (bearish) to +1 (bullish)
        (bullish_count as f64 - bearish_count as f64) / total as f64
    }

    /// Interpret sentiment score as human-readable text
    fn interpret_sentiment(&self, score: f64) -> &'static str {
        if score > 0.5 {
            "Very Bullish"
        } else if score > 0.2 {
            "Bullish"
        } else if score > -0.2 {
            "Neutral"
        } else if score > -0.5 {
            "Bearish"
        } else {
            "Very Bearish"
        }
    }

    /// Fallback sentiment when Reddit API is unavailable
    async fn fallback_sentiment(&self, symbol: &str) -> DataResult<serde_json::Value> {
        let sentiment = json!({
            "symbol": symbol,
            "score": 0.0,
            "source": "reddit",
            "sample_size": 0,
            "timestamp": Utc::now().to_rfc3339(),
            "interpretation": "Neutral (fallback)",
            "note": "Reddit API credentials not configured or unavailable"
        });

        self.persist_sentiment(&sentiment).await?;
        Ok(sentiment)
    }

    /// Persist sentiment data to database
    async fn persist_sentiment(&self, sentiment: &serde_json::Value) -> DataResult<()> {
        let symbol = sentiment["symbol"].as_str().unwrap_or("UNKNOWN");
        let score = sentiment["score"].as_f64().unwrap_or(0.0);

        sqlx::query(
            r#"
            INSERT INTO sentiment (captured_at, source, symbol, score, meta)
            VALUES ($1, $2, $3, $4, $5)
            "#,
        )
        .bind(Utc::now())
        .bind("reddit")
        .bind(symbol)
        .bind(score)
        .bind(sentiment.clone())
        .execute(&self.pool)
        .await?;

        tracing::info!("Reddit sentiment persisted for {}: {:.2}", symbol, score);
        Ok(())
    }
}

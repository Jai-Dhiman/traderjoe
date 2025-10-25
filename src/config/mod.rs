use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub database: DatabaseConfig,
    pub apis: ApiConfig,
    pub llm: LlmConfig,
    pub trading: TradingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub min_connections: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub exa_api_key: Option<String>,
    pub reddit_client_id: Option<String>,
    pub reddit_client_secret: Option<String>,
    pub news_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub anthropic_api_key: Option<String>,
    pub cerebras_api_key: Option<String>,
    pub github_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: String, // "cerebras" or "openrouter"
    pub cerebras_url: String,
    pub openrouter_url: String,
    pub primary_model: String,
    pub fallback_model: String,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingConfig {
    pub paper_trading: bool,
    pub max_position_size_pct: f64,
    pub max_daily_loss_pct: f64,
    pub max_weekly_loss_pct: f64,
}

impl Config {
    pub fn load() -> Result<Self> {
        // Load .env file if it exists (fail silently if not found)
        dotenv::dotenv().ok();

        // Database configuration - DATABASE_URL is required
        let database_url = env::var("DATABASE_URL")
            .context("DATABASE_URL environment variable is required but not set")?;

        let config = Config {
            database: DatabaseConfig {
                url: database_url,
                max_connections: env::var("DB_MAX_CONNECTIONS")
                    .unwrap_or_else(|_| "5".to_string())
                    .parse()
                    .context("Invalid DB_MAX_CONNECTIONS value")?,
                min_connections: env::var("DB_MIN_CONNECTIONS")
                    .unwrap_or_else(|_| "1".to_string())
                    .parse()
                    .context("Invalid DB_MIN_CONNECTIONS value")?,
            },
            apis: ApiConfig {
                exa_api_key: env::var("EXA_API_KEY").ok(),
                reddit_client_id: env::var("REDDIT_CLIENT_ID").ok(),
                reddit_client_secret: env::var("REDDIT_CLIENT_SECRET").ok(),
                news_api_key: env::var("NEWS_API_KEY").ok(),
                openai_api_key: env::var("OPENAI_API_KEY").ok(),
                anthropic_api_key: env::var("ANTHROPIC_API_KEY").ok(),
                cerebras_api_key: env::var("CEREBRAS_API_KEY").ok(),
                github_token: env::var("GITHUB_TOKEN").ok(),
            },
            llm: LlmConfig {
                provider: env::var("LLM_PROVIDER").unwrap_or_else(|_| "cerebras".to_string()),
                cerebras_url: env::var("CEREBRAS_URL")
                    .unwrap_or_else(|_| "https://api.cerebras.ai/v1".to_string()),
                openrouter_url: env::var("OPENROUTER_URL")
                    .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string()),
                primary_model: env::var("PRIMARY_MODEL")
                    .unwrap_or_else(|_| "llama-3.3-70b".to_string()),
                fallback_model: env::var("FALLBACK_MODEL")
                    .unwrap_or_else(|_| "llama-3.1-8b".to_string()),
                timeout_seconds: env::var("LLM_TIMEOUT_SECONDS")
                    .unwrap_or_else(|_| "30".to_string())
                    .parse()
                    .context("Invalid LLM_TIMEOUT_SECONDS value")?,
            },
            trading: TradingConfig {
                paper_trading: env::var("PAPER_TRADING")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .context("Invalid PAPER_TRADING value (use true/false)")?,
                max_position_size_pct: env::var("MAX_POSITION_SIZE_PCT")
                    .unwrap_or_else(|_| "5.0".to_string())
                    .parse()
                    .context("Invalid MAX_POSITION_SIZE_PCT value")?,
                max_daily_loss_pct: env::var("MAX_DAILY_LOSS_PCT")
                    .unwrap_or_else(|_| "3.0".to_string())
                    .parse()
                    .context("Invalid MAX_DAILY_LOSS_PCT value")?,
                max_weekly_loss_pct: env::var("MAX_WEEKLY_LOSS_PCT")
                    .unwrap_or_else(|_| "10.0".to_string())
                    .parse()
                    .context("Invalid MAX_WEEKLY_LOSS_PCT value")?,
            },
        };

        Ok(config)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            database: DatabaseConfig {
                url: "postgresql://localhost/traderjoe".to_string(),
                max_connections: 5,
                min_connections: 1,
            },
            apis: ApiConfig {
                exa_api_key: None,
                reddit_client_id: None,
                reddit_client_secret: None,
                news_api_key: None,
                openai_api_key: None,
                anthropic_api_key: None,
                cerebras_api_key: None,
                github_token: None,
            },
            llm: LlmConfig {
                provider: "cerebras".to_string(),
                cerebras_url: "https://api.cerebras.ai/v1".to_string(),
                openrouter_url: "https://openrouter.ai/api/v1".to_string(),
                primary_model: "llama-3.3-70b".to_string(),
                fallback_model: "llama-3.1-8b".to_string(),
                timeout_seconds: 30,
            },
            trading: TradingConfig {
                paper_trading: true,
                max_position_size_pct: 5.0,
                max_daily_loss_pct: 3.0,
                max_weekly_loss_pct: 10.0,
            },
        }
    }
}

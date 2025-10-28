use anyhow::{Context, Result};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub database: DatabaseConfig,
    pub apis: ApiConfig,
    pub llm: LlmConfig,
    pub trading: TradingConfig,

    // Backtest mode configuration (set programmatically, not from env vars)
    #[serde(skip)]
    pub backtest_mode: Option<bool>,
    #[serde(skip)]
    pub backtest_date: Option<NaiveDate>,
    #[serde(skip)]
    pub skip_sentiment: Option<bool>,
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
    pub cloudflare_account_id: Option<String>,
    pub cloudflare_api_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: String, // "workers_ai" or "ollama"
    pub workers_ai_url: String,
    pub ollama_url: String,
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
                cloudflare_account_id: env::var("CLOUDFLARE_ACCOUNT_ID").ok(),
                cloudflare_api_token: env::var("CLOUDFLARE_API_TOKEN").ok(),
            },
            llm: LlmConfig {
                provider: {
                    let provider = env::var("LLM_PROVIDER").unwrap_or_else(|_| "workers_ai".to_string());
                    eprintln!("🔧 Config: LLM_PROVIDER = {}", provider);
                    provider
                },
                workers_ai_url: env::var("WORKERS_AI_URL")
                    .unwrap_or_else(|_| "https://api.cloudflare.com/client/v4/accounts".to_string()),
                ollama_url: env::var("OLLAMA_URL")
                    .unwrap_or_else(|_| "http://localhost:11434/v1".to_string()),
                primary_model: {
                    let model = env::var("PRIMARY_MODEL")
                        .unwrap_or_else(|_| "@cf/meta/llama-4-scout-17b-16e-instruct".to_string());
                    eprintln!("🔧 Config: PRIMARY_MODEL = {}", model);
                    model
                },
                fallback_model: {
                    let model = env::var("FALLBACK_MODEL")
                        .unwrap_or_else(|_| "@cf/meta/llama-4-scout-17b-16e-instruct".to_string());
                    eprintln!("🔧 Config: FALLBACK_MODEL = {}", model);
                    model
                },
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
            // Backtest mode is not loaded from env vars - set programmatically
            backtest_mode: None,
            backtest_date: None,
            skip_sentiment: None,
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
                cloudflare_account_id: None,
                cloudflare_api_token: None,
            },
            llm: LlmConfig {
                provider: "workers_ai".to_string(),
                workers_ai_url: "https://api.cloudflare.com/client/v4/accounts".to_string(),
                ollama_url: "http://localhost:11434".to_string(),
                primary_model: "@cf/meta/llama-4-scout-17b-16e-instruct".to_string(),
                fallback_model: "@cf/meta/llama-4-scout-17b-16e-instruct".to_string(),
                timeout_seconds: 30,
            },
            trading: TradingConfig {
                paper_trading: true,
                max_position_size_pct: 5.0,
                max_daily_loss_pct: 3.0,
                max_weekly_loss_pct: 10.0,
            },
            backtest_mode: None,
            backtest_date: None,
            skip_sentiment: None,
        }
    }
}

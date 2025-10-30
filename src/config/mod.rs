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
    pub openai_api_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: String, // "workers_ai" or "openai"
    pub workers_ai_url: String,
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
    /// Returns the effective datetime for operations, respecting backtest mode.
    /// In backtest mode, returns the backtest_date at 9:30 AM (market open).
    /// In live mode, returns the current time.
    pub fn get_effective_datetime(&self) -> chrono::DateTime<chrono::Utc> {
        if let Some(backtest_date) = self.backtest_date {
            backtest_date
                .and_hms_opt(9, 30, 0)
                .expect("Invalid time: 9:30:00")
                .and_utc()
        } else {
            chrono::Utc::now()
        }
    }

    /// Returns the effective date for operations, respecting backtest mode.
    /// In backtest mode, returns the backtest_date.
    /// In live mode, returns the current date.
    pub fn get_effective_date(&self) -> chrono::NaiveDate {
        self.backtest_date.unwrap_or_else(|| chrono::Utc::now().date_naive())
    }

    pub fn load() -> Result<Self> {
        // Load .env file - this sets env vars that aren't already set
        dotenv::dotenv().ok();

        // Force override critical LLM config from .env file if it exists
        // This ensures .env file takes precedence over shell environment
        if let Ok(vars) = dotenv::from_filename_iter(".env") {
            for item in vars {
                if let Ok((key, val)) = item {
                    if key == "LLM_PROVIDER" || key == "PRIMARY_MODEL" || key == "FALLBACK_MODEL" {
                        env::set_var(&key, &val);
                    }
                }
            }
        }

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
                openai_api_key: env::var("OPENAI_API_KEY").ok(),
            },
            llm: LlmConfig {
                provider: {
                    let provider = env::var("LLM_PROVIDER").unwrap_or_else(|_| "openai".to_string());
                    eprintln!("🔧 Config: LLM_PROVIDER = {}", provider);
                    provider
                },
                workers_ai_url: env::var("WORKERS_AI_URL")
                    .unwrap_or_else(|_| "https://api.cloudflare.com/client/v4/accounts".to_string()),
                primary_model: {
                    let model = env::var("PRIMARY_MODEL")
                        .unwrap_or_else(|_| "gpt-5-nano-2025-08-07".to_string());
                    eprintln!("🔧 Config: PRIMARY_MODEL = {}", model);
                    model
                },
                fallback_model: {
                    let model = env::var("FALLBACK_MODEL")
                        .unwrap_or_else(|_| "gpt-5-nano-2025-08-07".to_string());
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
                openai_api_key: None,
            },
            llm: LlmConfig {
                provider: "openai".to_string(),
                workers_ai_url: "https://api.cloudflare.com/client/v4/accounts".to_string(),
                primary_model: "gpt-5-nano-2025-08-07".to_string(),
                fallback_model: "gpt-5-nano-2025-08-07".to_string(),
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

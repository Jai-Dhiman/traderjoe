use clap::{Parser, Subcommand};
use chrono::NaiveDate;
use uuid::Uuid;
use sqlx::PgPool;
use anyhow::Result;
use tracing::info;

pub mod commands;
pub mod migrate;

#[derive(Parser)]
#[command(
    name = "traderjoe",
    about = "ACE-Enhanced Daily Trading System",
    version = "0.1.0",
    author = "jdhiman"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Configuration file path
    #[arg(short, long, global = true)]
    pub config: Option<String>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run morning market analysis and generate trading recommendations
    Analyze {
        /// Date to analyze (defaults to today)
        #[arg(short, long)]
        date: Option<NaiveDate>,
        
        /// Symbol to analyze
        #[arg(short, long, default_value = "SPY")]
        symbol: String,
    },

    /// Execute a paper trade based on recommendation
    Execute {
        /// Recommendation ID to execute
        #[arg(short, long)]
        recommendation_id: Uuid,
    },

    /// Run evening review and update ACE playbook
    Review {
        /// Date to review (defaults to today)
        #[arg(short, long)]
        date: Option<NaiveDate>,
    },

    /// Generate weekly performance report and deep analysis
    Weekly {
        /// Start date for weekly analysis
        #[arg(short, long)]
        start_date: Option<NaiveDate>,
    },

    /// Fetch market data from various sources
    Fetch {
        /// Symbol to fetch data for
        #[arg(short, long, default_value = "SPY")]
        symbol: String,
        
        /// Type of data to fetch
        #[arg(short = 't', long, default_value = "ohlcv")]
        data_type: String,
        
        /// Number of days to fetch (default: 30)
        #[arg(short, long)]
        days: Option<u32>,
        
        /// Data source (yahoo, alpha_vantage, etc.)
        #[arg(short = 'o', long)]
        source: Option<String>,
    },

    /// Run deep research query using Exa API
    Research {
        /// Research query
        #[arg(short, long)]
        query: String,
    },

    /// Collect sentiment data from various sources
    Sentiment {
        /// Sentiment source (reddit, news, etc.)
        #[arg(short, long, default_value = "reddit")]
        source: String,
    },

    /// Run database migrations
    Migrate,

    /// Query ACE context database for similar patterns
    AceQuery {
        /// Natural language query for ACE context
        #[arg(short, long)]
        query: String,
    },

    /// Display ACE playbook statistics and patterns
    PlaybookStats,

    /// Run backtesting on historical data
    Backtest {
        /// Start date for backtest
        #[arg(short, long)]
        start_date: NaiveDate,

        /// End date for backtest
        #[arg(short, long)]
        end_date: NaiveDate,

        /// Strategy to test
        #[arg(short = 't', long, default_value = "ace")]
        strategy: String,
    },

    /// Display open positions and account status
    Positions,

    /// Display performance metrics and statistics
    Performance {
        /// Number of days to analyze (default: 30)
        #[arg(short, long)]
        days: Option<i32>,
    },

    /// Review all pending contexts
    ReviewAll,

    /// Close an open position manually
    Close {
        /// Trade ID to close
        #[arg(short, long)]
        trade_id: Uuid,

        /// Exit reason (optional note)
        #[arg(short, long)]
        reason: Option<String>,
    },

    /// Run auto-exit checks for open positions
    AutoExit {
        /// Custom auto-exit time (e.g., "15:00" for 3:00 PM ET)
        #[arg(short = 't', long)]
        exit_time: Option<String>,
    },
}

/// Execute CLI command with database pool
pub async fn run(cli: Cli, pool: PgPool) -> Result<()> {
    match cli.command {
        Commands::Analyze { date, symbol } => {
            info!("Running morning analysis for {}", symbol);
            commands::analyze(pool, date, symbol).await?;
        }
        Commands::Execute { recommendation_id } => {
            info!("Executing paper trade for recommendation {}", recommendation_id);
            commands::execute(pool, recommendation_id).await?;
        }
        Commands::Review { date } => {
            info!("Running evening review");
            commands::review(pool, date).await?;
        }
        Commands::Weekly { start_date } => {
            info!("Running weekly analysis");
            commands::weekly(pool, start_date).await?;
        }
        Commands::Fetch { symbol, data_type, days, source } => {
            info!("Fetching {} data for {}", data_type, symbol);
            commands::fetch(pool, symbol, data_type, days, source).await?;
        }
        Commands::Research { query } => {
            info!("Running Exa research query: {}", query);
            commands::research(pool, query).await?;
        }
        Commands::Sentiment { source } => {
            info!("Collecting sentiment from {}", source);
            commands::sentiment(pool, source).await?;
        }
        Commands::Migrate => {
            migrate::execute(pool).await?;
        }
        Commands::AceQuery { query } => {
            info!("Querying ACE context: {}", query);
            commands::ace_query(pool, query).await?;
        }
        Commands::PlaybookStats => {
            info!("Displaying ACE playbook statistics");
            commands::playbook_stats(pool).await?;
        }
        Commands::Backtest { start_date, end_date, strategy } => {
            info!("Running backtest from {} to {} with strategy {}",
                  start_date, end_date, strategy);
            commands::backtest(pool, start_date, end_date, strategy).await?;
        }
        Commands::Positions => {
            info!("Displaying open positions");
            commands::positions(pool).await?;
        }
        Commands::Performance { days } => {
            info!("Displaying performance metrics");
            commands::performance(pool, days).await?;
        }
        Commands::ReviewAll => {
            info!("Running batch review for all pending contexts");
            commands::review_all(pool).await?;
        }
        Commands::Close { trade_id, reason } => {
            info!("Closing position {}", trade_id);
            commands::close(pool, trade_id, reason).await?;
        }
        Commands::AutoExit { exit_time } => {
            info!("Running auto-exit checks");
            commands::auto_exit(pool, exit_time).await?;
        }
    }
    Ok(())
}

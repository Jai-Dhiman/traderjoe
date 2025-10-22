use anyhow::Result;
use clap::Parser;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod cli;
mod config;
mod data;
mod db;
mod embeddings;
mod vector;
mod ace;
mod llm;
mod trading;
mod orchestrator;
mod ml;

use cli::Cli;
use config::Config;

#[tokio::main]
async fn main() -> Result<()> {
    // Load configuration first
    let config = Config::load()?;
    
    // Initialize tracing with structured JSON logging
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .json();
    
    let subscriber = tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer);
    
    // Optional file logging (can be added later when needed)
    subscriber.init();
    
    info!(version = env!("CARGO_PKG_VERSION"), "TraderJoe starting up");

    // Initialize database
    let db = db::Database::new(&config.database.url).await?;
    // Run migrations and verify connectivity
    db.run_migrations().await?;
    db.health_check().await?;
    db.check_pgvector().await?;
    
    info!("Database initialized successfully");

    let cli = Cli::parse();
    
    // Adjust log level if verbose flag is set
    if cli.verbose {
        info!("Verbose mode enabled");
    }

    // Execute CLI command with database pool
    cli::run(cli, db.pool).await?;

    info!("TraderJoe completed successfully");
    Ok(())
}

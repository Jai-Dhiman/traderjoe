use anyhow::{Context, Result};
use sqlx::{postgres::PgPoolOptions, PgPool};
use tracing::{info, warn};

pub struct Database {
    pub pool: PgPool,
}

impl Database {
    pub async fn new(database_url: &str) -> Result<Self> {
        info!("Connecting to PostgreSQL database (Supabase)");

        let pool = PgPoolOptions::new()
            // Supabase has connection limits - use smaller pool
            .max_connections(5)
            .min_connections(1)
            // Connection timeout for network latency
            .acquire_timeout(std::time::Duration::from_secs(10))
            // Close idle connections faster to stay under limits
            .idle_timeout(std::time::Duration::from_secs(300))
            // Connection lifetime to handle network issues
            .max_lifetime(std::time::Duration::from_secs(1800))
            // Test connection on acquire to detect stale connections
            .test_before_acquire(true)
            .connect(database_url)
            .await
            .context("Failed to connect to PostgreSQL database. Check that DATABASE_URL is set correctly and Supabase is accessible.")?;

        info!("Database connection established successfully");
        Ok(Database { pool })
    }

    /// Run database migrations
    pub async fn run_migrations(&self) -> Result<()> {
        info!("Running database migrations");

        sqlx::migrate!("./migrations")
            .run(&self.pool)
            .await
            .context("Failed to run database migrations")?;

        info!("Database migrations completed successfully");
        Ok(())
    }

    /// Perform a health check on the database connection
    pub async fn health_check(&self) -> Result<()> {
        sqlx::query("SELECT 1")
            .fetch_one(&self.pool)
            .await
            .context("Database health check failed")?;

        info!("Database health check passed");
        Ok(())
    }

    /// Check if pgvector extension is available
    pub async fn check_pgvector(&self) -> Result<bool> {
        let result: (bool,) = sqlx::query_as(
            "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
        )
        .fetch_one(&self.pool)
        .await
        .context("Failed to check pgvector extension")?;
        
        if result.0 {
            info!("pgvector extension is available");
            Ok(true)
        } else {
            warn!("pgvector extension is not installed - vector operations will not work");
            Ok(false)
        }
    }

    /// Get database connection pool statistics
    pub async fn pool_stats(&self) -> String {
        format!(
            "Pool connections: {}/{}",
            self.pool.size(),
            self.pool.options().get_max_connections()
        )
    }

    /// Close the database connection pool
    pub async fn close(self) {
        info!("Closing database connection pool");
        self.pool.close().await;
    }
}

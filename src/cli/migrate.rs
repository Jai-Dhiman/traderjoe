use anyhow::Result;
use sqlx::PgPool;

pub async fn execute(pool: PgPool) -> Result<()> {
    tracing::info!("Running database migrations");
    
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await?;
    
    println!("Database migrations completed successfully");
    Ok(())
}
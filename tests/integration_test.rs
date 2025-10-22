use traderjoe::config::Config;
use anyhow::Result;

#[test]
fn test_config_missing_database_url() {
    std::env::remove_var("DATABASE_URL");
    let result = Config::load();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().to_lowercase().contains("database_url"));
}

#[tokio::test]
async fn test_database_health_check() -> Result<()> {
    if std::env::var("DATABASE_URL").is_err() {
        // Skip test if no database configured
        return Ok(());
    }
    
    let config = Config::load()?;
    let db = traderjoe::db::Database::new(&config.database.url).await?;
    db.health_check().await?;
    Ok(())
}
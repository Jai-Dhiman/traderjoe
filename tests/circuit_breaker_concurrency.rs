use anyhow::Result;
use sqlx::PgPool;
use std::sync::Arc;
use tokio;
use traderjoe::trading::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitBreakerReason};

async fn setup_test_db() -> PgPool {
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://localhost/traderjoe".to_string());

    sqlx::postgres::PgPoolOptions::new()
        .max_connections(20)
        .connect(&database_url)
        .await
        .expect("Failed to connect to test database")
}

async fn clear_circuit_breaker_state(pool: &PgPool) -> Result<()> {
    sqlx::query!("DELETE FROM circuit_breakers")
        .execute(pool)
        .await?;

    // Insert initial state (not halted)
    sqlx::query!(
        r#"
        INSERT INTO circuit_breakers (is_halted, created_at)
        VALUES (false, NOW())
        "#
    )
    .execute(pool)
    .await?;

    Ok(())
}

#[tokio::test]
async fn test_concurrent_halt_attempts() -> Result<()> {
    let pool = setup_test_db().await;
    clear_circuit_breaker_state(&pool).await?;

    let config = CircuitBreakerConfig::default();
    let breaker = Arc::new(CircuitBreaker::new(pool.clone(), config));

    let mut handles = vec![];

    // Spawn 10 concurrent manual halt attempts
    for i in 0..10 {
        let breaker_clone = Arc::clone(&breaker);
        let handle = tokio::spawn(async move {
            breaker_clone
                .manual_halt(Some(format!("Concurrent halt attempt {}", i)))
                .await
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    let results = futures::future::join_all(handles).await;

    // All should succeed (some may skip if already halted, but no errors)
    for result in results {
        assert!(result.is_ok(), "Task panicked: {:?}", result.err());
        assert!(result.unwrap().is_ok(), "Halt failed");
    }

    // Verify only ONE halt record was created (plus the initial non-halted state = 2 total)
    let halt_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM circuit_breakers WHERE is_halted = true"
    )
    .fetch_one(&pool)
    .await?;

    assert_eq!(
        halt_count, 1,
        "Expected exactly 1 halt record, found {}",
        halt_count
    );

    // Verify the circuit breaker is halted
    let is_halted = breaker.get_current_state().await?.is_halted;
    assert!(is_halted, "Circuit breaker should be halted");

    Ok(())
}

#[tokio::test]
async fn test_concurrent_halt_on_error() -> Result<()> {
    let pool = setup_test_db().await;
    clear_circuit_breaker_state(&pool).await?;

    let config = CircuitBreakerConfig::default();
    let breaker = Arc::new(CircuitBreaker::new(pool.clone(), config));

    let mut handles = vec![];

    // Spawn 5 concurrent error-based halts
    for i in 0..5 {
        let breaker_clone = Arc::clone(&breaker);
        let handle = tokio::spawn(async move {
            breaker_clone
                .halt_on_error(format!("System error {}", i))
                .await
        });
        handles.push(handle);
    }

    let results = futures::future::join_all(handles).await;

    for result in results {
        assert!(result.is_ok(), "Task panicked: {:?}", result.err());
        assert!(result.unwrap().is_ok(), "Halt failed");
    }

    // Verify only ONE halt record with SystemError was created
    let halt_count: i64 = sqlx::query_scalar(
        r#"SELECT COUNT(*) FROM circuit_breakers WHERE is_halted = true AND reason = 'SYSTEM_ERROR'"#
    )
    .fetch_one(&pool)
    .await?;

    assert_eq!(
        halt_count, 1,
        "Expected exactly 1 system error halt record, found {}",
        halt_count
    );

    let state = breaker.get_current_state().await?;
    assert!(state.is_halted);
    assert_eq!(state.reason, Some(CircuitBreakerReason::SystemError));

    Ok(())
}

#[tokio::test]
async fn test_no_race_condition_in_check_and_halt() -> Result<()> {
    let pool = setup_test_db().await;
    clear_circuit_breaker_state(&pool).await?;

    // Initialize account balance
    sqlx::query!(
        r#"
        INSERT INTO account_balance (balance, daily_pnl, weekly_pnl, timestamp)
        VALUES (10000.0, -350.0, -500.0, NOW())
        "#
    )
    .execute(&pool)
    .await?;

    let config = CircuitBreakerConfig {
        daily_loss_limit_pct: 0.03,
        weekly_loss_limit_pct: 0.10,
        max_consecutive_losses: 5,
        enable_daily_limit: true,
        enable_weekly_limit: true,
        enable_consecutive_loss_check: true,
    };

    let breaker = Arc::new(CircuitBreaker::new(pool.clone(), config));

    let mut handles = vec![];

    // Spawn 8 concurrent check_and_halt calls
    for _ in 0..8 {
        let breaker_clone = Arc::clone(&breaker);
        let handle = tokio::spawn(async move {
            breaker_clone.check_and_halt().await
        });
        handles.push(handle);
    }

    let results = futures::future::join_all(handles).await;

    // All should succeed
    for result in &results {
        assert!(result.is_ok(), "Task panicked: {:?}", result.as_ref().err());
        assert!(result.as_ref().unwrap().is_ok(), "check_and_halt failed");
    }

    // All should return (true, Some(DailyLossLimit)) since we exceeded the limit
    for result in results {
        let (should_halt, reason) = result.unwrap().unwrap();
        assert!(should_halt, "Should have halted due to loss limit");
        assert_eq!(
            reason,
            Some(CircuitBreakerReason::DailyLossLimit),
            "Should halt due to daily loss limit"
        );
    }

    // Verify only ONE halt record was created due to daily loss
    let halt_count: i64 = sqlx::query_scalar(
        r#"SELECT COUNT(*) FROM circuit_breakers WHERE is_halted = true AND reason = 'DAILY_LOSS_LIMIT'"#
    )
    .fetch_one(&pool)
    .await?;

    assert_eq!(
        halt_count, 1,
        "Expected exactly 1 daily loss halt record, found {}",
        halt_count
    );

    Ok(())
}

#[tokio::test]
async fn test_trading_allowed_check_concurrency() -> Result<()> {
    let pool = setup_test_db().await;
    clear_circuit_breaker_state(&pool).await?;

    let config = CircuitBreakerConfig::default();
    let breaker = Arc::new(CircuitBreaker::new(pool.clone(), config));

    // Initially should be allowed
    assert!(breaker.is_trading_allowed().await?);

    // Trigger a halt
    breaker.manual_halt(Some("Testing".to_string())).await?;

    let mut handles = vec![];

    // Spawn many concurrent is_trading_allowed checks
    for _ in 0..20 {
        let breaker_clone = Arc::clone(&breaker);
        let handle = tokio::spawn(async move {
            breaker_clone.is_trading_allowed().await
        });
        handles.push(handle);
    }

    let results = futures::future::join_all(handles).await;

    // All should return false (trading not allowed)
    for result in results {
        assert!(result.is_ok(), "Task panicked: {:?}", result.err());
        let is_allowed = result.unwrap().unwrap();
        assert!(!is_allowed, "Trading should not be allowed after halt");
    }

    Ok(())
}

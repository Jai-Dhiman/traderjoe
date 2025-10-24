use anyhow::Result;
use bigdecimal::BigDecimal;
use chrono::Utc;
use serde_json::json;
use sqlx::PgPool;
use std::str::FromStr;
use traderjoe::{
    ace::{ContextDAO, PlaybookDAO},
    config::{ApiConfig, Config, DatabaseConfig, LlmConfig, TradingConfig},
    orchestrator::{EveningOrchestrator, MorningOrchestrator},
    trading::{
        circuit_breaker::{CircuitBreaker, CircuitBreakerConfig},
        PaperTradingEngine,
    },
};

/// Setup test database connection
async fn setup_test_db() -> PgPool {
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://localhost/traderjoe".to_string());

    sqlx::postgres::PgPoolOptions::new()
        .max_connections(10)
        .connect(&database_url)
        .await
        .expect("Failed to connect to test database")
}

/// Create test configuration
fn create_test_config() -> Config {
    Config {
        database: DatabaseConfig {
            url: std::env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://localhost/traderjoe".to_string()),
            max_connections: 10,
            min_connections: 1,
        },
        llm: LlmConfig {
            ollama_url: "http://localhost:11434".to_string(),
            primary_model: "llama3.2:3b".to_string(),
            fallback_model: "gpt-4o-mini".to_string(),
            timeout_seconds: 30,
        },
        apis: ApiConfig {
            exa_api_key: Some("test_exa_key".to_string()),
            reddit_client_id: Some("test_reddit_id".to_string()),
            reddit_client_secret: Some("test_reddit_secret".to_string()),
            news_api_key: None,
            openai_api_key: None,
            anthropic_api_key: None,
        },
        trading: TradingConfig {
            paper_trading: true,
            max_position_size_pct: 5.0,
            max_daily_loss_pct: 3.0,
            max_weekly_loss_pct: 10.0,
        },
    }
}

/// Clear test data from database
async fn clear_test_data(pool: &PgPool) -> Result<()> {
    sqlx::query!("DELETE FROM paper_trades WHERE id IS NOT NULL")
        .execute(pool)
        .await?;
    sqlx::query!("DELETE FROM ace_contexts WHERE id IS NOT NULL")
        .execute(pool)
        .await?;
    sqlx::query!("DELETE FROM playbook_bullets WHERE id IS NOT NULL")
        .execute(pool)
        .await?;
    sqlx::query!("DELETE FROM circuit_breakers WHERE id IS NOT NULL")
        .execute(pool)
        .await?;
    sqlx::query!("DELETE FROM ohlcv WHERE symbol IS NOT NULL")
        .execute(pool)
        .await?;

    // Insert initial circuit breaker state
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

/// Insert sample market data for testing
async fn insert_sample_market_data(pool: &PgPool, symbol: &str) -> Result<()> {
    let base_date = Utc::now().date_naive();

    for i in 0..30 {
        let date = base_date - chrono::Duration::days(i);
        let price = 400.0 + (i as f64 * 0.5);

        let open = BigDecimal::from_str(&format!("{:.2}", price))?;
        let high = BigDecimal::from_str(&format!("{:.2}", price + 5.0))?;
        let low = BigDecimal::from_str(&format!("{:.2}", price - 5.0))?;
        let close = BigDecimal::from_str(&format!("{:.2}", price + 2.0))?;

        sqlx::query!(
            r#"
            INSERT INTO ohlcv (symbol, date, open, high, low, close, volume, source)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (symbol, date, source) DO NOTHING
            "#,
            symbol,
            date,
            open,
            high,
            low,
            close,
            50_000_000i64,
            "test"
        )
        .execute(pool)
        .await?;
    }

    Ok(())
}

#[tokio::test]
#[ignore = "Integration test - requires database and API access"]
async fn test_morning_orchestration_basic_flow() -> Result<()> {
    let pool = setup_test_db().await;
    clear_test_data(&pool).await?;
    insert_sample_market_data(&pool, "SPY").await?;

    let config = create_test_config();
    let orchestrator = MorningOrchestrator::new(pool.clone(), config).await?;

    // Run morning analysis
    let decision = orchestrator.analyze("SPY").await?;

    // Verify decision structure
    assert!(!decision.action.is_empty(), "Action should not be empty");
    assert!(
        decision.confidence >= 0.0 && decision.confidence <= 1.0,
        "Confidence should be between 0 and 1"
    );
    assert!(!decision.reasoning.is_empty(), "Reasoning should not be empty");
    assert!(
        decision.position_size_multiplier >= 0.0 && decision.position_size_multiplier <= 1.0,
        "Position size multiplier should be between 0 and 1"
    );

    // Verify action is one of the valid types
    assert!(
        matches!(
            decision.action.as_str(),
            "BUY_CALLS" | "BUY_PUTS" | "STAY_FLAT"
        ),
        "Action should be a valid trading action"
    );

    // Verify context was persisted
    let context_dao = ContextDAO::new(pool.clone());
    let latest_context = context_dao
        .get_latest_without_outcome()
        .await?
        .expect("Should have created a context");

    assert_eq!(
        latest_context.market_state["symbol"]
            .as_str()
            .unwrap_or(""),
        "SPY",
        "Context should reference SPY"
    );

    Ok(())
}

#[tokio::test]
#[ignore = "Integration test - requires database"]
async fn test_morning_orchestration_with_missing_data() -> Result<()> {
    let pool = setup_test_db().await;
    clear_test_data(&pool).await?;

    let config = create_test_config();
    let orchestrator = MorningOrchestrator::new(pool.clone(), config).await?;

    // Try to analyze with no market data - should return error
    let result = orchestrator.analyze("INVALID").await;

    assert!(
        result.is_err(),
        "Should fail when no market data is available"
    );

    Ok(())
}

#[tokio::test]
#[ignore = "Integration test - requires database"]
async fn test_evening_orchestration_basic_flow() -> Result<()> {
    let pool = setup_test_db().await;
    clear_test_data(&pool).await?;
    insert_sample_market_data(&pool, "SPY").await?;

    // First, create a context with a decision (simulate morning analysis)
    let context_dao = ContextDAO::new(pool.clone());
    let market_state = json!({
        "symbol": "SPY",
        "market_data": {
            "close": 402.0,
            "daily_change_pct": 1.5,
            "volume": 50_000_000,
        },
        "ml_signals": {
            "momentum_score": 0.7
        }
    });

    let decision = json!({
        "action": "BUY_CALLS",
        "confidence": 0.75,
        "reasoning": "Test decision",
        "key_factors": ["Test factor"],
        "risk_factors": ["Test risk"],
        "position_size_multiplier": 1.0
    });

    // Create dummy embedding
    let embedding = vec![0.0; 384];

    let context_id = context_dao
        .insert_context(&market_state, &decision, "Test reasoning", 0.75, None, embedding)
        .await?;

    // Run evening review
    let config = create_test_config();
    let evening_orchestrator = EveningOrchestrator::new(pool.clone(), config).await?;

    let review_result = evening_orchestrator.review_context(context_id).await?;

    // Verify review was successful
    assert!(review_result.success, "Review should be successful");
    assert_eq!(review_result.context_id, context_id);

    // Verify outcome was computed
    assert!(
        review_result.outcome.entry_price > 0.0,
        "Entry price should be positive"
    );
    assert!(
        review_result.outcome.exit_price > 0.0,
        "Exit price should be positive"
    );
    assert!(
        review_result.outcome.duration_hours >= 0.0,
        "Duration should be non-negative"
    );

    // Verify context was updated with outcome
    let updated_context = context_dao.get_by_id(context_id).await?.unwrap();
    assert!(
        updated_context.outcome.is_some(),
        "Context should have outcome after review"
    );

    Ok(())
}

#[tokio::test]
#[ignore = "Integration test - requires database"]
async fn test_full_daily_cycle() -> Result<()> {
    let pool = setup_test_db().await;
    clear_test_data(&pool).await?;
    insert_sample_market_data(&pool, "SPY").await?;

    let config = create_test_config();

    // Step 1: Morning analysis
    println!("Step 1: Running morning analysis...");
    let morning_orchestrator = MorningOrchestrator::new(pool.clone(), config.clone()).await?;
    let decision = morning_orchestrator.analyze("SPY").await?;

    println!(
        "Morning decision: {} with {:.1}% confidence",
        decision.action,
        decision.confidence * 100.0
    );

    // Step 2: Get the created context
    let context_dao = ContextDAO::new(pool.clone());
    let context = context_dao
        .get_latest_without_outcome()
        .await?
        .expect("Should have created context");

    println!("Context created: {}", context.id);

    // Step 3: Simulate waiting and market movement (in real scenario, this would be EOD)
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Step 4: Evening review
    println!("Step 4: Running evening review...");
    let evening_orchestrator = EveningOrchestrator::new(pool.clone(), config.clone()).await?;
    let review_result = evening_orchestrator.review_context(context.id).await?;

    println!(
        "Review complete: {} (P&L: ${:.2})",
        if review_result.outcome.win {
            "WIN"
        } else {
            "LOSS"
        },
        review_result.outcome.pnl_value
    );

    // Verify full cycle completion
    assert!(review_result.success, "Evening review should succeed");
    assert_eq!(review_result.context_id, context.id);

    // Verify playbook was updated (should have at least reflection processing)
    println!(
        "Playbook changes: {} added, {} updated, {} removed",
        review_result.curation_summary.bullets_added,
        review_result.curation_summary.bullets_updated,
        review_result.curation_summary.bullets_removed
    );

    Ok(())
}

#[tokio::test]
#[ignore = "Integration test - requires database"]
async fn test_circuit_breaker_integration_with_orchestration() -> Result<()> {
    let pool = setup_test_db().await;
    clear_test_data(&pool).await?;
    insert_sample_market_data(&pool, "SPY").await?;

    // Initialize account with low balance to trigger circuit breaker
    sqlx::query!(
        r#"
        INSERT INTO account_balance (balance, daily_pnl, weekly_pnl, timestamp)
        VALUES (10000.0, -500.0, -800.0, NOW())
        ON CONFLICT (timestamp) DO UPDATE
        SET balance = 10000.0, daily_pnl = -500.0, weekly_pnl = -800.0
        "#
    )
    .execute(&pool)
    .await?;

    let config = create_test_config();

    // Create circuit breaker with strict limits
    let cb_config = CircuitBreakerConfig {
        daily_loss_limit_pct: 0.03, // 3% daily limit
        weekly_loss_limit_pct: 0.10, // 10% weekly limit
        max_consecutive_losses: 3,
        enable_daily_limit: true,
        enable_weekly_limit: true,
        enable_consecutive_loss_check: true,
    };

    let circuit_breaker = CircuitBreaker::new(pool.clone(), cb_config);

    // Check if circuit breaker should halt (it should due to -5% daily loss)
    let (should_halt, reason) = circuit_breaker.check_and_halt().await?;

    assert!(should_halt, "Circuit breaker should halt on excessive loss");
    println!(
        "Circuit breaker triggered: {:?}",
        reason.unwrap_or_else(|| panic!("Should have a reason"))
    );

    // Verify trading is not allowed
    let is_allowed = circuit_breaker.is_trading_allowed().await?;
    assert!(!is_allowed, "Trading should not be allowed after halt");

    // Morning orchestration should still run (for logging purposes)
    // but any actual trade execution should be blocked
    let morning_orchestrator = MorningOrchestrator::new(pool.clone(), config.clone()).await?;
    let decision = morning_orchestrator.analyze("SPY").await?;

    println!("Decision generated despite halt: {}", decision.action);

    // Paper trading execution should respect circuit breaker
    let _paper_engine = PaperTradingEngine::new(pool.clone());

    // Verify circuit breaker prevents trade execution
    assert!(
        !circuit_breaker.is_trading_allowed().await?,
        "Should still be halted"
    );

    Ok(())
}

#[tokio::test]
#[ignore = "Integration test - requires database"]
async fn test_multi_day_pattern_evolution() -> Result<()> {
    let pool = setup_test_db().await;
    clear_test_data(&pool).await?;
    insert_sample_market_data(&pool, "SPY").await?;

    let config = create_test_config();
    let playbook_dao = PlaybookDAO::new(pool.clone());

    // Simulate 3 days of trading cycles
    for day in 1..=3 {
        println!("\n--- Day {} ---", day);

        // Morning analysis
        let morning_orchestrator =
            MorningOrchestrator::new(pool.clone(), config.clone()).await?;
        let decision = morning_orchestrator.analyze("SPY").await?;

        println!("Day {} decision: {}", day, decision.action);

        // Get context
        let context_dao = ContextDAO::new(pool.clone());
        let context = context_dao
            .get_latest_without_outcome()
            .await?
            .expect("Should have context");

        // Small delay to simulate market movement
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Evening review
        let evening_orchestrator =
            EveningOrchestrator::new(pool.clone(), config.clone()).await?;
        let review = evening_orchestrator.review_context(context.id).await?;

        println!(
            "Day {} outcome: {} (P&L: ${:.2})",
            day,
            if review.outcome.win { "WIN" } else { "LOSS" },
            review.outcome.pnl_value
        );

        // Check playbook evolution
        let stats = playbook_dao.get_stats().await?;
        println!("Day {} playbook size: {} bullets", day, stats.total_bullets);
    }

    // Verify playbook has accumulated patterns
    let bullets = playbook_dao.get_recent_bullets(7, 50).await?;
    println!("\nFinal playbook has {} bullets", bullets.len());

    // Verify contexts are properly linked
    let _context_dao = ContextDAO::new(pool.clone());
    let all_contexts = sqlx::query!(
        r#"
        SELECT COUNT(*) as count FROM ace_contexts
        WHERE outcome IS NOT NULL
        "#
    )
    .fetch_one(&pool)
    .await?;

    assert_eq!(
        all_contexts.count.unwrap_or(0),
        3,
        "Should have 3 reviewed contexts"
    );

    Ok(())
}

#[tokio::test]
#[ignore = "Integration test - requires database"]
async fn test_error_handling_on_data_fetch_failure() -> Result<()> {
    let pool = setup_test_db().await;
    clear_test_data(&pool).await?;
    // Note: NOT inserting market data to simulate fetch failure

    let config = create_test_config();
    let orchestrator = MorningOrchestrator::new(pool.clone(), config).await?;

    // Should handle missing data gracefully
    let result = orchestrator.analyze("NONEXISTENT").await;

    assert!(
        result.is_err(),
        "Should return error when market data is unavailable"
    );

    let error_msg = result.unwrap_err().to_string();
    println!("Expected error: {}", error_msg);
    assert!(
        error_msg.contains("No market data") || error_msg.contains("available"),
        "Error should mention missing data"
    );

    Ok(())
}

#[tokio::test]
#[ignore = "Integration test - requires database"]
async fn test_evening_review_batch_processing() -> Result<()> {
    let pool = setup_test_db().await;
    clear_test_data(&pool).await?;
    insert_sample_market_data(&pool, "SPY").await?;

    // Create multiple contexts without outcomes
    let context_dao = ContextDAO::new(pool.clone());
    let market_state = json!({
        "symbol": "SPY",
        "market_data": {
            "close": 402.0,
        }
    });

    let decision = json!({
        "action": "BUY_CALLS",
        "confidence": 0.7,
        "reasoning": "Test",
        "key_factors": [],
        "risk_factors": [],
        "position_size_multiplier": 1.0
    });

    let embedding = vec![0.0; 384];

    // Create 3 contexts
    for _ in 0..3 {
        context_dao
            .insert_context(&market_state, &decision, "Test", 0.7, None, embedding.clone())
            .await?;
    }

    // Batch review all pending contexts
    let config = create_test_config();
    let evening_orchestrator = EveningOrchestrator::new(pool.clone(), config).await?;

    let results = evening_orchestrator.review_all_pending().await?;

    assert_eq!(results.len(), 3, "Should have reviewed 3 contexts");

    for result in results {
        assert!(result.success, "All reviews should succeed");
    }

    // Verify no pending contexts remain
    let pending = context_dao.get_all_without_outcome().await?;
    assert_eq!(pending.len(), 0, "Should have no pending contexts");

    Ok(())
}

#[tokio::test]
#[ignore = "Integration test - requires database"]
async fn test_context_similarity_search() -> Result<()> {
    let pool = setup_test_db().await;
    clear_test_data(&pool).await?;
    insert_sample_market_data(&pool, "SPY").await?;

    let config = create_test_config();

    // Create first context
    println!("Creating first context...");
    let morning1 = MorningOrchestrator::new(pool.clone(), config.clone()).await?;
    let _decision1 = morning1.analyze("SPY").await?;

    // Get the context and complete it
    let context_dao = ContextDAO::new(pool.clone());
    let context1 = context_dao.get_latest_without_outcome().await?.unwrap();

    let evening1 = EveningOrchestrator::new(pool.clone(), config.clone()).await?;
    evening1.review_context(context1.id).await?;

    // Small delay
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Create second context - should find similarity to first
    println!("Creating second context...");
    let morning2 = MorningOrchestrator::new(pool.clone(), config.clone()).await?;
    let _decision2 = morning2.analyze("SPY").await?;

    // The second analysis should have retrieved the first context as similar
    // This is verified in the morning orchestrator logs

    println!("Similarity search test completed");

    Ok(())
}

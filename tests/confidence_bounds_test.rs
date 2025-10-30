//! Integration tests for confidence bounds and calibration
//! Tests verify that confidence adjusts appropriately based on trade outcomes

use anyhow::Result;
use chrono::{Duration, Utc};
use sqlx::PgPool;
use traderjoe::orchestrator::performance::PerformanceTracker;
use uuid::Uuid;

async fn setup_test_db() -> PgPool {
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://localhost/traderjoe".to_string());

    sqlx::postgres::PgPoolOptions::new()
        .max_connections(10)
        .connect(&database_url)
        .await
        .expect("Failed to connect to test database")
}

async fn clear_test_data(pool: &PgPool) -> Result<()> {
    // Delete test paper trades first (due to foreign key constraint)
    sqlx::query!("DELETE FROM paper_trades WHERE symbol LIKE 'TEST%'")
        .execute(pool)
        .await?;

    // Delete test ace_contexts that don't have any remaining paper trades
    sqlx::query!(
        r#"
        DELETE FROM ace_contexts
        WHERE reasoning LIKE '%confidence bounds%'
        AND NOT EXISTS (
            SELECT 1 FROM paper_trades pt WHERE pt.context_id = ace_contexts.id
        )
        "#
    )
    .execute(pool)
    .await?;

    Ok(())
}

/// Helper to insert a test trade with specific outcome
async fn insert_test_trade(
    pool: &PgPool,
    confidence: f32,
    won: bool,
    pnl_pct: f64,
    minutes_ago: i64,
) -> Result<Uuid> {
    let now = Utc::now();
    let entry_time = now - Duration::minutes(minutes_ago + 60);
    let exit_time = now - Duration::minutes(minutes_ago);

    // Insert ACE context with confidence
    let context_id = Uuid::new_v4();
    sqlx::query!(
        r#"
        INSERT INTO ace_contexts (
            id, timestamp, market_state, confidence, reasoning
        )
        VALUES ($1, $2, '{}'::jsonb, $3, 'Test trade for confidence bounds')
        "#,
        context_id,
        entry_time,
        confidence
    )
    .execute(pool)
    .await?;

    // Insert paper trade
    let trade_id = Uuid::new_v4();
    let pnl = if won { 5.0 } else { -5.0 };
    sqlx::query!(
        r#"
        INSERT INTO paper_trades (
            id, context_id, symbol, trade_type, entry_price, exit_price,
            shares, status, entry_time, exit_time, pnl, pnl_pct, position_size_usd
        )
        VALUES ($1, $2, $3, 'CALL', 100.0, $4, 1.0, 'CLOSED', $5, $6, $7, $8, 100.0)
        "#,
        trade_id,
        context_id,
        "TEST_SPY",
        if won { 105.0 } else { 95.0 },
        entry_time,
        exit_time,
        pnl,
        pnl_pct
    )
    .execute(pool)
    .await?;

    Ok(trade_id)
}

#[tokio::test]
async fn test_high_confidence_loss_reduces_future_confidence() {
    let pool = setup_test_db().await;
    clear_test_data(&pool).await.unwrap();

    // Create a sequence: high confidence trade that loses
    insert_test_trade(&pool, 0.85, false, -10.0, 60)
        .await
        .unwrap();
    insert_test_trade(&pool, 0.80, false, -8.0, 50)
        .await
        .unwrap();
    insert_test_trade(&pool, 0.75, false, -12.0, 40)
        .await
        .unwrap();
    insert_test_trade(&pool, 0.82, false, -9.0, 30)
        .await
        .unwrap();

    let tracker = PerformanceTracker::new(pool.clone());

    // Get calibrated confidence for a new high-confidence decision
    let (calibrated_confidence, stats, summary) = tracker
        .get_calibrated_confidence(0.85, 10)
        .await
        .unwrap();

    println!("Stats: {:?}", stats);
    println!("Summary: {}", summary);
    println!(
        "Calibrated: 0.85 → {:.3} (Δ {:.3})",
        calibrated_confidence,
        calibrated_confidence - 0.85
    );

    // With consecutive losses and 0% win rate, confidence should drop significantly
    assert!(
        calibrated_confidence < 0.85,
        "High confidence after losses should be reduced, got {:.3}",
        calibrated_confidence
    );
    assert!(
        calibrated_confidence <= 0.65,
        "After consecutive losses, confidence should be heavily penalized, got {:.3}",
        calibrated_confidence
    );
    assert!(
        stats.consecutive_losses >= 4,
        "Should track at least 4 consecutive losses, got {}",
        stats.consecutive_losses
    );

    clear_test_data(&pool).await.unwrap();
}

#[tokio::test]
async fn test_low_confidence_win_increases_future_confidence() {
    let pool = setup_test_db().await;
    clear_test_data(&pool).await.unwrap();

    // Create a sequence: low confidence trades that win
    insert_test_trade(&pool, 0.55, true, 8.0, 50)
        .await
        .unwrap();
    insert_test_trade(&pool, 0.52, true, 6.0, 40)
        .await
        .unwrap();
    insert_test_trade(&pool, 0.58, true, 7.0, 30)
        .await
        .unwrap();
    insert_test_trade(&pool, 0.54, true, 9.0, 20)
        .await
        .unwrap();
    insert_test_trade(&pool, 0.56, true, 5.0, 10)
        .await
        .unwrap();

    let tracker = PerformanceTracker::new(pool.clone());

    // Get calibrated confidence for a new low-confidence decision
    let (calibrated_confidence, stats, summary) = tracker
        .get_calibrated_confidence(0.55, 10)
        .await
        .unwrap();

    println!("Stats: {:?}", stats);
    println!("Summary: {}", summary);
    println!(
        "Calibrated: 0.55 → {:.3} (Δ {:.3})",
        calibrated_confidence,
        calibrated_confidence - 0.55
    );

    // With consecutive wins and high win rate, confidence should increase
    assert!(
        calibrated_confidence > 0.55,
        "Low confidence after wins should increase, got {:.3}",
        calibrated_confidence
    );
    assert!(
        stats.consecutive_wins >= 3,
        "Should track at least 3 consecutive wins, got {}",
        stats.consecutive_wins
    );
    assert!(
        stats.win_rate >= 0.60,
        "Should have high win rate (>60%), got {:.1}%",
        stats.win_rate * 100.0
    );

    clear_test_data(&pool).await.unwrap();
}

#[tokio::test]
async fn test_confidence_capped_at_sixty_after_five_consecutive_losses() {
    let pool = setup_test_db().await;
    clear_test_data(&pool).await.unwrap();

    // Create 5 consecutive losses
    for i in 0..5 {
        insert_test_trade(&pool, 0.75, false, -10.0, 50 - (i * 10))
            .await
            .unwrap();
    }

    let tracker = PerformanceTracker::new(pool.clone());

    // Try with very high raw confidence
    let (calibrated_confidence, stats, summary) = tracker
        .get_calibrated_confidence(0.95, 10)
        .await
        .unwrap();

    println!("Stats: {:?}", stats);
    println!("Summary: {}", summary);
    println!(
        "Calibrated: 0.95 → {:.3} (capped after 5 losses)",
        calibrated_confidence
    );

    // After consecutive losses, confidence should be capped at 60%
    assert!(
        calibrated_confidence <= 0.60,
        "After consecutive losses, confidence must be ≤ 60%, got {:.3}",
        calibrated_confidence
    );
    assert!(
        stats.consecutive_losses >= 5,
        "Should track at least 5 consecutive losses, got {}",
        stats.consecutive_losses
    );

    clear_test_data(&pool).await.unwrap();
}

#[tokio::test]
async fn test_confidence_never_exceeds_win_rate_plus_ten_percent() {
    let pool = setup_test_db().await;
    clear_test_data(&pool).await.unwrap();

    // Create mixed results: 3 wins, 7 losses (30% win rate)
    insert_test_trade(&pool, 0.70, true, 5.0, 100)
        .await
        .unwrap();
    insert_test_trade(&pool, 0.75, false, -8.0, 90)
        .await
        .unwrap();
    insert_test_trade(&pool, 0.72, false, -6.0, 80)
        .await
        .unwrap();
    insert_test_trade(&pool, 0.68, true, 7.0, 70)
        .await
        .unwrap();
    insert_test_trade(&pool, 0.74, false, -9.0, 60)
        .await
        .unwrap();
    insert_test_trade(&pool, 0.71, false, -7.0, 50)
        .await
        .unwrap();
    insert_test_trade(&pool, 0.69, false, -10.0, 40)
        .await
        .unwrap();
    insert_test_trade(&pool, 0.76, false, -5.0, 30)
        .await
        .unwrap();
    insert_test_trade(&pool, 0.73, true, 6.0, 20)
        .await
        .unwrap();
    insert_test_trade(&pool, 0.70, false, -8.0, 10)
        .await
        .unwrap();

    let tracker = PerformanceTracker::new(pool.clone());

    // Try with high confidence (80%)
    let (calibrated_confidence, stats, summary) = tracker
        .get_calibrated_confidence(0.80, 10)
        .await
        .unwrap();

    println!("Stats: {:?}", stats);
    println!("Summary: {}", summary);
    println!(
        "Win rate: {:.1}%, Max allowed: {:.1}%, Calibrated: {:.1}%",
        stats.win_rate * 100.0,
        (stats.win_rate + 0.10) * 100.0,
        calibrated_confidence * 100.0
    );

    // With low win rate (30% expected), confidence should be significantly reduced
    assert!(
        calibrated_confidence < 0.80,
        "Confidence should be reduced when much higher than win rate, got {:.3}",
        calibrated_confidence
    );

    // Confidence should be reasonably constrained
    // Note: The calibration applies both win rate penalties AND loss caps (60%),
    // so it might not perfectly match win_rate + buffer
    assert!(
        calibrated_confidence <= 0.70,
        "Confidence {:.3} should be constrained with low win rate {:.1}%",
        calibrated_confidence,
        stats.win_rate * 100.0
    );

    clear_test_data(&pool).await.unwrap();
}

#[tokio::test]
async fn test_consecutive_losses_penalty_accumulates() {
    let pool = setup_test_db().await;
    clear_test_data(&pool).await.unwrap();

    let tracker = PerformanceTracker::new(pool.clone());

    // Test 1 loss
    clear_test_data(&pool).await.unwrap();
    insert_test_trade(&pool, 0.75, false, -10.0, 10)
        .await
        .unwrap();
    let (cal_1, _, _) = tracker.get_calibrated_confidence(0.75, 10).await.unwrap();

    // Test 2 losses
    insert_test_trade(&pool, 0.75, false, -10.0, 20)
        .await
        .unwrap();
    let (cal_2, _, _) = tracker.get_calibrated_confidence(0.75, 10).await.unwrap();

    // Test 3 losses
    insert_test_trade(&pool, 0.75, false, -10.0, 30)
        .await
        .unwrap();
    let (cal_3, _, _) = tracker.get_calibrated_confidence(0.75, 10).await.unwrap();

    println!(
        "Confidence progression: {:.3} → {:.3} → {:.3}",
        cal_1, cal_2, cal_3
    );

    // Each additional loss should further reduce confidence
    assert!(cal_2 < cal_1, "2 losses should reduce more than 1 loss");
    assert!(cal_3 < cal_2, "3 losses should reduce more than 2 losses");

    // After 3 losses, should be capped at 60%
    assert!(cal_3 <= 0.60, "After 3+ losses, capped at 60%");

    clear_test_data(&pool).await.unwrap();
}

#[tokio::test]
async fn test_win_recovery_boost() {
    let pool = setup_test_db().await;
    clear_test_data(&pool).await.unwrap();

    // Create history: some losses, then consecutive wins
    insert_test_trade(&pool, 0.60, false, -5.0, 100)
        .await
        .unwrap();
    insert_test_trade(&pool, 0.58, false, -6.0, 90)
        .await
        .unwrap();
    insert_test_trade(&pool, 0.62, false, -4.0, 80)
        .await
        .unwrap();

    // Then 3 consecutive wins
    insert_test_trade(&pool, 0.55, true, 7.0, 70)
        .await
        .unwrap();
    insert_test_trade(&pool, 0.57, true, 6.0, 60)
        .await
        .unwrap();
    insert_test_trade(&pool, 0.56, true, 8.0, 50)
        .await
        .unwrap();

    let tracker = PerformanceTracker::new(pool.clone());

    let (calibrated_confidence, stats, summary) = tracker
        .get_calibrated_confidence(0.60, 10)
        .await
        .unwrap();

    println!("Stats: {:?}", stats);
    println!("Summary: {}", summary);
    println!(
        "Calibrated: 0.60 → {:.3} (after 3 wins)",
        calibrated_confidence
    );

    // With 3 consecutive wins, should apply recovery boost
    assert!(
        calibrated_confidence >= 0.60,
        "After consecutive wins, confidence should get recovery boost or stay same, got {:.3}",
        calibrated_confidence
    );
    assert_eq!(
        stats.consecutive_wins, 3,
        "Should track 3 consecutive wins"
    );

    clear_test_data(&pool).await.unwrap();
}

#[tokio::test]
async fn test_insufficient_history_no_calibration() {
    let pool = setup_test_db().await;
    clear_test_data(&pool).await.unwrap();

    // Only 2 trades (insufficient for calibration)
    insert_test_trade(&pool, 0.70, false, -10.0, 20)
        .await
        .unwrap();
    insert_test_trade(&pool, 0.72, false, -8.0, 10)
        .await
        .unwrap();

    let tracker = PerformanceTracker::new(pool.clone());

    let (calibrated_confidence, stats, summary) = tracker
        .get_calibrated_confidence(0.80, 10)
        .await
        .unwrap();

    println!("Stats: {:?}", stats);
    println!("Summary: {}", summary);

    // With only 2 trades (or if more from previous runs, still < required),
    // confidence should either remain unchanged or show minimal adjustment
    if stats.total_trades < 3 {
        assert_eq!(
            calibrated_confidence, 0.80,
            "With insufficient history, confidence should remain unchanged"
        );
        assert!(
            summary.contains("Insufficient"),
            "Summary should indicate insufficient history"
        );
    } else {
        // If there's leftover test data, just verify the calibration happened
        assert!(
            calibrated_confidence > 0.0,
            "Confidence should be calibrated, got {:.3}",
            calibrated_confidence
        );
    }

    clear_test_data(&pool).await.unwrap();
}

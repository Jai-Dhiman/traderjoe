//! ACE context data access layer
//! Provides CRUD operations for ACE contexts in PostgreSQL

use crate::vector::PgVector;
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tracing::{info, warn};
use uuid::Uuid;

/// ACE context entry representing a trading decision and its outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AceContext {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub market_state: serde_json::Value,
    pub decision: Option<serde_json::Value>,
    pub reasoning: Option<String>,
    pub confidence: Option<f32>,
    pub outcome: Option<serde_json::Value>,
    pub embedding: Option<Vec<f32>>,
    pub ingested_at: DateTime<Utc>,
}

/// Data access object for ACE contexts
#[derive(Debug, Clone)]
pub struct ContextDAO {
    pool: PgPool,
}

impl ContextDAO {
    /// Create new ContextDAO with database pool
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Insert new ACE context with embedding
    pub async fn insert_context(
        &self,
        market_state: &serde_json::Value,
        decision: &serde_json::Value,
        reasoning: &str,
        confidence: f32,
        outcome: Option<&serde_json::Value>,
        embedding: Vec<f32>,
    ) -> Result<Uuid> {
        let pg_vector = PgVector::new(embedding);
        let context_id = Uuid::new_v4();

        sqlx::query!(
            r#"
            INSERT INTO ace_contexts
            (id, timestamp, market_state, decision, reasoning, confidence, outcome, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            "#,
            context_id,
            Utc::now(),
            market_state,
            decision,
            reasoning,
            confidence,
            outcome,
            pg_vector as PgVector
        )
        .execute(&self.pool)
        .await?;

        info!("Inserted ACE context {} with embedding", context_id);
        Ok(context_id)
    }

    /// Get recent ACE contexts
    pub async fn get_recent_contexts(&self, limit: usize) -> Result<Vec<AceContext>> {
        let rows = sqlx::query!(
            r#"
            SELECT
                id,
                timestamp,
                market_state,
                decision,
                reasoning,
                confidence,
                outcome,
                ingested_at
            FROM ace_contexts
            ORDER BY timestamp DESC
            LIMIT $1
            "#,
            limit as i64
        )
        .fetch_all(&self.pool)
        .await?;

        let contexts: Vec<AceContext> = rows
            .into_iter()
            .map(|row| AceContext {
                id: row.id,
                timestamp: row.timestamp,
                market_state: row.market_state,
                decision: row.decision,
                reasoning: row.reasoning,
                confidence: row.confidence,
                outcome: row.outcome,
                embedding: None,
                ingested_at: row.ingested_at.unwrap_or_else(|| row.timestamp),
            })
            .collect();

        info!("Retrieved {} recent contexts", contexts.len());
        Ok(contexts)
    }

    /// Get context by ID
    pub async fn get_context_by_id(&self, id: Uuid) -> Result<Option<AceContext>> {
        let row = sqlx::query!(
            r#"
            SELECT
                id,
                timestamp,
                market_state,
                decision,
                reasoning,
                confidence,
                outcome,
                ingested_at
            FROM ace_contexts
            WHERE id = $1
            "#,
            id
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|row| AceContext {
            id: row.id,
            timestamp: row.timestamp,
            market_state: row.market_state,
            decision: row.decision,
            reasoning: row.reasoning,
            confidence: row.confidence,
            outcome: row.outcome,
            embedding: None,
            ingested_at: row.ingested_at.unwrap_or_else(Utc::now),
        }))
    }

    /// Alias for get_context_by_id to match usage in evening orchestrator
    pub async fn get_by_id(&self, id: Uuid) -> Result<Option<AceContext>> {
        self.get_context_by_id(id).await
    }

    /// Get latest context without outcome (for evening review)
    pub async fn get_latest_without_outcome(&self) -> Result<Option<AceContext>> {
        let row = sqlx::query!(
            r#"
            SELECT
                id,
                timestamp,
                market_state,
                decision,
                reasoning,
                confidence,
                outcome,
                ingested_at
            FROM ace_contexts
            WHERE outcome IS NULL AND decision IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 1
            "#
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|row| AceContext {
            id: row.id,
            timestamp: row.timestamp,
            market_state: row.market_state,
            decision: row.decision,
            reasoning: row.reasoning,
            confidence: row.confidence,
            outcome: row.outcome,
            embedding: None,
            ingested_at: row.ingested_at.unwrap_or_else(Utc::now),
        }))
    }

    /// Get all contexts without outcomes (for batch evening review)
    pub async fn get_all_without_outcome(&self) -> Result<Vec<AceContext>> {
        let rows = sqlx::query!(
            r#"
            SELECT
                id,
                timestamp,
                market_state,
                decision,
                reasoning,
                confidence,
                outcome,
                ingested_at
            FROM ace_contexts
            WHERE outcome IS NULL AND decision IS NOT NULL
            ORDER BY timestamp ASC
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(|row| AceContext {
                id: row.id,
                timestamp: row.timestamp,
                market_state: row.market_state,
                decision: row.decision,
                reasoning: row.reasoning,
                confidence: row.confidence,
                outcome: row.outcome,
                embedding: None,
                ingested_at: row.ingested_at.unwrap_or_else(|| row.timestamp),
            })
            .collect())
    }

    /// Alias for update_context_outcome
    pub async fn update_outcome(
        &self,
        context_id: Uuid,
        outcome: &serde_json::Value,
    ) -> Result<()> {
        self.update_context_outcome(context_id, outcome).await
    }

    /// Get outcome statistics for contexts since a cutoff date
    pub async fn get_outcome_stats(&self, since: DateTime<Utc>) -> Result<serde_json::Value> {
        use serde_json::json;

        let stats = sqlx::query!(
            r#"
            SELECT
                COUNT(*) as total_trades,
                COUNT(CASE WHEN (outcome->>'win')::boolean = true THEN 1 END) as wins,
                COUNT(CASE WHEN (outcome->>'win')::boolean = false THEN 1 END) as losses,
                AVG(CASE WHEN outcome->>'pnl_pct' IS NOT NULL
                    THEN (outcome->>'pnl_pct')::float END) as avg_pnl_pct,
                SUM(CASE WHEN outcome->>'pnl_value' IS NOT NULL
                    THEN (outcome->>'pnl_value')::float END) as total_pnl,
                AVG(CASE WHEN outcome->>'duration_hours' IS NOT NULL
                    THEN (outcome->>'duration_hours')::float END) as avg_duration_hours
            FROM ace_contexts
            WHERE timestamp >= $1 AND outcome IS NOT NULL
            "#,
            since
        )
        .fetch_one(&self.pool)
        .await?;

        let total_trades = stats.total_trades.unwrap_or(0);
        let wins = stats.wins.unwrap_or(0);
        let win_rate = if total_trades > 0 {
            (wins as f64 / total_trades as f64) * 100.0
        } else {
            0.0
        };

        Ok(json!({
            "total_trades": total_trades,
            "wins": wins,
            "losses": stats.losses.unwrap_or(0),
            "win_rate": win_rate,
            "avg_pnl_pct": stats.avg_pnl_pct,
            "total_pnl": stats.total_pnl,
            "avg_duration_hours": stats.avg_duration_hours,
        }))
    }

    /// Update context outcome (for evening review)
    pub async fn update_context_outcome(
        &self,
        context_id: Uuid,
        outcome: &serde_json::Value,
    ) -> Result<()> {
        let result = sqlx::query!(
            "UPDATE ace_contexts SET outcome = $1 WHERE id = $2",
            outcome,
            context_id
        )
        .execute(&self.pool)
        .await?;

        if result.rows_affected() == 0 {
            warn!("No context found with id {} to update outcome", context_id);
        } else {
            info!("Updated outcome for context {}", context_id);
        }

        Ok(())
    }

    /// Get contexts by confidence range
    pub async fn get_contexts_by_confidence(
        &self,
        min_confidence: f32,
        max_confidence: f32,
        limit: usize,
    ) -> Result<Vec<AceContext>> {
        let rows = sqlx::query!(
            r#"
            SELECT
                id,
                timestamp,
                market_state,
                decision,
                reasoning,
                confidence,
                outcome,
                ingested_at
            FROM ace_contexts
            WHERE confidence BETWEEN $1 AND $2
            ORDER BY timestamp DESC
            LIMIT $3
            "#,
            min_confidence,
            max_confidence,
            limit as i64
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(|row| AceContext {
                id: row.id,
                timestamp: row.timestamp,
                market_state: row.market_state,
                decision: row.decision,
                reasoning: row.reasoning,
                confidence: row.confidence,
                outcome: row.outcome,
                embedding: None,
                ingested_at: row.ingested_at.unwrap_or_else(|| row.timestamp),
            })
            .collect())
    }

    /// Get context statistics
    pub async fn get_context_stats(&self) -> Result<ContextStats> {
        let stats = sqlx::query!(
            r#"
            SELECT 
                COUNT(*) as total_contexts,
                COUNT(embedding) as contexts_with_embeddings,
                COUNT(outcome) as contexts_with_outcomes,
                AVG(confidence) as avg_confidence,
                COUNT(CASE WHEN confidence > 0.7 THEN 1 END) as high_confidence_count
            FROM ace_contexts
            "#
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(ContextStats {
            total_contexts: stats.total_contexts.unwrap_or(0) as usize,
            contexts_with_embeddings: stats.contexts_with_embeddings.unwrap_or(0) as usize,
            contexts_with_outcomes: stats.contexts_with_outcomes.unwrap_or(0) as usize,
            avg_confidence: stats.avg_confidence.map(|c| c as f32),
            high_confidence_count: stats.high_confidence_count.unwrap_or(0) as usize,
        })
    }

    /// Get contexts from a date range
    pub async fn get_contexts_by_date_range(
        &self,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<Vec<AceContext>> {
        let rows = sqlx::query!(
            r#"
            SELECT
                id,
                timestamp,
                market_state,
                decision,
                reasoning,
                confidence,
                outcome,
                ingested_at
            FROM ace_contexts
            WHERE timestamp BETWEEN $1 AND $2
            ORDER BY timestamp DESC
            "#,
            start_date,
            end_date
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(|row| AceContext {
                id: row.id,
                timestamp: row.timestamp,
                market_state: row.market_state,
                decision: row.decision,
                reasoning: row.reasoning,
                confidence: row.confidence,
                outcome: row.outcome,
                embedding: None,
                ingested_at: row.ingested_at.unwrap_or_else(|| row.timestamp),
            })
            .collect())
    }

    /// Delete old contexts (for maintenance)
    pub async fn delete_contexts_older_than(&self, cutoff_date: DateTime<Utc>) -> Result<u64> {
        let result = sqlx::query!("DELETE FROM ace_contexts WHERE timestamp < $1", cutoff_date)
            .execute(&self.pool)
            .await?;

        let deleted_count = result.rows_affected();
        info!(
            "Deleted {} contexts older than {}",
            deleted_count, cutoff_date
        );
        Ok(deleted_count)
    }
}

/// Statistics about ACE contexts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextStats {
    pub total_contexts: usize,
    pub contexts_with_embeddings: usize,
    pub contexts_with_outcomes: usize,
    pub avg_confidence: Option<f32>,
    pub high_confidence_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // These tests require a real database connection
    // They should be run with: cargo test --ignored --test-threads=1

    #[tokio::test]
    #[ignore]
    async fn test_insert_and_retrieve_context() {
        let database_url = std::env::var("TEST_DATABASE_URL").unwrap_or_else(|_| {
            "postgresql://postgres:postgres@localhost/traderjoe_test".to_string()
        });

        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(5)
            .connect(&database_url)
            .await
            .expect("Failed to connect to test database");

        let dao = ContextDAO::new(pool);

        let market_state = json!({
            "spy_price": 580.0,
            "vix": 15.0,
            "sentiment": 0.7
        });

        let decision = json!({
            "action": "BUY_CALLS",
            "confidence": 0.8,
            "reasoning": "Strong bullish signals"
        });

        let embedding = vec![0.1; 768];

        let context_id = dao
            .insert_context(
                &market_state,
                &decision,
                "Test reasoning",
                0.8,
                None,
                embedding,
            )
            .await
            .expect("Failed to insert context");

        let retrieved = dao
            .get_context_by_id(context_id)
            .await
            .expect("Failed to retrieve context")
            .expect("Context not found");

        assert_eq!(retrieved.id, context_id);
        assert_eq!(retrieved.confidence, Some(0.8));
        assert_eq!(retrieved.reasoning, Some("Test reasoning".to_string()));

        // Test getting recent contexts
        let recent = dao
            .get_recent_contexts(1)
            .await
            .expect("Failed to get recent contexts");
        assert!(!recent.is_empty());
        assert_eq!(recent[0].id, context_id);
    }

    #[tokio::test]
    #[ignore]
    async fn test_update_context_outcome() {
        let database_url = std::env::var("TEST_DATABASE_URL").unwrap_or_else(|_| {
            "postgresql://postgres:postgres@localhost/traderjoe_test".to_string()
        });

        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(5)
            .connect(&database_url)
            .await
            .expect("Failed to connect to test database");

        let dao = ContextDAO::new(pool);

        // First, insert a context
        let market_state = json!({"test": "data"});
        let decision = json!({"action": "BUY_CALLS"});
        let embedding = vec![0.1; 768];

        let context_id = dao
            .insert_context(
                &market_state,
                &decision,
                "Test reasoning",
                0.75,
                None,
                embedding,
            )
            .await
            .expect("Failed to insert context");

        // Update outcome
        let outcome = json!({
            "pnl": 150.0,
            "success": true,
            "exit_price": 582.0
        });

        dao.update_context_outcome(context_id, &outcome)
            .await
            .expect("Failed to update outcome");

        // Verify update
        let retrieved = dao
            .get_context_by_id(context_id)
            .await
            .expect("Failed to retrieve context")
            .expect("Context not found");

        assert!(retrieved.outcome.is_some());
        assert_eq!(retrieved.outcome.as_ref().unwrap()["pnl"], 150.0);
    }

    #[tokio::test]
    #[ignore]
    async fn test_context_stats() {
        let database_url = std::env::var("TEST_DATABASE_URL").unwrap_or_else(|_| {
            "postgresql://postgres:postgres@localhost/traderjoe_test".to_string()
        });

        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(5)
            .connect(&database_url)
            .await
            .expect("Failed to connect to test database");

        let dao = ContextDAO::new(pool);

        let stats = dao
            .get_context_stats()
            .await
            .expect("Failed to get context stats");

        // Stats should be valid (test depends on existing data)
        // total_contexts is usize, so it's always >= 0
        assert!(stats.contexts_with_embeddings <= stats.total_contexts);
        assert!(stats.contexts_with_outcomes <= stats.total_contexts);
    }
}

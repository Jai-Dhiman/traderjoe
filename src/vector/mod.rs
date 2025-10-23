// Vector operations and similarity search using PostgreSQL + pgvector
// Provides HNSW indexing and efficient similarity search for embeddings

mod pgvector_sqlx;
pub use pgvector_sqlx::PgVector;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use tracing::{info, warn};
use uuid::Uuid;

/// VectorStore provides vector similarity search capabilities
pub struct VectorStore {
    pool: PgPool,
}

/// ACE context entry for similarity search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEntry {
    pub id: Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub market_state: serde_json::Value,
    pub decision: serde_json::Value,
    pub reasoning: String,
    pub confidence: f32,
    pub outcome: Option<serde_json::Value>,
    pub similarity: Option<f32>,
}

impl VectorStore {
    /// Create a new VectorStore instance
    pub async fn new(pool: PgPool) -> Result<Self> {
        // Verify pgvector extension
        sqlx::query("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            .fetch_optional(&pool)
            .await
            .context("Failed to verify pgvector extension")?
            .ok_or_else(|| {
                anyhow::anyhow!("pgvector extension not found. Please run migrations.")
            })?;

        Ok(Self { pool })
    }

    /// Perform similarity search against ace_contexts table
    pub async fn similarity_search(
        &self,
        embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<ContextEntry>> {
        // Convert Vec<f32> to PgVector type
        let pg_vector = PgVector::new(embedding);

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
                (embedding <=> $1) AS similarity
            FROM ace_contexts
            WHERE embedding IS NOT NULL
            AND decision IS NOT NULL
            AND reasoning IS NOT NULL
            AND confidence IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT $2
            "#,
            pg_vector as PgVector,
            limit as i64
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to execute similarity search")?;

        let results: Vec<ContextEntry> = rows.into_iter().filter_map(|row| {
            match (row.decision, row.reasoning, row.confidence) {
                (Some(decision), Some(reasoning), Some(confidence)) => Some(ContextEntry {
                    id: row.id,
                    timestamp: row.timestamp,
                    market_state: row.market_state,
                    decision,
                    reasoning,
                    confidence,
                    outcome: row.outcome,
                    similarity: row.similarity.map(|s| s as f32),
                }),
                _ => {
                    warn!("Skipping context {} with missing required fields (decision, reasoning, or confidence)", row.id);
                    None
                }
            }
        }).collect();

        info!("Found {} similar contexts", results.len());
        Ok(results)
    }

    /// Insert or update context with embedding
    pub async fn upsert_context_embedding(
        &self,
        id: Uuid,
        timestamp: chrono::DateTime<chrono::Utc>,
        market_state: serde_json::Value,
        decision: serde_json::Value,
        reasoning: String,
        confidence: f32,
        outcome: Option<serde_json::Value>,
        embedding: Vec<f32>,
    ) -> Result<()> {
        let pg_vector = PgVector::new(embedding);

        sqlx::query!(
            r#"
            INSERT INTO ace_contexts (
                id, timestamp, market_state, decision, reasoning, confidence, outcome, embedding
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (id)
            DO UPDATE SET
                timestamp = EXCLUDED.timestamp,
                market_state = EXCLUDED.market_state,
                decision = EXCLUDED.decision,
                reasoning = EXCLUDED.reasoning,
                confidence = EXCLUDED.confidence,
                outcome = EXCLUDED.outcome,
                embedding = EXCLUDED.embedding
            "#,
            id,
            timestamp,
            market_state,
            decision,
            reasoning,
            confidence,
            outcome,
            pg_vector as PgVector
        )
        .execute(&self.pool)
        .await
        .context("Failed to upsert context with embedding")?;

        info!("Upserted context {} with embedding", id);
        Ok(())
    }

    /// Ensure HNSW index exists on specified table and column
    pub async fn ensure_hnsw_index(&self, table: &str, column: &str) -> Result<()> {
        // Check if index already exists
        let index_name = format!("idx_{}_{}", table, column);
        let check_query = "SELECT 1 FROM pg_indexes WHERE indexname = $1";

        let exists = sqlx::query(check_query)
            .bind(&index_name)
            .fetch_optional(&self.pool)
            .await
            .context("Failed to check if HNSW index exists")?;

        if exists.is_some() {
            info!("HNSW index {} already exists", index_name);
            return Ok(());
        }

        // Create HNSW index
        let create_query = format!(
            "CREATE INDEX {} ON {} USING hnsw ({} vector_cosine_ops)",
            index_name, table, column
        );

        sqlx::query(&create_query)
            .execute(&self.pool)
            .await
            .context("Failed to create HNSW index")?;

        info!("Created HNSW index {} for {}.{}", index_name, table, column);
        Ok(())
    }

    /// Get statistics about ace_contexts table
    pub async fn context_stats(&self) -> Result<(usize, usize, usize)> {
        let query = r#"
            SELECT 
                COUNT(*) as total_contexts,
                COUNT(embedding) as contexts_with_embeddings,
                COUNT(outcome) as contexts_with_outcomes
            FROM ace_contexts
        "#;

        let row = sqlx::query(query)
            .fetch_one(&self.pool)
            .await
            .context("Failed to get context statistics")?;

        let total: i64 = row.get("total_contexts");
        let with_embeddings: i64 = row.get("contexts_with_embeddings");
        let with_outcomes: i64 = row.get("contexts_with_outcomes");

        Ok((
            total as usize,
            with_embeddings as usize,
            with_outcomes as usize,
        ))
    }

    /// Get recent contexts for analysis
    pub async fn get_recent_contexts(&self, limit: usize) -> Result<Vec<ContextEntry>> {
        let query = r#"
            SELECT 
                id, timestamp, market_state, decision, reasoning, confidence, outcome
            FROM ace_contexts 
            ORDER BY timestamp DESC 
            LIMIT $1
        "#;

        let rows = sqlx::query(query)
            .bind(limit as i64)
            .fetch_all(&self.pool)
            .await
            .context("Failed to get recent contexts")?;

        let mut results = Vec::new();
        for row in rows {
            let entry = ContextEntry {
                id: row.get("id"),
                timestamp: row.get("timestamp"),
                market_state: row.get("market_state"),
                decision: row.get("decision"),
                reasoning: row.get("reasoning"),
                confidence: row.get("confidence"),
                outcome: row.get("outcome"),
                similarity: None,
            };
            results.push(entry);
        }

        Ok(results)
    }

    /// Update outcome for existing context
    pub async fn update_context_outcome(
        &self,
        context_id: Uuid,
        outcome: serde_json::Value,
    ) -> Result<()> {
        let query = "UPDATE ace_contexts SET outcome = $1 WHERE id = $2";

        let result = sqlx::query(query)
            .bind(outcome)
            .bind(context_id)
            .execute(&self.pool)
            .await
            .context("Failed to update context outcome")?;

        if result.rows_affected() == 0 {
            warn!("No context found with id {} to update outcome", context_id);
        } else {
            info!("Updated outcome for context {}", context_id);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::EmbeddingGemma;

    // Note: These tests require a running PostgreSQL instance with pgvector
    // Run: docker run -d --name postgres-test -e POSTGRES_PASSWORD=test -p 5432:5432 pgvector/pgvector:pg16

    #[tokio::test]
    #[ignore] // Requires database setup
    async fn test_vector_operations() {
        let database_url = std::env::var("TEST_DATABASE_URL")
            .unwrap_or_else(|_| "postgresql://postgres:test@localhost/traderjoe_test".to_string());

        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(5)
            .connect(&database_url)
            .await
            .expect("Failed to connect to test database");

        let vector_store = VectorStore::new(pool)
            .await
            .expect("Failed to create VectorStore");

        // Ensure HNSW index
        vector_store
            .ensure_hnsw_index("ace_contexts", "embedding")
            .await
            .expect("Failed to create HNSW index");

        // Test embedding and similarity search
        let embedder = EmbeddingGemma::load()
            .await
            .expect("Failed to load embedder");
        let embedding = embedder
            .embed("bullish market sentiment")
            .await
            .expect("Failed to generate embedding");

        // Insert test context
        let context_id = Uuid::new_v4();
        let market_state = serde_json::json!({
            "spy_price": 450.0,
            "vix": 15.0,
            "sentiment": 0.7
        });

        let decision = serde_json::json!({
            "action": "BUY_CALLS",
            "strike": 452.0,
            "expiry": "2025-01-17"
        });

        vector_store
            .upsert_context_embedding(
                context_id,
                chrono::Utc::now(),
                market_state,
                decision,
                "Strong bullish signals from technical analysis".to_string(),
                0.85,
                None,
                embedding.clone(),
            )
            .await
            .expect("Failed to insert context");

        // Test similarity search
        let similar_embedding = embedder
            .embed("positive market outlook")
            .await
            .expect("Failed to generate similar embedding");

        let results = vector_store
            .similarity_search(similar_embedding, 5)
            .await
            .expect("Failed to perform similarity search");

        assert!(!results.is_empty());
        assert!(results[0].similarity.is_some());

        // Test context stats
        let (total, with_embeddings, _) = vector_store
            .context_stats()
            .await
            .expect("Failed to get context stats");

        assert!(total > 0);
        assert!(with_embeddings > 0);
    }
}

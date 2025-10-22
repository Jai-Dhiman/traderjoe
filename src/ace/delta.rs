//! Delta engine for ACE incremental context updates
//! Handles embedding-based deduplication and deterministic merging

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use tracing::{info, warn, debug};

use crate::{
    ace::playbook::{PlaybookDAO, PlaybookSection, PlaybookBullet},
    embeddings::EmbeddingGemma,
};

/// Delta operation types for incremental updates
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeltaOp {
    /// Add new bullet to playbook
    Add,
    /// Update existing bullet (content or counters)
    Update,
    /// Remove bullet from playbook
    Remove,
}

/// Delta representing a single incremental change to the playbook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    /// Type of operation to perform
    pub op: DeltaOp,
    /// Target section for the operation
    pub section: PlaybookSection,
    /// Content for Add/Update operations
    pub content: Option<String>,
    /// Target bullet ID for Update/Remove operations
    pub bullet_id: Option<Uuid>,
    /// Counter adjustments for Update operations
    pub helpful_delta: Option<i32>,
    pub harmful_delta: Option<i32>,
    pub confidence_adjustment: Option<f32>,
    /// Metadata for traceability and additional context
    pub meta: Option<serde_json::Value>,
}

impl Delta {
    /// Create an Add delta
    pub fn add(
        section: PlaybookSection,
        content: String,
        meta: Option<serde_json::Value>,
    ) -> Self {
        Self {
            op: DeltaOp::Add,
            section,
            content: Some(content),
            bullet_id: None,
            helpful_delta: None,
            harmful_delta: None,
            confidence_adjustment: None,
            meta,
        }
    }

    /// Create an Update delta for content
    pub fn update_content(
        bullet_id: Uuid,
        section: PlaybookSection,
        content: String,
        meta: Option<serde_json::Value>,
    ) -> Self {
        Self {
            op: DeltaOp::Update,
            section,
            content: Some(content),
            bullet_id: Some(bullet_id),
            helpful_delta: None,
            harmful_delta: None,
            confidence_adjustment: None,
            meta,
        }
    }

    /// Create an Update delta for counters
    pub fn update_counters(
        bullet_id: Uuid,
        section: PlaybookSection,
        helpful_delta: i32,
        harmful_delta: i32,
        confidence_adjustment: f32,
        meta: Option<serde_json::Value>,
    ) -> Self {
        Self {
            op: DeltaOp::Update,
            section,
            content: None,
            bullet_id: Some(bullet_id),
            helpful_delta: Some(helpful_delta),
            harmful_delta: Some(harmful_delta),
            confidence_adjustment: Some(confidence_adjustment),
            meta,
        }
    }

    /// Create a Remove delta
    pub fn remove(
        bullet_id: Uuid,
        section: PlaybookSection,
        meta: Option<serde_json::Value>,
    ) -> Self {
        Self {
            op: DeltaOp::Remove,
            section,
            content: None,
            bullet_id: Some(bullet_id),
            helpful_delta: None,
            harmful_delta: None,
            confidence_adjustment: None,
            meta,
        }
    }

    /// Validate delta has required fields for its operation
    pub fn validate(&self) -> Result<()> {
        match self.op {
            DeltaOp::Add => {
                if self.content.is_none() {
                    return Err(anyhow::anyhow!("Add delta requires content"));
                }
            }
            DeltaOp::Update => {
                if self.bullet_id.is_none() {
                    return Err(anyhow::anyhow!("Update delta requires bullet_id"));
                }
                if self.content.is_none() 
                    && self.helpful_delta.is_none() 
                    && self.harmful_delta.is_none() 
                    && self.confidence_adjustment.is_none() {
                    return Err(anyhow::anyhow!(
                        "Update delta requires at least one field to update"
                    ));
                }
            }
            DeltaOp::Remove => {
                if self.bullet_id.is_none() {
                    return Err(anyhow::anyhow!("Remove delta requires bullet_id"));
                }
            }
        }
        Ok(())
    }
}

/// Result of applying deltas to the playbook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyReport {
    /// Number of deltas successfully applied
    pub applied_count: usize,
    /// Number of deltas skipped (e.g., due to deduplication)
    pub skipped_count: usize,
    /// Number of deltas that failed to apply
    pub failed_count: usize,
    /// Details about each delta application
    pub delta_results: Vec<DeltaResult>,
    /// Overall success status
    pub success: bool,
}

/// Result of applying a single delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaResult {
    /// The delta that was processed
    pub delta: Delta,
    /// Status of the application
    pub status: DeltaStatus,
    /// Additional information about the result
    pub message: Option<String>,
    /// ID of bullet created/modified (if applicable)
    pub bullet_id: Option<Uuid>,
    /// Similarity score for duplicates (if applicable)
    pub similarity_score: Option<f32>,
}

/// Status of delta application
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeltaStatus {
    /// Delta was successfully applied
    Applied,
    /// Delta was skipped due to deduplication
    SkippedDuplicate,
    /// Delta was skipped for other reasons
    SkippedOther,
    /// Delta failed to apply due to error
    Failed,
}

/// Configuration for delta application
#[derive(Debug, Clone)]
pub struct DeltaEngineConfig {
    /// Cosine similarity threshold for deduplication (0.0 to 1.0)
    pub similarity_threshold: f32,
    /// Minimum content length to check for duplicates
    pub min_content_length: usize,
    /// Maximum number of existing bullets to check for similarity per section
    pub max_similarity_checks: usize,
}

impl Default for DeltaEngineConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.90,
            min_content_length: 10,
            max_similarity_checks: 100,
        }
    }
}

/// Delta engine for processing incremental playbook updates
pub struct DeltaEngine {
    playbook_dao: PlaybookDAO,
    embedder: EmbeddingGemma,
    config: DeltaEngineConfig,
}

impl DeltaEngine {
    /// Create new delta engine
    pub async fn new(
        playbook_dao: PlaybookDAO,
        config: Option<DeltaEngineConfig>,
    ) -> Result<Self> {
        let embedder = EmbeddingGemma::load().await?;
        let config = config.unwrap_or_default();

        Ok(Self {
            playbook_dao,
            embedder,
            config,
        })
    }

    /// Apply a batch of deltas to the playbook
    pub async fn apply_deltas(&self, deltas: Vec<Delta>) -> Result<ApplyReport> {
        info!("Applying {} deltas to playbook", deltas.len());

        let mut delta_results = Vec::new();
        let mut applied_count = 0;
        let mut skipped_count = 0;
        let mut failed_count = 0;

        for delta in deltas {
            let result = self.apply_single_delta(delta).await;
            
            match &result.status {
                DeltaStatus::Applied => applied_count += 1,
                DeltaStatus::SkippedDuplicate | DeltaStatus::SkippedOther => skipped_count += 1,
                DeltaStatus::Failed => failed_count += 1,
            }

            delta_results.push(result);
        }

        let report = ApplyReport {
            applied_count,
            skipped_count,
            failed_count,
            delta_results,
            success: failed_count == 0,
        };

        info!(
            "Delta application complete: {} applied, {} skipped, {} failed",
            applied_count, skipped_count, failed_count
        );

        Ok(report)
    }

    /// Apply a single delta
    async fn apply_single_delta(&self, delta: Delta) -> DeltaResult {
        // Validate delta structure
        if let Err(e) = delta.validate() {
            return DeltaResult {
                delta,
                status: DeltaStatus::Failed,
                message: Some(format!("Validation failed: {}", e)),
                bullet_id: None,
                similarity_score: None,
            };
        }

        match delta.op {
            DeltaOp::Add => self.apply_add_delta(delta).await,
            DeltaOp::Update => self.apply_update_delta(delta).await,
            DeltaOp::Remove => self.apply_remove_delta(delta).await,
        }
    }

    /// Apply an Add delta with deduplication
    async fn apply_add_delta(&self, delta: Delta) -> DeltaResult {
        let content = delta.content.as_ref().unwrap();

        // Skip deduplication check for very short content
        if content.len() < self.config.min_content_length {
            debug!("Skipping deduplication check for short content: {}", content);
        } else {
            // Check for duplicates in the same section
            match self.check_for_duplicates(&delta.section, content).await {
                Ok(Some((similarity, existing_bullet))) => {
                    debug!(
                        "Found duplicate bullet {} with similarity {:.3}: {}",
                        existing_bullet.id, similarity, content
                    );
                    return DeltaResult {
                        delta,
                        status: DeltaStatus::SkippedDuplicate,
                        message: Some(format!(
                            "Content is {:.1}% similar to existing bullet {}",
                            similarity * 100.0,
                            existing_bullet.id
                        )),
                        bullet_id: Some(existing_bullet.id),
                        similarity_score: Some(similarity),
                    };
                }
                Ok(None) => {
                    // No duplicate found, proceed with insertion
                }
                Err(e) => {
                    warn!("Failed to check for duplicates: {}", e);
                    // Continue with insertion despite deduplication failure
                }
            }
        }

        // Insert new bullet
        match self.playbook_dao.insert_bullet(
            delta.section.clone(),
            content.clone(),
            delta.meta.as_ref().and_then(|m| m.get("source_context_id"))
                .and_then(|id| id.as_str())
                .and_then(|id| Uuid::parse_str(id).ok()),
            delta.meta.clone(),
        ).await {
            Ok(bullet_id) => DeltaResult {
                delta,
                status: DeltaStatus::Applied,
                message: Some("Successfully added new bullet".to_string()),
                bullet_id: Some(bullet_id),
                similarity_score: None,
            },
            Err(e) => DeltaResult {
                delta,
                status: DeltaStatus::Failed,
                message: Some(format!("Failed to insert bullet: {}", e)),
                bullet_id: None,
                similarity_score: None,
            },
        }
    }

    /// Apply an Update delta
    async fn apply_update_delta(&self, delta: Delta) -> DeltaResult {
        let bullet_id = delta.bullet_id.unwrap();

        // Update content if provided
        if let Some(content) = &delta.content {
            match self.playbook_dao.update_bullet_content(bullet_id, content.clone()).await {
                Ok(true) => {
                    return DeltaResult {
                        delta,
                        status: DeltaStatus::Applied,
                        message: Some("Successfully updated bullet content".to_string()),
                        bullet_id: Some(bullet_id),
                        similarity_score: None,
                    };
                }
                Ok(false) => {
                    return DeltaResult {
                        delta,
                        status: DeltaStatus::Failed,
                        message: Some("Bullet not found for content update".to_string()),
                        bullet_id: Some(bullet_id),
                        similarity_score: None,
                    };
                }
                Err(e) => {
                    return DeltaResult {
                        delta,
                        status: DeltaStatus::Failed,
                        message: Some(format!("Failed to update content: {}", e)),
                        bullet_id: Some(bullet_id),
                        similarity_score: None,
                    };
                }
            }
        }

        // Update counters if provided
        if delta.helpful_delta.is_some() || delta.harmful_delta.is_some() || delta.confidence_adjustment.is_some() {
            let helpful_delta = delta.helpful_delta.unwrap_or(0);
            let harmful_delta = delta.harmful_delta.unwrap_or(0);
            let confidence_adjustment = delta.confidence_adjustment.unwrap_or(0.0);

            match self.playbook_dao.update_counters(
                bullet_id,
                helpful_delta,
                harmful_delta,
                confidence_adjustment,
            ).await {
                Ok(true) => DeltaResult {
                    delta,
                    status: DeltaStatus::Applied,
                    message: Some("Successfully updated bullet counters".to_string()),
                    bullet_id: Some(bullet_id),
                    similarity_score: None,
                },
                Ok(false) => DeltaResult {
                    delta,
                    status: DeltaStatus::Failed,
                    message: Some("Bullet not found for counter update".to_string()),
                    bullet_id: Some(bullet_id),
                    similarity_score: None,
                },
                Err(e) => DeltaResult {
                    delta,
                    status: DeltaStatus::Failed,
                    message: Some(format!("Failed to update counters: {}", e)),
                    bullet_id: Some(bullet_id),
                    similarity_score: None,
                },
            }
        } else {
            DeltaResult {
                delta,
                status: DeltaStatus::SkippedOther,
                message: Some("No updates specified".to_string()),
                bullet_id: Some(bullet_id),
                similarity_score: None,
            }
        }
    }

    /// Apply a Remove delta
    async fn apply_remove_delta(&self, delta: Delta) -> DeltaResult {
        let bullet_id = delta.bullet_id.unwrap();

        match self.playbook_dao.delete_bullet(bullet_id).await {
            Ok(true) => DeltaResult {
                delta,
                status: DeltaStatus::Applied,
                message: Some("Successfully removed bullet".to_string()),
                bullet_id: Some(bullet_id),
                similarity_score: None,
            },
            Ok(false) => DeltaResult {
                delta,
                status: DeltaStatus::Failed,
                message: Some("Bullet not found for removal".to_string()),
                bullet_id: Some(bullet_id),
                similarity_score: None,
            },
            Err(e) => DeltaResult {
                delta,
                status: DeltaStatus::Failed,
                message: Some(format!("Failed to remove bullet: {}", e)),
                bullet_id: Some(bullet_id),
                similarity_score: None,
            },
        }
    }

    /// Check for duplicate content in the given section
    async fn check_for_duplicates(
        &self,
        section: &PlaybookSection,
        content: &str,
    ) -> Result<Option<(f32, PlaybookBullet)>> {
        // Get existing bullets in the same section
        let existing_bullets = self.playbook_dao
            .get_by_section(section.clone(), Some(self.config.max_similarity_checks))
            .await
            .context("Failed to get existing bullets for deduplication")?;

        if existing_bullets.is_empty() {
            return Ok(None);
        }

        // Generate embedding for the new content
        let new_embedding = self.embedder.embed(content).await
            .context("Failed to generate embedding for new content")?;

        // Check similarity against existing bullets
        let mut best_match: Option<(f32, PlaybookBullet)> = None;

        for bullet in existing_bullets {
            // Generate embedding for existing bullet content
            let existing_embedding = self.embedder.embed(&bullet.content).await
                .context("Failed to generate embedding for existing bullet")?;

            // Calculate cosine similarity
            let similarity = cosine_similarity(&new_embedding, &existing_embedding);

            if similarity >= self.config.similarity_threshold {
                if let Some((current_best_similarity, _)) = &best_match {
                    if similarity > *current_best_similarity {
                        best_match = Some((similarity, bullet));
                    }
                } else {
                    best_match = Some((similarity, bullet));
                }
            }
        }

        Ok(best_match)
    }

    /// Get configuration for debugging/monitoring
    pub fn get_config(&self) -> &DeltaEngineConfig {
        &self.config
    }
}

/// Calculate cosine similarity between two embeddings
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ace::playbook::PlaybookDAO;
    use serde_json::json;

    #[test]
    fn test_delta_validation() {
        // Valid Add delta
        let add_delta = Delta::add(
            PlaybookSection::PatternInsights,
            "Test content".to_string(),
            None,
        );
        assert!(add_delta.validate().is_ok());

        // Invalid Add delta (no content)
        let invalid_add = Delta {
            op: DeltaOp::Add,
            section: PlaybookSection::PatternInsights,
            content: None,
            bullet_id: None,
            helpful_delta: None,
            harmful_delta: None,
            confidence_adjustment: None,
            meta: None,
        };
        assert!(invalid_add.validate().is_err());

        // Valid Update delta
        let update_delta = Delta::update_counters(
            Uuid::new_v4(),
            PlaybookSection::FailureModes,
            1,
            0,
            0.1,
            None,
        );
        assert!(update_delta.validate().is_ok());

        // Invalid Update delta (no bullet_id)
        let invalid_update = Delta {
            op: DeltaOp::Update,
            section: PlaybookSection::PatternInsights,
            content: Some("content".to_string()),
            bullet_id: None,
            helpful_delta: None,
            harmful_delta: None,
            confidence_adjustment: None,
            meta: None,
        };
        assert!(invalid_update.validate().is_err());

        // Valid Remove delta
        let remove_delta = Delta::remove(
            Uuid::new_v4(),
            PlaybookSection::RegimeRules,
            None,
        );
        assert!(remove_delta.validate().is_ok());
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        // Orthogonal vectors
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);

        // Opposite vectors
        let a = vec![1.0, 2.0];
        let b = vec![-1.0, -2.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);

        // Different length vectors
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);

        // Zero vectors
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_delta_creation_helpers() {
        let bullet_id = Uuid::new_v4();

        // Test Add delta
        let add_delta = Delta::add(
            PlaybookSection::PatternInsights,
            "VIX spikes predict reversals".to_string(),
            Some(json!({"confidence": 0.8})),
        );
        assert_eq!(add_delta.op, DeltaOp::Add);
        assert_eq!(add_delta.section, PlaybookSection::PatternInsights);
        assert!(add_delta.content.is_some());
        assert!(add_delta.bullet_id.is_none());

        // Test Update content delta
        let update_content = Delta::update_content(
            bullet_id,
            PlaybookSection::FailureModes,
            "Updated content".to_string(),
            None,
        );
        assert_eq!(update_content.op, DeltaOp::Update);
        assert_eq!(update_content.bullet_id, Some(bullet_id));
        assert!(update_content.content.is_some());
        assert!(update_content.helpful_delta.is_none());

        // Test Update counters delta
        let update_counters = Delta::update_counters(
            bullet_id,
            PlaybookSection::RegimeRules,
            2,
            1,
            0.05,
            None,
        );
        assert_eq!(update_counters.op, DeltaOp::Update);
        assert_eq!(update_counters.bullet_id, Some(bullet_id));
        assert!(update_counters.content.is_none());
        assert_eq!(update_counters.helpful_delta, Some(2));
        assert_eq!(update_counters.harmful_delta, Some(1));
        assert_eq!(update_counters.confidence_adjustment, Some(0.05));

        // Test Remove delta
        let remove_delta = Delta::remove(
            bullet_id,
            PlaybookSection::ModelReliability,
            Some(json!({"reason": "outdated"})),
        );
        assert_eq!(remove_delta.op, DeltaOp::Remove);
        assert_eq!(remove_delta.bullet_id, Some(bullet_id));
        assert!(remove_delta.content.is_none());
        assert!(remove_delta.meta.is_some());
    }

    #[tokio::test]
    #[ignore] // Requires TEST_DATABASE_URL and actual database setup
    async fn test_delta_engine_integration() {
        let database_url = std::env::var("TEST_DATABASE_URL")
            .unwrap_or_else(|_| "postgresql://postgres:postgres@localhost/traderjoe_test".to_string());

        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(5)
            .connect(&database_url)
            .await
            .expect("Failed to connect to test database");

        // Run migrations
        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .expect("Failed to run migrations");

        let playbook_dao = PlaybookDAO::new(pool);
        let engine = DeltaEngine::new(playbook_dao, None)
            .await
            .expect("Failed to create delta engine");

        // Test adding a new bullet
        let deltas = vec![
            Delta::add(
                PlaybookSection::PatternInsights,
                "High VIX usually leads to mean reversion within 3-5 days".to_string(),
                Some(json!({"test": true})),
            ),
        ];

        let report = engine.apply_deltas(deltas)
            .await
            .expect("Failed to apply deltas");

        assert_eq!(report.applied_count, 1);
        assert_eq!(report.skipped_count, 0);
        assert_eq!(report.failed_count, 0);
        assert!(report.success);

        let bullet_id = report.delta_results[0].bullet_id.expect("Should have bullet ID");

        // Test updating the bullet
        let update_deltas = vec![
            Delta::update_counters(
                bullet_id,
                PlaybookSection::PatternInsights,
                3,
                1,
                0.1,
                None,
            ),
        ];

        let update_report = engine.apply_deltas(update_deltas)
            .await
            .expect("Failed to apply update deltas");

        assert_eq!(update_report.applied_count, 1);
        assert!(update_report.success);

        // Test deduplication - add very similar content
        let duplicate_deltas = vec![
            Delta::add(
                PlaybookSection::PatternInsights,
                "High VIX typically leads to mean reversion in 3-5 trading days".to_string(),
                None,
            ),
        ];

        let dup_report = engine.apply_deltas(duplicate_deltas)
            .await
            .expect("Failed to apply duplicate deltas");

        assert_eq!(dup_report.skipped_count, 1);
        assert_eq!(dup_report.delta_results[0].status, DeltaStatus::SkippedDuplicate);
        assert!(dup_report.delta_results[0].similarity_score.unwrap() >= 0.90);

        // Test removing the bullet
        let remove_deltas = vec![
            Delta::remove(bullet_id, PlaybookSection::PatternInsights, None),
        ];

        let remove_report = engine.apply_deltas(remove_deltas)
            .await
            .expect("Failed to apply remove deltas");

        assert_eq!(remove_report.applied_count, 1);
        assert!(remove_report.success);
    }
}
//! ACE Playbook data access layer
//! Manages playbook bullets for incremental context evolution

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Valid playbook sections based on ACE framework
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PlaybookSection {
    PatternInsights,
    FailureModes,
    RegimeRules,
    ModelReliability,
    NewsImpact,
    StrategyLifecycle,
}

impl PlaybookSection {
    /// Convert to database string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            PlaybookSection::PatternInsights => "pattern_insights",
            PlaybookSection::FailureModes => "failure_modes",
            PlaybookSection::RegimeRules => "regime_rules",
            PlaybookSection::ModelReliability => "model_reliability",
            PlaybookSection::NewsImpact => "news_impact",
            PlaybookSection::StrategyLifecycle => "strategy_lifecycle",
        }
    }

    /// Parse from database string representation
    pub fn from_str(s: &str) -> Result<Self> {
        match s {
            "pattern_insights" => Ok(PlaybookSection::PatternInsights),
            "failure_modes" => Ok(PlaybookSection::FailureModes),
            "regime_rules" => Ok(PlaybookSection::RegimeRules),
            "model_reliability" => Ok(PlaybookSection::ModelReliability),
            "news_impact" => Ok(PlaybookSection::NewsImpact),
            "strategy_lifecycle" => Ok(PlaybookSection::StrategyLifecycle),
            _ => Err(anyhow::anyhow!("Invalid playbook section: {}", s)),
        }
    }

    /// All valid sections
    pub fn all() -> Vec<PlaybookSection> {
        vec![
            PlaybookSection::PatternInsights,
            PlaybookSection::FailureModes,
            PlaybookSection::RegimeRules,
            PlaybookSection::ModelReliability,
            PlaybookSection::NewsImpact,
            PlaybookSection::StrategyLifecycle,
        ]
    }
}

/// ACE playbook bullet representing accumulated knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybookBullet {
    pub id: Uuid,
    pub section: PlaybookSection,
    pub content: String,
    pub helpful_count: i32,
    pub harmful_count: i32,
    pub confidence: f32,
    pub last_used: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub source_context_id: Option<Uuid>,
    pub meta: Option<serde_json::Value>,
}

impl PlaybookBullet {
    /// Calculate the effectiveness ratio (helpful / total feedback)
    pub fn effectiveness_ratio(&self) -> f32 {
        let total = self.helpful_count + self.harmful_count;
        if total > 0 {
            self.helpful_count as f32 / total as f32
        } else {
            0.5 // Neutral when no feedback
        }
    }

    /// Check if bullet is stale (unused for more than days)
    pub fn is_stale(&self, days_threshold: i64) -> bool {
        match self.last_used {
            Some(last_used) => {
                let threshold = Utc::now() - chrono::Duration::days(days_threshold);
                last_used < threshold
            }
            None => {
                // Never used - check creation age
                let threshold = Utc::now() - chrono::Duration::days(days_threshold);
                self.created_at < threshold
            }
        }
    }

    /// Check if bullet should be pruned based on performance
    pub fn should_prune(&self, min_confidence: f32, max_staleness_days: i64) -> bool {
        self.confidence < min_confidence
            && self.harmful_count > self.helpful_count
            && self.is_stale(max_staleness_days)
    }
}

/// Playbook query filters
#[derive(Debug, Clone, Default)]
pub struct PlaybookFilters {
    pub sections: Option<Vec<PlaybookSection>>,
    pub min_confidence: Option<f32>,
    pub max_confidence: Option<f32>,
    pub min_helpful_count: Option<i32>,
    pub only_recent: Option<i64>, // Days
    pub content_search: Option<String>,
}

/// Data access object for playbook bullets
#[derive(Debug, Clone)]
pub struct PlaybookDAO {
    pool: PgPool,
}

impl PlaybookDAO {
    /// Create new PlaybookDAO with database pool
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Insert new playbook bullet
    pub async fn insert_bullet(
        &self,
        section: PlaybookSection,
        content: String,
        source_context_id: Option<Uuid>,
        meta: Option<serde_json::Value>,
    ) -> Result<Uuid> {
        let bullet_id = Uuid::new_v4();
        let section_str = section.as_str();

        sqlx::query!(
            r#"
            INSERT INTO playbook_bullets 
            (id, section, content, source_context_id, meta)
            VALUES ($1, $2, $3, $4, $5)
            "#,
            bullet_id,
            section_str,
            content,
            source_context_id,
            meta
        )
        .execute(&self.pool)
        .await
        .context("Failed to insert playbook bullet")?;

        info!(
            "Inserted playbook bullet {} in section {}",
            bullet_id, section_str
        );
        Ok(bullet_id)
    }

    /// Update bullet content
    pub async fn update_bullet_content(&self, bullet_id: Uuid, content: String) -> Result<bool> {
        let result = sqlx::query!(
            "UPDATE playbook_bullets SET content = $1 WHERE id = $2",
            content,
            bullet_id
        )
        .execute(&self.pool)
        .await
        .context("Failed to update bullet content")?;

        let updated = result.rows_affected() > 0;
        if updated {
            info!("Updated content for bullet {}", bullet_id);
        } else {
            warn!("No bullet found with id {} to update content", bullet_id);
        }

        Ok(updated)
    }

    /// Update bullet counters and confidence
    pub async fn update_counters(
        &self,
        bullet_id: Uuid,
        helpful_delta: i32,
        harmful_delta: i32,
        confidence_adjustment: f32,
    ) -> Result<bool> {
        let result = sqlx::query!(
            r#"
            UPDATE playbook_bullets 
            SET 
                helpful_count = helpful_count + $2,
                harmful_count = harmful_count + $3,
                confidence = GREATEST(0.01, LEAST(0.99, confidence + $4))
            WHERE id = $1
            "#,
            bullet_id,
            helpful_delta,
            harmful_delta,
            confidence_adjustment
        )
        .execute(&self.pool)
        .await
        .context("Failed to update bullet counters")?;

        let updated = result.rows_affected() > 0;
        if updated {
            debug!(
                "Updated counters for bullet {}: +{} helpful, +{} harmful, {:+.3} confidence",
                bullet_id, helpful_delta, harmful_delta, confidence_adjustment
            );
        } else {
            warn!("No bullet found with id {} to update counters", bullet_id);
        }

        Ok(updated)
    }

    /// Update last_used timestamp
    pub async fn update_last_used(&self, bullet_id: Uuid) -> Result<bool> {
        let result = sqlx::query!(
            "UPDATE playbook_bullets SET last_used = NOW() WHERE id = $1",
            bullet_id
        )
        .execute(&self.pool)
        .await
        .context("Failed to update last_used timestamp")?;

        let updated = result.rows_affected() > 0;
        if updated {
            debug!("Updated last_used for bullet {}", bullet_id);
        }

        Ok(updated)
    }

    /// Delete bullet by ID
    pub async fn delete_bullet(&self, bullet_id: Uuid) -> Result<bool> {
        let result = sqlx::query!("DELETE FROM playbook_bullets WHERE id = $1", bullet_id)
            .execute(&self.pool)
            .await
            .context("Failed to delete bullet")?;

        let deleted = result.rows_affected() > 0;
        if deleted {
            info!("Deleted bullet {}", bullet_id);
        } else {
            warn!("No bullet found with id {} to delete", bullet_id);
        }

        Ok(deleted)
    }

    /// Get bullets by section
    pub async fn get_by_section(
        &self,
        section: PlaybookSection,
        limit: Option<usize>,
    ) -> Result<Vec<PlaybookBullet>> {
        let section_str = section.as_str();
        let limit = limit.unwrap_or(100) as i64;

        let rows = sqlx::query!(
            r#"
            SELECT 
                id, section, content, helpful_count, harmful_count, confidence,
                last_used, created_at, updated_at, source_context_id, meta
            FROM playbook_bullets 
            WHERE section = $1
            ORDER BY confidence DESC, helpful_count DESC
            LIMIT $2
            "#,
            section_str,
            limit
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to get bullets by section")?;

        let bullets: Vec<PlaybookBullet> = rows
            .into_iter()
            .filter_map(|row| {
                match PlaybookSection::from_str(&row.section) {
                    Ok(section) => Some(PlaybookBullet {
                        id: row.id,
                        section,
                        content: row.content,
                        helpful_count: row.helpful_count,
                        harmful_count: row.harmful_count,
                        confidence: row.confidence,
                        last_used: row.last_used,
                        created_at: row.created_at,
                        updated_at: row.updated_at,
                        source_context_id: row.source_context_id,
                        meta: row.meta,
                    }),
                    Err(e) => {
                        warn!("Skipping bullet {} with invalid section '{}': {}", row.id, row.section, e);
                        None
                    }
                }
            })
            .collect();

        debug!(
            "Retrieved {} bullets for section {}",
            bullets.len(),
            section_str
        );
        Ok(bullets)
    }

    /// Search bullets by content text
    pub async fn search_by_text(
        &self,
        query: &str,
        limit: Option<usize>,
    ) -> Result<Vec<PlaybookBullet>> {
        let limit = limit.unwrap_or(20) as i64;

        let rows = sqlx::query!(
            r#"
            SELECT 
                id, section, content, helpful_count, harmful_count, confidence,
                last_used, created_at, updated_at, source_context_id, meta
            FROM playbook_bullets 
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
            ORDER BY confidence DESC, helpful_count DESC
            LIMIT $2
            "#,
            query,
            limit
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to search bullets by text")?;

        let bullets: Vec<PlaybookBullet> = rows
            .into_iter()
            .filter_map(|row| {
                match PlaybookSection::from_str(&row.section) {
                    Ok(section) => Some(PlaybookBullet {
                        id: row.id,
                        section,
                        content: row.content,
                        helpful_count: row.helpful_count,
                        harmful_count: row.harmful_count,
                        confidence: row.confidence,
                        last_used: row.last_used,
                        created_at: row.created_at,
                        updated_at: row.updated_at,
                        source_context_id: row.source_context_id,
                        meta: row.meta,
                    }),
                    Err(e) => {
                        warn!("Skipping bullet {} with invalid section '{}': {}", row.id, row.section, e);
                        None
                    }
                }
            })
            .collect();

        debug!(
            "Found {} bullets matching text query: {}",
            bullets.len(),
            query
        );
        Ok(bullets)
    }

    /// Get top K most confident bullets across all sections
    pub async fn get_top_confident(&self, limit: usize) -> Result<Vec<PlaybookBullet>> {
        let limit = limit as i64;

        let rows = sqlx::query!(
            r#"
            SELECT 
                id, section, content, helpful_count, harmful_count, confidence,
                last_used, created_at, updated_at, source_context_id, meta
            FROM playbook_bullets 
            ORDER BY confidence DESC, helpful_count DESC
            LIMIT $1
            "#,
            limit
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to get top confident bullets")?;

        let bullets: Vec<PlaybookBullet> = rows
            .into_iter()
            .filter_map(|row| {
                match PlaybookSection::from_str(&row.section) {
                    Ok(section) => Some(PlaybookBullet {
                        id: row.id,
                        section,
                        content: row.content,
                        helpful_count: row.helpful_count,
                        harmful_count: row.harmful_count,
                        confidence: row.confidence,
                        last_used: row.last_used,
                        created_at: row.created_at,
                        updated_at: row.updated_at,
                        source_context_id: row.source_context_id,
                        meta: row.meta,
                    }),
                    Err(e) => {
                        warn!("Skipping bullet {} with invalid section '{}': {}", row.id, row.section, e);
                        None
                    }
                }
            })
            .collect();

        debug!("Retrieved {} top confident bullets", bullets.len());
        Ok(bullets)
    }

    /// Get recently used bullets for context generation
    pub async fn get_recent_bullets(&self, days: i64, limit: usize) -> Result<Vec<PlaybookBullet>> {
        let limit = limit as i64;
        let since = Utc::now() - chrono::Duration::days(days);

        let rows = sqlx::query!(
            r#"
            SELECT 
                id, section, content, helpful_count, harmful_count, confidence,
                last_used, created_at, updated_at, source_context_id, meta
            FROM playbook_bullets 
            WHERE last_used >= $1 OR (last_used IS NULL AND created_at >= $1)
            ORDER BY COALESCE(last_used, created_at) DESC, confidence DESC
            LIMIT $2
            "#,
            since,
            limit
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to get recent bullets")?;

        let bullets: Vec<PlaybookBullet> = rows
            .into_iter()
            .filter_map(|row| {
                match PlaybookSection::from_str(&row.section) {
                    Ok(section) => Some(PlaybookBullet {
                        id: row.id,
                        section,
                        content: row.content,
                        helpful_count: row.helpful_count,
                        harmful_count: row.harmful_count,
                        confidence: row.confidence,
                        last_used: row.last_used,
                        created_at: row.created_at,
                        updated_at: row.updated_at,
                        source_context_id: row.source_context_id,
                        meta: row.meta,
                    }),
                    Err(e) => {
                        warn!("Skipping bullet {} with invalid section '{}': {}", row.id, row.section, e);
                        None
                    }
                }
            })
            .collect();

        debug!(
            "Retrieved {} bullets used in last {} days",
            bullets.len(),
            days
        );
        Ok(bullets)
    }

    /// Get stale bullets for cleanup
    pub async fn get_stale_bullets(&self, days_threshold: i64) -> Result<Vec<PlaybookBullet>> {
        let threshold = Utc::now() - chrono::Duration::days(days_threshold);

        let rows = sqlx::query!(
            r#"
            SELECT 
                id, section, content, helpful_count, harmful_count, confidence,
                last_used, created_at, updated_at, source_context_id, meta
            FROM playbook_bullets 
            WHERE COALESCE(last_used, created_at) < $1
            ORDER BY confidence ASC, harmful_count DESC
            "#,
            threshold
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to get stale bullets")?;

        let bullets: Vec<PlaybookBullet> = rows
            .into_iter()
            .filter_map(|row| {
                match PlaybookSection::from_str(&row.section) {
                    Ok(section) => Some(PlaybookBullet {
                        id: row.id,
                        section,
                        content: row.content,
                        helpful_count: row.helpful_count,
                        harmful_count: row.harmful_count,
                        confidence: row.confidence,
                        last_used: row.last_used,
                        created_at: row.created_at,
                        updated_at: row.updated_at,
                        source_context_id: row.source_context_id,
                        meta: row.meta,
                    }),
                    Err(e) => {
                        warn!("Skipping bullet {} with invalid section '{}': {}", row.id, row.section, e);
                        None
                    }
                }
            })
            .collect();

        debug!(
            "Found {} stale bullets older than {} days",
            bullets.len(),
            days_threshold
        );
        Ok(bullets)
    }

    /// Query bullets with advanced filters
    pub async fn query_bullets(
        &self,
        filters: PlaybookFilters,
        limit: usize,
    ) -> Result<Vec<PlaybookBullet>> {
        let mut query = "SELECT id, section, content, helpful_count, harmful_count, confidence, last_used, created_at, updated_at, source_context_id, meta FROM playbook_bullets WHERE 1=1".to_string();
        let mut params: Vec<String> = Vec::new();
        let mut param_count = 1;

        // Build dynamic query based on filters
        if let Some(sections) = &filters.sections {
            let section_strs: Vec<String> = sections
                .iter()
                .map(|s| {
                    format!("${}", {
                        param_count += 1;
                        param_count - 1
                    })
                })
                .collect();
            query.push_str(&format!(" AND section IN ({})", section_strs.join(",")));
            params.extend(sections.iter().map(|s| s.as_str().to_string()));
        }

        if let Some(min_conf) = filters.min_confidence {
            query.push_str(&format!(" AND confidence >= ${}", param_count));
            params.push(min_conf.to_string());
            param_count += 1;
        }

        if let Some(max_conf) = filters.max_confidence {
            query.push_str(&format!(" AND confidence <= ${}", param_count));
            params.push(max_conf.to_string());
            param_count += 1;
        }

        if let Some(min_helpful) = filters.min_helpful_count {
            query.push_str(&format!(" AND helpful_count >= ${}", param_count));
            params.push(min_helpful.to_string());
            param_count += 1;
        }

        if let Some(days) = filters.only_recent {
            let since = Utc::now() - chrono::Duration::days(days);
            query.push_str(&format!(
                " AND (last_used >= ${} OR (last_used IS NULL AND created_at >= ${}))",
                param_count, param_count
            ));
            params.push(since.to_rfc3339());
            param_count += 1;
        }

        if let Some(search_text) = &filters.content_search {
            query.push_str(&format!(
                " AND to_tsvector('english', content) @@ plainto_tsquery('english', ${})",
                param_count
            ));
            params.push(search_text.clone());
            param_count += 1;
        }

        query.push_str(&format!(
            " ORDER BY confidence DESC, helpful_count DESC LIMIT ${}",
            param_count
        ));
        params.push(limit.to_string());

        // Note: This is a simplified implementation. In production, you'd want to use sqlx's
        // query builder or a proper dynamic query system to avoid SQL injection risks.
        // For now, we'll use the simpler individual methods above.

        warn!("Advanced query_bullets not fully implemented - use specific methods instead");
        Ok(vec![])
    }

    /// Get playbook statistics
    pub async fn get_stats(&self) -> Result<PlaybookStats> {
        let stats = sqlx::query!(
            r#"
            SELECT 
                COUNT(*) as total_bullets,
                COUNT(CASE WHEN confidence > 0.7 THEN 1 END) as high_confidence_bullets,
                COUNT(CASE WHEN last_used IS NOT NULL THEN 1 END) as used_bullets,
                AVG(confidence) as avg_confidence,
                SUM(helpful_count) as total_helpful,
                SUM(harmful_count) as total_harmful
            FROM playbook_bullets
            "#
        )
        .fetch_one(&self.pool)
        .await
        .context("Failed to get playbook statistics")?;

        Ok(PlaybookStats {
            total_bullets: stats.total_bullets.unwrap_or(0) as usize,
            high_confidence_bullets: stats.high_confidence_bullets.unwrap_or(0) as usize,
            used_bullets: stats.used_bullets.unwrap_or(0) as usize,
            avg_confidence: stats.avg_confidence.map(|c| c as f32),
            total_helpful: stats.total_helpful.unwrap_or(0) as i32,
            total_harmful: stats.total_harmful.unwrap_or(0) as i32,
        })
    }

    /// Get section-wise bullet counts
    pub async fn get_section_counts(&self) -> Result<Vec<(PlaybookSection, usize)>> {
        let rows = sqlx::query!(
            "SELECT section, COUNT(*) as count FROM playbook_bullets GROUP BY section ORDER BY count DESC"
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to get section counts")?;

        let mut counts = Vec::new();
        for row in rows {
            let section = PlaybookSection::from_str(&row.section)?;
            counts.push((section, row.count.unwrap_or(0) as usize));
        }

        Ok(counts)
    }
}

/// Playbook statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybookStats {
    pub total_bullets: usize,
    pub high_confidence_bullets: usize,
    pub used_bullets: usize,
    pub avg_confidence: Option<f32>,
    pub total_helpful: i32,
    pub total_harmful: i32,
}

impl PlaybookStats {
    /// Calculate overall effectiveness ratio
    pub fn effectiveness_ratio(&self) -> f32 {
        let total_feedback = self.total_helpful + self.total_harmful;
        if total_feedback > 0 {
            self.total_helpful as f32 / total_feedback as f32
        } else {
            0.5
        }
    }

    /// Usage percentage
    pub fn usage_percentage(&self) -> f32 {
        if self.total_bullets > 0 {
            (self.used_bullets as f32 / self.total_bullets as f32) * 100.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    #[ignore] // Requires TEST_DATABASE_URL
    async fn test_playbook_bullet_lifecycle() {
        let database_url = std::env::var("TEST_DATABASE_URL").unwrap_or_else(|_| {
            "postgresql://postgres:postgres@localhost/traderjoe_test".to_string()
        });

        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(5)
            .connect(&database_url)
            .await
            .expect("Failed to connect to test database");

        // Run migrations to ensure playbook_bullets table exists
        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .expect("Failed to run migrations");

        let dao = PlaybookDAO::new(pool);

        // Test insert
        let bullet_id = dao
            .insert_bullet(
                PlaybookSection::PatternInsights,
                "When VIX > 30, calls have 65% win rate".to_string(),
                None,
                Some(json!({"test": true})),
            )
            .await
            .expect("Failed to insert bullet");

        // Test retrieval by section
        let bullets = dao
            .get_by_section(PlaybookSection::PatternInsights, Some(10))
            .await
            .expect("Failed to get bullets by section");

        assert!(!bullets.is_empty());
        let bullet = &bullets[0];
        assert_eq!(bullet.id, bullet_id);
        assert_eq!(bullet.section, PlaybookSection::PatternInsights);
        assert_eq!(bullet.content, "When VIX > 30, calls have 65% win rate");
        assert_eq!(bullet.helpful_count, 0);
        assert_eq!(bullet.harmful_count, 0);
        assert_eq!(bullet.confidence, 0.5);

        // Test counter updates
        let updated = dao
            .update_counters(bullet_id, 3, 1, 0.1)
            .await
            .expect("Failed to update counters");
        assert!(updated);

        // Test last_used update
        let updated = dao
            .update_last_used(bullet_id)
            .await
            .expect("Failed to update last_used");
        assert!(updated);

        // Verify updates
        let bullets = dao
            .get_by_section(PlaybookSection::PatternInsights, Some(10))
            .await
            .expect("Failed to get bullets after update");

        let bullet = &bullets[0];
        assert_eq!(bullet.helpful_count, 3);
        assert_eq!(bullet.harmful_count, 1);
        assert!((bullet.confidence - 0.6).abs() < 0.01); // 0.5 + 0.1 = 0.6
        assert!(bullet.last_used.is_some());

        // Test search
        let found = dao
            .search_by_text("VIX", Some(5))
            .await
            .expect("Failed to search bullets");
        assert!(!found.is_empty());

        // Test stats
        let stats = dao.get_stats().await.expect("Failed to get stats");
        assert!(stats.total_bullets > 0);

        // Test delete
        let deleted = dao
            .delete_bullet(bullet_id)
            .await
            .expect("Failed to delete bullet");
        assert!(deleted);
    }

    #[test]
    fn test_playbook_section_conversion() {
        let section = PlaybookSection::PatternInsights;
        assert_eq!(section.as_str(), "pattern_insights");

        let parsed = PlaybookSection::from_str("pattern_insights").unwrap();
        assert_eq!(parsed, PlaybookSection::PatternInsights);

        let invalid = PlaybookSection::from_str("invalid_section");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_bullet_lifecycle_methods() {
        let bullet = PlaybookBullet {
            id: Uuid::new_v4(),
            section: PlaybookSection::FailureModes,
            content: "Test bullet".to_string(),
            helpful_count: 5,
            harmful_count: 2,
            confidence: 0.7,
            last_used: Some(Utc::now() - chrono::Duration::days(10)),
            created_at: Utc::now() - chrono::Duration::days(30),
            updated_at: Utc::now(),
            source_context_id: None,
            meta: None,
        };

        // Test effectiveness ratio
        let ratio = bullet.effectiveness_ratio();
        assert!((ratio - (5.0 / 7.0)).abs() < 0.01);

        // Test staleness
        assert!(!bullet.is_stale(15)); // Not stale within 15 days
        assert!(bullet.is_stale(5)); // Stale beyond 5 days

        // Test pruning logic
        assert!(!bullet.should_prune(0.5, 30)); // High confidence, not prunable

        let low_conf_bullet = PlaybookBullet {
            confidence: 0.3,
            harmful_count: 5,
            helpful_count: 1,
            last_used: Some(Utc::now() - chrono::Duration::days(40)),
            ..bullet
        };
        assert!(low_conf_bullet.should_prune(0.5, 30)); // Low confidence, harmful, stale
    }
}

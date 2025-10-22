//! ACE Curator module for orchestrating playbook evolution and delta management
//! Coordinates Generator, Reflector, and Delta Engine to maintain the playbook

use anyhow::{Context, Result};
use serde_json::json;
use tracing::{info, warn, debug};
use uuid::Uuid;

use crate::{
    ace::{
        delta::{Delta, DeltaEngine, ApplyReport, DeltaEngineConfig},
        playbook::{PlaybookDAO, PlaybookBullet, PlaybookSection, PlaybookStats},
        generator::{Generator, GeneratorInput},
        reflector::{Reflector, ReflectionInput, TradingOutcome},
        prompts::TradingDecision,
    },
    llm::LLMClient,
    vector::ContextEntry,
};

/// Configuration for the Curator's behavior
#[derive(Debug, Clone)]
pub struct CuratorConfig {
    /// Maximum number of recent bullets to include in generation context
    pub max_context_bullets: usize,
    /// Minimum confidence threshold for bullet pruning during maintenance
    pub min_confidence_for_pruning: f32,
    /// Days after which bullets are considered stale if unused
    pub staleness_threshold_days: i64,
    /// Whether to auto-update last_used timestamps for referenced bullets
    pub auto_update_usage: bool,
    /// Maximum deltas to apply in a single batch
    pub max_deltas_per_batch: usize,
}

impl Default for CuratorConfig {
    fn default() -> Self {
        Self {
            max_context_bullets: 50,
            min_confidence_for_pruning: 0.2,
            staleness_threshold_days: 30,
            auto_update_usage: true,
            max_deltas_per_batch: 100,
        }
    }
}

/// Summary of playbook changes after curation
#[derive(Debug, Clone)]
pub struct CurationSummary {
    /// Number of new bullets added
    pub bullets_added: usize,
    /// Number of bullets updated
    pub bullets_updated: usize,
    /// Number of bullets removed
    pub bullets_removed: usize,
    /// Total helpful count changes
    pub total_helpful_delta: i32,
    /// Total harmful count changes
    pub total_harmful_delta: i32,
    /// Number of confidence adjustments made
    pub confidence_adjustments: usize,
    /// Average confidence change
    pub avg_confidence_change: f32,
    /// Playbook statistics before curation
    pub stats_before: PlaybookStats,
    /// Playbook statistics after curation
    pub stats_after: PlaybookStats,
}

impl CurationSummary {
    /// Create empty summary
    pub fn empty() -> Self {
        Self {
            bullets_added: 0,
            bullets_updated: 0,
            bullets_removed: 0,
            total_helpful_delta: 0,
            total_harmful_delta: 0,
            confidence_adjustments: 0,
            avg_confidence_change: 0.0,
            stats_before: PlaybookStats {
                total_bullets: 0,
                high_confidence_bullets: 0,
                used_bullets: 0,
                avg_confidence: None,
                total_helpful: 0,
                total_harmful: 0,
            },
            stats_after: PlaybookStats {
                total_bullets: 0,
                high_confidence_bullets: 0,
                used_bullets: 0,
                avg_confidence: None,
                total_helpful: 0,
                total_harmful: 0,
            },
        }
    }

    /// Display human-readable summary
    pub fn display_summary(&self) {
        println!("📚 ACE Playbook Curation Summary");
        println!("================================");
        println!("📈 Changes Applied:");
        println!("  • {} bullets added", self.bullets_added);
        println!("  • {} bullets updated", self.bullets_updated);
        println!("  • {} bullets removed", self.bullets_removed);
        
        if self.total_helpful_delta != 0 || self.total_harmful_delta != 0 {
            println!("  • Helpful votes: {:+}", self.total_helpful_delta);
            println!("  • Harmful votes: {:+}", self.total_harmful_delta);
        }
        
        if self.confidence_adjustments > 0 {
            println!("  • {} confidence adjustments (avg: {:+.3})", 
                     self.confidence_adjustments, self.avg_confidence_change);
        }
        
        println!("📊 Before → After:");
        println!("  • Total bullets: {} → {}", 
                 self.stats_before.total_bullets, self.stats_after.total_bullets);
        println!("  • High confidence: {} → {}", 
                 self.stats_before.high_confidence_bullets, self.stats_after.high_confidence_bullets);
        
        if let (Some(before), Some(after)) = (self.stats_before.avg_confidence, self.stats_after.avg_confidence) {
            println!("  • Avg confidence: {:.1}% → {:.1}%", 
                     before * 100.0, after * 100.0);
        }
        
        println!("  • Usage rate: {:.1}% → {:.1}%", 
                 self.stats_before.usage_percentage(), self.stats_after.usage_percentage());
    }
}

/// ACE Curator for orchestrating playbook evolution
pub struct Curator {
    playbook_dao: PlaybookDAO,
    delta_engine: DeltaEngine,
    generator: Generator,
    reflector: Reflector,
    config: CuratorConfig,
}

impl Curator {
    /// Create new Curator with all required components
    pub async fn new(
        playbook_dao: PlaybookDAO,
        llm_client: LLMClient,
        config: Option<CuratorConfig>,
        delta_config: Option<DeltaEngineConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        
        // Create delta engine
        let delta_engine = DeltaEngine::new(playbook_dao.clone(), delta_config).await?;
        
        // Create generator and reflector
        let generator = Generator::new(llm_client.clone());
        let reflector = Reflector::new(llm_client);
        
        Ok(Self {
            playbook_dao,
            delta_engine,
            generator,
            reflector,
            config,
        })
    }

    /// Generate new patterns from market analysis and add to playbook
    pub async fn generate_and_apply_patterns(
        &self,
        market_state: serde_json::Value,
        ml_signals: serde_json::Value,
        similar_contexts: Vec<ContextEntry>,
        source_context_id: Option<Uuid>,
    ) -> Result<ApplyReport> {
        info!("Starting pattern generation and application");

        // Get existing playbook for context
        let existing_playbook = self.playbook_dao
            .get_recent_bullets(7, self.config.max_context_bullets)
            .await
            .context("Failed to get existing playbook for context")?;

        // Build generator input
        let generator_input = GeneratorInput {
            market_state,
            ml_signals,
            similar_contexts,
            existing_playbook,
            source_context_id,
        };

        // Generate patterns
        let deltas = self.generator.generate_patterns(generator_input).await
            .context("Failed to generate patterns")?;

        if deltas.is_empty() {
            info!("No patterns generated");
            return Ok(ApplyReport {
                applied_count: 0,
                skipped_count: 0,
                failed_count: 0,
                delta_results: vec![],
                success: true,
            });
        }

        // Apply generated deltas
        self.delta_engine.apply_deltas(deltas).await
            .context("Failed to apply generated deltas")
    }

    /// Reflect on trading outcome and update playbook
    pub async fn reflect_and_update_playbook(
        &self,
        decision: TradingDecision,
        market_state: serde_json::Value,
        outcome: TradingOutcome,
        context_id: Uuid,
    ) -> Result<CurationSummary> {
        info!("Starting reflection and playbook update");

        // Get stats before changes
        let stats_before = self.playbook_dao.get_stats().await?;

        // Identify which bullets were referenced in the decision
        let available_bullets = self.playbook_dao
            .get_recent_bullets(30, 100)
            .await
            .context("Failed to get bullets for reference identification")?;
        
        let referenced_bullets = self.reflector.identify_referenced_bullets(
            &decision.reasoning,
            &available_bullets,
        );

        info!("Identified {} referenced bullets in decision", referenced_bullets.len());

        // Update last_used timestamps for referenced bullets
        if self.config.auto_update_usage {
            for bullet in &referenced_bullets {
                let _ = self.playbook_dao.update_last_used(bullet.id).await;
            }
        }

        // Build reflection input
        let reflection_input = ReflectionInput {
            decision,
            market_state,
            outcome,
            referenced_bullets,
            context_id,
            date: chrono::Utc::now().format("%Y-%m-%d").to_string(),
        };

        // Generate reflection deltas
        let deltas = self.reflector.reflect_on_outcome(reflection_input).await
            .context("Failed to generate reflection deltas")?;

        if deltas.is_empty() {
            info!("No reflection deltas generated");
            let stats_after = stats_before.clone();
            return Ok(self.build_summary(vec![], stats_before, stats_after));
        }

        // Apply reflection deltas
        let apply_report = self.delta_engine.apply_deltas(deltas).await
            .context("Failed to apply reflection deltas")?;

        // Get stats after changes
        let stats_after = self.playbook_dao.get_stats().await?;

        Ok(self.build_summary(vec![apply_report], stats_before, stats_after))
    }

    /// Get recent playbook entries for morning context generation
    pub async fn summarize_recent_playbook(&self, limit: usize) -> Result<Vec<String>> {
        // Get high-confidence pattern insights
        let pattern_insights = self.playbook_dao
            .get_by_section(PlaybookSection::PatternInsights, Some(limit / 3))
            .await?;

        // Get recent failure modes as cautions
        let failure_modes = self.playbook_dao
            .get_by_section(PlaybookSection::FailureModes, Some(limit / 3))
            .await?;

        // Get regime rules for current conditions
        let regime_rules = self.playbook_dao
            .get_by_section(PlaybookSection::RegimeRules, Some(limit / 3))
            .await?;

        let mut summaries = Vec::new();

        // Format pattern insights
        for bullet in pattern_insights.iter().take(limit / 3) {
            if bullet.confidence > 0.6 {
                summaries.push(format!("✅ {}", bullet.content));
            }
        }

        // Format failure modes as warnings
        for bullet in failure_modes.iter().take(limit / 3) {
            if bullet.confidence > 0.5 {
                summaries.push(format!("⚠️ {}", bullet.content));
            }
        }

        // Format regime rules
        for bullet in regime_rules.iter().take(limit / 3) {
            if bullet.confidence > 0.5 {
                summaries.push(format!("📊 {}", bullet.content));
            }
        }

        // Sort by confidence descending, but keep limit
        summaries.truncate(limit);

        info!("Generated {} playbook summaries for context", summaries.len());
        Ok(summaries)
    }

    /// Get playbook statistics and health metrics
    pub async fn get_playbook_health(&self) -> Result<serde_json::Value> {
        let stats = self.playbook_dao.get_stats().await?;
        let section_counts = self.playbook_dao.get_section_counts().await?;
        
        // Get stale bullets count
        let stale_bullets = self.playbook_dao
            .get_stale_bullets(self.config.staleness_threshold_days)
            .await?;

        let health = json!({
            "total_bullets": stats.total_bullets,
            "high_confidence_bullets": stats.high_confidence_bullets,
            "used_bullets": stats.used_bullets,
            "avg_confidence": stats.avg_confidence,
            "effectiveness_ratio": stats.effectiveness_ratio(),
            "usage_percentage": stats.usage_percentage(),
            "stale_bullets": stale_bullets.len(),
            "section_distribution": section_counts.iter()
                .map(|(section, count)| (section.as_str(), count))
                .collect::<std::collections::HashMap<_, _>>(),
            "health_score": self.calculate_health_score(&stats, stale_bullets.len())
        });

        Ok(health)
    }

    /// Apply batch of deltas with comprehensive reporting
    pub async fn apply_deltas_with_summary(
        &self,
        deltas: Vec<Delta>,
    ) -> Result<CurationSummary> {
        let stats_before = self.playbook_dao.get_stats().await?;

        let apply_report = self.delta_engine.apply_deltas(deltas).await?;

        let stats_after = self.playbook_dao.get_stats().await?;

        Ok(self.build_summary(vec![apply_report], stats_before, stats_after))
    }

    /// Calculate overall health score for the playbook
    fn calculate_health_score(&self, stats: &PlaybookStats, stale_count: usize) -> f32 {
        let mut score = 100.0;

        // Penalty for low usage
        let usage_pct = stats.usage_percentage();
        if usage_pct < 50.0 {
            score -= (50.0 - usage_pct) * 0.5;
        }

        // Penalty for low confidence bullets
        if let Some(avg_conf) = stats.avg_confidence {
            if avg_conf < 0.5 {
                score -= (0.5 - avg_conf) * 100.0;
            }
        }

        // Penalty for too many stale bullets
        let stale_pct = if stats.total_bullets > 0 {
            (stale_count as f32 / stats.total_bullets as f32) * 100.0
        } else {
            0.0
        };
        if stale_pct > 30.0 {
            score -= (stale_pct - 30.0) * 0.5;
        }

        // Bonus for good effectiveness ratio
        let effectiveness = stats.effectiveness_ratio();
        if effectiveness > 0.6 {
            score += (effectiveness - 0.6) * 50.0;
        }

        score.max(0.0).min(100.0)
    }

    /// Build curation summary from apply reports
    fn build_summary(
        &self,
        reports: Vec<ApplyReport>,
        stats_before: PlaybookStats,
        stats_after: PlaybookStats,
    ) -> CurationSummary {
        let mut bullets_added = 0;
        let mut bullets_updated = 0;
        let mut bullets_removed = 0;
        let mut total_helpful_delta = 0;
        let mut total_harmful_delta = 0;
        let mut confidence_adjustments = 0;
        let mut confidence_changes = Vec::new();

        for report in reports {
            bullets_added += report.applied_count;
            
            for result in report.delta_results {
                match result.delta.op {
                    crate::ace::delta::DeltaOp::Add => bullets_added += 1,
                    crate::ace::delta::DeltaOp::Update => {
                        bullets_updated += 1;
                        
                        if let Some(helpful) = result.delta.helpful_delta {
                            total_helpful_delta += helpful;
                        }
                        if let Some(harmful) = result.delta.harmful_delta {
                            total_harmful_delta += harmful;
                        }
                        if let Some(conf_adj) = result.delta.confidence_adjustment {
                            confidence_adjustments += 1;
                            confidence_changes.push(conf_adj);
                        }
                    }
                    crate::ace::delta::DeltaOp::Remove => bullets_removed += 1,
                }
            }
        }

        let avg_confidence_change = if !confidence_changes.is_empty() {
            confidence_changes.iter().sum::<f32>() / confidence_changes.len() as f32
        } else {
            0.0
        };

        CurationSummary {
            bullets_added,
            bullets_updated,
            bullets_removed,
            total_helpful_delta,
            total_harmful_delta,
            confidence_adjustments,
            avg_confidence_change,
            stats_before,
            stats_after,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{llm::LLMConfig, ace::playbook::PlaybookBullet};
    use serde_json::json;

    #[test]
    fn test_curator_config() {
        let config = CuratorConfig::default();
        assert_eq!(config.max_context_bullets, 50);
        assert_eq!(config.min_confidence_for_pruning, 0.2);
        assert_eq!(config.staleness_threshold_days, 30);
        assert!(config.auto_update_usage);
        assert_eq!(config.max_deltas_per_batch, 100);
    }

    #[test]
    fn test_curation_summary() {
        let mut summary = CurationSummary::empty();
        summary.bullets_added = 5;
        summary.bullets_updated = 3;
        summary.confidence_adjustments = 7;
        summary.avg_confidence_change = 0.02;

        // Test that display doesn't panic
        summary.display_summary();
    }

    #[test]
    fn test_calculate_health_score() {
        // This would need a proper curator instance, but we can test the logic
        let stats = PlaybookStats {
            total_bullets: 100,
            high_confidence_bullets: 60,
            used_bullets: 70,
            avg_confidence: Some(0.65),
            total_helpful: 150,
            total_harmful: 50,
        };

        // Would need actual curator instance to test
        // let score = curator.calculate_health_score(&stats, 10);
        // assert!(score > 70.0);  // Should be a decent score
    }

    #[tokio::test]
    async fn test_curator_creation() {
        // This would require full database and LLM setup
        // Keeping it as a placeholder for integration testing

        // let pool = setup_test_database().await;
        // let playbook_dao = PlaybookDAO::new(pool);
        // let llm_client = LLMClient::new(LLMConfig::default()).await.unwrap();
        // 
        // let curator = Curator::new(playbook_dao, llm_client, None, None).await;
        // assert!(curator.is_ok());
    }
}
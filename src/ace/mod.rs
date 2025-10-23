//! ACE (Agentic Context Engineering) framework implementation
//! Core ACE framework components based on the research paper:
//! - Context Types: Market state, decision patterns, outcome tracking
//! - Vector Store: pgvector integration for context retrieval  
//! - Playbook System: Pattern storage and lifecycle management
//! - Delta Updates: Incremental context modifications
//! - Grow-and-Refine: Deduplication and context optimization
//! - Reasoner: Context fusion and decision synthesis using Llama 3.2

use anyhow::Result;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub mod context;
pub mod curator;
pub mod delta;
pub mod generator;
pub mod playbook;
pub mod prompts;
pub mod reflector;
pub mod sanitize;

// Re-export commonly used types
pub use context::ContextDAO;
pub use curator::{CurationSummary, Curator};
pub use delta::Delta;
pub use generator::{Generator, GeneratorInput};
pub use playbook::{PlaybookBullet, PlaybookDAO};
pub use prompts::{ACEPrompts, TradingDecision};
pub use reflector::Reflector;

use crate::embeddings::EmbeddingGemma;
use crate::llm::LLMClient;
use crate::vector::VectorStore;
use sqlx::PgPool;
use std::sync::Arc;
use tracing::info;

pub struct ACEEngine {
    llm_client: LLMClient,
    embedder: Arc<EmbeddingGemma>,
    vector_store: VectorStore,
    generator: Generator,
    reflector: Reflector,
    curator: Curator,
    playbook_dao: PlaybookDAO,
    context_dao: ContextDAO,
    pool: PgPool,
}

impl ACEEngine {
    pub async fn new(pool: PgPool, llm_client: LLMClient) -> Result<Self> {
        info!("Initializing ACE Engine");

        // Load embedding model
        info!("Loading embedding model");
        let embedder = Arc::new(EmbeddingGemma::load().await?);

        // Initialize vector store
        info!("Initializing vector store");
        let vector_store = VectorStore::new(pool.clone()).await?;

        // Initialize DAOs
        let playbook_dao = PlaybookDAO::new(pool.clone());
        let context_dao = ContextDAO::new(pool.clone());

        // Initialize ACE components
        let generator = Generator::new(llm_client.clone());
        let reflector = Reflector::new(llm_client.clone());
        let curator = Curator::new(
            playbook_dao.clone(),
            llm_client.clone(),
            None,
            None,
        )
        .await?;

        info!("ACE Engine initialized successfully");
        Ok(Self {
            llm_client,
            embedder,
            vector_store,
            generator,
            reflector,
            curator,
            playbook_dao,
            context_dao,
            pool,
        })
    }

    pub async fn generate_recommendation(
        &self,
        market_state: &crate::data::MarketData,
    ) -> Result<TradingRecommendation> {
        info!("Generating trading recommendation via ACE pipeline");

        // Step 1: Convert market state to JSON for embedding
        let market_context = serde_json::to_value(market_state)?;

        // Step 2: Generate embedding for current market state
        let context_text = format!(
            "Market: {} | Price: {} | High: {} | Low: {} | Volume: {}",
            market_state.symbol,
            market_state.close,
            market_state.high,
            market_state.low,
            market_state.volume
        );
        let context_embedding = self.embedder.embed(&context_text).await?;

        // Step 3: Search for similar historical contexts
        let similar_contexts = self
            .vector_store
            .similarity_search(context_embedding, 10)
            .await?;

        info!(
            "Found {} similar historical contexts",
            similar_contexts.len()
        );

        // Step 4: Get relevant playbook bullets (get recent ones)
        let playbook_bullets = self.playbook_dao.get_recent_bullets(30, 20).await?;

        info!("Retrieved {} playbook bullets", playbook_bullets.len());

        // Step 5: Generate patterns using Generator
        let generator_input = GeneratorInput {
            market_state: market_context.clone(),
            ml_signals: serde_json::json!({}), // Empty for now
            similar_contexts: similar_contexts.clone(),
            existing_playbook: playbook_bullets.clone(),
            source_context_id: None,
        };

        let deltas = self.generator.generate_patterns(generator_input).await?;

        // Step 6: Synthesize recommendation using LLM
        let recommendation_prompt = self.build_recommendation_prompt(
            market_state,
            &similar_contexts,
            &playbook_bullets,
            &deltas,
        );

        let llm_response = self
            .llm_client
            .generate(&recommendation_prompt, None)
            .await?;

        // Step 7: Parse LLM response into structured recommendation
        let recommendation = self.parse_recommendation(llm_response.content)?;

        // Step 8: Basic safety check - ensure confidence is reasonable
        if recommendation.confidence > 0.95 {
            info!("Recommendation confidence unusually high, reducing to 0.9");
            return Ok(TradingRecommendation {
                id: recommendation.id,
                action: recommendation.action,
                confidence: 0.9,
                reasoning: recommendation.reasoning,
                risk_factors: recommendation.risk_factors,
            });
        }

        info!("Generated recommendation: {:?}", recommendation.action);
        Ok(recommendation)
    }

    fn build_recommendation_prompt(
        &self,
        market_state: &crate::data::MarketData,
        similar_contexts: &[crate::vector::ContextEntry],
        playbook_bullets: &[PlaybookBullet],
        deltas: &[Delta],
    ) -> String {
        let mut prompt = String::from("You are an expert trading advisor using the ACE framework.\n\n");

        // Current market state
        prompt.push_str(&format!(
            "CURRENT MARKET STATE:\nSymbol: {}\nPrice: {}\nHigh: {}\nLow: {}\nVolume: {}\n\n",
            market_state.symbol,
            market_state.close,
            market_state.high,
            market_state.low,
            market_state.volume
        ));

        // Historical contexts
        if !similar_contexts.is_empty() {
            prompt.push_str("SIMILAR HISTORICAL CONTEXTS:\n");
            for (i, ctx) in similar_contexts.iter().take(5).enumerate() {
                prompt.push_str(&format!(
                    "{}. Context ID: {} | Confidence: {} | Similarity: {:?}\n",
                    i + 1,
                    ctx.id,
                    ctx.confidence,
                    ctx.similarity
                ));
            }
            prompt.push_str("\n");
        }

        // Playbook bullets
        if !playbook_bullets.is_empty() {
            prompt.push_str("RELEVANT PLAYBOOK ENTRIES:\n");
            for (i, bullet) in playbook_bullets.iter().take(5).enumerate() {
                prompt.push_str(&format!(
                    "{}. {} (Confidence: {})\n",
                    i + 1,
                    bullet.content,
                    bullet.confidence
                ));
            }
            prompt.push_str("\n");
        }

        // Deltas/Patterns
        if !deltas.is_empty() {
            prompt.push_str("IDENTIFIED PATTERNS (Deltas):\n");
            for (i, delta) in deltas.iter().take(3).enumerate() {
                let desc = match &delta.content {
                    Some(content) => format!("{:?}: {}", delta.op, content),
                    None => format!("{:?}", delta.op),
                };
                prompt.push_str(&format!("{}. {}\n", i + 1, desc));
            }
            prompt.push_str("\n");
        }

        prompt.push_str("Based on the above context, provide a trading recommendation in JSON format:\n");
        prompt.push_str("{\n");
        prompt.push_str("  \"action\": \"BUY\" | \"SELL\" | \"HOLD\",\n");
        prompt.push_str("  \"confidence\": 0.0-1.0,\n");
        prompt.push_str("  \"reasoning\": \"Brief explanation\",\n");
        prompt.push_str("  \"risk_factors\": [\"factor1\", \"factor2\"]\n");
        prompt.push_str("}\n");

        prompt
    }

    fn parse_recommendation(&self, llm_response: String) -> Result<TradingRecommendation> {
        // Try to extract JSON from response
        let json_start = llm_response.find('{');
        let json_end = llm_response.rfind('}');

        if let (Some(start), Some(end)) = (json_start, json_end) {
            let json_str = &llm_response[start..=end];
            let parsed: serde_json::Value = serde_json::from_str(json_str)?;

            Ok(TradingRecommendation {
                id: Uuid::new_v4(),
                action: parsed["action"]
                    .as_str()
                    .unwrap_or("HOLD")
                    .to_string(),
                confidence: parsed["confidence"].as_f64().unwrap_or(0.0) as f32,
                reasoning: parsed["reasoning"]
                    .as_str()
                    .unwrap_or("No reasoning provided")
                    .to_string(),
                risk_factors: parsed["risk_factors"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default(),
            })
        } else {
            // Fallback if JSON parsing fails
            Ok(TradingRecommendation {
                id: Uuid::new_v4(),
                action: "HOLD".to_string(),
                confidence: 0.0,
                reasoning: "Failed to parse LLM response".to_string(),
                risk_factors: vec!["Parsing error".to_string()],
            })
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingRecommendation {
    pub id: Uuid,
    pub action: String,
    pub confidence: f32,
    pub reasoning: String,
    pub risk_factors: Vec<String>,
}

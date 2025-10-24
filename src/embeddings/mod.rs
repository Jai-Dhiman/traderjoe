//! Embeddings module using GitHub Models API
//! Provides text embeddings via OpenAI-compatible text-embedding-3-small model

use anyhow::Result;
use async_openai::{
    config::OpenAIConfig,
    types::{CreateEmbeddingRequestArgs, EmbeddingInput},
    Client as OpenAIClient,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Text embedder using GitHub Models API
pub struct EmbeddingGemma {
    client: OpenAIClient<OpenAIConfig>,
    model_name: String,
    cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    dimension: usize,
}

impl EmbeddingGemma {
    /// Initialize embedder with GitHub Models provider
    pub async fn from_github_models(api_key: String) -> Result<Self> {
        info!("Initializing GitHub Models embeddings (text-embedding-3-small)");

        let config = OpenAIConfig::new()
            .with_api_key(&api_key)
            .with_api_base("https://models.inference.ai.azure.com");

        let client = OpenAIClient::with_config(config);

        Ok(Self {
            client,
            model_name: "text-embedding-3-small".to_string(),
            cache: Arc::new(RwLock::new(HashMap::new())),
            dimension: 1536, // text-embedding-3-small dimension
        })
    }

    /// Alias for from_github_models for backwards compatibility
    pub async fn load() -> Result<Self> {
        let api_key = std::env::var("GITHUB_TOKEN")
            .or_else(|_| std::env::var("OPENAI_API_KEY"))
            .map_err(|_| anyhow::anyhow!("GITHUB_TOKEN or OPENAI_API_KEY not set"))?;

        Self::from_github_models(api_key).await
    }

    /// Generate embeddings for text
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(embedding) = cache.get(text) {
                return Ok(embedding.clone());
            }
        }

        // Generate embedding
        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.model_name)
            .input(EmbeddingInput::String(text.to_string()))
            .build()?;

        let response = self.client.embeddings().create(request).await?;

        let embedding: Vec<f32> = response
            .data
            .first()
            .ok_or_else(|| anyhow::anyhow!("No embedding returned"))?
            .embedding
            .iter()
            .map(|&v| v as f32)
            .collect();

        info!(
            "Generated embedding with {} dimensions for text (first 50 chars): {}",
            embedding.len(),
            &text.chars().take(50).collect::<String>()
        );

        // Cache the result
        {
            let mut cache = self.cache.write().await;
            cache.insert(text.to_string(), embedding.clone());
        }

        Ok(embedding)
    }

    /// Generate batch embeddings efficiently
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let texts_owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.model_name)
            .input(EmbeddingInput::StringArray(texts_owned))
            .build()?;

        let response = self.client.embeddings().create(request).await?;

        let embeddings: Vec<Vec<f32>> = response
            .data
            .iter()
            .map(|data| data.embedding.iter().map(|&v| v as f32).collect())
            .collect();

        info!(
            "Generated {} embeddings with {} dimensions",
            embeddings.len(),
            embeddings.first().map(|e| e.len()).unwrap_or(0)
        );

        // Cache the results
        {
            let mut cache = self.cache.write().await;
            for (text, embedding) in texts.iter().zip(embeddings.iter()) {
                cache.insert(text.to_string(), embedding.clone());
            }
        }

        Ok(embeddings)
    }

    /// Get the embedding dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Clear the embedding cache
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
        info!("Embedding cache cleared");
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.read().await;
        let entries = cache.len();
        let estimated_memory = entries * (self.dimension * 4 + 64); // dimension f32s + key overhead
        (entries, estimated_memory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Ignore by default - requires API key
    async fn test_embedding_generation() {
        let embedder = EmbeddingGemma::load()
            .await
            .expect("Failed to load embedder");

        let text = "The market is showing strong bullish sentiment today";
        let embedding = embedder
            .embed(text)
            .await
            .expect("Failed to generate embedding");

        // Check dimension
        assert_eq!(embedding.len(), 1536);

        // Check deterministic - same text should produce same embedding (from cache)
        let embedding2 = embedder
            .embed(text)
            .await
            .expect("Failed to generate embedding");
        assert_eq!(embedding, embedding2);
    }

    #[tokio::test]
    #[ignore] // Ignore by default - requires API key
    async fn test_batch_embedding() {
        let embedder = EmbeddingGemma::load()
            .await
            .expect("Failed to load embedder");

        let texts = vec!["bullish market", "bearish sentiment", "neutral outlook"];
        let embeddings = embedder
            .embed_batch(&texts)
            .await
            .expect("Failed to generate batch embeddings");

        assert_eq!(embeddings.len(), 3);
        for embedding in &embeddings {
            assert_eq!(embedding.len(), 1536);
        }

        // Different texts should produce different embeddings
        assert_ne!(embeddings[0], embeddings[1]);
    }

    #[tokio::test]
    #[ignore] // Ignore by default - requires API key
    async fn test_caching() {
        let embedder = EmbeddingGemma::load()
            .await
            .expect("Failed to load embedder");

        let text = "test caching behavior";

        // First call - should cache
        let _embedding1 = embedder.embed(text).await.expect("Failed to embed");
        let (entries, _) = embedder.cache_stats().await;
        assert_eq!(entries, 1);

        // Second call - should use cache
        let _embedding2 = embedder.embed(text).await.expect("Failed to embed");
        let (entries, _) = embedder.cache_stats().await;
        assert_eq!(entries, 1); // Still just one entry
    }
}

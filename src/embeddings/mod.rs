//! Embeddings module using Cloudflare Workers AI
//! Provides text embeddings via @cf/baai/bge-base-en-v1.5 model (768 dimensions)

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Request payload for Cloudflare Workers AI embeddings
#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    text: Vec<String>,
}

/// Response from Cloudflare Workers AI embeddings
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    result: EmbeddingResult,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResult {
    shape: Vec<usize>,
    data: Vec<Vec<f32>>,
}

/// Text embedder using Cloudflare Workers AI
pub struct EmbeddingGemma {
    client: reqwest::Client,
    account_id: String,
    api_token: String,
    model_name: String,
    cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    dimension: usize,
}

impl EmbeddingGemma {
    /// Initialize embedder with Cloudflare Workers AI
    pub async fn from_cloudflare(account_id: String, api_token: String) -> Result<Self> {
        info!("Initializing Cloudflare Workers AI embeddings (@cf/baai/bge-base-en-v1.5)");

        let client = reqwest::Client::new();

        Ok(Self {
            client,
            account_id,
            api_token,
            model_name: "@cf/baai/bge-base-en-v1.5".to_string(),
            cache: Arc::new(RwLock::new(HashMap::new())),
            dimension: 768, // bge-base-en-v1.5 dimension
        })
    }

    /// Alias for backwards compatibility - uses Cloudflare Workers AI
    pub async fn load() -> Result<Self> {
        let account_id = std::env::var("CLOUDFLARE_ACCOUNT_ID")
            .map_err(|_| anyhow::anyhow!("CLOUDFLARE_ACCOUNT_ID not set"))?;
        let api_token = std::env::var("CLOUDFLARE_API_TOKEN")
            .map_err(|_| anyhow::anyhow!("CLOUDFLARE_API_TOKEN not set"))?;

        Self::from_cloudflare(account_id, api_token).await
    }

    /// Backwards compatibility alias
    pub async fn from_github_models(api_key: String) -> Result<Self> {
        // Try to get Cloudflare credentials from environment
        let account_id = std::env::var("CLOUDFLARE_ACCOUNT_ID")
            .map_err(|_| anyhow::anyhow!("CLOUDFLARE_ACCOUNT_ID not set. Migration required: use Cloudflare Workers AI instead of GitHub Models."))?;

        Self::from_cloudflare(account_id, api_key).await
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

        // Generate embedding using Workers AI
        let url = format!(
            "https://api.cloudflare.com/client/v4/accounts/{}/ai/run/{}",
            self.account_id, self.model_name
        );

        let request_body = EmbeddingRequest {
            text: vec![text.to_string()],
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_token))
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow::anyhow!(
                "Cloudflare Workers AI request failed with status {}: {}",
                status,
                error_text
            ));
        }

        let embedding_response: EmbeddingResponse = response.json().await?;

        let embedding = embedding_response
            .result
            .data
            .first()
            .ok_or_else(|| anyhow::anyhow!("No embedding returned"))?
            .clone();

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
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Cloudflare Workers AI supports up to 100 texts per batch
        if texts.len() > 100 {
            return Err(anyhow::anyhow!(
                "Batch size {} exceeds maximum of 100",
                texts.len()
            ));
        }

        let url = format!(
            "https://api.cloudflare.com/client/v4/accounts/{}/ai/run/{}",
            self.account_id, self.model_name
        );

        let request_body = EmbeddingRequest {
            text: texts.iter().map(|s| s.to_string()).collect(),
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_token))
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow::anyhow!(
                "Cloudflare Workers AI batch request failed with status {}: {}",
                status,
                error_text
            ));
        }

        let embedding_response: EmbeddingResponse = response.json().await?;
        let embeddings = embedding_response.result.data;

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
    #[ignore] // Ignore by default - requires API credentials
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
        assert_eq!(embedding.len(), 768);

        // Check deterministic - same text should produce same embedding (from cache)
        let embedding2 = embedder
            .embed(text)
            .await
            .expect("Failed to generate embedding");
        assert_eq!(embedding, embedding2);
    }

    #[tokio::test]
    #[ignore] // Ignore by default - requires API credentials
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
            assert_eq!(embedding.len(), 768);
        }

        // Different texts should produce different embeddings
        assert_ne!(embeddings[0], embeddings[1]);
    }

    #[tokio::test]
    #[ignore] // Ignore by default - requires API credentials
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

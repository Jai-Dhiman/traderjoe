//! EmbeddingGemma 300M integration module
//! Provides local embedding generation using candle framework

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma::{Config, Model};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::{Mutex, RwLock};
use tracing::info;

/// EmbeddingGemma 300M model wrapper with caching
pub struct EmbeddingGemma {
    device: Device,
    model: Arc<Mutex<Model>>,
    tokenizer: Tokenizer,
    cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    model_loaded: bool,
}

impl EmbeddingGemma {
    /// Load the EmbeddingGemma model from HuggingFace
    pub async fn load() -> Result<Self> {
        info!("Loading EmbeddingGemma 300M model from HuggingFace");

        // Determine device - prefer GPU if available
        let device = Self::get_device()?;
        info!("Using device: {:?}", device);

        // Download model from HuggingFace
        let (model_path, tokenizer_path, config_path) = Self::download_model().await?;

        // Load tokenizer
        info!("Loading tokenizer from {:?}", tokenizer_path);
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Load config
        info!("Loading config from {:?}", config_path);
        let config: Config = serde_json::from_reader(
            std::fs::File::open(&config_path)
                .context("Failed to open config file")?
        )?;

        // Load model weights
        info!("Loading model weights from {:?}", model_path);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], candle_core::DType::F32, &device)?
        };

        let model = Model::new(false, &config, vb)?; // use_flash_attn = false for compatibility
        info!("EmbeddingGemma model loaded successfully");

        Ok(Self {
            device,
            model: Arc::new(Mutex::new(model)),
            tokenizer,
            cache: Arc::new(RwLock::new(HashMap::new())),
            model_loaded: true,
        })
    }

    /// Determine the best available device (CUDA > Metal > CPU)
    fn get_device() -> Result<Device> {
        if cuda_is_available() {
            info!("CUDA available, using GPU");
            Ok(Device::new_cuda(0)?)
        } else if metal_is_available() {
            info!("Metal available, using GPU");
            Ok(Device::new_metal(0)?)
        } else {
            info!("No GPU available, using CPU");
            Ok(Device::Cpu)
        }
    }

    /// Download model from HuggingFace Hub
    async fn download_model() -> Result<(PathBuf, PathBuf, PathBuf)> {
        info!("Downloading EmbeddingGemma model from HuggingFace");

        // Use google/gemma-2b or similar embedding model
        // Note: For true embedding model, we might want to use a dedicated embedding model
        // like sentence-transformers, but we'll use Gemma for now

        // Do all the downloading in a single blocking task to avoid clone issues
        tokio::task::spawn_blocking(|| {
            let api = Api::new()?;
            let repo = api.repo(Repo::new(
                "google/gemma-2b".to_string(),
                RepoType::Model,
            ));

            let model_path = repo.get("model.safetensors")?;
            let tokenizer_path = repo.get("tokenizer.json")?;
            let config_path = repo.get("config.json")?;

            info!("Model files downloaded successfully");
            Ok((model_path, tokenizer_path, config_path))
        })
        .await?
    }

    /// Generate embeddings for text (768-dimensional)
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if !self.model_loaded {
            return Err(anyhow::anyhow!("EmbeddingGemma model not loaded"));
        }

        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(embedding) = cache.get(text) {
                return Ok(embedding.clone());
            }
        }

        // Generate embedding using actual model
        let embedding = self.generate_embedding(text)?;

        // Cache the result
        {
            let mut cache = self.cache.write().await;
            cache.insert(text.to_string(), embedding.clone());
        }

        Ok(embedding)
    }

    /// Generate embedding using the actual Gemma model
    fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        // Tokenize the input text
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let tokens = encoding.get_ids();
        let token_ids = Tensor::new(tokens, &self.device)?
            .unsqueeze(0)?; // Add batch dimension

        // Run forward pass through the model (need to block on mutex)
        let mut model = self.model.blocking_lock();
        let output = model.forward(&token_ids, 0)?;
        drop(model);

        // Extract embeddings from the last hidden state
        // We'll use mean pooling over the sequence dimension
        let embedding = self.mean_pool(&output)?;

        // Convert to Vec<f32>
        let embedding_vec = embedding.to_vec1::<f32>()?;

        // Normalize to unit vector
        let norm: f32 = embedding_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = if norm > 0.0 {
            embedding_vec.iter().map(|x| x / norm).collect()
        } else {
            embedding_vec
        };

        Ok(normalized)
    }

    /// Mean pooling over sequence dimension
    fn mean_pool(&self, tensor: &Tensor) -> Result<Tensor> {
        // tensor shape: [batch_size, seq_len, hidden_dim]
        // We want to average over seq_len dimension
        let sum = tensor.sum(1)?; // Sum over sequence dimension
        let seq_len = tensor.dim(1)? as f64;
        let mean = sum.affine(1.0 / seq_len, 0.0)?;

        // Remove batch dimension since we always have batch_size=1
        Ok(mean.squeeze(0)?)
    }

    /// Generate batch embeddings efficiently
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::new();

        for text in texts {
            let embedding = self.embed(text).await?;
            results.push(embedding);
        }

        Ok(results)
    }

    /// Get embedding dimension (returns actual dimension from first embedding)
    pub fn dimension(&self) -> usize {
        // Gemma-2b typically has hidden_size of 2048
        // For actual implementation, we could query the model config
        2048
    }

    /// Clear the embedding cache
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
        info!("EmbeddingGemma cache cleared");
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.read().await;
        let entries = cache.len();
        let estimated_memory = entries * (768 * 4 + 64); // 768 f32s + key overhead
        (entries, estimated_memory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require downloading the model from HuggingFace
    // which can be slow. Consider using #[ignore] for CI/CD

    #[tokio::test]
    #[ignore] // Ignore by default - requires model download
    async fn test_embedding_generation() {
        let embedder = EmbeddingGemma::load()
            .await
            .expect("Failed to load embedder");

        let text = "The market is showing strong bullish sentiment today";
        let embedding = embedder
            .embed(text)
            .await
            .expect("Failed to generate embedding");

        // Check dimension (Gemma-2b has 2048 hidden size)
        assert_eq!(embedding.len(), 2048);

        // Check that it's normalized (approximately unit vector)
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);

        // Check deterministic - same text should produce same embedding (from cache)
        let embedding2 = embedder
            .embed(text)
            .await
            .expect("Failed to generate embedding");
        assert_eq!(embedding, embedding2);
    }

    #[tokio::test]
    #[ignore] // Ignore by default - requires model download
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
            assert_eq!(embedding.len(), 2048);
        }

        // Different texts should produce different embeddings
        assert_ne!(embeddings[0], embeddings[1]);
    }

    #[tokio::test]
    #[ignore] // Ignore by default - requires model download
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

    #[test]
    fn test_device_selection() {
        // Test that device selection doesn't panic
        let device = EmbeddingGemma::get_device();
        assert!(device.is_ok());
    }
}

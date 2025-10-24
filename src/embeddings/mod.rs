//! Embeddings module supporting multiple providers
//! - GitHub Models (text-embedding-3-small) - OpenAI-compatible
//! - EmbeddingGemma 300M (local, via PyO3) - for development
//! - Google AI Studio (gemini-embedding-001) - alternative

use anyhow::Result;
use async_openai::{
    config::OpenAIConfig,
    types::{CreateEmbeddingRequestArgs, EmbeddingInput},
    Client as OpenAIClient,
};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Embedding provider enum
#[derive(Clone)]
enum EmbeddingProvider {
    GitHubModels(OpenAIClient<OpenAIConfig>),
    LocalPython, // For EmbeddingGemma via PyO3
}

/// Flexible embedder supporting multiple providers
pub struct EmbeddingGemma {
    provider: EmbeddingProvider,
    model_name: String,
    cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    dimension: usize,
}

impl EmbeddingGemma {
    /// Load embedder with GitHub Models provider
    pub async fn from_github_models(api_key: String) -> Result<Self> {
        info!("Initializing GitHub Models embeddings (text-embedding-3-small)");

        let config = OpenAIConfig::new()
            .with_api_key(&api_key)
            .with_api_base("https://models.inference.ai.azure.com");

        let client = OpenAIClient::with_config(config);

        Ok(Self {
            provider: EmbeddingProvider::GitHubModels(client),
            model_name: "text-embedding-3-small".to_string(),
            cache: Arc::new(RwLock::new(HashMap::new())),
            dimension: 1536, // text-embedding-3-small dimension
        })
    }

    /// Load the local EmbeddingGemma model via Python (for development)
    pub async fn load() -> Result<Self> {
        info!("Loading EmbeddingGemma 300M model via PyO3 (local development mode)");

        // Initialize Python and check that sentence-transformers is available
        Python::with_gil(|py| {
            // Check if sentence-transformers is installed
            let sys = py.import_bound("sys")?;
            let version: String = sys.getattr("version")?.extract()?;
            info!("Using Python version: {}", version);

            // Try to import sentence_transformers
            match py.import_bound("sentence_transformers") {
                Ok(_) => {
                    info!("sentence-transformers library found");
                    Ok(())
                }
                Err(e) => {
                    anyhow::bail!(
                        "sentence-transformers not installed. Please install it with: pip install sentence-transformers\nError: {}",
                        e
                    );
                }
            }
        })?;

        info!("EmbeddingGemma Python environment initialized");

        Ok(Self {
            provider: EmbeddingProvider::LocalPython,
            model_name: "google/embeddinggemma-300m".to_string(),
            cache: Arc::new(RwLock::new(HashMap::new())),
            dimension: 768,
        })
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

        // Generate embedding based on provider
        let embedding = match &self.provider {
            EmbeddingProvider::GitHubModels(client) => {
                let request = CreateEmbeddingRequestArgs::default()
                    .model(&self.model_name)
                    .input(EmbeddingInput::String(text.to_string()))
                    .build()?;

                let response = client.embeddings().create(request).await?;

                let embedding_vec: Vec<f32> = response
                    .data
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("No embedding returned"))?
                    .embedding
                    .iter()
                    .map(|&v| v as f32)
                    .collect();

                info!(
                    "Generated GitHub Models embedding with {} dimensions for text (first 50 chars): {}",
                    embedding_vec.len(),
                    &text.chars().take(50).collect::<String>()
                );

                embedding_vec
            }
            EmbeddingProvider::LocalPython => {
                // Clone text for the blocking task
                let text_owned = text.to_string();

                // Generate embedding using Python in a blocking task
                tokio::task::spawn_blocking(move || {
                    Self::generate_embedding_sync(&text_owned)
                })
                .await??
            }
        };

        // Cache the result
        {
            let mut cache = self.cache.write().await;
            cache.insert(text.to_string(), embedding.clone());
        }

        Ok(embedding)
    }

    /// Generate embedding using Python (synchronous, CPU-intensive)
    fn generate_embedding_sync(text: &str) -> Result<Vec<f32>> {
        Python::with_gil(|py| {
            // Import sentence_transformers
            let st = py.import_bound("sentence_transformers")?;
            let sentence_transformer = st.getattr("SentenceTransformer")?;

            // Load model (will use cache after first load)
            let model = sentence_transformer.call1(("google/embeddinggemma-300m",))?;

            // Encode the text
            let embedding_py = model.call_method1("encode", (text,))?;

            // Convert to Python list for easier extraction
            let tolist = embedding_py.call_method0("tolist")?;

            // Extract as Vec<f32>
            let embedding: Vec<f32> = tolist.extract()?;

            info!("Generated embedding with {} dimensions for text (first 50 chars): {}",
                  embedding.len(),
                  &text.chars().take(50).collect::<String>());

            Ok(embedding)
        })
    }

    /// Generate batch embeddings efficiently
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Generate embeddings based on provider
        let embeddings = match &self.provider {
            EmbeddingProvider::GitHubModels(client) => {
                let texts_owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

                let request = CreateEmbeddingRequestArgs::default()
                    .model(&self.model_name)
                    .input(EmbeddingInput::StringArray(texts_owned))
                    .build()?;

                let response = client.embeddings().create(request).await?;

                let embeddings: Vec<Vec<f32>> = response
                    .data
                    .iter()
                    .map(|data| data.embedding.iter().map(|&v| v as f32).collect())
                    .collect();

                info!(
                    "Generated {} GitHub Models embeddings with {} dimensions",
                    embeddings.len(),
                    embeddings.first().map(|e| e.len()).unwrap_or(0)
                );

                embeddings
            }
            EmbeddingProvider::LocalPython => {
                // Convert texts to owned strings
                let texts_owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();

                // Generate embeddings using Python in a blocking task
                tokio::task::spawn_blocking(move || {
                    Self::generate_batch_embedding_sync(&texts_owned)
                })
                .await??
            }
        };

        // Cache the results
        {
            let mut cache = self.cache.write().await;
            for (text, embedding) in texts.iter().zip(embeddings.iter()) {
                cache.insert(text.to_string(), embedding.clone());
            }
        }

        Ok(embeddings)
    }

    /// Generate batch embeddings using Python (synchronous)
    fn generate_batch_embedding_sync(texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Python::with_gil(|py| {
            // Import sentence_transformers
            let st = py.import_bound("sentence_transformers")?;
            let sentence_transformer = st.getattr("SentenceTransformer")?;

            // Load model (will use cache after first load)
            let model = sentence_transformer.call1(("google/embeddinggemma-300m",))?;

            // Convert texts to Python list
            let py_texts = PyList::new_bound(py, texts);

            // Encode the texts
            let embeddings_py = model.call_method1("encode", (py_texts,))?;

            // Convert to Python list for easier extraction
            let tolist = embeddings_py.call_method0("tolist")?;

            // Extract as Vec<Vec<f32>>
            let embeddings: Vec<Vec<f32>> = tolist.extract()?;

            Ok(embeddings)
        })
    }

    /// Get the embedding dimension
    pub fn dimension(&self) -> usize {
        self.dimension
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

    #[tokio::test]
    #[ignore] // Ignore by default - requires Python and sentence-transformers
    async fn test_embedding_generation() {
        let embedder = EmbeddingGemma::load()
            .await
            .expect("Failed to load embedder");

        let text = "The market is showing strong bullish sentiment today";
        let embedding = embedder
            .embed(text)
            .await
            .expect("Failed to generate embedding");

        // Check dimension (EmbeddingGemma has 768 hidden size)
        assert_eq!(embedding.len(), 768);

        // Check that it's normalized (approximately unit vector)
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.1); // Allow some variance

        // Check deterministic - same text should produce same embedding (from cache)
        let embedding2 = embedder
            .embed(text)
            .await
            .expect("Failed to generate embedding");
        assert_eq!(embedding, embedding2);
    }

    #[tokio::test]
    #[ignore] // Ignore by default - requires Python and sentence-transformers
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
    #[ignore] // Ignore by default - requires Python and sentence-transformers
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

    #[tokio::test]
    async fn test_python_availability() {
        // Test that Python is available
        let result = Python::with_gil(|py| {
            let sys = py.import_bound("sys")?;
            let version: String = sys.getattr("version")?.extract()?;
            Ok::<String, PyErr>(version)
        });

        assert!(result.is_ok(), "Python should be available");
    }
}

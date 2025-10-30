//! Reranker module using Cloudflare Workers AI
//! Provides text reranking via @cf/baai/bge-reranker-base model

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Context object for reranking
#[derive(Debug, Clone, Serialize)]
struct RerankContext {
    text: String,
}

/// Request payload for Cloudflare Workers AI reranker
#[derive(Debug, Serialize)]
struct RerankRequest {
    query: String,
    contexts: Vec<RerankContext>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<usize>,
}

/// Cloudflare API wrapper response
#[derive(Debug, Deserialize)]
struct CloudflareApiResponse {
    result: RerankResult,
    success: bool,
}

/// Result from Cloudflare Workers AI reranker
#[derive(Debug, Deserialize)]
struct RerankResult {
    response: Vec<RerankScore>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RerankScore {
    pub id: usize,  // Note: Cloudflare uses 'id' not 'index'
    pub score: f64,
}

/// Text reranker using Cloudflare Workers AI
pub struct Reranker {
    client: reqwest::Client,
    account_id: String,
    api_token: String,
    model_name: String,
}

impl Reranker {
    // Maximum text length for reranking (characters)
    // Cloudflare has token limits, so we cap text length conservatively
    const MAX_TEXT_LENGTH: usize = 2000;
    const MIN_TEXT_LENGTH: usize = 3;

    /// Initialize reranker with Cloudflare Workers AI
    pub fn new(account_id: String, api_token: String) -> Self {
        info!("Initializing Cloudflare Workers AI reranker (@cf/baai/bge-reranker-base)");

        Self {
            client: reqwest::Client::new(),
            account_id,
            api_token,
            model_name: "@cf/baai/bge-reranker-base".to_string(),
        }
    }

    /// Sanitize and validate text for reranking
    fn sanitize_text(text: &str) -> Option<String> {
        let trimmed = text.trim();

        // Filter out empty or too-short text
        if trimmed.len() < Self::MIN_TEXT_LENGTH {
            return None;
        }

        // Truncate if too long
        let sanitized = if trimmed.len() > Self::MAX_TEXT_LENGTH {
            &trimmed[..Self::MAX_TEXT_LENGTH]
        } else {
            trimmed
        };

        // Remove any null bytes or other problematic characters
        let cleaned: String = sanitized
            .chars()
            .filter(|c| !c.is_control() || c.is_whitespace())
            .collect();

        if cleaned.len() >= Self::MIN_TEXT_LENGTH {
            Some(cleaned)
        } else {
            None
        }
    }

    /// Rerank documents given a query
    /// Returns the indices and scores of documents sorted by relevance (highest first)
    pub async fn rerank(&self, query: &str, documents: Vec<String>) -> Result<Vec<RerankScore>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        // Validate and sanitize query
        let sanitized_query = Self::sanitize_text(query)
            .ok_or_else(|| anyhow::anyhow!("Query text is empty or invalid after sanitization"))?;

        // Sanitize documents and track original indices
        let mut valid_contexts = Vec::new();
        let mut index_mapping = Vec::new();
        let total_docs = documents.len();

        for (idx, doc) in documents.into_iter().enumerate() {
            if let Some(sanitized) = Self::sanitize_text(&doc) {
                valid_contexts.push(RerankContext { text: sanitized });
                index_mapping.push(idx);
            } else {
                warn!("Filtered out document {} (empty or invalid)", idx);
            }
        }

        // Check if we have any valid documents left
        if valid_contexts.is_empty() {
            warn!("All documents were filtered out during sanitization");
            return Ok(vec![]);
        }

        let filtered_count = total_docs - valid_contexts.len();
        info!(
            "Reranking query (length: {}) against {} valid documents{}",
            sanitized_query.len(),
            valid_contexts.len(),
            if filtered_count > 0 {
                format!(" (filtered {} invalid)", filtered_count)
            } else {
                String::new()
            }
        );

        let url = format!(
            "https://api.cloudflare.com/client/v4/accounts/{}/ai/run/{}",
            self.account_id, self.model_name
        );

        let request_body = RerankRequest {
            query: sanitized_query.clone(),
            contexts: valid_contexts.clone(),
            top_k: None, // Let API return all, we'll sort and filter
        };

        // Log detailed request information for debugging
        info!(
            "🔍 Reranker request details: query_len={}, num_contexts={}, context_lens=[{}]",
            sanitized_query.len(),
            valid_contexts.len(),
            valid_contexts.iter()
                .take(3)
                .map(|c| c.text.len().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );

        // Log first 100 chars of query and first context for debugging
        info!(
            "🔍 Query preview: {:?}, First context preview: {:?}",
            &sanitized_query.chars().take(100).collect::<String>(),
            valid_contexts.first().map(|c| c.text.chars().take(100).collect::<String>())
        );

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
                "Cloudflare Workers AI reranker request failed with status {}: {}",
                status,
                error_text
            ));
        }

        let api_response: CloudflareApiResponse = response.json().await?;

        // Map returned IDs back to original document indices
        // The API returns IDs based on the sanitized contexts array (0-indexed)
        // We need to map these back to the original document indices
        let mut mapped_scores: Vec<RerankScore> = api_response
            .result
            .response
            .into_iter()
            .filter_map(|score| {
                // Map the API's ID (which indexes into valid_contexts) to the original doc index
                index_mapping.get(score.id).map(|&original_idx| RerankScore {
                    id: original_idx,
                    score: score.score,
                })
            })
            .collect();

        // Sort by score descending (highest relevance first)
        mapped_scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        info!(
            "Reranked {} documents, top score: {:.4}",
            mapped_scores.len(),
            mapped_scores.first().map(|s| s.score).unwrap_or(0.0)
        );

        Ok(mapped_scores)
    }

    /// Rerank and return only the top N results
    pub async fn rerank_top_n(
        &self,
        query: &str,
        documents: Vec<String>,
        top_n: usize,
    ) -> Result<Vec<RerankScore>> {
        let all_scores = self.rerank(query, documents).await?;
        Ok(all_scores.into_iter().take(top_n).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Ignore by default - requires API credentials
    async fn test_reranking() {
        let account_id = std::env::var("CLOUDFLARE_ACCOUNT_ID")
            .expect("CLOUDFLARE_ACCOUNT_ID not set");
        let api_token = std::env::var("CLOUDFLARE_API_TOKEN")
            .expect("CLOUDFLARE_API_TOKEN not set");

        let reranker = Reranker::new(account_id, api_token);

        let query = "What is the capital of France?";
        let documents = vec![
            "Paris is the capital and most populous city of France.".to_string(),
            "Berlin is the capital of Germany.".to_string(),
            "London is the capital of England.".to_string(),
        ];

        let scores = reranker
            .rerank(query, documents)
            .await
            .expect("Failed to rerank");

        assert_eq!(scores.len(), 3);
        // The first document should be most relevant
        assert_eq!(scores[0].id, 0);
        assert!(scores[0].score > scores[1].score);
    }

    #[tokio::test]
    #[ignore] // Ignore by default - requires API credentials
    async fn test_rerank_top_n() {
        let account_id = std::env::var("CLOUDFLARE_ACCOUNT_ID")
            .expect("CLOUDFLARE_ACCOUNT_ID not set");
        let api_token = std::env::var("CLOUDFLARE_API_TOKEN")
            .expect("CLOUDFLARE_API_TOKEN not set");

        let reranker = Reranker::new(account_id, api_token);

        let query = "stock market trading";
        let documents = vec![
            "The stock market showed strong gains today.".to_string(),
            "I like to eat pizza for dinner.".to_string(),
            "Options trading can be profitable.".to_string(),
            "The weather is nice today.".to_string(),
            "Technical analysis helps predict price movements.".to_string(),
        ];

        let top_3 = reranker
            .rerank_top_n(query, documents, 3)
            .await
            .expect("Failed to rerank");

        assert_eq!(top_3.len(), 3);
        // Check that results are sorted by score descending
        assert!(top_3[0].score >= top_3[1].score);
        assert!(top_3[1].score >= top_3[2].score);
    }
}

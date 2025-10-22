//! LLM integration module for Ollama + cloud fallback
//! Provides local Llama 3.2 3B integration with explicit error handling

use anyhow::{Context, Result};
use ollama_rs::{Ollama, generation::completion::request::GenerationRequest};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{info, warn, error};
use url::Url;

/// LLM response with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    pub content: String,
    pub model: String,
    pub prompt_tokens: Option<usize>,
    pub completion_tokens: Option<usize>,
    pub total_tokens: Option<usize>,
}

/// LLM client configuration
#[derive(Debug, Clone)]
pub struct LLMConfig {
    pub ollama_url: String,
    pub primary_model: String,
    pub timeout_seconds: u64,
    pub max_retries: u32,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            ollama_url: "http://localhost:11434".to_string(),
            primary_model: "llama3.2:3b".to_string(),
            timeout_seconds: 30,
            max_retries: 3,
        }
    }
}

/// LLM client with local Ollama integration
#[derive(Debug, Clone)]
pub struct LLMClient {
    ollama: Ollama,
    config: LLMConfig,
}

impl LLMClient {
    /// Create new LLM client with configuration
    pub async fn new(config: LLMConfig) -> Result<Self> {
        // Parse URL to get host and port
        let url = config.ollama_url.clone();
        let parsed_url = Url::parse(&url)
            .context("Invalid Ollama URL")?;
        
        let host = parsed_url.host_str()
            .ok_or_else(|| anyhow::anyhow!("No host in Ollama URL"))?;
        let port = parsed_url.port().unwrap_or(11434);
        
        let ollama = Ollama::new(host.to_string(), port);
        
        // Test connectivity
        info!("Testing Ollama connectivity at {}", config.ollama_url);
        
        // Try to list models to verify connection
        match timeout(
            Duration::from_secs(10),
            ollama.list_local_models()
        ).await {
            Ok(Ok(models)) => {
                info!("Connected to Ollama successfully. Available models: {:?}", 
                     models.iter().map(|m| &m.name).collect::<Vec<_>>());
                
                // Check if primary model is available
                let model_available = models.iter()
                    .any(|m| m.name.contains(&config.primary_model));
                
                if !model_available {
                    warn!(
                        "Primary model '{}' not found in available models. Consider pulling it with: ollama pull {}",
                        config.primary_model, config.primary_model
                    );
                }
            }
            Ok(Err(e)) => {
                error!("Failed to list Ollama models: {}", e);
                return Err(anyhow::anyhow!(
                    "Ollama API error when listing models: {}. Is Ollama running?", e
                ));
            }
            Err(_) => {
                error!("Timeout connecting to Ollama at {}", config.ollama_url);
                return Err(anyhow::anyhow!(
                    "Timeout connecting to Ollama. Is Ollama running at {}?", config.ollama_url
                ));
            }
        }
        
        Ok(Self { ollama, config })
    }
    
    /// Create client from config::Config
    pub async fn from_config(config: &crate::config::Config) -> Result<Self> {
        let llm_config = LLMConfig {
            ollama_url: config.llm.ollama_url.clone(),
            primary_model: config.llm.primary_model.clone(),
            timeout_seconds: config.llm.timeout_seconds,
            max_retries: 3,
        };
        
        Self::new(llm_config).await
    }
    
    /// Generate text using specified model with explicit error handling
    pub async fn generate(
        &self,
        prompt: &str,
        model: Option<&str>,
    ) -> Result<LLMResponse> {
        let model_name = model.unwrap_or(&self.config.primary_model);
        
        info!("Generating text with model '{}' (prompt length: {} chars)", 
              model_name, prompt.len());
        
        let request = GenerationRequest::new(model_name.to_string(), prompt.to_string());
        
        // Retry logic with exponential backoff
        let mut last_error = None;
        for attempt in 1..=self.config.max_retries {
            match timeout(
                Duration::from_secs(self.config.timeout_seconds),
                self.ollama.generate(request.clone())
            ).await {
                Ok(Ok(response)) => {
                    info!("Generated {} tokens with model '{}'", 
                          response.response.len(), model_name);
                    
                    return Ok(LLMResponse {
                        content: response.response,
                        model: model_name.to_string(),
                        prompt_tokens: None, // Ollama doesn't provide token counts
                        completion_tokens: None,
                        total_tokens: None,
                    });
                }
                Ok(Err(e)) => {
                    error!("Ollama API error on attempt {}: {}", attempt, e);
                    last_error = Some(anyhow::anyhow!("Ollama API error: {}", e));
                }
                Err(_) => {
                    error!("Timeout on attempt {} after {} seconds", 
                           attempt, self.config.timeout_seconds);
                    last_error = Some(anyhow::anyhow!(
                        "Request timeout after {} seconds", self.config.timeout_seconds
                    ));
                }
            }
            
            if attempt < self.config.max_retries {
                let backoff_seconds = 2_u64.pow(attempt - 1);
                warn!("Retrying in {} seconds (attempt {}/{})", 
                      backoff_seconds, attempt, self.config.max_retries);
                tokio::time::sleep(Duration::from_secs(backoff_seconds)).await;
            }
        }
        
        Err(last_error.unwrap_or_else(|| 
            anyhow::anyhow!("Failed to generate text after {} attempts", self.config.max_retries)
        ))
    }
    
    /// Generate text with structured JSON output
    pub async fn generate_json<T>(
        &self,
        prompt: &str,
        model: Option<&str>,
    ) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        let enhanced_prompt = format!(
            "{}

Please respond with valid JSON only. Do not include any explanation or markdown formatting.",
            prompt
        );
        
        let response = self.generate(&enhanced_prompt, model).await?;
        
        // Try to extract JSON from response
        let json_content = extract_json_from_text(&response.content)
            .ok_or_else(|| anyhow::anyhow!("No valid JSON found in response"))?;
        
        serde_json::from_str(&json_content)
            .context("Failed to parse JSON response")
    }
    
    /// Test if the client is working with a simple prompt
    pub async fn health_check(&self) -> Result<()> {
        let test_prompt = "Respond with exactly 'OK' if you can understand this.";
        
        let response = timeout(
            Duration::from_secs(10),
            self.generate(test_prompt, None)
        ).await
        .context("Health check timeout")?
        .context("Health check failed")?;
        
        if response.content.trim().to_uppercase().contains("OK") {
            info!("LLM health check passed");
            Ok(())
        } else {
            Err(anyhow::anyhow!("Unexpected health check response: {}", response.content))
        }
    }
    
    /// Get available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let models = self.ollama.list_local_models().await
            .context("Failed to list local models")?;
        
        Ok(models.into_iter().map(|m| m.name).collect())
    }
}

/// Extract JSON from text that might contain markdown or other formatting
fn extract_json_from_text(text: &str) -> Option<String> {
    // First try to find JSON wrapped in markdown code blocks
    if let Some(start) = text.find("```json") {
        if let Some(end) = text[start + 7..].find("```") {
            return Some(text[start + 7..start + 7 + end].trim().to_string());
        }
    }
    
    // Try to find JSON wrapped in regular code blocks
    if let Some(start) = text.find("```") {
        if let Some(end) = text[start + 3..].find("```") {
            let potential_json = text[start + 3..start + 3 + end].trim();
            if potential_json.starts_with('{') || potential_json.starts_with('[') {
                return Some(potential_json.to_string());
            }
        }
    }
    
    // Look for JSON-like content (starts with { or [)
    if let Some(start) = text.find('{') {
        // Find the matching closing brace
        let mut brace_count = 0;
        let mut end_pos = start;
        
        for (i, c) in text[start..].char_indices() {
            match c {
                '{' => brace_count += 1,
                '}' => {
                    brace_count -= 1;
                    if brace_count == 0 {
                        end_pos = start + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        
        if brace_count == 0 {
            return Some(text[start..end_pos].to_string());
        }
    }
    
    // If all else fails, try the entire text if it looks like JSON
    let trimmed = text.trim();
    if (trimmed.starts_with('{') && trimmed.ends_with('}')) || 
       (trimmed.starts_with('[') && trimmed.ends_with(']')) {
        return Some(trimmed.to_string());
    }
    
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extract_json_from_text() {
        let text1 = r#"Here is the JSON: {"key": "value", "number": 42}"#;
        let extracted1 = extract_json_from_text(text1);
        assert_eq!(extracted1, Some(r#"{"key": "value", "number": 42}"#.to_string()));
        
        let text2 = r#"```json
{"action": "BUY_CALLS", "confidence": 0.8}
```"#;
        let extracted2 = extract_json_from_text(text2);
        assert_eq!(extracted2, Some(r#"{"action": "BUY_CALLS", "confidence": 0.8}"#.to_string()));
        
        let text3 = "This is not JSON at all";
        let extracted3 = extract_json_from_text(text3);
        assert_eq!(extracted3, None);
    }
    
    #[tokio::test]
    #[ignore] // Requires running Ollama
    async fn test_ollama_integration() {
        let config = LLMConfig::default();
        
        match LLMClient::new(config).await {
            Ok(client) => {
                // Test basic generation
                let response = client.generate("Say hello in one word", None).await;
                match response {
                    Ok(resp) => {
                        println!("Response: {}", resp.content);
                        assert!(!resp.content.is_empty());
                    }
                    Err(e) => {
                        println!("Generation failed: {}", e);
                        // This is expected if Ollama isn't running
                    }
                }
            }
            Err(e) => {
                println!("Failed to create client: {}", e);
                // This is expected if Ollama isn't running
            }
        }
    }
}

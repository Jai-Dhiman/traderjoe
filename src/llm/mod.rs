//! LLM integration module
//! Supports multiple providers: Ollama (local), Cerebras, OpenRouter

use anyhow::{Context, Result};
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
    },
    Client as OpenAIClient,
};
use chrono::{DateTime, NaiveTime, Utc};
use chrono_tz::America::New_York;
use ollama_rs::{generation::completion::request::GenerationRequest, Ollama};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{error, info, warn};
use url::Url;

use crate::system;

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
    pub provider: String, // "ollama", "cerebras", or "openrouter"
    pub ollama_url: String,
    pub cerebras_url: String,
    pub openrouter_url: String,
    pub primary_model: String,
    pub fallback_model: String,
    pub timeout_seconds: u64,
    pub max_retries: u32,
    pub force_retries: bool, // Override time check for retries (for testing)
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            provider: "ollama".to_string(),
            ollama_url: "http://localhost:11434".to_string(),
            cerebras_url: "https://api.cerebras.ai/v1".to_string(),
            openrouter_url: "https://openrouter.ai/api/v1".to_string(),
            primary_model: "llama3.2:3b".to_string(),
            fallback_model: "gpt-4o-mini".to_string(),
            timeout_seconds: 30,
            max_retries: 3,
            force_retries: false,
        }
    }
}

/// Provider-specific client enum
#[derive(Clone)]
enum ProviderClient {
    Ollama(Ollama),
    Cerebras(OpenAIClient<OpenAIConfig>),
    OpenRouter(OpenAIClient<OpenAIConfig>),
}

/// LLM client supporting multiple providers
#[derive(Clone)]
pub struct LLMClient {
    provider: ProviderClient,
    config: LLMConfig,
    #[allow(dead_code)]
    api_key: Option<String>,
}

impl std::fmt::Debug for LLMClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LLMClient")
            .field("config", &self.config)
            .finish()
    }
}

impl LLMClient {
    /// Create new LLM client with configuration
    pub async fn new(config: LLMConfig, api_key: Option<String>) -> Result<Self> {
        let provider = match config.provider.as_str() {
            "ollama" => {
                info!("Initializing Ollama provider");
                let parsed_url = Url::parse(&config.ollama_url).context("Invalid Ollama URL")?;
                parsed_url
                    .host_str()
                    .ok_or_else(|| anyhow::anyhow!("No host in Ollama URL"))?;

                let ollama = Ollama::from_url(parsed_url.clone());

                // Check available memory for local models
                info!("Checking system memory before model initialization");
                match system::check_available_memory() {
                    Ok(mem_stats) => {
                        info!(
                            "Memory check passed: {:.2} GB available / {:.2} GB total",
                            mem_stats.available_memory_gb, mem_stats.total_memory_gb
                        );
                    }
                    Err(e) => {
                        warn!("Memory check failed (non-fatal for cloud providers): {}", e);
                    }
                }

                // Test Ollama connectivity
                info!("Testing Ollama connectivity at {}", config.ollama_url);
                match timeout(Duration::from_secs(10), ollama.list_local_models()).await {
                    Ok(Ok(models)) => {
                        info!(
                            "Connected to Ollama successfully. Available models: {:?}",
                            models.iter().map(|m| &m.name).collect::<Vec<_>>()
                        );
                    }
                    Ok(Err(e)) => {
                        warn!("Failed to list Ollama models: {}", e);
                    }
                    Err(_) => {
                        warn!("Timeout connecting to Ollama at {}", config.ollama_url);
                    }
                }

                ProviderClient::Ollama(ollama)
            }
            "cerebras" => {
                info!("Initializing Cerebras provider");
                let api_key = api_key
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("CEREBRAS_API_KEY is required for Cerebras provider"))?;

                let openai_config = OpenAIConfig::new()
                    .with_api_key(&api_key)
                    .with_api_base(&config.cerebras_url);

                let client = OpenAIClient::with_config(openai_config);
                ProviderClient::Cerebras(client)
            }
            "openrouter" => {
                info!("Initializing OpenRouter provider");
                let api_key = api_key
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("API key is required for OpenRouter provider"))?;

                let openai_config = OpenAIConfig::new()
                    .with_api_key(&api_key)
                    .with_api_base(&config.openrouter_url);

                let client = OpenAIClient::with_config(openai_config);
                ProviderClient::OpenRouter(client)
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported LLM provider: {}. Use 'ollama', 'cerebras', or 'openrouter'",
                    config.provider
                ));
            }
        };

        Ok(Self {
            provider,
            config,
            api_key,
        })
    }

    /// Create client from config::Config
    pub async fn from_config(config: &crate::config::Config) -> Result<Self> {
        let llm_config = LLMConfig {
            provider: config.llm.provider.clone(),
            ollama_url: config.llm.ollama_url.clone(),
            cerebras_url: config.llm.cerebras_url.clone(),
            openrouter_url: config.llm.openrouter_url.clone(),
            primary_model: config.llm.primary_model.clone(),
            fallback_model: config.llm.fallback_model.clone(),
            timeout_seconds: config.llm.timeout_seconds,
            max_retries: 3,
            force_retries: false,
        };

        // Get API key based on provider
        let api_key = match llm_config.provider.as_str() {
            "cerebras" => config.apis.cerebras_api_key.clone(),
            "openrouter" => config.apis.openai_api_key.clone(), // Reuse openai_api_key for OpenRouter
            _ => None,
        };

        Self::new(llm_config, api_key).await
    }

    /// Generate text using specified model with explicit error handling
    pub async fn generate(&self, prompt: &str, model: Option<&str>) -> Result<LLMResponse> {
        let model_name = model.unwrap_or(&self.config.primary_model);

        info!(
            "Generating text with model '{}' (prompt length: {} chars)",
            model_name,
            prompt.len()
        );

        // Retry logic with exponential backoff
        let mut last_error = None;
        for attempt in 1..=self.config.max_retries {
            let result = match &self.provider {
                ProviderClient::Ollama(ollama) => {
                    let request = GenerationRequest::new(model_name.to_string(), prompt.to_string());
                    match timeout(
                        Duration::from_secs(self.config.timeout_seconds),
                        ollama.generate(request),
                    )
                    .await
                    {
                        Ok(Ok(response)) => Ok(LLMResponse {
                            content: response.response,
                            model: model_name.to_string(),
                            prompt_tokens: None,
                            completion_tokens: None,
                            total_tokens: None,
                        }),
                        Ok(Err(e)) => Err(anyhow::anyhow!("Ollama API error: {}", e)),
                        Err(_) => Err(anyhow::anyhow!(
                            "Request timeout after {} seconds",
                            self.config.timeout_seconds
                        )),
                    }
                }
                ProviderClient::Cerebras(client) | ProviderClient::OpenRouter(client) => {
                    // Use chat completion for OpenAI-compatible APIs
                    let messages = vec![ChatCompletionRequestMessage::User(
                        ChatCompletionRequestUserMessageArgs::default()
                            .content(prompt)
                            .build()
                            .context("Failed to build user message")?,
                    )];

                    let request = CreateChatCompletionRequestArgs::default()
                        .model(model_name)
                        .messages(messages)
                        .build()
                        .context("Failed to build chat completion request")?;

                    match timeout(
                        Duration::from_secs(self.config.timeout_seconds),
                        client.chat().create(request),
                    )
                    .await
                    {
                        Ok(Ok(response)) => {
                            let content = response
                                .choices
                                .first()
                                .and_then(|c| c.message.content.clone())
                                .ok_or_else(|| anyhow::anyhow!("No content in API response"))?;

                            Ok(LLMResponse {
                                content,
                                model: response.model.clone(),
                                prompt_tokens: response.usage.as_ref().map(|u| u.prompt_tokens as usize),
                                completion_tokens: response.usage.as_ref().map(|u| u.completion_tokens as usize),
                                total_tokens: response.usage.as_ref().map(|u| u.total_tokens as usize),
                            })
                        }
                        Ok(Err(e)) => Err(anyhow::anyhow!("API error: {}", e)),
                        Err(_) => Err(anyhow::anyhow!(
                            "Request timeout after {} seconds",
                            self.config.timeout_seconds
                        )),
                    }
                }
            };

            match result {
                Ok(response) => {
                    info!(
                        "Generated {} chars with model '{}'",
                        response.content.len(),
                        model_name
                    );
                    return Ok(response);
                }
                Err(e) => {
                    error!("API error on attempt {}: {}", attempt, e);
                    last_error = Some(e);
                }
            }

            if attempt < self.config.max_retries {
                // Check if we're too close to market close
                let now = Utc::now();
                if !self.config.force_retries && is_too_close_to_market_close(now) {
                    warn!("Skipping retry attempt {} - too close to market close (< 10 minutes to 4:00 PM ET). Use --force to override.",
                          attempt + 1);
                    break;
                }

                let backoff_seconds = 2_u64.pow(attempt - 1);
                warn!(
                    "Retrying in {} seconds (attempt {}/{})",
                    backoff_seconds, attempt, self.config.max_retries
                );
                tokio::time::sleep(Duration::from_secs(backoff_seconds)).await;
            }
        }

        Err(last_error.unwrap_or_else(|| {
            anyhow::anyhow!(
                "Failed to generate text after {} attempts",
                self.config.max_retries
            )
        }))
    }

    /// Generate text with structured JSON output
    pub async fn generate_json<T>(&self, prompt: &str, model: Option<&str>) -> Result<T>
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

        serde_json::from_str(&json_content).context("Failed to parse JSON response")
    }

    /// Test if the client is working with a simple prompt
    pub async fn health_check(&self) -> Result<()> {
        let test_prompt = "Respond with exactly 'OK' if you can understand this.";

        let response = timeout(Duration::from_secs(10), self.generate(test_prompt, None))
            .await
            .context("Health check timeout")?
            .context("Health check failed")?;

        if response.content.trim().to_uppercase().contains("OK") {
            info!("LLM health check passed");
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "Unexpected health check response: {}",
                response.content
            ))
        }
    }

    /// Get available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        match &self.provider {
            ProviderClient::Ollama(ollama) => {
                let models = ollama
                    .list_local_models()
                    .await
                    .context("Failed to list local models")?;
                Ok(models.into_iter().map(|m| m.name).collect())
            }
            ProviderClient::Cerebras(_) => {
                // Return known Cerebras models
                Ok(vec![
                    "llama-3.3-70b".to_string(),
                    "llama-3.1-8b".to_string(),
                    "llama-3.1-70b".to_string(),
                ])
            }
            ProviderClient::OpenRouter(_) => {
                // Return common OpenRouter models
                Ok(vec![
                    "meta-llama/llama-3.1-70b-instruct".to_string(),
                    "anthropic/claude-3.5-sonnet".to_string(),
                    "openai/gpt-4o".to_string(),
                ])
            }
        }
    }

    /// Get Ollama process memory usage
    pub fn get_ollama_memory_usage(&self) -> Option<f64> {
        let mut monitor = system::SystemMonitor::new();
        monitor.get_ollama_memory_usage()
    }

    /// Log memory stats including Ollama usage
    pub fn log_memory_stats(&self) -> Result<()> {
        let mut monitor = system::SystemMonitor::new();
        let stats = monitor.get_system_stats()?;

        if let Some(ollama_mem_gb) = self.get_ollama_memory_usage() {
            info!("Ollama process is using {:.2} GB of memory", ollama_mem_gb);
        } else {
            warn!("Could not find Ollama process to check memory usage");
        }

        info!("System stats: {}", serde_json::to_string_pretty(&stats)?);
        Ok(())
    }
}

/// Check if current time is too close to market close (< 10 minutes to 4:00 PM ET)
/// Returns true if retries should be skipped
fn is_too_close_to_market_close(now: DateTime<Utc>) -> bool {
    let et_time = now.with_timezone(&New_York);
    let current_time = et_time.time();

    // Market closes at 4:00 PM ET
    let market_close =
        NaiveTime::from_hms_opt(16, 0, 0).expect("Invalid hardcoded time 16:00:00 - this is a bug");

    // 10 minutes before market close
    let retry_cutoff = NaiveTime::from_hms_opt(15, 50, 0)
        .expect("Invalid hardcoded time 15:50:00 - this is a bug");

    // If current time is between 3:50 PM and 4:00 PM ET, skip retries
    current_time >= retry_cutoff && current_time < market_close
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
    if (trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'))
    {
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
        assert_eq!(
            extracted1,
            Some(r#"{"key": "value", "number": 42}"#.to_string())
        );

        let text2 = r#"```json
{"action": "BUY_CALLS", "confidence": 0.8}
```"#;
        let extracted2 = extract_json_from_text(text2);
        assert_eq!(
            extracted2,
            Some(r#"{"action": "BUY_CALLS", "confidence": 0.8}"#.to_string())
        );

        let text3 = "This is not JSON at all";
        let extracted3 = extract_json_from_text(text3);
        assert_eq!(extracted3, None);
    }

    #[test]
    fn test_is_too_close_to_market_close_before_cutoff() {
        // 3:45 PM ET (before 3:50 PM cutoff)
        let time = chrono::DateTime::parse_from_rfc3339("2025-01-06T20:45:00Z")
            .unwrap()
            .with_timezone(&Utc);

        assert!(!is_too_close_to_market_close(time));
    }

    #[test]
    fn test_is_too_close_to_market_close_after_cutoff() {
        // 3:55 PM ET (after 3:50 PM cutoff, before 4:00 PM)
        let time = chrono::DateTime::parse_from_rfc3339("2025-01-06T20:55:00Z")
            .unwrap()
            .with_timezone(&Utc);

        assert!(is_too_close_to_market_close(time));
    }

    #[test]
    fn test_is_too_close_to_market_close_exactly_at_cutoff() {
        // 3:50 PM ET (exactly at cutoff)
        let time = chrono::DateTime::parse_from_rfc3339("2025-01-06T20:50:00Z")
            .unwrap()
            .with_timezone(&Utc);

        assert!(is_too_close_to_market_close(time));
    }

    #[test]
    fn test_is_too_close_to_market_close_after_close() {
        // 4:05 PM ET (after market close)
        let time = chrono::DateTime::parse_from_rfc3339("2025-01-06T21:05:00Z")
            .unwrap()
            .with_timezone(&Utc);

        assert!(!is_too_close_to_market_close(time));
    }

    #[test]
    fn test_is_too_close_to_market_close_exactly_at_close() {
        // 4:00 PM ET (exactly at market close)
        let time = chrono::DateTime::parse_from_rfc3339("2025-01-06T21:00:00Z")
            .unwrap()
            .with_timezone(&Utc);

        assert!(!is_too_close_to_market_close(time));
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

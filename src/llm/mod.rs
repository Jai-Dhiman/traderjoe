//! LLM integration module for Ollama + OpenAI fallback
//! Provides local Llama 3.2 3B integration with OpenAI GPT-4o-mini fallback

use anyhow::{Context, Result};
use async_openai::{Client as OpenAIClient, config::OpenAIConfig, types::{ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage, CreateChatCompletionRequest}};
use chrono::{DateTime, NaiveTime, Utc};
use chrono_tz::America::New_York;
use ollama_rs::{Ollama, generation::completion::request::GenerationRequest};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::time::{timeout, Instant};
use tracing::{info, warn, error};
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
    pub ollama_url: String,
    pub primary_model: String,
    pub fallback_model: String,  // OpenAI model (e.g., "gpt-4o-mini")
    pub openai_api_key: Option<String>,
    pub timeout_seconds: u64,
    pub max_retries: u32,
    pub enable_fallback: bool,
    pub force_retries: bool,  // Override time check for retries (for testing)
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            ollama_url: "http://localhost:11434".to_string(),
            primary_model: "llama3.2:3b".to_string(),
            fallback_model: "gpt-4o-mini".to_string(),
            openai_api_key: None,
            timeout_seconds: 30,
            max_retries: 3,
            enable_fallback: true,
            force_retries: false,
        }
    }
}

/// Rate limiter for OpenAI API calls (60 requests per minute)
#[derive(Debug)]
struct RateLimiter {
    requests: Arc<Mutex<Vec<Instant>>>,
    max_requests_per_minute: usize,
}

impl RateLimiter {
    fn new(max_requests_per_minute: usize) -> Self {
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            max_requests_per_minute,
        }
    }

    async fn acquire(&self) -> Result<()> {
        loop {
            let mut requests = self.requests.lock().await;
            let now = Instant::now();
            let one_minute_ago = now - Duration::from_secs(60);

            // Remove requests older than 1 minute
            requests.retain(|&time| time > one_minute_ago);

            if requests.len() >= self.max_requests_per_minute {
                let Some(oldest) = requests.first() else {
                    // This should never happen since we just checked len >= max_requests_per_minute
                    // But we handle it gracefully just in case
                    drop(requests);
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    continue;
                };
                let wait_duration = Duration::from_secs(60) - (now - *oldest);
                drop(requests); // Release lock before sleeping

                warn!("Rate limit reached. Waiting {:?} before next request", wait_duration);
                tokio::time::sleep(wait_duration).await;
                continue; // Loop instead of recursive call
            }

            requests.push(now);
            return Ok(());
        }
    }
}

/// LLM client with local Ollama integration and OpenAI fallback
#[derive(Clone)]
pub struct LLMClient {
    ollama: Ollama,
    openai_client: Option<OpenAIClient<OpenAIConfig>>,
    config: LLMConfig,
    rate_limiter: Option<Arc<RateLimiter>>,
    fallback_usage_count: Arc<Mutex<usize>>,
}

impl std::fmt::Debug for LLMClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LLMClient")
            .field("config", &self.config)
            .field("has_openai_client", &self.openai_client.is_some())
            .finish()
    }
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

        // Initialize OpenAI client if API key is provided
        let (openai_client, rate_limiter) = if config.enable_fallback {
            if let Some(api_key) = config.openai_api_key.as_ref() {
                info!("Initializing OpenAI fallback with model: {}", config.fallback_model);
                let openai_config = OpenAIConfig::new()
                    .with_api_key(api_key);
                let client = OpenAIClient::with_config(openai_config);
                let limiter = Arc::new(RateLimiter::new(60)); // 60 requests per minute
                (Some(client), Some(limiter))
            } else {
                warn!("OpenAI fallback enabled but no API key provided");
                (None, None)
            }
        } else {
            (None, None)
        };

        // Check available memory before proceeding
        info!("Checking system memory before model initialization");
        match system::check_available_memory() {
            Ok(mem_stats) => {
                info!(
                    "Memory check passed: {:.2} GB available / {:.2} GB total",
                    mem_stats.available_memory_gb,
                    mem_stats.total_memory_gb
                );
            }
            Err(e) => {
                error!("Memory check failed: {}", e);
                return Err(e);
            }
        }

        // Test Ollama connectivity
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
                if openai_client.is_none() {
                    return Err(anyhow::anyhow!(
                        "Ollama API error when listing models: {}. Is Ollama running? No fallback available.", e
                    ));
                } else {
                    warn!("Ollama unavailable but fallback is configured: {}", e);
                }
            }
            Err(_) => {
                error!("Timeout connecting to Ollama at {}", config.ollama_url);
                if openai_client.is_none() {
                    return Err(anyhow::anyhow!(
                        "Timeout connecting to Ollama. Is Ollama running at {}? No fallback available.", config.ollama_url
                    ));
                } else {
                    warn!("Ollama timeout but fallback is configured");
                }
            }
        }

        Ok(Self {
            ollama,
            openai_client,
            config,
            rate_limiter,
            fallback_usage_count: Arc::new(Mutex::new(0)),
        })
    }
    
    /// Create client from config::Config
    pub async fn from_config(config: &crate::config::Config) -> Result<Self> {
        let llm_config = LLMConfig {
            ollama_url: config.llm.ollama_url.clone(),
            primary_model: config.llm.primary_model.clone(),
            fallback_model: config.llm.fallback_model.clone(),
            openai_api_key: std::env::var("OPENAI_API_KEY").ok(),
            timeout_seconds: config.llm.timeout_seconds,
            max_retries: 3,
            enable_fallback: true,
            force_retries: false,
        };

        Self::new(llm_config).await
    }

    /// Get fallback usage statistics
    pub async fn fallback_usage_count(&self) -> usize {
        *self.fallback_usage_count.lock().await
    }

    /// Generate text using OpenAI (fallback)
    async fn generate_with_openai(&self, prompt: &str) -> Result<LLMResponse> {
        let client = self.openai_client.as_ref()
            .ok_or_else(|| anyhow::anyhow!("OpenAI client not initialized"))?;

        let limiter = self.rate_limiter.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Rate limiter not initialized"))?;

        // Acquire rate limit slot
        limiter.acquire().await?;

        // Increment fallback usage counter
        {
            let mut count = self.fallback_usage_count.lock().await;
            *count += 1;
        }

        info!("Using OpenAI fallback with model: {}", self.config.fallback_model);

        let request = CreateChatCompletionRequest{
            model: self.config.fallback_model.clone(),
            messages: vec![
                ChatCompletionRequestMessage::System(
                    ChatCompletionRequestSystemMessage {
                        content: async_openai::types::ChatCompletionRequestSystemMessageContent::Text(
                            "You are a helpful assistant that provides concise, accurate responses.".to_string()
                        ),
                        ..Default::default()
                    }
                ),
                ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessage {
                        content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(prompt.to_string()),
                        ..Default::default()
                    }
                ),
            ],
            ..Default::default()
        };

        let response = timeout(
            Duration::from_secs(self.config.timeout_seconds),
            client.chat().create(request)
        ).await
        .context("OpenAI request timeout")??;

        let content = response.choices.first()
            .and_then(|choice| choice.message.content.as_ref())
            .ok_or_else(|| anyhow::anyhow!("No content in OpenAI response"))?
            .to_string();

        Ok(LLMResponse {
            content,
            model: self.config.fallback_model.clone(),
            prompt_tokens: response.usage.as_ref().map(|u| u.prompt_tokens as usize),
            completion_tokens: response.usage.as_ref().map(|u| u.completion_tokens as usize),
            total_tokens: response.usage.as_ref().map(|u| u.total_tokens as usize),
        })
    }
    
    /// Generate text using specified model with explicit error handling and fallback
    pub async fn generate(
        &self,
        prompt: &str,
        model: Option<&str>,
    ) -> Result<LLMResponse> {
        let model_name = model.unwrap_or(&self.config.primary_model);

        info!("Generating text with model '{}' (prompt length: {} chars)",
              model_name, prompt.len());

        let request = GenerationRequest::new(model_name.to_string(), prompt.to_string());

        // Retry logic with exponential backoff for Ollama
        let mut last_error = None;
        for attempt in 1..=self.config.max_retries {
            match timeout(
                Duration::from_secs(self.config.timeout_seconds),
                self.ollama.generate(request.clone())
            ).await {
                Ok(Ok(response)) => {
                    info!("Generated {} chars with model '{}'",
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
                // Check if we're too close to market close
                let now = Utc::now();
                if !self.config.force_retries && is_too_close_to_market_close(now) {
                    warn!("Skipping retry attempt {} - too close to market close (< 10 minutes to 4:00 PM ET). Use --force to override.",
                          attempt + 1);
                    break; // Exit retry loop
                }

                let backoff_seconds = 2_u64.pow(attempt - 1);
                warn!("Retrying in {} seconds (attempt {}/{})",
                      backoff_seconds, attempt, self.config.max_retries);
                tokio::time::sleep(Duration::from_secs(backoff_seconds)).await;
            }
        }

        // Ollama failed after all retries - try OpenAI fallback if available
        if self.config.enable_fallback && self.openai_client.is_some() {
            warn!("Ollama failed after {} attempts. Falling back to OpenAI", self.config.max_retries);
            match self.generate_with_openai(prompt).await {
                Ok(response) => {
                    info!("Successfully generated response using OpenAI fallback");
                    return Ok(response);
                }
                Err(e) => {
                    error!("OpenAI fallback also failed: {}", e);
                    return Err(anyhow::anyhow!(
                        "Both Ollama and OpenAI fallback failed. Ollama: {:?}, OpenAI: {}",
                        last_error, e
                    ));
                }
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
    let market_close = NaiveTime::from_hms_opt(16, 0, 0)
        .expect("Invalid hardcoded time 16:00:00 - this is a bug");

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

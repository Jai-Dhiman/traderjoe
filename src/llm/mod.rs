//! LLM integration module
//! Supports Workers AI (OpenAI-compatible) and native Ollama API

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
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{error, info, warn};

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
    pub provider: String, // "workers_ai" or "openai"
    pub workers_ai_url: String,
    pub workers_ai_account_id: Option<String>,
    pub workers_ai_api_token: Option<String>,
    pub primary_model: String,
    pub fallback_model: String,
    pub timeout_seconds: u64,
    pub max_retries: u32,
    pub force_retries: bool, // Override time check for retries (for testing)
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            workers_ai_url: "https://api.cloudflare.com/client/v4/accounts".to_string(),
            workers_ai_account_id: None,
            workers_ai_api_token: None,
            primary_model: "gpt-5-nano-2025-08-07".to_string(),
            fallback_model: "gpt-5-nano-2025-08-07".to_string(),
            timeout_seconds: 30,
            max_retries: 3,
            force_retries: false,
        }
    }
}

/// Provider-specific client enum
#[derive(Clone)]
enum ProviderClient {
    WorkersAI(OpenAIClient<OpenAIConfig>),
    OpenAI(OpenAIClient<OpenAIConfig>),
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
            "workers_ai" => {
                info!("Initializing Cloudflare Workers AI provider");

                let account_id = config.workers_ai_account_id
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("CLOUDFLARE_ACCOUNT_ID is required for Workers AI provider"))?;

                let api_token = api_key
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("CLOUDFLARE_API_TOKEN is required for Workers AI provider"))?;

                // Cloudflare Workers AI uses format: https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1
                let api_base = format!("{}/{}/ai/v1", config.workers_ai_url, account_id);

                let openai_config = OpenAIConfig::new()
                    .with_api_key(&api_token)
                    .with_api_base(&api_base);

                let client = OpenAIClient::with_config(openai_config);
                ProviderClient::WorkersAI(client)
            }
            "openai" => {
                info!("Initializing OpenAI provider");

                let api_key = api_key
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("OPENAI_API_KEY is required for OpenAI provider"))?;

                let openai_config = OpenAIConfig::new()
                    .with_api_key(&api_key);

                let client = OpenAIClient::with_config(openai_config);
                ProviderClient::OpenAI(client)
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported LLM provider: {}. Use 'workers_ai' or 'openai'",
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
            workers_ai_url: config.llm.workers_ai_url.clone(),
            workers_ai_account_id: config.apis.cloudflare_account_id.clone(),
            workers_ai_api_token: config.apis.cloudflare_api_token.clone(),
            primary_model: config.llm.primary_model.clone(),
            fallback_model: config.llm.fallback_model.clone(),
            timeout_seconds: config.llm.timeout_seconds,
            max_retries: 3,
            force_retries: false,
        };

        // Get API key based on provider
        let api_key = match llm_config.provider.as_str() {
            "workers_ai" => config.apis.cloudflare_api_token.clone(),
            "openai" => config.apis.openai_api_key.clone(),
            _ => None,
        };

        Self::new(llm_config, api_key).await
    }

    /// Generate text using specified model with explicit error handling
    pub async fn generate(&self, prompt: &str, model: Option<&str>) -> Result<LLMResponse> {
        self.generate_internal(prompt, model, false).await
    }

    /// Internal generate method with JSON mode support
    async fn generate_internal(&self, prompt: &str, model: Option<&str>, _json_mode: bool) -> Result<LLMResponse> {
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
                ProviderClient::WorkersAI(client) => {
                    self.generate_with_workers_ai(client, prompt, model_name).await
                }
                ProviderClient::OpenAI(client) => {
                    self.generate_with_openai(client, prompt, model_name).await
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

    /// Generate using Workers AI (OpenAI-compatible)
    async fn generate_with_workers_ai(
        &self,
        client: &OpenAIClient<OpenAIConfig>,
        prompt: &str,
        model_name: &str,
    ) -> Result<LLMResponse> {
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

    /// Generate using OpenAI API
    async fn generate_with_openai(
        &self,
        client: &OpenAIClient<OpenAIConfig>,
        prompt: &str,
        model_name: &str,
    ) -> Result<LLMResponse> {
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
            Ok(Err(e)) => Err(anyhow::anyhow!("OpenAI API error: {}", e)),
            Err(_) => Err(anyhow::anyhow!(
                "Request timeout after {} seconds",
                self.config.timeout_seconds
            )),
        }
    }


    /// Generate text with structured JSON output
    pub async fn generate_json<T>(&self, prompt: &str, model: Option<&str>) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        let mut last_error = None;

        // Retry JSON parsing up to 3 times with progressively clearer instructions
        for attempt in 1..=3 {
            let enhanced_prompt = if attempt == 1 {
                format!(
                    "{}

Please respond with valid JSON only. Do not include any explanation or markdown formatting.",
                    prompt
                )
            } else {
                // On retry, be even more explicit
                format!(
                    "{}

CRITICAL: The previous response failed to parse as valid JSON.
Respond with ONLY valid JSON matching the schema provided above.
Do NOT include:
- Markdown code blocks (```json or ```)
- Any explanatory text before or after the JSON
- Comments in the JSON
- Trailing commas
Just return the raw JSON object starting with {{ and ending with }}.",
                    prompt
                )
            };

            info!("Generating JSON response (attempt {}/3)", attempt);

            // Use JSON mode to hint JSON output
            let response = match self.generate_internal(&enhanced_prompt, model, true).await {
                Ok(r) => r,
                Err(e) => {
                    error!("LLM API error on attempt {}: {}", attempt, e);
                    last_error = Some(e);
                    continue;
                }
            };

            // Try to extract JSON from response
            let json_content = match extract_json_from_text(&response.content) {
                Some(content) => content,
                None => {
                    error!(
                        "Failed to extract JSON from LLM response (attempt {}). Raw response:\n{}",
                        attempt, response.content
                    );
                    last_error = Some(anyhow::anyhow!(
                        "No valid JSON found in response. Raw response: {}",
                        response.content
                    ));

                    // Don't retry if we're on the last attempt
                    if attempt < 3 {
                        warn!("Retrying with clearer JSON instructions...");
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        continue;
                    } else {
                        return Err(last_error.unwrap());
                    }
                }
            };

            // Try to parse the JSON
            match serde_json::from_str::<T>(&json_content) {
                Ok(parsed) => {
                    info!("Successfully parsed JSON response on attempt {}", attempt);
                    return Ok(parsed);
                }
                Err(e) => {
                    error!(
                        "Failed to parse JSON on attempt {}: {}. Extracted JSON:\n{}",
                        attempt, e, json_content
                    );
                    last_error = Some(anyhow::anyhow!(
                        "JSON parse error: {}. Content: {}",
                        e, json_content
                    ));

                    if attempt < 3 {
                        warn!("Retrying with clearer JSON instructions...");
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        continue;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            anyhow::anyhow!("Failed to generate valid JSON after 3 attempts")
        }))
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
            ProviderClient::WorkersAI(_) => {
                // Return known Cloudflare Workers AI models
                Ok(vec![
                    "@cf/meta/llama-4-scout-17b-16e-instruct".to_string(),
                    "@cf/meta/llama-3.3-70b-instruct".to_string(),
                    "@cf/meta/llama-3.1-8b-instruct".to_string(),
                ])
            }
            ProviderClient::OpenAI(_) => {
                // Return common OpenAI models
                Ok(vec![
                    "gpt-5-nano-2025-08-07".to_string(),
                    "gpt-4-turbo".to_string(),
                    "gpt-4".to_string(),
                    "gpt-3.5-turbo".to_string(),
                ])
            }
        }
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
}

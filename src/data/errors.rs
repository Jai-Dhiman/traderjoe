use thiserror::Error;

/// Comprehensive error types for data operations
#[derive(Error, Debug)]
pub enum DataError {
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    #[error("Parse error: {message}")]
    Parse { message: String },
    
    #[error("API error: {message} (status: {status_code})")]
    Api { status_code: u16, message: String },
    
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Rate limit exceeded, retry after {retry_after} seconds")]
    RateLimit { retry_after: u64 },
    
    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),
    
    #[error("No data available for {symbol} between {start} and {end}")]
    NoData {
        symbol: String,
        start: String,
        end: String,
    },
    
    #[error("Authentication failed: {0}")]
    Authentication(String),
    
    #[error("Timeout error: operation took longer than {timeout_seconds}s")]
    Timeout { timeout_seconds: u64 },
    
    #[error("Data validation error: {field} - {message}")]
    Validation { field: String, message: String },
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for data operations
pub type DataResult<T> = Result<T, DataError>;

impl DataError {
    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            DataError::Network(_) => true,
            DataError::RateLimit { .. } => true,
            DataError::Timeout { .. } => true,
            DataError::Api { status_code, .. } => {
                // Retry on server errors (5xx) and rate limiting (429)
                *status_code >= 500 || *status_code == 429
            }
            _ => false,
        }
    }
    
    /// Get retry delay in seconds for retryable errors
    pub fn retry_delay(&self) -> Option<u64> {
        match self {
            DataError::RateLimit { retry_after } => Some(*retry_after),
            DataError::Network(_) => Some(1), // 1 second base delay
            DataError::Timeout { .. } => Some(2), // 2 second delay for timeouts
            DataError::Api { status_code, .. } if *status_code >= 500 => Some(5), // 5 seconds for server errors
            _ => None,
        }
    }
    
    /// Create a parse error with context
    pub fn parse_error<S: Into<String>>(message: S) -> Self {
        DataError::Parse {
            message: message.into(),
        }
    }
    
    /// Create a validation error with field context
    pub fn validation_error<S: Into<String>>(field: S, message: S) -> Self {
        DataError::Validation {
            field: field.into(),
            message: message.into(),
        }
    }
    
    /// Create an API error with status code
    pub fn api_error<S: Into<String>>(status_code: u16, message: S) -> Self {
        DataError::Api {
            status_code,
            message: message.into(),
        }
    }
}
use tokio_retry::{strategy::{ExponentialBackoff, jitter}, Retry};
use std::time::Duration;
use super::{DataError, DataResult};

pub async fn retry_with_backoff<F, Fut, T>(
    operation: F,
    max_attempts: usize,
) -> DataResult<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = DataResult<T>>,
{
    let retry_strategy = ExponentialBackoff::from_millis(100)
        .max_delay(Duration::from_secs(10))
        .map(jitter)
        .take(max_attempts);
    
    Retry::spawn(retry_strategy, || async {
        match operation().await {
            Ok(result) => Ok(result),
            Err(e) => {
                match &e {
                    DataError::Network(_) => {
                        tracing::warn!("Retryable network error: {}", e);
                        Err(e)
                    }
                    DataError::RateLimit { retry_after } => {
                        tracing::warn!("Rate limited, retry after {} seconds", retry_after);
                        tokio::time::sleep(Duration::from_secs(*retry_after)).await;
                        Err(e)
                    }
                    _ => {
                        tracing::error!("Non-retryable error: {}", e);
                        Err(e)
                    }
                }
            }
        }
    }).await
}
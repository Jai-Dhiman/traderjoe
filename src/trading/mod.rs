pub mod account;
pub mod auto_exit;
pub mod circuit_breaker;
pub mod execution;
pub mod paper;
pub mod position_sizing;

// Re-export commonly used types
pub use account::AccountManager;
pub use auto_exit::{AutoExitConfig, AutoExitManager};
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
pub use execution::{ExecutionParams, OrderSide};
pub use paper::{ExitReason, PaperTradingEngine, TradeStatus, TradeType};
pub use position_sizing::PositionSizer;

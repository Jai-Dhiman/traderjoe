pub mod account;
pub mod auto_exit;
pub mod black_scholes;
pub mod calibration;
pub mod circuit_breaker;
pub mod execution;
pub mod paper;
pub mod position_sizing;
pub mod regime;
pub mod risk_manager;
pub mod signals;
pub mod slippage;

// Re-export commonly used types
pub use account::AccountManager;
pub use auto_exit::{AutoExitConfig, AutoExitManager};
pub use black_scholes::{call_price, delta, put_price, strike_from_delta};
pub use calibration::{CalibrationBin, PlattScaler};
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
pub use execution::{ExecutionParams, OrderSide};
pub use paper::{ExitReason, PaperTradingEngine, TradeStatus, TradeType};
pub use position_sizing::PositionSizer;
pub use regime::{MarketRegime, VolatilityRegime};
pub use risk_manager::{RiskConfig, RiskManager};
pub use signals::{SignalCombiner, SignalStats};
pub use slippage::{SlippageCalculator, SlippageConfig};

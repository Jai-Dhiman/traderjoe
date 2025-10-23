//! Orchestrator module for coordinating ACE analysis pipelines
//! Combines data fetching, ML signals, vector search, and LLM reasoning

pub mod evening;
pub mod morning;

// Re-export main orchestrators
pub use evening::EveningOrchestrator;
pub use morning::MorningOrchestrator;

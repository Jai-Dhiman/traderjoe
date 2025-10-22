//! Orchestrator module for coordinating ACE analysis pipelines
//! Combines data fetching, ML signals, vector search, and LLM reasoning

pub mod morning;
pub mod evening;

// Re-export main orchestrators
pub use morning::MorningOrchestrator;
pub use evening::{EveningOrchestrator, EveningReviewResult};
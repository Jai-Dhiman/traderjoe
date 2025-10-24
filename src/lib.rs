#![deny(clippy::unwrap_used)]

pub mod ace;
pub mod config;
pub mod data;
pub mod db;
pub mod embeddings;
pub mod llm;
pub mod ml;
pub mod orchestrator;
pub mod system;
pub mod trading;
pub mod vector;

// Re-export commonly used items
pub use config::Config;
pub use data::{MarketData, NewsItem, ResearchResult, SentimentData};

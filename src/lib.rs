// TraderJoe - ACE-Enhanced Daily Trading System
// A comprehensive trading system that combines Extended Thinking (ACE - Augmented Context Evolution)
// with traditional machine learning to make daily trading decisions.

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

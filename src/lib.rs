//! TraderJoe - ACE-Enhanced Daily Trading System
//! 
//! A comprehensive trading system that combines Extended Thinking (ACE - Augmented Context Evolution) 
//! with traditional machine learning to make daily trading decisions.

pub mod config;
pub mod data;
pub mod db;
pub mod embeddings;
pub mod vector;
pub mod ace;
pub mod llm;
pub mod trading;
pub mod orchestrator;
pub mod ml;

// Re-export commonly used items
pub use config::Config;
pub use data::{MarketData, NewsItem, SentimentData, ResearchResult};
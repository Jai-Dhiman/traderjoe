//! Morning analysis orchestrator
//! Coordinates the full ACE pipeline: data → ML → context retrieval → LLM → decision

use anyhow::Result;
use chrono::Utc;
use serde_json::{json, Value};
use sqlx::PgPool;
use tracing::{info, warn, error};

use crate::{
    config::Config,
    data::{MarketDataClient, ResearchClient, SentimentClient},
    embeddings::EmbeddingGemma,
    vector::VectorStore,
    ace::{ContextDAO, ACEPrompts, TradingDecision},
    llm::LLMClient,
};

/// Morning analysis orchestrator
pub struct MorningOrchestrator {
    pool: PgPool,
    config: Config,
    market_client: MarketDataClient,
    research_client: ResearchClient,
    sentiment_client: SentimentClient,
    embedder: EmbeddingGemma,
    vector_store: VectorStore,
    context_dao: ContextDAO,
    llm_client: LLMClient,
}

impl MorningOrchestrator {
    /// Create new morning orchestrator
    pub async fn new(pool: PgPool, config: Config) -> Result<Self> {
        info!("Initializing Morning Orchestrator");
        
        // Initialize all clients
        let market_client = MarketDataClient::new(pool.clone());
        let research_client = ResearchClient::new(pool.clone(), config.apis.exa_api_key.clone());
        let sentiment_client = SentimentClient::new(
            pool.clone(),
            config.apis.reddit_client_id.clone(),
            config.apis.reddit_client_secret.clone(),
        );
        let embedder = EmbeddingGemma::load().await?;
        let vector_store = VectorStore::new(pool.clone()).await?;
        let context_dao = ContextDAO::new(pool.clone());
        let llm_client = LLMClient::from_config(&config).await?;
        
        // Ensure HNSW index exists
        vector_store.ensure_hnsw_index("ace_contexts", "embedding").await?;
        
        info!("Morning Orchestrator initialized successfully");
        
        Ok(Self {
            pool,
            config,
            market_client,
            research_client,
            sentiment_client,
            embedder,
            vector_store,
            context_dao,
            llm_client,
        })
    }
    
    /// Run full morning analysis for a symbol
    pub async fn analyze(&self, symbol: &str) -> Result<TradingDecision> {
        info!("🌅 Starting morning analysis for {}", symbol);
        
        // Step 1: Fetch market data
        info!("📊 Fetching market data...");
        let market_data = self.fetch_market_data(symbol).await?;
        
        // Step 2: Compute ML signals
        info!("🤖 Computing ML signals...");
        let ml_signals = self.compute_ml_signals(&market_data).await?;
        
        // Step 3: Fetch research and sentiment
        info!("🔍 Gathering research and sentiment...");
        let (research_data, sentiment_data) = tokio::try_join!(
            self.fetch_research(symbol),
            self.fetch_sentiment(symbol)
        )?;
        
        // Step 4: Build market state representation
        info!("🏗️ Building market state representation...");
        let market_state = self.build_market_state(
            symbol,
            &market_data,
            &ml_signals,
            &research_data,
            &sentiment_data
        ).await?;
        
        // Step 5: Retrieve similar historical contexts
        info!("🧠 Retrieving similar historical contexts...");
        let similar_contexts = self.retrieve_similar_contexts(&market_state).await?;
        
        // Step 6: Get relevant playbook entries (placeholder for now)
        let playbook_entries = self.get_playbook_entries(&market_state).await?;
        
        // Step 7: Generate decision via LLM
        info!("💡 Generating trading decision...");
        let decision = self.generate_decision(
            &market_state,
            &ml_signals,
            &similar_contexts,
            &playbook_entries
        ).await?;
        
        // Step 8: Persist context with embedding
        info!("💾 Persisting ACE context...");
        let context_embedding = self.embedder.embed(&serde_json::to_string(&market_state)?).await?;
        let context_id = self.context_dao.insert_context(
            &market_state,
            &serde_json::to_value(&decision)?,
            &decision.reasoning,
            decision.confidence,
            None, // No outcome yet - will be filled in evening review
            context_embedding
        ).await?;
        
        // Step 9: Display results
        self.display_analysis_results(&decision, context_id, &similar_contexts).await;
        
        info!("✅ Morning analysis complete for {}", symbol);
        Ok(decision)
    }
    
    /// Fetch and persist market data
    async fn fetch_market_data(&self, symbol: &str) -> Result<Value> {
        let ohlcv_data = self.market_client.fetch_ohlcv(symbol, 30).await?;
        
        if ohlcv_data.is_empty() {
            return Err(anyhow::anyhow!("No market data available for {}", symbol));
        }
        
        // Persist to database
        let count = self.market_client.persist_ohlcv(&ohlcv_data).await?;
        info!("Persisted {} OHLCV records for {}", count, symbol);
        
        // Return latest data point and some statistics
        let latest = &ohlcv_data[ohlcv_data.len() - 1];
        let previous = if ohlcv_data.len() > 1 {
            Some(&ohlcv_data[ohlcv_data.len() - 2])
        } else {
            None
        };
        
        let daily_change = if let Some(prev) = previous {
            ((latest.close - prev.close) / prev.close) * 100.0
        } else {
            0.0
        };
        
        let volatility = self.calculate_volatility(&ohlcv_data);
        
        Ok(json!({
            "symbol": symbol,
            "latest_price": latest.close,
            "daily_change_pct": daily_change,
            "volume": latest.volume,
            "volatility_20d": volatility,
            "high_52w": ohlcv_data.iter().map(|d| d.high).fold(0.0, f64::max),
            "low_52w": ohlcv_data.iter().map(|d| d.low).fold(f64::INFINITY, f64::min),
            "avg_volume_20d": ohlcv_data.iter().map(|d| d.volume as f64).sum::<f64>() / ohlcv_data.len() as f64,
            "data_points": ohlcv_data.len()
        }))
    }
    
    /// Compute ML signals from market data
    async fn compute_ml_signals(&self, market_data: &Value) -> Result<Value> {
        // For now, create placeholder ML signals
        // In a full implementation, this would call actual ML models
        let daily_change = market_data["daily_change_pct"].as_f64().unwrap_or(0.0);
        let volatility = market_data["volatility_20d"].as_f64().unwrap_or(0.0);
        
        // Placeholder technical indicators
        let rsi = 50.0 + daily_change * 2.0; // Simplified RSI-like signal
        let macd_signal = if daily_change > 0.0 { "bullish" } else { "bearish" };
        let volume_signal = if market_data["volume"].as_i64().unwrap_or(0) > 
                           market_data["avg_volume_20d"].as_i64().unwrap_or(0) {
            "high"
        } else {
            "normal"
        };
        
        // Simple momentum score
        let momentum_score = match () {
            _ if daily_change > 2.0 => 0.8,
            _ if daily_change > 0.5 => 0.6,
            _ if daily_change > -0.5 => 0.4,
            _ if daily_change > -2.0 => 0.2,
            _ => 0.1
        };
        
        Ok(json!({
            "technical_indicators": {
                "rsi_estimate": rsi.max(0.0).min(100.0),
                "macd_signal": macd_signal,
                "volume_signal": volume_signal,
                "volatility": volatility
            },
            "momentum_score": momentum_score,
            "price_signals": {
                "daily_change_pct": daily_change,
                "trend": if daily_change > 0.0 { "up" } else { "down" },
                "strength": daily_change.abs()
            },
            "ml_confidence": 0.65, // Placeholder confidence in ML signals
            "signal_summary": format!("{} momentum with {} volume", 
                                       if daily_change > 0.0 { "Positive" } else { "Negative" },
                                       volume_signal)
        }))
    }
    
    /// Fetch research data
    async fn fetch_research(&self, symbol: &str) -> Result<Value> {
        let query = format!("{} stock market outlook earnings analysis", symbol);
        match self.research_client.search(&query).await {
            Ok(research) => Ok(research),
            Err(e) => {
                warn!("Research fetch failed: {}, using fallback", e);
                Ok(json!({
                    "query": query,
                    "results": [],
                    "source": "fallback",
                    "message": "Research API unavailable"
                }))
            }
        }
    }
    
    /// Fetch sentiment data
    async fn fetch_sentiment(&self, symbol: &str) -> Result<Value> {
        match self.sentiment_client.analyze_reddit(Some(symbol)).await {
            Ok(sentiment) => Ok(sentiment),
            Err(e) => {
                warn!("Sentiment fetch failed: {}, using neutral sentiment", e);
                Ok(json!({
                    "symbol": symbol,
                    "score": 0.5,
                    "source": "fallback",
                    "message": "Sentiment API unavailable"
                }))
            }
        }
    }
    
    /// Build comprehensive market state representation
    async fn build_market_state(
        &self,
        symbol: &str,
        market_data: &Value,
        ml_signals: &Value,
        research_data: &Value,
        sentiment_data: &Value,
    ) -> Result<Value> {
        let current_time = Utc::now();
        
        Ok(json!({
            "timestamp": current_time,
            "symbol": symbol,
            "market_data": market_data,
            "ml_signals": ml_signals,
            "research_summary": {
                "source_count": research_data["results"].as_array().map(|r| r.len()).unwrap_or(0),
                "key_themes": research_data["autoprompt_string"].as_str().unwrap_or("No themes available"),
                "availability": if research_data["source"] == "fallback" { "limited" } else { "full" }
            },
            "sentiment": {
                "reddit_score": sentiment_data["score"].as_f64().unwrap_or(0.5),
                "sentiment_label": match sentiment_data["score"].as_f64().unwrap_or(0.5) {
                    s if s > 0.7 => "bullish",
                    s if s > 0.3 => "neutral",
                    _ => "bearish"
                },
                "source": sentiment_data["source"].as_str().unwrap_or("unknown")
            },
            "market_regime": self.assess_market_regime(market_data, ml_signals).await,
            "analysis_quality": self.assess_analysis_quality(market_data, research_data, sentiment_data).await
        }))
    }
    
    /// Retrieve similar historical contexts using vector search
    async fn retrieve_similar_contexts(&self, market_state: &Value) -> Result<Vec<crate::vector::ContextEntry>> {
        let search_text = format!(
            "Market analysis for {} with {} trend, {} volume, {} sentiment",
            market_state["symbol"].as_str().unwrap_or("unknown"),
            market_state["ml_signals"]["price_signals"]["trend"].as_str().unwrap_or("unknown"),
            market_state["ml_signals"]["technical_indicators"]["volume_signal"].as_str().unwrap_or("unknown"),
            market_state["sentiment"]["sentiment_label"].as_str().unwrap_or("unknown")
        );
        
        let embedding = self.embedder.embed(&search_text).await?;
        let similar_contexts = self.vector_store.similarity_search(embedding, 5).await?;
        
        info!("Found {} similar contexts", similar_contexts.len());
        Ok(similar_contexts)
    }
    
    /// Get relevant playbook entries (placeholder implementation)
    async fn get_playbook_entries(&self, _market_state: &Value) -> Result<Vec<String>> {
        // Placeholder for playbook system - would query actual playbook database
        Ok(vec![
            "When VIX < 20 and momentum positive, calls have 68% win rate".to_string(),
            "High volume + positive sentiment often leads to continuation".to_string(),
            "Avoid trading on low confidence days (< 60%)".to_string()
        ])
    }
    
    /// Generate trading decision using LLM
    async fn generate_decision(
        &self,
        market_state: &Value,
        ml_signals: &Value,
        similar_contexts: &[crate::vector::ContextEntry],
        playbook_entries: &[String],
    ) -> Result<TradingDecision> {
        let prompt = ACEPrompts::morning_decision_prompt(
            market_state,
            ml_signals,
            similar_contexts,
            playbook_entries,
            &Utc::now().format("%Y-%m-%d").to_string()
        );
        
        // Try to get structured decision from LLM
        match self.llm_client.generate_json::<TradingDecision>(&prompt, None).await {
            Ok(decision) => {
                info!("LLM generated decision: {} with {:.1}% confidence", 
                      decision.action, decision.confidence * 100.0);
                Ok(decision)
            }
            Err(e) => {
                error!("LLM decision generation failed: {}", e);
                // Fallback decision based on signals
                Ok(self.generate_fallback_decision(market_state, ml_signals).await)
            }
        }
    }
    
    /// Generate fallback decision when LLM is unavailable
    async fn generate_fallback_decision(
        &self,
        market_state: &Value,
        ml_signals: &Value
    ) -> TradingDecision {
        let momentum_score = ml_signals["momentum_score"].as_f64().unwrap_or(0.5);
        let daily_change = market_state["market_data"]["daily_change_pct"].as_f64().unwrap_or(0.0);
        let sentiment_score = market_state["sentiment"]["reddit_score"].as_f64().unwrap_or(0.5);
        
        let action = if momentum_score > 0.6 && daily_change > 0.5 {
            "BUY_CALLS"
        } else if momentum_score < 0.4 && daily_change < -0.5 {
            "BUY_PUTS"
        } else {
            "STAY_FLAT"
        };
        
        let confidence = ((momentum_score - 0.5).abs() + sentiment_score.max(1.0 - sentiment_score)) / 2.0;
        
        TradingDecision {
            action: action.to_string(),
            confidence: confidence.max(0.1).min(0.9),
            reasoning: format!(
                "Fallback decision based on momentum score {:.2}, daily change {:.2}%, sentiment {:.2}",
                momentum_score, daily_change, sentiment_score
            ),
            key_factors: vec![
                format!("Momentum: {:.1}%", momentum_score * 100.0),
                format!("Price change: {:.2}%", daily_change),
                format!("Sentiment: {:.2}", sentiment_score)
            ],
            risk_factors: vec![
                "LLM unavailable - using simplified logic".to_string(),
                "Limited market analysis".to_string()
            ],
            similar_pattern_reference: None,
            position_size_multiplier: 0.5, // Reduced size for fallback
        }
    }
    
    /// Display analysis results to user
    async fn display_analysis_results(
        &self,
        decision: &TradingDecision,
        context_id: uuid::Uuid,
        similar_contexts: &[crate::vector::ContextEntry]
    ) {
        println!("\n🎯 TRADING RECOMMENDATION");
        println!("========================");
        println!("Action: {}", decision.action);
        println!("Confidence: {:.1}%", decision.confidence * 100.0);
        println!("Position Size: {:.0}% of base", decision.position_size_multiplier * 100.0);
        println!("\n💭 REASONING:");
        println!("{}", decision.reasoning);
        
        if !decision.key_factors.is_empty() {
            println!("\n✅ KEY FACTORS:");
            for factor in &decision.key_factors {
                println!("  • {}", factor);
            }
        }
        
        if !decision.risk_factors.is_empty() {
            println!("\n⚠️  RISK FACTORS:");
            for risk in &decision.risk_factors {
                println!("  • {}", risk);
            }
        }
        
        if let Some(pattern) = &decision.similar_pattern_reference {
            println!("\n📊 SIMILAR PATTERN: {}", pattern);
        }
        
        println!("\n🧠 ACE CONTEXT:");
        println!("Context ID: {}", context_id);
        println!("Similar contexts found: {}", similar_contexts.len());
        
        if !similar_contexts.is_empty() {
            println!("\n📈 MOST SIMILAR PAST DECISIONS:");
            for (i, ctx) in similar_contexts.iter().take(3).enumerate() {
                println!("  {}. {} (similarity: {:.2}, confidence: {:.1}%)", 
                         i + 1, 
                         ctx.reasoning,
                         ctx.similarity.unwrap_or(0.0),
                         ctx.confidence * 100.0);
            }
        }
        
        println!("\n{}", "=".repeat(50));
    }
    
    /// Calculate price volatility from OHLCV data
    fn calculate_volatility(&self, ohlcv_data: &[crate::data::OHLCV]) -> f64 {
        if ohlcv_data.len() < 2 {
            return 0.0;
        }
        
        let returns: Vec<f64> = ohlcv_data.windows(2)
            .map(|pair| ((pair[1].close - pair[0].close) / pair[0].close))
            .collect();
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        variance.sqrt() * 100.0 // Convert to percentage
    }
    
    /// Assess current market regime
    async fn assess_market_regime(&self, market_data: &Value, ml_signals: &Value) -> String {
        let daily_change = market_data["daily_change_pct"].as_f64().unwrap_or(0.0);
        let volatility = market_data["volatility_20d"].as_f64().unwrap_or(0.0);
        let momentum = ml_signals["momentum_score"].as_f64().unwrap_or(0.5);
        
        match () {
            _ if volatility > 3.0 => "HIGH_VOLATILITY".to_string(),
            _ if daily_change.abs() > 2.0 => "TRENDING".to_string(),
            _ if momentum > 0.7 || momentum < 0.3 => "MOMENTUM".to_string(),
            _ => "RANGING".to_string(),
        }
    }
    
    /// Assess quality of available analysis data
    async fn assess_analysis_quality(&self, market_data: &Value, research_data: &Value, sentiment_data: &Value) -> Value {
        let market_quality = if market_data["data_points"].as_u64().unwrap_or(0) >= 20 { "good" } else { "limited" };
        let research_quality = if research_data["source"] != "fallback" { "good" } else { "limited" };
        let sentiment_quality = if sentiment_data["source"] != "fallback" { "good" } else { "limited" };
        
        json!({
            "market_data": market_quality,
            "research": research_quality,
            "sentiment": sentiment_quality,
            "overall": if market_quality == "good" && (research_quality == "good" || sentiment_quality == "good") {
                "sufficient"
            } else {
                "limited"
            }
        })
    }
}
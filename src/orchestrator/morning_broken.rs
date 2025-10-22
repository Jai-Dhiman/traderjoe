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
        let (research_data, sentiment_data) = tokio::try_join!(\n            self.fetch_research(symbol),\n            self.fetch_sentiment(symbol)\n        )?;
        
        // Step 4: Build market state representation
        info!("🏗️ Building market state representation...");
        let market_state = self.build_market_state(\n            symbol,\n            &market_data,\n            &ml_signals,\n            &research_data,\n            &sentiment_data\n        ).await?;
        
        // Step 5: Retrieve similar historical contexts
        info!("🧠 Retrieving similar historical contexts...");
        let similar_contexts = self.retrieve_similar_contexts(&market_state).await?;
        
        // Step 6: Get relevant playbook entries (placeholder for now)
        let playbook_entries = self.get_playbook_entries(&market_state).await?;
        
        // Step 7: Generate decision via LLM
        info!("💡 Generating trading decision...");
        let decision = self.generate_decision(\n            &market_state,\n            &ml_signals,\n            &similar_contexts,\n            &playbook_entries\n        ).await?;
        
        // Step 8: Persist context with embedding
        info!("💾 Persisting ACE context...");
        let context_embedding = self.embedder.embed(&serde_json::to_string(&market_state)?).await?;
        let context_id = self.context_dao.insert_context(\n            &market_state,\n            &serde_json::to_value(&decision)?,\n            &decision.reasoning,\n            decision.confidence,\n            None, // No outcome yet - will be filled in evening review\n            context_embedding\n        ).await?;
        
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
        let previous = if ohlcv_data.len() > 1 {\n            Some(&ohlcv_data[ohlcv_data.len() - 2])\n        } else {\n            None\n        };
        
        let daily_change = if let Some(prev) = previous {\n            ((latest.close - prev.close) / prev.close) * 100.0\n        } else {\n            0.0\n        };
        
        let volatility = self.calculate_volatility(&ohlcv_data);
        
        Ok(json!({\n            \"symbol\": symbol,\n            \"latest_price\": latest.close,\n            \"daily_change_pct\": daily_change,\n            \"volume\": latest.volume,\n            \"volatility_20d\": volatility,\n            \"high_52w\": ohlcv_data.iter().map(|d| d.high).fold(0.0, f64::max),\n            \"low_52w\": ohlcv_data.iter().map(|d| d.low).fold(f64::INFINITY, f64::min),\n            \"avg_volume_20d\": ohlcv_data.iter().map(|d| d.volume as f64).sum::<f64>() / ohlcv_data.len() as f64,\n            \"data_points\": ohlcv_data.len()\n        }))
    }
    
    /// Compute ML signals from market data
    async fn compute_ml_signals(&self, market_data: &Value) -> Result<Value> {
        // For now, create placeholder ML signals
        // In a full implementation, this would call actual ML models
        let latest_price = market_data["latest_price"].as_f64().unwrap_or(0.0);
        let daily_change = market_data["daily_change_pct"].as_f64().unwrap_or(0.0);
        let volatility = market_data["volatility_20d"].as_f64().unwrap_or(0.0);
        
        // Placeholder technical indicators
        let rsi = 50.0 + daily_change * 2.0; // Simplified RSI-like signal
        let macd_signal = if daily_change > 0.0 { "bullish" } else { "bearish" };
        let volume_signal = if market_data["volume"].as_i64().unwrap_or(0) > \n                           market_data["avg_volume_20d"].as_i64().unwrap_or(0) {\n            "high"\n        } else {\n            "normal"\n        };
        
        // Simple momentum score
        let momentum_score = match () {\n            _ if daily_change > 2.0 => 0.8,\n            _ if daily_change > 0.5 => 0.6,\n            _ if daily_change > -0.5 => 0.4,\n            _ if daily_change > -2.0 => 0.2,\n            _ => 0.1\n        };
        
        Ok(json!({\n            \"technical_indicators\": {\n                \"rsi_estimate\": rsi.max(0.0).min(100.0),\n                \"macd_signal\": macd_signal,\n                \"volume_signal\": volume_signal,\n                \"volatility\": volatility\n            },\n            \"momentum_score\": momentum_score,\n            \"price_signals\": {\n                \"daily_change_pct\": daily_change,\n                \"trend\": if daily_change > 0.0 { \"up\" } else { \"down\" },\n                \"strength\": daily_change.abs()\n            },\n            \"ml_confidence\": 0.65, // Placeholder confidence in ML signals\n            \"signal_summary\": format!(\"{} momentum with {} volume\", \n                                       if daily_change > 0.0 { \"Positive\" } else { \"Negative\" },\n                                       volume_signal)\n        }))
    }
    
    /// Fetch research data
    async fn fetch_research(&self, symbol: &str) -> Result<Value> {
        let query = format!(\"{} stock market outlook earnings analysis\", symbol);
        match self.research_client.search(&query).await {\n            Ok(research) => Ok(research),\n            Err(e) => {\n                warn!(\"Research fetch failed: {}, using fallback\", e);\n                Ok(json!({\n                    \"query\": query,\n                    \"results\": [],\n                    \"source\": \"fallback\",\n                    \"message\": \"Research API unavailable\"\n                }))\n            }\n        }\n    }
    
    /// Fetch sentiment data
    async fn fetch_sentiment(&self, symbol: &str) -> Result<Value> {
        match self.sentiment_client.analyze_reddit(Some(symbol)).await {\n            Ok(sentiment) => Ok(sentiment),\n            Err(e) => {\n                warn!(\"Sentiment fetch failed: {}, using neutral sentiment\", e);\n                Ok(json!({\n                    \"symbol\": symbol,\n                    \"score\": 0.5,\n                    \"source\": \"fallback\",\n                    \"message\": \"Sentiment API unavailable\"\n                }))\n            }\n        }\n    }
    
    /// Build comprehensive market state representation
    async fn build_market_state(\n        &self,\n        symbol: &str,\n        market_data: &Value,\n        ml_signals: &Value,\n        research_data: &Value,\n        sentiment_data: &Value,\n    ) -> Result<Value> {\n        let current_time = Utc::now();\n        \n        Ok(json!({\n            \"timestamp\": current_time,\n            \"symbol\": symbol,\n            \"market_data\": market_data,\n            \"ml_signals\": ml_signals,\n            \"research_summary\": {\n                \"source_count\": research_data[\"results\"].as_array().map(|r| r.len()).unwrap_or(0),\n                \"key_themes\": research_data[\"autoprompt_string\"].as_str().unwrap_or(\"No themes available\"),\n                \"availability\": if research_data[\"source\"] == \"fallback\" { \"limited\" } else { \"full\" }\n            },\n            \"sentiment\": {\n                \"reddit_score\": sentiment_data[\"score\"].as_f64().unwrap_or(0.5),\n                \"sentiment_label\": match sentiment_data[\"score\"].as_f64().unwrap_or(0.5) {\n                    s if s > 0.7 => \"bullish\",\n                    s if s > 0.3 => \"neutral\",\n                    _ => \"bearish\"\n                },\n                \"source\": sentiment_data[\"source\"].as_str().unwrap_or(\"unknown\")\n            },\n            \"market_regime\": self.assess_market_regime(market_data, ml_signals).await,\n            \"analysis_quality\": self.assess_analysis_quality(market_data, research_data, sentiment_data).await\n        }))\n    }
    
    /// Retrieve similar historical contexts using vector search
    async fn retrieve_similar_contexts(&self, market_state: &Value) -> Result<Vec<crate::vector::ContextEntry>> {\n        let search_text = format!(\n            \"Market analysis for {} with {} trend, {} volume, {} sentiment\",\n            market_state[\"symbol\"].as_str().unwrap_or(\"unknown\"),\n            market_state[\"ml_signals\"][\"price_signals\"][\"trend\"].as_str().unwrap_or(\"unknown\"),\n            market_state[\"ml_signals\"][\"technical_indicators\"][\"volume_signal\"].as_str().unwrap_or(\"unknown\"),\n            market_state[\"sentiment\"][\"sentiment_label\"].as_str().unwrap_or(\"unknown\")\n        );\n        \n        let embedding = self.embedder.embed(&search_text).await?;\n        let similar_contexts = self.vector_store.similarity_search(embedding, 5).await?;\n        \n        info!(\"Found {} similar contexts\", similar_contexts.len());\n        Ok(similar_contexts)\n    }
    
    /// Get relevant playbook entries (placeholder implementation)
    async fn get_playbook_entries(&self, _market_state: &Value) -> Result<Vec<String>> {\n        // Placeholder for playbook system - would query actual playbook database\n        Ok(vec![\n            \"When VIX < 20 and momentum positive, calls have 68% win rate\".to_string(),\n            \"High volume + positive sentiment often leads to continuation\".to_string(),\n            \"Avoid trading on low confidence days (< 60%)\".to_string()\n        ])\n    }
    
    /// Generate trading decision using LLM
    async fn generate_decision(\n        &self,\n        market_state: &Value,\n        ml_signals: &Value,\n        similar_contexts: &[crate::vector::ContextEntry],\n        playbook_entries: &[String],\n    ) -> Result<TradingDecision> {\n        let prompt = ACEPrompts::morning_decision_prompt(\n            market_state,\n            ml_signals,\n            similar_contexts,\n            playbook_entries,\n            &Utc::now().format(\"%Y-%m-%d\").to_string()\n        );\n        \n        // Try to get structured decision from LLM\n        match self.llm_client.generate_json::<TradingDecision>(&prompt, None).await {\n            Ok(decision) => {\n                info!(\"LLM generated decision: {} with {:.1}% confidence\", \n                      decision.action, decision.confidence * 100.0);\n                Ok(decision)\n            }\n            Err(e) => {\n                error!(\"LLM decision generation failed: {}\", e);\n                // Fallback decision based on signals\n                Ok(self.generate_fallback_decision(market_state, ml_signals).await)\n            }\n        }\n    }
    
    /// Generate fallback decision when LLM is unavailable
    async fn generate_fallback_decision(\n        &self,\n        market_state: &Value,\n        ml_signals: &Value\n    ) -> TradingDecision {\n        let momentum_score = ml_signals[\"momentum_score\"].as_f64().unwrap_or(0.5);\n        let daily_change = market_state[\"market_data\"][\"daily_change_pct\"].as_f64().unwrap_or(0.0);\n        let sentiment_score = market_state[\"sentiment\"][\"reddit_score\"].as_f64().unwrap_or(0.5);\n        \n        let action = if momentum_score > 0.6 && daily_change > 0.5 {\n            \"BUY_CALLS\"\n        } else if momentum_score < 0.4 && daily_change < -0.5 {\n            \"BUY_PUTS\"\n        } else {\n            \"STAY_FLAT\"\n        };\n        \n        let confidence = ((momentum_score - 0.5).abs() + sentiment_score.max(1.0 - sentiment_score)) / 2.0;\n        \n        TradingDecision {\n            action: action.to_string(),\n            confidence: confidence.max(0.1).min(0.9),\n            reasoning: format!(\n                \"Fallback decision based on momentum score {:.2}, daily change {:.2}%, sentiment {:.2}\",\n                momentum_score, daily_change, sentiment_score\n            ),\n            key_factors: vec![\n                format!(\"Momentum: {:.1}%\", momentum_score * 100.0),\n                format!(\"Price change: {:.2}%\", daily_change),\n                format!(\"Sentiment: {:.2}\", sentiment_score)\n            ],\n            risk_factors: vec![\n                \"LLM unavailable - using simplified logic\".to_string(),\n                \"Limited market analysis\".to_string()\n            ],\n            similar_pattern_reference: None,\n            position_size_multiplier: 0.5, // Reduced size for fallback\n        }\n    }
    
    /// Display analysis results to user
    async fn display_analysis_results(\n        &self,\n        decision: &TradingDecision,\n        context_id: uuid::Uuid,\n        similar_contexts: &[crate::vector::ContextEntry]\n    ) {\n        println!(\"\\n🎯 TRADING RECOMMENDATION\");\n        println!(\"========================\");\n        println!(\"Action: {}\", decision.action);\n        println!(\"Confidence: {:.1}%\", decision.confidence * 100.0);\n        println!(\"Position Size: {:.0}% of base\", decision.position_size_multiplier * 100.0);\n        println!(\"\\n💭 REASONING:\");\n        println!(\"{}\", decision.reasoning);\n        \n        if !decision.key_factors.is_empty() {\n            println!(\"\\n✅ KEY FACTORS:\");\n            for factor in &decision.key_factors {\n                println!(\"  • {}\", factor);\n            }\n        }\n        \n        if !decision.risk_factors.is_empty() {\n            println!(\"\\n⚠️  RISK FACTORS:\");\n            for risk in &decision.risk_factors {\n                println!(\"  • {}\", risk);\n            }\n        }\n        \n        if let Some(pattern) = &decision.similar_pattern_reference {\n            println!(\"\\n📊 SIMILAR PATTERN: {}\", pattern);\n        }\n        \n        println!(\"\\n🧠 ACE CONTEXT:\");\n        println!(\"Context ID: {}\", context_id);\n        println!(\"Similar contexts found: {}\", similar_contexts.len());\n        \n        if !similar_contexts.is_empty() {\n            println!(\"\\n📈 MOST SIMILAR PAST DECISIONS:\");\n            for (i, ctx) in similar_contexts.iter().take(3).enumerate() {\n                println!(\"  {}. {} (similarity: {:.2}, confidence: {:.1}%)\", \n                         i + 1, \n                         ctx.reasoning,\n                         ctx.similarity.unwrap_or(0.0),\n                         ctx.confidence * 100.0);\n            }\n        }\n        \n        println!(\"\\n\" + &\"=\".repeat(50));\n    }
    
    /// Calculate price volatility from OHLCV data
    fn calculate_volatility(&self, ohlcv_data: &[crate::data::OHLCV]) -> f64 {\n        if ohlcv_data.len() < 2 {\n            return 0.0;\n        }\n        \n        let returns: Vec<f64> = ohlcv_data.windows(2)\n            .map(|pair| ((pair[1].close - pair[0].close) / pair[0].close))\n            .collect();\n        \n        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;\n        let variance = returns.iter()\n            .map(|r| (r - mean_return).powi(2))\n            .sum::<f64>() / returns.len() as f64;\n        \n        variance.sqrt() * 100.0 // Convert to percentage\n    }
    
    /// Assess current market regime\n    async fn assess_market_regime(&self, market_data: &Value, ml_signals: &Value) -> String {\n        let daily_change = market_data[\"daily_change_pct\"].as_f64().unwrap_or(0.0);\n        let volatility = market_data[\"volatility_20d\"].as_f64().unwrap_or(0.0);\n        let momentum = ml_signals[\"momentum_score\"].as_f64().unwrap_or(0.5);\n        \n        match () {\n            _ if volatility > 3.0 => \"HIGH_VOLATILITY\".to_string(),\n            _ if daily_change.abs() > 2.0 => \"TRENDING\".to_string(),\n            _ if momentum > 0.7 || momentum < 0.3 => \"MOMENTUM\".to_string(),\n            _ => \"RANGING\".to_string(),\n        }\n    }
    \n    /// Assess quality of available analysis data\n    async fn assess_analysis_quality(&self, market_data: &Value, research_data: &Value, sentiment_data: &Value) -> Value {\n        let market_quality = if market_data[\"data_points\"].as_u64().unwrap_or(0) >= 20 { \"good\" } else { \"limited\" };\n        let research_quality = if research_data[\"source\"] != \"fallback\" { \"good\" } else { \"limited\" };\n        let sentiment_quality = if sentiment_data[\"source\"] != \"fallback\" { \"good\" } else { \"limited\" };\n        \n        json!({\n            \"market_data\": market_quality,\n            \"research\": research_quality,\n            \"sentiment\": sentiment_quality,\n            \"overall\": if market_quality == \"good\" && (research_quality == \"good\" || sentiment_quality == \"good\") {\n                \"sufficient\"\n            } else {\n                \"limited\"\n            }\n        })\n    }\n}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n    use serde_json::json;\n    \n    #[tokio::test]\n    #[ignore] // Requires full database and API setup\n    async fn test_morning_orchestrator() {\n        // This test would require:\n        // 1. Test database with migrations\n        // 2. Mock or real API clients\n        // 3. Local LLM setup\n        // Include in integration testing suite\n    }\n    \n    #[test]\n    fn test_volatility_calculation() {\n        use crate::data::OHLCV;\n        use chrono::NaiveDate;\n        \n        // Mock orchestrator for testing\n        let pool = todo!(\"Mock pool\");\n        let config = todo!(\"Mock config\");\n        // let orchestrator = MorningOrchestrator::new(pool, config).await.unwrap();\n        \n        let test_data = vec![\n            OHLCV {\n                symbol: \"TEST\".to_string(),\n                date: NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),\n                open: 100.0, high: 105.0, low: 95.0, close: 102.0, volume: 1000,\n                source: \"test\".to_string(),\n            },\n            OHLCV {\n                symbol: \"TEST\".to_string(),\n                date: NaiveDate::from_ymd_opt(2025, 1, 2).unwrap(),\n                open: 102.0, high: 108.0, low: 98.0, close: 104.0, volume: 1200,\n                source: \"test\".to_string(),\n            },\n        ];\n        \n        // This test is incomplete due to the struct initialization requirements\n        // Would need dependency injection or factory pattern for proper testing\n    }\n}
//! Morning analysis orchestrator
//! Coordinates the full ACE pipeline: data → ML → context retrieval → LLM → decision

use anyhow::Result;
use chrono::Utc;
use serde_json::{json, Value};
use sqlx::PgPool;
use tracing::{error, info, warn};

use crate::{
    ace::{
        sanitize::validate_trading_decision, ACEPrompts, ContextDAO, PlaybookDAO, TradingDecision,
    },
    config::Config,
    data::{MarketDataClient, ResearchClient, SentimentClient},
    embeddings::EmbeddingGemma,
    llm::LLMClient,
    vector::VectorStore,
};

/// Morning analysis orchestrator
pub struct MorningOrchestrator {
    _pool: PgPool,  // Kept for potential future use
    _config: Config,  // Kept for potential future use
    market_client: MarketDataClient,
    research_client: ResearchClient,
    sentiment_client: SentimentClient,
    embedder: EmbeddingGemma,
    vector_store: VectorStore,
    context_dao: ContextDAO,
    playbook_dao: PlaybookDAO,
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

        // Initialize embedder using Cloudflare Workers AI
        let embedder = if let (Some(cf_account_id), Some(cf_api_token)) = (
            config.apis.cloudflare_account_id.clone(),
            config.apis.cloudflare_api_token.clone(),
        ) {
            info!("Using Cloudflare Workers AI for embeddings (@cf/baai/bge-base-en-v1.5, 768 dimensions)");
            EmbeddingGemma::from_cloudflare(cf_account_id, cf_api_token).await?
        } else {
            return Err(anyhow::anyhow!("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are required for embeddings"));
        };

        let vector_store = VectorStore::new(pool.clone()).await?;
        let context_dao = ContextDAO::new(pool.clone());
        let playbook_dao = PlaybookDAO::new(pool.clone());
        let llm_client = LLMClient::from_config(&config).await?;

        // Ensure HNSW index exists
        vector_store
            .ensure_hnsw_index("ace_contexts", "embedding")
            .await?;

        info!("Morning Orchestrator initialized successfully");

        Ok(Self {
            _pool: pool,
            _config: config,
            market_client,
            research_client,
            sentiment_client,
            embedder,
            vector_store,
            context_dao,
            playbook_dao,
            llm_client,
        })
    }

    /// Run full morning analysis for a symbol
    pub async fn analyze(&self, symbol: &str) -> Result<TradingDecision> {
        info!("🌅 Starting morning analysis for {}", symbol);

        // Step 1: Fetch market data
        info!("📊 Fetching market data...");
        let market_data = self.fetch_market_data(symbol).await?;

        // Step 2: Compute technical indicators
        info!("📊 Computing technical indicators...");
        let ml_signals = self.compute_ml_signals(&market_data).await?;

        // Step 3: Fetch research and sentiment (both can fail gracefully)
        info!("🔍 Gathering research and sentiment...");

        // Fetch research (required, falls back to limited data)
        let research_data = self.fetch_research(symbol).await?;

        // Fetch sentiment (optional, may be excluded if unavailable)
        let sentiment_data = match self.fetch_sentiment(symbol).await {
            Ok(data) => Some(data),
            Err(e) => {
                warn!("Sentiment data unavailable: {} - continuing without sentiment", e);
                None
            }
        };

        // Log research results
        let research_count = research_data["results"].as_array().map(|r| r.len()).unwrap_or(0);
        let research_source = research_data["source"].as_str().unwrap_or("unknown");
        info!("📚 Retrieved {} research results from {}", research_count, research_source);

        // Check for degraded services and alert
        self.check_service_health(&research_data, &sentiment_data).await;

        // Step 4: Build market state representation
        info!("🏗️ Building market state representation...");
        let market_state = self
            .build_market_state(
                symbol,
                &market_data,
                &ml_signals,
                &research_data,
                &sentiment_data,
            )
            .await?;

        // Step 5: Retrieve similar historical contexts
        info!("🧠 Retrieving similar historical contexts...");
        let similar_contexts = self.retrieve_similar_contexts(&market_state).await?;

        // Step 6: Get relevant playbook entries
        let playbook_entries = self.get_playbook_entries(&market_state).await?;

        // Step 7: Generate decision via LLM
        info!("💡 Generating trading decision...");
        let decision = self
            .generate_decision(
                &market_state,
                &ml_signals,
                &similar_contexts,
                &playbook_entries,
            )
            .await?;

        // Step 8: Persist context with embedding
        info!("💾 Persisting ACE context...");
        let context_embedding = self
            .embedder
            .embed(&serde_json::to_string(&market_state)?)
            .await?;
        let context_id = self
            .context_dao
            .insert_context(
                &market_state,
                &serde_json::to_value(&decision)?,
                &decision.reasoning,
                decision.confidence,
                None, // No outcome yet - will be filled in evening review
                context_embedding,
            )
            .await?;

        // Step 9: Display results
        self.display_analysis_results(&decision, context_id, &similar_contexts)
            .await;

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

    /// Compute technical indicators from market data
    /// Note: This calculates RSI, MACD-like signals, volume analysis, and momentum scores
    /// These are NOT machine learning models, but traditional technical analysis indicators
    async fn compute_ml_signals(&self, market_data: &Value) -> Result<Value> {
        // Extract market data
        let daily_change = market_data["daily_change_pct"].as_f64().unwrap_or(0.0);
        let volatility = market_data["volatility_20d"].as_f64().unwrap_or(0.0);
        let _latest_price = market_data["latest_price"].as_f64().unwrap_or(0.0);

        // Calculate simplified technical indicators for quick momentum assessment
        // RSI approximation based on daily change momentum (not standard 14-period RSI)
        let rsi = 50.0 + (daily_change * 5.0).clamp(-50.0, 50.0);

        // MACD signal based on trend
        let macd_signal = if daily_change > 0.0 {
            "bullish"
        } else {
            "bearish"
        };
        let volume_signal = if market_data["volume"].as_i64().unwrap_or(0)
            > market_data["avg_volume_20d"].as_i64().unwrap_or(0)
        {
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
            _ => 0.1,
        };

        // Calculate confidence based on signal alignment and strength
        let signal_confidence = {
            let mut conf: f32 = 0.5; // Base confidence

            // Increase confidence when signals align
            if (daily_change > 0.5 && volume_signal == "high")
                || (daily_change < -0.5 && volume_signal == "high")
            {
                conf += 0.2; // Strong volume confirms move
            }

            // Reduce confidence in high volatility
            if volatility > 3.0 {
                conf -= 0.15;
            } else if volatility < 1.5 {
                conf += 0.1; // Low volatility is more predictable
            }

            // RSI extremes reduce confidence (overbought/oversold)
            if rsi > 70.0 || rsi < 30.0 {
                conf -= 0.1;
            }

            conf.clamp(0.2, 0.95)
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
            "indicator_confidence": signal_confidence,
            "signal_summary": format!("{} momentum with {} volume (technical confidence: {:.1}%)",
                                       if daily_change > 0.0 { "Positive" } else { "Negative" },
                                       volume_signal,
                                       signal_confidence * 100.0)
        }))
    }

    /// Fetch research data
    async fn fetch_research(&self, symbol: &str) -> Result<Value> {
        let query = format!("{} stock market outlook earnings analysis", symbol);
        self.research_client
            .search(&query)
            .await
            .map_err(|e| {
                error!("Research fetch failed: {}", e);
                anyhow::anyhow!("Failed to fetch research data: {}. Ensure Exa API key is configured and service is available.", e)
            })
    }

    /// Fetch sentiment data
    async fn fetch_sentiment(&self, symbol: &str) -> Result<Value> {
        self.sentiment_client
            .analyze_reddit(Some(symbol))
            .await
            .map_err(|e| {
                error!("Sentiment fetch failed: {}", e);
                anyhow::anyhow!("Failed to fetch sentiment data: {}. Ensure Reddit API credentials are configured.", e)
            })
    }

    /// Build comprehensive market state representation
    async fn build_market_state(
        &self,
        symbol: &str,
        market_data: &Value,
        ml_signals: &Value,
        research_data: &Value,
        sentiment_data: &Option<Value>,
    ) -> Result<Value> {
        let current_time = Utc::now();

        let mut state = json!({
            "timestamp": current_time,
            "symbol": symbol,
            "market_data": market_data,
            "ml_signals": ml_signals,
            "research_summary": {
                "source_count": research_data["results"].as_array().map(|r| r.len()).unwrap_or(0),
                "key_themes": research_data["autoprompt_string"].as_str().unwrap_or("Research data available (no summary generated by API)"),
                "availability": if research_data["source"] == "fallback" { "limited" } else { "full" },
                "data_source": research_data["source"].as_str().unwrap_or("unknown")
            },
            "market_regime": self.assess_market_regime(market_data, ml_signals).await,
        });

        // Only include sentiment if available
        if let Some(sentiment) = sentiment_data {
            state["sentiment"] = json!({
                "reddit_score": sentiment["score"].as_f64().unwrap_or(0.5),
                "sentiment_label": match sentiment["score"].as_f64().unwrap_or(0.5) {
                    s if s > 0.7 => "bullish",
                    s if s > 0.3 => "neutral",
                    _ => "bearish"
                },
                "source": sentiment["source"].as_str().unwrap_or("reddit"),
                "sample_size": sentiment["sample_size"].as_u64().unwrap_or(0)
            });
        } else {
            state["sentiment"] = json!({"availability": "unavailable", "note": "Sentiment data not available for this analysis"});
        }

        state["analysis_quality"] = self.assess_analysis_quality(market_data, research_data, sentiment_data).await;

        Ok(state)
    }

    /// Retrieve similar historical contexts using vector search
    async fn retrieve_similar_contexts(
        &self,
        market_state: &Value,
    ) -> Result<Vec<crate::vector::ContextEntry>> {
        let search_text = format!(
            "Market analysis for {} with {} trend, {} volume, {} sentiment",
            market_state["symbol"].as_str().unwrap_or("unknown"),
            market_state["ml_signals"]["price_signals"]["trend"]
                .as_str()
                .unwrap_or("unknown"),
            market_state["ml_signals"]["technical_indicators"]["volume_signal"]
                .as_str()
                .unwrap_or("unknown"),
            market_state["sentiment"]["sentiment_label"]
                .as_str()
                .unwrap_or("unknown")
        );

        let embedding = self.embedder.embed(&search_text).await?;
        let similar_contexts = self.vector_store.similarity_search(embedding, 5).await?;

        info!("Found {} similar contexts", similar_contexts.len());
        Ok(similar_contexts)
    }

    /// Get relevant playbook entries from the ACE playbook database
    async fn get_playbook_entries(&self, _market_state: &Value) -> Result<Vec<String>> {
        // Query playbook bullets from the last 30 days with high confidence
        let bullets = self.playbook_dao.get_recent_bullets(30, 20).await?;

        // Filter for high confidence bullets and return their content
        let entries: Vec<String> = bullets
            .iter()
            .filter(|b| b.confidence > 0.6)
            .map(|b| b.content.clone())
            .collect();

        info!("Retrieved {} playbook entries for context", entries.len());
        Ok(entries)
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
            &Utc::now().format("%Y-%m-%d").to_string(),
        );

        // Get structured decision from LLM - no fallback
        let decision = self
            .llm_client
            .generate_json::<TradingDecision>(&prompt, None)
            .await
            .map_err(|e| {
                error!("LLM decision generation failed: {}", e);
                anyhow::anyhow!("Failed to generate trading decision from LLM: {}. Ensure Ollama is running with 'ollama serve' and model is available.", e)
            })?;

        info!(
            "LLM generated decision: {} with {:.1}% confidence",
            decision.action,
            decision.confidence * 100.0
        );

        // Validate the decision
        let decision_json = serde_json::to_value(&decision)?;
        validate_trading_decision(&decision_json).map_err(|e| {
            error!("LLM decision failed validation: {}", e);
            anyhow::anyhow!("LLM generated invalid trading decision: {}. This indicates a problem with the LLM output format.", e)
        })?;

        info!("Decision passed validation");
        Ok(decision)
    }


    /// Display analysis results to user
    async fn display_analysis_results(
        &self,
        decision: &TradingDecision,
        context_id: uuid::Uuid,
        similar_contexts: &[crate::vector::ContextEntry],
    ) {
        println!("\n🎯 TRADING RECOMMENDATION");
        println!("========================");
        println!("Action: {}", decision.action);
        println!("Confidence: {:.1}%", decision.confidence * 100.0);
        println!(
            "Position Size: {:.0}% of base",
            decision.position_size_multiplier * 100.0
        );
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
                println!(
                    "  {}. {} (similarity: {:.2}, confidence: {:.1}%)",
                    i + 1,
                    ctx.reasoning,
                    ctx.similarity.unwrap_or(0.0),
                    ctx.confidence * 100.0
                );
            }
        }

        println!("\n{}", "=".repeat(50));
    }

    /// Calculate price volatility from OHLCV data
    fn calculate_volatility(&self, ohlcv_data: &[crate::data::OHLCV]) -> f64 {
        if ohlcv_data.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = ohlcv_data
            .windows(2)
            .map(|pair| (pair[1].close - pair[0].close) / pair[0].close)
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;

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
    async fn assess_analysis_quality(
        &self,
        market_data: &Value,
        research_data: &Value,
        sentiment_data: &Option<Value>,
    ) -> Value {
        let market_quality = if market_data["data_points"].as_u64().unwrap_or(0) >= 20 {
            "good"
        } else {
            "limited"
        };
        let research_quality = if research_data["results"].as_array().map(|r| !r.is_empty()).unwrap_or(false) {
            "good"
        } else {
            "limited"
        };
        let sentiment_quality = if sentiment_data.as_ref().and_then(|s| s.get("score")).is_some() {
            "good"
        } else {
            "unavailable"
        };

        json!({
            "market_data": market_quality,
            "research": research_quality,
            "sentiment": sentiment_quality,
            "overall": if market_quality == "good" && research_quality == "good" {
                "sufficient"
            } else if market_quality == "good" && sentiment_quality == "good" {
                "sufficient"
            } else {
                "limited"
            }
        })
    }

    /// Check service health and alert on degraded services
    /// This detects when external APIs are using fallback mode or unavailable
    async fn check_service_health(&self, research_data: &Value, sentiment_data: &Option<Value>) {
        let mut degraded_services = Vec::new();

        // Check if research is using fallback
        if research_data["source"].as_str() == Some("fallback") {
            warn!("⚠️  DEGRADED SERVICE: Research API (Exa) is unavailable - using fallback data");
            degraded_services.push("Research/Exa");
        }

        // Check if sentiment is unavailable
        if sentiment_data.is_none() {
            warn!("⚠️  SERVICE UNAVAILABLE: Reddit API - sentiment data excluded from analysis");
            degraded_services.push("Reddit Sentiment");
        }

        // If any services are degraded, emit a consolidated warning
        if !degraded_services.is_empty() {
            error!(
                degraded_services = ?degraded_services,
                "❌ SYSTEM HEALTH WARNING: {} external service(s) degraded - {}. Trading decisions may be less reliable. Check API credentials in .env file.",
                degraded_services.len(),
                degraded_services.join(", ")
            );

            // Also print to stdout so user sees it in the terminal
            eprintln!("\n╔════════════════════════════════════════════════════════════╗");
            eprintln!("║          ⚠️  SYSTEM HEALTH WARNING ⚠️                      ║");
            eprintln!("╚════════════════════════════════════════════════════════════╝");
            eprintln!("The following external services are degraded:");
            for service in &degraded_services {
                eprintln!("  ❌ {}", service);
            }
            eprintln!("\nImpact: Trading decisions will rely on limited data.");
            eprintln!("Action: Check your .env file and verify API credentials.");
            eprintln!("═══════════════════════════════════════════════════════════\n");
        } else {
            info!("✅ All external services healthy (Research, Sentiment)");
        }
    }
}

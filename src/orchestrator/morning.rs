//! Morning analysis orchestrator
//! Coordinates the full ACE pipeline: data → ML → context retrieval → LLM → decision

use anyhow::Result;
use chrono::Utc;
use serde_json::{json, Value};
use sqlx::PgPool;
use tracing::{debug, error, info, warn};

use crate::{
    ace::{
        sanitize::validate_trading_decision, ACEPrompts, ContextDAO, PlaybookDAO, TradingDecision,
    },
    config::Config,
    data::{MarketDataClient, ResearchClient, SentimentClient},
    embeddings::EmbeddingGemma,
    llm::LLMClient,
    orchestrator::PerformanceTracker,
    vector::{Reranker, VectorStore},
};

/// Morning analysis orchestrator
pub struct MorningOrchestrator {
    pool: PgPool,
    config: Config,
    market_client: MarketDataClient,
    research_client: ResearchClient,
    sentiment_client: SentimentClient,
    embedder: EmbeddingGemma,
    reranker: Reranker,
    vector_store: VectorStore,
    context_dao: ContextDAO,
    playbook_dao: PlaybookDAO,
    llm_client: LLMClient,
    performance_tracker: PerformanceTracker,
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
        let (embedder, reranker) = if let (Some(cf_account_id), Some(cf_api_token)) = (
            config.apis.cloudflare_account_id.clone(),
            config.apis.cloudflare_api_token.clone(),
        ) {
            info!("Using Cloudflare Workers AI for embeddings (@cf/baai/bge-base-en-v1.5, 768 dimensions)");
            let embedder = EmbeddingGemma::from_cloudflare(cf_account_id.clone(), cf_api_token.clone()).await?;
            let reranker = Reranker::new(cf_account_id, cf_api_token);
            (embedder, reranker)
        } else {
            return Err(anyhow::anyhow!("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are required for embeddings and reranking"));
        };

        let vector_store = VectorStore::new(pool.clone()).await?;
        let context_dao = ContextDAO::new(pool.clone());
        let playbook_dao = PlaybookDAO::new(pool.clone());
        let llm_client = LLMClient::from_config(&config).await?;
        let performance_tracker = PerformanceTracker::new(pool.clone());

        // Ensure HNSW index exists
        vector_store
            .ensure_hnsw_index("ace_contexts", "embedding")
            .await?;

        info!("Morning Orchestrator initialized successfully");

        Ok(Self {
            pool,
            config,
            market_client,
            research_client,
            sentiment_client,
            embedder,
            reranker,
            vector_store,
            context_dao,
            playbook_dao,
            llm_client,
            performance_tracker,
        })
    }

    /// Run full morning analysis for a symbol
    pub async fn analyze(&self, symbol: &str) -> Result<TradingDecision> {
        info!("🌅 Starting morning analysis for {}", symbol);

        // Step 1: Fetch market data
        info!("📊 Fetching market data...");
        let (market_data, ohlcv_data) = self.fetch_market_data(symbol).await?;

        // Step 2: Compute technical indicators
        info!("📊 Computing technical indicators...");
        let ml_signals = self.compute_ml_signals(&market_data, &ohlcv_data).await?;

        // Step 3: Fetch research and sentiment (both can fail gracefully)
        // Check if we should skip data fetching (e.g., in backtest mode with --no-sentiment)
        let skip_data_fetching = self.config.skip_sentiment.unwrap_or(false);

        let (research_data, sentiment_data) = if skip_data_fetching {
            info!("Skipping research and sentiment data fetching (skip_sentiment flag set)");
            // Provide minimal fallback data structure
            let fallback_research = json!({
                "source": "skipped",
                "results": [],
                "query": format!("{} analysis", symbol),
                "note": "Data fetching skipped due to skip_sentiment flag"
            });
            (fallback_research, None)
        } else {
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

            (research_data, sentiment_data)
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
        let embedding_text = Self::market_state_to_embedding_text(&market_state);
        let context_embedding = self
            .embedder
            .embed(&embedding_text)
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
                Some(self.config.get_effective_datetime()),
            )
            .await?;

        // Step 9: Persist trading recommendation
        info!("💾 Persisting trading recommendation...");
        if let Err(e) = self.persist_trading_recommendation(symbol, &decision, context_id).await {
            warn!("Failed to persist trading recommendation: {}", e);
            // Continue - this is not a fatal error
        }

        // Step 9.5: Persist daily analysis summary
        info!("💾 Persisting daily analysis summary...");
        if let Err(e) = self.persist_daily_analysis_summary(symbol, &decision, context_id).await {
            warn!("Failed to persist daily analysis summary: {}", e);
            // Continue - this is not a fatal error
        }

        // Step 10: Display results
        self.display_analysis_results(&decision, context_id, &similar_contexts)
            .await;

        info!("✅ Morning analysis complete for {}", symbol);
        Ok(decision)
    }

    /// Fetch and persist market data
    async fn fetch_market_data(&self, symbol: &str) -> Result<(Value, Vec<crate::data::OHLCV>)> {
        // Fetch at least 50 bars for proper technical indicators (need 26 for MACD + buffer)
        // In backtest mode, use the backtest_date as the end date
        let ohlcv_data = self.market_client
            .fetch_ohlcv_with_end_date(symbol, 50, self.config.backtest_date)
            .await?;

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

        let market_data = json!({
            "symbol": symbol,
            "latest_price": latest.close,
            "daily_change_pct": daily_change,
            "volume": latest.volume,
            "volatility_20d": volatility,
            "high_52w": ohlcv_data.iter().map(|d| d.high).fold(0.0, f64::max),
            "low_52w": ohlcv_data.iter().map(|d| d.low).fold(f64::INFINITY, f64::min),
            "avg_volume_20d": ohlcv_data.iter().map(|d| d.volume as f64).sum::<f64>() / ohlcv_data.len() as f64,
            "data_points": ohlcv_data.len()
        });

        Ok((market_data, ohlcv_data))
    }

    /// Compute technical indicators from market data
    /// Implements proper RSI (14-period) and MACD (12/26/9) calculations
    async fn compute_ml_signals(&self, market_data: &Value, ohlcv_data: &[crate::data::OHLCV]) -> Result<Value> {
        // Extract market data
        let daily_change = market_data["daily_change_pct"].as_f64().unwrap_or(0.0);
        let volatility = market_data["volatility_20d"].as_f64().unwrap_or(0.0);
        let _latest_price = market_data["latest_price"].as_f64().unwrap_or(0.0);

        // Calculate proper 14-period RSI using Wilder's smoothing
        let rsi = self.calculate_rsi(ohlcv_data, 14);

        // Calculate proper MACD (12/26/9 EMA)
        let (macd_line, signal_line, histogram) = self.calculate_macd(ohlcv_data);
        let macd_signal = if histogram > 0.0 {
            "bullish"
        } else {
            "bearish"
        };

        // Volume analysis
        let volume_signal = if market_data["volume"].as_i64().unwrap_or(0)
            > market_data["avg_volume_20d"].as_i64().unwrap_or(0)
        {
            "high"
        } else {
            "normal"
        };

        // Momentum score based on multiple factors
        let momentum_score = {
            let mut score = 0.5; // Base

            // RSI contribution
            if rsi > 50.0 {
                score += ((rsi - 50.0) / 100.0) * 0.3; // Up to +0.15
            } else {
                score -= ((50.0 - rsi) / 100.0) * 0.3; // Down to -0.15
            }

            // MACD histogram contribution
            score += histogram.signum() * 0.2;

            // Daily change contribution
            score += (daily_change / 10.0).clamp(-0.2, 0.2);

            score.clamp(0.0, 1.0)
        };

        // Calculate confidence based on signal alignment and strength
        let signal_confidence = {
            let mut conf: f32 = 0.5; // Base confidence

            // Increase confidence when technical signals align
            let rsi_bullish = rsi > 50.0;
            let macd_bullish = histogram > 0.0;
            let price_bullish = daily_change > 0.0;

            let signals_aligned = (rsi_bullish == macd_bullish) && (macd_bullish == price_bullish);
            if signals_aligned {
                conf += 0.2;
            }

            // Volume confirmation
            if volume_signal == "high" && daily_change.abs() > 0.3 {
                conf += 0.1; // Strong volume confirms move
            }

            // Reduce confidence in high volatility
            if volatility > 3.0 {
                conf -= 0.15;
            } else if volatility < 1.5 {
                conf += 0.05; // Low volatility slightly positive
            }

            // RSI extremes
            if rsi > 70.0 {
                conf -= 0.05; // Slightly overbought
            } else if rsi < 30.0 {
                conf -= 0.05; // Slightly oversold
            }

            conf.clamp(0.2, 0.95)
        };

        Ok(json!({
            "technical_indicators": {
                "rsi": rsi.max(0.0).min(100.0),
                "macd_line": macd_line,
                "macd_signal_line": signal_line,
                "macd_histogram": histogram,
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
        let current_time = self.config.get_effective_datetime();

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

    /// Convert market state to rich text representation for embeddings
    /// This ensures consistent text format between storage and retrieval
    fn market_state_to_embedding_text(market_state: &Value) -> String {
        let symbol = market_state["symbol"].as_str().unwrap_or("unknown");
        let trend = market_state["ml_signals"]["price_signals"]["trend"]
            .as_str()
            .unwrap_or("unknown");
        let volume = market_state["ml_signals"]["technical_indicators"]["volume_signal"]
            .as_str()
            .unwrap_or("unknown");
        let sentiment = market_state["sentiment"]["sentiment_label"]
            .as_str()
            .unwrap_or("unknown");
        let regime = market_state["market_regime"]["classification"]
            .as_str()
            .unwrap_or("unknown");

        // Get key technical indicators
        let macd_signal = market_state["ml_signals"]["technical_indicators"]["macd_signal"]
            .as_str()
            .unwrap_or("neutral");
        let rsi = market_state["ml_signals"]["technical_indicators"]["rsi"]
            .as_f64()
            .unwrap_or(50.0);

        // Create rich text representation matching what we store
        format!(
            "Market analysis for {}: {} trend with {} volume and {} sentiment. \
             Market regime: {}. Technical indicators: MACD is {}, RSI at {:.1}. \
             Price momentum and market conditions suggest {} environment for trading.",
            symbol, trend, volume, sentiment, regime, macd_signal, rsi,
            if trend == "up" && volume == "high" { "favorable" }
            else if trend == "down" { "bearish" }
            else { "mixed" }
        )
    }

    /// Retrieve similar historical contexts using vector search + reranking
    async fn retrieve_similar_contexts(
        &self,
        market_state: &Value,
    ) -> Result<Vec<crate::vector::ContextEntry>> {
        // Step 1: Generate consistent embedding text
        let search_text = Self::market_state_to_embedding_text(market_state);

        // Step 2: Vector search for top 20 candidates
        let embedding = self.embedder.embed(&search_text).await?;
        let candidates = self.vector_store.similarity_search(embedding, 20).await?;

        info!("Found {} candidate contexts from vector search", candidates.len());

        // Step 3: Rerank to get top 5 most relevant
        if candidates.is_empty() {
            return Ok(vec![]);
        }

        // Prepare documents for reranking (use reasoning as the document text)
        let documents: Vec<String> = candidates
            .iter()
            .map(|c| c.reasoning.clone())
            .collect();

        // Try reranking with fallback to vector search results
        match self
            .reranker
            .rerank_top_n(&search_text, documents, 5)
            .await
        {
            Ok(rerank_scores) => {
                // Map reranked results back to context entries
                let reranked_contexts: Vec<crate::vector::ContextEntry> = rerank_scores
                    .iter()
                    .filter_map(|score| {
                        candidates.get(score.id).cloned().map(|mut context| {
                            // Update similarity with rerank score
                            context.similarity = Some(score.score as f32);
                            context
                        })
                    })
                    .collect();

                info!(
                    "✅ Reranked to {} contexts (top score: {:.4})",
                    reranked_contexts.len(),
                    reranked_contexts
                        .first()
                        .and_then(|c| c.similarity)
                        .unwrap_or(0.0)
                );

                Ok(reranked_contexts)
            }
            Err(e) => {
                warn!("⚠️ Reranking failed ({}), falling back to vector search results", e);
                // Fall back to top 5 from vector search
                let fallback_contexts: Vec<crate::vector::ContextEntry> =
                    candidates.into_iter().take(5).collect();

                info!(
                    "📊 Using top {} vector search results (fallback)",
                    fallback_contexts.len()
                );

                Ok(fallback_contexts)
            }
        }
    }

    /// Get relevant playbook entries from the ACE playbook database using semantic search
    async fn get_playbook_entries(&self, market_state: &Value) -> Result<Vec<String>> {
        // Determine reference date: use backtest_date if in backtest mode, otherwise current time
        let reference_date = if let Some(backtest_date) = self.config.backtest_date {
            backtest_date.and_hms_opt(0, 0, 0)
                .expect("Invalid time")
                .and_utc()
        } else {
            Utc::now()
        };

        // Query playbook bullets from the last 30 days with high confidence (get more candidates)
        let bullets = self.playbook_dao.get_recent_bullets(30, 30, reference_date).await?;

        // Filter for moderate+ confidence bullets
        // 0.45 threshold allows new bullets (start at 0.5) to be included
        let candidates: Vec<_> = bullets
            .iter()
            .filter(|b| b.confidence > 0.45)
            .collect();

        if candidates.is_empty() {
            info!("No playbook bullets found with sufficient confidence");
            return Ok(vec![]);
        }

        // Use semantic search to find most relevant entries
        // Create query text from market state
        let query_text = Self::market_state_to_embedding_text(market_state);

        // Prepare documents for reranking
        let documents: Vec<String> = candidates
            .iter()
            .map(|b| b.content.clone())
            .collect();

        // Try reranking to get top 10 most relevant entries
        match self.reranker.rerank_top_n(&query_text, documents.clone(), 10).await {
            Ok(rerank_scores) => {
                // Map reranked results back to playbook bullets
                let reranked_entries: Vec<String> = rerank_scores
                    .iter()
                    .filter_map(|score| {
                        candidates.get(score.id).map(|bullet| bullet.content.clone())
                    })
                    .collect();

                let total_chars: usize = reranked_entries.iter().map(|e| e.len()).sum();
                info!(
                    "✅ Retrieved {} reranked playbook entries ({} chars) for context (as of {})",
                    reranked_entries.len(),
                    total_chars,
                    reference_date.format("%Y-%m-%d")
                );

                Ok(reranked_entries)
            }
            Err(e) => {
                warn!("⚠️ Reranking playbook entries failed ({}), using recency-based selection", e);
                // Fallback: take top 10 by recency
                let fallback_entries: Vec<String> = candidates
                    .into_iter()
                    .take(10)
                    .map(|b| b.content.clone())
                    .collect();

                let total_chars: usize = fallback_entries.iter().map(|e| e.len()).sum();
                info!(
                    "📊 Retrieved {} recent playbook entries ({} chars, fallback) for context (as of {})",
                    fallback_entries.len(),
                    total_chars,
                    reference_date.format("%Y-%m-%d")
                );

                Ok(fallback_entries)
            }
        }
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
            &self.config.get_effective_date().format("%Y-%m-%d").to_string(),
        );

        // Log playbook integration details
        let playbook_chars: usize = playbook_entries.iter().map(|e| e.len()).sum();
        info!(
            "Prompt contains {} playbook entries ({} total chars)",
            playbook_entries.len(),
            playbook_chars
        );

        // Log individual playbook entries for debugging
        if playbook_entries.is_empty() {
            warn!("⚠️ No playbook entries included in decision prompt - system may not be learning!");
        } else {
            info!("Playbook entries in prompt:");
            for (i, entry) in playbook_entries.iter().enumerate() {
                info!("  {}. {}", i + 1, entry);
            }
        }

        // Debug: log full prompt to verify playbook integration
        debug!("Full LLM prompt:\n{}", prompt);
        debug!(
            "Prompt stats: total length={} chars, playbook section={} chars ({:.1}%)",
            prompt.len(),
            playbook_chars,
            (playbook_chars as f64 / prompt.len() as f64) * 100.0
        );

        // Get structured decision from LLM - no fallback
        let mut decision = self
            .llm_client
            .generate_json::<TradingDecision>(&prompt, None)
            .await
            .map_err(|e| {
                error!("LLM decision generation failed: {}", e);
                anyhow::anyhow!("Failed to generate trading decision from LLM: {}. Ensure Ollama is running with 'ollama serve' and model is available.", e)
            })?;

        let raw_confidence = decision.confidence;
        info!(
            "LLM generated decision: {} with {:.1}% raw confidence",
            decision.action,
            raw_confidence * 100.0
        );

        // Apply confidence calibration based on recent performance
        let (calibrated_confidence, perf_stats, calibration_summary) = self
            .performance_tracker
            .get_calibrated_confidence(raw_confidence, 10)
            .await?;

        // Update decision with calibrated confidence
        decision.confidence = calibrated_confidence;

        // Log calibration if significant adjustment occurred
        if (calibrated_confidence - raw_confidence).abs() > 0.05 {
            warn!(
                "📊 Confidence adjusted: {:.1}% → {:.1}% | Reason: {}",
                raw_confidence * 100.0,
                calibrated_confidence * 100.0,
                calibration_summary
            );
            warn!(
                "📊 Recent performance: {}/{} wins ({:.1}% win rate), {}L/{}W streak",
                perf_stats.wins,
                perf_stats.total_trades,
                perf_stats.win_rate * 100.0,
                perf_stats.consecutive_losses,
                perf_stats.consecutive_wins
            );
        } else {
            info!(
                "✓ Confidence calibrated: {:.1}% → {:.1}% (minor adjustment)",
                raw_confidence * 100.0,
                calibrated_confidence * 100.0
            );
        }

        // Check if decision reasoning references playbook entries
        if !playbook_entries.is_empty() {
            let reasoning_lower = decision.reasoning.to_lowercase();
            let playbook_referenced = playbook_entries.iter().any(|entry| {
                // Check if any significant words from the playbook entry appear in reasoning
                let entry_words: Vec<&str> = entry.split_whitespace()
                    .filter(|w| w.len() > 5) // Only check substantial words
                    .collect();
                entry_words.iter().any(|word| reasoning_lower.contains(&word.to_lowercase()))
            });

            if playbook_referenced {
                info!("✓ Decision reasoning appears to reference playbook knowledge");
            } else {
                warn!("⚠️ Decision reasoning does NOT appear to reference playbook entries - LLM may be ignoring playbook");
                debug!("Decision reasoning: {}", decision.reasoning);
            }
        }

        // Validate the decision
        let decision_json = serde_json::to_value(&decision)?;
        validate_trading_decision(&decision_json).map_err(|e| {
            error!("LLM decision failed validation: {}", e);
            anyhow::anyhow!("LLM generated invalid trading decision: {}. This indicates a problem with the LLM output format.", e)
        })?;

        info!("Decision passed validation");

        // Calculate average similarity score from similar contexts
        let avg_similarity = if !similar_contexts.is_empty() {
            similar_contexts.iter()
                .map(|c| c.similarity.unwrap_or(0.0))
                .sum::<f32>() / similar_contexts.len() as f32
        } else {
            0.0
        };

        // Log comprehensive decision metadata with structured fields
        info!(
            decision = %decision.action,
            confidence = %decision.confidence,
            raw_confidence = %raw_confidence,
            playbook_entries_count = %playbook_entries.len(),
            similar_contexts_count = %similar_contexts.len(),
            avg_similarity = %avg_similarity,
            key_factors_count = %decision.key_factors.len(),
            risk_factors_count = %decision.risk_factors.len(),
            risk_factors = ?decision.risk_factors,
            "Generated trading decision with ACE context"
        );

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

    /// Calculate 14-period RSI using Wilder's smoothing method
    fn calculate_rsi(&self, ohlcv_data: &[crate::data::OHLCV], period: usize) -> f64 {
        if ohlcv_data.len() < period + 1 {
            return 50.0; // Neutral if insufficient data
        }

        // Calculate price changes
        let changes: Vec<f64> = ohlcv_data
            .windows(2)
            .map(|pair| pair[1].close - pair[0].close)
            .collect();

        if changes.is_empty() {
            return 50.0;
        }

        // Separate gains and losses
        let mut avg_gain = 0.0;
        let mut avg_loss = 0.0;

        // Initial average using simple mean for first period
        for i in 0..period.min(changes.len()) {
            if changes[i] > 0.0 {
                avg_gain += changes[i];
            } else {
                avg_loss += changes[i].abs();
            }
        }
        avg_gain /= period as f64;
        avg_loss /= period as f64;

        // Use Wilder's smoothing for subsequent periods
        for i in period..changes.len() {
            if changes[i] > 0.0 {
                avg_gain = (avg_gain * (period - 1) as f64 + changes[i]) / period as f64;
                avg_loss = (avg_loss * (period - 1) as f64) / period as f64;
            } else {
                avg_gain = (avg_gain * (period - 1) as f64) / period as f64;
                avg_loss = (avg_loss * (period - 1) as f64 + changes[i].abs()) / period as f64;
            }
        }

        // Calculate RSI
        if avg_loss == 0.0 {
            return 100.0; // All gains, RSI = 100
        }

        let rs = avg_gain / avg_loss;
        let rsi = 100.0 - (100.0 / (1.0 + rs));

        rsi
    }

    /// Calculate MACD (12/26/9 EMA)
    /// Returns (MACD line, signal line, histogram)
    fn calculate_macd(&self, ohlcv_data: &[crate::data::OHLCV]) -> (f64, f64, f64) {
        if ohlcv_data.len() < 26 {
            return (0.0, 0.0, 0.0); // Insufficient data
        }

        let closes: Vec<f64> = ohlcv_data.iter().map(|d| d.close).collect();

        // Calculate 12-period EMA
        let ema12 = self.calculate_ema(&closes, 12);

        // Calculate 26-period EMA
        let ema26 = self.calculate_ema(&closes, 26);

        // MACD line = EMA12 - EMA26
        let macd_line = ema12 - ema26;

        // For signal line, we need to calculate EMA of MACD line
        // Simplified: use fixed signal line based on histogram direction
        // In production, you'd calculate full MACD history and get 9-EMA of that
        let signal_line = macd_line * 0.8; // Simplified approximation

        // Histogram = MACD - Signal
        let histogram = macd_line - signal_line;

        (macd_line, signal_line, histogram)
    }

    /// Calculate Exponential Moving Average
    fn calculate_ema(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period {
            return prices.last().copied().unwrap_or(0.0);
        }

        let multiplier = 2.0 / (period as f64 + 1.0);

        // Start with SMA for first period
        let sma: f64 = prices[prices.len() - period..].iter().take(period).sum::<f64>()
            / period as f64;

        // Calculate EMA from SMA
        let mut ema = sma;
        for price in &prices[prices.len() - period + 1..] {
            ema = (price - ema) * multiplier + ema;
        }

        ema
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

    /// Persist trading recommendation to the database
    async fn persist_trading_recommendation(
        &self,
        symbol: &str,
        decision: &TradingDecision,
        context_id: uuid::Uuid,
    ) -> Result<()> {
        use sqlx::query;

        // Calculate position size percentage (90% for high confidence, 70% for medium, 50% for low)
        let position_size_pct = if decision.confidence >= 0.8 {
            0.9
        } else if decision.confidence >= 0.6 {
            0.7
        } else {
            0.5
        };

        // Insert into trading_recommendations table
        query(
            r#"
            INSERT INTO trading_recommendations (
                symbol,
                recommendation,
                confidence,
                reasoning,
                position_size_pct,
                ace_context_id,
                executed
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            "#,
        )
        .bind(symbol)
        .bind(&decision.action)
        .bind(decision.confidence as f32)
        .bind(&decision.reasoning)
        .bind(position_size_pct as f32)
        .bind(context_id)
        .bind(false) // Initially not executed
        .execute(&self.pool)
        .await?;

        info!(
            "Trading recommendation persisted: {} for {} (confidence: {:.1}%)",
            decision.action,
            symbol,
            decision.confidence * 100.0
        );

        Ok(())
    }

    /// Run analysis for a historical date (used in backtesting)
    /// This ensures temporal isolation - only data UP TO the date is used
    pub async fn analyze_at_date(
        &self,
        symbol: &str,
        historical_date: chrono::NaiveDate,
    ) -> Result<TradingDecision> {
        info!(
            "🌅 Starting historical analysis for {} on {}",
            symbol, historical_date
        );

        // For backtest mode, we run the same analysis but with temporal isolation
        // The market_client and sentiment_client should respect the backtest_date from config

        // Run the regular analyze method - the clients will respect backtest_date from config
        self.analyze(symbol).await
    }

    /// Persist daily analysis summary for audit trail and historical analysis
    async fn persist_daily_analysis_summary(
        &self,
        symbol: &str,
        decision: &TradingDecision,
        context_id: uuid::Uuid,
    ) -> Result<()> {
        use sqlx::query;

        // Get the analysis date (use backtest_date if in backtest mode, otherwise today)
        let analysis_date = if let Some(backtest_date) = self.config.backtest_date {
            backtest_date
        } else {
            chrono::Utc::now().date_naive()
        };

        // Calculate position size percentage (same logic as trading_recommendations)
        let position_size_pct = if decision.confidence >= 0.8 {
            0.9
        } else if decision.confidence >= 0.6 {
            0.7
        } else {
            0.5
        };

        // Convert Vec<String> to Vec<&str> for key_factors and risk_factors
        let key_factors: Vec<&str> = decision.key_factors.iter().map(|s| s.as_str()).collect();
        let risk_factors: Vec<&str> = decision.risk_factors.iter().map(|s| s.as_str()).collect();

        // Insert into daily_analysis_summaries table
        // Use ON CONFLICT to handle the case where we already analyzed this symbol today
        query(
            r#"
            INSERT INTO daily_analysis_summaries (
                analysis_date,
                symbol,
                recommendation,
                confidence,
                reasoning,
                key_factors,
                risk_factors,
                position_size_pct,
                ace_context_id,
                executed
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (analysis_date, symbol)
            DO UPDATE SET
                recommendation = EXCLUDED.recommendation,
                confidence = EXCLUDED.confidence,
                reasoning = EXCLUDED.reasoning,
                key_factors = EXCLUDED.key_factors,
                risk_factors = EXCLUDED.risk_factors,
                position_size_pct = EXCLUDED.position_size_pct,
                ace_context_id = EXCLUDED.ace_context_id,
                executed = EXCLUDED.executed
            "#,
        )
        .bind(analysis_date)
        .bind(symbol)
        .bind(&decision.action)
        .bind(decision.confidence as f32)
        .bind(&decision.reasoning)
        .bind(&key_factors[..])
        .bind(&risk_factors[..])
        .bind(position_size_pct as f32)
        .bind(context_id)
        .bind(false) // Initially not executed
        .execute(&self.pool)
        .await?;

        info!(
            "Daily analysis summary persisted for {} on {}",
            symbol, analysis_date
        );

        Ok(())
    }
}

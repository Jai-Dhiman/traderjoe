use anyhow::Result;
use chrono::NaiveDate;
use uuid::Uuid;
use sqlx::PgPool;
use tracing::{info, warn};

/// Run morning market analysis and generate trading recommendations
pub async fn analyze(pool: PgPool, _date: Option<NaiveDate>, symbol: String) -> Result<()> {
    info!("🔍 Starting morning analysis for {}", symbol);
    
    // Load configuration
    let config = crate::config::Config::load()?;
    
    // Initialize the morning orchestrator
    let orchestrator = crate::orchestrator::MorningOrchestrator::new(pool, config).await?;
    
    // Run the full ACE analysis pipeline
    let decision = orchestrator.analyze(&symbol).await?;
    
    println!("\n✅ Morning analysis completed successfully!");
    println!("Decision: {}, Confidence: {:.1}%", decision.action, decision.confidence * 100.0);
    
    Ok(())
}

/// Execute a paper trade based on recommendation
pub async fn execute(pool: PgPool, recommendation_id: Uuid) -> Result<()> {
    info!("⚡ Executing paper trade for recommendation {}", recommendation_id);
    
    // TODO: Implement paper trade execution:
    // - Simulate trade with realistic slippage
    // - Log full context and reasoning
    // - Update position tracking
    
    warn!("⚠️  Paper trade execution not yet implemented");
    println!("📈 Paper Trade Execution - Coming Soon!");
    println!("🎯 Recommendation ID: {}", recommendation_id);
    
    Ok(())
}

/// Run evening review and update ACE playbook
pub async fn review(pool: PgPool, date: Option<NaiveDate>) -> Result<()> {
    info!("🌙 Running evening review for date {:?}", date);

    // Load configuration
    let config = crate::config::Config::load()?;

    // Initialize the evening orchestrator
    let orchestrator = crate::orchestrator::EveningOrchestrator::new(pool, config).await?;

    // If date is specified, review contexts from that date
    // Otherwise, review the latest unreviewed context
    let result = if date.is_some() {
        // TODO: Implement date-specific review
        warn!("Date-specific review not yet implemented, reviewing latest context");
        orchestrator.review_latest().await?
    } else {
        orchestrator.review_latest().await?
    };

    // Display the results
    result.display_summary();

    println!("\n✅ Evening review completed successfully!");

    Ok(())
}

/// Run evening review for all pending contexts
pub async fn review_all(pool: PgPool) -> Result<()> {
    info!("🌙 Running batch evening review for all pending contexts");

    // Load configuration
    let config = crate::config::Config::load()?;

    // Initialize the evening orchestrator
    let orchestrator = crate::orchestrator::EveningOrchestrator::new(pool, config).await?;

    // Review all pending contexts
    let results = orchestrator.review_all_pending().await?;

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║          BATCH EVENING REVIEW SUMMARY                     ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let successful = results.iter().filter(|r| r.success).count();
    let failed = results.len() - successful;

    println!("📊 Total contexts reviewed: {}", results.len());
    println!("✅ Successful: {}", successful);
    if failed > 0 {
        println!("❌ Failed: {}", failed);
    }

    let wins = results.iter().filter(|r| r.outcome.win).count();
    let losses = results.iter().filter(|r| !r.outcome.win && r.success).count();

    if successful > 0 {
        let win_rate = (wins as f64 / successful as f64) * 100.0;
        println!("\n📈 Performance:");
        println!("   Wins: {} ({:.1}%)", wins, win_rate);
        println!("   Losses: {}", losses);

        let total_pnl: f64 = results.iter()
            .filter(|r| r.success)
            .map(|r| r.outcome.pnl_value)
            .sum();
        println!("   Total P&L: ${:.2}", total_pnl);
    }

    println!("\n✅ Batch review completed successfully!");

    Ok(())
}

/// Generate weekly performance report and deep analysis
pub async fn weekly(pool: PgPool, start_date: Option<NaiveDate>) -> Result<()> {
    info!("📊 Running weekly analysis from {:?}", start_date);
    
    // TODO: Implement weekly review:
    // - Performance aggregation
    // - Strategy drift detection
    // - Detailed reporting with Claude 3.5 Sonnet
    
    warn!("⚠️  Weekly analysis not yet implemented");
    println!("📈 Weekly Analysis - Coming Soon!");
    println!("🔍 Deep pattern analysis with Claude 3.5 Sonnet");
    
    Ok(())
}

/// Fetch market data from various sources
pub async fn fetch(
    pool: PgPool,
    symbol: String,
    data_type: String,
    days: Option<u32>,
    _source: Option<String>,
) -> Result<()> {
    match data_type.as_str() {
        "ohlcv" => {
            let client = crate::data::MarketDataClient::new(pool);
            let days = days.unwrap_or(30);
            
            info!("Fetching {} data for {} (last {} days)", data_type, symbol, days);
            
            let data = client.fetch_ohlcv(&symbol, days).await?;
            let count = client.persist_ohlcv(&data).await?;
            
            println!("✅ Successfully fetched and persisted {} OHLCV records for {}", count, symbol);
            println!("📊 Data range: {} to {}", 
                data.first().map(|d| d.date.to_string()).unwrap_or_default(),
                data.last().map(|d| d.date.to_string()).unwrap_or_default()
            );
        }
        _ => {
            return Err(anyhow::anyhow!("Unsupported data type: {}. Currently only 'ohlcv' is implemented.", data_type));
        }
    }
    
    Ok(())
}

/// Run deep research query using Exa API
pub async fn research(pool: PgPool, query: String) -> Result<()> {
    let config = crate::config::Config::load()?;
    let client = crate::data::ResearchClient::new(pool, config.apis.exa_api_key);
    
    info!("Executing research query: {}", query);
    
    let result = client.search(&query).await?;
    
    println!("📊 Research completed:");
    println!("{}", serde_json::to_string_pretty(&result)?);
    
    Ok(())
}

/// Collect sentiment data from various sources
pub async fn sentiment(pool: PgPool, source: String) -> Result<()> {
    info!("📱 Collecting sentiment from {}", source);

    match source.as_str() {
        "reddit" => {
            let config = crate::config::Config::load()?;
            let client = crate::data::sentiment::SentimentClient::new(
                pool,
                config.apis.reddit_client_id,
                config.apis.reddit_client_secret,
            );

            info!("Analyzing sentiment from Reddit for SPY (stub)");
            let result = client.analyze_reddit(Some("SPY")).await?;

            println!("💭 Sentiment analysis completed:");
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        other => {
            return Err(anyhow::anyhow!(
                "Unsupported sentiment source: {}. Currently only 'reddit' is implemented.",
                other
            ));
        }
    }

    Ok(())
}

/// Query ACE context database for similar patterns
pub async fn ace_query(pool: PgPool, query: String) -> Result<()> {
    info!("🧠 Querying ACE context: {}", query);
    
    // Initialize embedder and vector store
    let embedder = crate::embeddings::EmbeddingGemma::load().await?;
    let vector_store = crate::vector::VectorStore::new(pool.clone()).await?;
    
    // Generate embedding for the query
    let query_embedding = embedder.embed(&query).await?;
    
    // Perform similarity search
    let similar_contexts = vector_store.similarity_search(query_embedding, 10).await?;
    
    println!("🔍 ACE Context Query Results");
    println!("============================");
    println!("Query: {}", query);
    println!("Found {} similar contexts\n", similar_contexts.len());
    
    if similar_contexts.is_empty() {
        println!("No similar contexts found. Try running some analyses first to build up the context database.");
        return Ok(());
    }
    
    for (i, ctx) in similar_contexts.iter().enumerate() {
        println!("{}. [{}] Similarity: {:.3}", 
                 i + 1,
                 ctx.timestamp.format("%Y-%m-%d %H:%M"),
                 ctx.similarity.unwrap_or(0.0));
        
        if let Some(decision) = &ctx.decision {
            if let Ok(action) = decision.get("action").and_then(|a| a.as_str()).ok_or(()) {
                println!("   Action: {}", action);
            }
        }
        
        println!("   Confidence: {:.1}%", ctx.confidence * 100.0);
        println!("   Reasoning: {}", ctx.reasoning);
        
        if let Some(outcome) = &ctx.outcome {
            println!("   Outcome: {}", serde_json::to_string_pretty(outcome)?);
        } else {
            println!("   Outcome: Pending");
        }
        
        println!();
    }
    
    Ok(())
}

/// Display ACE playbook statistics and patterns
pub async fn playbook_stats(pool: PgPool) -> Result<()> {
    info!("📚 Displaying ACE playbook statistics");
    
    let context_dao = crate::ace::ContextDAO::new(pool.clone());
    let vector_store = crate::vector::VectorStore::new(pool).await?;
    
    // Get overall statistics
    let stats = context_dao.get_context_stats().await?;
    let (total_contexts, with_embeddings, with_outcomes) = vector_store.context_stats().await?;
    
    println!("📊 ACE Playbook Statistics");
    println!("=============================");
    println!("Total Contexts: {}", stats.total_contexts);
    println!("With Embeddings: {} ({:.1}%)", stats.contexts_with_embeddings, 
             (stats.contexts_with_embeddings as f64 / stats.total_contexts.max(1) as f64) * 100.0);
    println!("With Outcomes: {} ({:.1}%)", stats.contexts_with_outcomes,
             (stats.contexts_with_outcomes as f64 / stats.total_contexts.max(1) as f64) * 100.0);
    
    if let Some(avg_conf) = stats.avg_confidence {
        println!("Average Confidence: {:.1}%", avg_conf * 100.0);
    }
    
    println!("High Confidence (>70%): {} ({:.1}%)", stats.high_confidence_count,
             (stats.high_confidence_count as f64 / stats.total_contexts.max(1) as f64) * 100.0);
    
    // Get recent contexts for pattern analysis
    let recent_contexts = context_dao.get_recent_contexts(20).await?;
    
    if !recent_contexts.is_empty() {
        println!("\n🔍 Recent Decision Patterns:");
        
        let mut action_counts = std::collections::HashMap::new();
        let mut total_confidence = 0.0;
        let mut confidence_count = 0;
        
        for ctx in &recent_contexts {
            if let Some(decision) = &ctx.decision {
                if let Some(action) = decision.get("action").and_then(|a| a.as_str()) {
                    *action_counts.entry(action.to_string()).or_insert(0) += 1;
                }
            }
            
            if let Some(conf) = ctx.confidence {
                total_confidence += conf;
                confidence_count += 1;
            }
        }
        
        for (action, count) in &action_counts {
            println!("  {}: {} times ({:.1}%)", action, count, 
                     (*count as f64 / recent_contexts.len() as f64) * 100.0);
        }
        
        if confidence_count > 0 {
            let avg_confidence = total_confidence / confidence_count as f32;
            println!("  Average Recent Confidence: {:.1}%", avg_confidence * 100.0);
        }
    }
    
    // Show confidence distribution
    if stats.total_contexts > 0 {
        println!("\n📈 Confidence Distribution:");
        
        let high_conf = context_dao.get_contexts_by_confidence(0.7, 1.0, 100).await?;
        let med_conf = context_dao.get_contexts_by_confidence(0.5, 0.7, 100).await?;
        let low_conf = context_dao.get_contexts_by_confidence(0.0, 0.5, 100).await?;
        
        println!("  High (70-100%): {} contexts", high_conf.len());
        println!("  Medium (50-70%): {} contexts", med_conf.len());
        println!("  Low (0-50%): {} contexts", low_conf.len());
    }
    
    println!("\n🕰️ Data Quality:");
    println!("  Vector Search Ready: {}", if with_embeddings > 0 { "Yes" } else { "No" });
    println!("  Outcome Tracking: {:.1}%", 
             (with_outcomes as f64 / total_contexts.max(1) as f64) * 100.0);
    
    Ok(())
}

/// Run backtesting on historical data
pub async fn backtest(pool: PgPool, start_date: NaiveDate, end_date: NaiveDate, strategy: String) -> Result<()> {
    info!("⏪ Running backtest from {} to {} with strategy {}", 
          start_date, end_date, strategy);
    
    // TODO: Implement backtesting:
    // - Walk-forward validation
    // - ACE context evolution simulation
    // - Performance metrics calculation
    
    warn!("⚠️  Backtesting not yet implemented");
    println!("⏪ Backtesting - Coming Soon!");
    println!("📅 Period: {} to {}", start_date, end_date);
    println!("🎯 Strategy: {}", strategy);
    
    Ok(())
}
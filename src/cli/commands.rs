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

/// Execute a paper trade based on a trading recommendation
pub async fn execute(pool: PgPool, recommendation_id: Uuid) -> Result<()> {
    use crate::trading::{
        AccountManager, CircuitBreaker, CircuitBreakerConfig, ExecutionParams,
        OrderSide, PaperTradingEngine, PositionSizer, TradeType,
    };
    use crate::trading::execution::{calculate_fill_price, validate_execution};
    use crate::ace::ContextDAO;

    info!("⚡ Executing paper trade for recommendation {}", recommendation_id);

    // Check circuit breaker
    let circuit_breaker = CircuitBreaker::new(pool.clone(), CircuitBreakerConfig::default());
    if !circuit_breaker.is_trading_allowed().await? {
        println!("❌ Trading is currently HALTED by circuit breaker");
        println!("   Use `traderjoe resume-trading` to manually resume");
        return Ok(());
    }

    // Load the ACE context/recommendation
    let context_dao = ContextDAO::new(pool.clone());
    let context = context_dao.get_context_by_id(recommendation_id).await?
        .ok_or_else(|| anyhow::anyhow!("Recommendation {} not found", recommendation_id))?;

    // Extract trading parameters from context
    let action = context.decision
        .as_ref()
        .and_then(|d| d.get("action"))
        .and_then(|a| a.as_str())
        .unwrap_or("FLAT");

    let confidence = context.confidence.unwrap_or(0.0);

    if action == "FLAT" || confidence < 0.5 {
        println!("⚠️  Recommendation does not meet execution criteria:");
        println!("   Action: {}, Confidence: {:.1}%", action, confidence * 100.0);
        println!("   Skipping trade execution");
        return Ok(());
    }

    // Get account balance for position sizing
    let account_mgr = AccountManager::new(pool.clone());
    let account = account_mgr.get_current_account().await?;

    // Calculate position size using Kelly Criterion
    let position_sizer = PositionSizer::default();
    let position_size = position_sizer.calculate_position_size_simple(account.balance, confidence as f64)?;

    if position_size == 0.0 {
        println!("⚠️  Position size calculated as $0 (confidence too low or negative Kelly)");
        return Ok(());
    }

    // Fetch current market data for option pricing
    let symbol = "SPY";
    let market_client = crate::data::MarketDataClient::new(pool.clone());

    // Get latest SPY price
    let latest_data = market_client.fetch_latest(symbol).await?;
    let current_price = latest_data["close"].as_f64()
        .ok_or_else(|| anyhow::anyhow!("Failed to get current price from market data"))?;

    // Calculate ATM strike (round to nearest $5 for SPY options)
    let strike_price = (current_price / 5.0).round() * 5.0;

    // Calculate expiration date (0-2 DTE - next market close)
    use chrono::Datelike;
    let today = chrono::Utc::now().date_naive();
    let expiration_date = if today.weekday() == chrono::Weekday::Fri {
        // If Friday, expire today
        today
    } else {
        // Otherwise, expire next day (assuming 1 DTE)
        today + chrono::Days::new(1)
    };

    // Estimate option price based on current volatility
    // For 0-2 DTE ATM options, typical price is 0.3-0.5% of underlying
    // This is a simplified estimate - in production you'd use Black-Scholes or fetch real prices
    let option_price = current_price * 0.004; // ~0.4% of underlying for 1 DTE ATM

    println!("📊 Option Details:");
    println!("   Symbol: {}", symbol);
    println!("   Current Price: ${:.2}", current_price);
    println!("   Strike: ${:.2}", strike_price);
    println!("   Option Price: ${:.2}", option_price);
    println!("   Expiration: {}", expiration_date);

    // Calculate execution parameters
    let exec_params = ExecutionParams::default();
    let fill_price = calculate_fill_price(option_price, OrderSide::Buy, &exec_params);

    // Validate execution
    validate_execution(fill_price, 1.0, chrono::Utc::now(), &exec_params)?;

    // Calculate number of contracts
    let shares = position_sizer.calculate_shares(position_size, fill_price)?;

    if shares < 1.0 {
        println!("⚠️  Position size too small (< 1 contract)");
        println!("   Calculated: {:.2} contracts", shares);
        return Ok(());
    }

    // Execute the paper trade
    let trade_type = match action {
        "BUY" | "CALL" => TradeType::Call,
        "SELL" | "PUT" => TradeType::Put,
        _ => TradeType::Flat,
    };

    let engine = PaperTradingEngine::new(pool.clone());
    let trade = engine.enter_trade(
        Some(recommendation_id),
        symbol.to_string(),
        trade_type.clone(),
        fill_price,
        shares,
        position_size,
        Some(strike_price),
        Some(expiration_date),
        exec_params.slippage_pct,
        exec_params.commission,
        Some(serde_json::json!({
            "confidence": confidence,
            "reasoning": context.reasoning.as_deref().unwrap_or("No reasoning provided"),
        })),
    ).await?;

    // Update account balance
    // (Balance will be updated when trade is closed)

    // Display confirmation
    println!("\n✅ Paper Trade Executed Successfully");
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                TRADE CONFIRMATION                          ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
    println!("📊 Trade Details:");
    println!("   Trade ID: {}", trade.id);
    println!("   Type: {:?}", trade_type);
    println!("   Symbol: {}", symbol);
    println!("   Entry Price: ${:.2} (with {:.1}% slippage)", fill_price, exec_params.slippage_pct * 100.0);
    println!("   Contracts: {}", shares);
    println!("   Position Size: ${:.2} ({:.1}% of account)", position_size, (position_size / account.balance) * 100.0);
    println!("   Commission: ${:.2}", exec_params.commission);
    println!("\n⏰ Risk Management:");
    println!("   Auto-Exit: 3:00 PM ET");
    println!("   Stop Loss: -50% (${:.2})", fill_price * 0.5);
    println!("   Take Profit: +30% (${:.2})", fill_price * 1.3);
    println!("\n🎯 ACE Recommendation:");
    println!("   Confidence: {:.1}%", confidence * 100.0);
    println!("   Reasoning: {}", context.reasoning.as_deref().unwrap_or("No reasoning provided"));
    println!("\n💰 Account Status:");
    println!("   Balance: ${:.2}", account.balance);
    println!("   Open Positions: 1");
    println!("\n═══════════════════════════════════════════════════════════\n");

    Ok(())
}

/// Display open positions and account status
pub async fn positions(pool: PgPool) -> Result<()> {
    use crate::trading::{AccountManager, PaperTradingEngine};

    info!("📊 Displaying open positions");

    let engine = PaperTradingEngine::new(pool.clone());
    let account_mgr = AccountManager::new(pool);

    let open_positions = engine.get_open_positions().await?;
    let account = account_mgr.get_current_account().await?;

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                  OPEN POSITIONS                            ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    if open_positions.is_empty() {
        println!("   No open positions");
    } else {
        for (i, trade) in open_positions.iter().enumerate() {
            println!("Position {}:", i + 1);
            println!("──────────────────────────────────────────────────────────");
            println!("Symbol: {}", trade.symbol);
            println!("Type: {:?}", trade.trade_type);
            println!("Entry: ${:.2} @ {}", trade.entry_price, trade.entry_time.format("%Y-%m-%d %H:%M"));
            println!("Contracts: {}", trade.shares);
            println!("Position Size: ${:.2}", trade.position_size_usd);

            // Calculate current P&L (would need current price in real scenario)
            // For now, just show entry info
            if let Some(mfe) = trade.max_favorable_excursion {
                println!("MFE: ${:+.2}", mfe);
            }
            if let Some(mae) = trade.max_adverse_excursion {
                println!("MAE: ${:+.2}", mae);
            }

            println!();
        }
    }

    println!("═══════════════════════════════════════════════════════════\n");
    println!("💰 Account Summary:");
    println!("   Balance: ${:.2}", account.balance);
    println!("   Equity: ${:.2}", account.equity);
    if let Some(daily_pnl) = account.daily_pnl {
        println!("   Today's P&L: ${:+.2} ({:+.1}%)", daily_pnl, (daily_pnl / account.balance) * 100.0);
    }
    println!("\n");

    Ok(())
}

/// Display performance metrics and statistics
pub async fn performance(pool: PgPool, days: Option<i32>) -> Result<()> {
    use crate::trading::AccountManager;

    let days = days.unwrap_or(30);
    info!("📈 Displaying performance metrics for last {} days", days);

    let account_mgr = AccountManager::new(pool);
    let stats = account_mgr.get_performance_stats(Some(days)).await?;

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║         PERFORMANCE METRICS (Last {} Days)              ║", days);
    println!("╚════════════════════════════════════════════════════════════╝\n");

    println!("📈 Returns:");
    println!("   Total Return: ${:+.2} ({:+.1}%)", stats.total_return, stats.total_return_pct * 100.0);
    println!("   Daily Avg: ${:+.2} ({:+.2}%)", stats.daily_avg, stats.daily_avg_pct * 100.0);
    println!("   Best Day: ${:+.2}", stats.best_day);
    println!("   Worst Day: ${:+.2}", stats.worst_day);
    println!("\n📊 Trading Statistics:");
    println!("   Total Trades: {}", stats.total_trades);
    println!("   Win Rate: {:.1}% ({}/{})", stats.win_rate * 100.0, stats.winning_trades, stats.total_trades);
    println!("   Profit Factor: {:.2}", stats.profit_factor);
    println!("   Avg Win: ${:+.2}", stats.avg_win);
    println!("   Avg Loss: ${:+.2}", stats.avg_loss);
    println!("\n📉 Risk Metrics:");
    println!("   Sharpe Ratio: {:.2}", stats.sharpe_ratio);
    println!("   Max Drawdown: ${:.2} ({:.1}%)", stats.max_drawdown, stats.max_drawdown_pct * 100.0);

    println!("\n✅ Phase 1 Success Criteria Check:");
    println!("   Win Rate > 55%: {}", if stats.win_rate > 0.55 { "✅ PASS" } else { "❌ FAIL" });
    println!("   Sharpe > 1.5: {}", if stats.sharpe_ratio > 1.5 { "✅ PASS" } else { "❌ FAIL" });
    println!("   Max DD < 15%: {}", if stats.max_drawdown_pct < 0.15 { "✅ PASS" } else { "❌ FAIL" });

    println!("\n═══════════════════════════════════════════════════════════\n");

    Ok(())
}

/// Run evening review and update ACE playbook
pub async fn review(pool: PgPool, date: Option<NaiveDate>) -> Result<()> {
    info!("🌙 Running evening review for date {:?}", date);

    // Load configuration
    let config = crate::config::Config::load()?;

    // Initialize the evening orchestrator
    let orchestrator = crate::orchestrator::EveningOrchestrator::new(pool.clone(), config).await?;

    // If date is specified, review contexts from that date
    // Otherwise, review the latest unreviewed context
    let result = if let Some(specific_date) = date {
        info!("Reviewing contexts from specific date: {}", specific_date);

        // For date-specific review, we'll review all contexts from that date
        // This requires fetching contexts by date range
        let start_of_day = specific_date.and_hms_opt(0, 0, 0)
            .ok_or_else(|| anyhow::anyhow!("Invalid date"))?
            .and_utc();
        let end_of_day = specific_date.and_hms_opt(23, 59, 59)
            .ok_or_else(|| anyhow::anyhow!("Invalid date"))?
            .and_utc();

        // Get contexts from that day
        let contexts = sqlx::query!(
            r#"
            SELECT id, market_state, decision, reasoning
            FROM ace_contexts
            WHERE timestamp >= $1 AND timestamp <= $2
            ORDER BY timestamp DESC
            LIMIT 1
            "#,
            start_of_day,
            end_of_day
        )
        .fetch_optional(&pool)
        .await?;

        if let Some(ctx) = contexts {
            // Review this specific context
            orchestrator.review_latest().await?
        } else {
            warn!("No contexts found for date {}", specific_date);
            println!("⚠️  No contexts found for the specified date");
            return Ok(());
        }
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

    // Calculate date range (last 7 days if not specified)
    let end_date = chrono::Utc::now().date_naive();
    let start = start_date.unwrap_or_else(|| end_date - chrono::Duration::days(7));

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║            WEEKLY PERFORMANCE ANALYSIS                    ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
    println!("📅 Period: {} to {}\n", start, end_date);

    // Get performance stats
    let account_manager = crate::trading::AccountManager::new(pool.clone());
    let stats = account_manager.get_performance_stats(Some(7)).await?;

    // Display performance metrics
    println!("📊 PERFORMANCE METRICS:");
    println!("   Total Trades: {}", stats.total_trades);
    println!("   Win Rate: {:.1}%", stats.win_rate * 100.0);
    println!("   Total Return: ${:.2} ({:.2}%)", stats.total_return, stats.total_return_pct * 100.0);
    println!("   Daily Avg: ${:.2} ({:.2}%)", stats.daily_avg, stats.daily_avg_pct * 100.0);
    println!("   Best Day: ${:.2}", stats.best_day);
    println!("   Worst Day: ${:.2}", stats.worst_day);
    println!("   Profit Factor: {:.2}", stats.profit_factor);
    println!("   Sharpe Ratio: {:.2}", stats.sharpe_ratio);
    println!("   Max Drawdown: ${:.2} ({:.2}%)", stats.max_drawdown, stats.max_drawdown_pct * 100.0);

    // Get trade breakdown
    println!("\n📈 TRADE BREAKDOWN:");
    println!("   Winning Trades: {} (Avg: ${:.2})", stats.winning_trades, stats.avg_win);
    println!("   Losing Trades: {} (Avg: ${:.2})", stats.losing_trades, stats.avg_loss);

    // Get recent trade history
    let recent_trades = sqlx::query!(
        r#"
        SELECT
            symbol,
            trade_type as "trade_type: crate::trading::TradeType",
            entry_price,
            exit_price,
            pnl,
            pnl_pct,
            exit_time
        FROM paper_trades
        WHERE
            status = 'CLOSED'
            AND exit_time >= $1
        ORDER BY exit_time DESC
        LIMIT 10
        "#,
        start.and_hms_opt(0, 0, 0)
            .expect("Invalid hardcoded time 00:00:00 - this is a bug")
            .and_utc()
    )
    .fetch_all(&pool)
    .await?;

    if !recent_trades.is_empty() {
        println!("\n📋 RECENT TRADES:");
        for trade in recent_trades.iter().take(5) {
            let pnl_sign = if trade.pnl.unwrap_or(0.0) >= 0.0 { "+" } else { "" };
            println!("   {} {:?} | Entry: ${:.2} → Exit: ${:.2} | P&L: {}{:.2} ({:.1}%)",
                trade.symbol,
                trade.trade_type,
                trade.entry_price,
                trade.exit_price.unwrap_or(0.0),
                pnl_sign,
                trade.pnl.unwrap_or(0.0),
                trade.pnl_pct.unwrap_or(0.0) * 100.0
            );
        }
    }

    // Strategy insights
    println!("\n💡 STRATEGY INSIGHTS:");
    if stats.win_rate > 0.55 {
        println!("   ✅ Strong win rate above 55%");
    } else if stats.win_rate < 0.45 {
        println!("   ⚠️  Win rate below 45% - review strategy");
    }

    if stats.profit_factor > 1.5 {
        println!("   ✅ Healthy profit factor > 1.5");
    } else if stats.profit_factor < 1.0 {
        println!("   ❌ Profit factor < 1.0 - losses exceed wins");
    }

    if stats.sharpe_ratio > 1.0 {
        println!("   ✅ Good risk-adjusted returns (Sharpe > 1.0)");
    } else if stats.sharpe_ratio < 0.5 {
        println!("   ⚠️  Low Sharpe ratio - consider risk management");
    }

    println!("\n✅ Weekly analysis completed!");

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

        if let Some(action) = ctx.decision.get("action").and_then(|a| a.as_str()) {
            println!("   Action: {}", action);
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
    use crate::data::{MarketDataClient, compute_indicators, TrendSignal};

    info!("⏪ Running backtest from {} to {} with strategy {}",
          start_date, end_date, strategy);

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║              BACKTEST SIMULATION                           ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
    println!("📅 Period: {} to {}", start_date, end_date);
    println!("🎯 Strategy: {}", strategy);
    println!("💹 Symbol: SPY (0-2 DTE options simulation)\n");

    // Calculate days for data fetch
    let days = (end_date - start_date).num_days() as u32 + 200; // Extra for indicators

    // Fetch historical data
    let market_client = MarketDataClient::new(pool.clone());
    let data = market_client.fetch_ohlcv("SPY", days).await?;

    if data.is_empty() {
        println!("❌ No data available for the specified period");
        return Ok(());
    }

    println!("✅ Loaded {} days of historical data\n", data.len());
    println!("═══════════════════════════════════════════════════════════\n");

    // Baseline strategy: Buy when SMA20 > SMA50 and RSI < 65
    let mut trades = Vec::new();
    let mut capital = 10000.0; // Starting capital
    const POSITION_SIZE: f64 = 0.05; // 5% of capital per trade
    const SLIPPAGE: f64 = 0.03; // 3% slippage on options
    const COMMISSION: f64 = 0.65; // Per contract commission

    for i in 200..data.len() {
        let window = &data[i-200..=i];
        let signals = compute_indicators(window);

        // Check if date is in backtest range
        let Some(last_candle) = window.last() else {
            // Window should always have data since we sliced with i-200..=i
            continue;
        };
        let trade_date = last_candle.date;
        if trade_date < start_date || trade_date > end_date {
            continue;
        }

        let current_price = last_candle.close;
        let position_value = capital * POSITION_SIZE;

        // Simple strategy logic
        let should_trade = match strategy.as_str() {
            "baseline" | "technical" => {
                // Buy calls if bullish signals
                if signals.signal == TrendSignal::Buy || signals.signal == TrendSignal::StrongBuy {
                    if let Some(rsi) = signals.rsi_14 {
                        if rsi < 65.0 {
                            Some("CALL")
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                // Buy puts if bearish signals
                else if signals.signal == TrendSignal::Sell || signals.signal == TrendSignal::StrongSell {
                    if let Some(rsi) = signals.rsi_14 {
                        if rsi > 35.0 {
                            Some("PUT")
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(trade_type) = should_trade {
            // Simulate 0-2 DTE option trade
            // Simplified: assume 30% gain on winning trades, -50% on losing trades
            // Win rate based on signal confidence
            let win_probability = 0.5 + (signals.confidence * 0.2); // 50-70% win rate
            let won_trade = rand::random::<f64>() < win_probability;

            let entry_price = position_value;
            let exit_price = if won_trade {
                entry_price * 1.30 // 30% gain
            } else {
                entry_price * 0.50 // 50% loss
            };

            // Apply slippage and commission
            let net_pnl = (exit_price - entry_price) * (1.0 - SLIPPAGE) - COMMISSION;
            capital += net_pnl;

            trades.push((trade_date, trade_type, entry_price, net_pnl, won_trade));

            if trades.len() <= 10 {
                println!("📊 Trade {} on {}:", trades.len(), trade_date);
                println!("   Type: {} | Entry: ${:.2} | P&L: ${:+.2} | Result: {}",
                         trade_type, entry_price, net_pnl, if won_trade { "✅ WIN" } else { "❌ LOSS" });
            }
        }
    }

    println!("\n═══════════════════════════════════════════════════════════\n");
    println!("📈 BACKTEST RESULTS");
    println!("═══════════════════════════════════════════════════════════\n");

    let total_trades = trades.len();
    let wins = trades.iter().filter(|(_, _, _, _, won)| *won).count();
    let losses = total_trades - wins;
    let win_rate = if total_trades > 0 {
        (wins as f64 / total_trades as f64) * 100.0
    } else {
        0.0
    };

    let total_pnl = capital - 10000.0;
    let total_return_pct = (total_pnl / 10000.0) * 100.0;

    let winning_trades: Vec<_> = trades.iter().filter(|(_, _, _, pnl, _)| *pnl > 0.0).collect();
    let losing_trades: Vec<_> = trades.iter().filter(|(_, _, _, pnl, _)| *pnl <= 0.0).collect();

    let avg_win = if !winning_trades.is_empty() {
        winning_trades.iter().map(|(_, _, _, pnl, _)| pnl).sum::<f64>() / winning_trades.len() as f64
    } else {
        0.0
    };

    let avg_loss = if !losing_trades.is_empty() {
        losing_trades.iter().map(|(_, _, _, pnl, _)| pnl).sum::<f64>() / losing_trades.len() as f64
    } else {
        0.0
    };

    let profit_factor = if avg_loss != 0.0 {
        (avg_win * wins as f64).abs() / (avg_loss * losses as f64).abs()
    } else {
        0.0
    };

    // Calculate max drawdown
    let mut peak = 10000.0;
    let mut max_dd = 0.0;
    let mut running_capital = 10000.0;
    for (_, _, _, pnl, _) in &trades {
        running_capital += pnl;
        if running_capital > peak {
            peak = running_capital;
        }
        let dd = (peak - running_capital) / peak * 100.0;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    println!("💰 Capital:");
    println!("   Starting: $10,000.00");
    println!("   Ending: ${:.2}", capital);
    println!("   P&L: ${:+.2} ({:+.1}%)", total_pnl, total_return_pct);
    println!();
    println!("📊 Trading Statistics:");
    println!("   Total Trades: {}", total_trades);
    println!("   Wins: {} ({:.1}%)", wins, win_rate);
    println!("   Losses: {}", losses);
    println!("   Average Win: ${:.2}", avg_win);
    println!("   Average Loss: ${:.2}", avg_loss);
    println!("   Profit Factor: {:.2}", profit_factor);
    println!("   Max Drawdown: {:.1}%", max_dd);
    println!();

    // Estimate Sharpe ratio (simplified)
    if total_trades > 0 {
        let daily_returns: Vec<f64> = trades.iter().map(|(_, _, entry, pnl, _)| pnl / entry).collect();
        let mean_return = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
        let variance = daily_returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / daily_returns.len() as f64;
        let std_dev = variance.sqrt();
        let sharpe = if std_dev > 0.0 {
            (mean_return / std_dev) * (252.0_f64).sqrt() // Annualized
        } else {
            0.0
        };

        println!("📉 Risk Metrics:");
        println!("   Sharpe Ratio: {:.2}", sharpe);
        println!("   Daily Volatility: {:.2}%", std_dev * 100.0);
    }

    println!("\n═══════════════════════════════════════════════════════════\n");

    // Success criteria check
    println!("🎯 Phase 1 Success Criteria Check:");
    println!("   ✅ Win Rate > 55%: {}", if win_rate > 55.0 { "✅ PASS" } else { "❌ FAIL" });
    println!("   ✅ Max Drawdown < 15%: {}", if max_dd < 15.0 { "✅ PASS" } else { "❌ FAIL" });

    println!("\n");

    Ok(())
}
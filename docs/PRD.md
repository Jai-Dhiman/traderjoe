# Product Requirements Document: ACE-Enhanced Daily Trading System

Version: 2.0
Last Updated: January 11, 2025
Status: Development Phase - Architecture Finalized

Executive Summary
This document defines a single-button automated trading system that combines Extended Thinking (ACE - Augmented Context Evolution) with traditional machine learning to make daily trading decisions. The system learns from outcomes, builds contextual understanding of market patterns, and provides a single trading recommendation each morning.
Core Philosophy: Test rigorously in paper trading before risking capital. Learn what doesn't work as valuable as learning what does.

1. Product Vision
1.1 What We're Building
A system where the user:

Wakes up each morning
Runs a single command or clicks one button
Receives a clear trading recommendation with reasoning
Approves or rejects the trade
System executes and manages the position automatically
End of day: system learns from the outcome

1.2 What Makes This Different
Traditional Algo Trading:

Rules-based or ML models trained on features
No reasoning about why patterns exist
Doesn't accumulate strategic wisdom over time
Black box decisions

This System (ACE-Enhanced):

Combines quantitative signals with contextual reasoning
Maintains an evolving "playbook" of market patterns
Explains decisions in human-understandable terms
Learns meta-strategies: "when to trust which signals"
Adapts to regime changes through accumulated context

1.3 Success Criteria
Phase 1 (90 days paper trading):

✅ System runs daily without manual intervention
✅ Makes falsifiable predictions with confidence scores
✅ Win rate > 55% on directional calls
✅ Sharpe ratio > 1.5
✅ Max drawdown < 15%
✅ ACE playbook contains non-obvious, specific insights

Phase 2 (Real money consideration):

✅ All Phase 1 criteria sustained for 90+ days
✅ Returns > transaction costs + API costs + 10% margin
✅ Survives at least 3 unexpected market events (volatility spikes, surprise news)
✅ User can articulate the edge clearly

2. Market Selection
2.1 Primary Target: SPY/QQQ Options (0-2 DTE) - Day 1 Implementation
Rationale:

Clean daily outcomes (options expire, clear right/wrong)
High leverage on predictions (20-100% returns possible)
Defined risk (can't lose more than premium)
Market hours only (9:30 AM - 4 PM ET = mental closure)
Large volume/tight spreads (easy execution)

Requirements:

$5,000 minimum capital (realistically $25K for unrestricted day trading)
Pattern Day Trader rule consideration
Options approval level (broker-specific)

2.2 Phase 2 Target: BTC/ETH Spot Trading
Rationale:

Lower capital requirements ($500-2000 start)
No day trading restrictions
24/7 markets (flexibility)
High sentiment-driven volatility (ACE advantage)

Trade-off:

Harder to ignore (markets never close)
Longer holding periods = slower learning
Lower leverage without margin

2.3 Why Not Others
Forex: Institutional-dominated, news already priced in microseconds, lower volatility
Futures: High leverage risk, margin complexity, fast execution requirements
Individual Stocks: Too many choices, lower volatility, PDT rule applies

3. System Architecture
3.1 High-Level Components
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                           │
│              (Single Button / CLI Command)                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  ORCHESTRATION LAYER                         │
│        (Morning Routine → Analysis → Decision → Execution)   │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ DATA LAYER   │    │  ACE LAYER   │    │  ML LAYER    │
│              │    │              │    │              │
│ • Market     │    │ • Context DB │    │ • Technical  │
│ • News       │◄───┤ • Vector     │◄───┤   Models     │
│ • Sentiment  │    │   Search     │    │ • Feature    │
│ • Economic   │    │ • Playbook   │    │   Engineering│
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                 ┌──────────────────────┐
                 │  DECISION ENGINE     │
                 │  (ACE + ML Fusion)   │
                 └──────────────────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  EXECUTION LAYER     │
                 │  • Paper Trading     │
                 │  • Live Trading      │
                 │  • Risk Management   │
                 └──────────────────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  LEARNING LAYER      │
                 │  • Outcome Logging   │
                 │  • ACE Update        │
                 │  • Model Retraining  │
                 └──────────────────────┘
3.2 Component Details
3.2.1 Data Layer
Responsibility: Fetch and normalize all input data needed for decisions
Data Sources:
Data TypeSourceUpdate FrequencyCostMarket pricesYahoo Finance APIReal-timeFreeOptions dataCBOE, broker APIIntradayFree-$50/moNews headlinesNewsAPI, RSS feedsContinuousFree tier availableEconomic calendarTrading Economics APIDailyFreeSocial sentimentReddit (PRAW), Twitter APIHourlyFree tierVIX/Fear gaugeYahoo FinanceReal-timeFreeFutures (ES, NQ)Yahoo FinanceReal-timeFreeCrypto pricesCoinGecko, BinanceReal-timeFreeOn-chain data (crypto)Whale Alert, GlassnodeHourlyFree tier
Data Pipeline Requirements:

Fault tolerance (if one source fails, continue with others)
Caching (don't re-fetch same data)
Normalization (standardize timestamps, formats)
Storage (time-series database for historical analysis)

3.2.2 ML Layer
Responsibility: Generate quantitative signals from data
Model Types:

Technical Models:

Moving average crossovers
RSI/MACD momentum indicators
Support/resistance levels
Volume profile analysis
Volatility regime detection (GARCH models)

Sentiment Models:

Local LLM (Llama 3.2 3B via Ollama) for news classification
Social media sentiment aggregation
Entity recognition (Fed speakers, companies mentioned)
Sentiment polarity scoring (-1 to +1)

Predictive Models:

XGBoost/LightGBM for next-day direction
LSTM for time series forecasting
Regime classification (trending/ranging/volatile)
Feature importance tracking

Feature Engineering:

Price-based: returns, volatility, ranges
Volume-based: volume spikes, volume profile
Sentiment-based: news count, polarity, entity mentions
Calendar-based: day of week, FOMC weeks, earnings season
Inter-market: VIX level, bond yields, dollar index
Historical: similar past price patterns

Model Management:

Weekly retraining on rolling window (6-12 months)
Walk-forward validation (train on past, test on future)
Feature drift detection
Performance tracking per model

3.2.3 ACE Layer
Responsibility: Maintain evolving contextual understanding of markets
Core Concept:
ACE is not just a one-time LLM call. It's a persistent knowledge system that:

Accumulates observations over time
Stores them in searchable context database
Retrieves relevant past patterns when analyzing new situations
Updates its "playbook" based on outcomes

ACE Context Database Structure:
Context Entry:
├─ Timestamp: When this pattern occurred
├─ Market State:
│  ├─ Price action (trending/ranging/volatile)
│  ├─ VIX level
│  ├─ Sentiment scores
│  ├─ Economic events that day
│  └─ Inter-market correlations
├─ Decision Made:
│  ├─ What ACE recommended
│  ├─ Reasoning (in natural language)
│  ├─ Confidence score
│  └─ Which models agreed/disagreed
├─ Outcome:
│  ├─ Actual market move
│  ├─ P&L
│  ├─ Max favorable/adverse excursion
│  └─ What happened that was unexpected
└─ Learnings:
   ├─ What went right
   ├─ What went wrong
   └─ Playbook updates triggered
Vector Search:

Embed each context entry into vector space
When analyzing new situation, find K most similar past contexts
Extract: "What worked last time we saw this pattern?"

Playbook Structure:
The playbook is a living document of accumulated wisdom:
Playbook Entry Types:

1. PATTERN INSIGHTS
   "When VIX > 25 and SPY futures are green pre-market,
    calls have worked 8/11 times (73%). Average return: 42%."

2. FAILURE MODES
   "Never trade on FOMC days when ACE confidence < 70%.
    Lost money 6/7 times doing this."

3. REGIME RULES
   "In trending markets (ADX > 25), momentum signals beat
    mean-reversion 4:1."

4. MODEL RELIABILITY
   "Technical models have 68% accuracy in normal volatility
    (VIX 12-20), but drop to 51% when VIX > 30."

5. NEWS IMPACT PATTERNS
   "Fed 'hawkish pivot' language → SPY usually drops 1-2%
    next day, then recovers within week."

6. STRATEGY LIFECYCLE
   "Breakout strategy stopped working after Q2 2024.
    Win rate dropped from 63% to 47%. PRUNED from active set."
ACE Reasoning Process:
Each morning, ACE:

Receives current market state + ML model outputs
Searches vector DB for similar past contexts
Reviews relevant playbook entries
Synthesizes: "Here's what usually happens in this situation"
Generates recommendation with confidence and reasoning
Identifies uncertainty: "I've only seen this pattern 3 times, low confidence"

LLM Integration:
Two modes:

Local LLM (Llama 3.1): Fast, free, for routine analysis
Cloud LLM (GPT-5-nano): Weekly deep reviews, complex reasoning

Prompt Structure for Daily Decision:
You are an expert trading system that learns from experience.

CURRENT MARKET STATE:
{market_data}

ML MODEL SIGNALS:
{technical_signals}
{sentiment_scores}
{predictive_model_outputs}

SIMILAR PAST CONTEXTS (from vector search):
{top_5_similar_situations}

RELEVANT PLAYBOOK ENTRIES:
{applicable_patterns}

TASK:
Analyze this situation and recommend ONE of:

1. BUY CALLS (bullish)
2. BUY PUTS (bearish)
3. STAY FLAT (no edge)

Provide:

- Decision
- Confidence (0-100%)
- Reasoning (2-3 sentences)
- Key risk factors
- Which past pattern this most resembles

Think step by step about:

- What do ML models agree/disagree on?
- What happened in similar past situations?
- What could go wrong with this prediction?
- Is this a high-confidence setup or marginal edge?
3.2.4 Decision Engine
Responsibility: Fuse ACE reasoning with ML signals into actionable trade
Decision Framework:
IF ACE confidence > 70% AND ML models agree (>60% agree):
    → EXECUTE TRADE with standard position size

ELSE IF ACE confidence 50-70% AND ML models mixed:
    → EXECUTE TRADE with reduced position size (50%)

ELSE IF ACE confidence < 50% OR ML models disagree strongly:
    → STAY FLAT (no trade today)

SPECIAL RULES:

- Never trade on high-impact news days unless ACE confidence > 80%
- Never exceed 5% account risk on single trade
- If 3 consecutive losses, reduce position size by 50% until win
- If drawdown > 10%, halt trading and review system
Position Sizing:
Kelly Criterion (modified for safety):
Optimal Position Size = (Win Rate × Avg Win - Loss Rate × Avg Loss) / Avg Win

Apply 1/4 Kelly (for safety):
Actual Position = Optimal × 0.25 × Account Size

Further modified by ACE confidence:
Final Position = Actual Position × (ACE Confidence / 100)
Trade Structure (Options):
For Bullish Signal:

- Buy ATM or slightly OTM calls
- 0-2 DTE (expire today or within 2 days)
- Target: 20-50% gain
- Stop loss: -50% (options can go to zero)
- Exit: 3:00 PM ET (before close)

For Bearish Signal:

- Buy ATM or slightly OTM puts
- Same risk management as calls

For High Uncertainty:

- Buy straddle (both call + put)
- Profits from big move in either direction
- Higher cost, but hedged
3.2.5 Execution Layer
Paper Trading Mode:

Simulates real trades without capital risk
Uses actual market prices (no lookahead bias)
Includes realistic slippage (2-5% on options)
Tracks all costs (commissions, bid-ask spread)
Maintains paper account balance

Live Trading Mode (future):

Integrates with broker API (TD Ameritrade, Interactive Brokers)
Pre-trade risk checks (sufficient capital, margin requirements)
Order types: Market orders at open, limit orders for exits
Position monitoring: Track P&L, set alerts

Risk Management:

Maximum position size: 5% of account per trade
Maximum daily loss: 3% of account
Maximum weekly loss: 10% of account
Circuit breaker: Halt if any limit breached

3.2.6 Learning Layer
Responsibility: Update ACE context and retrain models based on outcomes
End-of-Day Routine:

Outcome Logging:

Fetch actual market close
Calculate P&L
Log max favorable excursion (how much profit we could have had)
Log max adverse excursion (worst drawdown during trade)

ACE Context Update:

Store today's full context + outcome
Generate embeddings for vector search
Update playbook if new pattern emerges

Pattern Extraction:

   IF win rate on specific pattern > 70% after 10+ occurrences:
       → Add to playbook as "HIGH CONFIDENCE PATTERN"

   IF strategy that used to work now has 5 consecutive losses:
       → Flag for review, potentially prune from active strategies

Model Retraining:

Weekly: Retrain predictive models on rolling window
Check for feature drift
Update feature importance

Performance Reporting:

Daily: Simple P&L summary
Weekly: Deep dive with ACE insights
Monthly: Full system audit

4. User Interaction Flow
4.1 Daily Morning Routine (Primary Flow)
User Action: Runs python morning_trade.py or clicks "Analyze Today" button
System Response:
╔════════════════════════════════════════════════════════════╗
║              DAILY TRADING ANALYSIS                        ║
║              October 10, 2025 - 7:15 AM ET                 ║
╚════════════════════════════════════════════════════════════╝

📊 MARKET OVERVIEW:
   SPY: $585.40 (↑0.3% premarket)
   VIX: 14.2 (Low volatility)
   ES Futures: +12 points (Bullish overnight)

📰 KEY NEWS:
   • Fed minutes released yesterday (neutral tone)
   • Tech earnings mostly beat expectations
   • No major economic data today

🤖 ML SIGNALS:
   Technical Models: 4/5 BULLISH
   Sentiment: +0.65 (Moderately bullish)
   Predictive Models: 68% probability SPY closes green

🧠 ACE ANALYSIS:
   This setup resembles 7 past instances where:

- Low VIX + positive futures + tech earnings season
- SPY rallied 8:30-11:00 AM, then consolidated
- Win rate: 6/7 (86%)
- Average return: +34%

🎯 RECOMMENDATION:
   Action: BUY SPY CALLS
   Strike: $587 (slightly OTM)
   Expiry: 0 DTE (today)
   Position Size: $450 (4.5% of account)
   Confidence: 78%

⚠️  RISK FACTORS:

- Market could reverse if tech sector sells off
- Low volume = potential choppy trading
- Exit by 3:00 PM to avoid EOD decay

📝 REASONING:
   ACE believes this is a typical "morning gap-up continuation"
   pattern. Historical data shows strong edge in first 90 minutes.
   ML models agree on bullish bias. Risk/reward favorable.

═══════════════════════════════════════════════════════════

[E] Execute Trade
[S] Skip Today (stay flat)
[D] See Detailed Analysis
[Q] Quit

Your choice:
User presses 'E':
✅ Trade scheduled for 9:30 AM market open
📧 Email confirmation sent
⏰ Auto-exit set for 3:00 PM

You're all set! Check back at 4:00 PM for results.
4.2 Evening Review (Automated)
System automatically runs at 4:30 PM:
╔════════════════════════════════════════════════════════════╗
║              DAILY RESULTS - October 10, 2025              ║
╚════════════════════════════════════════════════════════════╝

📈 TRADE OUTCOME:
   Entry: $587 SPY calls @ $2.30 (9:31 AM)
   Exit: Sold @ $3.15 (3:00 PM)
   Return: +$85 (+37%)

✅ WINNER - ACE was correct

📊 PERFORMANCE UPDATE:
   Today: +$85 (+0.85% account)
   Week: +$220 (+2.2%)
   Month: +$1,140 (+11.4%)
   Win Rate (30d): 18/27 (67%)

🧠 ACE LEARNING:
   ✓ Pattern "morning gap-up continuation" reinforced
   ✓ Confidence in low-VIX bullish setups increased
   ✓ No playbook updates needed (pattern performed as expected)

📧 Full report emailed to you.
4.3 Weekly Deep Review (Manual)
User runs: `traderjoe weekly-review`
System provides:

📊 Win/loss breakdown by ACE pattern type
🤖 ML model performance analysis (which models performed best/worst)
📚 ACE playbook evolution this week (new patterns, updated confidence)
📈 Trade journal with annotated charts
🔍 Deep pattern analysis using Claude 3.5 Sonnet
⚡ System performance metrics and recommendations
🧠 ACE learning effectiveness (context quality, retrieval accuracy)

5. Technical Stack - FINALIZED

5.1 Primary Language: Rust
**Rationale**: Performance, memory safety, lower resource usage, excellent async ecosystem

**Core Dependencies**:

```toml
[dependencies]
tokio = { version = "1.0", features = ["full"] }           # Async runtime
sqlx = { version = "0.7", features = ["postgres", "runtime-tokio-rustls"] }
clap = { version = "4.0", features = ["derive"] }          # CLI framework
reqwest = { version = "0.11", features = ["json"] }       # HTTP client
serde = { version = "1.0", features = ["derive"] }        # Serialization
serde_json = "1.0"                                        # JSON support
ollama-rs = "0.1"                                         # Local LLM client
candle = "0.4"                                           # ML models (EmbeddingGemma)
pyo3 = "0.20"                                            # Python interop
maturin = "1.0"                                          # Python packaging
tracing = "0.1"                                          # Structured logging
tracing-subscriber = "0.3"                               # Log formatting
anyhow = "1.0"                                           # Error handling
thiserror = "1.0"                                        # Custom errors
```

**Module Structure**:

- `src/config/` - Settings and environment management
- `src/data/` - Market data, Exa API, Reddit, NewsAPI integration
- `src/embeddings/` - EmbeddingGemma 300M wrapper
- `src/vector/` - PostgreSQL + pgvector operations
- `src/ace/` - ACE framework implementation (Generator, Reflector, Curator)
- `src/llm/` - Ollama client + cloud fallback
- `src/trading/` - Paper trading engine and risk management
- `src/cli/` - Command-line interface

5.2 Data Sources - EXPANDED

| Data Type | Source | Frequency | Cost | Status |
|-----------|--------|-----------|------|--------|
| Market Prices | Yahoo Finance API | Real-time | Free | ✅ Day 1 |
| Options Data | CBOE, Broker API | Intraday | Free-$50/mo | ✅ Day 1 |
| **Deep Research** | **Exa API** | **On-demand** | **$20-50/mo** | **✅ Day 1** |
| News Headlines | NewsAPI, RSS | Continuous | Free tier | ✅ Day 1 |
| Economic Calendar | Trading Economics | Daily | Free | Phase 2 |
| **Social Sentiment** | **Reddit API** | **Hourly** | **Free** | **✅ Day 1** |
| VIX/Fear Gauge | Yahoo Finance | Real-time | Free | ✅ Day 1 |
| Futures (ES, NQ) | Yahoo Finance | Real-time | Free | Phase 2 |
| Crypto Prices | CoinGecko | Real-time | Free | Phase 3 |

5.3 LLM Strategy - LOCAL FIRST

**Primary**: Llama 3.2 3B via Ollama (Local)

- Cost: $0 operational cost
- Performance: ~200ms inference on Intel Mac
- Use case: Daily analysis, pattern recognition, routine reasoning

**Fallback**: GPT-4o-mini (Cloud)

- Cost: $0.15/1M tokens (~$5-15/month)
- Use case: Complex reasoning when local model fails
- Automatic fallback on local model timeout/error

**Deep Analysis**: Claude 3.5 Sonnet (Cloud, Weekly)

- Cost: $3/1M tokens (~$10/month)
- Use case: Weekly deep reviews, complex pattern analysis

5.4 Vector Database: PostgreSQL + pgvector
**Migration from ChromaDB rationale**:

- ACID compliance for financial data
- Better concurrent access patterns
- Mature backup/replication
- Single database for both time-series and vectors
- HNSW indexing for sub-linear similarity search

**Schema**:

```sql
CREATE TABLE ace_contexts (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    market_state JSONB NOT NULL,
    decision JSONB NOT NULL,
    reasoning TEXT NOT NULL,
    confidence REAL NOT NULL,
    outcome JSONB,
    embedding VECTOR(768) NOT NULL  -- EmbeddingGemma dimensions
);

CREATE INDEX ON ace_contexts USING hnsw (embedding vector_cosine_ops);
```

5.5 Embeddings: Google EmbeddingGemma 300M
**Local model advantages**:

- Best-in-class offline embedding quality
- 300M parameters, runs efficiently on Intel Mac
- No API costs or latency
- Integration via Rust `candle` crate

5.6 Python ML Components (via PyO3)
**Hybrid approach**: Rust for system, Python for ML where ecosystem is superior

```bash
# Python ML environment
uv init ml_components
uv add pandas numpy scikit-learn xgboost lightgbm vectorbt
```

**Components**:

- Feature engineering pipeline
- Model training (XGBoost/LightGBM)
- Backtesting with vectorbt
- Performance metrics

**PyO3 Integration**:

- Async Python function calls from Rust
- Efficient data serialization (serde ↔ pandas)
- Error propagation between languages

5.7 Updated Cost Breakdown

| Component | Service | Monthly Cost | Notes |
|-----------|---------|--------------|-------|
| **Primary LLM** | **Ollama (Local)** | **$0** | **Llama 3.2 3B** |
| Fallback LLM | GPT-4o-mini | $5-15 | Error handling |
| Deep Analysis | Claude 3.5 Sonnet | $10 | Weekly reviews |
| **Research API** | **Exa** | **$20-50** | **Deep market research** |
| Market Data | Yahoo Finance | $0 | OHLCV, VIX |
| News Data | NewsAPI free tier | $0 | Headlines |
| **Social Data** | **Reddit API** | **$0** | **Sentiment** |
| Options Data | Broker API | $0-50 | Depends on broker |
| **Vector DB** | **PostgreSQL + pgvector** | **$0** | **Local** |
| Hosting | Local machine | $0 | Intel Mac |
| **Total** | | **$35-75/mo** | **vs. $200+ cloud-only** |

5.8 Data Storage Requirements

| Data Type | 1 Year Storage | Notes |
|-----------|----------------|-------|
| Market data | ~500MB | OHLCV, indicators |
| News/sentiment | ~300MB | **Expanded with Reddit** |
| **ACE contexts** | **~200MB** | **Vector embeddings** |
| **Model weights** | **~1.2GB** | **EmbeddingGemma + ML** |
| Logs | ~100MB | Structured tracing |
| **Total** | **~2.3GB** | **Still fits easily** |

6. Development Phases - RUST IMPLEMENTATION

Phase 0: Environment Setup (Week 1)
Goal: Configure Rust development environment and core infrastructure
Deliverables:

✅ Rust toolchain installed (cargo, rustc, rustfmt, clippy)
✅ PostgreSQL + pgvector configured locally
✅ Ollama installed with Llama 3.2 3B model
✅ Basic Cargo.toml with core dependencies
✅ Database migrations working (sqlx)
✅ Configuration management (.env, settings)

Success Criteria:

```bash
cargo test        # All tests pass
cargo clippy      # No warnings
psql traderjoe -c "SELECT version();"  # DB accessible
ollama run llama3.2:3b "Test"  # LLM working
```

Phase 1: Data Pipeline (Weeks 2-3)
Goal: Async data fetching and storage in Rust
Deliverables:

📊 Market data pipeline (SPY/QQQ via Yahoo Finance)
📰 News aggregation (NewsAPI + RSS feeds)
🔍 **Exa API integration** for deep research
📱 **Reddit sentiment collection**
💾 Time-series data storage in PostgreSQL
⚡ Async error handling and retry logic

Success Criteria:

```bash
traderjoe fetch --symbol SPY        # Downloads latest data
traderjoe research "market outlook"  # Exa API working
traderjoe sentiment --source reddit  # Social data flowing
```

Phase 2: ML Integration (Week 4)
Goal: Python ↔ Rust ML pipeline via PyO3
Deliverables:

🐍 Python ML environment setup (uv + dependencies)
📈 Technical indicators (RSI, MACD, moving averages)
🤖 XGBoost baseline models for direction prediction
🔗 PyO3 bindings for Rust ↔ Python data transfer
📊 Simple strategy: "technical + sentiment signals"

Success Criteria:

```bash
traderjoe analyze --date 2025-01-10  # Generates ML signals
traderjoe backtest --start 2024-01-01  # Historical testing
```

Phase 3: ACE Framework (Weeks 5-7)
Goal: Implement full ACE system based on research paper
Deliverables:

🧠 **EmbeddingGemma 300M** integration via Candle
📚 **ACE Context Database** (PostgreSQL + pgvector)
🔄 **Generator-Reflector-Curator** architecture
📝 **Incremental Delta Updates** system
🔍 **Vector similarity search** for context retrieval
📈 **Grow-and-Refine** mechanism for playbook evolution
🧑‍💻 **Ollama LLM** integration for reasoning

Success Criteria:

```bash
traderjoe ace-query "Show me similar trading days to today"
# Returns: Found 5 similar contexts with 0.85+ similarity

traderjoe playbook-stats
# Returns: 15+ unique patterns, confidence scores, win rates

traderjoe morning-analysis
# Returns: Recommendation with ACE reasoning + ML signals
```

Phase 4: Paper Trading Validation (Weeks 8-20)
Goal: 90-day live validation with full ACE learning
Deliverables:

📅 **90 days** of paper trading execution
📊 **Daily ACE learning** cycle (morning → trade → evening → update)
📈 **Weekly performance reviews** with pattern analysis
🔧 **System improvements** based on failure analysis
📝 **Documented trading edge** or clear "no-go" decision

Success Criteria:

✅ Hit ALL success criteria from Section 1.3:

- Win rate > 55%
- Sharpe ratio > 1.5
- Max drawdown < 15%
- ACE playbook shows non-obvious insights

🚀 **Stretch goals**:

- Match performance from ACE paper (+10.6% over baseline)
- System runs 85+ days without manual intervention
- Playbook accumulates 50+ unique, validated patterns

Phase 5: Live Trading (If Warranted)
Goal: Transition to real capital with extreme caution
Deliverables:

💹 **Broker API integration** (Alpaca or Interactive Brokers)
⚙️ **Real money risk checks** and circuit breakers
🔐 **Start with $5K**, single SPY option contracts only
🚨 **Strict safeguards**: 3% daily loss limit, 10% weekly limit

Success Criteria:

✅ **First 30 days** live performance matches paper trading
💰 **No catastrophic losses** (max 15% account drawdown ever)
🤖 **System autonomy**: Runs 25+ days without manual override
⚡ **Risk management**: All circuit breakers tested and working

7. Risk Management & Safeguards
7.1 Pre-Trade Risk Checks
Before executing any trade, system verifies:

✅ Sufficient capital in account
✅ Position size within limits (max 5% of account)
✅ No excessive correlation (don't double down on same bet)
✅ Broker API connectivity working
✅ Market is open and liquid
✅ No pending orders that would conflict

7.2 Intra-Trade Monitoring
While trade is active:

Track P&L every 5 minutes
Alert if loss exceeds -30% (consider early exit)
Alert if profit exceeds +50% (consider taking profits)
Monitor volume and spreads (detect liquidity issues)

7.3 Circuit Breakers
Automatic trading halts if:

Daily loss > 3% of account
Weekly loss > 10% of account
5 consecutive losing trades
System error or data feed failure
Unusual market conditions (VIX spike >50%, circuit breakers)

Resume trading only after:

Manual review of what went wrong
System fixes implemented
User explicitly re-enables trading

7.4 Position Limits
Hard limits:

Max 1 active position at a time (initially)
Max 5% of account per trade
Max 3 trades per day
No overnight options positions (too risky with theta decay)

7.5 Data Quality Checks
Before using data for decisions:

Verify timestamps are recent (< 15 minutes old)
Check for missing data (if news feed fails, halt trading)
Validate ranges (if SPY price is $0, something's wrong)
Compare across sources (if APIs disagree significantly, investigate)

8. Evaluation Metrics
8.1 Trading Performance Metrics
Primary:

Total Return: Percentage gain/loss over period
Sharpe Ratio: Risk-adjusted returns (target > 1.5)
Win Rate: Percentage of profitable trades (target > 55%)
Profit Factor: (Sum of wins) / (Sum of losses) (target > 1.8)
Max Drawdown: Largest peak-to-trough decline (target < 15%)

Secondary:

Average win: Mean profit on winning trades
Average loss: Mean loss on losing trades
Expectancy: (Win rate × Avg win) - (Loss rate × Avg loss)
Recovery time: Days to recover from drawdown
Consistency: Standard deviation of daily returns

8.2 ACE-Specific Metrics
Context Quality:

Number of unique patterns in playbook
Average similarity score of retrieved contexts (are we finding truly similar days?)
Playbook staleness (when was each pattern last validated?)

Prediction Accuracy:

ACE confidence vs actual outcome correlation
Confidence calibration: when ACE says 70% confident, is it right 70% of time?
False positive rate (predicted edge, but lost money)
False negative rate (stayed flat, but would have made money)

Learning Effectiveness:

Pattern discovery rate (new insights per month)
Pattern validation rate (how often do patterns hold up over time?)
Adaptation speed (how quickly does ACE adjust to regime changes?)

8.3 System Health Metrics
Operational:

Uptime (percentage of trading days system ran successfully)
Data fetch success rate
API error rate
Execution latency (time from decision to trade placed)

Cost Efficiency:

API costs per trade
Cost per profitable trade
Break-even return rate (minimum return needed to cover costs)

9. Testing & Validation Strategy
9.1 Backtesting (Historical Validation)
Methodology:

Walk-forward analysis: Train on 6 months, test on 1 month, roll forward
No lookahead bias: Only use data available at decision time
Include realistic costs: Slippage (2-5%), commissions ($0.65 per contract)
Multiple time periods: Test across different market regimes

Backtesting Limitations:

Cannot fully simulate ACE learning (ACE evolves, backtest is static)
Options data may be sparse or missing
News sentiment must be reconstructed from archives
Over-optimization risk (curve-fitting to past)

Use backtesting to:

Validate ML models work at all
Test position sizing rules
Estimate potential returns
Identify obvious failure modes

Do NOT use backtesting to:

Guarantee future performance
Justify going straight to real money
Fine-tune every parameter (leads to overfitting)

9.2 Paper Trading (Forward Testing)
This is the PRIMARY validation method
Why paper trading is superior to backtesting:

Tests ACE learning in real-time (evolving playbook)
Uses actual market prices (no historical data issues)
Includes psychological element (seeing real losses, even if paper)
Validates entire system pipeline (data fetching, decision, execution)

Paper trading protocol:

Run for minimum 90 calendar days (60+ trading days)
Treat it like real money (no "oh that was a mistake, redo")
Log every decision with full reasoning
Review weekly: what worked, what didn't
Make system improvements, but track performance separately before/after changes

9.3 Hypothesis-Driven Testing
Instead of "hoping for profit," test specific hypotheses:
Example Hypotheses:
H1: "When VIX < 15 and futures are green premarket, SPY calls have > 60% win rate"

Test: Track this specific setup over 30 days
Measure: Win rate, average return, max drawdown
Result: Accept/reject hypothesis

H2: "ACE context retrieval improves win rate by 10%+ vs baseline ML"

Test: Alternate days (Day 1: use ACE, Day 2: ML only, etc.)
Measure: Difference in win rates
Result: Quantify ACE value

H3: "Sentiment divergence (bearish news + bullish price action) predicts reversals"

Test: Only trade when sentiment and technicals disagree
Measure: Return vs random trades
Result: Is divergence signal real?

9.4 Stress Testing
Test system under adverse conditions:
Scenario 1: Volatility Spike

Simulate VIX jump from 15 to 40
Does ACE recommend staying flat?
Are position sizes reduced appropriately?

Scenario 2: Data Feed Failure

Disable news API, does system halt or continue?
Proper error handling and alerts?

Scenario 3: Losing Streak

Manually inject 10 consecutive losses
Do circuit breakers trigger correctly?
Does ACE update playbook to be more cautious?

Scenario 4: Regime Change

Test on historical regime shifts (e.g., 2020 COVID crash, 2022 rate hikes)
How long does ACE take to adapt?
Does system survive or blow up?

10. Success Scenarios & Failure Modes
10.1 Success Scenario: "It Actually Works"
What this looks like:

After 90 days: 15%+ returns, 65% win rate, Sharpe > 2
ACE playbook has 20+ specific, non-obvious patterns
System survives 2-3 unexpected market events
User understands exactly why it works

Next steps:

Transition to real money with $5K
Run for another
RetryThis response paused because Claude reached its max length for a message. Hit continue to nudge Claude along.Continue

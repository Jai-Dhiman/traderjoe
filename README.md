# TraderJoe: ACE-Enhanced Daily Trading System

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **⚠️ ALPHA SOFTWARE**: This is pre-alpha software currently in active development. Not suitable for production use.

An intelligent daily trading system that combines **Agentic Context Engineering (ACE)** with traditional machine learning to make data-driven trading decisions. Built in Rust for performance and safety.

## 🚀 Quick Start

```bash
# Clone and build
git clone <repo-url>
cd traderjoe
cargo build --release

# View available commands
./target/release/traderjoe --help

# Example usage (once implemented)
./target/release/traderjoe analyze --symbol SPY
./target/release/traderjoe research "market outlook"
./target/release/traderjoe ace-query "similar trading days to today"
```

## 🎯 What Makes This Different

### Traditional Algorithmic Trading

- Rules-based or ML models trained on features
- No reasoning about why patterns exist  
- Doesn't accumulate strategic wisdom over time
- Black box decisions

### TraderJoe (ACE-Enhanced)

- **Combines quantitative signals with contextual reasoning**
- **Maintains an evolving "playbook" of market patterns**
- **Explains decisions in human-understandable terms**
- **Learns meta-strategies: "when to trust which signals"**
- **Adapts to regime changes through accumulated context**

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                   CLI INTERFACE                              │
│            (Manual Daily Trading Workflow)                  │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ DATA LAYER   │    │  ACE LAYER   │    │  ML LAYER    │
│              │    │              │    │              │
│ • SPY/QQQ    │    │ • PostgreSQL │    │ • Technical  │
│ • Exa API    │◄───┤   + pgvector │◄───┤   Models     │
│ • Reddit     │    │ • Context    │    │ • Python/    │
│ • News       │    │   Evolution  │    │   PyO3       │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                 ┌──────────────────────┐
                 │  DECISION ENGINE     │
                 │  (ACE + ML Fusion)   │
                 └──────────────────────┘
```

### Key Technologies

| Component | Technology | Status |
|-----------|------------|--------|
| **Core Language** | Rust 🦀 | ✅ Active |
| **Vector Database** | PostgreSQL + pgvector | 🚧 Planned |
| **Local LLM** | Llama 3.2 3B (Ollama) | 🚧 Planned |
| **Embeddings** | Google EmbeddingGemma 300M | 🚧 Planned |
| **Research API** | Exa API | 🚧 Planned |
| **Sentiment Data** | Reddit API | 🚧 Planned |
| **ML Components** | Python via PyO3 | 🚧 Planned |

## 📊 Data Sources

- **Market Data**: SPY/QQQ OHLCV via Yahoo Finance
- **Deep Research**: Exa API for market intelligence  
- **Social Sentiment**: Reddit API for retail sentiment
- **News**: NewsAPI + RSS feeds for headline analysis
- **Economic**: VIX, futures, economic calendar

## 🧠 ACE Framework

Based on the research paper ["Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models"](docs/ace_framework.md), our ACE implementation treats contexts as **evolving playbooks** rather than static prompts.

### Three Specialized Agents

1. **Generator**: Produces reasoning trajectories for new market situations
2. **Reflector**: Critiques trades to extract concrete lessons  
3. **Curator**: Synthesizes insights into structured context updates

### Key Benefits

- **+10.6%** improvement over baseline systems (from paper)
- **86.9%** reduction in adaptation latency
- **Self-improving** without manual intervention
- **Interpretable** reasoning and decision process

## 🎮 CLI Commands

### Daily Workflow

```bash
# Morning analysis
traderjoe analyze --symbol SPY
traderjoe research "Federal Reserve policy outlook"

# Paper trade execution  
traderjoe execute --recommendation-id <uuid>

# Evening review and learning
traderjoe review
```

### Data & Research

```bash
# Fetch market data
traderjoe fetch --symbol QQQ --data-type ohlcv

# Deep research queries
traderjoe research "inflation impact on tech stocks"

# Sentiment collection
traderjoe sentiment --source reddit
```

### ACE Context Queries

```bash
# Query similar historical patterns
traderjoe ace-query "high VIX with bullish futures"

# View playbook statistics
traderjoe playbook-stats
```

### Analysis & Testing

```bash
# Weekly deep analysis
traderjoe weekly --start-date 2025-01-01

# Backtesting
traderjoe backtest --start-date 2024-01-01 --end-date 2024-12-31
```

## 🚀 Development Phases

### ✅ Phase 0: Foundation (Completed)

- [x] Rust project structure
- [x] CLI framework with clap
- [x] Comprehensive dependency setup
- [x] Module architecture defined

### 🚧 Phase 1: Data Pipeline (Current)

- [ ] Market data fetching (Yahoo Finance)  
- [ ] Exa API integration
- [ ] Reddit sentiment collection
- [ ] PostgreSQL + pgvector setup
- [ ] Async error handling

### 📋 Phase 2: ML Integration  

- [ ] Python ML environment (uv)
- [ ] Technical indicators (RSI, MACD, etc.)
- [ ] XGBoost baseline models
- [ ] PyO3 Rust ↔ Python bindings

### 📋 Phase 3: ACE Framework

- [ ] EmbeddingGemma 300M integration
- [ ] ACE Context Database (pgvector)
- [ ] Generator-Reflector-Curator architecture
- [ ] Incremental delta updates
- [ ] Ollama LLM integration

### 📋 Phase 4: Paper Trading (90 days)

- [ ] Paper trading engine
- [ ] Daily ACE learning cycle
- [ ] Weekly performance reviews
- [ ] Strategy effectiveness measurement

### 📋 Phase 5: Live Trading (If Warranted)

- [ ] Broker API integration
- [ ] Real money risk checks  
- [ ] Circuit breakers and safeguards

## 🔧 Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Database
DATABASE_URL=postgresql://localhost/traderjoe

# API Keys  
EXA_API_KEY=your_key_here
REDDIT_CLIENT_ID=your_client_id
OPENAI_API_KEY=your_key_here

# LLM Configuration
OLLAMA_URL=http://localhost:11434
PRIMARY_MODEL=llama3.2:3b

# Trading Settings
PAPER_TRADING=true
MAX_POSITION_SIZE_PCT=5.0
MAX_DAILY_LOSS_PCT=3.0
```

## 📈 Success Metrics (Phase 1 Goals)

After 90 days of paper trading:

- [ ] **Win rate > 55%** on directional calls
- [ ] **Sharpe ratio > 1.5**
- [ ] **Max drawdown < 15%**
- [ ] **ACE playbook** contains non-obvious, specific insights
- [ ] **System autonomy**: Runs daily without manual intervention

## 🛡️ Risk Management

### Pre-Trade Checks

- ✅ Sufficient capital verification
- ✅ Position size limits (max 5% account)
- ✅ No excessive correlation
- ✅ Market hours and liquidity validation

### Circuit Breakers

- Daily loss > 3% of account → Trading halt
- Weekly loss > 10% of account → Manual review required
- 5 consecutive losses → Reduce position size
- System errors → Automatic trading suspension

## Quick Start Guide

### Prerequisites
- Rust 1.70+ (install via rustup)
- PostgreSQL 14+ with pgvector extension
- Optional: Docker for containerized PostgreSQL

### Database Setup

#### Option 1: Docker (Recommended)
```bash
# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: traderjoe_dev
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
EOF

# Start PostgreSQL
docker-compose up -d

# Set environment variable
export DATABASE_URL="postgresql://postgres:postgres@localhost/traderjoe_dev"
```

#### Option 2: Native PostgreSQL

```bash
# Install PostgreSQL and pgvector
brew install postgresql@14
brew services start postgresql@14

# Install pgvector extension
# (ensure you have Xcode command line tools installed)
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
make install

# Create database
createdb traderjoe_dev

# Enable pgvector
psql traderjoe_dev -c "CREATE EXTENSION vector;"

# Set environment variable
export DATABASE_URL="postgresql://localhost/traderjoe_dev"
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# Required for Phase 1:
# DATABASE_URL=postgresql://postgres:postgres@localhost/traderjoe_dev
# 
# Optional for Phase 1 stubs:
# EXA_API_KEY=your_key_here
# REDDIT_CLIENT_ID=your_id_here
# REDDIT_CLIENT_SECRET=your_secret_here
```

### Build and Run

```bash
# Build the project
cargo build --release

# Run database migrations (via app)
./target/release/traderjoe --help # triggers migrations on startup

# Fetch OHLCV data for SPY
cargo run -- fetch --symbol SPY --data-type ohlcv --days 30

# Enable verbose logging
RUST_LOG=debug cargo run -- fetch --symbol QQQ --data-type ohlcv --verbose

# Run tests
cargo test
```

### Troubleshooting

**Database connection failed:**
- Verify PostgreSQL is running: `pg_isready`
- Check DATABASE_URL format: `postgresql://[user]:[password]@[host]/[database]`
- Ensure database exists: `psql -l`

**pgvector not found:**
- Install pgvector extension (see Database Setup)
- Rerun the app to run migrations

**No data returned:**
- Yahoo Finance may rate-limit; wait and retry
- Check network connectivity
- Verify symbol is valid (e.g., SPY, QQQ, AAPL)

## 📊 Expected Cost Structure

| Component | Monthly Cost | Notes |
|-----------|--------------|-------|
| Llama 3.2 3B (Local) | $0 | Primary LLM |
| GPT-4o-mini (Fallback) | $5-15 | Error handling |
| Claude 3.5 Sonnet | $10 | Weekly deep analysis |
| Exa API | $20-50 | Market research |
| Reddit/News APIs | $0 | Sentiment data |
| **Total** | **$35-75/mo** | vs. $200+ cloud-only |

## 🤝 Contributing

This is currently a personal research project. Once the core system is proven effective, we may open source components.

## ⚠️ Disclaimer

**FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY**

This software is provided for educational and research purposes only. It is not financial advice and should not be used with real money without extensive testing and understanding of the risks involved. Trading involves substantial risk of loss and is not suitable for all investors.

- **Past performance does not guarantee future results**
- **All trading involves risk of loss**
- **Only trade with capital you can afford to lose**
- **This is experimental software in early development**

## 📚 References

- [Agentic Context Engineering Paper](docs/ace_framework.md) - Core ACE methodology
- [Product Requirements Document](PRD.md) - Detailed system specifications
- [PostgreSQL + pgvector](https://github.com/pgvector/pgvector) - Vector database
- [Ollama](https://ollama.ai/) - Local LLM inference
- [Exa API](https://exa.ai/) - AI-powered search and research

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Built with ❤️ and ☕ in Rust** 🦀

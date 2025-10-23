# ACE-Enhanced Daily Trading System

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)

A daily trading system that combines **Agentic Context Engineering (ACE)** with traditional machine learning to make trading decisions.

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                   CLI INTERFACE                             │
│               (Daily Trading Workflow)                      │
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

| Component | Technology |
|-----------|------------|
| **Core Language** | Rust 🦀 |
| **Vector Database** | PostgreSQL + pgvector |
| **Local LLM** | Llama 3.2 3B (Ollama) |
| **Embeddings** | Google EmbeddingGemma 300M |
| **Research API** | Exa API |
| **Sentiment Data** | Reddit API |
| **ML Components** | Python via PyO3 |

## 📊 Data Sources

- **Market Data**: SPY/QQQ OHLCV via Yahoo Finance
- **Deep Research**: Exa API for market intelligence  
- **Social Sentiment**: Reddit API for retail sentiment
- **News**: NewsAPI + RSS feeds for headline analysis
- **Economic**: VIX, futures, economic calendar

## 🧠 ACE Framework

Based on the research paper ["Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models"](docs/ace_framework.md), ther ACE implementation treats contexts as **evolving playbooks** rather than static prompts.

### Three Specialized Agents

1. **Generator**: Produces reasoning trajectories for new market situations
2. **Reflector**: Critiques trades to extract concrete lessons  
3. **Curator**: Synthesizes insights into structured context updates

# Trading Agent Swarm - Hackathon Pitch

## 🎯 The Big Idea

An **autonomous agent swarm** that learns to trade by coordinating multiple specialized agents, each with their own role. The system gets smarter over time by remembering what worked and what didn't, then evolving its strategy.

## 🤖 How It Works

### Three Core Agents (ACE Framework)

1. **Generator Agent** - Creates new trading strategies based on current market conditions
2. **Reflector Agent** - Analyzes past trades to extract lessons ("why did we lose money here?")
3. **Curator Agent** - Synthesizes insights into an evolving "playbook" of trading patterns

These agents coordinate together, sharing a collective memory that improves with every trade.

## 🔌 Technology Stack (Sponsor Integrations)

### **Perplexity.ai** - Market Intelligence
- Real-time research on market conditions, news, earnings, economic events
- Agents query Perplexity to understand *why* markets are moving
- Provides context that pure price data can't capture

### **Hyperspell** - Agent Memory
- Long-term memory across all trading sessions
- Agents remember patterns from weeks/months ago
- Persists the evolving "playbook" of what works and what doesn't

### **Convex.dev** - Real-time Coordination
- Agents share state and coordinate in real-time
- Handles all data sync between Generator, Reflector, and Curator
- Stores market data, trade history, and agent decisions

### **Moss** - Pattern Recognition
- Semantic search over historical trades and market conditions
- "Find me situations similar to today's market"
- Runs fast enough for real-time decision making

### **Browser Use** - Data Collection
- Automates scraping of financial websites, social sentiment, unusual activity
- Goes beyond APIs to gather signals humans look at
- Fills gaps where APIs don't exist (e.g., Twitter sentiment, forum discussions)

## 🎪 Hackathon Tracks

### Primary: **Track 1 - The Summoners (Agents that Coordinate)**
- Three specialized agents working together as a swarm
- Each agent has a distinct role: generate, reflect, curate
- Agents share knowledge through a collective memory system

### Secondary: **Track 4 - The Reapers (Agents that Earn)**
- The entire system exists to make money through trading
- Agents actively buy, sell, and optimize for profit
- Self-improving: learns from losses to earn better over time

## 🎬 Demo Flow

1. **Market Opens** → BrowserUse scrapes sentiment, Perplexity researches news
2. **Generator Agent** → Uses Moss to find similar past situations, proposes trading strategy
3. **Decision Made** → System executes trade through paper trading account
4. **Market Closes** → Reflector analyzes the trade outcome
5. **Curator Updates Playbook** → Hyperspell stores new learnings for future trades
6. **Next Day** → System is smarter, uses yesterday's lessons via Moss semantic search

## 💡 Why This Wins

- **Agent Coordination**: Clear example of multiple agents working together (Track 1)
- **Monetization**: Actually tries to make money autonomously (Track 4)
- **Sponsor Integration**: Uses 5 sponsors naturally, not forced
- **Self-Improving**: Gets better over time - can show before/after performance
- **Real-World**: Trading is tangible - everyone understands win/loss

## 📊 What We'll Show

- Live dashboard showing agents coordinating in real-time
- Playbook evolution over multiple trading sessions
- Agent "thought process" as they analyze and decide
- Performance metrics showing improvement as agents learn
- Side-by-side comparison: "Agent with memory" vs "Agent without memory"

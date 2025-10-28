#!/bin/bash

echo "Testing Ollama Models for Trading Decisions"
echo "==========================================="
echo ""

# Test prompt (simplified version of what backtest uses)
PROMPT='You are a trading AI. Given this market data, make a trading decision.

Market: SPY
Price: $550.50
Trend: Bullish
RSI: 55 (neutral)
MACD: Positive signal

Respond ONLY with valid JSON in this format:
{
  "action": "BUY_CALLS",
  "confidence": 0.75,
  "reasoning": "Brief explanation"
}

Your JSON response:'

echo "=== Testing llama3.2:3b (Fast) ==="
echo "Starting at: $(date)"
time curl -X POST http://localhost:11434/api/generate \
  -d "{
    \"model\": \"llama3.2:3b\",
    \"prompt\": $(echo "$PROMPT" | jq -Rs .),
    \"stream\": false
  }" 2>/dev/null | jq -r '.response'

echo ""
echo ""
echo "=== Testing llama4:scout (Powerful) ==="
echo "Starting at: $(date)"
time curl -X POST http://localhost:11434/api/generate \
  -d "{
    \"model\": \"llama4:scout\",
    \"prompt\": $(echo "$PROMPT" | jq -Rs .),
    \"stream\": false
  }" 2>/dev/null | jq -r '.response'

echo ""
echo "Done at: $(date)"

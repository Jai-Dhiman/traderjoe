#!/bin/bash
# TraderJoe Morning Routine
# Run this script every trading day before market open (before 9:30 AM ET)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WRAPPER="$SCRIPT_DIR/traderjoe"

# Load environment variables from .env file
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Start Ollama once for the entire routine
OLLAMA_STARTED=false
OLLAMA_PID=""

start_ollama_once() {
    # Check if Ollama is already running
    if pgrep -x "ollama" >/dev/null; then
        echo -e "${GREEN}✓ Ollama is already running${NC}"
        OLLAMA_STARTED=false
        return 0
    fi

    echo -e "${YELLOW}Starting Ollama for morning routine...${NC}"
    if [[ -x "/opt/homebrew/bin/ollama" ]]; then
        /opt/homebrew/bin/ollama serve >/tmp/ollama-morning-routine.log 2>&1 &
        OLLAMA_PID=$!
        OLLAMA_STARTED=true

        # Wait for Ollama to be ready
        for i in {1..30}; do
            if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
                echo -e "${GREEN}✓ Ollama is ready (PID: $OLLAMA_PID)${NC}"
                return 0
            fi
            sleep 1
        done

        echo -e "${RED}✗ Ollama failed to start${NC}"
        exit 1
    else
        echo -e "${RED}✗ Ollama not found. Please install: brew install ollama${NC}"
        exit 1
    fi
}

stop_ollama_if_started() {
    if [[ "$OLLAMA_STARTED" == "true" && -n "$OLLAMA_PID" ]]; then
        echo -e "${YELLOW}Stopping Ollama (PID $OLLAMA_PID)...${NC}"
        kill "$OLLAMA_PID" 2>/dev/null || true

        # Wait for graceful shutdown
        for i in {1..5}; do
            if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
                echo -e "${GREEN}✓ Ollama stopped${NC}"
                return 0
            fi
            sleep 1
        done

        # Force kill if still running
        kill -9 "$OLLAMA_PID" 2>/dev/null || true
        echo -e "${GREEN}✓ Ollama stopped${NC}"
    fi
}

# Cleanup trap
cleanup() {
    echo ""
    stop_ollama_if_started
}
trap cleanup EXIT INT TERM

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}      TraderJoe Morning Routine - $(date '+%Y-%m-%d')       ${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

# Start Ollama once for all commands
start_ollama_once

# Export flag so wrapper script skips Ollama management
export SKIP_OLLAMA_MANAGEMENT=1
echo ""

# 1. System Health Check
echo -e "${YELLOW}[1/5] Checking System Health...${NC}"
if ! $WRAPPER migrate 2>/dev/null; then
    echo -e "${RED}✗ Database migration check failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Database is healthy${NC}\n"

# 2. Fetch Latest Market Data
echo -e "${YELLOW}[2/5] Fetching Latest Market Data...${NC}"
if ! $WRAPPER fetch --symbol SPY --data-type ohlcv --days 5; then
    echo -e "${RED}✗ Market data fetch failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Market data updated${NC}\n"

# 3. Run Morning Analysis with ACE
echo -e "${YELLOW}[3/5] Running Morning Analysis (ACE Pipeline)...${NC}"
ANALYSIS_OUTPUT=$($WRAPPER analyze --symbol SPY 2>&1)
echo "$ANALYSIS_OUTPUT"
echo -e "${GREEN}✓ Analysis complete${NC}\n"

# 3b. Auto-Execute Trade if Confidence Meets Threshold
echo -e "${YELLOW}[3b/5] Checking Auto-Execution Criteria...${NC}"

# Confidence threshold for auto-execution (adjust as needed)
# Paper trading phase: Lower threshold (0.50) to generate more learning data
# Live trading phase: Raise threshold (0.70+) for higher quality setups
CONFIDENCE_THRESHOLD=0.50

# Extract recommendation ID (UUID format)
RECOMMENDATION_ID=$(echo "$ANALYSIS_OUTPUT" | grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | head -1)

if [[ -n "$RECOMMENDATION_ID" ]]; then
    echo -e "${BLUE}Found recommendation ID: $RECOMMENDATION_ID${NC}"

    # Extract confidence from analysis output (looks for patterns like "Confidence: 0.75" or "confidence: 75%")
    CONFIDENCE=$(echo "$ANALYSIS_OUTPUT" | grep -i "confidence" | grep -oE '[0-9]+(\.[0-9]+)?' | head -1)

    # Convert percentage to decimal if needed (e.g., 75 -> 0.75)
    if [[ -n "$CONFIDENCE" ]]; then
        if (( $(echo "$CONFIDENCE > 1" | bc -l) )); then
            CONFIDENCE=$(echo "scale=2; $CONFIDENCE / 100" | bc)
        fi

        echo -e "${BLUE}Confidence: $CONFIDENCE (threshold: $CONFIDENCE_THRESHOLD)${NC}"

        # Execute if confidence meets threshold
        if (( $(echo "$CONFIDENCE >= $CONFIDENCE_THRESHOLD" | bc -l) )); then
            echo -e "${GREEN}✓ Confidence meets threshold - AUTO-EXECUTING TRADE${NC}"
            echo -e "${YELLOW}  (Paper trading mode - no real money at risk)${NC}"
            if $WRAPPER execute --recommendation-id "$RECOMMENDATION_ID"; then
                echo -e "${GREEN}✓ Trade executed successfully${NC}\n"
            else
                echo -e "${RED}✗ Trade execution failed${NC}\n"
            fi
        else
            echo -e "${YELLOW}⊘ Confidence below threshold - SKIPPING execution${NC}\n"
        fi
    else
        echo -e "${YELLOW}⊘ Could not parse confidence - SKIPPING execution${NC}\n"
    fi
else
    echo -e "${YELLOW}⊘ No recommendation ID found - SKIPPING execution${NC}\n"
fi

# 4. Display Current Account Status
echo -e "${YELLOW}[4/5] Checking Account & Positions...${NC}"
$WRAPPER positions
echo ""

# 5. Display ACE Playbook Stats
echo -e "${YELLOW}[5/5] ACE Playbook Statistics...${NC}"
$WRAPPER playbook-stats
echo ""

# Final Summary
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                 MORNING ROUTINE COMPLETE                   ${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

echo -e "${GREEN}Auto-Execution Status:${NC}"
echo -e "  • Trades are AUTOMATICALLY executed when confidence ≥ 50%"
echo -e "  • Lower threshold for LEARNING PHASE (paper trading only)"
echo -e "  • All executions are PAPER TRADES (no real money at risk)\n"

echo -e "${YELLOW}Active Risk Controls:${NC}"
echo -e "  • Max position size: 5% of account"
echo -e "  • Stop loss: -50% from entry"
echo -e "  • Take profit: +30% from entry"
echo -e "  • Time-based exit: 3:00 PM ET (via auto_exit.sh)"
echo -e "  • Confidence threshold: ≥50% (learning phase)\n"

echo -e "${BLUE}Note: Increase threshold to 70%+ before live trading${NC}"

echo -e "${BLUE}Next Auto-Run: 12:00 PM PST (auto_exit.sh checks positions)${NC}\n"

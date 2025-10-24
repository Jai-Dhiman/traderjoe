#!/bin/bash
# TraderJoe Evening Routine
# Run this script every trading day after market close (after 4:00 PM ET)

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

    echo -e "${YELLOW}Starting Ollama for evening routine...${NC}"
    if [[ -x "/opt/homebrew/bin/ollama" ]]; then
        /opt/homebrew/bin/ollama serve >/tmp/ollama-evening-routine.log 2>&1 &
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
echo -e "${BLUE}      TraderJoe Evening Routine - $(date '+%Y-%m-%d')      ${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

# Start Ollama once for all commands
start_ollama_once

# Export flag so wrapper script skips Ollama management
export SKIP_OLLAMA_MANAGEMENT=1
echo ""

# 1. Display Today's Performance
echo -e "${YELLOW}[1/4] Today's Trading Performance...${NC}"
$WRAPPER positions
echo ""

# 2. Run Evening Review (ACE Learning Cycle)
echo -e "${YELLOW}[2/4] Running Evening Review (ACE Reflection)...${NC}"
echo -e "${BLUE}This will update the playbook based on today's outcomes${NC}\n"
$WRAPPER review
echo ""

# 3. Display Updated Performance Metrics
echo -e "${YELLOW}[3/4] Performance Metrics (Last 30 Days)...${NC}"
$WRAPPER performance --days 30
echo ""

# 4. Check Playbook Evolution
echo -e "${YELLOW}[4/4] Checking Playbook Evolution...${NC}"
$WRAPPER playbook-stats
echo ""

# Final Summary
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                 EVENING ROUTINE COMPLETE                   ${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

echo -e "${GREEN}Daily Review Complete!${NC}\n"

echo -e "${YELLOW}Key Metrics to Monitor:${NC}"
echo -e "  • Win Rate should stay > 55%"
echo -e "  • Sharpe Ratio should be > 1.5"
echo -e "  • Max Drawdown should be < 15%"
echo -e "  • ACE confidence should correlate with outcomes\n"

echo -e "${BLUE}See you tomorrow!${NC}\n"

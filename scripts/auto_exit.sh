#!/bin/bash
# TraderJoe Auto-Exit Routine
# Run this script at 3:00 PM ET to automatically exit positions

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

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}     TraderJoe Auto-Exit Routine - $(date '+%Y-%m-%d %H:%M')     ${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

# 1. Check Current Positions
echo -e "${YELLOW}[1/3] Checking Open Positions...${NC}"
$WRAPPER positions
echo ""

# 2. Run Auto-Exit Checks
echo -e "${YELLOW}[2/3] Running Auto-Exit Checks...${NC}"
echo -e "${BLUE}Checking stop-loss, take-profit, and time-based exits${NC}\n"
$WRAPPER auto-exit
echo ""

# 3. Display Updated Positions
echo -e "${YELLOW}[3/3] Updated Positions Status...${NC}"
$WRAPPER positions
echo ""

# Final Summary
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}               AUTO-EXIT ROUTINE COMPLETE                   ${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

echo -e "${GREEN}Auto-Exit Check Complete!${NC}\n"

echo -e "${YELLOW}Exit Criteria:${NC}"
echo -e "  • Time-based: 3:00 PM ET"
echo -e "  • Stop-loss: -50% from entry"
echo -e "  • Take-profit: +30% from entry"
echo -e "  • All criteria checked automatically\n"

echo -e "${BLUE}Positions will be closed if exit conditions are met${NC}\n"

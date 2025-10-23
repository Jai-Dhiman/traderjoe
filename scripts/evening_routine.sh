#!/bin/bash
# TraderJoe Evening Routine
# Run this script every trading day after market close (after 4:00 PM ET)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TRADERJOE="$PROJECT_ROOT/target/release/traderjoe"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}      TraderJoe Evening Routine - $(date '+%Y-%m-%d')      ${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

# 1. Display Today's Performance
echo -e "${YELLOW}[1/4] Today's Trading Performance...${NC}"
$TRADERJOE positions
echo ""

# 2. Run Evening Review (ACE Learning Cycle)
echo -e "${YELLOW}[2/4] Running Evening Review (ACE Reflection)...${NC}"
echo -e "${BLUE}This will update the playbook based on today's outcomes${NC}\n"
$TRADERJOE review
echo ""

# 3. Display Updated Performance Metrics
echo -e "${YELLOW}[3/4] Performance Metrics (Last 30 Days)...${NC}"
$TRADERJOE performance --days 30
echo ""

# 4. Check Playbook Evolution
echo -e "${YELLOW}[4/4] Checking Playbook Evolution...${NC}"
$TRADERJOE playbook-stats
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

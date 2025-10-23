#!/bin/bash
# TraderJoe Morning Routine
# Run this script every trading day before market open (before 9:30 AM ET)

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
echo -e "${BLUE}      TraderJoe Morning Routine - $(date '+%Y-%m-%d')       ${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

# 1. System Health Check
echo -e "${YELLOW}[1/5] Checking System Health...${NC}"
if ! $TRADERJOE migrate 2>/dev/null; then
    echo -e "${RED}✗ Database migration check failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Database is healthy${NC}\n"

# 2. Fetch Latest Market Data
echo -e "${YELLOW}[2/5] Fetching Latest Market Data...${NC}"
if ! $TRADERJOE fetch --symbol SPY --data-type ohlcv --days 5; then
    echo -e "${RED}✗ Market data fetch failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Market data updated${NC}\n"

# 3. Run Morning Analysis with ACE
echo -e "${YELLOW}[3/5] Running Morning Analysis (ACE Pipeline)...${NC}"
$TRADERJOE analyze --symbol SPY
echo -e "${GREEN}✓ Analysis complete${NC}\n"

# 4. Display Current Account Status
echo -e "${YELLOW}[4/5] Checking Account & Positions...${NC}"
$TRADERJOE positions
echo ""

# 5. Display ACE Playbook Stats
echo -e "${YELLOW}[5/5] ACE Playbook Statistics...${NC}"
$TRADERJOE playbook-stats
echo ""

# Final Instructions
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                 MORNING ROUTINE COMPLETE                   ${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

echo -e "${GREEN}Next Steps:${NC}"
echo -e "  1. Review the ACE recommendation above"
echo -e "  2. If confidence > 70%, consider executing the trade:"
echo -e "     ${YELLOW}$TRADERJOE execute --recommendation-id <UUID>${NC}"
echo -e "  3. Or skip and wait for better setups\n"

echo -e "${YELLOW}Risk Reminders:${NC}"
echo -e "  • Max position size: 5% of account"
echo -e "  • Stop loss: -50%"
echo -e "  • Take profit: +30%"
echo -e "  • Auto-exit: 3:00 PM ET\n"

#!/usr/bin/env bash
# Test script to validate confidence learning mechanism
# Simulates wins and losses to ensure playbook bullets update correctly

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Database connection
DATABASE_URL="${DATABASE_URL:-postgresql://localhost/traderjoe}"
PSQL="/opt/homebrew/opt/postgresql@16/bin/psql"

echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}     CONFIDENCE LEARNING TEST HARNESS                       ${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo ""

# Function to create a test ACE context with outcome
create_test_context() {
    local action=$1
    local win=$2
    local pnl_pct=$3
    local entry_price=$4
    local exit_price=$5

    echo -e "${BLUE}Creating test context: ${action}, win=${win}, P&L=${pnl_pct}%${NC}"

    $PSQL "$DATABASE_URL" -c "
    INSERT INTO ace_contexts (
        id,
        timestamp,
        market_state,
        decision,
        reasoning,
        confidence,
        outcome,
        embedding
    ) VALUES (
        gen_random_uuid(),
        NOW(),
        '{\"symbol\": \"SPY\", \"market_data\": {\"latest_price\": ${entry_price}}}',
        '{\"action\": \"${action}\", \"confidence\": 0.6, \"reasoning\": \"Test decision for confidence learning\"}',
        'Test decision for confidence learning',
        0.6,
        '{\"win\": ${win}, \"pnl_pct\": ${pnl_pct}, \"pnl_value\": 0, \"entry_price\": ${entry_price}, \"exit_price\": ${exit_price}, \"duration_hours\": 6.5, \"notes\": \"Test outcome\"}',
        NULL
    )
    RETURNING id;
    " -t
}

# Test 1: STAY_FLAT decision (0% P&L) - Tests the fix for outcome_severity
echo -e "\n${YELLOW}${BOLD}[Test 1/4] STAY_FLAT Decision (0% P&L)${NC}"
echo "Expected: Should use 0.05 severity multiplier (not 0.01)"
echo "─────────────────────────────────────────────────────────"
context_id=$(create_test_context "STAY_FLAT" "false" "0.0" "100.0" "100.0")
echo -e "Context created: ${context_id}"
echo ""

# Test 2: BUY_CALLS win with moderate P&L (10%)
echo -e "${YELLOW}${BOLD}[Test 2/4] BUY_CALLS Win (10% gain)${NC}"
echo "Expected: Should use 0.05 severity multiplier (5-20% range)"
echo "─────────────────────────────────────────────────────────"
context_id=$(create_test_context "BUY_CALLS" "true" "10.0" "100.0" "110.0")
echo -e "Context created: ${context_id}"
echo ""

# Test 3: BUY_PUTS loss with large move (25%)
echo -e "${YELLOW}${BOLD}[Test 3/4] BUY_PUTS Loss (25% loss)${NC}"
echo "Expected: Should use 0.10 severity multiplier (20-50% range)"
echo "─────────────────────────────────────────────────────────"
context_id=$(create_test_context "BUY_PUTS" "false" "-25.0" "100.0" "75.0")
echo -e "Context created: ${context_id}"
echo ""

# Test 4: BUY_CALLS huge win (60%)
echo -e "${YELLOW}${BOLD}[Test 4/4] BUY_CALLS Huge Win (60% gain)${NC}"
echo "Expected: Should use 0.15 severity multiplier (50%+ range)"
echo "─────────────────────────────────────────────────────────"
context_id=$(create_test_context "BUY_CALLS" "true" "60.0" "100.0" "160.0")
echo -e "Context created: ${context_id}"
echo ""

echo -e "${GREEN}${BOLD}✅ Test contexts created successfully!${NC}"
echo ""
echo -e "${BOLD}Next Steps:${NC}"
echo "1. Check initial playbook state:"
echo -e "   ${BLUE}./scripts/monitor_playbook.sh${NC}"
echo ""
echo "2. Run evening review to process these outcomes:"
echo -e "   ${BLUE}./scripts/evening_routine.sh${NC}"
echo ""
echo "3. Check playbook updates (should see confidence changes):"
echo -e "   ${BLUE}./scripts/monitor_playbook.sh${NC}"
echo ""
echo "4. Verify detailed logging in evening review output:"
echo -e "   ${BLUE}Look for '📊 PLAYBOOK UPDATE' messages with before→after confidence${NC}"
echo ""
echo -e "${YELLOW}Expected Behavior:${NC}"
echo "  • STAY_FLAT (0% P&L): Confidence should change by ±0.005 per vote (0.05 severity)"
echo "  • Moderate win (10%): Confidence should change by ±0.005 per vote"
echo "  • Large loss (25%):   Confidence should change by ±0.010 per vote"
echo "  • Huge win (60%):     Confidence should change by ±0.015 per vote"
echo ""
echo -e "${RED}${BOLD}Important:${NC} These are TEST data. To clean up after testing:"
echo -e "   ${BLUE}psql \$DATABASE_URL -c \"DELETE FROM ace_contexts WHERE reasoning LIKE '%Test decision%';\"${NC}"
echo ""

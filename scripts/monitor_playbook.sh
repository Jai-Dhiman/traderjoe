#!/usr/bin/env bash
# Playbook Evolution Monitoring Script
# Shows key metrics for ACE playbook learning progress

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Database connection
DATABASE_URL="${DATABASE_URL:-postgresql://localhost/traderjoe}"
PSQL="/opt/homebrew/opt/postgresql@16/bin/psql"

echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}        ACE PLAYBOOK EVOLUTION MONITOR                      ${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo ""

# 1. Overall Statistics
echo -e "${CYAN}${BOLD}[1/5] Overall Playbook Statistics${NC}"
echo "─────────────────────────────────────────────────────────"
$PSQL "$DATABASE_URL" -c "
SELECT
    COUNT(*) as total_bullets,
    COUNT(CASE WHEN confidence > 0.6 THEN 1 END) as high_confidence,
    COUNT(CASE WHEN confidence BETWEEN 0.4 AND 0.6 THEN 1 END) as medium_confidence,
    COUNT(CASE WHEN confidence < 0.4 THEN 1 END) as low_confidence,
    ROUND(AVG(confidence)::numeric, 3) as avg_confidence,
    ROUND(MIN(confidence)::numeric, 3) as min_confidence,
    ROUND(MAX(confidence)::numeric, 3) as max_confidence
FROM playbook_bullets;
"
echo ""

# 2. Bullets by Section
echo -e "${CYAN}${BOLD}[2/5] Bullets by Section${NC}"
echo "─────────────────────────────────────────────────────────"
$PSQL "$DATABASE_URL" -c "
SELECT
    section,
    COUNT(*) as count,
    COUNT(CASE WHEN confidence > 0.6 THEN 1 END) as high_conf,
    ROUND(AVG(confidence)::numeric, 3) as avg_conf,
    SUM(helpful_count) as total_helpful,
    SUM(harmful_count) as total_harmful
FROM playbook_bullets
GROUP BY section
ORDER BY count DESC;
"
echo ""

# 3. Top Performing Bullets (Highest Confidence)
echo -e "${GREEN}${BOLD}[3/5] Top 5 High-Confidence Bullets${NC}"
echo "─────────────────────────────────────────────────────────"
$PSQL "$DATABASE_URL" -c "
SELECT
    section,
    SUBSTRING(content, 1, 70) as content_preview,
    ROUND(confidence::numeric, 3) as conf,
    helpful_count as helpful,
    harmful_count as harmful,
    CASE
        WHEN last_used IS NOT NULL THEN AGE(NOW(), last_used)::text
        ELSE 'Never used'
    END as last_used_age
FROM playbook_bullets
ORDER BY confidence DESC
LIMIT 5;
"
echo ""

# 4. Bullets Needing Attention (Low Confidence or Harmful)
echo -e "${YELLOW}${BOLD}[4/5] Bottom 5 Bullets (Need Review)${NC}"
echo "─────────────────────────────────────────────────────────"
$PSQL "$DATABASE_URL" -c "
SELECT
    section,
    SUBSTRING(content, 1, 70) as content_preview,
    ROUND(confidence::numeric, 3) as conf,
    helpful_count as helpful,
    harmful_count as harmful,
    CASE
        WHEN harmful_count > helpful_count THEN '⚠️  More harmful'
        WHEN helpful_count = 0 AND harmful_count = 0 THEN 'Untested'
        ELSE 'OK'
    END as status
FROM playbook_bullets
ORDER BY confidence ASC
LIMIT 5;
"
echo ""

# 5. Learning Progress (Contexts with Outcomes)
echo -e "${CYAN}${BOLD}[5/5] ACE Learning Progress${NC}"
echo "─────────────────────────────────────────────────────────"
$PSQL "$DATABASE_URL" -c "
SELECT
    COUNT(*) as total_contexts,
    COUNT(CASE WHEN outcome IS NOT NULL THEN 1 END) as with_outcomes,
    COUNT(CASE WHEN outcome IS NULL THEN 1 END) as pending_review,
    ROUND(AVG(confidence)::numeric, 3) as avg_decision_confidence,
    COUNT(CASE WHEN (decision->>'action') = 'STAY_FLAT' THEN 1 END) as stay_flat_count,
    COUNT(CASE WHEN (decision->>'action') = 'BUY_CALLS' THEN 1 END) as buy_calls_count,
    COUNT(CASE WHEN (decision->>'action') = 'BUY_PUTS' THEN 1 END) as buy_puts_count
FROM ace_contexts;
"
echo ""

# 6. Recent Confidence Changes (if we have outcomes)
echo -e "${BLUE}${BOLD}Recent Outcome Win/Loss Summary${NC}"
echo "─────────────────────────────────────────────────────────"
$PSQL "$DATABASE_URL" -c "
SELECT
    COUNT(*) as total_reviewed,
    COUNT(CASE WHEN (outcome->>'win')::boolean = true THEN 1 END) as wins,
    COUNT(CASE WHEN (outcome->>'win')::boolean = false THEN 1 END) as losses,
    ROUND(AVG((outcome->>'pnl_pct')::numeric), 2) as avg_pnl_pct
FROM ace_contexts
WHERE outcome IS NOT NULL;
"
echo ""

echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}Monitor Complete!${NC}"
echo ""
echo "Next Steps:"
echo "  • Bullets with confidence > 0.6 will be used in decision-making"
echo "  • Run evening review to update playbook based on outcomes"
echo "  • Monitor harmful_count - may indicate poor bullets to remove"
echo ""
echo "Commands:"
echo "  • Watch in real-time: watch -n 60 ./scripts/monitor_playbook.sh"
echo "  • Check specific section: psql \$DATABASE_URL -c \"SELECT * FROM playbook_bullets WHERE section = 'pattern_insights';\""
echo ""

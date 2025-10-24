#!/usr/bin/env bash
# Daily summary email (requires macOS mail or sendmail)

set -euo pipefail

DATABASE_URL="${DATABASE_URL:-postgresql://localhost/traderjoe}"
PSQL="/opt/homebrew/opt/postgresql@16/bin/psql"

# Generate summary
SUMMARY=$(cat <<EOF
TraderJoe Daily Summary - $(date +%Y-%m-%d)
========================================

Today's Decision:
$($PSQL "$DATABASE_URL" -t -c "SELECT (decision->>'action') as action, confidence FROM ace_contexts ORDER BY timestamp DESC LIMIT 1;")

Playbook Status:
$($PSQL "$DATABASE_URL" -t -c "SELECT COUNT(*) as total, COUNT(CASE WHEN confidence > 0.6 THEN 1 END) as high_conf FROM playbook_bullets;")

Recent Performance (last 7 days):
$($PSQL "$DATABASE_URL" -t -c "SELECT COUNT(*) as trades, COUNT(CASE WHEN (outcome->>'win')::boolean THEN 1 END) as wins FROM ace_contexts WHERE timestamp > NOW() - INTERVAL '7 days' AND outcome IS NOT NULL;")

Latest Logs:
tail -20 ~/Documents/traderjoe/logs/evening_$(date +%Y%m%d).log
EOF
)

# Send email (macOS)
echo "$SUMMARY" | mail -s "TraderJoe Daily Summary" your_email@example.com

# Alternative: Save to file
echo "$SUMMARY" > ~/Documents/traderjoe/logs/daily_summary_$(date +%Y%m%d).txt

#!/bin/bash
# Setup Cron Jobs for TraderJoe Automated Paper Trading
# This script configures daily cron jobs for morning and evening routines

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}         TraderJoe Cron Job Setup                           ${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

# Check if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    CRON_CMD="crontab"
    echo -e "${YELLOW}Detected macOS${NC}"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    CRON_CMD="crontab"
    echo -e "${YELLOW}Detected Linux${NC}"
else
    echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
    exit 1
fi

# Backup existing crontab
echo -e "${YELLOW}Backing up existing crontab...${NC}"
$CRON_CMD -l > /tmp/crontab_backup_$(date +%Y%m%d_%H%M%S).txt 2>/dev/null || echo "No existing crontab found"

# Create log directory
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
echo -e "${GREEN}✓ Created log directory: $LOG_DIR${NC}"

# Get current crontab
CURRENT_CRON=$($CRON_CMD -l 2>/dev/null || echo "")

# Remove any existing TraderJoe entries
FILTERED_CRON=$(echo "$CURRENT_CRON" | grep -v "traderjoe" || true)

# Create new crontab with TraderJoe jobs
# PST timezone: 6:00 AM PST = 9:00 AM ET, 12:00 PM PST = 3:00 PM ET, 2:00 PM PST = 5:00 PM ET
# 6:00 AM PST - Morning routine (before market open at 9:30 AM ET / 6:30 AM PST)
# 12:00 PM PST - Auto-exit routine (3:00 PM ET - time-based exit)
# 2:00 PM PST - Evening routine (after market close at 4:00 PM ET / 1:00 PM PST)

NEW_CRON="$FILTERED_CRON

# TraderJoe Automated Paper Trading (PST timezone)
# Morning routine: 6:00 AM PST (9:00 AM ET)
0 6 * * 1-5 cd $PROJECT_ROOT && $SCRIPT_DIR/morning_routine.sh >> $LOG_DIR/morning_\$(date +\%Y\%m\%d).log 2>&1

# Auto-exit routine: 12:00 PM PST (3:00 PM ET)
0 12 * * 1-5 cd $PROJECT_ROOT && $SCRIPT_DIR/auto_exit.sh >> $LOG_DIR/auto_exit_\$(date +\%Y\%m\%d).log 2>&1

# Evening routine: 2:00 PM PST (5:00 PM ET)
0 14 * * 1-5 cd $PROJECT_ROOT && $SCRIPT_DIR/evening_routine.sh >> $LOG_DIR/evening_\$(date +\%Y\%m\%d).log 2>&1
"

# Install new crontab
echo "$NEW_CRON" | $CRON_CMD -

echo -e "\n${GREEN}✓ Cron jobs installed successfully!${NC}\n"

# Display the installed cron jobs
echo -e "${BLUE}Installed Cron Jobs:${NC}"
echo -e "${YELLOW}─────────────────────────────────────────────────────────${NC}"
$CRON_CMD -l | grep -A 5 "TraderJoe"
echo -e "${YELLOW}─────────────────────────────────────────────────────────${NC}\n"

# Important notes
echo -e "${BLUE}Important Notes:${NC}"
echo -e "${YELLOW}1. Time Zone:${NC}"
echo -e "   Configured for PST timezone:"
echo -e "   • Morning: 6:00 AM PST (9:00 AM ET - before market open)"
echo -e "   • Auto-Exit: 12:00 PM PST (3:00 PM ET - position exit time)"
echo -e "   • Evening: 2:00 PM PST (5:00 PM ET - after market close)\n"

echo -e "${YELLOW}2. Database Connection:${NC}"
echo -e "   Ensure DATABASE_URL is set in your environment or .env file:\n"
echo -e "   ${GREEN}export DATABASE_URL=\"postgresql://localhost/traderjoe\"${NC}\n"

echo -e "${YELLOW}3. Environment Variables:${NC}"
echo -e "   Cron runs with a minimal environment. You may need to add:"
echo -e "   • PATH includes cargo/rust binaries"
echo -e "   • PostgreSQL connection settings"
echo -e "   • API keys (EXA_API_KEY, etc.)\n"

echo -e "${YELLOW}4. Logs:${NC}"
echo -e "   Daily logs are stored in: ${GREEN}$LOG_DIR/${NC}"
echo -e "   • morning_YYYYMMDD.log"
echo -e "   • auto_exit_YYYYMMDD.log"
echo -e "   • evening_YYYYMMDD.log\n"

echo -e "${YELLOW}5. Manual Execution:${NC}"
echo -e "   You can still run manually with:"
echo -e "   ${GREEN}$SCRIPT_DIR/morning_routine.sh${NC}"
echo -e "   ${GREEN}$SCRIPT_DIR/auto_exit.sh${NC}"
echo -e "   ${GREEN}$SCRIPT_DIR/evening_routine.sh${NC}\n"

echo -e "${YELLOW}6. Trade Execution:${NC}"
echo -e "   The morning routine generates recommendations but does NOT auto-execute."
echo -e "   You must review and manually execute trades with:"
echo -e "   ${GREEN}traderjoe execute --recommendation-id <UUID>${NC}\n"

echo -e "${RED}⚠️  SAFETY REMINDER:${NC}"
echo -e "   These are PAPER TRADES for validation purposes."
echo -e "   The system will NOT execute trades automatically."
echo -e "   You maintain full control over all trading decisions.\n"

# Offer to create a wrapper script for auto-execution (optional)
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
read -p "Do you want to enable AUTOMATIC trade execution? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "\n${YELLOW}Creating auto-execution wrapper...${NC}"

    cat > "$SCRIPT_DIR/morning_routine_auto.sh" << 'EOF'
#!/bin/bash
# TraderJoe Morning Routine with Auto-Execution
# WARNING: This will AUTOMATICALLY execute trades based on ACE recommendations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TRADERJOE="$PROJECT_ROOT/target/release/traderjoe"

# Run morning analysis
$SCRIPT_DIR/morning_routine.sh

# Extract recommendation ID from the last analysis
# This is a simplified approach - in production you'd want more robust parsing
RECOMMENDATION_ID=$($TRADERJOE analyze --symbol SPY 2>&1 | grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | head -1)

if [ -n "$RECOMMENDATION_ID" ]; then
    echo "Found recommendation: $RECOMMENDATION_ID"

    # Check confidence level before executing
    # You can add additional checks here

    echo "Auto-executing trade..."
    $TRADERJOE execute --recommendation-id "$RECOMMENDATION_ID"
else
    echo "No recommendation ID found - skipping execution"
fi
EOF

    chmod +x "$SCRIPT_DIR/morning_routine_auto.sh"

    echo -e "${GREEN}✓ Created auto-execution script: morning_routine_auto.sh${NC}"
    echo -e "${RED}⚠️  To enable auto-execution, update your crontab to use:${NC}"
    echo -e "   ${YELLOW}morning_routine_auto.sh${NC} instead of ${YELLOW}morning_routine.sh${NC}\n"
else
    echo -e "\n${GREEN}Manual execution mode maintained (recommended).${NC}\n"
fi

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

echo -e "Next steps:"
echo -e "1. Test the morning routine manually: ${GREEN}$SCRIPT_DIR/morning_routine.sh${NC}"
echo -e "2. Check logs after first cron run: ${GREEN}$LOG_DIR/${NC}"
echo -e "3. Monitor the system during the 90-day validation period"
echo -e "4. View cron jobs anytime: ${GREEN}crontab -l${NC}"
echo -e "5. Remove cron jobs: ${GREEN}crontab -e${NC} (then delete TraderJoe lines)\n"

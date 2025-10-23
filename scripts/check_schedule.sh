#!/bin/bash
# Check Next Scheduled TraderJoe Runs
# Shows when the next morning and evening routines will execute

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}      TraderJoe Next Scheduled Runs                         ${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

if ! crontab -l 2>/dev/null | grep -q "traderjoe"; then
    echo -e "${YELLOW}No cron jobs configured.${NC}"
    echo -e "Run ${GREEN}./scripts/setup_cron.sh${NC} to enable automated trading.\n"
    exit 0
fi

echo -e "${GREEN}Current Time:${NC} $(date '+%Y-%m-%d %H:%M:%S %Z')\n"

# Parse cron schedule
MORNING_SCHEDULE=$(crontab -l | grep "morning_routine" | grep -v "^#" | awk '{print $1, $2, $3, $4, $5}')
EVENING_SCHEDULE=$(crontab -l | grep "evening_routine" | grep -v "^#" | awk '{print $1, $2, $3, $4, $5}')

if [ -n "$MORNING_SCHEDULE" ]; then
    echo -e "${YELLOW}Morning Routine Schedule:${NC}"
    echo "  Cron: $MORNING_SCHEDULE"
    echo "  Human: Weekdays (Mon-Fri) at 6:00 AM PST (9:00 AM ET)"
    echo ""
fi

if [ -n "$EVENING_SCHEDULE" ]; then
    echo -e "${YELLOW}Evening Routine Schedule:${NC}"
    echo "  Cron: $EVENING_SCHEDULE"
    echo "  Human: Weekdays (Mon-Fri) at 2:00 PM PST (5:00 PM ET)"
    echo ""
fi

# Calculate next runs (simplified)
CURRENT_HOUR=$(date +%H)
CURRENT_MINUTE=$(date +%M)
CURRENT_DAY=$(date +%u)  # 1=Monday, 5=Friday, 6=Saturday, 7=Sunday

echo -e "${BLUE}Next Scheduled Runs:${NC}"

# Morning (6:00 AM PST)
if [ "$CURRENT_DAY" -ge 1 ] && [ "$CURRENT_DAY" -le 5 ]; then
    # It's a weekday
    if [ "$CURRENT_HOUR" -lt 6 ]; then
        # Before 6 AM today
        echo -e "  ${GREEN}Morning:${NC} Today at 6:00 AM PST"
    else
        # After 6 AM, show next weekday
        if [ "$CURRENT_DAY" -eq 5 ]; then
            echo -e "  ${GREEN}Morning:${NC} Monday at 6:00 AM PST"
        else
            echo -e "  ${GREEN}Morning:${NC} Tomorrow at 6:00 AM PST"
        fi
    fi
else
    # It's weekend
    echo -e "  ${GREEN}Morning:${NC} Monday at 6:00 AM PST"
fi

# Evening (2:00 PM PST / 14:00)
if [ "$CURRENT_DAY" -ge 1 ] && [ "$CURRENT_DAY" -le 5 ]; then
    # It's a weekday
    if [ "$CURRENT_HOUR" -lt 14 ]; then
        # Before 2 PM today
        echo -e "  ${GREEN}Evening:${NC} Today at 2:00 PM PST"
    else
        # After 2 PM, show next weekday
        if [ "$CURRENT_DAY" -eq 5 ]; then
            echo -e "  ${GREEN}Evening:${NC} Monday at 2:00 PM PST"
        else
            echo -e "  ${GREEN}Evening:${NC} Tomorrow at 2:00 PM PST"
        fi
    fi
else
    # It's weekend
    echo -e "  ${GREEN}Evening:${NC} Monday at 2:00 PM PST"
fi

echo ""

# Show market status
echo -e "${BLUE}Market Status:${NC}"
if [ "$CURRENT_DAY" -ge 6 ]; then
    echo -e "  ${YELLOW}Weekend${NC} - Markets closed"
elif [ "$CURRENT_HOUR" -ge 9 ] && [ "$CURRENT_HOUR" -lt 16 ]; then
    echo -e "  ${GREEN}Open${NC} - Trading hours (9:30 AM - 4:00 PM ET)"
elif [ "$CURRENT_HOUR" -ge 16 ]; then
    echo -e "  ${YELLOW}After Hours${NC} - Market closed"
else
    echo -e "  ${YELLOW}Pre-Market${NC} - Market opens at 9:30 AM ET"
fi

echo -e "\n${BLUE}════════════════════════════════════════════════════════════${NC}\n"

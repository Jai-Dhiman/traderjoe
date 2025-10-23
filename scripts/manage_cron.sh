#!/bin/bash
# Manage TraderJoe Cron Jobs
# Utility script to enable/disable/status cron jobs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

show_status() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}         TraderJoe Cron Job Status                          ${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

    if crontab -l 2>/dev/null | grep -q "traderjoe"; then
        echo -e "${GREEN}✓ Cron jobs are ENABLED${NC}\n"
        echo -e "${YELLOW}Current schedule:${NC}"
        crontab -l | grep -A 5 "TraderJoe" || echo "No TraderJoe cron jobs found"
    else
        echo -e "${RED}✗ Cron jobs are DISABLED${NC}"
        echo -e "Run ${GREEN}./scripts/setup_cron.sh${NC} to enable automated trading.\n"
    fi

    # Check recent logs
    LOG_DIR="$PROJECT_ROOT/logs"
    if [ -d "$LOG_DIR" ]; then
        echo -e "\n${YELLOW}Recent log files:${NC}"
        ls -lht "$LOG_DIR"/*.log 2>/dev/null | head -5 || echo "No log files found"
    fi
}

disable_cron() {
    echo -e "${YELLOW}Disabling TraderJoe cron jobs...${NC}"

    # Backup current crontab
    crontab -l > /tmp/crontab_backup_$(date +%Y%m%d_%H%M%S).txt 2>/dev/null || true

    # Remove TraderJoe entries
    CURRENT_CRON=$(crontab -l 2>/dev/null || echo "")
    FILTERED_CRON=$(echo "$CURRENT_CRON" | grep -v "traderjoe" | grep -v "TraderJoe" || true)

    echo "$FILTERED_CRON" | crontab -

    echo -e "${GREEN}✓ Cron jobs disabled${NC}"
    echo -e "Backup saved to: /tmp/crontab_backup_$(date +%Y%m%d)_*.txt\n"
}

enable_cron() {
    echo -e "${YELLOW}Enabling TraderJoe cron jobs...${NC}"
    "$SCRIPT_DIR/setup_cron.sh"
}

view_logs() {
    LOG_DIR="$PROJECT_ROOT/logs"

    if [ ! -d "$LOG_DIR" ]; then
        echo -e "${RED}No logs directory found${NC}"
        return
    fi

    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}         TraderJoe Logs                                     ${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

    # Show latest morning log
    LATEST_MORNING=$(ls -t "$LOG_DIR"/morning_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_MORNING" ]; then
        echo -e "${GREEN}Latest Morning Log:${NC} $LATEST_MORNING"
        echo -e "${YELLOW}─────────────────────────────────────────────────────────${NC}"
        tail -50 "$LATEST_MORNING"
        echo -e "${YELLOW}─────────────────────────────────────────────────────────${NC}\n"
    fi

    # Show latest evening log
    LATEST_EVENING=$(ls -t "$LOG_DIR"/evening_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_EVENING" ]; then
        echo -e "${GREEN}Latest Evening Log:${NC} $LATEST_EVENING"
        echo -e "${YELLOW}─────────────────────────────────────────────────────────${NC}"
        tail -50 "$LATEST_EVENING"
        echo -e "${YELLOW}─────────────────────────────────────────────────────────${NC}\n"
    fi
}

test_cron() {
    echo -e "${YELLOW}Testing cron job execution...${NC}\n"

    echo -e "${BLUE}Running morning routine manually:${NC}"
    "$SCRIPT_DIR/morning_routine.sh"

    echo -e "\n${BLUE}Running evening routine manually:${NC}"
    "$SCRIPT_DIR/evening_routine.sh"

    echo -e "\n${GREEN}✓ Manual test completed${NC}"
    echo -e "If this works, cron should work too (check logs after scheduled run)\n"
}

show_help() {
    echo -e "${BLUE}TraderJoe Cron Management${NC}\n"
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  status    Show current cron job status and recent logs"
    echo "  enable    Enable automated trading cron jobs"
    echo "  disable   Disable automated trading cron jobs"
    echo "  logs      View recent execution logs"
    echo "  test      Test run both routines manually"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 status"
    echo "  $0 enable"
    echo "  $0 logs"
    echo ""
}

# Main command dispatcher
case "${1:-status}" in
    status)
        show_status
        ;;
    enable)
        enable_cron
        ;;
    disable)
        disable_cron
        ;;
    logs)
        view_logs
        ;;
    test)
        test_cron
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}\n"
        show_help
        exit 1
        ;;
esac

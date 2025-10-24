#!/usr/bin/env bash
# Check that required services are running and will auto-start

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables from .env file
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "════════════════════════════════════════════════════════════"
echo "         TraderJoe Service Health Check                    "
echo "════════════════════════════════════════════════════════════"
echo ""

# Check PostgreSQL
echo "Checking PostgreSQL..."
if pgrep -x "postgres" >/dev/null; then
    echo -e "${GREEN}✓ PostgreSQL is running${NC}"
else
    echo -e "${RED}✗ PostgreSQL is NOT running${NC}"
    echo "  Start with: brew services start postgresql@16"
fi

# Check Ollama
echo ""
echo "Checking Ollama..."
if pgrep -x "ollama" >/dev/null; then
    echo -e "${GREEN}✓ Ollama is running${NC}"

    # Check if model is available
    if ollama list | grep -q "llama3.2:3b"; then
        echo -e "${GREEN}✓ llama3.2:3b model is available${NC}"
    else
        echo -e "${YELLOW}⚠ llama3.2:3b model not found${NC}"
        echo "  Pull with: ollama pull llama3.2:3b"
    fi
else
    echo -e "${RED}✗ Ollama is NOT running${NC}"
    echo "  Start with: brew services start ollama"
fi

# Check if services will auto-start on boot
echo ""
echo "Checking auto-start configuration..."
if brew services list | grep "postgresql@16" | grep -q "started"; then
    echo -e "${GREEN}✓ PostgreSQL will auto-start on boot${NC}"
else
    echo -e "${YELLOW}⚠ PostgreSQL NOT configured to auto-start${NC}"
    echo "  Enable with: brew services start postgresql@16"
fi

if brew services list | grep "ollama" | grep -q "started"; then
    echo -e "${GREEN}✓ Ollama will auto-start on boot${NC}"
else
    echo -e "${YELLOW}⚠ Ollama NOT configured to auto-start${NC}"
    echo "  Enable with: brew services start ollama"
fi

# Check database connection
echo ""
echo "Testing database connection..."
if psql "${DATABASE_URL:-postgresql://localhost/traderjoe}" -c "SELECT 1" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Database connection successful${NC}"
else
    echo -e "${RED}✗ Database connection failed${NC}"
    echo "  Check DATABASE_URL: ${DATABASE_URL:-postgresql://localhost/traderjoe}"
fi

# Check environment variables
echo ""
echo "Checking environment variables..."
if [ -n "${DATABASE_URL:-}" ]; then
    echo -e "${GREEN}✓ DATABASE_URL is set${NC}"
else
    echo -e "${YELLOW}⚠ DATABASE_URL not set (will use default)${NC}"
fi

if [ -n "${EXA_API_KEY:-}" ]; then
    echo -e "${GREEN}✓ EXA_API_KEY is set${NC}"
else
    echo -e "${RED}✗ EXA_API_KEY not set${NC}"
    echo "  Research functionality will fail"
fi

if [ -n "${POLYGON_API_KEY:-}" ]; then
    echo -e "${GREEN}✓ POLYGON_API_KEY is set${NC}"
else
    echo -e "${RED}✗ POLYGON_API_KEY not set${NC}"
    echo "  Market data fetching will fail"
fi

# Check Python environment
echo ""
echo "Checking Python environment..."
if [ -d "/Users/jdhiman/Documents/traderjoe/.venv" ]; then
    echo -e "${GREEN}✓ Python virtual environment exists${NC}"
else
    echo -e "${RED}✗ Python virtual environment not found${NC}"
    echo "  Create with: uv venv && uv pip install sentence-transformers"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Summary:"
echo "  If all checks pass, you're ready for hands-off operation"
echo "  If any checks fail, fix them before enabling cron"
echo ""

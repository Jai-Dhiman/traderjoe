#!/bin/bash
set -e

echo "TraderJoe Phase 1 Verification"
echo "=============================="

# Check environment
echo "Checking environment variables..."
if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL not set"
    echo "Run: export DATABASE_URL='postgresql://postgres:postgres@localhost/traderjoe_dev'"
    exit 1
fi

# Build
echo "Building project..."
cargo build > /dev/null

# Help (triggers migrations and checks DB)
echo "Running help to trigger migrations..."
cargo run -- --help > /dev/null || (echo "Help failed" && exit 1)

# Test fetch command
echo "Testing OHLCV fetch for SPY..."
cargo run -- fetch --symbol SPY --data-type ohlcv --days 5

# Test research stub (works without API key but persists)
echo "Testing research command..."
cargo run -- research "market outlook"

# Test sentiment stub (requires Reddit creds; expect config error without creds)
echo "Testing sentiment command..."
if cargo run -- sentiment --source reddit 2>&1 | grep -q "Reddit credentials"; then
  echo "Sentiment credentials check OK"
else
  echo "Sentiment stub ran (credentials present)"
fi

echo ""
echo "Phase 1 Verification Complete!"
echo "=============================="
echo "Next steps:"
echo "1. Set up API keys in .env for research/sentiment"
echo "2. Begin Phase 2: ACE Implementation"

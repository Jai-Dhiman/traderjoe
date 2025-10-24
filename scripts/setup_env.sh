#!/bin/bash
# Setup environment variables for TraderJoe

# Detect project root
if [[ -n "${BASH_SOURCE[0]}" ]]; then
    # Script is being sourced or executed
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    # Fallback to current directory
    PROJECT_ROOT="$(pwd)"
fi

# =============================================================================
# LOAD .env FILE
# =============================================================================
# Try multiple locations for .env file
ENV_FILE=""
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    ENV_FILE="$PROJECT_ROOT/.env"
elif [[ -f "$(pwd)/.env" ]]; then
    ENV_FILE="$(pwd)/.env"
elif [[ -f ".env" ]]; then
    ENV_FILE=".env"
fi

if [[ -n "$ENV_FILE" ]]; then
    echo "Loading environment variables from $ENV_FILE..."
    # Export all variables from .env (skip comments and empty lines)
    set -a
    source "$ENV_FILE"
    set +a
    echo "✓ Environment variables loaded"
else
    echo "WARNING: .env file not found"
    echo "Copy .env.example to .env and fill in your API keys:"
    echo "  cp .env.example .env"
    echo ""
fi

# =============================================================================
# FALLBACK DEFAULTS (if not set in .env)
# =============================================================================

# Database
export DATABASE_URL="${DATABASE_URL:-postgresql://localhost/traderjoe}"

# Python environment for PyO3 (embeddings)
export PYO3_PYTHON="${PYO3_PYTHON:-/Users/jdhiman/Documents/traderjoe/.venv/bin/python}"
export PYTHONHOME="${PYTHONHOME:-/Users/jdhiman/.local/share/uv/python/cpython-3.11.14-macos-aarch64-none}"
export PYTHONPATH="${PYTHONPATH:-/Users/jdhiman/Documents/traderjoe/.venv/lib/python3.11/site-packages}"

# =============================================================================
# API KEY VALIDATION
# =============================================================================

if [[ -z "$POLYGON_API_KEY" ]]; then
    echo "WARNING: POLYGON_API_KEY is not set!"
    echo "Market data fetching will FAIL without this."
    echo "Add it to your .env file or get your free API key from: https://polygon.io"
fi

if [[ -z "$EXA_API_KEY" ]]; then
    echo "WARNING: EXA_API_KEY is not set!"
    echo "Research and market intelligence will FAIL without this."
    echo "Add it to your .env file or get your API key from: https://exa.ai"
fi

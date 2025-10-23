-- Phase 4: Paper Trading Engine
-- Migration 005: Paper Trades and Account Management

-- Paper trades table
CREATE TABLE IF NOT EXISTS paper_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    context_id UUID REFERENCES ace_contexts(id),
    symbol TEXT NOT NULL,
    trade_type TEXT NOT NULL CHECK (trade_type IN ('CALL', 'PUT', 'FLAT')),
    entry_price DOUBLE PRECISION NOT NULL,
    entry_time TIMESTAMPTZ NOT NULL,
    exit_price DOUBLE PRECISION,
    exit_time TIMESTAMPTZ,
    shares DOUBLE PRECISION NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('OPEN', 'CLOSED', 'CANCELLED')),
    pnl DOUBLE PRECISION,
    pnl_pct DOUBLE PRECISION,
    notes JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Position tracking
    strike_price DOUBLE PRECISION,
    expiration_date DATE,
    position_size_usd DOUBLE PRECISION NOT NULL,
    commission DOUBLE PRECISION DEFAULT 0.65,
    slippage_pct DOUBLE PRECISION DEFAULT 0.03,

    -- Risk metrics
    max_favorable_excursion DOUBLE PRECISION,  -- MFE: Best P&L reached
    max_adverse_excursion DOUBLE PRECISION,    -- MAE: Worst P&L reached

    -- Exit reason tracking
    exit_reason TEXT CHECK (exit_reason IN ('AUTO_EXIT', 'STOP_LOSS', 'TAKE_PROFIT', 'MANUAL', 'CIRCUIT_BREAKER'))
);

CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status);
CREATE INDEX IF NOT EXISTS idx_paper_trades_entry_time ON paper_trades(entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_paper_trades_symbol ON paper_trades(symbol);
CREATE INDEX IF NOT EXISTS idx_paper_trades_context_id ON paper_trades(context_id);

-- Account balance tracking
CREATE TABLE IF NOT EXISTS account_balance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    balance DOUBLE PRECISION NOT NULL,
    equity DOUBLE PRECISION NOT NULL,  -- balance + open position value
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    daily_pnl DOUBLE PRECISION,
    weekly_pnl DOUBLE PRECISION,
    monthly_pnl DOUBLE PRECISION,

    -- Risk metrics
    max_drawdown_pct DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    win_rate DOUBLE PRECISION,
    total_trades INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_account_balance_timestamp ON account_balance(timestamp DESC);

-- Circuit breaker tracking
CREATE TABLE IF NOT EXISTS circuit_breakers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    is_halted BOOLEAN NOT NULL DEFAULT FALSE,
    reason TEXT CHECK (reason IN ('DAILY_LOSS_LIMIT', 'WEEKLY_LOSS_LIMIT', 'CONSECUTIVE_LOSSES', 'SYSTEM_ERROR', 'MANUAL_HALT')),
    halted_at TIMESTAMPTZ,
    resumed_at TIMESTAMPTZ,
    triggered_by TEXT,  -- What triggered the halt (specific loss amount, error message, etc.)
    notes JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_circuit_breakers_is_halted ON circuit_breakers(is_halted);
CREATE INDEX IF NOT EXISTS idx_circuit_breakers_created_at ON circuit_breakers(created_at DESC);

-- Trade journal for detailed logging
CREATE TABLE IF NOT EXISTS trade_journal (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_id UUID REFERENCES paper_trades(id) ON DELETE CASCADE,
    market_snapshot JSONB NOT NULL,  -- VIX, futures, sentiment at trade time
    ace_reasoning TEXT,
    ml_signals JSONB,  -- Technical indicators, confidence scores
    post_trade_notes TEXT,
    similar_trades JSONB,  -- Vector similarity results
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trade_journal_trade_id ON trade_journal(trade_id);
CREATE INDEX IF NOT EXISTS idx_trade_journal_created_at ON trade_journal(created_at DESC);

-- Insert initial account balance (starting with $10,000)
INSERT INTO account_balance (balance, equity, daily_pnl, weekly_pnl, monthly_pnl, total_trades)
VALUES (10000.00, 10000.00, 0.00, 0.00, 0.00, 0);

-- Insert initial circuit breaker state (not halted)
INSERT INTO circuit_breakers (is_halted)
VALUES (FALSE);

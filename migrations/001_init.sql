-- TraderJoe Complete Database Schema
-- Consolidated migration for paper trading system

-- Enable pgvector extension for similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- OHLCV market data
CREATE TABLE IF NOT EXISTS ohlcv (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(10,2) NOT NULL,
    high DECIMAL(10,2) NOT NULL,
    low DECIMAL(10,2) NOT NULL,
    close DECIMAL(10,2) NOT NULL,
    volume BIGINT NOT NULL,
    source VARCHAR(50) NOT NULL,
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, date, source)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_date ON ohlcv(symbol, date);
CREATE INDEX IF NOT EXISTS idx_ohlcv_source ON ohlcv(source);

-- News data
CREATE TABLE IF NOT EXISTS news (
    id SERIAL PRIMARY KEY,
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    source VARCHAR(100) NOT NULL,
    title TEXT NOT NULL,
    url TEXT,
    content TEXT,
    sentiment DECIMAL(3,2),
    symbols TEXT[],
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_news_published ON news(published_at);
CREATE INDEX IF NOT EXISTS idx_news_symbols ON news USING GIN(symbols);
CREATE INDEX IF NOT EXISTS idx_news_source ON news(source);

-- Sentiment analysis data
CREATE TABLE IF NOT EXISTS sentiment (
    id SERIAL PRIMARY KEY,
    captured_at TIMESTAMP WITH TIME ZONE NOT NULL,
    source VARCHAR(50) NOT NULL,
    symbol VARCHAR(10),
    score DECIMAL(3,2) NOT NULL,
    meta JSONB,
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sentiment_symbol ON sentiment(symbol);
CREATE INDEX IF NOT EXISTS idx_sentiment_captured ON sentiment(captured_at);
CREATE INDEX IF NOT EXISTS idx_sentiment_source ON sentiment(source);

-- Research data from Exa API
CREATE TABLE IF NOT EXISTS research (
    id SERIAL PRIMARY KEY,
    captured_at TIMESTAMP WITH TIME ZONE NOT NULL,
    query TEXT NOT NULL,
    result JSONB NOT NULL,
    embedding VECTOR(768),
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_research_captured ON research(captured_at);
CREATE INDEX IF NOT EXISTS idx_research_query ON research USING GIN(to_tsvector('english', query));

-- ACE contexts for context evolution
CREATE TABLE IF NOT EXISTS ace_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    market_state JSONB NOT NULL,
    decision JSONB,
    reasoning TEXT,
    confidence REAL,
    outcome JSONB,
    embedding VECTOR(768),
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ace_contexts_timestamp ON ace_contexts(timestamp);
CREATE INDEX IF NOT EXISTS idx_ace_contexts_confidence ON ace_contexts(confidence);

-- Trading execution records
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL,
    action VARCHAR(10) NOT NULL CHECK (action IN ('BUY', 'SELL')),
    instrument VARCHAR(20) NOT NULL DEFAULT 'STOCK',
    quantity INTEGER NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    executed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    strategy VARCHAR(50),
    recommendation_id UUID,
    paper_trade BOOLEAN NOT NULL DEFAULT true,
    pnl DECIMAL(10,2),
    meta JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_executed_at ON trades(executed_at);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_trades_paper_trade ON trades(paper_trade);

-- Current positions (both paper and live)
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL,
    instrument VARCHAR(20) NOT NULL DEFAULT 'STOCK',
    quantity INTEGER NOT NULL,
    avg_cost DECIMAL(10,2) NOT NULL,
    current_price DECIMAL(10,2),
    paper_trade BOOLEAN NOT NULL DEFAULT true,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, instrument, paper_trade)
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_paper_trade ON positions(paper_trade);
CREATE INDEX IF NOT EXISTS idx_positions_updated_at ON positions(updated_at);

-- Trading recommendations from ACE
CREATE TABLE IF NOT EXISTS trading_recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL,
    recommendation VARCHAR(20) NOT NULL CHECK (recommendation IN ('BUY_CALLS', 'BUY_PUTS', 'STAY_FLAT')),
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    reasoning TEXT NOT NULL,
    price_target DECIMAL(10,2),
    stop_loss DECIMAL(10,2),
    position_size_pct REAL,
    ace_context_id UUID REFERENCES ace_contexts(id),
    executed BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_recommendations_symbol ON trading_recommendations(symbol);
CREATE INDEX IF NOT EXISTS idx_recommendations_created_at ON trading_recommendations(created_at);
CREATE INDEX IF NOT EXISTS idx_recommendations_executed ON trading_recommendations(executed);
CREATE INDEX IF NOT EXISTS idx_recommendations_confidence ON trading_recommendations(confidence);

-- ACE Playbook bullets table for incremental delta updates
CREATE TABLE IF NOT EXISTS playbook_bullets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    section TEXT NOT NULL CHECK (section IN (
        'pattern_insights',
        'failure_modes',
        'regime_rules',
        'model_reliability',
        'news_impact',
        'strategy_lifecycle'
    )),
    content TEXT NOT NULL,
    helpful_count INTEGER NOT NULL DEFAULT 0,
    harmful_count INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0.5 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    last_used TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source_context_id UUID REFERENCES ace_contexts(id),
    meta JSONB
);

CREATE INDEX IF NOT EXISTS idx_playbook_bullets_section ON playbook_bullets(section);
CREATE INDEX IF NOT EXISTS idx_playbook_bullets_confidence_desc ON playbook_bullets(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_playbook_bullets_last_used_desc ON playbook_bullets(last_used DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_playbook_bullets_created_at ON playbook_bullets(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_playbook_bullets_helpful_ratio ON playbook_bullets((helpful_count::REAL / GREATEST(helpful_count + harmful_count, 1)) DESC);
CREATE INDEX IF NOT EXISTS idx_playbook_bullets_content_search ON playbook_bullets USING GIN(to_tsvector('english', content));

-- Trigger to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_playbook_bullets_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_playbook_bullets_updated_at ON playbook_bullets;
CREATE TRIGGER trigger_playbook_bullets_updated_at
    BEFORE UPDATE ON playbook_bullets
    FOR EACH ROW
    EXECUTE FUNCTION update_playbook_bullets_updated_at();

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
    strike_price DOUBLE PRECISION,
    expiration_date DATE,
    position_size_usd DOUBLE PRECISION NOT NULL,
    commission DOUBLE PRECISION DEFAULT 0.65,
    slippage_pct DOUBLE PRECISION DEFAULT 0.03,
    max_favorable_excursion DOUBLE PRECISION,
    max_adverse_excursion DOUBLE PRECISION,
    exit_reason TEXT CHECK (exit_reason IN ('AUTO_EXIT', 'STOP_LOSS', 'TAKE_PROFIT', 'MANUAL', 'CIRCUIT_BREAKER')),
    estimated_slippage_pct DOUBLE PRECISION,
    actual_entry_slippage_pct DOUBLE PRECISION,
    actual_exit_slippage_pct DOUBLE PRECISION,
    market_vix DOUBLE PRECISION,
    option_moneyness TEXT CHECK (option_moneyness IN ('ITM', 'ATM', 'OTM')),
    vix_range TEXT,
    moneyness TEXT
);

CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status);
CREATE INDEX IF NOT EXISTS idx_paper_trades_entry_time ON paper_trades(entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_paper_trades_symbol ON paper_trades(symbol);
CREATE INDEX IF NOT EXISTS idx_paper_trades_context_id ON paper_trades(context_id);

-- Account balance tracking
CREATE TABLE IF NOT EXISTS account_balance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    balance DOUBLE PRECISION NOT NULL,
    equity DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    daily_pnl DOUBLE PRECISION,
    weekly_pnl DOUBLE PRECISION,
    monthly_pnl DOUBLE PRECISION,
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
    triggered_by TEXT,
    notes JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    version INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_circuit_breakers_is_halted ON circuit_breakers(is_halted);
CREATE INDEX IF NOT EXISTS idx_circuit_breakers_created_at ON circuit_breakers(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_circuit_breakers_halted ON circuit_breakers(is_halted, created_at DESC);

-- Trade journal for detailed logging
CREATE TABLE IF NOT EXISTS trade_journal (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_id UUID REFERENCES paper_trades(id) ON DELETE CASCADE,
    market_snapshot JSONB NOT NULL,
    ace_reasoning TEXT,
    ml_signals JSONB,
    post_trade_notes TEXT,
    similar_trades JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trade_journal_trade_id ON trade_journal(trade_id);
CREATE INDEX IF NOT EXISTS idx_trade_journal_created_at ON trade_journal(created_at DESC);

-- Slippage calibration table
CREATE TABLE IF NOT EXISTS slippage_calibration (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calibration_date DATE NOT NULL,
    vix_range TEXT NOT NULL,
    moneyness TEXT NOT NULL CHECK (moneyness IN ('ITM', 'ATM', 'OTM')),
    total_trades INTEGER NOT NULL,
    avg_estimated_slippage DOUBLE PRECISION NOT NULL,
    avg_actual_slippage DOUBLE PRECISION NOT NULL,
    slippage_error DOUBLE PRECISION NOT NULL,
    slippage_error_pct DOUBLE PRECISION NOT NULL,
    recommended_slippage DOUBLE PRECISION NOT NULL,
    applied BOOLEAN DEFAULT FALSE,
    applied_at TIMESTAMPTZ,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_slippage_calibration_date ON slippage_calibration(calibration_date DESC);
CREATE INDEX IF NOT EXISTS idx_slippage_calibration_vix_range ON slippage_calibration(vix_range);
CREATE INDEX IF NOT EXISTS idx_slippage_calibration_moneyness ON slippage_calibration(moneyness);

-- Initial data inserts
INSERT INTO account_balance (balance, equity, daily_pnl, weekly_pnl, monthly_pnl, total_trades)
VALUES (10000.00, 10000.00, 0.00, 0.00, 0.00, 0);

INSERT INTO circuit_breakers (is_halted)
VALUES (FALSE);

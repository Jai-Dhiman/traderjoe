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

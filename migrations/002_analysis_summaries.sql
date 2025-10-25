-- Daily analysis summaries table
-- Stores comprehensive morning analysis results for easy retrieval
CREATE TABLE IF NOT EXISTS daily_analysis_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_date DATE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    recommendation VARCHAR(20) NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    reasoning TEXT NOT NULL,
    key_factors TEXT[],
    risk_factors TEXT[],
    position_size_pct REAL,
    ace_context_id UUID REFERENCES ace_contexts(id),
    executed BOOLEAN NOT NULL DEFAULT false,
    execution_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(analysis_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_daily_analysis_date ON daily_analysis_summaries(analysis_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_analysis_symbol ON daily_analysis_summaries(symbol);
CREATE INDEX IF NOT EXISTS idx_daily_analysis_executed ON daily_analysis_summaries(executed);

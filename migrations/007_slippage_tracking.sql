-- Migration 007: Slippage Tracking and Analysis
-- Add fields to track estimated vs actual slippage for dynamic model calibration

-- Add slippage tracking fields to paper_trades
ALTER TABLE paper_trades
ADD COLUMN IF NOT EXISTS estimated_slippage_pct DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS actual_entry_slippage_pct DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS actual_exit_slippage_pct DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS market_vix DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS option_moneyness TEXT CHECK (option_moneyness IN ('ITM', 'ATM', 'OTM'));

-- Create slippage calibration table to track model performance over time
CREATE TABLE IF NOT EXISTS slippage_calibration (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calibration_date DATE NOT NULL,
    vix_range TEXT NOT NULL,  -- 'VIX_LOW', 'VIX_NORMAL', 'VIX_ELEVATED', 'VIX_HIGH'
    moneyness TEXT NOT NULL CHECK (moneyness IN ('ITM', 'ATM', 'OTM')),

    -- Aggregate statistics
    total_trades INTEGER NOT NULL,
    avg_estimated_slippage DOUBLE PRECISION NOT NULL,
    avg_actual_slippage DOUBLE PRECISION NOT NULL,
    slippage_error DOUBLE PRECISION NOT NULL,  -- avg(actual - estimated)
    slippage_error_pct DOUBLE PRECISION NOT NULL,  -- (actual - estimated) / estimated

    -- Recommendations
    recommended_slippage DOUBLE PRECISION NOT NULL,
    applied BOOLEAN DEFAULT FALSE,
    applied_at TIMESTAMPTZ,

    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_slippage_calibration_date ON slippage_calibration(calibration_date DESC);
CREATE INDEX IF NOT EXISTS idx_slippage_calibration_vix_range ON slippage_calibration(vix_range);
CREATE INDEX IF NOT EXISTS idx_slippage_calibration_moneyness ON slippage_calibration(moneyness);

-- Add comments for documentation
COMMENT ON COLUMN paper_trades.estimated_slippage_pct IS 'Estimated slippage at trade time from dynamic model';
COMMENT ON COLUMN paper_trades.actual_entry_slippage_pct IS 'Actual slippage observed at entry (if available from real fills)';
COMMENT ON COLUMN paper_trades.actual_exit_slippage_pct IS 'Actual slippage observed at exit (if available from real fills)';
COMMENT ON COLUMN paper_trades.market_vix IS 'VIX level at trade time';
COMMENT ON COLUMN paper_trades.option_moneyness IS 'Option moneyness at trade time (ITM/ATM/OTM)';

COMMENT ON TABLE slippage_calibration IS 'Tracks slippage model performance and calibration adjustments over time';

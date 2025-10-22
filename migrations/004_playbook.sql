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
    -- Optional relation to ace_contexts for traceability
    source_context_id UUID REFERENCES ace_contexts(id),
    -- Metadata for additional information
    meta JSONB
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_playbook_bullets_section ON playbook_bullets(section);
CREATE INDEX IF NOT EXISTS idx_playbook_bullets_confidence_desc ON playbook_bullets(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_playbook_bullets_last_used_desc ON playbook_bullets(last_used DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_playbook_bullets_created_at ON playbook_bullets(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_playbook_bullets_helpful_ratio ON playbook_bullets((helpful_count::REAL / GREATEST(helpful_count + harmful_count, 1)) DESC);

-- Full-text search on content for pattern matching
CREATE INDEX IF NOT EXISTS idx_playbook_bullets_content_search ON playbook_bullets USING GIN(to_tsvector('english', content));

-- Trigger to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_playbook_bullets_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_playbook_bullets_updated_at
    BEFORE UPDATE ON playbook_bullets
    FOR EACH ROW
    EXECUTE FUNCTION update_playbook_bullets_updated_at();
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

-- Create HNSW index for similarity search (will be populated later)
-- This will be created after we have some data
-- CREATE INDEX IF NOT EXISTS idx_ace_contexts_embedding 
-- ON ace_contexts USING hnsw (embedding vector_cosine_ops);
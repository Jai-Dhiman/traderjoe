-- Add version field for optimistic locking
ALTER TABLE circuit_breakers ADD COLUMN version INTEGER DEFAULT 1;

-- Add index for faster lookups on created_at (since we ORDER BY created_at DESC LIMIT 1)
CREATE INDEX IF NOT EXISTS idx_circuit_breakers_created_at ON circuit_breakers(created_at DESC);

-- Add index on is_halted for quick status checks
CREATE INDEX IF NOT EXISTS idx_circuit_breakers_halted ON circuit_breakers(is_halted, created_at DESC);

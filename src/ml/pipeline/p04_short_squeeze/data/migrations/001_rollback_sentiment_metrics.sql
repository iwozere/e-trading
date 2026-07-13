-- Rollback Migration: Remove enhanced sentiment metrics
-- Version: 1.0
-- Date: 2025-11-17
-- Description: Rolls back sentiment integration changes

BEGIN;

-- Drop cleanup function
DROP FUNCTION IF EXISTS cleanup_old_sentiment_history();

-- Drop sentiment history table
DROP TABLE IF EXISTS ss_sentiment_history;

-- Drop indexes
DROP INDEX IF EXISTS idx_ss_deep_metrics_virality;
DROP INDEX IF EXISTS idx_ss_deep_metrics_mentions_growth;
DROP INDEX IF EXISTS idx_ss_deep_metrics_mentions;

-- Remove constraints (PostgreSQL requires dropping them individually)
ALTER TABLE ss_deep_metrics
DROP CONSTRAINT IF EXISTS check_virality_range,
DROP CONSTRAINT IF EXISTS check_bot_pct_range,
DROP CONSTRAINT IF EXISTS check_mentions_positive;

-- Remove columns
ALTER TABLE ss_deep_metrics
DROP COLUMN IF EXISTS mentions_24h,
DROP COLUMN IF EXISTS mentions_growth_7d,
DROP COLUMN IF EXISTS virality_index,
DROP COLUMN IF EXISTS bot_pct,
DROP COLUMN IF EXISTS sentiment_data_quality;

COMMIT;

-- Verification
-- SELECT column_name
-- FROM information_schema.columns
-- WHERE table_name = 'ss_deep_metrics'
-- AND column_name IN ('mentions_24h', 'mentions_growth_7d', 'virality_index', 'bot_pct', 'sentiment_data_quality');
-- Should return no rows if rollback successful

-- Rollback Migration: Remove tech_discourse signal class columns + Hacker News corpus cache
-- Version: 2.0
-- Date: 2026-08-20
-- Description: Rolls back 002_add_tech_discourse_signal.sql

BEGIN;

DROP TABLE IF EXISTS ss_hn_corpus;

ALTER TABLE ss_deep_metrics
DROP CONSTRAINT IF EXISTS check_tech_mentions_24h,
DROP CONSTRAINT IF EXISTS check_tech_sentiment_score_24h,
DROP CONSTRAINT IF EXISTS check_tech_sentiment_24h,
DROP CONSTRAINT IF EXISTS check_tech_discussion_depth;

ALTER TABLE ss_deep_metrics
DROP COLUMN IF EXISTS tech_mentions_24h,
DROP COLUMN IF EXISTS tech_sentiment_score_24h,
DROP COLUMN IF EXISTS tech_sentiment_24h,
DROP COLUMN IF EXISTS tech_discussion_depth,
DROP COLUMN IF EXISTS tech_coverage_available;

COMMIT;

-- Verification
-- SELECT column_name FROM information_schema.columns
-- WHERE table_name = 'ss_deep_metrics' AND column_name LIKE 'tech_%';
-- SELECT to_regclass('ss_hn_corpus');
-- Both should return no rows / NULL if rollback successful

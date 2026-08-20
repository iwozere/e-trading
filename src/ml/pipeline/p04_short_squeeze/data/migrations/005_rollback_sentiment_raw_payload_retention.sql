-- Rollback Migration: Remove sentiment raw_payload retention function
-- Version: 5.0
-- Date: 2026-08-20
-- Description: Rolls back 005_add_sentiment_raw_payload_retention.sql

BEGIN;

DROP FUNCTION IF EXISTS purge_old_sentiment_raw_payload(INTEGER);

COMMIT;

-- Verification
-- SELECT proname FROM pg_proc WHERE proname = 'purge_old_sentiment_raw_payload';
-- Should return no rows if rollback successful

-- Rollback Migration: Remove per-source sentiment calibration table
-- Version: 3.0
-- Date: 2026-08-20
-- Description: Rolls back 003_add_sentiment_calibration.sql

BEGIN;

DROP TABLE IF EXISTS ss_sentiment_calibration;

COMMIT;

-- Verification
-- SELECT to_regclass('ss_sentiment_calibration');
-- Should return NULL if rollback successful

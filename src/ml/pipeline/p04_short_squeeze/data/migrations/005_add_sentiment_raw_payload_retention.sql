-- Migration: Add sentiment raw_payload retention function
-- Version: 5.0
-- Date: 2026-08-20
-- Description: sentiment-spec-rev2.md §1.3/§2.11 -- "raw_payload retained for audit under access
-- control. Add a retention policy -- default 90 days -- and a purge job. Indefinite retention of
-- third-party social content has no upside for a signal pipeline with a 7-day feature horizon."
--
-- ss_deep_metrics.raw_payload already existed (pre-dates this migration) but was never actually
-- populated until daily_deep_scan.py started writing it in Rev 2 Phase 4. This adds a purge
-- function, following the same pattern as cleanup_old_sentiment_history() in
-- 001_add_sentiment_metrics.sql, invoked by scripts/run_sentiment_retention.py.
--
-- Nulls raw_payload on old rows rather than deleting the rows themselves -- squeeze_score and
-- the other scalar sentiment columns remain useful for historical backtesting long after the
-- underlying third-party post content should be purged.

BEGIN;

CREATE OR REPLACE FUNCTION purge_old_sentiment_raw_payload(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    purged_count INTEGER;
BEGIN
    UPDATE ss_deep_metrics
    SET raw_payload = NULL
    WHERE date < CURRENT_DATE - (retention_days || ' days')::INTERVAL
      AND raw_payload IS NOT NULL;

    GET DIAGNOSTICS purged_count = ROW_COUNT;
    RETURN purged_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION purge_old_sentiment_raw_payload(INTEGER) IS
'Nulls ss_deep_metrics.raw_payload for rows older than retention_days (default 90). Returns the number of rows purged. Should be run periodically via scripts/run_sentiment_retention.py (spec §1.3/§2.11).';

COMMIT;

-- Verification queries
-- SELECT purge_old_sentiment_raw_payload(90);  -- dry-run equivalent: check count first, see the script's --dry-run
-- SELECT count(*) FROM ss_deep_metrics WHERE raw_payload IS NOT NULL AND date < CURRENT_DATE - INTERVAL '90 days';

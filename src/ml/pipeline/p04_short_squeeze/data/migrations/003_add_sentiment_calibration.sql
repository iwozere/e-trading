-- Migration: Add per-source sentiment calibration table
-- Version: 3.0
-- Date: 2026-08-20
-- Description: Rev 2 of the sentiment spec (see src/common/sentiments/docs/sentiment-spec-rev2.md
-- §2.5.6) calibrates each provider's raw sentiment score against its own trailing 30-day
-- distribution (z-score) before blending, rather than blending raw values directly -- raw scores
-- aren't comparable across platforms (Bluesky finance chatter skews promotional-positive, Hacker
-- News skews critical-negative). This migration adds ss_sentiment_calibration, one row per
-- (provider, day), written after each batch run and pooled over a trailing window at read time.

BEGIN;

CREATE TABLE IF NOT EXISTS ss_sentiment_calibration (
    provider    VARCHAR(32) NOT NULL,
    day         DATE NOT NULL,
    mean_score  NUMERIC(10, 6) NOT NULL,
    std_score   NUMERIC(10, 6) NOT NULL,
    n_obs       INTEGER NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (provider, day),
    CONSTRAINT check_calibration_n_obs_nonneg CHECK (n_obs >= 0),
    CONSTRAINT check_calibration_std_nonneg CHECK (std_score >= 0)
);

CREATE INDEX IF NOT EXISTS idx_ss_sentiment_calibration_day ON ss_sentiment_calibration(day);

COMMENT ON TABLE ss_sentiment_calibration IS
'Per-source, per-day sentiment score distribution (spec §2.5.6). A trailing window (default 30 days) of rows is pooled at read time into one (mean, std, n) used to z-score calibrate that provider''s raw scores before blending. Falls back to raw scores until n_obs across the window reaches min_observations (default 200) -- see processing/calibration.py.';

COMMIT;

-- Verification query
-- SELECT provider, day, mean_score, std_score, n_obs FROM ss_sentiment_calibration ORDER BY provider, day DESC;

-- Migration: Add tech_discourse signal class (Hacker News) + shared corpus cache
-- Version: 2.0
-- Date: 2026-08-20
-- Description: Rev 2 of the sentiment spec (see src/common/sentiments/docs/sentiment-spec-rev2.md)
-- adds a second, structurally distinct signal class -- tech_discourse (Hacker News) -- reported
-- separately from retail sentiment and never blended into it (spec §2.5.6). This migration adds:
--   1. tech_* columns on ss_deep_metrics
--   2. ss_hn_corpus, the shared Hacker News corpus cache (spec §2.4/§2.7)
-- It also backfills mentions_24h/virality_index/bot_pct/sentiment_data_quality persistence,
-- which migration 001 added columns for but which daily_deep_scan.py never actually wrote
-- (_store_results() silently dropped them -- fixed alongside this migration, no schema change
-- needed for that part since the columns already exist).

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. tech_discourse columns on ss_deep_metrics
-- ---------------------------------------------------------------------------
ALTER TABLE ss_deep_metrics
ADD COLUMN IF NOT EXISTS tech_mentions_24h INTEGER,
ADD COLUMN IF NOT EXISTS tech_sentiment_score_24h NUMERIC(4, 3),
ADD COLUMN IF NOT EXISTS tech_sentiment_24h NUMERIC(4, 3),
ADD COLUMN IF NOT EXISTS tech_discussion_depth NUMERIC(8, 2),
ADD COLUMN IF NOT EXISTS tech_coverage_available BOOLEAN;

-- PostgreSQL has no "ADD CONSTRAINT IF NOT EXISTS" (unlike ADD COLUMN IF NOT EXISTS) --
-- wrap each in a DO block that checks pg_constraint first so this migration stays idempotent.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'check_tech_mentions_24h') THEN
        ALTER TABLE ss_deep_metrics
            ADD CONSTRAINT check_tech_mentions_24h CHECK (tech_mentions_24h >= 0);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'check_tech_sentiment_score_24h') THEN
        ALTER TABLE ss_deep_metrics
            ADD CONSTRAINT check_tech_sentiment_score_24h
                CHECK (tech_sentiment_score_24h >= -1 AND tech_sentiment_score_24h <= 1);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'check_tech_sentiment_24h') THEN
        ALTER TABLE ss_deep_metrics
            ADD CONSTRAINT check_tech_sentiment_24h
                CHECK (tech_sentiment_24h >= -1 AND tech_sentiment_24h <= 1);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'check_tech_discussion_depth') THEN
        ALTER TABLE ss_deep_metrics
            ADD CONSTRAINT check_tech_discussion_depth CHECK (tech_discussion_depth >= 0);
    END IF;
END $$;

COMMENT ON COLUMN ss_deep_metrics.tech_mentions_24h IS
'Hacker News mentions in the last 24h for tickers covered by the entity map. NULL (not 0) when tech_coverage_available is false -- distinguishes "no data" from "covered, zero mentions".';

COMMENT ON COLUMN ss_deep_metrics.tech_sentiment_score_24h IS
'Raw (-1..1) tech_discourse sentiment score. Engineering-reputation signal, structurally different from retail sentiment_score_24h -- never blended with it. See sentiment-spec-rev2.md §2.5.6.';

COMMENT ON COLUMN ss_deep_metrics.tech_sentiment_24h IS
'Normalized (0..1) tech_discourse sentiment, mirroring sentiment_24h''s normalization for retail. Independent feature; add to squeeze_score only if a backtest justifies it (spec §2.5.6), not by default.';

COMMENT ON COLUMN ss_deep_metrics.tech_discussion_depth IS
'Mean comment count per Hacker News story that matched this ticker''s entity map. NULL when tech_coverage_available is false.';

COMMENT ON COLUMN ss_deep_metrics.tech_coverage_available IS
'Whether this ticker is present in the Hacker News entity map at all (entity/tickers.yml). False means "no data" -- the other tech_* columns are then NULL, never a fabricated neutral reading.';

-- ---------------------------------------------------------------------------
-- 2. Shared Hacker News corpus cache
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ss_hn_corpus (
    item_id     BIGINT PRIMARY KEY,
    item_type   VARCHAR(10) NOT NULL,
    parent_id   BIGINT,
    story_id    BIGINT,
    author_hash VARCHAR(64),
    created_utc TIMESTAMPTZ NOT NULL,
    text_clean  TEXT,
    score       INTEGER,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT check_hn_item_type CHECK (item_type IN ('story', 'comment', 'job', 'poll'))
);

CREATE INDEX IF NOT EXISTS idx_ss_hn_corpus_created_utc ON ss_hn_corpus(created_utc);
CREATE INDEX IF NOT EXISTS idx_ss_hn_corpus_story_id ON ss_hn_corpus(story_id);
CREATE INDEX IF NOT EXISTS idx_ss_hn_corpus_fetched_at ON ss_hn_corpus(fetched_at);

COMMENT ON TABLE ss_hn_corpus IS
'Shared Hacker News corpus cache (spec §2.4/§2.7). Populated once per batch run and entity-matched against every ticker in-process -- cost is O(corpus size), independent of batch size. author_hash is salted (spec §2.11); no raw HN usernames are stored.';

COMMIT;

-- Verification queries
-- SELECT column_name, data_type FROM information_schema.columns
-- WHERE table_name = 'ss_deep_metrics' AND column_name LIKE 'tech_%';
-- SELECT count(*) FROM ss_hn_corpus;

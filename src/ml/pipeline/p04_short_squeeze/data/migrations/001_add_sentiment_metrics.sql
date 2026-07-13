-- Migration: Add enhanced sentiment metrics to ss_deep_metrics
-- Version: 1.0
-- Date: 2025-11-17
-- Description: Adds multi-source sentiment fields for virality, mention tracking, and bot detection

BEGIN;

-- Add new columns for enhanced sentiment
ALTER TABLE ss_deep_metrics
ADD COLUMN IF NOT EXISTS mentions_24h INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS mentions_growth_7d FLOAT DEFAULT NULL,
ADD COLUMN IF NOT EXISTS virality_index FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS bot_pct FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS sentiment_data_quality JSONB DEFAULT '{}'::jsonb;

-- Add indexes for common queries
CREATE INDEX IF NOT EXISTS idx_ss_deep_metrics_virality
ON ss_deep_metrics(virality_index DESC)
WHERE virality_index > 0.5;

CREATE INDEX IF NOT EXISTS idx_ss_deep_metrics_mentions_growth
ON ss_deep_metrics(mentions_growth_7d DESC)
WHERE mentions_growth_7d IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_ss_deep_metrics_mentions
ON ss_deep_metrics(mentions_24h DESC)
WHERE mentions_24h > 0;

-- Add check constraints for data validation
ALTER TABLE ss_deep_metrics
ADD CONSTRAINT IF NOT EXISTS check_virality_range
    CHECK (virality_index >= 0 AND virality_index <= 1),
ADD CONSTRAINT IF NOT EXISTS check_bot_pct_range
    CHECK (bot_pct >= 0 AND bot_pct <= 1),
ADD CONSTRAINT IF NOT EXISTS check_mentions_positive
    CHECK (mentions_24h >= 0);

-- Add comments for documentation
COMMENT ON COLUMN ss_deep_metrics.mentions_24h IS
'Total mention count across all sentiment sources (StockTwits, Reddit, News, etc.) in the last 24 hours';

COMMENT ON COLUMN ss_deep_metrics.mentions_growth_7d IS
'Percentage growth in mentions compared to 7-day historical average. NULL if no historical data available. Formula: (current - avg_7d) / avg_7d';

COMMENT ON COLUMN ss_deep_metrics.virality_index IS
'Engagement-weighted virality score (0-1) based on likes, replies, retweets, and author influence. Higher = more viral.';

COMMENT ON COLUMN ss_deep_metrics.bot_pct IS
'Estimated percentage of bot activity (0-1) based on account age, posting patterns, and content similarity. Higher = more bot activity.';

COMMENT ON COLUMN ss_deep_metrics.sentiment_data_quality IS
'JSON object with per-provider status and metadata. Format: {"stocktwits": "ok", "reddit": "partial", "news": "failed", "providers_count": 2}';

-- Create table for historical mention tracking (for growth calculation)
CREATE TABLE IF NOT EXISTS ss_sentiment_history (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    mentions_count INTEGER NOT NULL DEFAULT 0,
    unique_authors INTEGER NOT NULL DEFAULT 0,
    sentiment_avg FLOAT NOT NULL DEFAULT 0.0,
    virality_avg FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_ticker_date UNIQUE (ticker, date)
);

-- Indexes for efficient lookups
CREATE INDEX IF NOT EXISTS idx_ss_sentiment_history_ticker_date
ON ss_sentiment_history(ticker, date DESC);

CREATE INDEX IF NOT EXISTS idx_ss_sentiment_history_date
ON ss_sentiment_history(date DESC);

-- Add comments
COMMENT ON TABLE ss_sentiment_history IS
'Historical sentiment metrics for calculating growth trends. Retention: 30 days (managed by cleanup job).';

COMMENT ON COLUMN ss_sentiment_history.mentions_count IS
'Daily total mentions across all sentiment sources';

COMMENT ON COLUMN ss_sentiment_history.unique_authors IS
'Count of unique authors/contributors for the day';

COMMENT ON COLUMN ss_sentiment_history.sentiment_avg IS
'Average sentiment score for the day (0-1 normalized)';

COMMENT ON COLUMN ss_sentiment_history.virality_avg IS
'Average virality index for the day (0-1)';

-- Create cleanup function for old sentiment history (30-day retention)
CREATE OR REPLACE FUNCTION cleanup_old_sentiment_history()
RETURNS void AS $$
BEGIN
    DELETE FROM ss_sentiment_history
    WHERE date < CURRENT_DATE - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION cleanup_old_sentiment_history() IS
'Cleanup function to remove sentiment history older than 30 days. Should be run daily via cron/scheduler.';

COMMIT;

-- Verification queries
-- SELECT column_name, data_type, is_nullable, column_default
-- FROM information_schema.columns
-- WHERE table_name = 'ss_deep_metrics'
-- AND column_name IN ('mentions_24h', 'mentions_growth_7d', 'virality_index', 'bot_pct', 'sentiment_data_quality');

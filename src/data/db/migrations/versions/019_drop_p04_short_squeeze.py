"""drop_p04_short_squeeze

Revision ID: 019_drop_p04_short_squeeze
Revises: 018_p22_trial_dates
Create Date: 2026-11-10

Drops the six tables of the retired P04 short-squeeze pipeline: ss_snapshot, ss_deep_metrics,
ss_alerts, ss_ad_hoc_candidates, ss_sentiment_calibration and ss_hn_corpus. P04 never ran on a
schedule; the only live writer was the Hacker News sentiment adapter's corpus cache
(ss_hn_corpus), which now caches to DATA_CACHE_DIR/sentiment/hn_corpus.json instead. No table
has a foreign key. The downgrade recreates all six empty, as the models last defined them.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "019_drop_p04_short_squeeze"
down_revision = "018_p22_trial_dates"
branch_labels = None
depends_on = None

_TABLES = [
    "ss_snapshot",
    "ss_deep_metrics",
    "ss_alerts",
    "ss_ad_hoc_candidates",
    "ss_sentiment_calibration",
    "ss_hn_corpus",
]

# CREATE TABLE / CREATE INDEX statements generated from model_short_squeeze.py before its removal.
_RECREATE_DDL = [
    """
CREATE TABLE ss_ad_hoc_candidates (
    id BIGSERIAL NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    reason TEXT,
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE,
    active BOOLEAN DEFAULT true NOT NULL,
    promoted_by_screener BOOLEAN DEFAULT false NOT NULL,
    CONSTRAINT pk_ss_ad_hoc_candidates PRIMARY KEY (id),
    CONSTRAINT ss_ad_hoc_candidates_ticker_key UNIQUE (ticker)
)
    """,
    """
CREATE INDEX idx_ss_adhoc_promoted ON ss_ad_hoc_candidates (promoted_by_screener, active)
    """,
    """
CREATE INDEX idx_ss_adhoc_active ON ss_ad_hoc_candidates (active, expires_at)
    """,
    """
CREATE INDEX idx_ss_adhoc_expires_at ON ss_ad_hoc_candidates (expires_at)
    """,
    """
CREATE TABLE ss_alerts (
    id BIGSERIAL NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    alert_level VARCHAR(10) NOT NULL,
    reason TEXT,
    squeeze_score NUMERIC(5, 4),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    sent BOOLEAN DEFAULT false NOT NULL,
    cooldown_expires TIMESTAMP WITH TIME ZONE,
    notification_id VARCHAR(50),
    CONSTRAINT pk_ss_alerts PRIMARY KEY (id),
    CONSTRAINT ss_alerts_alert_level_check CHECK (alert_level IN ('LOW', 'MEDIUM', 'HIGH')),
    CONSTRAINT ss_alerts_squeeze_score_check CHECK (squeeze_score >= 0 AND squeeze_score <= 1)
)
    """,
    """
CREATE INDEX idx_ss_alerts_ticker_cooldown ON ss_alerts (ticker, cooldown_expires)
    """,
    """
CREATE INDEX idx_ss_alerts_timestamp_desc ON ss_alerts ("timestamp" DESC)
    """,
    """
CREATE INDEX idx_ss_alerts_alert_level_timestamp ON ss_alerts (alert_level, "timestamp" DESC)
    """,
    """
CREATE INDEX idx_ss_alerts_sent_timestamp ON ss_alerts (sent, "timestamp" DESC)
    """,
    """
CREATE TABLE ss_deep_metrics (
    id BIGSERIAL NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    volume_spike NUMERIC(6, 2),
    call_put_ratio NUMERIC(6, 2),
    sentiment_24h NUMERIC(4, 3),
    borrow_fee_pct NUMERIC(5, 4),
    squeeze_score NUMERIC(5, 4),
    alert_level VARCHAR(10),
    raw_payload JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    mentions_24h INTEGER DEFAULT 0,
    mentions_growth_7d FLOAT,
    virality_index FLOAT DEFAULT 0.0,
    bot_pct FLOAT DEFAULT 0.0,
    sentiment_data_quality JSONB DEFAULT '{}'::jsonb,
    tech_mentions_24h INTEGER,
    tech_sentiment_score_24h NUMERIC(4, 3),
    tech_sentiment_24h NUMERIC(4, 3),
    tech_discussion_depth NUMERIC(8, 2),
    tech_coverage_available BOOLEAN,
    CONSTRAINT pk_ss_deep_metrics PRIMARY KEY (id),
    CONSTRAINT ss_deep_metrics_ticker_date_key UNIQUE (ticker, date),
    CONSTRAINT ss_deep_metrics_volume_spike_check CHECK (volume_spike >= 0),
    CONSTRAINT ss_deep_metrics_call_put_ratio_check CHECK (call_put_ratio >= 0),
    CONSTRAINT ss_deep_metrics_sentiment_24h_check CHECK (sentiment_24h >= -1 AND sentiment_24h <= 1),
    CONSTRAINT ss_deep_metrics_borrow_fee_pct_check CHECK (borrow_fee_pct >= 0),
    CONSTRAINT ss_deep_metrics_squeeze_score_check CHECK (squeeze_score >= 0 AND squeeze_score <= 1),
    CONSTRAINT ss_deep_metrics_alert_level_check CHECK (alert_level IN ('LOW', 'MEDIUM', 'HIGH')),
    CONSTRAINT check_tech_mentions_24h CHECK (tech_mentions_24h >= 0),
    CONSTRAINT check_tech_sentiment_score_24h CHECK (tech_sentiment_score_24h >= -1 AND tech_sentiment_score_24h <= 1),
    CONSTRAINT check_tech_sentiment_24h CHECK (tech_sentiment_24h >= -1 AND tech_sentiment_24h <= 1),
    CONSTRAINT check_tech_discussion_depth CHECK (tech_discussion_depth >= 0),
    CONSTRAINT check_mentions_positive CHECK (mentions_24h >= 0),
    CONSTRAINT check_virality_range CHECK (virality_index >= 0),
    CONSTRAINT check_bot_pct_range CHECK (bot_pct >= 0 AND bot_pct <= 1)
)
    """,
    """
CREATE INDEX idx_ss_deep_metrics_sentiment24 ON ss_deep_metrics (sentiment_24h)
    """,
    """
CREATE INDEX idx_ss_deep_metrics_created_at ON ss_deep_metrics (created_at)
    """,
    """
CREATE INDEX idx_ss_deep_metrics_squeeze_score_desc ON ss_deep_metrics (squeeze_score DESC, date DESC)
    """,
    """
CREATE INDEX idx_ss_deep_metrics_virality ON ss_deep_metrics (virality_index DESC)
    """,
    """
CREATE INDEX idx_ss_deep_metrics_date_desc ON ss_deep_metrics (date DESC)
    """,
    """
CREATE INDEX idx_ss_deep_metrics_ticker_date ON ss_deep_metrics (ticker, date)
    """,
    """
CREATE INDEX idx_ss_deep_metrics_mentions ON ss_deep_metrics (mentions_24h DESC) WHERE mentions_24h > 0
    """,
    """
CREATE INDEX idx_ss_deep_metrics_alert_level ON ss_deep_metrics (alert_level, date DESC)
    """,
    """
CREATE INDEX idx_ss_deep_metrics_mentions_growth ON ss_deep_metrics (mentions_growth_7d DESC) WHERE mentions_growth_7d IS NOT NULL
    """,
    """
CREATE TABLE ss_hn_corpus (
    item_id BIGINT NOT NULL,
    item_type VARCHAR(10) NOT NULL,
    parent_id BIGINT,
    story_id BIGINT,
    author_hash VARCHAR(64),
    created_utc TIMESTAMP WITH TIME ZONE NOT NULL,
    text_clean TEXT,
    score INTEGER,
    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    CONSTRAINT pk_ss_hn_corpus PRIMARY KEY (item_id),
    CONSTRAINT check_hn_item_type CHECK (item_type IN ('story', 'comment', 'job', 'poll'))
)
    """,
    """
CREATE INDEX idx_ss_hn_corpus_story_id ON ss_hn_corpus (story_id)
    """,
    """
CREATE INDEX idx_ss_hn_corpus_fetched_at ON ss_hn_corpus (fetched_at)
    """,
    """
CREATE INDEX idx_ss_hn_corpus_created_utc ON ss_hn_corpus (created_utc)
    """,
    """
CREATE TABLE ss_sentiment_calibration (
    provider VARCHAR(32) NOT NULL,
    day DATE NOT NULL,
    mean_score NUMERIC(10, 6) NOT NULL,
    std_score NUMERIC(10, 6) NOT NULL,
    n_obs INTEGER NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    CONSTRAINT pk_ss_sentiment_calibration PRIMARY KEY (provider, day),
    CONSTRAINT check_calibration_n_obs_nonneg CHECK (n_obs >= 0),
    CONSTRAINT check_calibration_std_nonneg CHECK (std_score >= 0)
)
    """,
    """
CREATE INDEX idx_ss_sentiment_calibration_day ON ss_sentiment_calibration (day)
    """,
    """
CREATE TABLE ss_snapshot (
    id BIGSERIAL NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    run_date DATE NOT NULL,
    short_interest_pct NUMERIC(5, 4),
    days_to_cover NUMERIC(8, 2),
    float_shares BIGINT,
    avg_volume_14d BIGINT,
    market_cap BIGINT,
    screener_score NUMERIC(5, 4),
    raw_payload JSONB,
    data_quality NUMERIC(3, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT pk_ss_snapshot PRIMARY KEY (id),
    CONSTRAINT ss_snapshot_short_interest_pct_check CHECK (short_interest_pct >= 0 AND short_interest_pct <= 1),
    CONSTRAINT ss_snapshot_days_to_cover_check CHECK (days_to_cover >= 0),
    CONSTRAINT ss_snapshot_screener_score_check CHECK (screener_score >= 0 AND screener_score <= 1),
    CONSTRAINT ss_snapshot_data_quality_check CHECK (data_quality >= 0 AND data_quality <= 1)
)
    """,
    """
CREATE INDEX idx_ss_snapshot_screener_score_desc ON ss_snapshot (screener_score DESC, run_date DESC)
    """,
    """
CREATE INDEX idx_ss_snapshot_ticker_date ON ss_snapshot (ticker, run_date)
    """,
    """
CREATE INDEX idx_ss_snapshot_run_date_desc ON ss_snapshot (run_date DESC)
    """,
    """
CREATE INDEX idx_ss_snapshot_created_at ON ss_snapshot (created_at)
    """,
]


def upgrade() -> None:
    """Drop the six P04 tables (their indexes go with them)."""
    for table in _TABLES:
        op.execute(f"DROP TABLE IF EXISTS {table}")


def downgrade() -> None:
    """Recreate the six tables and their indexes, empty."""
    for statement in _RECREATE_DDL:
        op.execute(statement)

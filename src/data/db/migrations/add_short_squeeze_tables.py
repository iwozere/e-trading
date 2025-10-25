"""
Migration: Add Short Squeeze Detection Pipeline Tables

Creates the four tables needed for the short squeeze detection pipeline:
- ss_snapshot: Weekly screener snapshots
- ss_deep_metrics: Daily deep scan metrics
- ss_alerts: Alert history and cooldown tracking
- ss_ad_hoc_candidates: Ad-hoc candidate management
"""

from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from sqlalchemy import text

from src.data.db.core.database import engine, session_scope
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def create_short_squeeze_tables():
    """Create all short squeeze detection pipeline tables."""

    # SQL for creating tables with proper constraints and indexes
    create_tables_sql = """
    -- Weekly screener snapshots (append-only)
    CREATE TABLE IF NOT EXISTS ss_snapshot (
        id BIGSERIAL PRIMARY KEY,
        ticker VARCHAR(10) NOT NULL,
        run_date DATE NOT NULL,
        short_interest_pct DECIMAL(5,4) CHECK (short_interest_pct >= 0 AND short_interest_pct <= 1),
        days_to_cover DECIMAL(8,2) CHECK (days_to_cover >= 0),
        float_shares BIGINT,
        avg_volume_14d BIGINT,
        market_cap BIGINT,
        screener_score DECIMAL(5,4) CHECK (screener_score >= 0 AND screener_score <= 1),
        raw_payload JSONB,
        data_quality DECIMAL(3,2) CHECK (data_quality >= 0 AND data_quality <= 1),
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    -- Daily deep scan metrics
    CREATE TABLE IF NOT EXISTS ss_deep_metrics (
        id BIGSERIAL PRIMARY KEY,
        ticker VARCHAR(10) NOT NULL,
        date DATE NOT NULL,
        volume_spike DECIMAL(6,2) CHECK (volume_spike >= 0),
        call_put_ratio DECIMAL(6,2) CHECK (call_put_ratio >= 0),
        sentiment_24h DECIMAL(4,3) CHECK (sentiment_24h >= -1 AND sentiment_24h <= 1),
        borrow_fee_pct DECIMAL(5,4) CHECK (borrow_fee_pct >= 0),
        squeeze_score DECIMAL(5,4) CHECK (squeeze_score >= 0 AND squeeze_score <= 1),
        alert_level VARCHAR(10) CHECK (alert_level IN ('LOW', 'MEDIUM', 'HIGH')),
        raw_payload JSONB,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, date)
    );

    -- Alert history and cooldown tracking
    CREATE TABLE IF NOT EXISTS ss_alerts (
        id BIGSERIAL PRIMARY KEY,
        ticker VARCHAR(10) NOT NULL,
        alert_level VARCHAR(10) NOT NULL CHECK (alert_level IN ('LOW', 'MEDIUM', 'HIGH')),
        reason TEXT,
        squeeze_score DECIMAL(5,4) CHECK (squeeze_score >= 0 AND squeeze_score <= 1),
        timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        sent BOOLEAN NOT NULL DEFAULT FALSE,
        cooldown_expires TIMESTAMP WITH TIME ZONE,
        notification_id VARCHAR(50)
    );

    -- Ad-hoc candidate management
    CREATE TABLE IF NOT EXISTS ss_ad_hoc_candidates (
        id BIGSERIAL PRIMARY KEY,
        ticker VARCHAR(10) NOT NULL UNIQUE,
        reason TEXT,
        first_seen TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP WITH TIME ZONE,
        active BOOLEAN NOT NULL DEFAULT TRUE,
        promoted_by_screener BOOLEAN NOT NULL DEFAULT FALSE
    );
    """

    # SQL for creating indexes
    create_indexes_sql = """
    -- Indexes for ss_snapshot
    CREATE INDEX IF NOT EXISTS idx_ss_snapshot_ticker_date ON ss_snapshot(ticker, run_date);
    CREATE INDEX IF NOT EXISTS idx_ss_snapshot_run_date_desc ON ss_snapshot(run_date DESC);
    CREATE INDEX IF NOT EXISTS idx_ss_snapshot_screener_score_desc ON ss_snapshot(screener_score DESC, run_date DESC);
    CREATE INDEX IF NOT EXISTS idx_ss_snapshot_created_at ON ss_snapshot(created_at);

    -- Indexes for ss_deep_metrics
    CREATE INDEX IF NOT EXISTS idx_ss_deep_metrics_ticker_date ON ss_deep_metrics(ticker, date);
    CREATE INDEX IF NOT EXISTS idx_ss_deep_metrics_date_desc ON ss_deep_metrics(date DESC);
    CREATE INDEX IF NOT EXISTS idx_ss_deep_metrics_squeeze_score_desc ON ss_deep_metrics(squeeze_score DESC, date DESC);
    CREATE INDEX IF NOT EXISTS idx_ss_deep_metrics_alert_level ON ss_deep_metrics(alert_level, date DESC);
    CREATE INDEX IF NOT EXISTS idx_ss_deep_metrics_created_at ON ss_deep_metrics(created_at);

    -- Indexes for ss_alerts
    CREATE INDEX IF NOT EXISTS idx_ss_alerts_ticker_cooldown ON ss_alerts(ticker, cooldown_expires);
    CREATE INDEX IF NOT EXISTS idx_ss_alerts_timestamp_desc ON ss_alerts(timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_ss_alerts_alert_level_timestamp ON ss_alerts(alert_level, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_ss_alerts_sent_timestamp ON ss_alerts(sent, timestamp DESC);

    -- Indexes for ss_ad_hoc_candidates
    CREATE INDEX IF NOT EXISTS idx_ss_adhoc_active ON ss_ad_hoc_candidates(active, expires_at);
    CREATE INDEX IF NOT EXISTS idx_ss_adhoc_expires_at ON ss_ad_hoc_candidates(expires_at);
    CREATE INDEX IF NOT EXISTS idx_ss_adhoc_promoted ON ss_ad_hoc_candidates(promoted_by_screener, active);
    """

    try:
        with engine.connect() as conn:
            # Create tables
            _logger.info("Creating short squeeze detection pipeline tables...")
            conn.execute(text(create_tables_sql))

            # Create indexes
            _logger.info("Creating indexes for short squeeze tables...")
            conn.execute(text(create_indexes_sql))

            conn.commit()
            _logger.info("Successfully created all short squeeze tables and indexes")

    except Exception as e:
        _logger.error("Failed to create short squeeze tables: %s", e)
        raise


def drop_short_squeeze_tables():
    """Drop all short squeeze detection pipeline tables."""

    drop_tables_sql = """
    DROP TABLE IF EXISTS ss_ad_hoc_candidates CASCADE;
    DROP TABLE IF EXISTS ss_alerts CASCADE;
    DROP TABLE IF EXISTS ss_deep_metrics CASCADE;
    DROP TABLE IF EXISTS ss_snapshot CASCADE;
    """

    try:
        with engine.connect() as conn:
            _logger.info("Dropping short squeeze detection pipeline tables...")
            conn.execute(text(drop_tables_sql))
            conn.commit()
            _logger.info("Successfully dropped all short squeeze tables")

    except Exception as e:
        _logger.error("Failed to drop short squeeze tables: %s", e)
        raise


def verify_tables_exist():
    """Verify that all short squeeze tables exist."""

    check_tables_sql = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_name LIKE 'ss_%'
    ORDER BY table_name;
    """

    try:
        with engine.connect() as conn:
            result = conn.execute(text(check_tables_sql))
            tables = [row[0] for row in result]

            expected_tables = [
                'ss_ad_hoc_candidates',
                'ss_alerts',
                'ss_deep_metrics',
                'ss_snapshot'
            ]

            missing_tables = set(expected_tables) - set(tables)

            if missing_tables:
                _logger.error("Missing short squeeze tables: %s", missing_tables)
                return False
            else:
                _logger.info("All short squeeze tables exist: %s", tables)
                return True

    except Exception as e:
        _logger.error("Failed to verify short squeeze tables: %s", e)
        return False


if __name__ == "__main__":
    """Run migration when executed directly."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "drop":
        drop_short_squeeze_tables()
    elif len(sys.argv) > 1 and sys.argv[1] == "verify":
        exists = verify_tables_exist()
        sys.exit(0 if exists else 1)
    else:
        create_short_squeeze_tables()
        verify_tables_exist()
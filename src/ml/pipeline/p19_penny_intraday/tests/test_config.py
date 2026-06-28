"""Tests for P19 configuration defaults."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p19_penny_intraday.config import P19Config


def test_create_default_shadow_mode_on():
    cfg = P19Config.create_default()
    assert cfg.shadow_mode is True             # Phase 1 ships log-only
    assert cfg.use_p17_watchlist and cfg.use_gappers


def test_default_filters_penny_regime():
    f = P19Config.create_default().filter_config
    assert f.max_price == 5.0
    assert f.max_float_shares == 25_000_000
    assert f.min_daily_volume == 500_000


def test_feed_reflects_probe_findings():
    feed = P19Config.create_default().feed_config
    # Finnhub real-time price; Polygon delayed volume on a slower cadence
    assert feed.price_provider == "finnhub"
    assert feed.volume_provider == "polygon"
    assert feed.volume_poll_interval_minutes >= feed.poll_interval_minutes


def test_independent_config_instances():
    a = P19Config.create_default()
    b = P19Config.create_default()
    a.manual_pins.append("ABCD")
    assert b.manual_pins == []                 # field(default_factory) — no shared state

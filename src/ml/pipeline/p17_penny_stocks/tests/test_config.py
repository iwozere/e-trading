"""Tests for P17 pipeline configuration."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p17_penny_stocks.config import (
    P17FilterConfig,
    P17PipelineConfig,
    P17ScoringConfig,
    P17ShortSqueezeConfig,
    P17TechnicalConfig,
)


def test_create_default_returns_config():
    config = P17PipelineConfig.create_default()
    assert isinstance(config, P17PipelineConfig)


def test_filter_config_price_range():
    cfg = P17FilterConfig()
    assert cfg.min_price == 0.50
    assert cfg.max_price == 10.00


def test_filter_config_market_cap_range():
    cfg = P17FilterConfig()
    assert cfg.min_market_cap == 30_000_000
    assert cfg.max_market_cap == 2_000_000_000


def test_filter_config_float_range():
    cfg = P17FilterConfig()
    assert cfg.min_float == 5_000_000
    assert cfg.max_float == 50_000_000


def test_scoring_weights_sum_to_one():
    cfg = P17ScoringConfig()
    total = (
        cfg.weight_momentum
        + cfg.weight_volume
        + cfg.weight_technical
        + cfg.weight_fundamentals
        + cfg.weight_catalyst
        + cfg.weight_short_squeeze
        + cfg.weight_accumulation
    )
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"


def test_scoring_config_tier_thresholds_ordered():
    cfg = P17ScoringConfig()
    assert cfg.tier_a_min_score > cfg.tier_b_min_score > cfg.tier_c_min_score > 0


def test_short_squeeze_thresholds_ordered():
    cfg = P17ShortSqueezeConfig()
    assert cfg.si_moderate_threshold < cfg.si_high_threshold < cfg.si_extreme_threshold


def test_technical_config_defaults():
    cfg = P17TechnicalConfig()
    assert cfg.rvol_strong_threshold == 3.0
    assert cfg.bb_period == 20
    assert cfg.atr_period == 14


def test_default_config_universe_excludes_etfs():
    config = P17PipelineConfig.create_default()
    assert config.universe_config.exclude_etfs is True
    assert config.universe_config.exclude_test_issues is True

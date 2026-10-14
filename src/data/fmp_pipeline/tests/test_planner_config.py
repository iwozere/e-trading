"""Tests for the planner configuration loader (spec Appendix B, plan WP-4)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.data.fmp_pipeline.planner.config import DEFAULT_CONFIG_PATH, PlanConfig, load_config


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "fmp.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_follow_the_spec():
    config = PlanConfig()
    assert config.rate_limit_per_min == 600
    assert (config.bandwidth.cap_gb, config.bandwidth.safety, config.bandwidth.basis) == (50.0, 0.90, "decoded")
    assert config.intraday.max_rows is None and config.intraday.bars_per_day == 7
    assert config.universe.include_otc is False
    assert config.priorities == ("P0", "P1")


def test_repository_config_file_loads():
    assert DEFAULT_CONFIG_PATH.exists()
    config = load_config()
    assert config.dump_id and config.bandwidth.basis in ("wire", "decoded")


def test_overrides_merge_into_defaults(tmp_path: Path):
    config = load_config(_write(tmp_path, """
fmp:
  dump_id: "2027-01"
  daily_from: 1995-01-01
  priorities: [P0]
  bandwidth: {basis: wire, cap_gb: 60}
  intraday: {max_rows: 5000, earliest: "2005-01-03", tiers: [A, B]}
  universe: {include_otc: true, tier_a: [SPY]}
"""))
    assert config.dump_id == "2027-01"
    assert config.daily_from == date(1995, 1, 1)
    assert config.priorities == ("P0",)
    assert config.bandwidth.basis == "wire" and config.bandwidth.cap_gb == 60.0
    assert config.bandwidth.safety == 0.90  # untouched sibling keeps its default
    assert config.intraday.max_rows == 5000 and config.intraday.earliest == date(2005, 1, 3)
    assert config.intraday.tiers == ("A", "B")
    assert config.universe.include_otc is True and config.universe.tier_a == ("SPY",)


def test_earliest_keyword_means_unset(tmp_path: Path):
    assert load_config(_write(tmp_path, "fmp:\n  intraday: {earliest: earliest}\n")).intraday.earliest is None


@pytest.mark.parametrize("text,message", [
    ("fmp:\n  dump_idd: x\n", "unknown setting fmp.dump_idd"),
    ("fmp:\n  bandwidth: {cap: 1}\n", "unknown setting fmp.bandwidth.cap"),
    ("fmp:\n  bandwidth: 5\n", "must be a mapping"),
    ("fmp:\n  priorities: P0\n", "expected a list"),
    ("fmp:\n  universe: {include_otc: 'no'}\n", "expected true/false"),
    ("fmp:\n  bandwidth: {basis: gzip}\n", "basis must be"),
    ("fmp:\n  daily_from: not-a-date\n", "daily_from"),
    ("- just a list\n", "top-level 'fmp:'"),
])
def test_invalid_settings_raise(tmp_path: Path, text: str, message: str):
    with pytest.raises(ValueError, match=message):
        load_config(_write(tmp_path, text))


def test_missing_explicit_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_missing_default_file_gives_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import src.data.fmp_pipeline.planner.config as module

    monkeypatch.setattr(module, "DEFAULT_CONFIG_PATH", tmp_path / "absent.yaml")
    assert load_config() == PlanConfig()


def test_empty_file_gives_defaults(tmp_path: Path):
    assert load_config(_write(tmp_path, "")) == PlanConfig()

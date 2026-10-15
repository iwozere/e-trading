"""
Pipeline configuration (spec Appendix B, §6.3, §6.5, §7.2, §7.5), loaded from ``config/fmp/fmp.yaml``.

Every field has a default, so a missing file or a missing key falls back to the spec's values. Numbers the paid
probe has not measured yet (intraday row cap, bars per day, intraday bytes per row) default to ``None`` or to an
explicitly *assumed* value; the planner reports which projections rest on assumptions.

The file lives in ``config/fmp/`` rather than at ``config/fmp.yaml`` because the repository's ``.gitignore`` only
un-ignores ``config/*/*.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "fmp" / "fmp.yaml"


@dataclass(frozen=True)
class BandwidthConfig:
    """The rolling bandwidth cap (spec §6.3.4)."""

    cap_gb: float = 50.0
    safety: float = 0.90
    basis: str = "decoded"
    window_days: int = 30


@dataclass(frozen=True)
class IntradayConfig:
    """Intraday download settings (spec §6.5). ``max_rows`` and ``bars_per_day`` come from the paid probe."""

    interval: str = "1hour"
    tiers: tuple[str, ...] = ("A", "B", "C", "D", "E")
    earliest: Optional[date] = None
    max_rows: Optional[int] = None
    bars_per_day: int = 7
    safety_factor: float = 1.25
    truncation_ratio: float = 0.95
    delisted_margin_days: int = 30
    assumed_max_rows: int = 2000
    wire_bytes_per_row: float = 18.0
    decoded_bytes_per_row: float = 110.0


@dataclass(frozen=True)
class UniverseConfig:
    """Universe rules (spec §7.2). Exchange codes are compared upper-case with spaces, ``-`` and ``_`` removed."""

    include_exchanges: tuple[str, ...] = ("NASDAQ", "NYSE", "AMEX", "NYSEAMERICAN", "NYSEARCA", "ARCA", "BATS",
                                          "CBOE", "CBOEBZX")
    otc_exchanges: tuple[str, ...] = ("OTC", "PNK", "OTCMKTS", "OTCBB", "OTCQB", "OTCQX", "PINK")
    include_otc: bool = False
    keep_unresolved: bool = False
    tier_a: tuple[str, ...] = ("SPY", "QQQ", "IWM", "DIA", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE",
                               "XLU", "XLV", "XLY", "^GSPC", "^IXIC", "^DJI", "^VIX", "^RUT")
    indices: tuple[str, ...] = ("^GSPC", "^IXIC", "^DJI", "^VIX", "^RUT")


@dataclass(frozen=True)
class FlagConfig:
    """Suspect-data flag thresholds (spec §7.5). Defaults are the spec's; calibrate and record in decisions.md."""

    spike_min_return: float = 0.35
    spike_z: float = 8.0
    spike_lookback: int = 60
    spike_min_history: int = 20
    spike_revert_ratio: float = 0.5
    event_window: int = 2
    jump_ratios: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 10.0, 1.5, 0.5, 1 / 3, 0.25, 0.2, 0.1, 2 / 3)
    jump_tolerance: float = 0.03
    stale_bars_daily: int = 5
    stale_bars_intraday: int = 35


@dataclass(frozen=True)
class BuildConfig:
    """``fmp build`` settings (spec §7). Intraday time zone and label are [VERIFY] by the paid probe (§8.4)."""

    intraday_tz: str = "America/New_York"
    intraday_label: str = "start"
    segment_reused_tickers: bool = False
    zstd_level: int = 6
    row_group_rows: int = 100_000
    batch_rows: int = 50_000
    schema_sample_files: int = 500
    flags: FlagConfig = field(default_factory=FlagConfig)


@dataclass(frozen=True)
class AdapterConfig:
    """
    The DataManager adapter (spec §9, plan WP-8). Off by default: turning it on is a deliberate act.

    ``data_root`` empty means ``DATA_CACHE_DIR/fmp-data``.
    """

    enabled: bool = False
    data_root: str = ""
    policy: str = "fmp_first"
    timeframes: tuple[str, ...] = ("1d", "1h")
    cache_entries: int = 256


@dataclass(frozen=True)
class IncrementsConfig:
    """
    The yfinance increment layer (spec §10, plan WP-9).

    ``writer_host`` must name the one machine allowed to ingest (owner decision Q4: the Pi); ingestion refuses to
    run anywhere else, and while it is empty.
    """

    writer_host: str = ""
    timeframes: tuple[str, ...] = ("1d", "1h")
    overlap_days: int = 5
    verify_tolerance: float = 0.02
    batch_size: int = 50
    pause_s: float = 2.0
    rate_limit_wait_s: float = 60.0
    intraday_warn_days: int = 700
    intraday_max_days: int = 729
    flag_context_bars: int = 60


@dataclass(frozen=True)
class PlanConfig:
    """Everything ``fmp plan`` needs."""

    dump_id: str = "2026-10"
    base_url: str = "https://financialmodelingprep.com/stable/"
    rate_limit_per_min: int = 600
    workers: int = 12
    priorities: tuple[str, ...] = ("P0", "P1")
    daily_from: date = date(2000, 1, 1)
    daily_window_years: int = 5
    calendar_window_days: int = 90
    page_size: int = 100
    max_pages: int = 2000
    statement_limit: int = 200
    statement_periods: tuple[str, ...] = ("annual", "quarter")
    exchanges: tuple[str, ...] = ("NYSE", "NASDAQ")
    economic_indicators: tuple[str, ...] = ("GDP", "realGDP", "CPI", "inflationRate", "federalFunds",
                                            "unemploymentRate", "initialClaims", "retailSales",
                                            "consumerSentiment", "industrialProductionTotalIndex")
    reconciliation_size: int = 300
    reconciliation_fixed: tuple[str, ...] = ("AAPL", "MSFT", "TSLA", "AMZN", "NVDA", "GOOGL", "JPM", "XOM", "KO",
                                             "T", "GE", "BRK-B")
    vintage_gb: float = 12.0
    build_temp_gb: float = 10.0
    archive_overhead: float = 1.25
    bandwidth: BandwidthConfig = field(default_factory=BandwidthConfig)
    intraday: IntradayConfig = field(default_factory=IntradayConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    build: BuildConfig = field(default_factory=BuildConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    increments: IncrementsConfig = field(default_factory=IncrementsConfig)
    #: Apply the settings measured by ``fmp probe-paid`` (and ``fmp calibrate-usage``) over this file's values.
    use_probe_results: bool = True


def _coerce(target: Any, value: Any) -> Any:
    """Convert a YAML value to the type of the default it replaces."""
    if isinstance(target, tuple):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"expected a list, got {value!r}")
        return tuple(value)
    if isinstance(target, bool):
        if not isinstance(value, bool):
            raise ValueError(f"expected true/false, got {value!r}")
        return value
    if isinstance(target, date) or (target is None and isinstance(value, (date, str))):
        if value in (None, "earliest"):
            return None
        return value if isinstance(value, date) else date.fromisoformat(str(value))
    if isinstance(target, float) and isinstance(value, (int, float)):
        return float(value)
    return value


def _merge(base: Any, overrides: dict[str, Any], where: str) -> Any:
    """Return ``base`` (a dataclass) with ``overrides`` applied; unknown keys raise ``ValueError``."""
    known = {f.name: f for f in fields(base)}
    changes: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in known:
            raise ValueError(f"unknown setting {where}{key}")
        current = getattr(base, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"{where}{key} must be a mapping")
            changes[key] = _merge(current, value, f"{where}{key}.")
        else:
            try:
                changes[key] = _coerce(current, value)
            except ValueError as exc:
                raise ValueError(f"{where}{key}: {exc}") from exc
    return replace(base, **changes)


def merge_settings(config: PlanConfig, overrides: dict[str, Any], where: str) -> PlanConfig:
    """
    Return ``config`` with a partial ``fmp:``-shaped mapping applied, validated like the YAML file.

    Args:
        config: Base configuration.
        overrides: Partial mapping (nested sections as dicts).
        where: Prefix for error messages.

    Raises:
        ValueError: Unknown keys or wrongly typed values.
    """
    return _merge(config, overrides, where)


def build_config_from_dict(data: dict[str, Any]) -> BuildConfig:
    """
    Rebuild a ``BuildConfig`` from its ``asdict`` form (as stored in a vintage's ``build_info.json``).

    Lists become tuples; unknown keys raise ``ValueError`` like the YAML loader.
    """
    return _merge(BuildConfig(), dict(data), "build_info.build_config.")


def load_config(path: Optional[Path] = None) -> PlanConfig:
    """
    Load the planner configuration.

    Args:
        path: YAML file with a top-level ``fmp:`` mapping. Defaults to ``config/fmp/fmp.yaml``; a missing default
            file yields the built-in defaults, a missing explicit file raises.

    Returns:
        The configuration.

    Raises:
        FileNotFoundError: An explicit ``path`` does not exist.
        ValueError: Unknown keys or values of the wrong type.
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
        if not path.exists():
            return PlanConfig()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = data.get("fmp", {}) if isinstance(data, dict) else None
    if not isinstance(section, dict):
        raise ValueError(f"{path}: expected a top-level 'fmp:' mapping")
    config = _merge(PlanConfig(), section, "fmp.")
    if config.bandwidth.basis not in ("wire", "decoded"):
        raise ValueError("fmp.bandwidth.basis must be 'wire' or 'decoded'")
    if config.adapter.policy != "fmp_first":
        raise ValueError("fmp.adapter.policy must be 'fmp_first' (the only policy, spec §9.2)")
    if config.build.intraday_label not in ("start", "end"):
        raise ValueError("fmp.build.intraday_label must be 'start' or 'end'")
    return config

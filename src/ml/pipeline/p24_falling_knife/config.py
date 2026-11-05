"""
P24 Falling Knife — configuration.

Every threshold of the detector, exits, universe filter, cost model and backtest window lives here (spec
``docs/pipeline-specification.md`` §4–§7, §14). Code elsewhere never hard-codes these numbers: it receives a
``TimeframeParams`` (one instance per timeframe) and derives variants with ``dataclasses.replace``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_DIR = RESULTS_ROOT / "p24_falling_knife"
BACKTEST_DIR = RESULTS_DIR / "backtest"
STATE_DIR = RESULTS_DIR / "_state"

DAILY = "daily"
WEEKLY = "weekly"


@dataclass(frozen=True)
class TimeframeParams:
    """Detector and exit parameters for one timeframe (spec §4, §4.9, §5.3)."""

    name: str

    # §4.1 indicators
    adx_period: int = 14
    ppo_fast: int = 12
    ppo_slow: int = 26
    ppo_signal: int = 9
    cci_period: int = 20

    # §4.2 reference peak and drop
    zz_pct: float = 0.25
    peak_lookback: int = 252
    min_drop: float = 0.40
    max_bars_since_low: int = 5

    # §4.4–§4.6 window (bars) for the local ADX peak, PPO trough and W minimum
    extreme_window: int = 40

    # §4.4 ADX mountain bulge
    adx_min_peak: float = 30.0
    adx_min_rise: float = 8.0
    adx_max_bars_after_peak: int = 10
    adx_min_decline: float = 0.5
    slope_bars: int = 3
    adx_flat_slope: float = 0.25
    adx_floor_margin: float = 8.0

    # §4.5 PPO valley bulge
    hist_near_zero_frac: float = 0.5
    ppo_max_bars_after_trough: int = 10
    ppo_range_bars: int = 504
    ppo_lower_frac: float = 2.0 / 3.0

    # §4.6 channel width
    w_narrow: float = 0.60
    max_xy_offset: int = 5
    max_bars_after_wmin: int = 10
    w_eps: float = 0.02
    max_low_to_wmin: int = 10
    w_rearm: float = 0.30

    # §4.7 intermediate support and CCI divergence
    support_bounce_pct: float = 0.08
    cci_trough_window: int = 2
    cci_stretch: float = -100.0
    cci_div_min: float = 25.0
    price_div_min: float = 0.02

    # §3.3 suspect data cliff
    cliff_drop: float = 0.30
    cliff_volume_ratio: float = 1.5
    cliff_volume_bars: int = 20
    jump_volume_ratio: float = 3.0
    splice_ratio: float = 4.0

    # §4.9 exits and §5.3 shakeout
    cci_exit: float = 200.0
    cci_bear_div_min: float = 25.0
    max_hold_bars: int = 126
    hard_stop_pct: Optional[float] = None
    use_shakeout_rule: bool = True
    cci_shakeout_margin: float = 10.0
    w_widen: float = 0.25

    # Ablation switches (§7.5); the drop + freshness rule is always on.
    require_direction: bool = True
    require_adx: bool = True
    require_ppo: bool = True
    require_channel: bool = True
    require_cci: bool = True
    ppo_lower_third: bool = True

    # §3.4 history needed before a bar may signal
    min_history_bars: int = 300

    def with_overrides(self, **overrides: object) -> "TimeframeParams":
        """
        Return a copy with some fields replaced (calibration grid, ablations).

        Args:
            **overrides: Field names and new values.

        Returns:
            New frozen instance.
        """
        return replace(self, **overrides)  # type: ignore[arg-type]


DAILY_PARAMS = TimeframeParams(name=DAILY)

WEEKLY_PARAMS = TimeframeParams(
    name=WEEKLY,
    extreme_window=20,
    peak_lookback=104,
    max_bars_since_low=8,
    adx_max_bars_after_peak=6,
    slope_bars=2,
    ppo_max_bars_after_trough=6,
    ppo_range_bars=260,
    max_xy_offset=3,
    max_bars_after_wmin=6,
    max_low_to_wmin=6,
    support_bounce_pct=0.10,
    cci_trough_window=1,
    max_hold_bars=52,
    min_history_bars=80,
)

PARAMS_BY_TIMEFRAME: Dict[str, TimeframeParams] = {DAILY: DAILY_PARAMS, WEEKLY: WEEKLY_PARAMS}


@dataclass(frozen=True)
class UniverseParams:
    """Point-in-time universe filters (spec §3.4); evaluated on daily bars."""

    min_price: float = 1.00
    min_dollar_volume: float = 1_000_000.0
    dollar_volume_bars: int = 20
    min_nonzero_volume_frac: float = 0.90
    nonzero_volume_bars: int = 300
    include_adr: bool = False


@dataclass(frozen=True)
class CostModel:
    """Per-side cost tiers by price (spec §7.2): ``(min_price, fraction)``, highest tier first."""

    tiers: Tuple[Tuple[float, float], ...] = ((10.0, 0.0015), (5.0, 0.0030), (1.0, 0.0075), (0.0, 0.0200))
    multiplier: float = 1.0

    def per_side(self, price: float) -> float:
        """
        Return the one-way cost fraction for a fill at ``price``.

        Args:
            price: Fill price before costs.

        Returns:
            Cost as a fraction of the price, scaled by ``multiplier``.
        """
        for floor, cost in self.tiers:
            if price >= floor:
                return cost * self.multiplier
        return self.tiers[-1][1] * self.multiplier


@dataclass(frozen=True)
class BacktestWindow:
    """Signal-date windows (spec §14.1). Indicator warm-up may use earlier data."""

    headline_start: date = date(2021, 1, 1)
    train_end: date = date(2023, 12, 31)
    holdout_start: date = date(2024, 1, 1)
    upper_bound_start: date = date(2004, 1, 1)
    upper_bound_end: date = date(2020, 12, 31)
    data_start: date = date(2000, 1, 1)
    # §7.4 / §14: years whose estimated delisted coverage is below the threshold are "optimistic upper bound" only.
    coverage_threshold: float = 0.50
    coverage_reference_years: Tuple[int, int] = (2021, 2025)


@dataclass(frozen=True)
class PortfolioParams:
    """Portfolio layer (spec §6.3)."""

    max_positions: int = 20
    liquidity_cap_frac: float = 0.05
    init_cash: float = 100_000.0


@dataclass(frozen=True)
class P24Config:
    """Everything one backtest run needs."""

    daily: TimeframeParams = DAILY_PARAMS
    weekly: TimeframeParams = WEEKLY_PARAMS
    universe: UniverseParams = field(default_factory=UniverseParams)
    costs: CostModel = field(default_factory=CostModel)
    window: BacktestWindow = field(default_factory=BacktestWindow)
    portfolio: PortfolioParams = field(default_factory=PortfolioParams)
    random_control_draws: int = 200
    seed: int = 24

    def params(self, timeframe: str) -> TimeframeParams:
        """
        Return the parameters for a timeframe.

        Args:
            timeframe: ``"daily"`` or ``"weekly"``.

        Returns:
            The matching ``TimeframeParams``.

        Raises:
            ValueError: Unknown timeframe.
        """
        if timeframe == DAILY:
            return self.daily
        if timeframe == WEEKLY:
            return self.weekly
        raise ValueError(f"unknown timeframe {timeframe!r}")


# Ablation variants (spec §7.5): name -> overrides applied on top of the timeframe's params.
VARIANTS: Dict[str, Dict[str, object]] = {
    "drop_only": dict(require_direction=False, require_adx=False, require_ppo=False, require_channel=False,
                      require_cci=False, ppo_lower_third=False),
    "drop_adx": dict(require_ppo=False, require_channel=False, require_cci=False, ppo_lower_third=False),
    "drop_adx_ppo": dict(require_channel=False, require_cci=False),
    "drop_adx_ppo_channel": dict(require_cci=False),
    "full": {},
    "full_no_lower_third": dict(ppo_lower_third=False),
    "full_hard_stop_30": dict(hard_stop_pct=0.30),
    "full_no_shakeout_rule": dict(use_shakeout_rule=False),
}

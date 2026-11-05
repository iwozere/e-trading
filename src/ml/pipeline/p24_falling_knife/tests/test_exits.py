"""Tests for P24 trade simulation and exits."""

from __future__ import annotations

from typing import Optional, Tuple

import pytest

from src.ml.pipeline.p24_falling_knife.config import DAILY_PARAMS, CostModel
from src.ml.pipeline.p24_falling_knife.detector import build_context, detect
from src.ml.pipeline.p24_falling_knife.exits import (
    BEARISH_DIVERGENCE,
    CCI_OVERBOUGHT,
    DELISTED,
    END_OF_DATA,
    HARD_STOP,
    SHAKEOUT_FAILED,
    TIME_STOP,
    Trade,
    simulate_trade,
)
from src.ml.pipeline.p24_falling_knife.tests.helpers import DEEP_KNIFE_BASE, knife_bars

COSTS = CostModel()


def _trade(tail: Tuple[Tuple[int, float], ...], truncate: Optional[int] = None, delisted: bool = False,
           **overrides: object) -> Trade:
    params = DAILY_PARAMS.with_overrides(**overrides)
    bars = knife_bars(DEEP_KNIFE_BASE + tail)
    if truncate is not None:
        bars = bars.iloc[:truncate]
    ctx = build_context(bars, params)
    signals, _ = detect(ctx, params)
    assert len(signals) == 1
    trade = simulate_trade(ctx, signals[0], params, COSTS, delisted=delisted)
    assert trade is not None
    return trade


def test_cci_overbought_exits_next_open_with_costs() -> None:
    params = DAILY_PARAMS
    bars = knife_bars(DEEP_KNIFE_BASE + ((4, 52.0), (16, 52.5), (2, 62.0), (20, 62.0)))
    ctx = build_context(bars, params)
    signal = detect(ctx, params)[0][0]
    trade = simulate_trade(ctx, signal, params, COSTS)
    assert trade is not None
    assert trade.exit_reason == CCI_OVERBOUGHT
    assert ctx.cci[trade.exit_bar - 1] >= params.cci_exit
    assert trade.entry_bar == signal.t + 1
    assert trade.entry_price == ctx.open[trade.entry_bar]
    assert trade.exit_price == ctx.open[trade.exit_bar]
    expected = (trade.exit_price * (1 - COSTS.per_side(trade.exit_price))) / \
               (trade.entry_price * (1 + COSTS.per_side(trade.entry_price))) - 1
    assert trade.net_return == pytest.approx(expected)
    assert trade.net_return < trade.gross_return


def test_bearish_divergence_exit() -> None:
    trade = _trade(((12, 62.0), (6, 56.0), (25, 64.5), (10, 60.0)))
    assert trade.exit_reason == BEARISH_DIVERGENCE
    assert trade.net_return > 0


def test_shakeout_with_shallower_cci_is_held() -> None:
    trade = _trade(((5, 54.0), (4, 48.0), (30, 60.0)))
    assert trade.shakeout_seen
    assert trade.exit_reason != SHAKEOUT_FAILED


def test_crash_below_signal_low_fails_shakeout_test() -> None:
    trade = _trade(((5, 55.0), (3, 38.0), (30, 42.0)))
    assert trade.exit_reason == SHAKEOUT_FAILED
    assert trade.net_return < 0


def test_shakeout_rule_can_be_switched_off() -> None:
    trade = _trade(((5, 55.0), (3, 38.0), (30, 42.0)), use_shakeout_rule=False, hard_stop_pct=0.10)
    assert trade.exit_reason == HARD_STOP


def test_time_stop() -> None:
    trade = _trade(((60, 75.0),), max_hold_bars=5)
    assert trade.exit_reason == TIME_STOP
    assert trade.bars_held == 6


def test_end_of_data_and_delisting() -> None:
    open_trade = _trade(((60, 75.0),))
    assert open_trade.exit_reason == END_OF_DATA
    gone = _trade(((60, 75.0),), truncate=open_trade.entry_bar + 4, delisted=True)
    assert gone.exit_reason == DELISTED
    assert gone.delisted_in_trade
    assert gone.exit_bar == open_trade.entry_bar + 3


def test_mae_and_mfe_bracket_the_path() -> None:
    trade = _trade(((5, 54.0), (4, 48.0), (30, 60.0)))
    assert trade.max_adverse_excursion < 0 < trade.max_favorable_excursion


def test_cost_tiers() -> None:
    assert COSTS.per_side(50.0) == pytest.approx(0.0015)
    assert COSTS.per_side(7.0) == pytest.approx(0.0030)
    assert COSTS.per_side(2.0) == pytest.approx(0.0075)
    assert COSTS.per_side(0.5) == pytest.approx(0.02)
    assert CostModel(multiplier=2.0).per_side(50.0) == pytest.approx(0.003)

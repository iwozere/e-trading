"""
P24 Falling Knife — trade simulation and exits (spec §4.9, §5.1, §5.3).

Entry fills at the open of the bar after the signal. Exit rules are checked on each closed bar; the exit fills at the
next bar's open. Costs are per side and tiered by price (``CostModel``). Returns exclude dividends (split-adjusted
prices), which is conservative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.ml.pipeline.p24_falling_knife.config import CostModel, TimeframeParams
from src.ml.pipeline.p24_falling_knife.detector import BarEval, SeriesContext
from src.ml.pipeline.p24_falling_knife.geometry import HIGH, Swing, window_max, window_min

DELISTED = "delisted"
END_OF_DATA = "end_of_data"
SHAKEOUT_FAILED = "shakeout_failed"
CCI_OVERBOUGHT = "cci_overbought"
BEARISH_DIVERGENCE = "bearish_divergence"
TIME_STOP = "time_stop"
HARD_STOP = "hard_stop"


@dataclass
class Trade:
    """One simulated trade (spec §6.2, security fields are added by the backtest)."""

    signal_bar: int
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    exit_reason: str
    bars_held: int
    gross_return: float
    net_return: float
    max_adverse_excursion: float
    max_favorable_excursion: float
    shakeout_seen: bool
    shakeout_widening: bool
    delisted_in_trade: bool
    suspect_in_trade: bool = False


def simulate_trade(ctx: SeriesContext, signal: BarEval, params: TimeframeParams, costs: CostModel,
                   delisted: bool = False) -> Optional[Trade]:
    """
    Simulate one trade from a signal.

    Args:
        ctx: Series context of the security.
        signal: The signal bar.
        params: Timeframe parameters (exit settings).
        costs: Cost model.
        delisted: The series ends because the security delisted (not because the data ends).

    Returns:
        ``Trade``, or ``None`` if there is no bar to enter on.
    """
    entry_bar = signal.t + 1
    if entry_bar >= ctx.n or not np.isfinite(ctx.open[entry_bar]) or ctx.open[entry_bar] <= 0:
        return None
    entry_open = float(ctx.open[entry_bar])
    l2_close = float(ctx.close[signal.l2])
    highs = [s for s in ctx.support_swings if s.kind == HIGH and s.bar >= entry_bar]
    below_start: Optional[int] = None
    shakeout_seen = False
    shakeout_widening = False
    reason: Optional[str] = None
    last = ctx.n - 1
    i = entry_bar
    for i in range(entry_bar, ctx.n):
        close = ctx.close[i]
        if params.use_shakeout_rule and np.isfinite(signal.cci_3):
            if close < l2_close:
                if below_start is None:
                    below_start = i
                    shakeout_seen = True
                    shakeout_widening = shakeout_widening or bool(ctx.w[i] - signal.w_min >= params.w_widen)
                cci_s = window_min(ctx.cci, below_start, i + 1)
                if not cci_s > signal.cci_3 + params.cci_shakeout_margin:
                    reason = SHAKEOUT_FAILED
            else:
                below_start = None
        if reason is None and ctx.cci[i] >= params.cci_exit:
            reason = CCI_OVERBOUGHT
        if reason is None and _bearish_divergence(ctx, i, highs, params):
            reason = BEARISH_DIVERGENCE
        if reason is None and i - entry_bar >= params.max_hold_bars:
            reason = TIME_STOP
        if reason is None and params.hard_stop_pct is not None and close < entry_open * (1.0 - params.hard_stop_pct):
            reason = HARD_STOP
        if reason is not None:
            break
    if reason is not None and i < last and np.isfinite(ctx.open[i + 1]):
        exit_bar, exit_price = i + 1, float(ctx.open[i + 1])
    else:
        exit_bar, exit_price = i, float(ctx.close[i])
        if reason is None or i >= last:
            reason = DELISTED if delisted else (reason or END_OF_DATA)
    # The cost tier depends on the price as traded, not on the split-adjusted price.
    entry_eff = entry_open * (1.0 + costs.per_side(entry_open * float(ctx.raw_scale[entry_bar])))
    exit_eff = exit_price * (1.0 - costs.per_side(exit_price * float(ctx.raw_scale[exit_bar])))
    return Trade(
        signal_bar=signal.t,
        entry_bar=entry_bar,
        exit_bar=exit_bar,
        entry_price=entry_open,
        exit_price=exit_price,
        exit_reason=reason,
        bars_held=exit_bar - entry_bar,
        gross_return=exit_price / entry_open - 1.0,
        net_return=exit_eff / entry_eff - 1.0,
        max_adverse_excursion=window_min(ctx.low, entry_bar, exit_bar + 1) / entry_open - 1.0,
        max_favorable_excursion=window_max(ctx.high, entry_bar, exit_bar + 1) / entry_open - 1.0,
        shakeout_seen=shakeout_seen,
        shakeout_widening=shakeout_widening,
        delisted_in_trade=delisted and exit_bar >= last,
        suspect_in_trade=bool(ctx.cliff[entry_bar + 1:exit_bar + 1].any()),
    )


def _bearish_divergence(ctx: SeriesContext, i: int, highs: List[Swing], params: TimeframeParams) -> bool:
    """
    Bearish CCI divergence at bar ``i`` (§4.9 rule 4).

    The latest swing high since entry confirmed before ``i`` is H1. Price has since closed above H1, CCI turned down
    on bar ``i``, and the highest CCI since price passed H1 is at least ``cci_bear_div_min`` below H1's CCI peak.
    """
    known = [s for s in highs if s.confirmed_bar < i]
    if not known or i < 1:
        return False
    h1 = known[-1]
    above = np.flatnonzero(ctx.close[h1.confirmed_bar + 1:i + 1] > h1.price)
    if above.size == 0:
        return False
    first_above = h1.confirmed_bar + 1 + int(above[0])
    j = params.cci_trough_window
    cci_h1 = window_max(ctx.cci, h1.bar - j, min(h1.bar + j, i) + 1)
    cci_run = window_max(ctx.cci, first_above, i + 1)
    return bool(ctx.cci[i] < ctx.cci[i - 1] and cci_run <= cci_h1 - params.cci_bear_div_min)

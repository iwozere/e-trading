"""
P24 Falling Knife — signal detector (spec §4.2–§4.8).

``build_context`` pre-computes everything that is vectorizable (indicators, channel width, swing points, reference
peaks, eligibility). ``evaluate_bar`` checks one closed bar ``t`` against every condition using only bars ``<= t``.
``detect`` walks the bars with the one-signal-per-knife state machine, including the re-arm rule for a second leg
of the same decline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.ml.pipeline.p24_falling_knife.config import TimeframeParams
from src.ml.pipeline.p24_falling_knife.geometry import (
    Swing,
    channel_width,
    low_since_peak,
    nan_argmax,
    nan_argmin,
    reference_peak_bars,
    swing_lows_between,
    window_min,
    zigzag,
)
from src.ml.pipeline.p24_falling_knife.indicators import INDICATOR_COLUMNS, compute_indicators


@dataclass
class SeriesContext:
    """Arrays of one security on one timeframe, ready for bar-by-bar evaluation."""

    index: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    adx: np.ndarray
    plus_di: np.ndarray
    minus_di: np.ndarray
    ppo: np.ndarray
    ppo_signal: np.ndarray
    ppo_hist: np.ndarray
    cci: np.ndarray
    w: np.ndarray
    ppo_floor: np.ndarray
    peak_bars: np.ndarray
    support_swings: List[Swing]
    eligible: np.ndarray
    cliff: np.ndarray
    raw_scale: np.ndarray
    low_bars: np.ndarray
    drop: np.ndarray
    _candidates: Dict[Tuple[float, int], np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        """Number of bars."""
        return len(self.close)

    def candidate(self, params: TimeframeParams) -> np.ndarray:
        """
        Bars that are eligible and pass the drop + freshness rule (§4.2), cached per threshold pair.

        Args:
            params: Timeframe parameters (``min_drop``, ``max_bars_since_low``).

        Returns:
            Boolean array.
        """
        key = (params.min_drop, params.max_bars_since_low)
        if key not in self._candidates:
            bars = np.arange(self.n)
            fresh = (self.low_bars >= 0) & (bars - self.low_bars <= params.max_bars_since_low)
            mask = self.eligible & fresh & (self.drop >= params.min_drop) & (self.peak_bars >= 0) & (bars >= 1)
            self._candidates[key] = mask
        return self._candidates[key]


@dataclass
class BarEval:
    """Every §4 condition and metric for one bar that passed the drop + freshness rule."""

    t: int
    p1: int
    l2: int
    drop: float
    seg_start: int
    bar_x: Optional[int] = None
    adx_peak: float = float("nan")
    bar_y: Optional[int] = None
    ppo_trough: float = float("nan")
    ppo_floor: float = float("nan")
    bar_wmin: Optional[int] = None
    w_min: float = float("nan")
    s3: Optional[int] = None
    cci_2: float = float("nan")
    cci_3: float = float("nan")
    direction_ok: bool = False
    adx_ok: bool = False
    ppo_core_ok: bool = False
    ppo_lower_third_ok: bool = False
    channel_ok: bool = False
    cci_ok: bool = False
    suspect_gap: bool = False
    score: float = float("nan")
    knife_seq: int = 1
    missing: List[str] = field(default_factory=list)

    def fires(self, params: TimeframeParams) -> bool:
        """
        Whether this bar is a signal under ``params``' ablation switches.

        Args:
            params: Timeframe parameters (``require_*`` flags, ``ppo_lower_third``).

        Returns:
            True if every required condition holds.
        """
        return not self.failed(params)

    def failed(self, params: TimeframeParams) -> List[str]:
        """
        Names of the required conditions that do not hold.

        Args:
            params: Timeframe parameters.

        Returns:
            Empty list for a signal.
        """
        checks: List[Tuple[bool, bool, str]] = [
            (params.require_direction, self.direction_ok, "direction"),
            (params.require_adx, self.adx_ok, "adx_mountain"),
            (params.require_ppo, self.ppo_core_ok, "ppo_valley"),
            (params.require_ppo and params.ppo_lower_third, self.ppo_lower_third_ok, "ppo_lower_third"),
            (params.require_channel, self.channel_ok, "narrow_channel"),
            (params.require_cci, self.cci_ok, "cci_divergence"),
        ]
        return [name for required, ok, name in checks if required and not ok]


def data_cliffs(close: np.ndarray, volume: np.ndarray, params: TimeframeParams,
                jump_no_split: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Bars that probably carry an unadjusted corporate event rather than a real crash (§3.3).

    Two rules, both keyed on volume because a real crash day trades many times its normal volume:

    - the close fell at least ``cliff_drop`` on volume below ``cliff_volume_ratio`` × the prior median (spin-offs);
    - FMP flagged the bar ``JUMP_NO_SPLIT`` (close ratio near a split ratio, no split recorded) and volume is below
      ``jump_volume_ratio`` × the prior median. A −33% or −50% crash on heavy volume also matches the price rule, so
      the flag alone is not enough;
    - the close rose at least ``splice_ratio``-fold in one bar on volume below ``jump_volume_ratio`` × the prior median
      (bankrupt shares spliced onto post-reorganisation equity, a missed reverse split, a test symbol). Such a bar
      inside a trade marks the trade ``suspect_in_trade``.

    Args:
        close: Closes.
        volume: Volumes.
        params: Timeframe parameters.
        jump_no_split: FMP ``JUMP_NO_SPLIT`` bar flags, or None.

    Returns:
        Boolean array.
    """
    close_s = pd.Series(close)
    volume_s = pd.Series(volume)
    fall = 1.0 - close_s / close_s.shift(1)
    median_volume = volume_s.shift(1).rolling(params.cliff_volume_bars, min_periods=5).median()
    quiet_cliff = (fall >= params.cliff_drop) & (volume_s < params.cliff_volume_ratio * median_volume)
    out = quiet_cliff.fillna(False).to_numpy(dtype=bool, copy=True)
    quiet_jump = (volume_s < params.jump_volume_ratio * median_volume).fillna(True).to_numpy(dtype=bool)
    splice = (close_s / close_s.shift(1) >= params.splice_ratio).fillna(False).to_numpy(dtype=bool)
    out |= splice & quiet_jump
    if jump_no_split is not None:
        out |= np.asarray(jump_no_split, dtype=bool) & quiet_jump
    return out


def build_context(bars: pd.DataFrame, params: TimeframeParams, eligible: Optional[np.ndarray] = None) -> SeriesContext:
    """
    Pre-compute indicators, geometry and eligibility for one security.

    Optional columns of ``bars``: ``eligible`` (point-in-time universe mask, §3.4), ``jump_no_split`` (FMP flag; see
    ``data_cliffs``), ``close_raw`` (as-traded close; gives ``raw_scale`` for the cost tier).

    Args:
        bars: OHLCV frame on a ``DatetimeIndex`` (indicator columns are computed if missing).
        params: Timeframe parameters.
        eligible: Optional universe mask overriding the ``eligible`` column, same length as ``bars``.

    Returns:
        ``SeriesContext``.
    """
    frame = bars if all(c in bars.columns for c in INDICATOR_COLUMNS) else compute_indicators(bars, params)
    close = frame["close"].to_numpy(dtype=np.float64)
    volume = frame["volume"].to_numpy(dtype=np.float64) if "volume" in frame.columns else np.full(len(close), np.nan)
    if eligible is None and "eligible" in frame.columns:
        eligible = frame["eligible"].fillna(False).to_numpy(dtype=bool)
    jumps = frame["jump_no_split"].fillna(False).to_numpy(dtype=bool) if "jump_no_split" in frame.columns else None
    raw_scale = (frame["close_raw"].to_numpy(dtype=np.float64) / close) if "close_raw" in frame.columns else np.ones(len(close))
    width = channel_width(frame["adx"], frame["ppo"], params.ppo_range_bars)
    peak_swings = zigzag(close, params.zz_pct)
    support_swings = zigzag(close, params.support_bounce_pct)
    n = len(close)
    history_ok = np.zeros(n, dtype=bool)
    valid = np.flatnonzero(~np.isnan(close))
    if valid.size:
        history_ok[min(n, int(valid[0]) + params.min_history_bars):] = True
    mask = history_ok if eligible is None else (history_ok & np.asarray(eligible, dtype=bool))
    peak_bars = reference_peak_bars(close, peak_swings, params.peak_lookback)
    low_bars = low_since_peak(close, peak_bars)
    valid_drop = (low_bars >= 0) & (peak_bars >= 0)
    peak_close = close[np.where(valid_drop, peak_bars, 0)]
    low_close = close[np.where(valid_drop, low_bars, 0)]
    with np.errstate(divide="ignore", invalid="ignore"):
        drop = np.where(valid_drop & (peak_close > 0), 1.0 - low_close / peak_close, np.nan)
    return SeriesContext(
        index=pd.DatetimeIndex(frame.index),
        open=frame["open"].to_numpy(dtype=np.float64),
        high=frame["high"].to_numpy(dtype=np.float64),
        low=frame["low"].to_numpy(dtype=np.float64),
        close=close,
        volume=volume,
        adx=frame["adx"].to_numpy(dtype=np.float64),
        plus_di=frame["plus_di"].to_numpy(dtype=np.float64),
        minus_di=frame["minus_di"].to_numpy(dtype=np.float64),
        ppo=frame["ppo"].to_numpy(dtype=np.float64),
        ppo_signal=frame["ppo_signal"].to_numpy(dtype=np.float64),
        ppo_hist=frame["ppo_hist"].to_numpy(dtype=np.float64),
        cci=frame["cci"].to_numpy(dtype=np.float64),
        w=width["w"].to_numpy(dtype=np.float64),
        ppo_floor=width["ppo_floor"].to_numpy(dtype=np.float64),
        peak_bars=peak_bars,
        support_swings=support_swings,
        eligible=mask,
        cliff=data_cliffs(close, volume, params, jumps),
        raw_scale=np.where(np.isfinite(raw_scale) & (raw_scale > 0), raw_scale, 1.0),
        low_bars=low_bars,
        drop=drop,
    )


def evaluate_bar(ctx: SeriesContext, t: int, params: TimeframeParams, seg_start: Optional[int] = None,
                 min_s3_bar: Optional[int] = None) -> Optional[BarEval]:
    """
    Check bar ``t`` against §4.2–§4.7 using only bars ``<= t``.

    Args:
        ctx: Series context.
        t: Bar to evaluate (closed).
        params: Timeframe parameters.
        seg_start: Start (exclusive) of the window for the ADX/PPO/W extremes; default P1. A second leg of the same
            knife starts at its re-arm bar.
        min_s3_bar: Point 3 must lie after this bar; default P1 (after an earlier signal: that signal's low).

    Returns:
        ``BarEval`` if the bar is eligible and passes the drop + freshness rule, else ``None``.
    """
    if t < 1 or not ctx.candidate(params)[t]:
        return None
    p1, l2 = int(ctx.peak_bars[t]), int(ctx.low_bars[t])
    start = p1 if seg_start is None else max(seg_start, p1)
    # ADX/PPO/W extremes are local: one reference peak can carry several mountains (the book's Dow 2008-09 table).
    start = max(start, t - params.extreme_window)
    ev = BarEval(t=t, p1=p1, l2=l2, drop=float(ctx.drop[t]), seg_start=start)
    ev.suspect_gap = bool(ctx.cliff[p1 + 1:l2 + 1].any())
    _check_adx(ctx, t, params, ev)
    _check_ppo(ctx, t, params, ev)
    _check_channel(ctx, t, params, ev)
    _check_cci(ctx, t, params, ev, p1 if min_s3_bar is None else max(min_s3_bar, p1))
    ev.score = _score(ev, params)
    ev.missing = ev.failed(params)
    return ev


def _check_adx(ctx: SeriesContext, t: int, params: TimeframeParams, ev: BarEval) -> None:
    adx = ctx.adx
    x = nan_argmax(adx, ev.seg_start + 1, t + 1)
    if x is None:
        return
    ev.bar_x = x
    ev.adx_peak = float(adx[x])
    ev.direction_ok = bool(ctx.minus_di[x] > ctx.plus_di[x])
    rise = ev.adx_peak - window_min(adx, ev.seg_start, x + 1)
    after = t - x
    declined = adx[t] <= ev.adx_peak - params.adx_min_decline
    slope = (adx[t] - adx[t - params.slope_bars]) / params.slope_bars if t >= params.slope_bars else np.nan
    flat = slope <= params.adx_flat_slope
    ev.adx_ok = bool(ev.adx_peak >= params.adx_min_peak and rise >= params.adx_min_rise
                     and 1 <= after <= params.adx_max_bars_after_peak and (declined or flat)
                     and adx[t] >= params.adx_min_peak - params.adx_floor_margin)


def _check_ppo(ctx: SeriesContext, t: int, params: TimeframeParams, ev: BarEval) -> None:
    ppo, sig, hist = ctx.ppo, ctx.ppo_signal, ctx.ppo_hist
    y = nan_argmin(ppo, ev.seg_start + 1, t + 1)
    if y is None:
        return
    ev.bar_y = y
    ev.ppo_trough = float(ppo[y])
    ev.ppo_floor = float(ctx.ppo_floor[t])
    hist_min = window_min(hist, ev.seg_start + 1, t + 1)
    turning = hist[t] > hist_min and hist[t] > hist[t - 1]
    near_zero = ppo[t] > sig[t] or hist[t] >= params.hist_near_zero_frac * hist_min
    formed = 1 <= t - y <= params.ppo_max_bars_after_trough and ppo[t] >= ppo[t - 1]
    ev.ppo_core_ok = bool(ppo[t] < 0 and turning and near_zero and formed)
    ev.ppo_lower_third_ok = bool(ev.ppo_floor < 0 and ev.ppo_trough <= params.ppo_lower_frac * ev.ppo_floor)


def _check_channel(ctx: SeriesContext, t: int, params: TimeframeParams, ev: BarEval) -> None:
    w = ctx.w
    bar_wmin = nan_argmin(w, ev.seg_start + 1, t + 1)
    if bar_wmin is None:
        return
    ev.bar_wmin = bar_wmin
    ev.w_min = float(w[bar_wmin])
    converge = ev.bar_x is not None and ev.bar_y is not None and abs(ev.bar_x - ev.bar_y) <= params.max_xy_offset
    reached = 1 <= t - bar_wmin <= params.max_bars_after_wmin and w[t] >= w[t - 1] - params.w_eps
    at_low = abs(ev.l2 - bar_wmin) <= params.max_low_to_wmin
    ev.channel_ok = bool(ev.w_min <= params.w_narrow and converge and reached and at_low)


def _check_cci(ctx: SeriesContext, t: int, params: TimeframeParams, ev: BarEval, min_s3_bar: int) -> None:
    lows = swing_lows_between(ctx.support_swings, min_s3_bar, ev.l2, t)
    if not lows:
        return
    s3 = lows[-1].bar
    j = params.cci_trough_window
    ev.s3 = s3
    ev.cci_3 = window_min(ctx.cci, s3 - j, min(s3 + j, t) + 1)
    ev.cci_2 = window_min(ctx.cci, ev.l2 - j, min(ev.l2 + j, t) + 1)
    lower_low = ctx.close[ev.l2] < ctx.close[s3] * (1.0 - params.price_div_min)
    ev.cci_ok = bool(ev.cci_3 <= params.cci_stretch and ev.cci_2 - ev.cci_3 >= params.cci_div_min and lower_low)


def _score(ev: BarEval, params: TimeframeParams) -> float:
    """Ranking score (§4.8); missing parts count as 0."""
    narrow = params.w_narrow - ev.w_min if np.isfinite(ev.w_min) else 0.0
    divergence = (ev.cci_2 - ev.cci_3) / 100.0 if np.isfinite(ev.cci_2) and np.isfinite(ev.cci_3) else 0.0
    return float(narrow + divergence + (ev.drop - params.min_drop))


@dataclass
class _KnifeState:
    p1: int
    t: int
    l2: int
    l2_close: float
    w_min: float
    seq: int
    rearm_bar: Optional[int] = None


def detect(ctx: SeriesContext, params: TimeframeParams, collect_evals: bool = False) -> Tuple[List[BarEval], List[BarEval]]:
    """
    Walk all bars and emit signals with the one-signal-per-knife rule (§4.8).

    After a signal, the same knife (same P1) can signal again only once the channel width has risen ``w_rearm`` above
    the signal's ``W`` minimum (re-arm) and price then makes a close below the signal low. The second leg measures its
    ADX/PPO/W extremes from the widest channel point since the re-arm, and its point 3 must lie after the first
    signal's low. A new reference peak starts a new knife.

    Args:
        ctx: Series context.
        params: Timeframe parameters (incl. ablation switches).
        collect_evals: Also return every evaluated bar (for the forming watchlist and diagnostics).

    Returns:
        ``(signals, evaluated_bars)``; the second list is empty unless ``collect_evals``.
    """
    signals: List[BarEval] = []
    evals: List[BarEval] = []
    state: Optional[_KnifeState] = None
    for t in (int(b) for b in np.flatnonzero(ctx.candidate(params))):
        p1 = int(ctx.peak_bars[t])
        if state is not None and bool(np.any(ctx.peak_bars[state.t + 1:t + 1] != state.p1)):
            state = None
        seg_start: Optional[int] = None
        min_s3: Optional[int] = None
        if state is not None:
            if state.rearm_bar is None:
                hits = np.flatnonzero(ctx.w[state.t + 1:t + 1] >= state.w_min + params.w_rearm)
                if hits.size == 0:
                    continue
                state.rearm_bar = state.t + 1 + int(hits[0])
                if state.rearm_bar == t:
                    continue
            widest = nan_argmax(ctx.w, state.rearm_bar, t + 1)
            seg_start, min_s3 = (state.rearm_bar if widest is None else widest), state.l2
        ev = evaluate_bar(ctx, t, params, seg_start, min_s3)
        if ev is None:
            continue
        if state is not None:
            if not ctx.close[ev.l2] < state.l2_close or ev.l2 <= (state.rearm_bar or 0):
                continue
            ev.knife_seq = state.seq + 1
        if collect_evals:
            evals.append(ev)
        if ev.fires(params):
            signals.append(ev)
            w_ref = ev.w_min if np.isfinite(ev.w_min) else float(ctx.w[t])
            state = _KnifeState(p1=p1, t=t, l2=ev.l2, l2_close=float(ctx.close[ev.l2]), w_min=w_ref, seq=ev.knife_seq)
    return signals, evals

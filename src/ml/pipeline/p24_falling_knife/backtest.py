"""
P24 Falling Knife — historical backtest on the FMP vintage (spec §6).

Per security and timeframe the expensive part (indicators, swing points, channel width) is built once, then every
ablation variant runs its trade simulation on the same context; variants that differ only in exit settings share one
signal walk. The matched-random control (§6.4)
re-enters each full-system signal's knife at random bars that also passed the 40% drop. Securities are processed in
chunks by a process pool; each worker opens the vintage once.

Signal periods (spec §14.1): ``upper_bound`` (survivors-only years, never used for calibration), ``train`` and
``holdout``. Holdout rows are produced only when explicitly requested.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.fmp_pipeline.datamanager_adapter.fmp_store import FmpStore
from src.ml.pipeline.p24_falling_knife.config import VARIANTS, P24Config, TimeframeParams
from src.ml.pipeline.p24_falling_knife.data import load_daily_batch, series_for_timeframe
from src.ml.pipeline.p24_falling_knife.detector import BarEval, SeriesContext, build_context, detect
from src.ml.pipeline.p24_falling_knife.exits import simulate_trade
from src.ml.pipeline.p24_falling_knife.indicators import compute_indicators
from src.ml.pipeline.p24_falling_knife.universe import Security
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

UPPER_BOUND = "upper_bound"
TRAIN = "train"
HOLDOUT = "holdout"
FULL = "full"
CONTROL_WINDOW_AFTER = 20


@dataclass
class ChunkResult:
    """What one worker returns for a chunk of securities."""

    trades: List[Dict[str, Any]] = field(default_factory=list)
    random_control: List[Dict[str, Any]] = field(default_factory=list)
    securities_with_data: int = 0
    errors: List[Tuple[int, str]] = field(default_factory=list)


@dataclass(frozen=True)
class RunSpec:
    """Inputs every worker needs (picklable)."""

    config: P24Config
    timeframes: Tuple[str, ...]
    variants: Tuple[str, ...]
    include_holdout: bool
    as_of: Optional[date]
    # Per-timeframe named overrides used instead of ``config.VARIANTS`` (calibration grid): timeframe -> variant ->
    # overrides. None means the ablation variants.
    variant_overrides: Optional[Dict[str, Dict[str, Dict[str, object]]]] = None
    # Only keep signals in these periods (None: every period the run allows).
    periods: Optional[Tuple[str, ...]] = None

    def overrides(self, variant: str, timeframe: str) -> Dict[str, object]:
        """
        Parameter overrides of a variant on a timeframe.

        Args:
            variant: Variant name.
            timeframe: ``daily`` or ``weekly``.

        Returns:
            Field overrides applied on top of the timeframe's parameters.
        """
        if self.variant_overrides is None:
            return VARIANTS[variant]
        return self.variant_overrides[timeframe][variant]


def period_of(day: date, config: P24Config) -> Optional[str]:
    """
    Signal period label for a signal date (spec §14.1).

    Args:
        day: Signal bar date.
        config: Run configuration (windows).

    Returns:
        ``upper_bound``, ``train``, ``holdout`` or None (outside every window).
    """
    window = config.window
    if window.upper_bound_start <= day <= window.upper_bound_end:
        return UPPER_BOUND
    if window.headline_start <= day <= window.train_end:
        return TRAIN
    if day >= window.holdout_start:
        return HOLDOUT
    return None


def price_bucket(raw_price: float) -> str:
    """Bucket label by as-traded price (spec §7.5)."""
    if raw_price >= 10.0:
        return ">=10"
    if raw_price >= 5.0:
        return "5-10"
    if raw_price >= 1.0:
        return "1-5"
    return "<1"


def _signal_row(security: Security, timeframe: str, variant: str, period: str, ctx: SeriesContext, sig: BarEval,
                bars: pd.DataFrame) -> Dict[str, Any]:
    idx = ctx.index
    dollar = bars["dollar_volume_20"].iloc[sig.t] if "dollar_volume_20" in bars.columns else np.nan
    return {
        "security_id": security.security_id,
        "symbol": security.symbol,
        "timeframe": timeframe,
        "variant": variant,
        "period": period,
        "signal_date": idx[sig.t].date(),
        "peak_date": idx[sig.p1].date(),
        "peak_close": float(ctx.close[sig.p1]),
        "l2_date": idx[sig.l2].date(),
        "l2_close": float(ctx.close[sig.l2]),
        "s3_date": idx[sig.s3].date() if sig.s3 is not None else None,
        "s3_close": float(ctx.close[sig.s3]) if sig.s3 is not None else np.nan,
        "drop_pct": sig.drop,
        "adx_peak": sig.adx_peak,
        "ppo_trough": sig.ppo_trough,
        "ppo_floor": sig.ppo_floor,
        "w_min": sig.w_min,
        "cci_2": sig.cci_2,
        "cci_3": sig.cci_3,
        "score": sig.score,
        "suspect_gap": sig.suspect_gap,
        "knife_seq": sig.knife_seq,
        "dollar_volume_20": float(dollar) if pd.notna(dollar) else np.nan,
    }


def run_security(security: Security, daily: pd.DataFrame, spec: RunSpec, rng_seed: int) -> ChunkResult:
    """
    All timeframes and variants for one security.

    Args:
        security: Vintage security.
        daily: Its P24 daily bars (``data.prepare_daily``).
        spec: Run inputs.
        rng_seed: Seed for this security's random control draws.

    Returns:
        Trades and random-control rows.
    """
    result = ChunkResult()
    config = spec.config
    rng = np.random.default_rng(rng_seed)
    for timeframe in spec.timeframes:
        base = config.params(timeframe)
        bars = series_for_timeframe(daily, timeframe, config.universe, spec.as_of)
        if len(bars) < base.min_history_bars + 2:
            continue
        frame = compute_indicators(bars, base)
        contexts: Dict[Tuple[float, float], SeriesContext] = {}
        walks: Dict[TimeframeParams, List[BarEval]] = {}
        for variant in spec.variants:
            params = base.with_overrides(**spec.overrides(variant, timeframe))
            key = (params.zz_pct, params.support_bounce_pct)
            if key not in contexts:
                contexts[key] = build_context(frame, params)
            ctx = contexts[key]
            detection = params.detection_params()
            if detection not in walks:
                walks[detection] = detect(ctx, params)[0]
            for sig in walks[detection]:
                period = period_of(ctx.index[sig.t].date(), config)
                if period is None or (period == HOLDOUT and not spec.include_holdout):
                    continue
                if spec.periods is not None and period not in spec.periods:
                    continue
                trade = simulate_trade(ctx, sig, params, config.costs, delisted=security.delisted)
                if trade is None:
                    continue
                row = _signal_row(security, timeframe, variant, period, ctx, sig, frame)
                row.update(_trade_fields(ctx, trade))
                result.trades.append(row)
                if variant == FULL and config.random_control_draws > 0:
                    result.random_control.append(_random_control(security, timeframe, period, ctx, sig, params,
                                                                 spec, rng))
    return result


def _trade_fields(ctx: SeriesContext, trade: Any) -> Dict[str, Any]:
    fields = asdict(trade)
    fields["entry_date"] = ctx.index[trade.entry_bar].date()
    fields["exit_date"] = ctx.index[trade.exit_bar].date()
    raw_entry = trade.entry_price * float(ctx.raw_scale[trade.entry_bar])
    fields["entry_price_raw"] = raw_entry
    fields["exit_price_raw"] = trade.exit_price * float(ctx.raw_scale[trade.exit_bar])
    fields["price_bucket"] = price_bucket(raw_entry)
    for key in ("signal_bar", "entry_bar", "exit_bar"):
        fields.pop(key)
    return fields


def _random_control(security: Security, timeframe: str, period: str, ctx: SeriesContext, sig: BarEval,
                    params: TimeframeParams, spec: RunSpec, rng: np.random.Generator) -> Dict[str, Any]:
    """
    Matched-random control for one full-system signal (spec §6.4 item 2).

    Draws ``random_control_draws`` entry bars from the same knife (``(P1, t + 20]``) among bars that are eligible and
    at least ``min_drop`` below P1, and simulates each with the signal's exit rules (same point-3 CCI for the shakeout
    test). Returns the per-draw net returns.
    """
    stop = min(ctx.n - 1, sig.t + CONTROL_WINDOW_AFTER)
    bars = np.arange(sig.p1 + 1, stop + 1)
    deep = ctx.close[bars] <= ctx.close[sig.p1] * (1.0 - params.min_drop)
    pool = bars[deep & ctx.eligible[bars]]
    draws = spec.config.random_control_draws
    returns: List[float] = []
    if pool.size:
        for b in rng.choice(pool, size=draws, replace=True):
            fake = BarEval(t=int(b), p1=sig.p1, l2=int(np.argmin(ctx.close[sig.p1 + 1:b + 1])) + sig.p1 + 1,
                           drop=sig.drop, seg_start=sig.p1, cci_3=sig.cci_3, w_min=sig.w_min)
            trade = simulate_trade(ctx, fake, params, spec.config.costs, delisted=security.delisted)
            returns.append(trade.net_return if trade is not None and not trade.suspect_in_trade else np.nan)
    return {
        "security_id": security.security_id,
        "timeframe": timeframe,
        "period": period,
        "signal_date": ctx.index[sig.t].date(),
        "pool_size": int(pool.size),
        "returns": returns,
    }


_STORE: Optional[FmpStore] = None


def _init_worker(data_root: str, vintage: Optional[str]) -> None:
    global _STORE
    _STORE = FmpStore(Path(data_root), vintage)


def _run_chunk(securities: Sequence[Security], spec: RunSpec, start: date, end: date) -> ChunkResult:
    """Worker entry point: load one chunk from the vintage and run every security in it."""
    if _STORE is None:
        raise RuntimeError("worker store not initialised")
    out = ChunkResult()
    daily = load_daily_batch(_STORE, [s.security_id for s in securities], start, end)
    for security in securities:
        frame = daily.get(security.security_id)
        if frame is None:
            continue
        out.securities_with_data += 1
        try:
            part = run_security(security, frame, spec, rng_seed=spec.config.seed * 1_000_003 + security.security_id)
        except (ValueError, IndexError, KeyError) as exc:
            _logger.exception("P24 backtest failed for security %s (%s)", security.security_id, security.symbol)
            out.errors.append((security.security_id, repr(exc)))
            continue
        out.trades.extend(part.trades)
        out.random_control.extend(part.random_control)
    return out


def run_backtest(store: FmpStore, securities: Sequence[Security], spec: RunSpec, workers: Optional[int] = None,
                 chunk_size: int = 150) -> ChunkResult:
    """
    Run the backtest over many securities.

    Args:
        store: Open vintage (its root and id are passed to the workers).
        securities: Universe to run.
        spec: Run inputs.
        workers: Process count; default ``os.cpu_count() - 1``; 1 runs in-process (debugging).
        chunk_size: Securities per worker task.

    Returns:
        Combined ``ChunkResult``.
    """
    start = spec.config.window.data_start
    end = spec.as_of or date.today()
    chunks = [list(securities[i:i + chunk_size]) for i in range(0, len(securities), chunk_size)]
    total = ChunkResult()
    n_workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    if n_workers == 1:
        _init_worker(str(store.data_root), store.vintage)
        for i, chunk in enumerate(chunks, 1):
            _merge(total, _run_chunk(chunk, spec, start, end))
            _logger.info("P24 backtest chunk %d/%d done (%d trades so far)", i, len(chunks), len(total.trades))
        return total
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             initargs=(str(store.data_root), store.vintage)) as pool:
        futures = [pool.submit(_run_chunk, chunk, spec, start, end) for chunk in chunks]
        for done, future in enumerate(as_completed(futures), 1):
            _merge(total, future.result())
            if done % 5 == 0 or done == len(futures):
                _logger.info("P24 backtest chunk %d/%d done (%d trades so far)", done, len(futures), len(total.trades))
    return total


def _merge(total: ChunkResult, part: ChunkResult) -> None:
    total.trades.extend(part.trades)
    total.random_control.extend(part.random_control)
    total.securities_with_data += part.securities_with_data
    total.errors.extend(part.errors)


def trades_frame(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """
    Trade rows as a deterministic frame (sorted by timeframe, variant, signal date, security).

    Args:
        rows: Trade dicts.

    Returns:
        Sorted frame.
    """
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return frame
    return frame.sort_values(["timeframe", "variant", "signal_date", "security_id"]).reset_index(drop=True)


def add_benchmark_excess(trades: pd.DataFrame, benchmark_close: pd.Series, name: str = "spy") -> pd.DataFrame:
    """
    Add the benchmark's return over each trade's holding period and the trade's excess over it.

    Benchmark return = close on the exit date / close on the last session before the entry date − 1 (close-to-close
    approximation of the open-to-open holding period).

    Args:
        trades: Trade frame with ``entry_date``, ``exit_date``, ``net_return``.
        benchmark_close: Benchmark daily closes on a ``DatetimeIndex``.
        name: Column prefix.

    Returns:
        Copy with ``<name>_return`` and ``excess_<name>``.
    """
    out = trades.copy()
    if out.empty:
        return out
    closes = benchmark_close.sort_index()
    entry = pd.to_datetime(out["entry_date"]) - pd.Timedelta(days=1)
    exit_ = pd.to_datetime(out["exit_date"])
    start_px = np.asarray(closes.asof(pd.DatetimeIndex(entry)), dtype=float)
    end_px = np.asarray(closes.asof(pd.DatetimeIndex(exit_)), dtype=float)
    out[f"{name}_return"] = end_px / start_px - 1.0
    out[f"excess_{name}"] = out["net_return"] - out[f"{name}_return"]
    return out

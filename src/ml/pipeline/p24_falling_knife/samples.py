"""
P24 Falling Knife — sample trade charts for the report (spec §9.3).

Draws a seeded random sample of full-system trades from a backtest run (headline periods first, survivor years only
to fill up), rebuilds each security's context with the run's parameters and writes the four-panel chart to
``<run_dir>/charts/samples/``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from src.data.fmp_pipeline.datamanager_adapter.fmp_store import FmpStore
from src.ml.pipeline.p24_falling_knife.backtest import FULL, UPPER_BOUND
from src.ml.pipeline.p24_falling_knife.chart import DAILY_WINDOW, WEEKLY_WINDOW, plot_signal
from src.ml.pipeline.p24_falling_knife.config import WEEKLY, P24Config
from src.ml.pipeline.p24_falling_knife.data import load_daily_batch, series_for_timeframe
from src.ml.pipeline.p24_falling_knife.detector import build_context, detect
from src.ml.pipeline.p24_falling_knife.exits import simulate_trade
from src.ml.pipeline.p24_falling_knife.indicators import compute_indicators
from src.ml.pipeline.p24_falling_knife.universe import Security
from src.ml.pipeline.p24_falling_knife.validation import clean_trades
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

SAMPLES_DIR = Path("charts") / "samples"


def sample_trades(trades: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Seeded sample of clean full-system trades, headline periods before survivor years.

    Args:
        trades: Trade rows of one or more timeframes.
        n: Sample size.
        seed: Random seed.

    Returns:
        Up to ``n`` rows.
    """
    if trades.empty:
        return trades
    full = clean_trades(trades)
    full = full[full["variant"] == FULL]
    headline = full[full["period"] != UPPER_BOUND]
    picked = headline.sample(n=min(n, len(headline)), random_state=seed)
    if len(picked) < n:
        rest = full[full["period"] == UPPER_BOUND]
        picked = pd.concat([picked, rest.sample(n=min(n - len(picked), len(rest)), random_state=seed)])
    return picked.sort_values(["timeframe", "signal_date"])


def write_sample_charts(store: FmpStore, securities: Sequence[Security], trades: pd.DataFrame, config: P24Config,
                        run_dir: Path, per_timeframe: int = 10) -> List[Path]:
    """
    Write sample trade charts for a run.

    Args:
        store: Open vintage.
        securities: Run universe (to resolve ``security_id``).
        trades: Every trade of the run (all timeframes).
        config: The run's configuration (calibration applied).
        run_dir: Backtest run folder.
        per_timeframe: Charts per timeframe.

    Returns:
        Written PNG paths.
    """
    by_id: Dict[int, Security] = {s.security_id: s for s in securities}
    paths: List[Path] = []
    if trades.empty:
        return paths
    for timeframe, part in trades.groupby("timeframe"):
        sample = sample_trades(part, per_timeframe, config.seed)
        if sample.empty:
            continue
        params = config.params(str(timeframe))
        ids = [int(i) for i in sample["security_id"].unique() if int(i) in by_id]
        daily = load_daily_batch(store, ids, config.window.data_start, store.as_of or date.today())
        for row in sample.itertuples(index=False):
            sid = int(getattr(row, "security_id"))
            security = by_id.get(sid)
            if security is None or sid not in daily:
                continue
            bars = series_for_timeframe(daily[sid], str(timeframe), config.universe, store.as_of)
            ctx = build_context(compute_indicators(bars, params), params)
            signal_date = pd.Timestamp(getattr(row, "signal_date")).date()
            match = [s for s in detect(ctx, params)[0] if ctx.index[s.t].date() == signal_date]
            if not match:
                _logger.warning("P24 sample chart: no signal for %s on %s", security.symbol, signal_date)
                continue
            trade = simulate_trade(ctx, match[0], params, config.costs, delisted=security.delisted)
            window = WEEKLY_WINDOW if timeframe == WEEKLY else DAILY_WINDOW
            net = getattr(row, "net_return")
            title = f"{security.symbol} {timeframe} signal {signal_date} ({getattr(row, 'period')}, net {net:+.1%}, {getattr(row, 'exit_reason')})"
            path = run_dir / SAMPLES_DIR / f"{timeframe}_{signal_date}_{security.symbol}.png"
            paths.append(plot_signal(ctx, match[0], path, title, trade, window))
    return paths

"""
P24 Falling Knife — backtest CLI (spec §6–§8).

Examples::

    .venv/Scripts/python.exe src/ml/pipeline/p24_falling_knife/jobs/run_p24_backtest.py --limit 300 --workers 1
    .venv/Scripts/python.exe src/ml/pipeline/p24_falling_knife/jobs/run_p24_backtest.py --timeframe both
    .venv/Scripts/python.exe src/ml/pipeline/p24_falling_knife/jobs/run_p24_backtest.py --holdout \
        --calibration results/p24_falling_knife/backtest/<run_id>/calibration_frozen.json

Without ``--holdout`` no holdout-period trade is produced (spec §7.3: the holdout is looked at once, after the
calibration is frozen).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.fmp_pipeline.datamanager_adapter.fmp_store import FmpStore  # noqa: E402
from src.ml.pipeline.p24_falling_knife.backtest import (  # noqa: E402
    RunSpec,
    add_benchmark_excess,
    run_backtest,
    trades_frame,
)
from src.ml.pipeline.p24_falling_knife.config import (  # noqa: E402
    BACKTEST_DIR,
    DAILY,
    VARIANTS,
    WEEKLY,
    P24Config,
)
from src.ml.pipeline.p24_falling_knife.data import load_daily_batch  # noqa: E402
from src.ml.pipeline.p24_falling_knife.report import write_report  # noqa: E402
from src.ml.pipeline.p24_falling_knife.samples import write_sample_charts  # noqa: E402
from src.ml.pipeline.p24_falling_knife.universe import (  # noqa: E402
    Security,
    is_common_stock,
    load_securities,
    survivorship_table,
)
from src.notification.logger import setup_logger  # noqa: E402

_logger = setup_logger(__name__)

DEFAULT_DATA_ROOT = Path("D:/data-cache/fmp-data")
BENCHMARK_SYMBOL = "SPY"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="P24 Falling Knife backtest on the FMP vintage")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="FMP data root (holds current.txt)")
    parser.add_argument("--vintage", default=None, help="Vintage id (default: current.txt)")
    parser.add_argument("--timeframe", choices=[DAILY, WEEKLY, "both"], default="both")
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS))
    parser.add_argument("--symbols", nargs="+", default=None, help="Only these symbols (any instrument type)")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N universe securities (smoke run)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--draws", type=int, default=None, help="Random-control draws per signal (default: config)")
    parser.add_argument("--min-price", type=float, default=None, help="Override the raw-price floor (e.g. 0.10)")
    parser.add_argument("--holdout", action="store_true", help="Also produce holdout-period trades (needs --calibration)")
    parser.add_argument("--calibration", type=Path, default=None, help="calibration_frozen.json to apply")
    parser.add_argument("--sample-charts", type=int, default=10, help="Sample trade charts per timeframe (0: none)")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args(argv)


def apply_calibration(config: P24Config, path: Optional[Path]) -> P24Config:
    """
    Apply a frozen calibration file ``{"daily": {...}, "weekly": {...}}`` to the config.

    Args:
        config: Base configuration.
        path: Calibration JSON, or None.

    Returns:
        Updated configuration.
    """
    if path is None:
        return config
    payload = json.loads(path.read_text(encoding="utf-8"))
    daily = config.daily.with_overrides(**payload.get(DAILY, {}))
    weekly = config.weekly.with_overrides(**payload.get(WEEKLY, {}))
    return replace(config, daily=daily, weekly=weekly)


def select_securities(store: FmpStore, args: argparse.Namespace, include_adr: bool) -> List[Security]:
    """Universe for the run: common stocks, or the explicitly named symbols."""
    securities = load_securities(store.dir)
    if args.symbols:
        wanted = {s.upper() for s in args.symbols}
        return [s for s in securities if s.symbol in wanted]
    chosen = [s for s in securities if is_common_stock(s, include_adr=include_adr)]
    return chosen[: args.limit] if args.limit else chosen


def random_control_summary(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Per-draw mean net return of the matched-random control, by timeframe and period.

    Args:
        rows: ``random_control`` rows from the backtest.

    Returns:
        Frame ``timeframe, period, draw, mean_net_return, n_signals``.
    """
    out: List[Dict[str, Any]] = []
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["timeframe", "period", "draw", "mean_net_return", "n_signals"])
    frame = frame[frame["returns"].map(len) > 0]
    for (timeframe, period), part in frame.groupby(["timeframe", "period"]):
        matrix = np.vstack([np.asarray(r, dtype=float) for r in part["returns"]])
        means = np.nanmean(matrix, axis=0)
        out.extend({"timeframe": timeframe, "period": period, "draw": i, "mean_net_return": float(m),
                    "n_signals": int(matrix.shape[0])} for i, m in enumerate(means))
    return pd.DataFrame(out)


def git_hash() -> str:
    """Current git commit (or 'unknown')."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_manifest(out_dir: Path, args: argparse.Namespace, config: P24Config, store: FmpStore, n_securities: int,
                   n_with_data: int, errors: int) -> None:
    """Write the run manifest (spec §8)."""
    import talib

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash(),
        "vintage": store.vintage,
        "vintage_as_of": str(store.as_of),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "securities_selected": n_securities,
        "securities_with_data": n_with_data,
        "security_errors": errors,
        "config": json.loads(json.dumps(asdict(config), default=str)),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                     "talib": talib.__version__},
        "returns_note": "split-adjusted prices, dividends excluded; costs tiered by as-traded price",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the backtest and write the result files; return the process exit code."""
    args = parse_args(argv)
    if args.holdout and args.calibration is None:
        _logger.error("--holdout needs --calibration: the holdout is evaluated once, with frozen parameters")
        return 2
    config = apply_calibration(P24Config(), args.calibration)
    if args.min_price is not None:
        config = replace(config, universe=replace(config.universe, min_price=args.min_price))
    if args.draws is not None:
        config = replace(config, random_control_draws=args.draws)
    store = FmpStore(args.data_root, args.vintage)
    securities = select_securities(store, args, config.universe.include_adr)
    timeframes = (DAILY, WEEKLY) if args.timeframe == "both" else (args.timeframe,)
    spec = RunSpec(config=config, timeframes=timeframes, variants=tuple(args.variants), include_holdout=args.holdout,
                   as_of=store.as_of)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = BACKTEST_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _logger.info("P24 backtest %s: %d securities, timeframes %s, variants %s, vintage %s", run_id, len(securities),
                 timeframes, args.variants, store.vintage)

    common = [s for s in load_securities(store.dir) if is_common_stock(s, include_adr=config.universe.include_adr)]
    window = config.window
    survivorship_table(common, window.upper_bound_start.year, (store.as_of or date.today()).year,
                       window.coverage_reference_years, window.coverage_threshold).to_csv(
        out_dir / "survivorship.csv", index=False)

    result = run_backtest(store, securities, spec, workers=args.workers)
    trades = trades_frame(result.trades)
    benchmark = store.resolve(BENCHMARK_SYMBOL)
    if benchmark is not None and not trades.empty:
        bench_daily = load_daily_batch(store, [benchmark.security_id], window.data_start, store.as_of or date.today())
        if benchmark.security_id in bench_daily:
            trades = add_benchmark_excess(trades, bench_daily[benchmark.security_id]["close"], name="spy")
    for timeframe in timeframes:
        part = trades[trades["timeframe"] == timeframe] if not trades.empty else trades
        part.to_csv(out_dir / f"trades_{timeframe}.csv", index=False)
    random_control_summary(result.random_control).to_csv(out_dir / "random_control_draws.csv", index=False)
    pd.DataFrame(result.errors, columns=["security_id", "error"]).to_csv(out_dir / "errors.csv", index=False)
    write_manifest(out_dir, args, config, store, len(securities), result.securities_with_data, len(result.errors))
    if args.sample_charts > 0 and not trades.empty:
        charts = write_sample_charts(store, securities, trades, config, out_dir, per_timeframe=args.sample_charts)
        _logger.info("P24 backtest %s: %d sample charts written", run_id, len(charts))
    write_report(out_dir, config.costs)
    _logger.info("P24 backtest %s done: %d trades, %d securities with data, %d errors -> %s", run_id, len(trades),
                 result.securities_with_data, len(result.errors), out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

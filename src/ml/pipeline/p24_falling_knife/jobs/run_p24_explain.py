"""
P24 Falling Knife — explain the detector for one symbol (diagnostics).

Prints, for every bar in the date range that passed the drop + freshness rule, the key metrics and the conditions that
failed. Use it to see why a chart that "looks like a knife" did or did not signal.

Example::

    .venv/Scripts/python.exe src/ml/pipeline/p24_falling_knife/jobs/run_p24_explain.py --symbol MPWR \
        --timeframe weekly --start 2008-09-01 --end 2009-03-31
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data.fmp_pipeline.datamanager_adapter.fmp_store import FmpStore  # noqa: E402
from src.ml.pipeline.p24_falling_knife.config import DAILY, WEEKLY, P24Config  # noqa: E402
from src.ml.pipeline.p24_falling_knife.data import load_daily_batch, series_for_timeframe  # noqa: E402
from src.ml.pipeline.p24_falling_knife.detector import SeriesContext, build_context, detect  # noqa: E402
from src.notification.logger import setup_logger  # noqa: E402

_logger = setup_logger(__name__)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Explain P24 detector decisions for one symbol")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", choices=[DAILY, WEEKLY], default=DAILY)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("D:/data-cache/fmp-data"))
    parser.add_argument("--set", nargs="*", default=[], help="Parameter overrides name=value (floats/ints)")
    parser.add_argument("--out", type=Path, default=None, help="Also write the table as CSV")
    return parser.parse_args(argv)


def _day(ctx: SeriesContext, bar: Optional[int]) -> Optional[date]:
    """Date of a bar index, or None."""
    return None if bar is None else ctx.index[bar].date()


def _overrides(pairs: Sequence[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for pair in pairs:
        name, _, raw = pair.partition("=")
        out[name] = int(raw) if raw.lstrip("-").isdigit() else float(raw)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print the condition table; return the exit code."""
    args = parse_args(argv)
    store = FmpStore(args.data_root)
    security = store.resolve(args.symbol, as_of=args.start)
    if security is None:
        _logger.error("Symbol %s not in vintage %s", args.symbol, store.vintage)
        return 1
    config = P24Config()
    params = config.params(args.timeframe).with_overrides(**_overrides(args.set))
    daily = load_daily_batch(store, [security.security_id], config.window.data_start, store.as_of or date.today())
    if security.security_id not in daily:
        _logger.error("No daily bars for %s", args.symbol)
        return 1
    bars = series_for_timeframe(daily[security.security_id], args.timeframe, config.universe, store.as_of)
    ctx = build_context(bars, params)
    signals, evals = detect(ctx, params, collect_evals=True)
    rows = []
    for ev in evals:
        when = ctx.index[ev.t].date()
        if not args.start <= when <= args.end:
            continue
        rows.append({"date": when, "close": round(float(ctx.close[ev.t]), 2), "p1": ctx.index[ev.p1].date(),
                     "drop": round(ev.drop, 3), "adx": round(float(ctx.adx[ev.t]), 1), "adx_peak": round(ev.adx_peak, 1),
                     "ppo": round(float(ctx.ppo[ev.t]), 2), "ppo_trough": round(ev.ppo_trough, 2),
                     "ppo_floor": round(ev.ppo_floor, 2), "w": round(float(ctx.w[ev.t]), 2), "w_min": round(ev.w_min, 2),
                     "l2": ctx.index[ev.l2].date(), "x": _day(ctx, ev.bar_x), "y": _day(ctx, ev.bar_y),
                     "wmin_at": _day(ctx, ev.bar_wmin),
                     "s3": ctx.index[ev.s3].date() if ev.s3 is not None else None, "cci_2": round(ev.cci_2) if ev.cci_2 == ev.cci_2 else None,
                     "cci_3": round(ev.cci_3) if ev.cci_3 == ev.cci_3 else None, "signal": not ev.missing,
                     "missing": ",".join(ev.missing)})
    table = pd.DataFrame(rows)
    in_range = [ctx.index[s.t].date() for s in signals if args.start <= ctx.index[s.t].date() <= args.end]
    _logger.info("%s %s %s..%s: %d evaluated bars, signals on %s\n%s", args.symbol, args.timeframe, args.start, args.end,
                 len(table), in_range, table.to_string(index=False) if not table.empty else "(no bar passed the drop rule)")
    if args.out is not None and not table.empty:
        table.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

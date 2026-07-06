"""
P17 Strategy Simulator & Optimizer

Path-dependent backtest of a stop-loss + trailing-stop strategy over the
screener's historical Tier A/B/C detections, plus an Optuna optimizer for the
strategy parameters and per-tier position sizing.

Strategy modelled
-----------------
For every ticker, on its first detection:
  * **Entry** at the detection close (``--entry close``, default) or the next
    session's open (``--entry next_open``).
  * **Initial stop** at ``-stop`` from entry.
  * **Trailing stop** arms once price reaches ``+activate`` above entry; once
    armed, the exit follows the running high by ``trail`` and the initial stop is
    discarded.
  * Stops are checked against the daily **low**; the initial stop is evaluated
    before the activation/high each day (pessimistic).

``stop`` and ``trail`` are interpreted as fixed fractions (``--stop-mode pct``)
or as multiples of each name's ATR (``--stop-mode atr``); penny-stock ATR varies
8–30 %/day, so an ATR-scaled stop adapts the leash to volatility.

Caveats
-------
Exits are assumed to fill **at** the stop price. Real penny stocks gap, so a stop
can fill well below — realised losses are an optimistic floor here. Entry at the
detection close is likewise optimistic. Treat results as relative guidance, not a
P&L promise; the historical sample is small.

Usage
-----
    # single parameter set
    python -m src.ml.pipeline.p17_penny_stocks.strategy_sim run \
        --stop 0.20 --trail 0.05 --activate 0.20

    # optimise stop/trail/activate (and optionally sizing) with Optuna
    python -m src.ml.pipeline.p17_penny_stocks.strategy_sim optimize \
        --trials 300 --objective total_pnl --optimize-sizing
"""

import argparse
import glob
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

DEFAULT_RESULTS_DIR = "results/p17_penny_stocks"
DEFAULT_SIZING = {"A": 1000.0, "B": 500.0, "C": 100.0}
TRADEABLE_TIERS = ("A", "B", "C")


# ── Data collection ─────────────────────────────────────────────────────────


def collect_detections(
    results_dir: str,
    since: str | None = None,
    tiers: Tuple[str, ...] = TRADEABLE_TIERS,
) -> List[Dict[str, Any]]:
    """
    First-detection record per ticker: ticker, detection_date, buy_price, tier, atr.

    Records keep the *earliest* appearance (files are date-sorted). Rows with a
    non-positive price or a tier outside ``tiers`` are dropped.
    """
    files = sorted(glob.glob(os.path.join(results_dir, "*", "*_candidates.csv")))
    seen: Dict[str, Dict[str, Any]] = {}
    for f in files:
        day = os.path.basename(os.path.dirname(f))
        if since and day < since:
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            _logger.warning("Could not read %s — skipping", f)
            continue
        if "ticker" not in df.columns or "price" not in df.columns:
            continue
        for r in df.itertuples():
            ticker = str(getattr(r, "ticker", "")).strip().upper()
            tier = str(getattr(r, "tier", "W"))
            if not ticker or ticker == "NAN" or ticker in seen or tier not in tiers:
                continue
            try:
                price = float(getattr(r, "price"))
            except (TypeError, ValueError):
                continue
            if price <= 0:
                continue
            seen[ticker] = {
                "ticker": ticker,
                "detection_date": day,
                "buy_price": price,
                "tier": tier,
                "atr": float(getattr(r, "atr_pct", 0.0) or 0.0),
            }
    return sorted(seen.values(), key=lambda d: (d["detection_date"], d["ticker"]))


def fetch_paths(
    records: List[Dict[str, Any]],
    cache_path: str | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Download daily OHLCV from each ticker's detection date to now, once.

    A pickle cache keyed by ``ticker -> (detection_date, DataFrame)`` lets repeated
    runs (and the optimizer) reuse downloads. Only tickers missing or stale in the
    cache are fetched.
    """
    cache: Dict[str, Tuple[str, pd.DataFrame]] = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                cache = pickle.load(fh)
        except Exception:
            _logger.warning("Could not load path cache %s — refetching", cache_path)

    dl = YahooDataDownloader()
    end = datetime.now(UTC).replace(tzinfo=None)
    paths: Dict[str, pd.DataFrame] = {}
    fetched = 0
    for i, rec in enumerate(records, 1):
        t, day = rec["ticker"], rec["detection_date"]
        if t in cache and cache[t][0] == day:
            paths[t] = cache[t][1]
            continue
        try:
            df = dl.get_ohlcv(t, "1d", datetime.strptime(day, "%Y-%m-%d"), end)
        except Exception:
            df = None
        if df is not None and not df.empty:
            df = df.reset_index(drop=True)
            paths[t] = df
            cache[t] = (day, df)
            fetched += 1
        if i % 25 == 0:
            _logger.info("  fetched price paths %d/%d", i, len(records))

    if cache_path:
        try:
            with open(cache_path, "wb") as fh:
                pickle.dump(cache, fh)
        except Exception:
            _logger.warning("Could not write path cache %s", cache_path)
    _logger.info("Price paths ready: %d tickers (%d newly fetched)", len(paths), fetched)
    return paths


# ── Simulation core ─────────────────────────────────────────────────────────


@dataclass
class StrategyParams:
    """Strategy knobs. ``stop``/``trail`` meaning depends on the *_mode fields."""

    stop: float = 0.20  # pct, or ATR-multiple when stop_mode == "atr"
    trail: float = 0.05  # pct, or ATR-multiple when trail_mode == "atr"
    activate: float = 0.20  # +fraction above entry that arms the trailing stop
    stop_mode: str = "pct"  # "pct" | "atr"
    trail_mode: str = "pct"  # "pct" | "atr"


def _resolve_frac(value: float, mode: str, atr: float) -> float:
    """Resolve a stop/trail setting to a fraction of price for one ticker."""
    frac = value if mode == "pct" else value * atr
    return min(max(frac, 0.02), 0.90)  # clamp to a sane [2%, 90%] band


def simulate_trade(
    buy: float,
    hold: pd.DataFrame,
    params: StrategyParams,
    atr: float,
) -> Tuple[float, str]:
    """
    Simulate one position over its holding-period bars.

    Args:
        buy: Entry price.
        hold: Daily OHLC for the holding period (rows *after* entry), with
            ``high``/``low``/``close`` columns.
        params: Strategy parameters.
        atr: The ticker's ATR fraction (used when a mode is "atr").

    Returns:
        ``(exit_price, reason)`` where reason is ``stop`` | ``trail`` | ``open``.
        ``open`` means never exited — valued at the last close.
    """
    if hold is None or len(hold) == 0:
        return buy, "open"

    stop_frac = _resolve_frac(params.stop, params.stop_mode, atr)
    trail_frac = _resolve_frac(params.trail, params.trail_mode, atr)
    stop_level = buy * (1.0 - stop_frac)
    arm_level = buy * (1.0 + params.activate)

    armed = False
    run_high = buy
    for r in hold.itertuples():
        hi = float(getattr(r, "high"))
        lo = float(getattr(r, "low"))
        if not armed:
            if lo <= stop_level:
                return stop_level, "stop"
            if hi >= arm_level:
                armed = True
                run_high = hi
                if lo <= run_high * (1.0 - trail_frac):
                    return run_high * (1.0 - trail_frac), "trail"
        else:
            run_high = max(run_high, hi)
            tstop = run_high * (1.0 - trail_frac)
            if lo <= tstop:
                return tstop, "trail"
    return float(hold.iloc[-1].close), "open"


def _holding(df: pd.DataFrame | None, entry: str) -> Tuple[float | None, pd.DataFrame | None]:
    """Return (buy_price, holding-rows) for the chosen entry convention."""
    if df is None or len(df) < 2:
        return None, None
    if entry == "next_open":
        buy = float(df.iloc[1].open)
        return buy, df.iloc[1:]
    # entry == "close": buy at detection close (row 0), hold from the next bar
    return float(df.iloc[0].close), df.iloc[1:]


def evaluate(
    records: List[Dict[str, Any]],
    paths: Dict[str, pd.DataFrame],
    params: StrategyParams,
    sizing: Dict[str, float],
    entry: str = "close",
) -> Dict[str, Any]:
    """
    Run every detection through the strategy and aggregate per-tier + total stats.

    Returns a dict with ``per_tier`` (dict keyed by tier and ``TOTAL``) and the raw
    ``trades`` list. Each group reports n, win_rate_pct, stop_rate_pct,
    avg_return_pct, invested, pnl, roi_pct.
    """
    trades: List[Dict[str, Any]] = []
    for rec in records:
        df = paths.get(rec["ticker"])
        buy, hold = _holding(df, entry)
        if buy is None or hold is None:
            continue
        exit_price, reason = simulate_trade(buy, hold, params, rec["atr"])
        ret = exit_price / buy - 1.0
        size = sizing.get(rec["tier"], 0.0)
        trades.append(
            {
                "ticker": rec["ticker"],
                "tier": rec["tier"],
                "reason": reason,
                "return_pct": ret * 100.0,
                "size": size,
                "pnl": size * ret,
            }
        )

    def agg(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
        n = len(rows)
        invested = sum(r["size"] for r in rows)
        pnl = sum(r["pnl"] for r in rows)
        wins = sum(1 for r in rows if r["return_pct"] > 0)
        stops = sum(1 for r in rows if r["reason"] == "stop")
        avg = sum(r["return_pct"] for r in rows) / n if n else 0.0
        return {
            "group": label,
            "n": n,
            "win_rate_pct": round(wins / n * 100, 1) if n else 0.0,
            "stop_rate_pct": round(stops / n * 100, 1) if n else 0.0,
            "avg_return_pct": round(avg, 1),
            "invested": round(invested, 2),
            "pnl": round(pnl, 2),
            "roi_pct": round(pnl / invested * 100, 1) if invested else 0.0,
        }

    per_tier = {"TOTAL": agg(trades, "TOTAL")}
    for tier in TRADEABLE_TIERS:
        sub = [t for t in trades if t["tier"] == tier]
        if sub:
            per_tier[tier] = agg(sub, f"Tier {tier}")
    return {"per_tier": per_tier, "trades": trades}


# ── Objectives ──────────────────────────────────────────────────────────────


def objective_value(result: Dict[str, Any], objective: str) -> float:
    """Scalar to maximise for a given evaluate() result."""
    total = result["per_tier"]["TOTAL"]
    if objective == "total_pnl":
        return total["pnl"]
    if objective == "roi":
        return total["roi_pct"]
    if objective == "sharpe":
        rets = [t["return_pct"] for t in result["trades"]]
        if len(rets) < 2:
            return 0.0
        s = pd.Series(rets)
        std = s.std()
        return s.mean() / std if std > 0 else 0.0
    raise ValueError(f"Unknown objective: {objective}")


# ── CLI ─────────────────────────────────────────────────────────────────────


def _print_result(result: Dict[str, Any]) -> None:
    cols = ["group", "n", "win_rate_pct", "stop_rate_pct", "avg_return_pct", "invested", "pnl", "roi_pct"]
    rows = [result["per_tier"]["TOTAL"]] + [result["per_tier"][t] for t in TRADEABLE_TIERS if t in result["per_tier"]]
    print(pd.DataFrame(rows)[cols].to_string(index=False))


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--since", default=None, help="Only detections on/after YYYY-MM-DD")
    p.add_argument("--entry", choices=["close", "next_open"], default="close")
    p.add_argument("--stop-mode", choices=["pct", "atr"], default="pct")
    p.add_argument("--trail-mode", choices=["pct", "atr"], default="pct")
    p.add_argument("--cache", default=None, help="Pickle cache for price paths (speeds re-runs/optimisation)")
    p.add_argument("--size-a", type=float, default=DEFAULT_SIZING["A"])
    p.add_argument("--size-b", type=float, default=DEFAULT_SIZING["B"])
    p.add_argument("--size-c", type=float, default=DEFAULT_SIZING["C"])


def _load(args) -> Tuple[List[Dict[str, Any]], Dict[str, pd.DataFrame]]:
    cache = args.cache or os.path.join(args.results_dir, "strategy_paths.pkl")
    records = collect_detections(args.results_dir, args.since)
    if not records:
        _logger.error("No detections found under %s", args.results_dir)
        sys.exit(1)
    _logger.info("Loaded %d detections; fetching price paths…", len(records))
    paths = fetch_paths(records, cache)
    return records, paths


def _cmd_run(args) -> int:
    records, paths = _load(args)
    params = StrategyParams(args.stop, args.trail, args.activate, args.stop_mode, args.trail_mode)
    sizing = {"A": args.size_a, "B": args.size_b, "C": args.size_c}
    result = evaluate(records, paths, params, sizing, args.entry)
    print(
        f"\n=== Strategy sim: stop={args.stop} ({args.stop_mode}) "
        f"trail={args.trail} ({args.trail_mode}) activate={args.activate} "
        f"entry={args.entry} ==="
    )
    print(f"sizing: A=${args.size_a:.0f} B=${args.size_b:.0f} C=${args.size_c:.0f}\n")
    _print_result(result)
    return 0


def _cmd_optimize(args) -> int:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    records, paths = _load(args)
    base_sizing = {"A": args.size_a, "B": args.size_b, "C": args.size_c}

    def objective(trial: "optuna.Trial") -> float:
        if args.stop_mode == "atr":
            stop = trial.suggest_float("stop", 0.5, 4.0)
        else:
            stop = trial.suggest_float("stop", 0.05, 0.40)
        if args.trail_mode == "atr":
            trail = trial.suggest_float("trail", 0.3, 3.0)
        else:
            trail = trial.suggest_float("trail", 0.03, 0.30)
        activate = trial.suggest_float("activate", 0.05, 0.60)

        sizing = dict(base_sizing)
        if args.optimize_sizing:
            sizing = {
                "A": trial.suggest_float("size_a", 0.0, 2000.0, step=50.0),
                "B": trial.suggest_float("size_b", 0.0, 2000.0, step=50.0),
                "C": trial.suggest_float("size_c", 0.0, 2000.0, step=50.0),
            }

        params = StrategyParams(stop, trail, activate, args.stop_mode, args.trail_mode)
        result = evaluate(records, paths, params, sizing, args.entry)
        return objective_value(result, args.objective)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    best = study.best_params
    sizing = (
        base_sizing if not args.optimize_sizing else {"A": best["size_a"], "B": best["size_b"], "C": best["size_c"]}
    )
    params = StrategyParams(best["stop"], best["trail"], best["activate"], args.stop_mode, args.trail_mode)
    result = evaluate(records, paths, params, sizing, args.entry)

    print(f"\n=== Optuna best ({args.trials} trials, objective={args.objective}) ===")
    print(f"best {args.objective} = {study.best_value:.4f}")
    print(f"  stop     = {best['stop']:.3f} ({args.stop_mode})")
    print(f"  trail    = {best['trail']:.3f} ({args.trail_mode})")
    print(f"  activate = {best['activate']:.3f}")
    if args.optimize_sizing:
        print(f"  sizing   = A=${sizing['A']:.0f} B=${sizing['B']:.0f} C=${sizing['C']:.0f}")
    else:
        print(f"  sizing   = A=${sizing['A']:.0f} B=${sizing['B']:.0f} C=${sizing['C']:.0f} (fixed)")
    print()
    _print_result(result)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="P17 stop/trailing strategy sim & optimizer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("run", help="Evaluate a single parameter set")
    _add_common(rp)
    rp.add_argument("--stop", type=float, default=0.20)
    rp.add_argument("--trail", type=float, default=0.05)
    rp.add_argument("--activate", type=float, default=0.20)
    rp.set_defaults(func=_cmd_run)

    op = sub.add_parser("optimize", help="Optuna search over strategy params")
    _add_common(op)
    op.add_argument("--trials", type=int, default=200)
    op.add_argument("--objective", choices=["total_pnl", "roi", "sharpe"], default="total_pnl")
    op.add_argument(
        "--optimize-sizing",
        action="store_true",
        help="Also optimise per-tier $ sizing (note: concentrates under total_pnl/roi)",
    )
    op.set_defaults(func=_cmd_optimize)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

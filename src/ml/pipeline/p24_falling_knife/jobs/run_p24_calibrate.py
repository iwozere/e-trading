"""
P24 Falling Knife — calibration grid on the train window (spec §7.3).

Runs every grid cell on train signals only (no holdout, no survivor years, no random control), scores each cell by
the median net trade return of its clean trades and picks a plateau cell per timeframe and shakeout setting. Writes
``calibration_grid.csv``, heatmaps, ``calibration_summary.md`` and ``calibration_frozen.json`` (the setting chosen with
``--frozen-shakeout``) into ``results/p24_falling_knife/calibration/<run_id>/``.

Examples::

    .venv/Scripts/python.exe src/ml/pipeline/p24_falling_knife/jobs/run_p24_calibrate.py --limit 300 --workers 1
    .venv/Scripts/python.exe src/ml/pipeline/p24_falling_knife/jobs/run_p24_calibrate.py --run-id 20261106-grid
    .venv/Scripts/python.exe src/ml/pipeline/p24_falling_knife/jobs/run_p24_calibrate.py \
        --repick results/p24_falling_knife/calibration/20261106-grid --frozen-shakeout off --default-timeframes weekly
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data.fmp_pipeline.datamanager_adapter.fmp_store import FmpStore  # noqa: E402
from src.ml.pipeline.p24_falling_knife.backtest import TRAIN, RunSpec, run_backtest, trades_frame  # noqa: E402
from src.ml.pipeline.p24_falling_knife.chart import plot_heatmap  # noqa: E402
from src.ml.pipeline.p24_falling_knife.config import DAILY, RESULTS_DIR, VARIANTS, WEEKLY, P24Config  # noqa: E402
from src.ml.pipeline.p24_falling_knife.jobs.run_p24_backtest import DEFAULT_DATA_ROOT, git_hash, select_securities  # noqa: E402
from src.ml.pipeline.p24_falling_knife.validation import clean_trades, plateau_pick, trade_metrics  # noqa: E402
from src.notification.logger import setup_logger  # noqa: E402

_logger = setup_logger(__name__)

CALIBRATION_DIR = RESULTS_DIR / "calibration"
OBJECTIVE = "median"
DIMS = ("zz_pct", "support_bounce_pct", "w_narrow", "cci_div_min", "max_bars_since_low")
SHARED_GRID: Dict[str, Tuple[float, ...]] = {
    "zz_pct": (0.20, 0.25, 0.30),
    "support_bounce_pct": (0.06, 0.10),
    "w_narrow": (0.40, 0.60, 0.80),
    "cci_div_min": (15.0, 25.0, 40.0),
}
MAX_BARS_SINCE_LOW: Dict[str, Tuple[int, ...]] = {DAILY: (5, 8), WEEKLY: (8, 12)}
SHAKEOUT = (True, False)
REFERENCES: Dict[str, Dict[str, object]] = {
    "ref_full_default": {},
    "ref_full_no_shakeout": {"use_shakeout_rule": False},
    "ref_drop_only": VARIANTS["drop_only"],
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="P24 calibration grid on the train window")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--vintage", default=None)
    parser.add_argument("--timeframe", choices=[DAILY, WEEKLY, "both"], default="both")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--min-trades", type=int, default=100, help="Minimum clean train trades of the picked cell")
    parser.add_argument("--frozen-shakeout", choices=["on", "off"], default="on",
                        help="Which shakeout setting's pick goes into calibration_frozen.json (book rule: on)")
    parser.add_argument("--default-timeframes", nargs="*", choices=[DAILY, WEEKLY], default=[],
                        help="Timeframes frozen at their default parameters (only the shakeout setting applied) and "
                             "reported for information only, outside the go/no-go gate")
    parser.add_argument("--repick", type=Path, default=None, help="Re-pick from an existing run folder's grid CSV")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args(argv)


def grid_tables(timeframes: Sequence[str]) -> Dict[str, Dict[str, Dict[str, object]]]:
    """
    Variant tables for the run: timeframe -> variant name -> overrides (grid cells plus references).

    Cell names are shared across timeframes (``c000`` …); only ``max_bars_since_low`` levels differ by timeframe.

    Args:
        timeframes: Timeframes to build.

    Returns:
        Per-timeframe variant tables.
    """
    out: Dict[str, Dict[str, Dict[str, object]]] = {}
    keys = list(SHARED_GRID)
    for timeframe in timeframes:
        table: Dict[str, Dict[str, object]] = dict(REFERENCES)
        combos = itertools.product(*(SHARED_GRID[k] for k in keys), MAX_BARS_SINCE_LOW[timeframe], SHAKEOUT)
        for i, values in enumerate(combos):
            cell: Dict[str, object] = dict(zip(keys, values[:len(keys)]))
            cell["max_bars_since_low"] = values[len(keys)]
            cell["use_shakeout_rule"] = values[len(keys) + 1]
            table[f"c{i:03d}"] = cell
        out[timeframe] = table
    return out


def grid_frame(trades: pd.DataFrame, tables: Dict[str, Dict[str, Dict[str, object]]], config: P24Config) -> pd.DataFrame:
    """
    One row per timeframe × variant with its full parameter values and clean-trade metrics.

    Args:
        trades: Train trades of every variant.
        tables: Variant tables used for the run.
        config: Base configuration (for the default values of references).

    Returns:
        Grid frame.
    """
    clean = clean_trades(trades) if not trades.empty else trades
    rows: List[Dict[str, Any]] = []
    for timeframe, table in tables.items():
        base = config.params(timeframe)
        part_tf = clean[clean["timeframe"] == timeframe] if not clean.empty else clean
        for variant, overrides in table.items():
            params = base.with_overrides(**overrides)
            part = part_tf[part_tf["variant"] == variant] if not part_tf.empty else part_tf
            m = trade_metrics(part) if not part.empty else {"count": 0.0}
            row: Dict[str, Any] = {"timeframe": timeframe, "variant": variant}
            row.update({d: getattr(params, d) for d in DIMS})
            row["use_shakeout_rule"] = params.use_shakeout_rule
            row.update({"n": int(m["count"]), "win_rate": m.get("win_rate"), "mean": m.get("mean"), "median": m.get("median"),
                        "trimmed_mean_5": m.get("trimmed_mean_5"), "profit_factor": m.get("profit_factor"),
                        "mean_excess_spy": m.get("mean_excess_spy"),
                        "months": int(pd.to_datetime(part["signal_date"]).dt.to_period("M").nunique()) if not part.empty else 0})
            rows.append(row)
    return pd.DataFrame(rows)


def pick_all(grid: pd.DataFrame, min_trades: int) -> Tuple[pd.DataFrame, Dict[Tuple[str, bool], Dict[str, Any]]]:
    """
    Plateau pick per timeframe and shakeout setting.

    Args:
        grid: ``grid_frame`` output (references are ignored).
        min_trades: Minimum trades of the picked cell.

    Returns:
        ``(annotated grid cells, picks)``; a pick holds the cell row as a dict plus ``plateau``.
    """
    cells = grid[grid["variant"].str.startswith("c")]
    parts: List[pd.DataFrame] = []
    picks: Dict[Tuple[str, bool], Dict[str, Any]] = {}
    for (timeframe, shakeout), part in cells.groupby(["timeframe", "use_shakeout_rule"]):
        annotated, idx, plateau = plateau_pick(part.reset_index(drop=True), list(DIMS), objective=OBJECTIVE,
                                               min_trades=min_trades)
        annotated["picked"] = False
        if idx is not None:
            annotated.loc[annotated.index[idx], "picked"] = True
            pick: Dict[str, Any] = {str(k): v for k, v in annotated.iloc[idx].to_dict().items()}
            pick["plateau"] = plateau
            picks[(str(timeframe), bool(shakeout))] = pick
        parts.append(annotated)
    return (pd.concat(parts, ignore_index=True) if parts else cells.copy()), picks


def write_heatmaps(cells: pd.DataFrame, picks: Dict[Tuple[str, bool], Dict[str, Any]], out_dir: Path) -> List[Path]:
    """Heatmap of the objective over ``w_narrow`` × ``cci_div_min`` at each pick's other parameter values."""
    paths: List[Path] = []
    for (timeframe, shakeout), pick in picks.items():
        part = cells[(cells["timeframe"] == timeframe) & (cells["use_shakeout_rule"] == shakeout)]
        for d in ("zz_pct", "support_bounce_pct", "max_bars_since_low"):
            part = part[part[d] == pick[d]]
        table = part.pivot_table(index="cci_div_min", columns="w_narrow", values=OBJECTIVE, aggfunc="first")
        label = "on" if shakeout else "off"
        title = (f"{timeframe} train median net, shakeout {label} (zz {pick['zz_pct']}, support "
                 f"{pick['support_bounce_pct']}, fresh {pick['max_bars_since_low']})")
        paths.append(plot_heatmap(table, out_dir / "charts" / f"heatmap_{timeframe}_shakeout_{label}.png", title))
    return paths


def _pct(value: Any) -> str:
    return f"{float(value):.1%}" if value is not None and pd.notna(value) else "—"


def summary_markdown(grid: pd.DataFrame, cells: pd.DataFrame, picks: Dict[Tuple[str, bool], Dict[str, Any]],
                     min_trades: int, frozen_shakeout: bool) -> str:
    """Markdown summary: references, picks and the number of cells tried."""
    lines = ["# P24 calibration (train window)\n",
             f"Objective: median net trade return of clean train trades; pick needs ≥ {min_trades} trades and every "
             "one-step neighbour within 25% of it (spec §7.3).\n"]
    for timeframe in sorted(grid["timeframe"].unique()):
        tried = int(((cells["timeframe"] == timeframe)).sum())
        lines.append(f"## {timeframe.capitalize()}\n\nCells tried: {tried} (multiple-testing context).\n")
        lines.append("| row | n | win | mean | median | zz | support | w_narrow | cci_div | fresh | plateau |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        refs = grid[(grid["timeframe"] == timeframe) & grid["variant"].str.startswith("ref_")]
        for _, r in refs.iterrows():
            lines.append(f"| {r['variant']} | {r['n']} | {_pct(r['win_rate'])} | {_pct(r['mean'])} | {_pct(r['median'])} | "
                         f"{r['zz_pct']} | {r['support_bounce_pct']} | {r['w_narrow']} | {r['cci_div_min']} | "
                         f"{r['max_bars_since_low']} | |")
        for shakeout in SHAKEOUT:
            pick = picks.get((timeframe, shakeout))
            label = f"pick, shakeout {'on' if shakeout else 'off'}"
            if pick is None:
                lines.append(f"| {label} | no cell with ≥ {min_trades} trades | | | | | | | | | |")
                continue
            lines.append(f"| {label} ({pick['variant']}) | {pick['n']} | {_pct(pick['win_rate'])} | {_pct(pick['mean'])} | "
                         f"{_pct(pick['median'])} | {pick['zz_pct']} | {pick['support_bounce_pct']} | {pick['w_narrow']} | "
                         f"{pick['cci_div_min']} | {pick['max_bars_since_low']} | {'yes' if pick['plateau'] else 'NO'} |")
        best = cells[(cells["timeframe"] == timeframe) & (cells["n"] >= min_trades)]
        if not best.empty:
            top = best.sort_values(OBJECTIVE, ascending=False).iloc[0]
            lines.append(f"\nBest single cell (not used): {top['variant']} median {_pct(top[OBJECTIVE])}, n {top['n']}.\n")
        share_pos = (best[OBJECTIVE] > 0).mean() if not best.empty else float("nan")
        lines.append(f"Cells with ≥ {min_trades} trades: {len(best)}; share with positive median: {_pct(share_pos)}.\n")
    lines.append(f"`calibration_frozen.json` holds the shakeout-{'on' if frozen_shakeout else 'off'} picks.\n")
    return "\n".join(lines)


def frozen_payload(picks: Dict[Tuple[str, bool], Dict[str, Any]], frozen_shakeout: bool, source: str,
                   default_timeframes: Sequence[str] = ()) -> Dict[str, Any]:
    """
    Calibration file contents (``apply_calibration`` format).

    Args:
        picks: Plateau picks per (timeframe, shakeout).
        frozen_shakeout: Shakeout setting to freeze.
        source: Calibration run folder.
        default_timeframes: Timeframes kept at default parameters (shakeout setting only) and left out of the gate.

    Returns:
        ``{"_meta": {...}, "daily": {...}, "weekly": {...}}``; ``_meta.gate_timeframes`` lists the gated timeframes.
    """
    timeframes = sorted({tf for tf, _ in picks} | set(default_timeframes))
    payload: Dict[str, Any] = {"_meta": {"created_utc": datetime.now(timezone.utc).isoformat(), "source": source,
                                         "objective": OBJECTIVE, "git_hash": git_hash(),
                                         "gate_timeframes": [tf for tf in timeframes if tf not in default_timeframes]}}
    for timeframe in default_timeframes:
        payload[timeframe] = {"use_shakeout_rule": bool(frozen_shakeout)}
        payload["_meta"][f"{timeframe}_info_only"] = True
    for (timeframe, shakeout), pick in picks.items():
        if shakeout != frozen_shakeout or timeframe in default_timeframes:
            continue
        values: Dict[str, Any] = {d: pick[d] for d in DIMS}
        values["max_bars_since_low"] = int(values["max_bars_since_low"])
        values["use_shakeout_rule"] = bool(shakeout)
        payload[timeframe] = values
        payload["_meta"][f"{timeframe}_plateau"] = bool(pick["plateau"])
    return payload


def finish(out_dir: Path, grid: pd.DataFrame, min_trades: int, frozen_shakeout: bool,
           default_timeframes: Sequence[str] = ()) -> None:
    """Pick, write the grid, heatmaps, summary and frozen calibration into ``out_dir``."""
    cells, picks = pick_all(grid, min_trades)
    refs = grid[grid["variant"].str.startswith("ref_")]
    pd.concat([refs, cells], ignore_index=True).to_csv(out_dir / "calibration_grid.csv", index=False)
    write_heatmaps(cells, picks, out_dir)
    (out_dir / "calibration_summary.md").write_text(summary_markdown(grid, cells, picks, min_trades, frozen_shakeout),
                                                    encoding="utf-8")
    payload = frozen_payload(picks, frozen_shakeout, str(out_dir), default_timeframes)
    (out_dir / "calibration_frozen.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _logger.info("P24 calibration picks written to %s", out_dir)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run (or re-pick) the calibration grid; return the process exit code."""
    args = parse_args(argv)
    frozen_shakeout = args.frozen_shakeout == "on"
    if args.repick is not None:
        grid = pd.read_csv(args.repick / "calibration_grid.csv")
        grid = grid[[c for c in grid.columns if c not in ("neighbours", "worst_neighbour", "on_plateau", "picked")]]
        finish(args.repick, grid, args.min_trades, frozen_shakeout, args.default_timeframes)
        return 0
    config = replace(P24Config(), random_control_draws=0)
    store = FmpStore(args.data_root, args.vintage)
    securities = select_securities(store, args, config.universe.include_adr)
    timeframes = (DAILY, WEEKLY) if args.timeframe == "both" else (args.timeframe,)
    tables = grid_tables(timeframes)
    variants = tuple(dict.fromkeys(name for table in tables.values() for name in table))
    spec = RunSpec(config=config, timeframes=timeframes, variants=variants, include_holdout=False, as_of=store.as_of,
                   variant_overrides=tables, periods=(TRAIN,))
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = CALIBRATION_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _logger.info("P24 calibration %s: %d securities, timeframes %s, %d variants", run_id, len(securities), timeframes,
                 len(variants))
    result = run_backtest(store, securities, spec, workers=args.workers)
    trades = trades_frame(result.trades)
    for timeframe in timeframes:
        part = trades[trades["timeframe"] == timeframe] if not trades.empty else trades
        part.to_csv(out_dir / f"trades_{timeframe}.csv.gz", index=False)
    pd.DataFrame(result.errors, columns=["security_id", "error"]).to_csv(out_dir / "errors.csv", index=False)
    finish(out_dir, grid_frame(trades, tables, config), args.min_trades, frozen_shakeout, args.default_timeframes)
    _logger.info("P24 calibration %s done: %d trades, %d errors", run_id, len(trades), len(result.errors))
    return 0


if __name__ == "__main__":
    sys.exit(main())

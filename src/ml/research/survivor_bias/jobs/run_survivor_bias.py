"""
Survivor-bias study — run every frozen rule on the three universes and write the tables (docs/Design.md).

Examples::

    .venv/Scripts/python.exe -X utf8 src/ml/research/survivor_bias/jobs/run_survivor_bias.py
    .venv/Scripts/python.exe -X utf8 src/ml/research/survivor_bias/jobs/run_survivor_bias.py --skip-knife

The month-end panel and the P24 trades are cached in ``<eodhd data root>/derived/`` (EODHD data, kept out of git);
``--rebuild`` recomputes them. Results (aggregates only) go to ``results/survivor_bias/<run_id>/``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.eodhd_pipeline.config import default_data_root  # noqa: E402
from src.data.eodhd_pipeline.store import EodhdStore  # noqa: E402
from src.ml.pipeline.p24_falling_knife.config import PortfolioParams  # noqa: E402
from src.ml.pipeline.p24_falling_knife.portfolio import simulate_portfolio  # noqa: E402
from src.ml.pipeline.p24_falling_knife.universe import Security, is_common_stock, load_securities  # noqa: E402
from src.ml.research.survivor_bias import config  # noqa: E402
from src.ml.research.survivor_bias.delisting import apply_to_trades  # noqa: E402
from src.ml.research.survivor_bias.knife import run_knife  # noqa: E402
from src.ml.research.survivor_bias.panel import StockInfo, build_panel  # noqa: E402
from src.ml.research.survivor_bias.portfolios import (  # noqa: E402
    eligible_rows,
    ew_universe,
    momentum_decile,
    momentum_p21,
    period_stats,
)
from src.notification.logger import setup_logger  # noqa: E402

_logger = setup_logger(__name__)

DERIVED_DIR = "derived"
TRADES_CACHE = "survivor_p24_trades.parquet"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Survivor-bias study on the EODHD store")
    parser.add_argument("--data-root", type=Path, default=None, help="EODHD data root (default: <DATA_CACHE_DIR>/eodhd-data)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true", help="Recompute the cached panel and P24 trades")
    parser.add_argument("--skip-knife", action="store_true", help="Monthly strategies only")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args(argv)


def universes(store: EodhdStore, securities: Sequence[Security]) -> Dict[str, Set[int]]:
    """
    Member ids of the three universes (docs/Design.md, "Universes").

    Args:
        store: Open EODHD store.
        securities: P24 ``Security`` rows of the store.

    Returns:
        ``{full, vintage_like, survivors}`` -> security ids.
    """
    common = [s.security_id for s in securities if is_common_stock(s)]
    return {
        config.FULL: set(common),
        config.VINTAGE_LIKE: {sid for sid in common if store.security(sid).fmp_security_id is not None},
        config.SURVIVORS: {sid for sid in common if store.security(sid).is_actively_trading},
    }


def load_panel(store: EodhdStore, securities: Sequence[Security], members: Set[int], rebuild: bool,
               workers: Optional[int]) -> pd.DataFrame:
    """Cached month-end panel of the ``full`` universe."""
    path = store.data_root / DERIVED_DIR / config.PANEL_CACHE
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    infos = [StockInfo(s.security_id, s.symbol, bool(s.is_actively_trading), s.last_price_date)
             for s in securities if s.security_id in members]
    panel = build_panel(store, infos, workers=workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    _logger.info("Survivor panel: %d rows for %d stocks -> %s", len(panel), panel["security_id"].nunique(), path)
    return panel


def monthly_tables(panel: pd.DataFrame, members: Dict[str, Set[int]]) -> pd.DataFrame:
    """Period statistics of every monthly strategy, universe, delisting version and clip setting."""
    rows: List[Dict[str, Any]] = []
    for universe, ids in members.items():
        for version in config.DELISTING_VERSIONS:
            for clip in (False, True):
                eligible = eligible_rows(panel, ids, version, clip=clip)
                bench = ew_universe(eligible)
                books = {"ew_universe": bench, "momentum_decile": momentum_decile(eligible), "momentum_p21": momentum_p21(eligible)}
                for strategy, book in books.items():
                    for stats in period_stats(book, bench):
                        rows.append({"strategy": strategy, "universe": universe, "delisting": version, "clipped": clip, **stats})
                _logger.info("Monthly strategies done: %s, %s, clip %s", universe, version, clip)
    return pd.DataFrame(rows)


def _trade_stats(part: pd.DataFrame) -> Dict[str, float]:
    returns = part["net_return"].to_numpy(dtype=float)
    n = len(returns)
    sd = float(np.std(returns, ddof=1)) if n > 1 else float("nan")
    return {"trades": n, "mean_net": float(returns.mean()) if n else float("nan"),
            "median_net": float(np.median(returns)) if n else float("nan"),
            "win_rate": float((returns > 0).mean()) if n else float("nan"),
            "t_mean": float(returns.mean() / sd * np.sqrt(n)) if n > 1 and sd > 0 else float("nan"),
            "delisted_share": float(part["delisted_in_trade"].astype(bool).mean()) if n else float("nan")}


def clean_trades(trades: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    """
    True for trades with no suspect month (Amendment 2) from 12 months before the signal to the exit.

    Args:
        trades: P24 trades with ``security_id``, ``signal_date``, ``exit_date``.
        panel: Month-end panel with ``suspect``.

    Returns:
        Boolean series on ``trades.index``.
    """
    if "suspect" not in panel.columns or trades.empty:
        return pd.Series(True, index=trades.index)
    bad = panel[panel["suspect"].astype(bool)]
    by_security: Dict[int, np.ndarray] = {}
    for _, group in bad.groupby("security_id"):
        by_security[int(group["security_id"].iloc[0])] = np.sort(group["month"].to_numpy(dtype="datetime64[ns]"))
    starts = (pd.to_datetime(trades["signal_date"]) - pd.DateOffset(months=12)).to_numpy(dtype="datetime64[ns]")
    ends = pd.to_datetime(trades["exit_date"]).to_numpy(dtype="datetime64[ns]")
    ok = np.ones(len(trades), dtype=bool)
    for i, sid in enumerate(trades["security_id"].to_numpy()):
        months = by_security.get(int(sid))
        if months is None:
            continue
        j = np.searchsorted(months, starts[i])
        ok[i] = not (j < len(months) and months[j] <= ends[i])
    return pd.Series(ok, index=trades.index)


def knife_tables(trades: pd.DataFrame, end_kinds: Dict[int, str], members: Dict[str, Set[int]],
                 clean: Optional[pd.Series] = None) -> pd.DataFrame:
    """Trade and portfolio statistics of the P24 variants per universe, delisting version, data screen and period."""
    rows: List[Dict[str, Any]] = []
    keep = ~trades["suspect_in_trade"].astype(bool)
    clean = clean if clean is not None else pd.Series(True, index=trades.index)
    for screened in (False, True):
        subset = trades[keep & clean] if screened else trades[keep]
        rows.extend(_knife_rows(subset, end_kinds, members, screened))
    return pd.DataFrame(rows)


def _knife_rows(trades: pd.DataFrame, end_kinds: Dict[int, str], members: Dict[str, Set[int]], screened: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    signal_day = pd.to_datetime(trades["signal_date"])
    for version in config.DELISTING_VERSIONS:
        adjusted = apply_to_trades(trades, end_kinds, version)
        for universe, ids in members.items():
            for variant in config.P24_VARIANTS:
                chosen = adjusted[(adjusted["variant"] == variant) & adjusted["security_id"].isin(ids)]
                for period, (start, end) in config.PERIODS.items():
                    in_period = signal_day.loc[chosen.index].between(pd.Timestamp(start), pd.Timestamp(end))
                    part = chosen[in_period]
                    stats: Dict[str, Any] = {"variant": variant, "universe": universe, "delisting": version, "screened": screened,
                                             "period": period}
                    stats.update(_trade_stats(part))
                    summary = simulate_portfolio(part, PortfolioParams()).summary() if not part.empty else {}
                    stats["portfolio_cagr"] = summary.get("cagr", float("nan"))
                    stats["portfolio_max_dd"] = summary.get("max_drawdown", float("nan"))
                    rows.append(stats)
    return rows


def load_trades(store: EodhdStore, securities: Sequence[Security], members: Set[int], rebuild: bool,
                workers: Optional[int]) -> pd.DataFrame:
    """Cached P24 trades of the ``full`` universe."""
    path = store.data_root / DERIVED_DIR / TRADES_CACHE
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    trades = run_knife(store, [s for s in securities if s.security_id in members], workers=workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(path, index=False)
    _logger.info("P24 on EODHD: %d trades -> %s", len(trades), path)
    return trades


def bias_view(table: pd.DataFrame, keys: List[str], value: str) -> pd.DataFrame:
    """
    One row per key with the value in each universe and the two biases.

    Args:
        table: Long statistics table with a ``universe`` column.
        keys: Columns that identify one result.
        value: Statistic to compare.

    Returns:
        Wide frame with ``full``, ``vintage_like``, ``survivors``, ``bias_vintage``, ``bias_survivors``.
    """
    wide = table.pivot_table(index=keys, columns="universe", values=value, aggfunc="first").reset_index()
    wide["bias_vintage"] = wide[config.VINTAGE_LIKE] - wide[config.FULL]
    wide["bias_survivors"] = wide[config.SURVIVORS] - wide[config.FULL]
    return wide


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the study and write the result files; return the process exit code."""
    args = parse_args(argv)
    store = EodhdStore(args.data_root or default_data_root())
    securities = load_securities(store.dir)
    members = universes(store, securities)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = config.RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _logger.info("Survivor bias %s: full %d, vintage_like %d, survivors %d stocks; store date %s", run_id,
                 len(members[config.FULL]), len(members[config.VINTAGE_LIKE]), len(members[config.SURVIVORS]), store.as_of)

    panel = load_panel(store, securities, members[config.FULL], args.rebuild, args.workers)
    monthly = monthly_tables(panel, members)
    monthly.to_csv(out_dir / "monthly_strategies.csv", index=False)
    keys = ["strategy", "delisting", "clipped", "period"]
    bias_view(monthly, keys, "cagr").to_csv(out_dir / "bias_cagr.csv", index=False)
    bias_view(monthly, keys, "excess_ann").to_csv(out_dir / "bias_excess.csv", index=False)

    if not args.skip_knife:
        trades = load_trades(store, securities, members[config.FULL], args.rebuild, args.workers)
        firsts = panel.groupby("security_id")["end_kind"].first()
        end_kinds = {int(sid): str(kind) for sid, kind in zip(firsts.index.to_numpy(), firsts.to_numpy())}
        knife = knife_tables(trades, end_kinds, members, clean_trades(trades, panel))
        knife.to_csv(out_dir / "p24.csv", index=False)
        bias_view(knife, ["variant", "delisting", "screened", "period"], "mean_net").to_csv(out_dir / "bias_p24_mean.csv", index=False)

    ends = panel.groupby("security_id")["end_kind"].first().value_counts()
    (out_dir / "end_kinds.csv").write_text(ends.to_csv(), encoding="utf-8")
    (out_dir / "run.txt").write_text(f"run {run_id}\nstore date {store.as_of}\ncreated {date.today()}\n", encoding="utf-8")
    _logger.info("Survivor bias %s done -> %s", run_id, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

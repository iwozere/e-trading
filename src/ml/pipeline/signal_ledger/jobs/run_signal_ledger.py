"""
Signal Ledger — nightly job: harvest signals, refresh prices, score, rebuild paper books.

Steps:
1. Harvest pick files of the last ``RECENT_HARVEST_DAYS`` (``--full``: whole archive = backfill)
   and upsert them into ``signal_ledger``.
2. Refresh the OHLC price cache for every ticker with a pending signal or a signal in the
   last ``ACTIVE_PRICE_DAYS`` (whole history from its earliest signal, so a ticker's prices
   always come from one fetch) plus benchmarks.
3. Evaluate pending signals and write their outcomes.
4. Rebuild each stream's simulated paper book under every exit policy (``exits.py``) into
   ``signal_book_nav`` and ``signal_book_trades``.
5. Write ``scorecard.json`` (scorecard, default-policy books, per-policy book and trade stats)
   to ``results/signal_ledger/YYYY-MM-DD/``.

Usage:
    python src/ml/pipeline/signal_ledger/jobs/run_signal_ledger.py [--full]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.signal_ledger_service import SignalLedgerService
from src.ml.pipeline.signal_ledger.config import ACTIVE_PRICE_DAYS, BENCHMARKS, RECENT_HARVEST_DAYS, RESULTS_DIR, RESULTS_ROOT
from src.ml.pipeline.signal_ledger.evaluator import STATUS_COMPLETE, STATUS_NO_DATA, evaluate_signals
from src.ml.pipeline.signal_ledger.harvesters import harvest_all
from src.ml.pipeline.signal_ledger.exits import DEFAULT_POLICY, policies_for
from src.ml.pipeline.signal_ledger.portfolio import simulate_book, simulate_uncapped, summarize_book, summarize_trades
from src.ml.pipeline.signal_ledger.prices import PriceCache
from src.ml.pipeline.signal_ledger.scorecard import build_scorecard
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Fetch a few days before the earliest signal so its entry session is always inside the cache.
_PRICE_LEAD_DAYS = 7


def run(
    today: Optional[date] = None,
    full: bool = False,
    service: Optional[SignalLedgerService] = None,
    cache: Optional[PriceCache] = None,
    results_root: Path = RESULTS_ROOT,
    results_dir: Path = RESULTS_DIR,
) -> Dict[str, Any]:
    """
    Execute one ledger cycle.

    Args:
        today: Run date (default: today).
        full: Harvest the whole archive instead of the recent folders.
        service: DB service (injectable for tests).
        cache: Price cache (injectable for tests).
        results_root: Project ``results/`` directory to harvest.
        results_dir: Where the dated ``scorecard.json`` is written.

    Returns:
        Summary dict for __SCHEDULER_RESULT__.
    """
    today = today or date.today()
    db = service or SignalLedgerService()
    prices = cache or PriceCache()

    since = None if full else today - timedelta(days=RECENT_HARVEST_DAYS)
    records = harvest_all(results_root, since)
    inserted = db.upsert_signals([r.to_row() for r in records])
    _logger.info("Harvested %d signals (%d new) since %s", len(records), inserted, since or "the beginning")

    pending = db.get_pending_signals()
    active = db.list_signals(since=today - timedelta(days=ACTIVE_PRICE_DAYS))
    refreshed = _refresh_prices(prices, [*pending, *active], today)

    panels = {field: prices.panel(field, before=today) for field in ("open", "high", "low", "close")}
    opens, closes = panels["open"], panels["close"]
    outcomes = evaluate_signals(pending, opens, closes, today, panels["high"], panels["low"]) if len(closes.index) else []
    db.update_outcomes(outcomes)
    completed = sum(1 for o in outcomes if o["status"] == STATUS_COMPLETE)
    no_data = sum(1 for o in outcomes if o["status"] == STATUS_NO_DATA)

    rows = db.list_signals()
    policies = _rebuild_books(db, rows, panels)
    books = {source: by_policy[DEFAULT_POLICY.name]["book"] for source, by_policy in policies.items()}
    scorecard = build_scorecard(rows)
    _write_scorecard(results_dir / today.isoformat(), scorecard, books, policies)

    result = {
        "date": today.isoformat(),
        "harvested": len(records),
        "new_signals": inserted,
        "pending_evaluated": len(outcomes),
        "completed": completed,
        "no_data": no_data,
        "prices_refreshed": refreshed,
        "streams": len(scorecard),
        "books_simulated": sum(len(by_policy) for by_policy in policies.values()),
        "edge_streams": sorted({s["stream"] for s in scorecard for h in s["horizons"] if h["verdict"] == "edge"}),
    }
    _logger.info("Signal ledger run complete: %s", result)
    return result


def _refresh_prices(cache: PriceCache, signals: List[Dict[str, Any]], today: date) -> int:
    """
    Refresh the tickers of the given signals (pending and recent ones) and the benchmarks.

    A ticker's columns are replaced as a whole, so its fetch starts at the earlier of
    its earliest given signal and its first cached price — older signals of the same
    ticker keep the prices the paper books need.
    """
    if not signals:
        return 0
    starts: Dict[str, date] = {}
    for signal in signals:
        ticker, first = signal["ticker"], signal["signal_date"] - timedelta(days=_PRICE_LEAD_DAYS)
        cached = cache.first_date(ticker)
        starts[ticker] = min(starts.get(ticker, first), first, cached or first)
    known = cache.sessions()
    bench_start = min(min(starts.values()), known[0].date()) if len(known) else min(starts.values())
    refreshed = cache.refresh(BENCHMARKS, bench_start, today)
    # One batch per start month keeps requests few without fetching years for a recent signal.
    by_month: Dict[date, List[str]] = {}
    for ticker, start in starts.items():
        if ticker not in BENCHMARKS:
            by_month.setdefault(start.replace(day=1), []).append(ticker)
    for month_start, tickers in sorted(by_month.items()):
        refreshed += cache.refresh(tickers, month_start, today)
    cache.save()
    return refreshed


def _rebuild_books(service: SignalLedgerService, rows: List[Dict[str, Any]], panels: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Simulate and store every stream's paper book under each of its exit policies.

    The capped book (NAV + trades) is stored; the uncapped run only contributes ``all_signals`` stats.

    Returns:
        ``{source: {policy: {"book": capped book summary, "trades": capped trade stats,
        "all_signals": uncapped trade stats}}}`` (JSON-safe).
    """
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    prices = (panels["open"], panels["close"], panels["high"], panels["low"])
    summaries: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for source, signals in sorted(by_source.items()):
        for policy in policies_for(source):
            result = simulate_book(source, signals, *prices, policy=policy)
            uncapped = simulate_uncapped(source, signals, *prices, policy=policy)
            service.replace_book_nav(source, result.nav, policy.name)
            service.replace_book_trades(source, policy.name, result.trades)
            stats = {"book": summarize_book(result.nav), "trades": summarize_trades(result.trades),
                     "all_signals": summarize_trades(uncapped.trades)}
            summaries.setdefault(source, {})[policy.name] = json.loads(json.dumps(stats, default=str))
        service.replace_book_stats(source, summaries.get(source, {}))
    return summaries


def _write_scorecard(
    day_dir: Path,
    scorecard: List[Dict[str, Any]],
    books: Dict[str, Dict[str, Any]],
    policies: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    """Write the dated scorecard snapshot (kept so the scorecard's history can be compared later)."""
    day_dir.mkdir(parents=True, exist_ok=True)
    payload = {"scorecard": scorecard, "books": books, "policies": policies}
    (day_dir / "scorecard.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    """Parse arguments, run the job and print the scheduler result."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="harvest the whole results archive (backfill)")
    args = parser.parse_args()
    result = run(full=args.full)
    print(f"__SCHEDULER_RESULT__:{json.dumps(result, default=str)}")


if __name__ == "__main__":
    main()

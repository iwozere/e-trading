"""End-to-end test of the nightly job with an in-memory service and a fake price fetcher."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from src.ml.pipeline.signal_ledger.jobs.run_signal_ledger import run
from src.ml.pipeline.signal_ledger.prices import PriceCache


class FakeService:
    """In-memory stand-in for SignalLedgerService."""

    def __init__(self) -> None:
        self.rows: Dict[tuple, Dict[str, Any]] = {}
        self.nav: Dict[tuple, List[Dict[str, Any]]] = {}
        self.trades: Dict[tuple, List[Dict[str, Any]]] = {}
        self.stats: Dict[str, Dict[str, Any]] = {}

    def upsert_signals(self, rows: List[Dict[str, Any]]) -> int:
        new = 0
        for row in rows:
            key = (row["source"], row["signal_date"], row["ticker"], row["side"])
            if key not in self.rows:
                self.rows[key] = {"id": len(self.rows) + 1, "status": "pending", **row}
                new += 1
        return new

    def get_pending_signals(self) -> List[Dict[str, Any]]:
        return [{k: r[k] for k in ("id", "signal_date", "ticker")} for r in self.rows.values() if r["status"] == "pending"]

    def update_outcomes(self, outcomes: List[Dict[str, Any]]) -> int:
        by_id = {r["id"]: r for r in self.rows.values()}
        for outcome in outcomes:
            by_id[outcome["id"]].update(outcome)
        return len(outcomes)

    def list_signals(self, **_: Any) -> List[Dict[str, Any]]:
        return sorted(self.rows.values(), key=lambda r: r["signal_date"])

    def replace_book_nav(self, source: str, rows: List[Dict[str, Any]], policy: str = "horizon") -> int:
        self.nav[(source, policy)] = list(rows)
        return len(rows)

    def replace_book_trades(self, source: str, policy: str, trades: List[Dict[str, Any]]) -> int:
        self.trades[(source, policy)] = list(trades)
        return len(trades)

    def replace_book_stats(self, source: str, stats_by_policy: Dict[str, Dict[str, Any]]) -> int:
        self.stats[source] = dict(stats_by_policy)
        return len(stats_by_policy)


def _fetcher(tickers: List[str], start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
    days = pd.bdate_range("2026-09-01", "2026-10-30")
    days = days[(days >= pd.Timestamp(start)) & (days < pd.Timestamp(end))]
    out = {}
    for ticker in tickers:
        if ticker == "DEAD":
            continue
        growth = 1.002 if ticker == "WIN" else 1.0
        closes = [100.0 * growth**i for i in range(len(days))]
        out[ticker] = pd.DataFrame({"timestamp": days, "open": closes, "high": [c * 1.01 for c in closes],
                                    "low": [c * 0.99 for c in closes], "close": closes})
    return out


def _results(root: Path) -> None:
    folder = root / "p05_ai_selector/2026-09-02"
    folder.mkdir(parents=True)
    (folder / "top_picks.csv").write_text("rank,ticker,confidence,bias\n1,WIN,8,long\n2,DEAD,7,long\n", encoding="utf-8")


def test_full_run_harvests_scores_and_builds_books(tmp_path: Path) -> None:
    _results(tmp_path / "results")
    service = FakeService()
    cache = PriceCache(path=tmp_path / "prices.parquet", fetcher=_fetcher)

    result = run(today=date(2026, 10, 30), full=True, service=service, cache=cache,  # type: ignore[arg-type]
                 results_root=tmp_path / "results", results_dir=tmp_path / "out")

    assert result["new_signals"] == 2
    assert result["completed"] == 0  # the 60/120/250-session horizons have not matured
    assert result["no_data"] == 1
    win = next(r for r in service.rows.values() if r["ticker"] == "WIN")
    assert win["entry_date"] == date(2026, 9, 3)
    assert win["ret_20d"] > 0.03
    assert win["ret_60d"] is None
    assert win["mae_20d"] == pytest.approx(-0.01) and win["mfe_20d"] > 0.03
    assert service.nav[("p05_top_picks", "horizon")][-1]["nav"] > 200_000
    # every generic policy plus the decay-based native exit for P05
    assert {policy for source, policy in service.trades if source == "p05_top_picks"} >= {"horizon", "fixed_20", "bracket", "trail_atr", "native"}
    assert result["books_simulated"] == len({k for k in service.nav})
    assert (tmp_path / "prices.parquet").exists()
    payload = json.loads((tmp_path / "out/2026-10-30/scorecard.json").read_text(encoding="utf-8"))
    assert payload["scorecard"][0]["stream"] == "p05_top_picks"
    assert payload["policies"]["p05_top_picks"]["native"]["trades"]["trades"] == 1
    assert service.stats["p05_top_picks"]["bracket"]["all_signals"]["trades"] == 1
    json.dumps(service.stats)  # stored stats are JSON-safe (dates as strings)


def test_second_run_is_idempotent(tmp_path: Path) -> None:
    _results(tmp_path / "results")
    service = FakeService()
    kwargs: Dict[str, Optional[Any]] = {"service": service, "results_root": tmp_path / "results", "results_dir": tmp_path / "out"}
    run(today=date(2026, 10, 30), full=True, cache=PriceCache(path=tmp_path / "p.parquet", fetcher=_fetcher), **kwargs)  # type: ignore[arg-type]

    result = run(today=date(2026, 10, 30), full=True, cache=PriceCache(path=tmp_path / "p.parquet", fetcher=_fetcher), **kwargs)  # type: ignore[arg-type]

    assert result["new_signals"] == 0
    assert result["pending_evaluated"] == 1  # WIN stays pending until its 250-session horizon

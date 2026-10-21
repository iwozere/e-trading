"""End-to-end test of the nightly job with an in-memory service and a fake price fetcher."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.ml.pipeline.signal_ledger.jobs.run_signal_ledger import run
from src.ml.pipeline.signal_ledger.prices import PriceCache


class FakeService:
    """In-memory stand-in for SignalLedgerService."""

    def __init__(self) -> None:
        self.rows: Dict[tuple, Dict[str, Any]] = {}
        self.nav: Dict[str, List[Dict[str, Any]]] = {}

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

    def replace_book_nav(self, source: str, rows: List[Dict[str, Any]]) -> int:
        self.nav[source] = list(rows)
        return len(rows)


def _fetcher(tickers: List[str], start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
    days = pd.bdate_range("2026-09-01", "2026-10-30")
    days = days[(days >= pd.Timestamp(start)) & (days < pd.Timestamp(end))]
    out = {}
    for ticker in tickers:
        if ticker == "DEAD":
            continue
        growth = 1.002 if ticker == "WIN" else 1.0
        closes = [100.0 * growth**i for i in range(len(days))]
        out[ticker] = pd.DataFrame({"timestamp": days, "open": closes, "close": closes})
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
    assert result["completed"] == 1
    assert result["no_data"] == 1
    win = next(r for r in service.rows.values() if r["ticker"] == "WIN")
    assert win["entry_date"] == date(2026, 9, 3)
    assert win["ret_20d"] > 0.03
    assert service.nav["p05_top_picks"][-1]["nav"] > 200_000
    assert (tmp_path / "prices.parquet").exists()
    payload = json.loads((tmp_path / "out/2026-10-30/scorecard.json").read_text(encoding="utf-8"))
    assert payload["scorecard"][0]["stream"] == "p05_top_picks"


def test_second_run_is_idempotent(tmp_path: Path) -> None:
    _results(tmp_path / "results")
    service = FakeService()
    kwargs: Dict[str, Optional[Any]] = {"service": service, "results_root": tmp_path / "results", "results_dir": tmp_path / "out"}
    run(today=date(2026, 10, 30), full=True, cache=PriceCache(path=tmp_path / "p.parquet", fetcher=_fetcher), **kwargs)  # type: ignore[arg-type]

    result = run(today=date(2026, 10, 30), full=True, cache=PriceCache(path=tmp_path / "p.parquet", fetcher=_fetcher), **kwargs)  # type: ignore[arg-type]

    assert result["new_signals"] == 0
    assert result["pending_evaluated"] == 0

"""Tests for P17 FilingsAgent (Form 4 + 13D/G annotation from the P15 cache)."""

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p17_penny_stocks.agents.filings_agent import FilingsAgent
from src.ml.pipeline.p17_penny_stocks.config import P17FilingsConfig
from src.ml.pipeline.p17_penny_stocks.models.candidate import Candidate

TARGET = "2026-11-05"  # Thursday → window 11-01..11-05 (weekdays 11-02..11-05)


class _FakeEdgar:
    """Stands in for EdgarDownloader: cache dirs + cache-only Form 4 reads + company_tickers map."""

    def __init__(self, root: Path, form4: Dict[str, pd.DataFrame]) -> None:
        self._form4_dir = root / "form4"
        self._13dg_dir = root / "13dg"
        self._form4_dir.mkdir()
        self._13dg_dir.mkdir()
        self.form4 = form4
        self.network_calls = 0

    def download_form4_filings(self, as_of_date: Optional[date] = None, allow_network: bool = True, **_: Any) -> pd.DataFrame:
        assert allow_network is False, "P17 must read the Form 4 cache only"
        df = self.form4.get(str(as_of_date))
        if df is not None:
            (self._form4_dir / f"{as_of_date}.csv.gz").write_text("x")
            return df
        return pd.DataFrame()

    def load_company_tickers(self) -> Dict[str, Any]:
        return {"0": {"ticker": "AAA", "cik_str": 1111}, "1": {"ticker": "BBB", "cik_str": 2222}}


def _f4(rows: list) -> pd.DataFrame:
    cols = ["ticker", "insider_name", "transaction_code", "acquired_disposed_code", "total_value_usd", "is_derivative"]
    return pd.DataFrame(rows, columns=cols)


def _write_13dg(edgar: _FakeEdgar, day: str, rows: list) -> None:
    df = pd.DataFrame(rows, columns=["cik", "entity_name", "accession_number", "filed_date", "form_type"])
    df.to_csv(edgar._13dg_dir / f"{day}.csv.gz", index=False, compression="gzip")


def test_window_is_lookback_calendar_days_ending_on_target(tmp_path: Path):
    agent = FilingsAgent(P17FilingsConfig(lookback_days=5), TARGET, edgar=_FakeEdgar(tmp_path, {}))  # type: ignore[arg-type]
    assert agent.window_days()[0] == date(2026, 11, 5)
    assert agent.window_days()[-1] == date(2026, 11, 1)


def test_form4_buys_sells_and_cluster(tmp_path: Path):
    form4 = {
        "2026-11-03": _f4([
            ["aaa", "Jane CEO", "P", "A", 50_000, False],
            ["AAA", "John CFO", "P", "A", 30_000, False],
            ["AAA", "Jane CEO", "M", "A", 99_999, False],  # option exercise — ignored
            ["BBB", "Bob Dir", "S", "D", 10_000, False],
        ]),
        "2026-10-30": _f4([["AAA", "Old Buyer", "P", "A", 1_000_000, False]]),  # outside window
    }
    edgar = _FakeEdgar(tmp_path, form4)
    a, b = Candidate(ticker="AAA"), Candidate(ticker="BBB")
    FilingsAgent(P17FilingsConfig(), TARGET, edgar=edgar).run([a, b])  # type: ignore[arg-type]
    assert (a.insider_buy_count, a.insider_buyers, a.insider_buy_value) == (2, 2, 80_000.0)
    assert a.insider_sell_count == 0
    assert (b.insider_sell_count, b.insider_sell_value) == (1, 10_000.0)


def test_13dg_matches_subject_cik_and_form_types(tmp_path: Path):
    edgar = _FakeEdgar(tmp_path, {})
    _write_13dg(edgar, "2026-11-04", [
        ["0000001111", "AAA Corp", "0001-26-1", "2026-11-04", "SC 13G"],  # subject line
        ["9999999", "Some Fund LP", "0001-26-1", "2026-11-04", "SC 13G"],  # filer line, same filing
        ["2222", "BBB Inc", "0001-26-2", "2026-11-04", "SC 13D/A"],
    ])
    a, b = Candidate(ticker="AAA"), Candidate(ticker="BBB")
    FilingsAgent(P17FilingsConfig(), TARGET, edgar=edgar).run([a, b])  # type: ignore[arg-type]
    assert a.fresh_13g and not a.fresh_13d and not a.fresh_13dg_amend
    assert not b.fresh_13g and b.fresh_13dg_amend


def test_13dg_outside_window_ignored(tmp_path: Path):
    edgar = _FakeEdgar(tmp_path, {})
    _write_13dg(edgar, "2026-10-30", [["1111", "AAA Corp", "0001-26-1", "2026-10-30", "SC 13G"]])
    a = Candidate(ticker="AAA")
    FilingsAgent(P17FilingsConfig(), TARGET, edgar=edgar).run([a])  # type: ignore[arg-type]
    assert not a.fresh_13g


def test_missing_cache_leaves_defaults(tmp_path: Path):
    a = Candidate(ticker="AAA")
    FilingsAgent(P17FilingsConfig(), TARGET, edgar=_FakeEdgar(tmp_path, {})).run([a])  # type: ignore[arg-type]
    assert a.insider_buy_count == 0 and not a.fresh_13g


def test_13dg_accepts_schedule_form_names(tmp_path: Path):
    edgar = _FakeEdgar(tmp_path, {})
    _write_13dg(edgar, "2026-11-04", [["1111", "AAA Corp", "0001-26-1", "2026-11-04", "SCHEDULE 13G"]])
    a = Candidate(ticker="AAA")
    FilingsAgent(P17FilingsConfig(), TARGET, edgar=edgar).run([a])  # type: ignore[arg-type]
    assert a.fresh_13g

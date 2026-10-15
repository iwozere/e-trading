"""Tests for ``fmp snapshot-universe`` (spec §12.1): no network, synthetic source files."""

from __future__ import annotations

import csv
import hashlib
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data.fmp_pipeline import cli
from src.data.fmp_pipeline.planner.config import SnapshotConfig
from src.data.fmp_pipeline.snapshots import universe as module
from src.data.fmp_pipeline.snapshots.universe import UniverseSnapshot, parse_constituents, parse_directory, sources

CFG = SnapshotConfig()
NASDAQ = ("Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
          "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
          "ZZZT|Test Issue|Q|Y|N|100|N|N\n"
          "{extra}"
          "File Creation Time: 1015202508:01|||||||\n")
OTHER = ("ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
         "BRK.B|Berkshire Hathaway Inc. Class B|N|BRK.B|N|100|N|BRK.B\n"
         "SPY|SPDR S&P 500|P|SPY|Y|100|N|SPY\n"
         "File Creation Time: 1015202508:01|||||||\n")
SP500 = "Symbol,Security\nAAPL,Apple\nBRK.B,Berkshire\n"
NDX = ("<html><body><table><tr><th>Company</th><th>Ticker</th></tr><tr><td>Apple</td><td>AAPL</td></tr>"
       "</table></body></html>")
DOW = ("<html><body><table><tr><th>Company</th><th>Symbol</th></tr><tr><td>Apple</td><td>AAPL</td></tr>"
       "</table></body></html>")


def _fetcher(extra: str = "", fail: tuple[str, ...] = ()) -> Any:
    bodies = {CFG.nasdaq_listed_url: NASDAQ.format(extra=extra), CFG.other_listed_url: OTHER, CFG.sp500_url: SP500,
              CFG.nasdaq100_url: NDX, CFG.dow_url: DOW}

    def fetch(url: str) -> bytes:
        if any(name in url for name in fail):
            raise ConnectionError("unreachable")
        return bodies[url].encode("utf-8")

    return fetch


class FakeYahoo:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, symbols: list[str], interval: str, start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
        self.calls.append(symbols)
        index = pd.DatetimeIndex(["2025-03-03", "2025-03-04"]).tz_localize("America/New_York")
        frame = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 10.0, "Dividends": 0.0,
                              "Stock Splits": 0.0}, index=index)
        return {"GONE": frame} if "GONE" in symbols else {}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_parsers():
    rows = parse_directory(NASDAQ.format(extra=""), "nasdaq_listed")
    assert [r["symbol"] for r in rows] == ["AAPL"] and rows[0]["exchange"] == "Q"
    other = parse_directory(OTHER, "other_listed")
    assert [(r["symbol"], r["exchange"], r["etf"]) for r in other] == [("BRK.B", "N", "N"), ("SPY", "P", "Y")]
    assert parse_directory("", "other_listed") == []
    by_name = {s.name: s for s in sources(CFG)}
    assert parse_constituents(SP500.encode(), by_name["sp500"]) == ["AAPL", "BRK.B"]
    assert parse_constituents(NDX.encode(), by_name["nasdaq100"]) == ["AAPL"]


def test_first_snapshot_then_diff_with_last_downloads(tmp_path: Path):
    first = UniverseSnapshot(tmp_path, CFG, _fetcher("GONE|Gone Corp|Q|N|N|100|N|N\n"), FakeYahoo(),
                             date(2026, 9, 15)).run()
    assert first.symbols == 4 and first.previous is None and not first.errors
    assert first.constituents == {"sp500": 2, "nasdaq100": 1, "dow": 1}
    yahoo = FakeYahoo()
    second = UniverseSnapshot(tmp_path, CFG, _fetcher("NEWX|New Corp|Q|N|N|100|N|N\n"), yahoo,
                              date(2026, 10, 15)).run()
    assert (second.previous, second.added, second.removed) == ("2026-09-15", ["NEWX"], ["GONE"])
    assert second.last_downloads == {"GONE": 2} and yahoo.calls == [["GONE"]]
    directory = tmp_path / "snapshots" / "2026-10-15"
    [suspect] = _read(directory / "delisted_suspected.csv")
    assert suspect == {"symbol": "GONE", "snapshot_date": "2026-10-15", "last_bar_date": "2025-03-04",
                       "bars_downloaded": "2"}
    assert (directory / "last_bars" / "GONE.parquet").exists()
    assert "Compared with 2026-09-15: 1 added, 1 removed." in (directory / "diff.md").read_text(encoding="utf-8")
    sums = (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    listed = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in sums}
    assert "nasdaqlisted.txt" in listed and "SHA256SUMS" not in listed
    assert listed["sp500.csv"] == hashlib.sha256((directory / "sp500.csv").read_bytes()).hexdigest()


def test_failed_directory_never_reports_mass_removals(tmp_path: Path):
    UniverseSnapshot(tmp_path, CFG, _fetcher(), FakeYahoo(), date(2026, 9, 15)).run()
    report = UniverseSnapshot(tmp_path, CFG, _fetcher(fail=("otherlisted",)), FakeYahoo(), date(2026, 10, 15)).run()
    assert "download failed" in report.errors["otherlisted"] and report.removed == []
    broken = UniverseSnapshot(tmp_path, SnapshotConfig(dow_url=CFG.sp500_url), _fetcher(), FakeYahoo(),
                              date(2026, 10, 16), download_removed=False).run()
    assert "parse failed" in broken.errors["dow"]  # the CSV has no HTML table; the raw file is still kept
    assert (tmp_path / "snapshots" / "2026-10-16" / "dow.html").exists()


def test_cli_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(module, "http_fetch", lambda user_agent: _fetcher())
    assert cli.main(["snapshot-universe", "--data-root", str(tmp_path), "--no-download"]) == cli.EXIT_OK
    monkeypatch.setattr(module, "http_fetch", lambda user_agent: _fetcher(fail=("nasdaqlisted",)))
    assert cli.main(["snapshot-universe", "--data-root", str(tmp_path), "--no-download"]) == cli.EXIT_STOPPED

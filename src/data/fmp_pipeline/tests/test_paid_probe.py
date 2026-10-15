"""Tests for ``fmp probe-paid`` and ``fmp calibrate-usage`` (spec §4.4, §8.4, Appendix A) — mock server, no FMP."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data.fmp_pipeline import cli
from src.data.fmp_pipeline.manifest.manifest import Manifest
from src.data.fmp_pipeline.planner.config import PlanConfig, load_config
from src.data.fmp_pipeline.probe.calibrate import GB, LocalUsage, calibrate, local_usage
from src.data.fmp_pipeline.probe.paid_analysis import (
    RowCapProbe,
    expected_bars,
    hourly_conventions,
    intraday_split_verdict,
    reuse_verdict,
    row_cap,
    window_complete,
)
from src.data.fmp_pipeline.probe.paid_probe import PaidFindings, PaidProbe, derive_settings
from src.data.fmp_pipeline.probe.paid_report import MARKER, render_paid_report
from src.data.fmp_pipeline.probe.probe_client import ProbeClient
from src.data.fmp_pipeline.probe.results import apply_probe_results, read_results, write_calibration, write_results
from src.data.fmp_pipeline.tests.mock_server import MockConfig, MockFmpServer

ET = "America/New_York"
#: Regular sessions around the 2024 and 2025 DST changes, plus the 2024-11-29 early close.
DAYS = ("2024-03-08", "2024-03-11", "2024-11-01", "2024-11-04", "2025-03-07", "2025-03-10", "2024-11-29")


def hourly_rows(days: tuple[str, ...], written_in: str = ET, label: str = "start",
                extra_hours: bool = False) -> list[dict[str, Any]]:
    """Synthetic hourly rows: regular-session bars (early close respected) written the way a vendor might."""
    rows = []
    for day in days:
        close = 13 * 60 if day == "2024-11-29" else 16 * 60
        starts = list(range(9 * 60 + 30, close, 60)) + ([16 * 60 + 30] if extra_hours else [])
        for minute in starts:
            stamp = pd.Timestamp(f"{day} {minute // 60:02d}:{minute % 60:02d}").tz_localize(ET)
            if label == "end":
                stamp += pd.Timedelta(minutes=60)
            local = stamp.tz_convert(written_in).tz_localize(None)
            rows.append({"date": local.strftime("%Y-%m-%d %H:%M:%S"), "close": 1.0})
    return rows


# --- analysis -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("written_in,label", [(ET, "start"), (ET, "end"), ("UTC", "start"), ("UTC", "end")])
def test_conventions_are_identified_across_dst(written_in: str, label: str):
    conventions = hourly_conventions(hourly_rows(DAYS, written_in, label))
    assert (conventions.timezone, conventions.label) == (written_in, label)
    assert conventions.bars_per_day == 7 and not conventions.extended_hours and conventions.conclusive
    assert conventions.early_close_bars == {"2024-11-29": 4} and conventions.days == len(DAYS)


def test_conventions_extended_hours_and_inconclusive():
    assert hourly_conventions(hourly_rows(DAYS, extra_hours=True)).extended_hours
    fixed_offset = [{"date": f"{d} 13:30:00"} for d in DAYS]  # one fixed UTC hour all year: fits no convention
    result = hourly_conventions(fixed_offset)
    assert not result.conclusive and result.first_bars == {"13:30": len(DAYS)}
    assert hourly_conventions([]).days == 0 and hourly_conventions([{"date": "junk"}]).days == 0


def test_row_cap_and_truncation_direction():
    full = RowCapProbe(date(2025, 1, 1), date(2025, 12, 31), 1750, date(2025, 1, 2), date(2025, 12, 31), 1750)
    assert row_cap([full]) == row_cap([full])
    assert (row_cap([full]).max_rows, row_cap([full]).capped) == (1750, False)
    newest = RowCapProbe(date(2016, 1, 1), date(2025, 12, 31), 5000, date(2023, 3, 1), date(2025, 12, 31), 17500)
    oldest = RowCapProbe(date(2016, 1, 1), date(2025, 12, 31), 5000, date(2016, 1, 4), date(2018, 11, 1), 17500)
    assert row_cap([full, newest]).keeps == "newest" and row_cap([full, newest]).max_rows == 5000
    assert row_cap([oldest]).keeps == "oldest" and row_cap([]).max_rows is None
    assert expected_bars(date(2024, 11, 25), date(2024, 11, 29), 7) == 4 * 7  # Thanksgiving is no session


def test_split_window_and_reuse_verdicts():
    rows = [{"date": "2024-06-07 15:30:00", "close": 1200.0}, {"date": "2024-06-10 09:30:00", "close": 121.0}]
    assert intraday_split_verdict(rows, "2024-06-10", 10.0) == "raw"
    rows[0]["close"] = 120.0
    assert intraday_split_verdict(rows, "2024-06-10", 10.0) == "split_adjusted"
    rows[0]["close"] = 300.0
    assert intraday_split_verdict(rows, "2024-06-10", 10.0) == "unclear"
    assert intraday_split_verdict(rows[:1], "2024-06-10", 10.0) == "no_data"
    window = [{"date": "2025-01-03"}, {"date": "2025-03-28"}]
    assert window_complete(window, date(2025, 1, 1), date(2025, 3, 31))
    assert not window_complete(window, date(2025, 1, 1), date(2025, 6, 29)) and not window_complete([], date.min,
                                                                                                    date.max)
    listing = [{"delistedDate": "2015-06-30"}]
    assert reuse_verdict(listing, [{"date": "2010-01-04"}, {"date": "2020-01-02"}]) == "merged series"
    assert reuse_verdict(listing, [{"date": "2018-01-02"}]) == "latest listing only"
    assert reuse_verdict(listing, [{"date": "2012-01-03"}]) == "earlier listing only"
    assert reuse_verdict([], [{"date": "2012-01-03"}]) == "latest listing only"
    assert reuse_verdict(listing, []) == "no data"


def test_derive_settings_only_from_conclusive_findings():
    assert derive_settings(PaidFindings()) == {}
    findings = PaidFindings(conventions=hourly_conventions(hourly_rows(DAYS)),
                            cap=row_cap([RowCapProbe(date(2016, 1, 1), date(2025, 12, 31), 5000, date(2023, 3, 1),
                                                     date(2025, 12, 31), 17500)]),
                            earliest_hourly=date(2010, 3, 1), page_sizes={"delisted-companies": 1000},
                            calendar_windows={"splits-calendar": 90, "earnings-calendar": 30},
                            indicators={"GDP": 40, "CPI": 0})
    assert derive_settings(findings) == {
        "intraday": {"max_rows": 5000, "bars_per_day": 7, "earliest": "2010-03-01"},
        "build": {"intraday_tz": ET, "intraday_label": "start"}, "page_size": 1000, "calendar_window_days": 30,
        "economic_indicators": ["GDP"]}


# --- results file and config ------------------------------------------------------------------------------------


def test_results_apply_over_the_yaml_config(tmp_path: Path):
    config = PlanConfig()
    assert apply_probe_results(config, tmp_path) is config  # no paid probe yet
    settings = {"intraday": {"max_rows": 5000, "earliest": "2010-03-01"}, "build": {"intraday_label": "end"},
                "page_size": 1000}
    write_results(tmp_path, PaidFindings(earliest_hourly=date(2010, 3, 1)), settings, 120)
    applied = apply_probe_results(config, tmp_path)
    assert applied.intraday.max_rows == 5000 and applied.intraday.earliest == date(2010, 3, 1)
    assert applied.build.intraday_label == "end" and applied.page_size == 1000
    assert apply_probe_results(PlanConfig(use_probe_results=False), tmp_path).intraday.max_rows is None
    write_calibration(tmp_path, {"basis": "wire"}, "wire")
    assert apply_probe_results(config, tmp_path).bandwidth.basis == "wire"
    write_results(tmp_path, PaidFindings(), {"page_size": 500}, 130)  # a re-run keeps the calibration
    results = read_results(tmp_path)
    assert results is not None and results["settings"]["bandwidth"] == {"basis": "wire"}
    assert results["calibration"] == {"basis": "wire"}
    write_results(tmp_path, PaidFindings(), {"nonsense": 1}, 130)
    with pytest.raises(ValueError, match="unknown setting probe_results.nonsense"):
        apply_probe_results(config, tmp_path)


# --- calibration ----------------------------------------------------------------------------------------------


def test_calibration_verdicts():
    usage = LocalUsage(int(1.0 * GB), int(7.0 * GB), ("m.sqlite",))
    assert calibrate(1.05, usage).basis == "wire"
    assert calibrate(6.8, usage).basis == "decoded"
    inconclusive = calibrate(4.0, usage)
    assert inconclusive.basis is None and "inconclusive" in inconclusive.message
    assert "other programs" in calibrate(20.0, usage).message
    assert calibrate(1.0, LocalUsage(0, 0, ())).basis is None
    with pytest.raises(ValueError):
        calibrate(0, usage)


def test_local_usage_reads_manifests_and_ledgers(tmp_path: Path):
    now = datetime(2026, 11, 20, tzinfo=timezone.utc)
    with Manifest(tmp_path / "manifests" / "d.sqlite", "d") as manifest:
        manifest.record_bandwidth(100, 700, now - timedelta(days=1))
        manifest.record_bandwidth(1000, 7000, now - timedelta(days=40))  # outside the window
    ledger = tmp_path / "probe-paid" / "ledger.jsonl"
    ledger.parent.mkdir()
    ledger.write_text(json.dumps({"at": (now - timedelta(days=2)).isoformat(), "bytes_wire": 10,
                                  "bytes_decoded": 70}) + "\n\n"
                      + json.dumps({"at": (now - timedelta(days=2)).isoformat(), "bytes_wire": 5}) + "\n",
                      encoding="utf-8")
    usage = local_usage(tmp_path, 30, now)
    assert (usage.wire, usage.decoded, usage.ledger_lines_without_decoded) == (115, 770, 1)
    assert usage.sources == ("d.sqlite", "probe-paid/ledger.jsonl")
    assert local_usage(tmp_path / "empty", 30, now).wire == 0


# --- the whole probe against the mock --------------------------------------------------------------------------

LISTS = {
    "stock-list": [{"symbol": "AAPL", "companyName": "A", "exchange": "NASDAQ"},
                   {"symbol": "IBM", "companyName": "I", "exchange": "NYSE"}],
    "actively-trading-list": [{"symbol": "AAPL", "name": "A"}, {"symbol": "SYM00001", "name": "Reused"}],
    "etf-list": [{"symbol": "SPY", "name": "S"}],
    "available-exchanges": [{"exchange": "NASDAQ", "name": "Nasdaq"}],
    "sp500-constituent": [{"symbol": "AAPL", "name": "A", "dateFirstAdded": "1982-11-30"}],
    "symbol-change": [{"date": "2025-01-02", "companyName": "X", "oldSymbol": "OLD", "newSymbol": "NEW"}],
    "cik-list": [{"cik": "0000000001", "companyName": "A"}],
    "mergers-acquisitions-latest": [{"symbol": "A", "transactionDate": "2025-01-02"}],
}


def _mock() -> MockFmpServer:
    return MockFmpServer(MockConfig(earliest=date(2010, 3, 1), restricted_symbols={"BRK.B", "^RUT"},
                                    restricted_endpoints={"historical-chart/1min"}, extra_fixtures=LISTS))


def test_paid_probe_against_the_mock(tmp_path: Path):
    with _mock() as fmp, ProbeClient(fmp.config.api_key, tmp_path / "cache", budget=400, min_interval_s=0,
                                     base_url=fmp.base_url) as client:
        run = PaidProbe(client, today=date(2026, 10, 14)).run()
        calls = client.calls_used
    f = run.findings
    assert f.intervals["1min"] == "plan_restricted" and f.intervals["1hour"] == "ok"
    assert f.coverage["BRK-B"] == "ok" and f.coverage["BRK.B"] == "plan_restricted"
    assert f.conventions is not None and (f.conventions.timezone, f.conventions.label) == (ET, "start")
    assert f.conventions.bars_per_day == 7  # the mock ignores holidays, so extended hours show up there
    assert f.cap is not None and f.cap.capped and f.cap.max_rows == 1000 and f.cap.keeps == "newest"
    assert f.earliest_hourly is not None and date(2010, 2, 1) <= f.earliest_hourly <= date(2010, 4, 1)
    assert f.intraday_split == "split_adjusted" and f.earliest_daily is None
    assert f.page_sizes["delisted-companies"] == 100 and f.lists["stock-list"]["rows"] == 2
    assert f.exchanges == {"NASDAQ": 2, "NYSE": 1}
    assert f.settings["intraday"]["max_rows"] == 1000 and f.settings["build"]["intraday_tz"] == ET
    assert f.settings["page_size"] == 100 and calls < 150
    assert list(f.reuse) == ["SYM00001"] and f.reuse["SYM00001"].startswith("delisted and actively trading: ")
    text = render_paid_report(run, calls, 400)
    assert text.startswith(MARKER) and "| §3.3 row cap / maximum window | 1000 rows, keeps the newest end |" in text


def test_paid_probe_without_hourly_bars(tmp_path: Path):
    with MockFmpServer(MockConfig(restricted_endpoints={f"historical-chart/{i}" for i in
                                                        ("1min", "5min", "15min", "30min", "1hour", "4hour")})) as fmp, \
            ProbeClient(fmp.config.api_key, tmp_path, budget=200, min_interval_s=0, base_url=fmp.base_url) as client:
        run = PaidProbe(client, today=date(2026, 10, 14)).run()
    assert "1hour bars are not served" in run.findings.notes[0] and "intraday" not in run.findings.settings


def test_cli_probe_paid_and_calibrate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    report = tmp_path / "paid.md"
    report.write_text("# Paid probe\n\nOwner notes.\n\n" + MARKER + "\nold\n", encoding="utf-8")
    with _mock() as fmp:
        real = ProbeClient

        def client(api_key: Any, cache_dir: Path, budget: int, offline: bool) -> ProbeClient:
            return real(api_key, cache_dir, budget=budget, offline=offline, min_interval_s=0, base_url=fmp.base_url)

        monkeypatch.setattr(cli, "ProbeClient", client)
        monkeypatch.setattr(cli, "_api_key", lambda: None)
        args = ["probe-paid", "--data-root", str(tmp_path), "--report", str(report)]
        assert cli.main(args) == cli.EXIT_USAGE  # no key
        monkeypatch.setattr(cli, "_api_key", lambda: fmp.config.api_key)
        assert cli.main(args) == cli.EXIT_OK
        served = len(fmp.requests)
        assert cli.main(args + ["--offline"]) == cli.EXIT_OK and len(fmp.requests) == served  # cache only
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# Paid probe\n\nOwner notes.") and "\nold\n" not in text  # generated part replaced
    config = tmp_path / "fmp.yaml"
    config.write_text("fmp:\n  dump_id: c\n", encoding="utf-8")
    plan_config = cli._config(argparse.Namespace(config=config), tmp_path)  # the CLI's config: probe applied
    assert plan_config.intraday.max_rows == 1000 and load_config(config).intraday.max_rows is None
    assert cli.main(["calibrate-usage", "--dashboard-gb", "0.001", "--data-root", str(tmp_path)]) in (
        cli.EXIT_OK, cli.EXIT_STOPPED)
    assert read_results(tmp_path)["calibration"]["dashboard_gb"] == 0.001  # type: ignore[index]
    assert cli.main(["calibrate-usage", "--dashboard-gb", "-1", "--data-root", str(tmp_path)]) == cli.EXIT_USAGE

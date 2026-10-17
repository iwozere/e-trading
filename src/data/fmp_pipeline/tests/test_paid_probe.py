"""Tests for ``fmp probe-paid`` and ``fmp calibrate-usage`` (spec §4.4, §8.4, Appendix A) — mock server, no FMP."""

from __future__ import annotations

import argparse
import gc
import json
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data.fmp_pipeline import cli
from src.data.fmp_pipeline.manifest.manifest import Manifest
from src.data.fmp_pipeline.planner.config import BandwidthConfig, IntradayConfig, PlanConfig, load_config
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
from src.data.fmp_pipeline.probe.paid_probe import (
    IntradayFindings,
    PaidFindings,
    PaidProbe,
    derive_settings,
    intraday_settings,
    session_bars,
)
from src.data.fmp_pipeline.probe.paid_report import MARKER, render_paid_report
from src.data.fmp_pipeline.probe.probe_client import ProbeClient
from src.data.fmp_pipeline.probe.results import apply_probe_results, read_results, write_calibration, write_results
from src.data.fmp_pipeline.tests.mock_server import MockConfig, MockFmpServer

ET = "America/New_York"
#: Regular sessions around the 2024 and 2025 DST changes, plus the 2024-11-29 early close.
DAYS = ("2024-03-08", "2024-03-11", "2024-11-01", "2024-11-04", "2025-03-07", "2025-03-10", "2024-11-29")


def hourly_rows(days: tuple[str, ...], written_in: str = ET, label: str = "start",
                extra_hours: bool = False, minutes: int = 60) -> list[dict[str, Any]]:
    """Synthetic intraday rows: regular-session bars (early close respected) written the way a vendor might."""
    rows = []
    for day in days:
        close = 13 * 60 if day == "2024-11-29" else 16 * 60
        starts = list(range(9 * 60 + 30, close, minutes)) + ([16 * 60 + 30] if extra_hours else [])
        for minute in starts:
            stamp = pd.Timestamp(f"{day} {minute // 60:02d}:{minute % 60:02d}").tz_localize(ET)
            if label == "end":
                stamp += pd.Timedelta(minutes=minutes)
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


@pytest.mark.parametrize("written_in,label", [(ET, "start"), ("UTC", "end")])
def test_five_minute_conventions(written_in: str, label: str):
    conventions = hourly_conventions(hourly_rows(DAYS, written_in, label, minutes=5), minutes=5)
    assert (conventions.timezone, conventions.label) == (written_in, label) and conventions.bars_per_day == 78
    assert conventions.early_close_bars == {"2024-11-29": 42}
    assert not hourly_conventions(hourly_rows(DAYS, minutes=5), minutes=60).conclusive  # wrong bar length


def test_session_bars_per_interval():
    assert [session_bars(i) for i in ("1min", "5min", "15min", "30min", "1hour", "4hour")] == [390, 78, 26, 13, 7, 2]
    assert IntradayConfig(interval="5min").session_bars == 78 and IntradayConfig().session_bars == 7
    assert IntradayConfig(interval="5min", bars_per_day=80).session_bars == 80


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
    hourly = IntradayFindings("1hour", conventions=hourly_conventions(hourly_rows(DAYS)),
                              cap=row_cap([RowCapProbe(date(2016, 1, 1), date(2025, 12, 31), 5000, date(2023, 3, 1),
                                                       date(2025, 12, 31), 17500)]),
                              earliest=date(2010, 3, 1), wire_bytes_per_row=20.5, decoded_bytes_per_row=140.0)
    findings = PaidFindings(intraday={"1hour": hourly}, page_sizes={"delisted-companies": 1000},
                            calendar_windows={"splits-calendar": 90, "earnings-calendar": 30},
                            indicators={"GDP": 40, "CPI": 0})
    assert derive_settings(findings) == {
        "intraday": {"max_rows": 5000, "bars_per_day": 7, "earliest": "2010-03-01", "wire_bytes_per_row": 20.5,
                     "decoded_bytes_per_row": 140.0},
        "build": {"intraday_tz": ET, "intraday_label": "start"}, "page_size": 1000, "calendar_window_days": 30,
        "economic_indicators": ["GDP"]}
    findings.interval = "5min"  # configured interval not measured: no intraday settings at all
    assert "intraday" not in derive_settings(findings) and "build" not in derive_settings(findings)
    assert intraday_settings(IntradayFindings("5min")) == {}


# --- results file and config ------------------------------------------------------------------------------------


def test_results_apply_over_the_yaml_config(tmp_path: Path):
    config = PlanConfig()
    assert apply_probe_results(config, tmp_path) is config  # no paid probe yet
    settings = {"intraday": {"max_rows": 5000, "earliest": "2010-03-01"}, "build": {"intraday_label": "end"},
                "page_size": 1000}
    write_results(tmp_path, PaidFindings(), settings, 120)
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


def test_results_follow_the_configured_interval(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    by_interval: dict[str, dict[str, Any]] = {
        "5min": {"intraday": {"max_rows": 9000, "bars_per_day": 78, "earliest": "2015-01-01"},
                 "build": {"intraday_label": "end"}},
        "1hour": {"intraday": {"max_rows": 5000, "bars_per_day": 7, "earliest": "2010-03-01"}}}
    settings = {**by_interval["5min"], "page_size": 1000}
    write_results(tmp_path, PaidFindings(interval="5min"), settings, 120, by_interval)
    five = apply_probe_results(PlanConfig(intraday=IntradayConfig(interval="5min")), tmp_path)
    assert (five.intraday.max_rows, five.intraday.session_bars, five.build.intraday_label) == (9000, 78, "end")
    hourly = apply_probe_results(PlanConfig(), tmp_path)  # owner switched back to 1hour: no re-probe needed
    assert (hourly.intraday.max_rows, hourly.intraday.earliest, hourly.build.intraday_label) == (
        5000, date(2010, 3, 1), "start")
    assert hourly.page_size == 1000
    unmeasured = apply_probe_results(PlanConfig(intraday=IntradayConfig(interval="15min")), tmp_path)
    assert unmeasured.intraday.max_rows is None and unmeasured.intraday.session_bars == 26
    assert unmeasured.page_size == 1000 and "did not measure intraday 15min" in caplog.text


def test_results_keep_earlier_intervals_and_feed_tier_intervals(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    five = {"5min": {"intraday": {"max_rows": 510, "bars_per_day": 78}}}
    write_results(tmp_path, PaidFindings(interval="5min"), five["5min"], 120, five)
    write_results(tmp_path, PaidFindings(interval="15min"), {}, 140, {"15min": {"intraday": {"max_rows": 500}}})
    assert set(read_results(tmp_path)["intraday_by_interval"]) == {"5min", "15min"}  # type: ignore[index]
    intraday = IntradayConfig(interval="5min", tier_intervals={"C": "15min", "E": "30min"})
    applied = apply_probe_results(PlanConfig(intraday=intraday), tmp_path).intraday
    assert applied.max_rows == 510 and applied.for_interval("15min").max_rows == 500
    assert applied.for_interval("30min").max_rows is None and "did not measure intraday 30min" in caplog.text


def test_config_rejects_bad_tier_intervals(tmp_path: Path):
    config = tmp_path / "fmp.yaml"
    config.write_text("fmp:\n  intraday:\n    tier_intervals: {C: 15min}\n", encoding="utf-8")
    assert load_config(config).intraday.tier_intervals == {"C": "15min"}
    for bad in ("{F: 15min}", "{C: 2hour}", "[C]"):
        config.write_text(f"fmp:\n  intraday:\n    tier_intervals: {bad}\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(config)


def test_config_rejects_an_unknown_interval(tmp_path: Path):
    config = tmp_path / "fmp.yaml"
    config.write_text("fmp:\n  intraday:\n    interval: 2hour\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fmp.intraday.interval"):
        load_config(config)


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


def test_calibration_records_the_dashboard_ratio(tmp_path: Path):
    usage = LocalUsage(int(1.0 * GB), int(7.0 * GB), ("m.sqlite",))
    fmp_style = calibrate(5.25, usage)  # what the 2026-10 dashboard does: ~0.75 of the decoded bytes
    assert (fmp_style.basis, fmp_style.scale) == ("decoded", 0.75) and "0.75 per byte" in fmp_style.message
    assert calibrate(1.05, usage).scale == 1.05
    lagging = calibrate(0.004, LocalUsage(int(0.05 * GB), int(0.1 * GB), ()))  # the meter showed 4 MB
    assert lagging.basis is None and lagging.scale is None and "at least 0.5 GB" in lagging.message
    write_calibration(tmp_path, vars(fmp_style), fmp_style.basis, fmp_style.scale)
    applied = apply_probe_results(PlanConfig(bandwidth=BandwidthConfig(cap_gb=150)), tmp_path).bandwidth
    assert (applied.basis, applied.scale) == ("decoded", 0.75)
    assert applied.limit_bytes == pytest.approx(0.9 * 150 * GB / 0.75)


def test_config_rejects_a_non_positive_scale(tmp_path: Path):
    config = tmp_path / "fmp.yaml"
    config.write_text("fmp:\n  bandwidth:\n    scale: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fmp.bandwidth.scale"):
        load_config(config)


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


def test_local_usage_closes_the_manifest_connection(tmp_path: Path):
    with Manifest(tmp_path / "manifests" / "d.sqlite", "d") as manifest:
        manifest.record_bandwidth(1, 7)
    gc.collect()  # objects other tests leaked must not be blamed on this call
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        local_usage(tmp_path)
        gc.collect()
    assert not [w for w in caught if issubclass(w.category, ResourceWarning) and "sqlite3.Connection" in str(w.message)]
    (tmp_path / "manifests" / "d.sqlite").unlink()  # no handle left open (Windows would refuse)


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
    assert list(f.intraday) == ["1hour"]
    m = f.intraday["1hour"]
    assert m.conventions is not None and (m.conventions.timezone, m.conventions.label) == (ET, "start")
    assert m.conventions.bars_per_day == 7  # the mock ignores holidays, so extended hours show up there
    assert m.cap is not None and m.cap.capped and m.cap.max_rows == 1000 and m.cap.keeps == "newest"
    assert m.earliest is not None and date(2010, 2, 1) <= m.earliest <= date(2010, 4, 1)
    assert m.split == "split_adjusted" and f.earliest_daily is None
    assert m.decoded_bytes_per_row and m.wire_bytes_per_row and m.wire_bytes_per_row < m.decoded_bytes_per_row
    assert f.page_sizes["delisted-companies"] == 100 and f.lists["stock-list"]["rows"] == 2
    assert f.exchanges == {"NASDAQ": 2, "NYSE": 1}
    assert f.settings["intraday"]["max_rows"] == 1000 and f.settings["build"]["intraday_tz"] == ET
    assert f.settings["page_size"] == 100 and calls < 150
    assert f.settings["intraday"]["decoded_bytes_per_row"] == m.decoded_bytes_per_row
    assert list(f.settings_by_interval) == ["1hour"]
    assert list(f.reuse) == ["SYM00001"] and f.reuse["SYM00001"].startswith("delisted and actively trading: ")
    text = render_paid_report(run, calls, 400)
    assert text.startswith(MARKER) and "| §3.3 row cap / maximum window (`1hour`) | 1000 rows, keeps the newest end |" in text


def test_paid_probe_measures_five_minutes_and_hourly(tmp_path: Path):
    with _mock() as fmp, ProbeClient(fmp.config.api_key, tmp_path / "cache", budget=400, min_interval_s=0,
                                     base_url=fmp.base_url) as client:
        run = PaidProbe(client, today=date(2026, 10, 14), interval="5min").run()
        calls = client.calls_used
    f = run.findings
    assert list(f.intraday) == ["5min", "1hour"] and f.interval == "5min"
    five = f.intraday["5min"]
    assert five.conventions is not None and five.conventions.bars_per_day == 78
    assert (five.conventions.timezone, five.conventions.label) == (ET, "start")
    assert five.cap is not None and five.cap.capped and five.cap.max_rows == 1000 and five.split == "split_adjusted"
    assert five.earliest is not None and date(2010, 2, 1) <= five.earliest <= date(2010, 4, 1)
    assert f.settings["intraday"]["bars_per_day"] == 78
    assert f.settings_by_interval["1hour"]["intraday"]["bars_per_day"] == 7 and calls < 200
    text = render_paid_report(run, calls, 400)
    assert "(intraday `5min`)" in text and "## Row-cap requests (AAPL `5min`)" in text
    assert "## Row-cap requests (AAPL `1hour`)" in text and "| §3.3 row cap / maximum window (`5min`) |" in text
    with pytest.raises(ValueError, match="unknown intraday interval"):
        PaidProbe(client, interval="2hour")


def test_paid_probe_when_the_configured_interval_is_not_served(tmp_path: Path):
    with MockFmpServer(MockConfig(restricted_endpoints={"historical-chart/5min"})) as fmp, \
            ProbeClient(fmp.config.api_key, tmp_path, budget=300, min_interval_s=0, base_url=fmp.base_url) as client:
        run = PaidProbe(client, today=date(2026, 10, 14), interval="5min").run()
    f = run.findings
    assert "5min bars are not served" in f.notes[0] and list(f.intraday) == ["1hour"]
    assert "intraday" not in f.settings and "intraday" in f.settings_by_interval["1hour"]


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
        args = ["probe-paid", "--data-root", str(tmp_path), "--report", str(report), "--interval", "1hour"]
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
    assert set(read_results(tmp_path)["intraday_by_interval"]) == {"1hour"}  # type: ignore[index]
    assert cli.main(args + ["--offline", "--interval", "2hour"]) == cli.EXIT_USAGE
    bad = tmp_path / "bad.yaml"
    bad.write_text("fmp:\n  intraday:\n    interval: 2hour\n", encoding="utf-8")
    assert cli.main(args[:-2] + ["--offline", "--config", str(bad)]) == cli.EXIT_USAGE
    assert cli.main(["calibrate-usage", "--dashboard-gb", "0.001", "--data-root", str(tmp_path)]) in (
        cli.EXIT_OK, cli.EXIT_STOPPED)
    assert read_results(tmp_path)["calibration"]["dashboard_gb"] == 0.001  # type: ignore[index]
    assert cli.main(["calibrate-usage", "--dashboard-gb", "-1", "--data-root", str(tmp_path)]) == cli.EXIT_USAGE

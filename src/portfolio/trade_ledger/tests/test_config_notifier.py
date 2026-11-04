"""Unit tests for the trade-ledger config loader, notifier messages and CLI wiring."""

import base64
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.portfolio.trade_ledger import cli
from src.portfolio.trade_ledger.config import DEFAULT_CONFIG_PATH, config_from_dict, load_config
from src.portfolio.trade_ledger.ledger_engine import build_ledger
from src.portfolio.trade_ledger.models import LedgerEvents, ReconciliationReport
from src.portfolio.trade_ledger.notifier import build_failure_message, build_report_message, send_report
from src.portfolio.trade_ledger.reconciler import STATUS_NO_SNAPSHOT
from src.portfolio.trade_ledger.report_builder import ReportView, build_report_view
from src.portfolio.trade_ledger.runner import RunSummary
from src.portfolio.trade_ledger.tests.event_factories import trade


def test_shipped_config_loads() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)

    assert cfg.recipient_id == 2
    assert cfg.channels == ["email"]
    assert cfg.cron == "0 6 2 * *"
    assert cfg.monthly_dir == Path("data/portfolio/trade_ledger/raw/monthly")


@pytest.mark.parametrize("raw", [{"channels": ["telegram"]}, {"channels": []}, {"cron": "0 6 2 *"}, {"raw_dir": ""}])
def test_invalid_config_is_rejected(raw: Dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        config_from_dict(raw)


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "missing.yaml"))


def _view() -> ReportView:
    result = build_ledger(LedgerEvents(executions=[trade("2025-12-05 10:00", "10", "20")]), date(2025, 12, 1), date(2025, 12, 31))
    rec = ReconciliationReport(as_of=date(2025, 12, 31), snapshot_date=None, status=STATUS_NO_SNAPSHOT)
    return build_report_view(result, rec, datetime(2026, 1, 2, tzinfo=timezone.utc))


def test_report_message_has_html_body_and_base64_xlsx(tmp_path: Path) -> None:
    xlsx = tmp_path / "trade_ledger_2025-12.xlsx"
    xlsx.write_bytes(b"PK\x03\x04fake")

    message = build_report_message(_view(), xlsx, 2, ["email"])

    content = message["content"]
    assert content["html"].startswith("<div") and "<html" not in content["html"]
    assert "Trade Ledger" in content["message"]
    attachment = content["attachments"]["trade_ledger_2025-12.xlsx"]
    assert base64.b64decode(attachment["data"]) == b"PK\x03\x04fake"
    assert message["recipient_id"] == "2"
    assert message["message_type"] == "REPORT"


def test_send_report_returns_false_when_queue_fails(tmp_path: Path) -> None:
    xlsx = tmp_path / "r.xlsx"
    xlsx.write_bytes(b"x")

    def broken(_message: Dict[str, Any]) -> None:
        raise RuntimeError("db down")

    assert send_report(_view(), xlsx, 2, ["email"], broken) is False


def test_failure_message_lists_errors() -> None:
    message = build_failure_message("2025-12", ["download failed", "no archive"], 2, ["email"])

    assert message["content"]["title"] == "Trade Ledger — 2025-12 FAILED"
    assert "download failed\nno archive" in message["content"]["message"]


def test_cli_run_passes_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Dict[str, Any]] = []

    def fake_run_month(cfg: Any, month: Any, *, dry_run: bool, download: Any) -> RunSummary:
        calls.append({"month": month, "dry_run": dry_run, "download": download})
        return RunSummary(month="2026-10", started_at="x")

    monkeypatch.setattr(cli, "run_month", fake_run_month)

    assert cli.main(["run", "--month", "2026-10", "--no-download", "--dry-run"]) == 0
    assert calls == [{"month": date(2026, 10, 1), "dry_run": True, "download": False}]


def test_cli_run_failure_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "run_month", lambda _cfg, _month, dry_run, download: RunSummary(month="m", started_at="x", errors=["boom"]))

    assert cli.main(["run"]) == 1


def test_cli_import_missing_path(tmp_path: Path) -> None:
    assert cli.main(["import", str(tmp_path / "missing.xml")]) == 1


def test_cli_rejects_bad_month() -> None:
    with pytest.raises(SystemExit):
        cli.main(["run", "--month", "October"])

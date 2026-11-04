"""Unit tests for the trade-ledger runner (fake storage, fake downloader, fake message queue; no DB, no network)."""

import asyncio
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
from openpyxl import load_workbook

from src.portfolio.pnl_alert.flex_downloader import FlexQueryError
from src.portfolio.trade_ledger import runner
from src.portfolio.trade_ledger.config import TradeLedgerConfig
from src.portfolio.trade_ledger.runner import (
    archive_path,
    import_files,
    month_bounds,
    parse_month,
    previous_month,
    run_month,
    summary_to_dict,
)
from src.portfolio.trade_ledger.tests.fake_storage import FakeStorage
from src.portfolio.trade_ledger.tests.flex_fixture import FLEX_XML

_NOW = datetime(2026, 1, 2, 6, 0, tzinfo=timezone.utc)
_DEC = date(2025, 12, 1)


@pytest.fixture(autouse=True)
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "_flex_credentials", lambda: ("token", "1695936"))


def _cfg(tmp_path: Path) -> TradeLedgerConfig:
    return TradeLedgerConfig(recipient_id=2, raw_dir=str(tmp_path / "raw"), reports_dir=str(tmp_path / "reports"))


def _ok_download(token: str, query_id: str) -> bytes:
    assert (token, query_id) == ("token", "1695936")
    return FLEX_XML


def _failing_download(_token: str, _query_id: str) -> bytes:
    raise FlexQueryError("Flex service unavailable")


class _Outbox:
    def __init__(self, fail: bool = False) -> None:
        self.messages: List[Dict[str, Any]] = []
        self.fail = fail

    def __call__(self, message: Dict[str, Any]) -> None:
        if self.fail:
            raise RuntimeError("db down")
        self.messages.append(message)


def _run(tmp_path: Path, downloader: Any = _ok_download, outbox: Optional[_Outbox] = None, **kwargs: Any) -> Tuple[runner.RunSummary, _Outbox, FakeStorage]:
    storage = FakeStorage()
    outbox = outbox or _Outbox()
    summary = run_month(_cfg(tmp_path), _DEC, storage=storage, downloader=downloader, create_message=outbox, now=_NOW, **kwargs)
    return summary, outbox, storage


def test_month_helpers() -> None:
    assert parse_month("2026-10") == date(2026, 10, 1)
    assert previous_month(date(2026, 1, 2)) == date(2025, 12, 1)
    assert month_bounds(date(2024, 2, 15)) == (date(2024, 2, 1), date(2024, 2, 29))
    with pytest.raises(ValueError):
        parse_month("2026/10")


def test_scheduled_run_downloads_archives_imports_and_emails(tmp_path: Path) -> None:
    summary, outbox, storage = _run(tmp_path)

    assert summary.ok and summary.downloaded and summary.email_sent
    archived = archive_path(tmp_path / "raw" / "monthly", _DEC)
    assert summary.archive_file == str(archived) and archived.read_bytes() == FLEX_XML
    assert summary.imported["executions"] == 2 and storage.tables["executions"]
    assert summary.reconciliation in {"ok", "mismatch"}
    assert Path(summary.xlsx_path or "").name == "trade_ledger_2025-12.xlsx"
    assert "P&L per Ticker" in load_workbook(summary.xlsx_path or "").sheetnames
    (message,) = outbox.messages
    assert message["recipient_id"] == "2" and message["channels"] == ["email"]
    assert message["content"]["title"] == "Trade Ledger — December 2025"
    assert "trade_ledger_2025-12.xlsx" in message["content"]["attachments"]
    assert message["content"]["html"].startswith("<div")


def test_default_month_is_previous_month(tmp_path: Path) -> None:
    summary = run_month(_cfg(tmp_path), storage=FakeStorage(), downloader=_ok_download, create_message=_Outbox(), now=_NOW, dry_run=True)

    assert summary.month == "2025-12"


def test_dry_run_writes_files_but_sends_nothing(tmp_path: Path) -> None:
    summary, outbox, _ = _run(tmp_path, dry_run=True)

    assert summary.ok and not summary.email_sent
    assert Path(summary.html_path or "").exists()
    assert outbox.messages == []


def test_failed_download_falls_back_to_archive(tmp_path: Path) -> None:
    archived = archive_path(tmp_path / "raw" / "monthly", _DEC)
    archived.parent.mkdir(parents=True)
    archived.write_bytes(FLEX_XML)

    summary, outbox, _ = _run(tmp_path, downloader=_failing_download)

    assert summary.ok and not summary.downloaded and summary.email_sent
    assert summary.imported["executions"] == 2
    assert any("download failed" in n for n in summary.notes)
    assert outbox.messages[0]["content"]["title"] == "Trade Ledger — December 2025"


def test_failed_download_without_archive_fails_and_emails_failure(tmp_path: Path) -> None:
    summary, outbox, _ = _run(tmp_path, downloader=_failing_download)

    assert not summary.ok and not summary.email_sent and summary.xlsx_path is None
    (message,) = outbox.messages
    assert message["content"]["title"] == "Trade Ledger — 2025-12 FAILED"
    assert message["priority"] == "HIGH"
    assert summary_to_dict(summary)["ok"] is False


def test_missing_credentials_without_archive_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "_flex_credentials", lambda: (None, None))

    summary, outbox, _ = _run(tmp_path)

    assert not summary.ok
    assert outbox.messages[0]["content"]["title"].endswith("FAILED")
    assert any("IBKR_FLEX_TOKEN" in n for n in summary.notes)


def test_no_download_reports_from_stored_events(tmp_path: Path) -> None:
    storage = FakeStorage()
    import_files([_write(tmp_path / "in" / "a.xml")], storage)

    summary = run_month(_cfg(tmp_path), date(2025, 5, 1), download=False, storage=storage, downloader=_failing_download, create_message=_Outbox(), now=_NOW)

    assert summary.ok and summary.executions == 1 and not summary.downloaded  # the 2025-05-21 buy


def test_queue_failure_marks_run_failed(tmp_path: Path) -> None:
    summary, _, _ = _run(tmp_path, outbox=_Outbox(fail=True))

    assert not summary.ok
    assert summary.errors == ["report email could not be queued"]
    assert Path(summary.xlsx_path or "").exists()  # report stays on disk


def test_invalid_download_is_kept_under_failed(tmp_path: Path) -> None:
    summary, _, _ = _run(tmp_path, downloader=lambda _token, _query_id: b"<html>maintenance</html>")

    assert not summary.ok
    assert list((tmp_path / "raw" / "monthly" / "failed").glob("*.xml"))


def _write(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(FLEX_XML)
    return path


def test_import_files_expands_directories_and_is_idempotent(tmp_path: Path) -> None:
    folder = tmp_path / "backfill"
    _write(folder / "My_Trades_2025.xml")
    storage = FakeStorage()

    first, skipped = import_files([folder], storage)
    second, _ = import_files([folder], storage)

    assert first["executions"] == 2 and second["executions"] == 0
    assert skipped


def test_import_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        import_files([tmp_path / "nope.xml"], FakeStorage())


def test_run_once_wraps_run_month(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Tuple[Any, ...]] = []

    def fake_run_month(cfg: TradeLedgerConfig, month: Optional[date], *, dry_run: bool, download: Optional[bool]) -> runner.RunSummary:
        calls.append((month, dry_run, download))
        return runner.RunSummary(month="2025-12", started_at="x")

    monkeypatch.setattr(runner, "run_month", fake_run_month)

    summary = asyncio.run(runner.run_once(_cfg(tmp_path), _DEC, dry_run=True))

    assert summary.month == "2025-12"
    assert calls == [(_DEC, True, None)]

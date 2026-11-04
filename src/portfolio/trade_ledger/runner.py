"""
Orchestration of the monthly trade-ledger report.

`run_month` (sync) does one full run: download the "My Trades" Flex statement
(period *Last Month*), archive it verbatim, import it (idempotent), rebuild
the ledger from every stored event, reconcile with IBKR's month-end
positions, write the XLSX / HTML copies and queue the email. `run_once` is
the async wrapper the scheduler awaits; it runs the blocking work in a thread
so the scheduler's event loop is not held up by the Flex polling.

A failed download falls back to the archived file for that month. With no
archive the run fails: a short failure email is queued and
`RunSummary.ok` is False.
"""

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from src.notification.logger import setup_logger
from src.portfolio.pnl_alert.flex_downloader import FlexQueryError, download_flex_statement
from src.portfolio.trade_ledger.config import TradeLedgerConfig
from src.portfolio.trade_ledger.ledger_engine import build_ledger
from src.portfolio.trade_ledger.models import ParseResult
from src.portfolio.trade_ledger.notifier import MessageCreator, send_failure, send_report
from src.portfolio.trade_ledger.reconciler import reconcile_period
from src.portfolio.trade_ledger.report_builder import build_report_view, format_html
from src.portfolio.trade_ledger.repository import TradeLedgerStorage, load_events, store_parse_result
from src.portfolio.trade_ledger.xlsx_writer import write_xlsx
from src.portfolio.trade_ledger.xml_parser import FlexParseError, parse_flex_bytes, parse_flex_file

_logger = setup_logger(__name__)

Downloader = Callable[[str, str], bytes]


@dataclass
class RunSummary:
    """
    Machine-readable outcome of one run (stored by the scheduler as the run result).

    Attributes:
        month: Report month "YYYY-MM".
        started_at: UTC ISO timestamp.
        dry_run: Files built, no email queued.
        downloaded: A fresh Flex statement was downloaded.
        archive_file: Archived XML imported in this run (download or fallback).
        imported: Newly inserted rows per table kind.
        skipped: Parser skip counters (non-STK rows etc.).
        executions: Executions in the report month.
        reconciliation: "ok" / "mismatch" / "no_snapshot".
        warnings: Ledger warnings in the report month.
        xlsx_path: Written workbook.
        html_path: Written HTML copy.
        email_sent: The report email was queued.
        notes: Non-fatal remarks (e.g. download failed, archive used).
        errors: Fatal problems; the run counts as failed when non-empty.
    """

    month: str
    started_at: str
    dry_run: bool = False
    downloaded: bool = False
    archive_file: Optional[str] = None
    imported: Dict[str, int] = field(default_factory=dict)
    skipped: Dict[str, int] = field(default_factory=dict)
    executions: int = 0
    reconciliation: str = ""
    warnings: int = 0
    xlsx_path: Optional[str] = None
    html_path: Optional[str] = None
    email_sent: bool = False
    notes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the run had no fatal error."""
        return not self.errors


def summary_to_dict(summary: RunSummary) -> Dict[str, Any]:
    """Convert a `RunSummary` to a JSON-serializable dict (adds `ok`)."""
    return {**asdict(summary), "ok": summary.ok}


# --------------------------------------------------------------------------
# Months
# --------------------------------------------------------------------------


def parse_month(text: str) -> date:
    """
    Parse "YYYY-MM" into the first day of that month.

    Args:
        text: Month string.

    Returns:
        First day of the month.

    Raises:
        ValueError: If `text` is not "YYYY-MM".
    """
    try:
        return datetime.strptime(text.strip(), "%Y-%m").date()
    except ValueError as exc:
        raise ValueError(f"month must be YYYY-MM, got {text!r}") from exc


def previous_month(today: date) -> date:
    """First day of the calendar month before `today`."""
    return (today.replace(day=1) - timedelta(days=1)).replace(day=1)


def month_bounds(month: date) -> Tuple[date, date]:
    """
    First and last day of the month containing `month`.

    Args:
        month: Any day of the month.

    Returns:
        (first day, last day).
    """
    start = month.replace(day=1)
    next_start = (start + timedelta(days=32)).replace(day=1)
    return start, next_start - timedelta(days=1)


# --------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------


def _merge(total: Dict[str, int], part: Dict[str, int]) -> None:
    for key, value in part.items():
        total[key] = total.get(key, 0) + value


def _xml_files(paths: Iterable[Path]) -> List[Path]:
    """Expand directories to their *.xml files (sorted); keep files as given."""
    files: List[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.xml")))
        else:
            files.append(path)
    return files


def import_files(paths: Iterable[Path], storage: Optional[TradeLedgerStorage] = None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Parse and store Flex XML files (idempotent).

    Args:
        paths: Files and/or directories (every *.xml inside).
        storage: Storage backend; defaults to `TradeLedgerService`.

    Returns:
        (newly inserted rows per table kind, parser skip counters), summed over all files.

    Raises:
        FileNotFoundError: If a path does not exist.
        FlexParseError: If a file is not a Flex statement.
    """
    inserted: Dict[str, int] = {}
    skipped: Dict[str, int] = {}
    for path in _xml_files(paths):
        if not path.exists():
            raise FileNotFoundError(f"Flex XML not found: {path}")
        result = parse_flex_file(path)
        _merge(inserted, store_parse_result(result, storage))
        _merge(skipped, result.skipped)
    return inserted, skipped


def archive_path(monthly_dir: Path, month: date) -> Path:
    """Archive file of one monthly statement: `<monthly_dir>/My_Trades_YYYY-MM.xml`."""
    return monthly_dir / f"My_Trades_{month:%Y-%m}.xml"


def _statement_month(result: ParseResult, fallback: date) -> date:
    """Month a downloaded statement covers (its latest `toDate`), or `fallback`."""
    ends = [s.to_date for s in result.statements]
    return max(ends).replace(day=1) if ends else fallback


def download_and_archive(
    token: str, query_id: str, monthly_dir: Path, requested: date, downloader: Downloader = download_flex_statement
) -> Tuple[Path, ParseResult]:
    """
    Download the Flex statement, archive it verbatim and parse it.

    The archive is named after the month the statement covers, which is the
    requested month when the query period is *Last Month* and the run is on
    schedule. A statement that fails to parse is kept under `failed/`.

    Args:
        token: Flex Web Service token.
        query_id: "My Trades" query id.
        monthly_dir: Archive folder.
        requested: Report month (used to name an empty statement).
        downloader: Flex download function (injected in tests).

    Returns:
        (archive path, parse result).

    Raises:
        FlexQueryError: If the download fails.
        FlexParseError: If the statement is not valid Flex XML.
    """
    xml = downloader(token, query_id)
    monthly_dir.mkdir(parents=True, exist_ok=True)
    try:
        parsed = parse_flex_bytes(xml, "download")
    except FlexParseError:
        failed = monthly_dir / "failed" / f"My_Trades_download_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.xml"
        failed.parent.mkdir(parents=True, exist_ok=True)
        failed.write_bytes(xml)
        _logger.error("Downloaded Flex statement is not valid; kept as %s", failed)
        raise
    path = archive_path(monthly_dir, _statement_month(parsed, requested))
    path.write_bytes(xml)
    _logger.info("Archived Flex statement to %s (%d bytes)", path, len(xml))
    # Re-parse from the archived file so stored rows reference the archive name.
    return path, parse_flex_file(path)


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def _flex_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Flex token and "My Trades" query id from donotshare (imported lazily)."""
    from config.donotshare.donotshare import IBKR_FLEX_TOKEN, IBKR_FLEX_TRADES_QUERY_ID

    return IBKR_FLEX_TOKEN, IBKR_FLEX_TRADES_QUERY_ID


def _ingest(cfg: TradeLedgerConfig, month: date, summary: RunSummary, storage: Optional[TradeLedgerStorage], downloader: Downloader) -> None:
    """Download + import this month's statement, falling back to the archive; records a fatal error when neither works."""
    token, query_id = _flex_credentials()
    archived = archive_path(cfg.monthly_dir, month)
    if token and query_id:
        try:
            path, parsed = download_and_archive(token, query_id, cfg.monthly_dir, month, downloader)
            summary.downloaded = True
            summary.archive_file = str(path)
            summary.imported = store_parse_result(parsed, storage)
            summary.skipped = dict(parsed.skipped)
            if path != archived:
                summary.notes.append(f"downloaded statement covers {path.stem[-7:]}, not {month:%Y-%m}")
            return
        except (FlexQueryError, FlexParseError, OSError) as exc:
            _logger.exception("Flex download failed for %s", f"{month:%Y-%m}")
            summary.notes.append(f"download failed: {exc}")
    else:
        summary.notes.append("IBKR_FLEX_TOKEN / IBKR_FLEX_TRADES_QUERY_ID not set")

    if archived.exists():
        summary.archive_file = str(archived)
        summary.imported, summary.skipped = import_files([archived], storage)
        summary.notes.append(f"used archived {archived.name}")
        return
    summary.errors.append(f"no statement for {month:%Y-%m}: download failed and {archived} does not exist")


def run_month(
    cfg: TradeLedgerConfig,
    month: Optional[date] = None,
    *,
    dry_run: bool = False,
    download: Optional[bool] = None,
    storage: Optional[TradeLedgerStorage] = None,
    downloader: Downloader = download_flex_statement,
    create_message: Optional[MessageCreator] = None,
    now: Optional[datetime] = None,
) -> RunSummary:
    """
    Build (and email) the report of one month.

    Args:
        cfg: Loaded configuration.
        month: Any day of the report month; defaults to the previous calendar month.
        dry_run: Write the files but queue no email (also no failure email).
        download: Override `cfg.download`.
        storage: Storage backend; defaults to `TradeLedgerService`.
        downloader: Flex download function (injected in tests).
        create_message: Notification message creator (injected in tests).
        now: Current time (UTC); defaults to now.

    Returns:
        RunSummary; `ok` is False when the report could not be built or sent.
    """
    now = now or datetime.now(timezone.utc)
    start, end = month_bounds(month or previous_month(now.date()))
    label = f"{start:%Y-%m}"
    summary = RunSummary(month=label, started_at=now.isoformat(), dry_run=dry_run)
    _logger.info("Trade ledger run for %s (dry_run=%s)", label, dry_run)

    try:
        if cfg.download if download is None else download:
            _ingest(cfg, start, summary, storage, downloader)
        if summary.ok:
            _build_and_send(cfg, start, end, summary, storage, create_message, now)
    except Exception as exc:
        _logger.exception("Trade ledger run for %s failed", label)
        summary.errors.append(f"{type(exc).__name__}: {exc}")

    if summary.errors and not dry_run:
        send_failure(label, summary.errors, cfg.recipient_id, cfg.channels, create_message)
    _logger.info("Trade ledger run for %s finished: ok=%s, email_sent=%s, notes=%s", label, summary.ok, summary.email_sent, summary.notes)
    return summary


def _build_and_send(
    cfg: TradeLedgerConfig,
    start: date,
    end: date,
    summary: RunSummary,
    storage: Optional[TradeLedgerStorage],
    create_message: Optional[MessageCreator],
    now: datetime,
) -> None:
    """Rebuild the ledger, write the files and queue the email."""
    events = load_events(storage)
    result = build_ledger(events, start, end)
    reconciliation = reconcile_period(events, start, end)
    view = build_report_view(result, reconciliation, now)
    summary.executions = len(view.executions)
    summary.reconciliation = reconciliation.status
    summary.warnings = len(view.warnings)

    reports_dir = Path(cfg.reports_dir)
    xlsx_path = write_xlsx(view, result, reports_dir / f"trade_ledger_{summary.month}.xlsx")
    html_path = reports_dir / f"trade_ledger_{summary.month}.html"
    html_path.write_text(format_html(view), encoding="utf-8")
    summary.xlsx_path, summary.html_path = str(xlsx_path), str(html_path)

    if summary.dry_run:
        _logger.info("Dry run: report written to %s, no email queued", xlsx_path)
        return
    summary.email_sent = send_report(view, xlsx_path, cfg.recipient_id, cfg.channels, create_message)
    if not summary.email_sent:
        summary.errors.append("report email could not be queued")


async def run_once(cfg: TradeLedgerConfig, month: Optional[date] = None, *, dry_run: bool = False, download: Optional[bool] = None) -> RunSummary:
    """
    Async entry point for the scheduler (runs `run_month` in a worker thread).

    Args:
        cfg: Loaded configuration.
        month: Any day of the report month; defaults to the previous calendar month.
        dry_run: Write the files but queue no email.
        download: Override `cfg.download`.

    Returns:
        RunSummary of the run.
    """
    return await asyncio.to_thread(run_month, cfg, month, dry_run=dry_run, download=download)

"""
``fmp snapshot-universe`` (spec §12.1): survivorship protection that needs **no FMP key**.

Each run stores, under ``<data_root>/snapshots/<YYYY-MM-DD>/``:

- the raw source files exactly as downloaded: Nasdaq Trader's ``nasdaqlisted.txt`` and ``otherlisted.txt`` (the
  daily symbol directories of every US-listed security), and the S&P 500, Nasdaq-100 and Dow constituents;
- ``symbols.csv`` (test issues removed) and ``constituents.csv`` parsed from them;
- ``SHA256SUMS`` (``sha256sum -c`` format) over every file of the snapshot;
- ``diff.json`` / ``diff.md`` against the previous snapshot: symbols added (new listings, renames) and removed
  (delisted, renamed);
- for each removed symbol, one last yfinance download (it may return nothing), stored **raw** like the increment
  layer in ``last_bars/``, and a row in ``delisted_suspected.csv`` with the snapshot date.

The vintage is immutable, so "mark them in ``securities`` as ``delisted_suspected``" (spec §12.1) is recorded
next to the snapshot instead (D71). Sources are configurable (``fmp.snapshots``); the defaults are documented in
``docs/decisions.md``. Recommended monthly; the owner schedules it.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.data.fmp_pipeline.archive.symbol_path import encode_symbol
from src.data.fmp_pipeline.builders.identity import yf_symbol
from src.data.fmp_pipeline.fsutil import atomic_write_bytes
from src.data.fmp_pipeline.planner.config import SnapshotConfig
from src.data.fmp_pipeline.validators.checksums import write_checksums
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

SNAPSHOTS_DIR = "snapshots"
SUMS_FILE = "SHA256SUMS"
#: ``url -> body``; raises on failure.
Fetch = Callable[[str], bytes]
#: ``(yf symbols, interval, start, end) -> {symbol: yfinance frame}`` (``YahooDataDownloader.get_unadjusted_history``).
YahooFetch = Callable[[list[str], str, datetime, datetime], dict[str, pd.DataFrame]]


@dataclass(frozen=True)
class Source:
    """One downloaded file: where it comes from and how its symbols are read."""

    name: str
    url: str
    kind: str
    file: str
    column: str = ""
    match: str = ""


def sources(cfg: SnapshotConfig) -> tuple[Source, ...]:
    """The configured sources (Nasdaq Trader directories plus three index lists)."""
    return (Source("nasdaqlisted", cfg.nasdaq_listed_url, "nasdaq_listed", "nasdaqlisted.txt"),
            Source("otherlisted", cfg.other_listed_url, "other_listed", "otherlisted.txt"),
            Source("sp500", cfg.sp500_url, "csv", "sp500.csv", column="Symbol"),
            Source("nasdaq100", cfg.nasdaq100_url, "html", "nasdaq100.html", column="Ticker", match="Ticker"),
            Source("dow", cfg.dow_url, "html", "dow.html", column="Symbol", match="Symbol"))


def http_fetch(user_agent: str, timeout_s: float = 60.0) -> Fetch:
    """A plain HTTPS GET (Wikipedia requires an identifying User-Agent)."""
    import httpx

    def fetch(url: str) -> bytes:
        response = httpx.get(url, headers={"User-Agent": user_agent}, timeout=timeout_s, follow_redirects=True)
        response.raise_for_status()
        return response.content

    return fetch


def parse_directory(text: str, kind: str) -> list[dict[str, str]]:
    """
    Parse a Nasdaq Trader symbol directory (pipe-delimited, header line, ``File Creation Time`` footer).

    Returns:
        Rows with ``symbol``, ``name``, ``exchange``, ``etf``; test issues are dropped.
    """
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("File Creation Time")]
    if not lines:
        return []
    reader = csv.DictReader(lines, delimiter="|")
    out = []
    for row in reader:
        if row.get("Test Issue", "N") == "Y":
            continue
        symbol = (row.get("Symbol") if kind == "nasdaq_listed" else row.get("ACT Symbol")) or ""
        if not symbol.strip():
            continue
        exchange = "Q" if kind == "nasdaq_listed" else row.get("Exchange", "")
        out.append({"symbol": symbol.strip(), "name": (row.get("Security Name") or "").strip(),
                    "exchange": exchange or "", "etf": row.get("ETF", "") or ""})
    return out


def parse_constituents(body: bytes, source: Source) -> list[str]:
    """Symbols of an index list: a CSV column, or the first HTML table containing ``source.match``."""
    if source.kind == "csv":
        frame = pd.read_csv(io.BytesIO(body))
    else:
        tables = pd.read_html(io.StringIO(body.decode("utf-8", errors="replace")), match=source.match, flavor="lxml")
        frame = next(t for t in tables if source.column in t.columns)
    return sorted({str(s).strip() for s in frame[source.column].dropna() if str(s).strip()})


@dataclass
class SnapshotReport:
    """What one snapshot run did."""

    date: str
    directory: Path
    symbols: int = 0
    constituents: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    previous: Optional[str] = None
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    last_downloads: dict[str, int] = field(default_factory=dict)


class UniverseSnapshot:
    """Takes one universe snapshot."""

    def __init__(self, data_root: Path, config: SnapshotConfig, fetch: Optional[Fetch] = None,
                 yahoo: Optional[YahooFetch] = None, today: Optional[date] = None, download_removed: bool = True) -> None:
        """
        Args:
            data_root: Pipeline data root.
            config: ``fmp.snapshots`` settings.
            fetch: HTTP access (default ``httpx`` with the configured User-Agent).
            yahoo: yfinance access for removed symbols (default ``YahooDataDownloader.get_unadjusted_history``).
            today: Snapshot date (default today).
            download_removed: Try one last yfinance download for removed symbols.
        """
        self.root = data_root / SNAPSHOTS_DIR
        self.config = config
        self.fetch = fetch or http_fetch(config.user_agent)
        self.yahoo = yahoo
        self.today = today or date.today()
        self.download_removed = download_removed

    def run(self) -> SnapshotReport:
        """Download, parse, checksum, diff, and try the removed symbols once more."""
        directory = self.root / self.today.isoformat()
        directory.mkdir(parents=True, exist_ok=True)
        report = SnapshotReport(self.today.isoformat(), directory)
        symbols: list[dict[str, str]] = []
        constituents: list[dict[str, str]] = []
        for source in sources(self.config):
            try:
                body = self.fetch(source.url)
            except Exception as exc:  # noqa: BLE001 - one unreachable source must not lose the others
                report.errors[source.name] = f"download failed: {type(exc).__name__}: {exc}"[:300]
                _logger.error("Snapshot source %s failed: %s", source.name, exc)
                continue
            atomic_write_bytes(directory / source.file, body)
            try:
                if source.kind in ("nasdaq_listed", "other_listed"):
                    symbols += [{**row, "source": source.name}
                                for row in parse_directory(body.decode("utf-8", errors="replace"), source.kind)]
                else:
                    members = parse_constituents(body, source)
                    constituents += [{"index": source.name, "symbol": s} for s in members]
                    report.constituents[source.name] = len(members)
            except Exception as exc:  # noqa: BLE001 - the raw file is kept for a later re-parse
                report.errors[source.name] = f"parse failed: {type(exc).__name__}: {exc}"[:300]
                _logger.error("Snapshot source %s could not be parsed: %s", source.name, exc)
        unique = {row["symbol"]: row for row in sorted(symbols, key=lambda r: (r["symbol"], r["source"]))}
        report.symbols = len(unique)
        _write_csv(directory / "symbols.csv", list(unique.values()), ["symbol", "name", "exchange", "etf", "source"])
        _write_csv(directory / "constituents.csv", constituents, ["index", "symbol"])
        complete = "nasdaqlisted" not in report.errors and "otherlisted" not in report.errors
        self._diff(report, set(unique), complete)
        if self.download_removed and report.removed:
            self._last_downloads(report, directory)
        atomic_write_bytes(directory / "diff.json", json.dumps(
            {"date": report.date, "previous": report.previous, "added": report.added, "removed": report.removed,
             "last_downloads": report.last_downloads, "errors": report.errors}, indent=1).encode("utf-8"))
        atomic_write_bytes(directory / "diff.md", _render(report).encode("utf-8"))
        sums = directory / SUMS_FILE
        sums.unlink(missing_ok=True)
        write_checksums(directory, self.root / f".{report.date}.sums")
        (self.root / f".{report.date}.sums").replace(sums)
        _logger.info("Snapshot %s: %d symbols, constituents %s, %d added, %d removed, errors %s", report.date,
                     report.symbols, report.constituents, len(report.added), len(report.removed), report.errors or "none")
        return report

    def _diff(self, report: SnapshotReport, current: set[str], complete: bool) -> None:
        earlier = sorted(p for p in self.root.iterdir() if p.is_dir() and p.name < report.date
                         and (p / "symbols.csv").exists()) if self.root.is_dir() else []
        if not earlier:
            return
        previous = earlier[-1]
        report.previous = previous.name
        with (previous / "symbols.csv").open(encoding="utf-8", newline="") as fh:
            before = {row["symbol"] for row in csv.DictReader(fh)}
        report.added = sorted(current - before)
        # A failed directory download would make every symbol look removed: only diff complete snapshots.
        report.removed = sorted(before - current) if complete and current else []

    def _last_downloads(self, report: SnapshotReport, directory: Path) -> None:
        from src.data.fmp_pipeline.increments.ingest import to_raw_bars

        yahoo = self.yahoo or _default_yahoo()
        candidates = [s for s in report.removed if "$" not in s and " " not in s][: self.config.max_last_downloads]
        start = datetime(1990, 1, 1)
        end = datetime.combine(self.today + timedelta(days=1), datetime.min.time())
        rows = []
        for i in range(0, len(candidates), 50):
            batch = candidates[i: i + 50]
            try:
                frames = yahoo([yf_symbol(s) for s in batch], "1d", start, end)
            except Exception as exc:  # noqa: BLE001 - throttling or a dead ticker: record and move on
                _logger.warning("Last yfinance download failed for a batch of %d: %s", len(batch), exc)
                frames = {}
            for symbol in batch:
                frame = frames.get(yf_symbol(symbol))
                bars = to_raw_bars(frame, "1d", "America/New_York") if frame is not None and not frame.empty else None
                count = 0 if bars is None else len(bars)
                if bars is not None and count:
                    path = directory / "last_bars" / f"{encode_symbol(symbol)}.parquet"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    bars.to_parquet(path)
                report.last_downloads[symbol] = count
                last = None if bars is None or not count else str(pd.Timestamp(bars.index.max()).date())
                rows.append({"symbol": symbol, "snapshot_date": report.date, "last_bar_date": last or "",
                             "bars_downloaded": str(count)})
        _write_csv(directory / "delisted_suspected.csv", rows,
                   ["symbol", "snapshot_date", "last_bar_date", "bars_downloaded"])


def _default_yahoo() -> YahooFetch:
    from src.data.downloader.yahoo_data_downloader import YahooDataDownloader

    return YahooDataDownloader().get_unadjusted_history


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def _render(report: SnapshotReport) -> str:
    lines = [f"# Universe snapshot {report.date}", "",
             f"{report.symbols:,} listed symbols (test issues removed); constituents: "
             + (", ".join(f"{k} {v}" for k, v in report.constituents.items()) or "none") + ".", ""]
    if report.errors:
        lines += ["## Source errors", ""] + [f"- {k}: {v}" for k, v in report.errors.items()] + [""]
    if report.previous is None:
        lines.append("First snapshot: nothing to compare with.")
    else:
        lines += [f"Compared with {report.previous}: {len(report.added)} added, {len(report.removed)} removed.", "",
                  "## Removed (delisted or renamed)", ""]
        lines += [f"- {s} — last yfinance download: {report.last_downloads.get(s, 'not tried')} bars"
                  for s in report.removed] or ["- none"]
        lines += ["", "## Added", ""] + ([f"- {s}" for s in report.added] or ["- none"])
    return "\n".join(lines) + "\n"

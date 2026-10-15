"""
``fmp ingest-increments`` (spec §10.2, §10.4, plan WP-9): extend the vintage with yfinance bars, stored raw.

For every security the vintage considers alive, per timeframe:

1. The last stored bar is the later of the vintage's and the increment layer's. Only bars **strictly after** it
   are appended (rule 1); stored bars are never rewritten (rule 5).
2. yfinance is asked with ``auto_adjust=False`` and actions (``YahooDataDownloader.get_unadjusted_history``,
   rule 2), from ``overlap_days`` before the last stored bar to today.
3. Bars are converted to raw with ``yfinance_to_raw`` using the splits yfinance reports in the same download
   (rule 3). That download always ends today, so it reports every split since the window start: a long gap
   with a split in it is converted correctly (rule 5). Hourly bars use every known split instead (vintage,
   increments and the day's daily download).
4. The overlap bars are only **verified** against what is stored (relative close difference above
   ``verify_tolerance`` is reported), never written (rule 1).
5. New splits and dividends (dividends restated on the raw basis) go to the events file; the adapter unions them
   with the vintage's, FMP winning on a shared date (rule 4).
6. Hourly: yfinance keeps about 730 days. Ingestion starts no earlier than ``intraday_max_days`` ago and warns
   loudly from ``intraday_warn_days`` of gap on (rule 8). Only securities that already have hourly history
   (vintage or increments) are extended.

Exactly one machine writes (§10.4, owner decision Q4): the run refuses unless ``increments.writer_host`` names
this host, and it holds ``<data_root>/locks/increments.lock``. Everything is resumable: a stopped run (Yahoo
throttling) leaves complete files, and the next run starts from the new last bars.
"""

from __future__ import annotations

import socket
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
from yfinance.exceptions import YFRateLimitError

from src.data.fmp_pipeline.adjust.yfinance_to_raw import (
    extract_yfinance_dividends,
    extract_yfinance_splits,
    yfinance_to_raw,
)
from src.data.fmp_pipeline.builders.flags import compute_flags
from src.data.fmp_pipeline.builders.prices import Bars, daily_table, intraday_table
from src.data.fmp_pipeline.datamanager_adapter.fmp_store import AmbiguousSymbol, FmpStore, SecurityInfo
from src.data.fmp_pipeline.fsutil import exclusive_lock
from src.data.fmp_pipeline.increments.store import EVENTS_SCHEMA, INGESTED_AT, SOURCE, IncrementStore, utc_now
from src.data.fmp_pipeline.planner.config import FlagConfig, PlanConfig
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

#: ``(yf symbols, interval, start, end exclusive) -> {symbol: yfinance frame}``.
Fetch = Callable[[list[str], str, datetime, datetime], dict[str, pd.DataFrame]]
TIMEFRAMES = ("1d", "1h")


class NotTheWriter(RuntimeError):
    """This machine is not ``increments.writer_host``."""


def check_writer(writer_host: str, hostname: Optional[str] = None) -> str:
    """
    Make sure this machine is the configured increment writer (spec §10.4).

    Args:
        writer_host: ``fmp.increments.writer_host``.
        hostname: This machine's name (default ``socket.gethostname()``).

    Returns:
        The host name.

    Raises:
        NotTheWriter: ``writer_host`` is empty or names another machine.
    """
    host = hostname or socket.gethostname()
    if not writer_host:
        raise NotTheWriter("fmp.increments.writer_host is not set; exactly one machine may ingest (spec §10.4)")
    if writer_host.lower() != host.lower():
        raise NotTheWriter(f"this host is {host!r} but the increment writer is {writer_host!r}; a reader never "
                           f"writes to increments/ (spec §10.4)")
    return host


def _flag_config(store: FmpStore) -> FlagConfig:
    """The flag thresholds the vintage was built with, so increments are flagged the same way."""
    stored = store.build_info.get("build_config", {}).get("flags")
    if isinstance(stored, dict):
        values: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in stored.items()}
        return FlagConfig(**values)
    return FlagConfig()


def default_fetch() -> Fetch:
    """The production fetch: ``YahooDataDownloader.get_unadjusted_history``."""
    from src.data.downloader.yahoo_data_downloader import YahooDataDownloader

    return YahooDataDownloader().get_unadjusted_history


def to_raw_bars(df_yf: pd.DataFrame, timeframe: str, tz: str,
                splits: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Convert one yfinance frame to raw bars on the vintage's time axis.

    Args:
        df_yf: yfinance history (``auto_adjust=False``, actions).
        timeframe: ``1d`` or ``1h``.
        tz: Exchange time zone, for the local date of hourly bars.
        splits: All known splits for the conversion; default: those in ``df_yf``.

    Returns:
        Frame on a tz-naive ``DatetimeIndex`` (daily: the date; hourly: the UTC bar start) with ``open`` …
        ``volume`` and, for hourly bars, ``date_et``.

    Raises:
        ValueError: Hourly bars without a time zone (their UTC time would be a guess).
    """
    raw = yfinance_to_raw(df_yf, splits)
    index = pd.DatetimeIndex(df_yf.index)
    if timeframe == "1d":
        local = index.tz_localize(None) if index.tz is not None else index
        raw.index = local.normalize().rename("timestamp")
    else:
        if index.tz is None:
            raise ValueError("hourly yfinance bars must carry a time zone")
        raw["date_et"] = index.tz_convert(tz).tz_localize(None).normalize()
        raw.index = index.tz_convert("UTC").tz_localize(None).rename("timestamp")
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    return raw[raw["close"].notna()]


@dataclass
class TimeframeReport:
    """Counters for one timeframe of one run."""

    candidates: int = 0
    requested: int = 0
    with_data: int = 0
    updated: int = 0
    new_bars: int = 0
    events_added: int = 0
    mismatched: dict[str, int] = field(default_factory=dict)
    gap_warnings: dict[str, int] = field(default_factory=dict)


@dataclass
class IngestReport:
    """What one ``fmp ingest-increments`` run did."""

    vintage: str
    stop_reason: str = "done"
    increment_seq: int = 0
    timeframes: dict[str, TimeframeReport] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly summary for ``_meta.json``."""
        return {"vintage": self.vintage, "stop_reason": self.stop_reason, "increment_seq": self.increment_seq,
                "timeframes": {tf: vars(r) for tf, r in self.timeframes.items()}}


class IncrementIngester:
    """Runs one ingestion over the alive securities of the active vintage."""

    def __init__(self, data_root: Path, config: PlanConfig, fetch: Optional[Fetch] = None,
                 store: Optional[FmpStore] = None, today: Optional[date] = None,
                 sleep: Callable[[float], None] = time.sleep, hostname: Optional[str] = None) -> None:
        """
        Args:
            data_root: Pipeline data root (vintage, ``increments/``, ``locks/``).
            config: Pipeline config (``increments`` section).
            fetch: yfinance access (default: ``YahooDataDownloader.get_unadjusted_history``).
            store: The vintage (default: the one ``current.txt`` names).
            today: Ingestion date (default: today, UTC).
            sleep: Pause between batches (injectable for tests).
            hostname: This machine's name (default ``socket.gethostname()``).
        """
        self.data_root = data_root
        self.cfg = config.increments
        self.fetch = fetch
        self.vintage = store or FmpStore(data_root)
        self.increments = IncrementStore(data_root)
        self.today = today or datetime.now().date()
        self.sleep = sleep
        self.hostname = hostname
        self.tz = self.vintage.intraday_tz
        self.flag_config = _flag_config(self.vintage)
        self._stopped = False
        self._daily_actions: dict[int, pd.DataFrame] = {}

    def run(self, symbols: Optional[Iterable[str]] = None, timeframes: Iterable[str] = TIMEFRAMES,
            limit: Optional[int] = None) -> IngestReport:
        """
        Ingest new bars and events.

        Args:
            symbols: Only these (caller symbols, resolved in the vintage); default every alive security.
            timeframes: Subset of ``1d``, ``1h`` (and of ``increments.timeframes``).
            limit: At most this many securities (for trial runs).

        Returns:
            The run report (also saved in ``increments/_meta.json``).

        Raises:
            NotTheWriter: This machine must not write increments.
            DumpLocked: Another ingestion holds the lock.
        """
        host = check_writer(self.cfg.writer_host, self.hostname)
        fetch = self.fetch or default_fetch()
        chosen = [tf for tf in TIMEFRAMES if tf in set(timeframes) and tf in self.cfg.timeframes]
        securities = self._securities(symbols, limit)
        report = IngestReport(self.vintage.vintage)
        with exclusive_lock(self.data_root / "locks" / "increments.lock"):
            meta = self.increments.meta()
            written = False
            for timeframe in chosen:
                tf_report = report.timeframes.setdefault(timeframe, TimeframeReport())
                written |= self._run_timeframe(fetch, timeframe, securities, tf_report)
                if self._stopped:
                    report.stop_reason = "rate_limited"
                    break
            seq = int(meta.get("increment_seq", 0)) + (1 if written else 0)
            report.increment_seq = seq
            meta.update({"increment_seq": seq, "writer_host": host, "vintage": self.vintage.vintage,
                         "last_run": {"at": utc_now().isoformat(timespec="seconds"), "today": str(self.today),
                                      **report.as_dict()}})
            self.increments.write_meta(meta)
        self._log(report)
        return report

    # --- selection ------------------------------------------------------------------------------------------

    def _securities(self, symbols: Optional[Iterable[str]], limit: Optional[int]) -> list[SecurityInfo]:
        if symbols is not None:
            chosen = []
            for symbol in symbols:
                info = self._resolve(symbol)
                if info is None:
                    _logger.warning("Increments: %s is not in vintage %s; skipped", symbol, self.vintage.vintage)
                elif not info.alive:
                    _logger.warning("Increments: %s is delisted in vintage %s; skipped", symbol,
                                    self.vintage.vintage)
                else:
                    chosen.append(info)
        else:
            chosen = [s for s in self.vintage.securities() if s.alive]
        return chosen[:limit] if limit is not None else chosen

    def _resolve(self, symbol: str) -> Optional[SecurityInfo]:
        """The listing trading today; else a former symbol of exactly one security; ambiguous ones are skipped."""
        try:
            return self.vintage.resolve(symbol, as_of=self.today) or self.vintage.resolve(symbol)
        except AmbiguousSymbol as exc:
            _logger.warning("Increments: %s skipped: %s", symbol, exc)
            return None

    def _stored(self, info: SecurityInfo, timeframe: str) -> pd.DataFrame:
        """Vintage bars plus increment bars of one security (raw, sorted)."""
        parts = [p for p in (self.vintage.prices(info.security_id, timeframe),
                             self.increments.bars(info.security_id, info.symbol, timeframe, None)) if p is not None]
        if not parts:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "flags", "source"])
        stored = pd.concat(parts)
        return stored[~stored.index.duplicated(keep="first")].sort_index()

    def _start(self, timeframe: str, last: Optional[pd.Timestamp]) -> Optional[date]:
        overlap = timedelta(days=self.cfg.overlap_days)
        if last is None:
            base = self.vintage.as_of or (self.today - timedelta(days=30))
            start = base - overlap
        else:
            start = last.date() - overlap
        if timeframe == "1h":
            start = max(start, self.today - timedelta(days=self.cfg.intraday_max_days))
        return start if start <= self.today else None

    # --- one timeframe ----------------------------------------------------------------------------------------

    def _run_timeframe(self, fetch: Fetch, timeframe: str, securities: list[SecurityInfo],
                       report: TimeframeReport) -> bool:
        groups: dict[date, list[tuple[SecurityInfo, pd.DataFrame]]] = defaultdict(list)
        for info in securities:
            stored = self._stored(info, timeframe)
            if timeframe == "1h" and stored.empty:
                continue  # hourly history is only extended, never started from yfinance's 730 days
            report.candidates += 1
            last = pd.Timestamp(stored.index.max()) if not stored.empty else None
            if timeframe == "1h" and last is not None:
                gap = (self.today - last.date()).days
                if gap >= self.cfg.intraday_warn_days:
                    report.gap_warnings[info.symbol] = gap
                    _logger.error("Increments: %s 1h gap is %d days; yfinance keeps ~730, older hours are being "
                                  "lost — ingest at least every 12 months (spec §10.2.8)", info.symbol, gap)
            start = self._start(timeframe, last)
            if start is not None:
                groups[start].append((info, stored))
        written = False
        for start in sorted(groups):
            members = groups[start]
            for i in range(0, len(members), self.cfg.batch_size):
                batch = members[i: i + self.cfg.batch_size]
                frames = self._fetch(fetch, [self._yf(info) for info, _ in batch], timeframe, start)
                if frames is None:
                    return written
                report.requested += len(batch)
                for info, stored in batch:
                    df = frames.get(self._yf(info))
                    if df is None or df.empty:
                        continue
                    report.with_data += 1
                    written |= self._ingest(info, timeframe, df, stored, report)
                self.sleep(self.cfg.pause_s)
        return written

    @staticmethod
    def _yf(info: SecurityInfo) -> str:
        return info.yf_symbol or info.symbol

    def _fetch(self, fetch: Fetch, symbols: list[str], timeframe: str, start: date) -> Optional[dict[str, Any]]:
        """One batch; on throttling wait once and retry, then stop the run (resumable)."""
        begin = datetime(start.year, start.month, start.day)
        end = datetime.combine(self.today + timedelta(days=1), datetime.min.time())
        for attempt in (1, 2):
            try:
                return fetch(symbols, timeframe, begin, end)
            except YFRateLimitError:
                if attempt == 2:
                    _logger.error("Increments: Yahoo still throttling after %.0f s; stopping (re-run resumes)",
                                  self.cfg.rate_limit_wait_s)
                    self._stopped = True
                    return None
                _logger.warning("Increments: Yahoo rate limit; waiting %.0f s", self.cfg.rate_limit_wait_s)
                self.sleep(self.cfg.rate_limit_wait_s)
        return None  # pragma: no cover - loop always returns

    # --- one security -----------------------------------------------------------------------------------------

    def _ingest(self, info: SecurityInfo, timeframe: str, df_yf: pd.DataFrame, stored: pd.DataFrame,
                report: TimeframeReport) -> bool:
        sid = info.security_id
        splits = None
        if timeframe == "1h":
            splits = self._known_splits(info, extract_yfinance_splits(self._daily_actions.get(sid, df_yf.iloc[:0])))
        try:
            bars = to_raw_bars(df_yf, timeframe, self.tz, splits)
        except ValueError as exc:
            _logger.warning("Increments: %s %s skipped: %s", info.symbol, timeframe, exc)
            return False
        last = pd.Timestamp(stored.index.max()) if not stored.empty else None
        new = bars[bars.index > last] if last is not None else bars
        self._verify(info, timeframe, bars[bars.index <= last] if last is not None else bars.iloc[:0], stored,
                     report)
        written = False
        if timeframe == "1d":
            self._daily_actions[sid] = df_yf
            added = self._events(info, df_yf)
            report.events_added += added
            written = added > 0
        if new.empty:
            return written
        flags = self._flags(info, timeframe, stored, new)
        table = self._table(sid, timeframe, new, flags)
        existing = self.increments.read_table(timeframe, sid)
        merged = pa.concat_tables([existing.cast(table.schema), table]) if existing is not None else table
        self.increments.write_bars(timeframe, sid, merged)
        report.updated += 1
        report.new_bars += len(new)
        return True

    def _known_splits(self, info: SecurityInfo, from_download: pd.DataFrame) -> pd.DataFrame:
        """Every split known for the security: vintage, increments and this run's daily download."""
        fmp_splits, _ = self.vintage.events(info.security_id)
        inc_splits, _ = self.increments.events(info.security_id, info.symbol)
        frames = [f for f in (fmp_splits, inc_splits, from_download) if f is not None and not f.empty]
        if not frames:
            return pd.DataFrame(columns=["date", "numerator", "denominator"])
        splits = pd.concat([f.assign(date=pd.to_datetime(f["date"])) for f in frames], ignore_index=True)
        return splits.drop_duplicates(subset=["date"], keep="first")

    def _verify(self, info: SecurityInfo, timeframe: str, overlap: pd.DataFrame, stored: pd.DataFrame,
                report: TimeframeReport) -> None:
        if overlap.empty or stored.empty:
            return
        joined = overlap[["close"]].join(stored[["close"]], how="inner", rsuffix="_stored")
        if joined.empty:
            return
        diff = (joined["close"] / joined["close_stored"] - 1.0).abs()
        bad = int((diff > self.cfg.verify_tolerance).sum())
        if bad:
            report.mismatched[info.symbol] = bad
            _logger.warning("Increments: %s %s — %d of %d overlap bars differ from the stored raw close by more "
                            "than %.0f%% (max %.2f%%); stored bars kept", info.symbol, timeframe, bad, len(joined),
                            self.cfg.verify_tolerance * 100, float(diff.max()) * 100)

    def _flags(self, info: SecurityInfo, timeframe: str, stored: pd.DataFrame, new: pd.DataFrame) -> np.ndarray:
        """Flag the new bars in the context of the last stored ones (spike and jump need history)."""
        context = stored.tail(self.cfg.flag_context_bars)[["open", "high", "low", "close", "volume"]]
        series = pd.concat([context, new[["open", "high", "low", "close", "volume"]]]).astype(float)
        if timeframe == "1d":
            dates = series.index.to_numpy(dtype="datetime64[D]")
        else:
            local = pd.concat([pd.Series(pd.DatetimeIndex(context.index).tz_localize("UTC").tz_convert(self.tz)
                                         .tz_localize(None).normalize(), dtype="datetime64[ns]"),
                               pd.Series(new["date_et"].to_numpy(), dtype="datetime64[ns]")])
            dates = local.to_numpy(dtype="datetime64[D]")
        splits = self._known_splits(info, pd.DataFrame(columns=["date", "numerator", "denominator"]))
        _, fmp_dividends = self.vintage.events(info.security_id)
        o, h, low, c, v = (series[col].to_numpy() for col in ("open", "high", "low", "close", "volume"))
        flags, _ = compute_flags(dates, o, h, low, c, v, cfg=self.flag_config, daily=timeframe == "1d",
                                 split_dates=pd.to_datetime(splits["date"]).to_numpy(dtype="datetime64[D]"),
                                 dividend_dates=pd.to_datetime(fmp_dividends["date"]).to_numpy(dtype="datetime64[D]"))
        return flags[len(context):]

    def _table(self, sid: int, timeframe: str, new: pd.DataFrame, flags: np.ndarray) -> pa.Table:
        index = pd.DatetimeIndex(new.index)
        o, h, low, c, v = (new[col].to_numpy(dtype=float) for col in ("open", "high", "low", "close", "volume"))
        if timeframe == "1d":
            dates = index.to_numpy(dtype="datetime64[D]")
            table = daily_table(sid, Bars(dates, dates, o, h, low, c, v), flags, self.vintage.vintage, SOURCE)
        else:
            bars = Bars(index.to_numpy(dtype="datetime64[us]"),
                        pd.DatetimeIndex(new["date_et"]).to_numpy(dtype="datetime64[D]"), o, h, low, c, v)
            table = intraday_table(sid, bars, flags, self.vintage.vintage, SOURCE)
        stamp = pa.array([utc_now()] * len(new), INGESTED_AT.type)
        return table.append_column(INGESTED_AT, stamp)

    def _events(self, info: SecurityInfo, df_yf: pd.DataFrame) -> int:
        """Store yfinance splits and dividends not yet in the increment layer; return how many were added."""
        sid = info.security_id
        splits = extract_yfinance_splits(df_yf)
        dividends = extract_yfinance_dividends(df_yf)
        existing = self.increments.read_events(sid)
        known: set[tuple[str, date]] = set()
        if existing is not None:
            known = {(t, d) for t, d in zip(existing.column("type").to_pylist(), existing.column("date").to_pylist())}
        rows: list[dict[str, Any]] = []
        now = utc_now()
        for _, row in splits.iterrows():
            when = pd.Timestamp(row["date"]).date()
            if ("split", when) not in known:
                rows.append({"type": "split", "date": when, "numerator": float(row["numerator"]),
                             "denominator": float(row["denominator"]), "amount": None})
        for _, row in dividends.iterrows():
            when = pd.Timestamp(row["date"]).date()
            if ("dividend", when) not in known:
                rows.append({"type": "dividend", "date": when, "numerator": None, "denominator": None,
                             "amount": float(row["amount"])})
        if not rows:
            return 0
        new = pa.Table.from_pylist([{"security_id": sid, **r, "source": SOURCE, "ingested_at": now} for r in rows],
                                   schema=EVENTS_SCHEMA)
        merged = pa.concat_tables([existing.cast(EVENTS_SCHEMA), new]) if existing is not None else new
        self.increments.write_events(sid, merged.sort_by([("date", "ascending"), ("type", "ascending")]))
        _logger.info("Increments: %s — %d new event(s) from yfinance", info.symbol, len(rows))
        return len(rows)

    def _log(self, report: IngestReport) -> None:
        for timeframe, r in report.timeframes.items():
            _logger.info("Increments %s (vintage %s): %d candidates, %d requested, %d with data, %d updated, "
                         "%d new bars, %d events, %d with overlap mismatches, %d gap warnings", timeframe,
                         report.vintage, r.candidates, r.requested, r.with_data, r.updated, r.new_bars,
                         r.events_added, len(r.mismatched), len(r.gap_warnings))
        _logger.info("Increments run finished: %s (increment_seq %d)", report.stop_reason, report.increment_seq)

"""
Resumable download of every universe code's daily bars, splits and dividends.

One gzip JSON file per security, ``raw/history/<id % 100>/<id>.json.gz`` (named by id: codes such as ``CON`` or
``PRN`` are reserved file names on Windows), holding the payloads as delivered. ``raw/progress.jsonl`` gets one line
per finished security; a security whose last line is ``ok`` or ``missing`` is not downloaded again, one that failed
is retried on the next run.

The run stops cleanly (everything finished so far is kept) when the call budget is spent, the daily quota is
exhausted, or the key's plan cuts the history (the free plan serves one year of bars and no splits or dividends).
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.data.eodhd_pipeline import config
from src.data.eodhd_pipeline.client import EodhdClient, EodhdError, PlanRestricted, QuotaExhausted
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

#: Securities handed to the worker pool between budget checks.
_BATCH_PER_WORKER = 10


class PlanLimited(EodhdError):
    """The key's plan cuts the history or lacks the event endpoints; the download would not be complete."""


@dataclass
class DownloadSummary:
    """What one download run did."""

    planned: int = 0
    done: int = 0
    missing: int = 0
    failed: int = 0
    calls: int = 0
    skipped_done: int = 0
    stop_reason: Optional[str] = None
    plan_limited: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def pending(self) -> int:
        """Securities of this run's plan not finished (failed or not reached)."""
        return self.planned - self.done - self.missing


def history_path(data_root: Path, security_id: int) -> Path:
    """Return the raw history file of a security."""
    return data_root / config.HISTORY_DIR / f"{security_id % 100:02d}" / f"{security_id}.json.gz"


def read_history(data_root: Path, security_id: int) -> Optional[dict[str, Any]]:
    """Return a security's raw payloads (``code``, ``eod``, ``splits``, ``dividends``, …), or None if not downloaded."""
    path = history_path(data_root, security_id)
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _write_history(data_root: Path, security_id: int, payload: dict[str, Any]) -> None:
    path = history_path(data_root, security_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    tmp.replace(path)


def load_progress(data_root: Path) -> dict[str, dict[str, Any]]:
    """Return the last progress line of each code (later lines win)."""
    path = data_root / config.PROGRESS_FILE
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut by a crash
            out[record["code"]] = record
    return out


def finished_codes(progress: dict[str, dict[str, Any]]) -> set[str]:
    """Codes whose last attempt ended ``ok`` or ``missing``."""
    return {code for code, record in progress.items() if record.get("status") in ("ok", "missing")}


def fetch_security(client: EodhdClient, data_root: Path, security_id: int, code: str, allow_limited: bool = False,
                   free_plan: bool = False) -> dict[str, Any]:
    """
    Download one security's bars, splits and dividends and write its raw file.

    Args:
        client: EODHD client.
        data_root: Data root.
        security_id: Our id of the code.
        code: EODHD code (``ABC`` or ``ABC_old``).
        allow_limited: Accept a plan-limited history and missing event endpoints (free-key dry runs only).
        free_plan: The key is on the free plan; the history is marked plan-limited even without EODHD's warning.

    Returns:
        The progress record (``status`` = ``ok`` or ``missing``).

    Raises:
        PlanLimited: The plan cut the history or lacks an event endpoint and ``allow_limited`` is False.
        QuotaExhausted: The quota ran out.
        EodhdError: Any other API error.
    """
    rows, limited = client.eod_rows(code)
    limited = limited or free_plan
    if limited and not allow_limited:
        raise PlanLimited(f"{code}: EODHD cut the history to the plan's limit")
    record: dict[str, Any] = {"code": code, "security_id": security_id, "plan_limited": limited,
                              "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if rows is None:
        return {**record, "status": "missing", "bars": 0, "splits": 0, "dividends": 0}
    payload: dict[str, Any] = {"code": code, "security_id": security_id, "fetched_at": record["at"], "eod": rows,
                               "plan_limited": limited, "events_restricted": False}
    try:
        payload["splits"] = client.splits(code)
        payload["dividends"] = client.dividends(code)
    except PlanRestricted as exc:
        if not allow_limited:
            raise PlanLimited(f"{code}: {exc}") from exc
        payload.update(splits=[], dividends=[], events_restricted=True)
    _write_history(data_root, security_id, payload)
    return {**record, "status": "ok", "bars": len(rows), "splits": len(payload["splits"]), "dividends": len(payload["dividends"]),
            "events_restricted": payload["events_restricted"]}


def _append_progress(data_root: Path, records: Iterable[dict[str, Any]]) -> None:
    path = data_root / config.PROGRESS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def plan_download(universe: pd.DataFrame, progress: dict[str, dict[str, Any]], force: bool = False,
                  codes: Optional[Iterable[str]] = None, limit: Optional[int] = None) -> tuple[pd.DataFrame, int]:
    """
    Choose what to download: delisted codes first, then live ones, skipping finished codes unless ``force``.

    Returns:
        ``(rows to download, number of finished codes skipped)``.
    """
    rows = universe if codes is None else universe[universe["eodhd_code"].isin(set(codes))]
    rows = rows.sort_values(["listed", "eodhd_code"])
    done = finished_codes(progress)
    skipped = 0 if force else int(rows["eodhd_code"].isin(done).sum())
    if not force:
        rows = rows[~rows["eodhd_code"].isin(done)]
    return (rows.head(limit) if limit is not None else rows).reset_index(drop=True), skipped


def download_histories(client: EodhdClient, data_root: Path, universe: pd.DataFrame, max_calls: Optional[int] = None,
                       workers: int = 4, allow_limited: bool = False, force: bool = False,
                       codes: Optional[Iterable[str]] = None, limit: Optional[int] = None,
                       reserve: int = config.DAILY_RESERVE) -> DownloadSummary:
    """
    Download the planned securities until done or out of budget.

    Args:
        client: EODHD client (paced; shared by the worker threads).
        data_root: Data root.
        universe: ``universe.parquet`` rows.
        max_calls: Spend at most this many calls in this run.
        workers: Parallel downloads.
        allow_limited: Accept a free plan's cut history (dry run of the mechanics; the data is then not usable).
        force: Download finished codes again.
        codes: Only these EODHD codes.
        limit: Only the first ``limit`` planned codes.
        reserve: Calls of the daily quota left unused (at most a tenth of the calls left).

    Returns:
        The run summary; ``stop_reason`` says why it ended early (``budget``, ``quota``, ``plan_limited``).
    """
    rows, skipped = plan_download(universe, load_progress(data_root), force, codes, limit)
    summary = DownloadSummary(planned=len(rows), skipped_done=skipped)
    free_plan = str(client.user().get("subscriptionType", "")).lower() == "free"
    summary.plan_limited = free_plan
    if free_plan and not allow_limited:
        # a whole-history request on the free plan silently returns one year, without the warning row
        summary.stop_reason = "plan_limited"
        summary.errors.append("the key is on the free plan (one year of bars, no splits or dividends)")
        _logger.error("EODHD key is on the free plan; nothing downloaded (use --allow-limited for a dry run)")
        return summary
    left = client.calls_left()
    budget = max_calls
    if left is not None:
        usable = max(left - min(reserve, left // 10), 0)  # never hold back more than a tenth of what is left
        budget = min(budget, usable) if budget is not None else usable
        _logger.info("EODHD quota: %d calls left today; this run may use %d", left, budget)
    start_calls = client.calls
    batch = max(workers, 1) * _BATCH_PER_WORKER
    records = list(zip(rows["security_id"].astype(int), rows["eodhd_code"]))
    position = rounds = 0
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        while position < len(records):
            size = batch
            if budget is not None:
                room = (budget - (client.calls - start_calls)) // config.CALLS_PER_STOCK
                if room <= 0:
                    summary.stop_reason = "budget"
                    break
                size = min(batch, room)
            chunk = records[position:position + size]
            position += len(chunk)
            rounds += 1
            futures = [(code, pool.submit(fetch_security, client, data_root, sid, code, allow_limited, free_plan)) for sid, code in chunk]
            finished: list[dict[str, Any]] = []
            for code, future in futures:
                try:
                    finished.append(future.result())
                except PlanLimited as exc:
                    summary.stop_reason, summary.plan_limited = "plan_limited", True
                    summary.errors.append(str(exc))
                except QuotaExhausted as exc:
                    summary.stop_reason = "quota"
                    summary.errors.append(str(exc))
                except (EodhdError, OSError, ValueError) as exc:
                    summary.failed += 1
                    summary.errors.append(f"{code}: {exc}")
                    finished.append({"code": code, "status": "error", "error": str(exc)[:300],
                                     "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            _append_progress(data_root, finished)
            summary.done += sum(1 for r in finished if r["status"] == "ok")
            summary.missing += sum(1 for r in finished if r["status"] == "missing")
            summary.plan_limited = summary.plan_limited or any(r.get("plan_limited") or r.get("events_restricted") for r in finished)
            if summary.stop_reason in ("plan_limited", "quota"):
                break
            if rounds % 10 == 0:
                _logger.info("Downloaded %d of %d (%d missing, %d failed), %d calls", summary.done, summary.planned,
                             summary.missing, summary.failed, client.calls - start_calls)
    summary.calls = client.calls - start_calls
    _logger.info("Download run: %d ok, %d missing, %d failed, %d not reached, %d calls; stop: %s", summary.done, summary.missing,
                 summary.failed, summary.pending - summary.failed, summary.calls, summary.stop_reason or "complete")
    return summary

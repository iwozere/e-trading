"""
Signal Ledger — harvesters that read each pipeline's archived outputs.

Pipelines are not modified: every pipeline already writes its picks to
``results/<pipeline>/YYYY-MM-DD/``. Each harvester turns one kind of file into
``SignalRecord`` objects. Harvesting is idempotent (the ledger upserts on
``source, signal_date, ticker, side``), so the nightly job re-reads the last few
folders and a ``--full`` run backfills the whole archive.

Folder date = ``signal_date``. Entry is assumed at the next session's open,
which is conservative for P05 (it runs before the US open of its folder date).
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import date
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import pandas as pd

from src.ml.pipeline.signal_ledger.config import RESULTS_ROOT
from src.ml.pipeline.signal_ledger.schemas import SignalRecord
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_DATE_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_P19_TIERS = ("T1", "T2", "T3")


def clean_ticker(value: object) -> Optional[str]:
    """
    Normalize a ticker cell, rejecting blanks, numbers and garbage.

    Args:
        value: Raw cell value.

    Returns:
        Upper-case ticker, or None if it does not look like a symbol.
    """
    if not isinstance(value, str):
        return None
    ticker = value.strip().upper()
    return ticker if _TICKER.match(ticker) else None


def _optional_float(value: object) -> Optional[float]:
    """Return value as float, or None for blanks/NaN/non-numeric."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def dated_files(pipeline_dir: Path, pattern: str, since: Optional[date]) -> Iterator[Tuple[date, Path]]:
    """
    Yield ``(folder_date, file)`` for files matching pattern in dated result folders.

    Args:
        pipeline_dir: ``results/<pipeline>``.
        pattern: Glob pattern inside each dated folder.
        since: Skip folders dated before this.

    Yields:
        Folder date and matching file path, oldest first.
    """
    if not pipeline_dir.is_dir():
        return
    for folder in sorted(pipeline_dir.iterdir()):
        if not (folder.is_dir() and _DATE_DIR.match(folder.name)):
            continue
        folder_date = date.fromisoformat(folder.name)
        if since and folder_date < since:
            continue
        for path in sorted(folder.glob(pattern)):
            yield folder_date, path


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read a CSV, logging and skipping unreadable/empty files."""
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, UnicodeDecodeError) as exc:
        _logger.warning("Skipping unreadable %s: %s", path, exc)
        return None


def harvest_p05(results_root: Path, since: Optional[date]) -> List[SignalRecord]:
    """P05 AI Selector ``top_picks.csv``: LLM-ranked picks with confidence and bias."""
    records: List[SignalRecord] = []
    for day, path in dated_files(results_root / "p05_ai_selector", "top_picks.csv", since):
        df = _read_csv(path)
        if df is None or "ticker" not in df:
            continue
        for row in df.to_dict("records"):
            ticker = clean_ticker(row.get("ticker"))
            if not ticker:
                continue
            side = "short" if str(row.get("bias", "long")).strip().lower() == "short" else "long"
            rank = _optional_float(row.get("rank"))
            records.append(SignalRecord(
                pipeline="p05", source="p05_top_picks", signal_date=day, ticker=ticker, side=side,
                rank=int(rank) if rank is not None else None,
                score=_optional_float(row.get("confidence")),
                meta={"time_horizon": str(row.get("time_horizon", ""))},
            ))
    return records


def harvest_p06(results_root: Path, since: Optional[date]) -> List[SignalRecord]:
    """P06 EMPS2 phase-2 alerts and final universe (one source each)."""
    records: List[SignalRecord] = []
    files = (("08_phase2_alerts.csv", "p06_phase2_alerts"), ("09_final_universe.csv", "p06_final_universe"))
    for pattern, source in files:
        for day, path in dated_files(results_root / "p06_emps2", pattern, since):
            df = _read_csv(path)
            if df is None or "ticker" not in df:
                continue
            for position, row in enumerate(df.to_dict("records"), start=1):
                ticker = clean_ticker(row.get("ticker"))
                if not ticker:
                    continue
                priority = row.get("alert_priority")
                records.append(SignalRecord(
                    pipeline="p06", source=source, signal_date=day, ticker=ticker, rank=position,
                    score=_optional_float(row.get("vol_zscore")),
                    meta={"alert_priority": priority} if isinstance(priority, str) else {},
                ))
    return records


def harvest_p10(results_root: Path, since: Optional[date]) -> List[SignalRecord]:
    """P10 EMPS3 final universe, PASSED rows only, ranked by pre-breakout score."""
    records: List[SignalRecord] = []
    for day, path in dated_files(results_root / "p10_emps3", "09_final_universe.csv", since):
        df = _read_csv(path)
        if df is None or "ticker" not in df:
            continue
        if "status" in df:
            df = df[df["status"] == "PASSED"]
        if "prebreakout_score" in df:
            df = df.sort_values("prebreakout_score", ascending=False)
        for position, row in enumerate(df.to_dict("records"), start=1):
            ticker = clean_ticker(row.get("ticker"))
            if ticker:
                records.append(SignalRecord(
                    pipeline="p10", source="p10_final_passed", signal_date=day, ticker=ticker, rank=position,
                    score=_optional_float(row.get("prebreakout_score")),
                ))
    return records


def harvest_p17(results_root: Path, since: Optional[date]) -> List[SignalRecord]:
    """P17 penny screener candidates, ranked by final score; tier kept in meta."""
    records: List[SignalRecord] = []
    for day, path in dated_files(results_root / "p17_penny_stocks", "????-??-??_candidates.csv", since):
        df = _read_csv(path)
        if df is None or "ticker" not in df:
            continue
        if "final_score" in df:
            df = df.sort_values("final_score", ascending=False)
        for position, row in enumerate(df.to_dict("records"), start=1):
            ticker = clean_ticker(row.get("ticker"))
            if not ticker:
                continue
            tier = row.get("tier")
            records.append(SignalRecord(
                pipeline="p17", source="p17_candidates", signal_date=day, ticker=ticker, rank=position,
                score=_optional_float(row.get("final_score")),
                meta={"tier": tier} if isinstance(tier, str) else {},
            ))
    return records


def harvest_p19(results_root: Path, since: Optional[date]) -> List[SignalRecord]:
    """
    P19 intraday momentum alerts (tiers T1-T3) from ``shadow.sqlite``, first alert per ticker-day.

    The alert fires intraday; the ledger still enters at the next open, so this
    measures "buy the day after the alert", with the alert price kept in meta.
    """
    db_path = results_root / "p19_penny_intraday" / "shadow.sqlite"
    if not db_path.exists():
        return []
    query = (
        "SELECT date, ticker, momentum_tier, momentum_score, price, MIN(ts) FROM shadow_log "
        f"WHERE momentum_tier IN ({','.join('?' * len(_P19_TIERS))}) AND date >= ? GROUP BY date, ticker"
    )
    try:
        with sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True) as conn:
            rows = conn.execute(query, (*_P19_TIERS, (since or date.min).isoformat())).fetchall()
    except sqlite3.Error as exc:
        _logger.warning("Cannot read P19 shadow store %s: %s", db_path, exc)
        return []
    records: List[SignalRecord] = []
    for day, raw_ticker, tier, score, price, first_ts in rows:
        ticker = clean_ticker(raw_ticker)
        if ticker:
            records.append(SignalRecord(
                pipeline="p19", source="p19_momentum_alerts", signal_date=date.fromisoformat(day), ticker=ticker,
                score=_optional_float(score), horizon_days=5,
                meta={"tier": tier, "alert_price": _optional_float(price), "alert_ts": first_ts},
            ))
    return records


def harvest_p21(results_root: Path, since: Optional[date]) -> List[SignalRecord]:
    """P21 momentum monthly rebalance ``targets.json`` (held ~one month)."""
    records: List[SignalRecord] = []
    for day, path in dated_files(results_root / "p21_momentum", "targets.json", since):
        try:
            targets = json.loads(path.read_text(encoding="utf-8")).get("targets", [])
        except (OSError, json.JSONDecodeError) as exc:
            _logger.warning("Skipping unreadable %s: %s", path, exc)
            continue
        for target in targets:
            ticker = clean_ticker(target.get("ticker"))
            if ticker:
                records.append(SignalRecord(
                    pipeline="p21", source="p21_targets", signal_date=day, ticker=ticker,
                    rank=target.get("rank"), horizon_days=21,
                    meta={"sector": target.get("sector")},
                ))
    return records


HARVESTERS: Dict[str, Callable[[Path, Optional[date]], List[SignalRecord]]] = {
    "p05": harvest_p05,
    "p06": harvest_p06,
    "p10": harvest_p10,
    "p17": harvest_p17,
    "p19": harvest_p19,
    "p21": harvest_p21,
}


def harvest_all(results_root: Path = RESULTS_ROOT, since: Optional[date] = None) -> List[SignalRecord]:
    """
    Run every harvester and de-duplicate on the ledger key.

    A failing harvester is logged and skipped so one broken pipeline output never blocks the others.

    Args:
        results_root: Project ``results/`` directory.
        since: Only folders dated on/after this (None = whole archive).

    Returns:
        Unique signal records.
    """
    unique: Dict[Tuple[str, date, str, str], SignalRecord] = {}
    for name, harvester in HARVESTERS.items():
        try:
            records = harvester(results_root, since)
        except Exception:
            _logger.exception("Harvester %s failed", name)
            continue
        for record in records:
            unique.setdefault((record.source, record.signal_date, record.ticker, record.side), record)
        _logger.info("Harvested %d %s signals", len(records), name)
    return list(unique.values())

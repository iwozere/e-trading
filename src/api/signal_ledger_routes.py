"""
Signal Ledger API Routes
------------------------

Read-only endpoints behind the Web UI's Monitoring > Signals tab: the per-stream
scorecard (forward returns after costs vs a benchmark), each stream's simulated
paper book, and the latest signals. Data is written by the nightly
``src/ml/pipeline/signal_ledger/jobs/run_signal_ledger.py`` job.
"""

from datetime import UTC, date, datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.auth import get_current_user
from src.data.db.models.model_users import User
from src.data.db.services.signal_ledger_service import SignalLedgerService
from src.ml.pipeline.signal_ledger.portfolio import summarize_book
from src.ml.pipeline.signal_ledger.scorecard import build_scorecard
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

router = APIRouter(prefix="/api/signals", tags=["signals"])

_BENCHMARKS = ("vt", "spy", "iwm")


def _service() -> SignalLedgerService:
    """Return the ledger service (a function so tests can patch it)."""
    return SignalLedgerService()


def _status_counts(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Count signals per source and status."""
    counts: Dict[str, Dict[str, int]] = {}
    for row in rows:
        per_source = counts.setdefault(row["source"], {})
        per_source[row["status"]] = per_source.get(row["status"], 0) + 1
    return counts


@router.get("/scorecard")
def get_scorecard(
    benchmark: str = Query("vt", description="Benchmark for excess returns: vt, spy or iwm"),
    _current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return the per-stream scorecard plus each stream's paper-book summary."""
    del _current_user  # auth gate only, value unused
    benchmark = benchmark.lower()
    if benchmark not in _BENCHMARKS:
        raise HTTPException(status_code=400, detail=f"Unknown benchmark: {benchmark}")
    try:
        service = _service()
        rows = service.list_signals()
        nav_by_source: Dict[str, List[Dict[str, Any]]] = {}
        for nav_row in service.get_book_nav():
            nav_by_source.setdefault(nav_row["source"], []).append(nav_row)
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "benchmark": benchmark.upper(),
            "streams": build_scorecard(rows, benchmark=benchmark),
            "books": {source: summarize_book(nav_rows) for source, nav_rows in nav_by_source.items()},
            "status_counts": _status_counts(rows),
        }
    except Exception:
        _logger.exception("Error building signal scorecard:")
        raise HTTPException(status_code=500, detail="Failed to build signal scorecard")


@router.get("/books/{source}/nav")
def get_book_nav(source: str, _current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Return the daily NAV of one stream's paper book (oldest first)."""
    del _current_user  # auth gate only, value unused
    try:
        rows = _service().get_book_nav(source)
    except Exception:
        _logger.exception("Error loading paper book NAV for %s:", source)
        raise HTTPException(status_code=500, detail="Failed to load paper book NAV")
    if not rows:
        raise HTTPException(status_code=404, detail=f"No paper book for source: {source}")
    return {"source": source, "nav": rows}


@router.get("/recent")
def get_recent_signals(
    source: Optional[str] = Query(None, description="Only this source"),
    since: Optional[date] = Query(None, description="Only signals dated on/after this"),
    limit: int = Query(200, ge=1, le=2000),
    _current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return the latest signals with their entry and forward returns."""
    del _current_user  # auth gate only, value unused
    try:
        rows = _service().list_signals(source=source, since=since, limit=limit, newest_first=True)
    except Exception:
        _logger.exception("Error loading recent signals:")
        raise HTTPException(status_code=500, detail="Failed to load recent signals")
    return {"signals": rows}

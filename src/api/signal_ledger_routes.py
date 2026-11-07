"""
Signal Ledger API Routes
------------------------

Read-only endpoints behind the Web UI's Monitoring > Signals tab: the per-stream
scorecard (forward returns after costs vs a benchmark), each stream's simulated
paper books per exit policy (NAV, policy comparison, trades, open positions),
and the latest signals. Data is written by the nightly
``src/ml/pipeline/signal_ledger/jobs/run_signal_ledger.py`` job.
"""

from datetime import UTC, date, datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.auth import get_current_user
from src.data.db.models.model_users import User
from src.data.db.services.signal_ledger_service import SignalLedgerService
from src.ml.pipeline.signal_ledger.exits import DEFAULT_POLICY, POLICIES, policy_by_name
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
        for nav_row in service.get_book_nav(policy=DEFAULT_POLICY.name):
            nav_by_source.setdefault(nav_row["source"], []).append(nav_row)
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "benchmark": benchmark.upper(),
            "streams": build_scorecard(rows, benchmark=benchmark),
            "books": {source: summarize_book(nav_rows) for source, nav_rows in nav_by_source.items()},
            "status_counts": _status_counts(rows),
            "policies": [{"name": p.name, "description": p.description} for p in POLICIES],
        }
    except Exception:
        _logger.exception("Error building signal scorecard:")
        raise HTTPException(status_code=500, detail="Failed to build signal scorecard")


def _policy_name(policy: str) -> str:
    """Validate an exit policy name (400 if unknown)."""
    if policy_by_name(policy) is None:
        raise HTTPException(status_code=400, detail=f"Unknown exit policy: {policy}")
    return policy


@router.get("/books/{source}/nav")
def get_book_nav(
    source: str,
    policy: str = Query(DEFAULT_POLICY.name, description="Exit policy"),
    _current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return the daily NAV of one stream's paper book under one exit policy (oldest first)."""
    del _current_user  # auth gate only, value unused
    policy = _policy_name(policy)
    try:
        rows = _service().get_book_nav(source, policy)
    except Exception:
        _logger.exception("Error loading paper book NAV for %s/%s:", source, policy)
        raise HTTPException(status_code=500, detail="Failed to load paper book NAV")
    if not rows:
        raise HTTPException(status_code=404, detail=f"No paper book for source {source} under policy {policy}")
    return {"source": source, "policy": policy, "nav": rows}


@router.get("/books/{source}/policies")
def get_book_policies(source: str, _current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Compare one stream's exit policies (nightly summaries).

    Per policy: ``book`` (capped paper book vs VT), ``trades`` (its trades) and ``all_signals``
    (every signal traded without the position cap — the fair way to compare exit rules).
    """
    del _current_user  # auth gate only, value unused
    try:
        stats_rows = _service().get_book_stats(source)
    except Exception:
        _logger.exception("Error comparing exit policies for %s:", source)
        raise HTTPException(status_code=500, detail="Failed to compare exit policies")
    if not stats_rows:
        raise HTTPException(status_code=404, detail=f"No paper book for source: {source}")
    by_policy = {row["policy"]: row for row in stats_rows}
    policies = [
        {"name": p.name, "description": p.description, **by_policy[p.name]["stats"]}
        for p in POLICIES if p.name in by_policy
    ]
    computed = max((row["computed_at"] for row in stats_rows if row.get("computed_at")), default=None)
    return {"source": source, "computed_at": computed, "policies": policies}


@router.get("/books/{source}/trades")
def get_book_trades(
    source: str,
    policy: str = Query(DEFAULT_POLICY.name, description="Exit policy"),
    open_only: bool = Query(False, description="Only positions still held"),
    limit: int = Query(500, ge=1, le=5000),
    _current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return one stream/policy book's paper trades, newest entry first."""
    del _current_user  # auth gate only, value unused
    policy = _policy_name(policy)
    try:
        trades = _service().get_book_trades(source=source, policy=policy, open_only=open_only, limit=limit)
    except Exception:
        _logger.exception("Error loading paper trades for %s/%s:", source, policy)
        raise HTTPException(status_code=500, detail="Failed to load paper trades")
    return {"source": source, "policy": policy, "trades": trades}


@router.get("/positions")
def get_open_positions(
    policy: str = Query(DEFAULT_POLICY.name, description="Exit policy"),
    _current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return every stream's currently open paper positions under one exit policy."""
    del _current_user  # auth gate only, value unused
    policy = _policy_name(policy)
    try:
        positions = _service().get_book_trades(policy=policy, open_only=True)
    except Exception:
        _logger.exception("Error loading open paper positions for %s:", policy)
        raise HTTPException(status_code=500, detail="Failed to load open positions")
    return {"policy": policy, "positions": positions}


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

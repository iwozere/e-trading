"""
Portfolio PnL alert runner.

Orchestrates the pipeline: load watchlist, pull IBKR positions, fetch current
prices, evaluate PnL, and dispatch one combined notification.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.notification.logger import setup_logger
from src.portfolio.pnl_alert.config import PnLAlertConfig
from src.portfolio.pnl_alert.notifier import send_alert
from src.portfolio.pnl_alert.pnl_evaluator import AlertRow, evaluate
from src.portfolio.pnl_alert.position_aggregator import aggregate_holdings
from src.portfolio.pnl_alert.price_fetcher import fetch_latest_closes
from src.portfolio.pnl_alert.watchlist_loader import load_watchlist

_logger = setup_logger(__name__)


@dataclass
class RunSummary:
    """
    Machine-readable outcome of one pipeline run.

    Attributes:
        ran_at: UTC timestamp when the run started.
        watchlist_count: Number of parsed watchlist entries.
        ibkr_count: Number of raw IBKR positions used (after STK filter).
        holdings_count: Total merged holdings.
        priced_count: Holdings for which a current price was obtained.
        alert_row_count: Rows above threshold (included in the notification).
        notification_sent: Whether the notifier reported success.
        dry_run: True if delivery was skipped.
        conflicts: Symbols present in both IBKR and watchlist (IBKR won).
        errors: Non-fatal errors collected during the run.
    """

    ran_at: str
    watchlist_count: int = 0
    ibkr_count: int = 0
    holdings_count: int = 0
    priced_count: int = 0
    alert_row_count: int = 0
    notification_sent: bool = False
    dry_run: bool = False
    conflicts: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


async def _build_ibkr_broker() -> Optional[Any]:
    """
    Build and connect an `IBKRBroker` instance using environment-sourced
    credentials. Returns `None` on failure.
    """
    try:
        from config.donotshare.donotshare import IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID
        from src.trading.broker.ibkr_broker import IBKRBroker
    except ImportError:
        _logger.exception("Could not import IBKR broker dependencies")
        return None

    if not IBKR_HOST or not IBKR_PORT:
        _logger.warning("IBKR_HOST / IBKR_PORT not configured; skipping IBKR positions")
        return None

    try:
        port = int(IBKR_PORT)
        client_id = int(IBKR_CLIENT_ID or 1)
    except ValueError:
        _logger.exception("IBKR env vars are not numeric; skipping IBKR positions")
        return None

    broker = IBKRBroker(host=IBKR_HOST, port=port, client_id=client_id)
    try:
        connected = await broker.connect()
    except Exception:
        _logger.exception("IBKR connect() raised; skipping IBKR positions")
        return None

    if not connected:
        _logger.warning("IBKR connect() returned False; skipping IBKR positions")
        return None

    return broker


async def run_once(
    cfg: PnLAlertConfig,
    *,
    dry_run: bool = False,
    threshold_override: Optional[float] = None,
    broker: Optional[Any] = None,
    data_manager: Optional[Any] = None,
    client: Optional[Any] = None,
) -> RunSummary:
    """
    Execute one pipeline run.

    Args:
        cfg: Loaded pipeline configuration.
        dry_run: If True, format the notification but do not send it.
        threshold_override: Optional threshold to override `cfg.threshold_pct`.
        broker: Optional pre-built `IBKRBroker` (mainly for tests).
        data_manager: Optional pre-built `DataManager` (mainly for tests).
        client: Optional pre-built `NotificationServiceClient` (for tests).

    Returns:
        `RunSummary` describing what happened.
    """
    ran_at = datetime.now(timezone.utc)
    threshold = threshold_override if threshold_override is not None else cfg.threshold_pct

    summary = RunSummary(ran_at=ran_at.isoformat(), dry_run=dry_run)

    try:
        watchlist = load_watchlist(cfg.watchlist_path)
    except FileNotFoundError:
        _logger.exception("Watchlist not found: %s", cfg.watchlist_path)
        summary.errors.append(f"watchlist_not_found:{cfg.watchlist_path}")
        return summary
    except Exception as exc:
        _logger.exception("Watchlist failed to load")
        summary.errors.append(f"watchlist_invalid:{exc}")
        return summary

    summary.watchlist_count = len(watchlist)

    owned_broker = False
    if cfg.include_ibkr and broker is None:
        broker = await _build_ibkr_broker()
        owned_broker = broker is not None

    try:
        holdings, conflicts = await aggregate_holdings(
            broker if cfg.include_ibkr else None,
            watchlist,
            stk_only=cfg.ibkr_stk_only,
        )
    finally:
        if owned_broker and broker is not None:
            try:
                await broker.disconnect()
            except Exception:
                _logger.exception("IBKR disconnect() raised")

    summary.ibkr_count = sum(1 for h in holdings if h.source == "ibkr")
    summary.holdings_count = len(holdings)
    summary.conflicts = conflicts

    if not holdings:
        _logger.info("No holdings to evaluate; exiting early")
        return summary

    symbols = [h.symbol for h in holdings]
    # fetch_latest_closes is synchronous (blocking network I/O); offload to
    # a thread pool so the scheduler's event loop is not blocked.
    import asyncio as _asyncio
    prices = await _asyncio.to_thread(fetch_latest_closes, symbols, data_manager)
    summary.priced_count = len(prices)

    if not prices:
        _logger.error("All price fetches failed; aborting run")
        summary.errors.append("all_price_fetches_failed")
        return summary

    rows: List[AlertRow] = evaluate(holdings, prices, threshold)
    summary.alert_row_count = len(rows)

    if not rows:
        _logger.info("No symbols above threshold; no notification sent")
        return summary

    sent = await send_alert(
        rows=rows,
        channels=cfg.channels,
        threshold_pct=threshold,
        recipient_id=cfg.recipient_id,
        client=client,
        dry_run=dry_run,
        as_of=ran_at,
    )
    summary.notification_sent = bool(sent)
    return summary


def summary_to_dict(summary: RunSummary) -> Dict[str, Any]:
    """Convert a `RunSummary` to a plain dict for JSON serialization."""
    return asdict(summary)

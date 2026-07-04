"""
Portfolio PnL alert runner.

Orchestrates the pipeline: load watchlist, pull IBKR positions, fetch current
prices, evaluate PnL, and dispatch one combined notification.
"""

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict, List

from src.notification.logger import setup_logger
from src.portfolio.pnl_alert.config import PnLAlertConfig
from src.portfolio.pnl_alert.ibkr_xml_loader import load_ibkr_xml
from src.portfolio.pnl_alert.notifier import send_alert
from src.portfolio.pnl_alert.pnl_evaluator import AlertRow, evaluate
from src.portfolio.pnl_alert.position_aggregator import (
    RawIbkrPosition,
    fetch_raw_ibkr_positions,
    merge_holdings,
)
from src.portfolio.pnl_alert.price_fetcher import fetch_latest_closes
from src.portfolio.pnl_alert.watchlist_loader import WatchlistEntry, load_watchlist

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


async def _build_ibkr_broker() -> Any | None:
    """
    Build and connect an `IBKRBroker` instance using environment-sourced
    credentials. Returns `None` on failure.
    """
    try:
        from config.donotshare.donotshare import IBKR_CLIENT_ID, IBKR_HOST, IBKR_PORT
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
    threshold_override: float | None = None,
    broker: Any | None = None,
    data_manager: Any | None = None,
    client: Any | None = None,
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
    ran_at = datetime.now(UTC)
    threshold = threshold_override if threshold_override is not None else cfg.threshold_pct

    summary = RunSummary(ran_at=ran_at.isoformat(), dry_run=dry_run)

    # --- watchlist (optional) ---
    watchlist: List[WatchlistEntry] = []
    if cfg.watchlist_path:
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

    # --- IBKR XML positions (optional) ---
    xml_positions: List[RawIbkrPosition] = []
    if cfg.ibkr_xml_path:
        try:
            xml_positions = load_ibkr_xml(cfg.ibkr_xml_path)
        except Exception as exc:
            _logger.exception("IBKR XML load failed: %s", cfg.ibkr_xml_path)
            summary.errors.append(f"ibkr_xml_failed:{exc}")

    # --- live IBKR broker (optional) ---
    owned_broker = False
    if cfg.include_ibkr and broker is None:
        broker = await _build_ibkr_broker()
        owned_broker = broker is not None

    try:
        live_ibkr: List[RawIbkrPosition] = []
        if cfg.include_ibkr and broker is not None:
            live_ibkr = fetch_raw_ibkr_positions(broker)
    finally:
        if owned_broker and broker is not None:
            try:
                await broker.disconnect()
            except Exception:
                _logger.exception("IBKR disconnect() raised")

    # Merge: live IBKR overrides XML on the same symbol; both win over watchlist.
    combined: dict[str, RawIbkrPosition] = {p.symbol: p for p in xml_positions}
    for p in live_ibkr:
        combined[p.symbol] = p

    holdings, conflicts = merge_holdings(list(combined.values()), watchlist, stk_only=cfg.ibkr_stk_only)

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

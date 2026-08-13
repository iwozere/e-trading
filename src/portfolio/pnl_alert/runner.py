"""
Portfolio PnL alert runner.

Orchestrates the pipeline: pull IBKR positions (Flex Query XML export merged
with any live broker positions), fetch current prices, evaluate PnL, and
dispatch one combined notification.
"""

import asyncio
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

from src.notification.logger import setup_logger
from src.portfolio.pnl_alert.config import PnLAlertConfig
from src.portfolio.pnl_alert.flex_downloader import download_open_positions_xml
from src.portfolio.pnl_alert.ibkr_xml_loader import load_ibkr_xml
from src.portfolio.pnl_alert.notifier import send_alert
from src.portfolio.pnl_alert.pnl_evaluator import AlertRow, evaluate
from src.portfolio.pnl_alert.position_aggregator import (
    RawIbkrPosition,
    fetch_raw_ibkr_positions,
    merge_holdings,
)
from src.portfolio.pnl_alert.price_fetcher import fetch_latest_closes

_logger = setup_logger(__name__)


@dataclass
class RunSummary:
    """
    Machine-readable outcome of one pipeline run.

    Attributes:
        ran_at: UTC timestamp when the run started.
        ibkr_count: Number of raw IBKR positions used (after STK filter).
        holdings_count: Total merged holdings.
        priced_count: Holdings for which a current price was obtained.
        alert_row_count: Rows above threshold (included in the notification).
        notification_sent: Whether the notifier reported success.
        dry_run: True if delivery was skipped.
        errors: Non-fatal errors collected during the run.
    """

    ran_at: str
    ibkr_count: int = 0
    holdings_count: int = 0
    priced_count: int = 0
    alert_row_count: int = 0
    notification_sent: bool = False
    dry_run: bool = False
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
    # ib_insync logs its own ERROR line straight to the "ib_insync.client" logger
    # on a failed connection (e.g. "Make sure API port on TWS/IBG is open"),
    # independent of our own logging below. We already treat this as a
    # best-effort, non-fatal condition, so drop that logger to CRITICAL for
    # the duration of the attempt to avoid a duplicate, unactionable ERROR
    # reaching the journal / monitoring pipeline.
    ib_client_logger = logging.getLogger("ib_insync.client")
    prev_level = ib_client_logger.level
    ib_client_logger.setLevel(logging.CRITICAL)
    try:
        connected = await broker.connect()
    except Exception:
        _logger.exception("IBKR connect() raised; skipping IBKR positions")
        return None
    finally:
        ib_client_logger.setLevel(prev_level)

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

    # --- IBKR XML positions (optional) ---
    xml_positions: List[RawIbkrPosition] = []
    if cfg.ibkr_xml_path:
        try:
            # Best-effort refresh via the Flex Web Service so the XML read below
            # reflects today's positions. On failure this falls back to whatever
            # file is already on disk (same tolerance as an unreachable live IBKR).
            await asyncio.to_thread(download_open_positions_xml, Path(cfg.ibkr_xml_path).parent)
        except Exception:
            _logger.exception("Flex Query download failed; using last cached Open Positions XML")

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

    # Merge: live IBKR overrides XML on the same symbol.
    combined: dict[str, RawIbkrPosition] = {p.symbol: p for p in xml_positions}
    for p in live_ibkr:
        combined[p.symbol] = p

    holdings = merge_holdings(list(combined.values()), stk_only=cfg.ibkr_stk_only)

    summary.ibkr_count = sum(1 for h in holdings if h.source == "ibkr")
    summary.holdings_count = len(holdings)

    if not holdings:
        _logger.info("No holdings to evaluate; exiting early")
        return summary

    symbols = [h.symbol for h in holdings]
    # fetch_latest_closes is synchronous (blocking network I/O); offload to
    # a thread pool so the scheduler's event loop is not blocked.
    prices = await asyncio.to_thread(fetch_latest_closes, symbols, data_manager)
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
    summary.notification_sent = sent
    return summary


def summary_to_dict(summary: RunSummary) -> Dict[str, Any]:
    """Convert a `RunSummary` to a plain dict for JSON serialization."""
    return asdict(summary)

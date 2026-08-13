"""
Earnings-triggered stop-loss coverage reminder runner.

Orchestrates: load holdings (reused from `pnl_alert`) -> resolve earnings
events for held tickers -> for tickers whose T-1day/T-1hour trigger just
fired, check live protective-order coverage -> send one combined reminder.

Deliberately does *not* top up holdings with a live broker connection the
way `pnl_alert` does: that live top-up target is the *paper* Gateway (see
`pnl_alert.runner._build_ibkr_broker`), which is irrelevant here, and adding
a second *live* connection just for same-day holdings freshness isn't worth
the complexity for a reminder that only fires once around each earnings
date. The Flex Query XML (refreshed at the top of this run, same as
`pnl_alert`) is precise enough.
"""

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.notification.logger import setup_logger
from src.portfolio.management.config import ManagementConfig
from src.portfolio.management.coverage import evaluate as evaluate_coverage
from src.portfolio.management.earnings_source import EarningsSource
from src.portfolio.management.earnings_window import EarningsEvent, matched_trigger, resolve_anchor_utc
from src.portfolio.management.notifier import TriggeredReminder, send_reminder
from src.portfolio.management.open_orders import IBKROpenOrdersFeed, fetch_protective_qty
from src.portfolio.pnl_alert.flex_downloader import download_open_positions_xml
from src.portfolio.pnl_alert.ibkr_xml_loader import load_ibkr_xml
from src.portfolio.pnl_alert.position_aggregator import RawIbkrPosition, merge_holdings

_logger = setup_logger(__name__)


@dataclass
class RunSummary:
    """
    Machine-readable outcome of one pipeline run.

    Attributes:
        ran_at: UTC timestamp when the run started.
        holdings_count: Held tickers loaded from the Flex Query XML.
        earnings_events_count: Held tickers with an earnings date within the
            lookahead window.
        triggered_count: Tickers whose T-1day/T-1hour trigger fired this run.
        notification_sent: Whether the notifier reported success.
        dry_run: True if delivery was skipped.
        errors: Non-fatal errors collected during the run.
    """

    ran_at: str
    holdings_count: int = 0
    earnings_events_count: int = 0
    triggered_count: int = 0
    notification_sent: bool = False
    dry_run: bool = False
    errors: List[str] = field(default_factory=list)


def _load_holdings(cfg: ManagementConfig, summary: RunSummary) -> Dict[str, float]:
    """Refresh + parse the Flex Query XML export into {ticker: quantity}."""
    if cfg.ibkr_xml_path:
        try:
            download_open_positions_xml(Path(cfg.ibkr_xml_path).parent)
        except Exception:
            _logger.exception("Flex Query download failed; using last cached Open Positions XML")

    try:
        xml_positions: List[RawIbkrPosition] = load_ibkr_xml(cfg.ibkr_xml_path) if cfg.ibkr_xml_path else []
    except Exception as exc:
        _logger.exception("IBKR XML load failed: %s", cfg.ibkr_xml_path)
        summary.errors.append(f"ibkr_xml_failed:{exc}")
        return {}

    holdings = merge_holdings(xml_positions, stk_only=cfg.ibkr_stk_only)
    return {h.symbol: h.quantity for h in holdings}


async def run_once(
    cfg: ManagementConfig,
    *,
    as_of_date: Optional[date] = None,
    now: Optional[datetime] = None,
    dry_run: bool = False,
    open_orders_feed: Optional[IBKROpenOrdersFeed] = None,
    earnings_source: Optional[EarningsSource] = None,
    client: Optional[Any] = None,
) -> RunSummary:
    """
    Execute one pipeline run.

    Args:
        cfg: Loaded pipeline configuration.
        as_of_date: Reference date for the earnings-window lookup. Defaults
            to `now`'s date.
        now: Current time (UTC, tz-aware) used for both the run timestamp
            and T-1day/T-1hour trigger matching. Defaults to
            `datetime.now(UTC)`. Injectable so trigger-matching tests can be
            deterministic instead of racing the real clock.
        dry_run: If True, format the notification but do not send it.
        open_orders_feed: Optional pre-built (unconnected) feed — mainly for
            tests, where it doubles as any object exposing `.connect()`,
            `.protective_order_qty()`, `.disconnect()`.
        earnings_source: Optional pre-built `EarningsSource` (mainly tests).
        client: Optional pre-built `NotificationServiceClient` (for tests).

    Returns:
        `RunSummary` describing what happened.
    """
    ran_at = now or datetime.now(UTC)
    today = as_of_date or ran_at.date()
    summary = RunSummary(ran_at=ran_at.isoformat(), dry_run=dry_run)

    # --- Holdings (reused from pnl_alert's Flex Query XML pipeline) ---
    positions = await asyncio.to_thread(_load_holdings, cfg, summary)
    summary.holdings_count = len(positions)
    if not positions:
        _logger.info("No holdings to check; exiting early")
        return summary

    # --- Earnings events for held tickers ---
    src = earnings_source or EarningsSource()
    try:
        events: List[EarningsEvent] = await asyncio.to_thread(
            src.get_upcoming_events, positions.keys(), today, cfg.earnings_window_days
        )
    except Exception as exc:
        _logger.exception("Earnings calendar lookup failed")
        summary.errors.append(f"earnings_lookup_failed:{exc}")
        return summary
    summary.earnings_events_count = len(events)

    # --- Which held tickers have a T-1day/T-1hour trigger firing right now ---
    triggered: List[tuple[EarningsEvent, str]] = []
    for event in events:
        if event.ticker not in positions:
            continue
        anchor = resolve_anchor_utc(event)
        trigger = matched_trigger(ran_at, anchor, cfg.trigger_window_minutes)
        if trigger is not None:
            triggered.append((event, trigger))

    summary.triggered_count = len(triggered)
    if not triggered:
        _logger.info("No earnings triggers in-window this run (%d upcoming event(s))", len(events))
        return summary

    # --- Live open-orders coverage check (only for triggered tickers) ---
    feed = open_orders_feed or IBKROpenOrdersFeed(
        host=cfg.ibkr_live_host, port=cfg.ibkr_live_port, client_id=cfg.ibkr_live_client_id
    )
    triggered_symbols = [event.ticker for event, _ in triggered]
    connected, protective_qty = await asyncio.to_thread(fetch_protective_qty, feed, triggered_symbols)
    if not connected:
        _logger.warning("Live IBKR unreachable; cannot verify stop coverage this run")
        summary.errors.append("live_ibkr_unreachable")

    triggered_positions = {event.ticker: positions[event.ticker] for event, _ in triggered}
    coverage_by_ticker = {row.ticker: row for row in evaluate_coverage(triggered_positions, protective_qty)}

    reminders = [
        TriggeredReminder(event=event, trigger=trigger, coverage=coverage_by_ticker[event.ticker])
        for event, trigger in triggered
    ]

    sent = await send_reminder(
        reminders=reminders,
        channels=cfg.channels,
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

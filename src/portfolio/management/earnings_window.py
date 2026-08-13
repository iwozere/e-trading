"""
Earnings-relative trigger window math.

Resolves a ticker's earnings date + best-effort session (BMO/AMC) into a
concrete "anchor" timestamp, and checks whether "now" falls inside the
T-1 day / T-1 hour trigger windows around that anchor. Pure functions, no I/O.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)

TRIGGER_T_MINUS_1_DAY = "t_minus_1_day"
TRIGGER_T_MINUS_1_HOUR = "t_minus_1_hour"


@dataclass(frozen=True)
class EarningsEvent:
    """
    One held ticker's upcoming earnings event.

    Attributes:
        ticker: Ticker symbol.
        earnings_date: Calendar date of the earnings release (exchange-local,
            i.e. America/New_York for US equities).
        session: "bmo" (before market open), "amc" (after market close), or
            "unknown" when the data source didn't report a session. Defaults
            to "unknown" since `EarningsCalendar` (the current source) does
            not surface session timing — see `earnings_source.py`.
    """

    ticker: str
    earnings_date: date
    session: str = field(default="unknown")


def resolve_anchor_utc(event: EarningsEvent) -> datetime:
    """
    Resolve an earnings event to a concrete UTC anchor timestamp.

    "amc" anchors at market close (16:00 ET) that day. "bmo" and "unknown"
    both anchor at market open (09:30 ET) that day — defaulting unknown to
    the earlier anchor means an unknown-session ticker still gets its
    T-1day/T-1hour triggers as early as possible rather than as late as
    possible, the safer direction for a coverage reminder (more lead time,
    not less).

    Args:
        event: Earnings event with date and (best-effort) session.

    Returns:
        UTC datetime anchor.
    """
    local_time = _MARKET_CLOSE if event.session == "amc" else _MARKET_OPEN
    local_dt = datetime.combine(event.earnings_date, local_time, tzinfo=_ET)
    return local_dt.astimezone(timezone.utc)


def matched_trigger(now_utc: datetime, anchor_utc: datetime, window_minutes: int) -> Optional[str]:
    """
    Check whether `now_utc` falls inside the T-1day or T-1hour window around
    `anchor_utc`.

    Args:
        now_utc: Current time (UTC, tz-aware).
        anchor_utc: Resolved earnings anchor (UTC, tz-aware; see
            `resolve_anchor_utc`).
        window_minutes: Half-width of the match window in minutes. Should be
            >= half the polling cadence so no trigger moment is skipped
            between two consecutive runs.

    Returns:
        `TRIGGER_T_MINUS_1_DAY`, `TRIGGER_T_MINUS_1_HOUR`, or `None` if
        neither trigger is currently in-window. When both would match (only
        possible with an unrealistically large `window_minutes`),
        `TRIGGER_T_MINUS_1_DAY` wins.
    """
    window = timedelta(minutes=window_minutes)
    t_minus_1_day = anchor_utc - timedelta(days=1)
    t_minus_1_hour = anchor_utc - timedelta(hours=1)

    if abs(now_utc - t_minus_1_day) <= window:
        return TRIGGER_T_MINUS_1_DAY
    if abs(now_utc - t_minus_1_hour) <= window:
        return TRIGGER_T_MINUS_1_HOUR
    return None

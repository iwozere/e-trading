"""
Pure analysis for the paid probe (spec §3 unknowns, §8.4, Appendix A): turn recorded responses into findings.

No I/O and no network: every function takes rows (as FMP returned them) and returns a small result, so each rule
is unit-tested on synthetic data.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional, cast

import pandas as pd

from src.data.fmp_pipeline.validators.scan import SESSION_OPEN_MINUTES, nyse_calendar

ET = "America/New_York"
#: Candidate interpretations of an intraday ``date`` string: (time zone it is written in, what it labels).
CONVENTIONS = ((ET, "start"), (ET, "end"), ("UTC", "start"), ("UTC", "end"))


@dataclass(frozen=True)
class HourlyConventions:
    """What the hourly bars of regular sessions say about FMP's conventions (spec §8.4)."""

    timezone: Optional[str]
    label: Optional[str]
    bars_per_day: Optional[int]
    extended_hours: bool
    days: int
    early_close_bars: dict[str, int] = field(default_factory=dict)
    first_bars: dict[str, int] = field(default_factory=dict)

    @property
    def conclusive(self) -> bool:
        """True when exactly one convention puts every regular day's first bar at 09:30 ET."""
        return self.timezone is not None


def _parse(value: Any) -> Optional[datetime]:
    try:
        return datetime.strptime(str(value)[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _to_et_start(stamps: list[datetime], tz: str, label: str, minutes: int) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(stamps).tz_localize(tz, ambiguous="NaT", nonexistent="NaT").tz_convert(ET)
    return index - pd.Timedelta(minutes=minutes) if label == "end" else index


def _expected_grid(close: int, minutes: int) -> list[int]:
    """Bar start minutes of a regular session: 09:30, then every ``minutes`` while before the close."""
    return list(range(SESSION_OPEN_MINUTES, close, minutes))


def _read(stamps: list[datetime], tz: str, label: str, minutes: int) -> pd.DataFrame:
    """The stamps read one way, as ET bar starts: columns ``day`` (date) and ``minute`` (after midnight)."""
    et = _to_et_start(stamps, tz, label, minutes).dropna()
    return pd.DataFrame({"day": [d.date() for d in et.tz_localize(None).normalize()],
                         "minute": et.hour * 60 + et.minute})


def hourly_conventions(rows: Iterable[dict[str, Any]], minutes: int = 60, min_fit: float = 0.9) -> HourlyConventions:
    """
    Infer time zone, bar label, bars per regular day and extended hours from hourly rows.

    Each candidate reading (ET or UTC, bar start or end) converts the stamps to ET bar starts. A regular session
    *fits* a reading when its in-session bars are exactly 09:30, 10:30, … up to the close (fewer on early
    closes). The reading that fits at least ``min_fit`` of the sessions, and fits more of them than any other, is
    the convention; the sample's DST weeks separate ET from UTC (spec §8.4).

    Args:
        rows: Hourly records with a ``date`` string (``YYYY-MM-DD HH:MM:SS``), possibly several windows.
        minutes: Bar length.
        min_fit: Share of sessions that must fit.

    Returns:
        The findings; ``timezone``/``label`` are ``None`` when no single reading fits.
    """
    stamps = sorted({s for s in (_parse(r.get("date")) for r in rows) if s is not None})
    if not stamps:
        return HourlyConventions(None, None, None, False, 0)
    calendar = nyse_calendar()
    first_bars = Counter(min(s for s in stamps if s.date() == day).strftime("%H:%M")
                         for day in sorted({s.date() for s in stamps}))
    scored = []
    for tz, label in CONVENTIONS:
        frame = _read(stamps, tz, label, minutes)
        sessions = fits = 0
        for day, group in frame.groupby("day"):
            close = calendar.close_minutes.get(cast(date, day))
            if close is None:
                continue
            sessions += 1
            inside = sorted(m for m in group["minute"] if SESSION_OPEN_MINUTES <= m < close)
            fits += int(inside == _expected_grid(close, minutes))
        scored.append((fits / sessions if sessions else 0.0, tz, label, frame))
    scored.sort(key=lambda item: -item[0])
    best = scored[0]
    if best[0] < min_fit or (len(scored) > 1 and scored[1][0] == best[0]):
        return HourlyConventions(None, None, None, False, len(first_bars), {}, dict(first_bars))
    _, tz, label, frame = best
    counts: Counter[int] = Counter()
    early: dict[str, int] = {}
    extended = False
    for day, group in frame.groupby("day"):
        close = calendar.close_minutes.get(cast(date, day))
        if close is None:
            extended = True  # bars on a day without a regular session
            continue
        inside = [m for m in group["minute"] if SESSION_OPEN_MINUTES <= m < close]
        extended |= len(inside) != len(group)
        if close == 16 * 60:
            counts[len(inside)] += 1
        else:
            early[str(day)] = len(inside)
    return HourlyConventions(tz, label, counts.most_common(1)[0][0] if counts else None, extended,
                             int(frame["day"].nunique()), early, dict(first_bars))


@dataclass(frozen=True)
class RowCapProbe:
    """One wide intraday request: what was asked and what came back."""

    start: date
    end: date
    rows: int
    first: Optional[date]
    last: Optional[date]
    expected: int


@dataclass(frozen=True)
class RowCap:
    """The row cap verdict."""

    max_rows: Optional[int]
    capped: bool
    keeps: Optional[str]
    largest_full: int


def expected_bars(start: date, end: date, bars_per_day: int) -> int:
    """Regular-session bars between two dates (NYSE calendar)."""
    sessions = nyse_calendar().expected_between(pd.Timestamp(start).to_datetime64().astype("datetime64[D]"),
                                                pd.Timestamp(end).to_datetime64().astype("datetime64[D]"))
    return int(len(sessions)) * bars_per_day


def row_cap(probes: Iterable[RowCapProbe], tolerance: float = 0.95) -> RowCap:
    """
    Decide whether intraday requests are capped, at how many rows, and which end of the window is kept.

    A response is capped when it returned fewer than ``tolerance`` × the expected bars. The cap is the largest
    such row count (they should all be equal). Without any capped response, the largest complete response is a
    safe ``max_rows``: the planner sizes windows from it (with its own 1.25 safety factor, spec §6.5).
    """
    probes = list(probes)
    capped = [p for p in probes if p.expected and p.rows < tolerance * p.expected]
    full = [p.rows for p in probes if p not in capped]
    if not capped:
        return RowCap(max(full) if full else None, False, None, max(full, default=0))
    cap = max(p.rows for p in capped)
    widest = max(capped, key=lambda p: (p.end - p.start).days)
    keeps = None
    if widest.first is not None and widest.last is not None:
        near_end = (widest.end - widest.last).days <= 7
        near_start = (widest.first - widest.start).days <= 7
        keeps = "newest" if near_end and not near_start else "oldest" if near_start and not near_end else None
    return RowCap(cap, True, keeps, max(full, default=0))


def first_date(rows: Iterable[dict[str, Any]]) -> Optional[date]:
    """Earliest ``date`` among rows (daily or intraday strings)."""
    dates = sorted(str(r["date"])[:10] for r in rows if r.get("date"))
    return date.fromisoformat(dates[0]) if dates else None


def last_date(rows: Iterable[dict[str, Any]]) -> Optional[date]:
    """Latest ``date`` among rows."""
    dates = sorted(str(r["date"])[:10] for r in rows if r.get("date"))
    return date.fromisoformat(dates[-1]) if dates else None


def intraday_split_verdict(rows: Iterable[dict[str, Any]], split_date: str, ratio: float,
                           tol: float = 0.15) -> str:
    """``raw`` if the last bar before the split and the first after differ by ``ratio``, ``split_adjusted`` if
    about equal, else ``unclear``/``no_data`` (spec §3 unknown 6)."""
    bars = sorted((str(r["date"]), float(r["close"])) for r in rows if r.get("close") is not None and r.get("date"))
    before = [b for b in bars if b[0][:10] < split_date]
    after = [b for b in bars if b[0][:10] >= split_date]
    if not before or not after:
        return "no_data"
    value = before[-1][1] / after[0][1]
    if abs(value / ratio - 1) <= tol:
        return "raw"
    if abs(value - 1) <= tol:
        return "split_adjusted"
    return "unclear"


def window_complete(rows: list[dict[str, Any]], start: date, end: date, slack_days: int = 10) -> bool:
    """True when a windowed response reaches both ends of the window (calendar maximum-window test)."""
    first, last = first_date(rows), last_date(rows)
    if first is None or last is None:
        return False
    return (first - start).days <= slack_days and (end - last).days <= slack_days


def reuse_verdict(listings: list[dict[str, Any]], prices: list[dict[str, Any]]) -> str:
    """
    What FMP's price series for a reused ticker contains (spec §4.2.8, Appendix A Q10).

    Args:
        listings: ``delisted-companies`` rows of the symbol (``ipoDate``, ``delistedDate``).
        prices: The symbol's daily rows over the whole history.

    Returns:
        ``merged series`` (bars on both sides of an earlier listing's delisting date), ``latest listing only``,
        ``earlier listing only`` or ``no data``.
    """
    first, last = first_date(prices), last_date(prices)
    if first is None or last is None:
        return "no data"
    delistings = sorted(d for d in (str(r.get("delistedDate") or "")[:10] for r in listings) if d)
    if not delistings:
        return "latest listing only"
    boundary = date.fromisoformat(delistings[0])
    if first < boundary - timedelta(days=30) and last > boundary + timedelta(days=30):
        return "merged series"
    return "latest listing only" if first > boundary else "earlier listing only"

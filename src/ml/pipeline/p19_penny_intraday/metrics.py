"""
P19 intraday metrics (pure functions).

Turns a raw IBKR delayed quote + the watchlist baseline into an ``IntradaySignal``:
% move from open / prev-close, dollar volume, and **RVOL-so-far**.

RVOL-so-far = cumulative day volume ÷ *expected* cumulative volume by this time of
day. The expected fraction uses a **linear** approximation of the regular session
(9:30–16:00 ET) for now — a placeholder until the shadow dataset yields a real
U-shaped intraday volume profile (spec §4.2). Because the raw day volume and the
timestamp are both logged, the true profile can be back-computed later.

Volume units: IBKR `reqMktData` day volume for US equities is typically reported in
**round lots (×100 shares)**; ``lot_size`` (default 100) converts to shares so RVOL
lines up with the share-based ``avg_volume_30d`` baseline. Verify the factor against
live numbers on the Pi during market hours and adjust if needed.
"""

from datetime import datetime, time
from typing import Any, Dict
from zoneinfo import ZoneInfo

from src.ml.pipeline.p19_penny_intraday.models.intraday_signal import IntradaySignal
from src.ml.pipeline.p19_penny_intraday.models.watchlist_entry import WatchlistEntry

_ET = ZoneInfo("America/New_York")
_OPEN = time(9, 30)
_CLOSE = time(16, 0)
_SESSION_MINUTES = 6 * 60 + 30  # 390


def session_fraction(ts_utc: datetime) -> float:
    """
    Fraction (0, 1] of the regular session elapsed at ``ts_utc``.

    Clamped to a small positive floor before/at the open so RVOL never divides by
    zero; 1.0 at/after the close. Linear placeholder (see module docstring).
    """
    et = ts_utc.astimezone(_ET)
    now = et.time()
    if now <= _OPEN:
        return 0.05  # pre/at open: avoid div-by-zero, treat as early
    if now >= _CLOSE:
        return 1.0
    elapsed = (et.hour * 60 + et.minute) - (_OPEN.hour * 60 + _OPEN.minute)
    return max(0.05, min(1.0, elapsed / _SESSION_MINUTES))


def _pct(curr: float, ref: float) -> float:
    return (curr / ref - 1.0) if ref and ref > 0 else 0.0


def compute_signal(
    entry: WatchlistEntry,
    quote: Dict[str, Any],
    ts_utc: datetime,
    lot_size: int = 100,
) -> IntradaySignal:
    """
    Build an IntradaySignal from a watchlist entry + a raw delayed quote.

    Args:
        entry: Watchlist baseline context (avg volume, prior close, dilution…).
        quote: Raw IBKR fields — ``last``/``open``/``high``/``low``/``prev_close``/``volume``.
        ts_utc: Snapshot time (UTC, tz-aware).
        lot_size: Multiplier converting IBKR day volume to shares.
    """
    price = float(quote.get("last") or 0.0)
    day_open = float(quote.get("open") or 0.0)
    day_high = float(quote.get("high") or 0.0)
    day_low = float(quote.get("low") or 0.0)
    prev_close = float(quote.get("prev_close") or entry.prior_close or 0.0)
    raw_vol = float(quote.get("volume") or 0.0)
    day_volume = raw_vol * lot_size if raw_vol > 0 else 0.0

    avg_vol = entry.avg_volume_30d
    expected = avg_vol * session_fraction(ts_utc) if avg_vol > 0 else 0.0
    rvol = (day_volume / expected) if expected > 0 else 0.0

    return IntradaySignal(
        ticker=entry.ticker,
        ts=ts_utc,
        source=entry.source,
        price=price,
        day_open=day_open,
        day_high=day_high,
        day_low=day_low,
        prev_close=prev_close,
        pct_from_open=_pct(price, day_open),
        pct_from_prev_close=_pct(price, prev_close),
        day_volume=day_volume,
        avg_volume_30d=avg_vol,
        rvol_so_far=round(rvol, 3),
        dollar_volume_so_far=round(price * day_volume, 2),
        volume_is_delayed=True,
        fresh_catalyst=entry.has_catalyst,
        catalyst_signals=list(entry.catalyst_signals),
        dilution_penalty=entry.dilution_penalty,
        tier=entry.tier,
    )

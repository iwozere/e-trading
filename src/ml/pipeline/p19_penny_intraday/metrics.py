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
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from src.ml.pipeline.p19_penny_intraday.config import P19TriggerConfig
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


def classify_momentum_tier(signal: IntradaySignal, cfg: P19TriggerConfig) -> Tuple[float, str]:
    """
    Momentum-only tier classification (spec v2 §4.3/§8.1) — **momentum evidence
    only, no structural term** (decision #6: the two axes stay orthogonal).

    Log-only in Phase 1.5 (spec §16 item 5): this exists so every shadow row
    carries a "simulated trigger point" even though there is no Disposition
    Engine, Alert Manager, or dedup/escalation state yet — those are Phase 2.
    Thresholds come from ``P19TriggerConfig`` and are launch placeholders,
    calibrated later against the shadow dataset (spec §15), not hand-tuned here.

    Gate: ``volume AND (price thrust OR fresh catalyst)`` when
    ``cfg.require_volume_and_price`` (the default); otherwise any one of the
    three. A fresh bullish 8-K both satisfies the gate on its own and adds to
    the score (spec: "escalates tier, lowers thresholds").

    Returns:
        (momentum_score 0–100, momentum_tier) — tier is "T0" (no trigger),
        "T1" (elevated), "T2" (strong), or "T3" (explosive).
    """
    vol_ok = cfg.rvol_trigger > 0 and signal.rvol_so_far >= cfg.rvol_trigger and signal.dollar_volume_so_far >= cfg.dollar_volume_floor
    price_ok = cfg.move_trigger_pct > 0 and abs(signal.pct_from_open) >= cfg.move_trigger_pct
    core = price_ok or signal.fresh_catalyst
    fires = (vol_ok and core) if cfg.require_volume_and_price else (vol_ok or core)

    vol_ratio = (signal.rvol_so_far / cfg.rvol_trigger) if cfg.rvol_trigger > 0 else 0.0
    price_ratio = (abs(signal.pct_from_open) / cfg.move_trigger_pct) if cfg.move_trigger_pct > 0 else 0.0
    score = min(100.0, max(vol_ratio, price_ratio) * 25.0)
    if signal.fresh_catalyst:
        score = min(100.0, score + 15.0)

    if not fires:
        tier = "T0"
    elif score >= 70.0:
        tier = "T3"
    elif score >= 40.0:
        tier = "T2"
    else:
        tier = "T1"
    return round(score, 1), tier


_TRIGGER_TIERS = {"T1", "T2", "T3"}


def compute_same_day_labels(polls: List[Dict[str, Any]], eod: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derive spec §12.2's same-day outcome labels from one name's poll rows +
    its EOD OHLC.

    Poll-granularity data (5/15-min snapshots, not true 1-min bars) means:
    - ``high_time`` is approximated as the first poll whose cumulative
      ``day_high`` already matches the EOD high — the true peak minute could
      be anywhere in that poll's interval, but this is the best resolution the
      stored data supports.
    - ``mae_from_alert``/``mfe_from_alert`` use each poll's cumulative
      ``day_high``/``day_low`` (plus the EOD bar itself as the final bound)
      rather than a true intrabar extreme, for the same reason.

    The "alert price" is the price at the first poll whose ``momentum_tier``
    is T1 or higher — the simulated trigger point (spec §16 item 5), since
    Phase 1.5 has no real Alert Manager yet. A name that never crossed T1 that
    day has no MAE/MFE (nothing to measure excursion from), but still gets
    ``close_retention``/``high_time`` from the EOD bar alone.

    Args:
        polls: Rows from ``ShadowStore.polls_for_date_ticker`` — ascending by
            ``ts``, each with ``ts`` (ISO string), ``price``, ``day_high``,
            ``day_low``, ``momentum_tier``.
        eod: ``open``/``high``/``low``/``close`` for the day.

    Returns:
        dict with ``high_time``, ``close_retention``, ``mae_from_alert``,
        ``mfe_from_alert`` — any of which may be None if unresolvable.
    """
    labels: Dict[str, Any] = {
        "high_time": None,
        "close_retention": None,
        "mae_from_alert": None,
        "mfe_from_alert": None,
    }
    if not polls:
        return labels

    o, h, l, c = eod.get("open"), eod.get("high"), eod.get("low"), eod.get("close")
    if o is not None and h is not None and c is not None and h != o:
        labels["close_retention"] = (c - o) / (h - o)

    if h is not None:
        for p in polls:
            day_high = p.get("day_high")
            if day_high is not None and day_high >= h - 1e-9:
                labels["high_time"] = _to_et_hhmm(p.get("ts"))
                break

    alert_idx = next((i for i, p in enumerate(polls) if p.get("momentum_tier") in _TRIGGER_TIERS), None)
    if alert_idx is None:
        return labels
    alert_price = polls[alert_idx].get("price")
    if not alert_price:
        return labels

    highs = [p["day_high"] for p in polls[alert_idx:] if p.get("day_high") is not None]
    lows = [p["day_low"] for p in polls[alert_idx:] if p.get("day_low") is not None]
    if h is not None:
        highs.append(h)
    if l is not None:
        lows.append(l)
    if highs:
        labels["mfe_from_alert"] = max(highs) / alert_price - 1.0
    if lows:
        labels["mae_from_alert"] = min(lows) / alert_price - 1.0
    return labels


def _to_et_hhmm(ts_iso: Optional[str]) -> Optional[str]:
    if not ts_iso:
        return None
    try:
        dt = datetime.fromisoformat(ts_iso)
    except ValueError:
        return None
    return dt.astimezone(_ET).strftime("%H:%M")

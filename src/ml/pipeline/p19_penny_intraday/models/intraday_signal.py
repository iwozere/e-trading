"""P19 IntradaySignal model."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass
class IntradaySignal:
    """
    One intraday detection / shadow-log row for a watchlist name (spec §11).

    Captured every poll for every name in shadow mode; the subset that crosses the
    trigger gate (`trigger_reason` set) becomes an alert candidate.
    """

    # ── Identity / time ────────────────────────────────────────────────────
    ticker: str
    ts: datetime  # detection time (UTC)
    source: str = ""  # watchlist source: p17 | gapper | manual
    price: float = 0.0

    # ── Live price action (IBKR delayed reqMktData) ────────────────────────
    day_open: float = 0.0
    day_high: float = 0.0
    day_low: float = 0.0
    prev_close: float = 0.0
    pct_from_open: float = 0.0
    pct_from_prev_close: float = 0.0

    # ── Volume / RVOL (delayed, confirming context) ────────────────────────
    day_volume: float = 0.0  # cumulative day volume (raw, IBKR units)
    avg_volume_30d: float = 0.0  # baseline (shares) for RVOL
    rvol_so_far: float = 0.0
    dollar_volume_so_far: float = 0.0
    volume_is_delayed: bool = True

    # ── Catalyst / fundamentals (reused P17 agents) ────────────────────────
    fresh_catalyst: bool = False  # bullish 8-K filed today
    catalyst_signals: List[str] = field(default_factory=list)
    short_squeeze_score: float = 0.0
    dilution_penalty: float = 0.0  # >0 → fade risk
    fresh_dilution_filing: bool = False  # v2 spec §9 — 424B5/S-1/8-K 3.02 filed today

    # ── Structural axis (v2 spec §11 — denormalised from Layer 0, pre-market) ─
    # Point-in-time snapshot as of this poll, never a live join against the
    # (mutable) profile cache — spec §12.1, avoids leaking future info into
    # calibration. All nullable: an unprofiled name still gets logged (decision #7).
    structural_grade: str = ""  # A / B / C / D — "" = not yet profiled
    dilution_urgency: float = 0.0
    insider_conviction: float = 0.0
    runway_quarters: float | None = None
    disqualifiers: List[str] = field(default_factory=list)
    structural_coverage: float = 0.0
    # StructuralSignals.md §2 — track FPIs separately in calibration, or the
    # grade-vs-close_retention test (spec §15) will be confounded by the two
    # very different populations that land in grade C (genuinely risky
    # domestic filers vs. opaque-by-structure FPIs). None = not yet profiled.
    is_fpi: bool | None = None

    # ── Sentiment (context only) ───────────────────────────────────────────
    sentiment: Dict[str, float] = field(default_factory=dict)

    # ── Scoring / alerting ─────────────────────────────────────────────────
    # momentum_score/momentum_tier (v2 spec §11) replace v1's flat `severity` —
    # momentum evidence only, no structural term (decision #6, orthogonal axes).
    # `severity` stays for backward-compat reads of pre-v2 shadow rows but is no
    # longer written.
    severity: float = 0.0
    momentum_score: float = 0.0
    momentum_tier: str = ""  # T0 / T1 / T2 / T3 — log-only in Phase 1.5, no Disposition Engine yet
    trigger_reason: str = ""  # which tripwire(s) fired; "" = no trigger
    tier: str = ""  # P17 quality tier passthrough (A/B/C), NOT momentum_tier

    # ── EOD backfill (shadow dataset) ──────────────────────────────────────
    eod_open: float | None = None
    eod_high: float | None = None
    eod_low: float | None = None
    eod_close: float | None = None

    # ── Outcome labels (v2 spec §12.2 — filled by eod-backfill / label-backfill) ─
    high_time: str | None = None  # HH:MM (ET) of the intraday high
    close_retention: float | None = None  # (close - open) / (high - open); primary fade measure
    mae_from_alert: float | None = None  # max adverse excursion from the simulated-trigger price
    mfe_from_alert: float | None = None  # max favourable excursion from the simulated-trigger price
    ret_t1: float | None = None
    ret_t3: float | None = None
    ret_t5: float | None = None
    ret_t10: float | None = None
    dilution_event_within_5d: bool | None = None
    reverse_split_within_180d: bool | None = None

    def to_dict(self) -> Dict:
        """Flat dict suitable for CSV / shadow-store rows."""
        return {
            "ticker": self.ticker,
            "ts": self.ts.isoformat() if self.ts else "",
            "source": self.source,
            "price": round(self.price, 4),
            "day_open": round(self.day_open, 4),
            "day_high": round(self.day_high, 4),
            "day_low": round(self.day_low, 4),
            "prev_close": round(self.prev_close, 4),
            "pct_from_open": round(self.pct_from_open, 4),
            "pct_from_prev_close": round(self.pct_from_prev_close, 4),
            "day_volume": round(self.day_volume, 2),
            "avg_volume_30d": round(self.avg_volume_30d, 2),
            "rvol_so_far": round(self.rvol_so_far, 2),
            "dollar_volume_so_far": round(self.dollar_volume_so_far, 2),
            "volume_is_delayed": self.volume_is_delayed,
            "fresh_catalyst": self.fresh_catalyst,
            "catalyst_signals": "|".join(self.catalyst_signals),
            "short_squeeze_score": round(self.short_squeeze_score, 1),
            "dilution_penalty": round(self.dilution_penalty, 1),
            "fresh_dilution_filing": self.fresh_dilution_filing,
            "structural_grade": self.structural_grade,
            "dilution_urgency": round(self.dilution_urgency, 2),
            "insider_conviction": round(self.insider_conviction, 2),
            "runway_quarters": self.runway_quarters,
            "disqualifiers": json.dumps(self.disqualifiers),
            "structural_coverage": round(self.structural_coverage, 3),
            "is_fpi": self.is_fpi,
            "sentiment": ";".join(f"{k}={v}" for k, v in self.sentiment.items()),
            "severity": round(self.severity, 1),
            "momentum_score": round(self.momentum_score, 2),
            "momentum_tier": self.momentum_tier,
            "trigger_reason": self.trigger_reason,
            "tier": self.tier,
            "eod_open": self.eod_open,
            "eod_high": self.eod_high,
            "eod_low": self.eod_low,
            "eod_close": self.eod_close,
            "high_time": self.high_time,
            "close_retention": self.close_retention,
            "mae_from_alert": self.mae_from_alert,
            "mfe_from_alert": self.mfe_from_alert,
            "ret_t1": self.ret_t1,
            "ret_t3": self.ret_t3,
            "ret_t5": self.ret_t5,
            "ret_t10": self.ret_t10,
            "dilution_event_within_5d": self.dilution_event_within_5d,
            "reverse_split_within_180d": self.reverse_split_within_180d,
        }

    @property
    def triggered(self) -> bool:
        return bool(self.trigger_reason)

"""P19 IntradaySignal model."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class IntradaySignal:
    """
    One intraday detection / shadow-log row for a watchlist name (spec §11).

    Captured every poll for every name in shadow mode; the subset that crosses the
    trigger gate (`trigger_reason` set) becomes an alert candidate.
    """

    # ── Identity / time ────────────────────────────────────────────────────
    ticker: str
    ts: datetime                              # detection time (UTC)
    price: float = 0.0

    # ── Live price action (Finnhub /quote, real-time) ──────────────────────
    pct_from_open: float = 0.0
    pct_from_prev_close: float = 0.0
    day_high: float = 0.0

    # ── Volume / RVOL (delayed, confirming context) ────────────────────────
    rvol_so_far: float = 0.0
    dollar_volume_so_far: float = 0.0
    volume_is_delayed: bool = True

    # ── Catalyst / fundamentals (reused P17 agents) ────────────────────────
    fresh_catalyst: bool = False              # bullish 8-K filed today
    catalyst_signals: List[str] = field(default_factory=list)
    short_squeeze_score: float = 0.0
    dilution_penalty: float = 0.0             # >0 → fade risk

    # ── Sentiment (context only) ───────────────────────────────────────────
    sentiment: Dict[str, float] = field(default_factory=dict)

    # ── Scoring / alerting ─────────────────────────────────────────────────
    severity: float = 0.0
    trigger_reason: str = ""                  # which tripwire(s) fired; "" = no trigger
    tier: str = ""                            # alert/escalation tier

    # ── EOD backfill (shadow dataset) ──────────────────────────────────────
    eod_open: Optional[float] = None
    eod_high: Optional[float] = None
    eod_low: Optional[float] = None
    eod_close: Optional[float] = None

    def to_dict(self) -> Dict:
        """Flat dict suitable for CSV / shadow-store rows."""
        return {
            "ticker": self.ticker,
            "ts": self.ts.isoformat() if self.ts else "",
            "price": round(self.price, 4),
            "pct_from_open": round(self.pct_from_open, 4),
            "pct_from_prev_close": round(self.pct_from_prev_close, 4),
            "day_high": round(self.day_high, 4),
            "rvol_so_far": round(self.rvol_so_far, 2),
            "dollar_volume_so_far": round(self.dollar_volume_so_far, 2),
            "volume_is_delayed": self.volume_is_delayed,
            "fresh_catalyst": self.fresh_catalyst,
            "catalyst_signals": "|".join(self.catalyst_signals),
            "short_squeeze_score": round(self.short_squeeze_score, 1),
            "dilution_penalty": round(self.dilution_penalty, 1),
            "sentiment": ";".join(f"{k}={v}" for k, v in self.sentiment.items()),
            "severity": round(self.severity, 1),
            "trigger_reason": self.trigger_reason,
            "tier": self.tier,
            "eod_open": self.eod_open,
            "eod_high": self.eod_high,
            "eod_low": self.eod_low,
            "eod_close": self.eod_close,
        }

    @property
    def triggered(self) -> bool:
        return bool(self.trigger_reason)

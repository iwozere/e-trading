"""P19 WatchlistEntry model."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class WatchlistEntry:
    """
    One name on the daily intraday watchlist (spec §4.1), with the baseline context
    the intraday loop needs (prior close, avg volume, float, dilution, catalyst).

    Sources: ``p17`` (Tier B/C / explosive from the daily screener), ``gapper``
    (pre-market gappers/most-active < price cap, via the IBKR scanner), ``manual``.
    """

    ticker: str
    source: str  # "p17" | "gapper" | "manual"
    tier: str = ""  # p17 tier, if applicable
    explosive: bool = False  # p17 explosive_candidate flag
    company_name: str = ""

    # ── Baseline context (for RVOL / % move / fade flags) ──────────────────
    prior_close: float = 0.0  # last known close (P17 detection price)
    avg_volume_30d: float = 0.0
    float_shares: float = 0.0
    market_cap: float = 0.0
    dilution_penalty: float = 0.0  # >0 → fade risk
    short_interest_pct_float: float | None = None
    has_catalyst: bool = False
    catalyst_signals: List[str] = field(default_factory=list)

    # ── Builder bookkeeping ────────────────────────────────────────────────
    priority: float = 0.0  # ranking score (higher = kept first)

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "source": self.source,
            "tier": self.tier,
            "explosive": self.explosive,
            "company_name": self.company_name,
            "prior_close": round(self.prior_close, 4),
            "avg_volume_30d": int(self.avg_volume_30d) if self.avg_volume_30d else 0,
            "float_shares": int(self.float_shares) if self.float_shares else 0,
            "market_cap": int(self.market_cap) if self.market_cap else 0,
            "dilution_penalty": round(self.dilution_penalty, 1),
            "short_interest_pct_float": self.short_interest_pct_float,
            "has_catalyst": self.has_catalyst,
            "catalyst_signals": list(self.catalyst_signals),
            "priority": round(self.priority, 2),
        }

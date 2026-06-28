"""
P19 Intraday Penny-Stock Monitor — configuration.

Dataclass config mirroring the pipeline specification (`docs/pipeline-specification.md`).
Defaults encode the locked design decisions and the 2026-06-28 feed-probe findings
(§13.1): Finnhub `/quote` is the real-time **price** trigger; volume/RVOL is
~15-min-delayed confirming context (Polygon/yfinance), not the live tripwire.

Thresholds here are launch placeholders — they MUST be calibrated against the
shadow-mode dataset (spec §15) before they are trusted for alerting.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class P19FilterConfig:
    """Watchlist eligibility (hard filters, spec §7)."""
    max_price: float = 5.0
    max_float_shares: float = 25_000_000
    ultra_low_float_shares: float = 10_000_000   # flagged as extra-explosive
    min_daily_volume: float = 500_000
    exclude_etfs: bool = True
    exclude_test_issues: bool = True


@dataclass
class P19FeedConfig:
    """
    Intraday data feed (spec §4.2, §13.1–13.2).

    Primary = **IBKR Gateway (delayed, free)** — unlike Finnhub/Polygon free, IBKR
    delayed bars include **volume**, giving real 1m/5m OHLCV (~15-min delayed) and
    therefore real RVOL-so-far. Connects to the paper Gateway on the same Pi.
    Finnhub (real-time price, no volume) is an optional faster cross-check.
    """
    primary_provider: str = "ibkr"         # delayed 1m/5m OHLCV+volume via Gateway
    price_crosscheck_provider: str = "finnhub"   # optional real-time price (no vol)
    fallback_provider: str = "yfinance"
    # IBKR Gateway connection (same-Pi paper Gateway; values from donotshare/.env)
    ibkr_host: str = "raspberrypi"
    ibkr_port: int = 4002                  # paper Gateway (live = 4001)
    ibkr_client_id: int = 19               # unique per process (p19)
    ibkr_market_data_type: int = 3         # 3 = delayed (free)
    use_streaming: bool = True             # keepUpToDate bars, not per-cycle historical
    poll_interval_minutes: int = 5         # 5/15/30 supported
    watchlist_cap: int = 100               # IBKR ~100 market-data lines (§13.2)
    premarket_enabled: bool = True


@dataclass
class P19TriggerConfig:
    """Intraday tripwires (spec §4.3). Placeholders — calibrate on shadow data."""
    move_trigger_pct: float = 0.20         # |% from open| price thrust (live trigger)
    rvol_trigger: float = 5.0              # RVOL-so-far (delayed, confirming)
    dollar_volume_floor: float = 50_000    # liquidity gate
    require_volume_and_price: bool = True   # vol AND (price OR fresh catalyst)
    fresh_catalyst_relaxes: bool = True    # a same-day bullish 8-K lowers thresholds


@dataclass
class P19AlertConfig:
    """Alerting + dedup state (spec §14)."""
    enabled: bool = True
    telegram_enabled: bool = True
    email_enabled: bool = False
    daily_alert_cap: int = 20
    dedup_per_day: bool = True
    realert_on_escalation: bool = True     # re-alert only on a higher severity tier


@dataclass
class P19Config:
    """Complete P19 pipeline configuration."""
    filter_config: P19FilterConfig = field(default_factory=P19FilterConfig)
    feed_config: P19FeedConfig = field(default_factory=P19FeedConfig)
    trigger_config: P19TriggerConfig = field(default_factory=P19TriggerConfig)
    alert_config: P19AlertConfig = field(default_factory=P19AlertConfig)

    # Watchlist sources (spec §4.1)
    use_p17_watchlist: bool = True
    use_gappers: bool = True
    manual_pins: List[str] = field(default_factory=list)

    shadow_mode: bool = True               # Phase 1: log only, no alerts
    user_id: Optional[str] = None          # scheduler-injected (alert recipient)

    @classmethod
    def create_default(cls) -> "P19Config":
        """Production defaults: shadow-mode on, alerting gated until calibrated."""
        return cls()

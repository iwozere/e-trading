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
from typing import Dict, List


@dataclass
class P19FilterConfig:
    """Watchlist eligibility (hard filters, spec §7)."""

    max_price: float = 5.0
    max_float_shares: float = 25_000_000
    ultra_low_float_shares: float = 10_000_000  # flagged as extra-explosive
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

    primary_provider: str = "ibkr"  # delayed 1m/5m OHLCV+volume via Gateway
    price_crosscheck_provider: str = "finnhub"  # optional real-time price (no vol)
    fallback_provider: str = "yfinance"
    # IBKR Gateway connection (same-Pi paper Gateway; values from donotshare/.env)
    ibkr_host: str = "127.0.0.1"  # same-Pi Gateway (host networking)
    ibkr_port: int = 4002  # paper Gateway API (verified working)
    ibkr_client_id: int = 19  # intraday loop (unique per process)
    ibkr_scanner_client_id: int = 20  # pre-market gappers scan (distinct id)
    ibkr_market_data_type: int = 3  # 3 = delayed (free)
    ibkr_volume_lot_size: int = 100  # reqMktData day-volume → shares (verify on Pi)
    use_streaming: bool = True  # keepUpToDate bars, not per-cycle historical
    poll_interval_minutes: int = 5  # 5/15/30 supported
    watchlist_cap: int = 100  # IBKR ~100 market-data lines (§13.2)
    premarket_enabled: bool = True


@dataclass
class P19TriggerConfig:
    """Intraday tripwires (spec §4.3). Placeholders — calibrate on shadow data."""

    move_trigger_pct: float = 0.20  # |% from open| price thrust (live trigger)
    rvol_trigger: float = 5.0  # RVOL-so-far (delayed, confirming)
    dollar_volume_floor: float = 50_000  # liquidity gate
    require_volume_and_price: bool = True  # vol AND (price OR fresh catalyst)
    fresh_catalyst_relaxes: bool = True  # a same-day bullish 8-K lowers thresholds


@dataclass
class P19StructuralConfig:
    """
    Layer 0 structural-integrity thresholds (spec v2 §7, StructuralSignals.md).

    Phase 1.5 added Tier 1+2 signals (N1,N2,N3,N4,N7,N8,N10,N11,N13,N14 /
    P1,P2,P3,P4,P5,P6,P7) plus the §5 baby-shelf arithmetic. Phase 3 added
    N5,N6,N15,N16,P8,P9,P11. **Still out of scope** (design-v2.md §Roadmap):
    N9 above the $75M baby-shelf threshold (needs prospectus text parsing —
    spec's own fallback is to stay boolean there), N12 (warrant overhang — no
    XBRL tag and no safe arithmetic shortcut like N9's baby-shelf rule exists),
    P10 (insider ownership stability — needs Form 3 baseline ingestion, not
    built). Those fields stay unresolved (None) and depress ``coverage``
    rather than being silently assumed clean.

    All thresholds here are **bootstrap placeholders** (spec §7.4, §15) — fit
    against shadow-dataset outcome labels once they exist, not hand-tuned.
    """

    # ── N1/N2 — reverse splits ───────────────────────────────────────────────
    reverse_split_window_months: int = 24

    # ── N3/N4 — share-count CAGR (split-adjusted, xbrl_facts.split_adjust) ──
    share_cagr_d_threshold: float = 0.25  # > this → grade D
    share_cagr_c_threshold: float = 0.10  # > this → grade C

    # ── N7 — exchange deficiency notice (8-K item 3.01) ──────────────────────
    deficiency_notice_lookback_days: int = 365

    # ── N8 — offering document filed recently ────────────────────────────────
    offering_window_c_days: int = 5  # trading days; C severity
    offering_window_d_days: int = 2  # escalates to D within this window

    # ── N10 — ATM usage ───────────────────────────────────────────────────────
    atm_lookback_quarters: int = 2
    # Excludes option/warrant-exercise noise on the same XBRL tag
    # (StructuralSignals.md N10 failure mode) — proceeds must clear this
    # fraction of market cap to count as a real ATM draw.
    atm_proceeds_pct_mcap_threshold: float = 0.01

    # ── N11 — cash runway ─────────────────────────────────────────────────────
    runway_c_threshold_quarters: float = 3.0
    runway_d_threshold_quarters: float = 1.5  # D only if ALSO shelf_active

    # ── N13 — unregistered equity sale (8-K item 3.02) ───────────────────────
    unregistered_sale_lookback_days: int = 30

    # ── N14 — clustered insider sales ────────────────────────────────────────
    insider_sell_window_days: int = 90
    n14_min_distinct_sellers: int = 2  # "clustered" — a single seller is not N14

    # ── P1/P2 — insider open-market buys ──────────────────────────────────────
    insider_buy_cluster_window_days: int = 30  # P1
    insider_buy_window_days: int = 90  # P2
    p1_min_distinct_buyers: int = 3

    # ── §5 baby-shelf arithmetic (N9 estimate, sub-$75M float only) ──────────
    baby_shelf_float_threshold_usd: float = 75_000_000.0

    # ── N17 / coverage ─────────────────────────────────────────────────────
    coverage_c_threshold: float = 0.4  # coverage below this → grade C

    # ── §7.5 grade assignment ─────────────────────────────────────────────
    insider_conviction_a_threshold: float = 40.0
    dilution_urgency_a_threshold: float = 30.0  # must be BELOW this for grade A

    # ── §7.4 dilution_urgency weights (bootstrap, spec §7.4) ─────────────────
    dilution_urgency_w_runway: float = 0.35
    dilution_urgency_w_shelf_capacity: float = 0.25
    dilution_urgency_w_recent_usage: float = 0.25
    dilution_urgency_w_dilution_history: float = 0.15
    runway_normalize_quarters: float = 8.0  # runway_pressure = 1 - clip(runway/this, 0, 1)
    recent_usage_lookback_quarters: int = 8
    dilution_history_cagr_clip: float = 0.5  # share_count_cagr_8q clipped at this for the urgency term

    # ── insider_conviction weights (bootstrap; P1-P9 + conditional P11) ──────
    # Renormalised over *resolved* signals only (requirements-v2.md rule #5) —
    # an unresolved signal contributes to neither the numerator nor denominator.
    # P11 is intentionally marked unresolved (excluded from this renormalisation)
    # whenever the pre-SI grade is C/D — StructuralSignals.md §4 P11: SI is
    # conviction evidence only at grade A/B, context-only at C/D (see
    # ``_eval_short_interest_conditional``).
    insider_conviction_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "P1": 35.0,  # clustered insider buying — highest
            "P2": 20.0,  # single officer/director buy — high
            "P3": 20.0,  # share count flat/declining — high
            "P4": 15.0,  # executed buyback — high
            "P5": 10.0,  # positive operating cash flow — medium
            "P6": 10.0,  # runway > 8q or net cash positive — medium
            "P7": 10.0,  # no dilution event in 24mo — medium
            "P8": 8.0,  # 13D/G accumulation activity, trailing 2q — medium (magnitude-blind proxy, see grading.py)
            "P9": 8.0,  # no near-term debt maturity — medium
            "P11": 5.0,  # high SI + rising days-to-cover, conditional on A/B — context/low
        }
    )

    # ── N5 — floating-rate/toxic convertible (EFTS phrase match) ─────────────
    # StructuralSignals.md open question 6: precision unmeasured (needs a
    # hand-labelled sample of ~50 filings). Spec's own explicit fallback:
    # "until precision is measured, consider firing N5 at grade C rather than
    # D." Promote to "D" once that study exists.
    n5_severity: str = "C"
    n5_convert_phrases: List[str] = field(
        default_factory=lambda: [
            "lowest VWAP",
            "conversion price equal to",
            "discount to the market price",
            "variable conversion price",
            "lowest trading price",
            "percentage of the lowest",
        ]
    )

    # ── N6 — going-concern qualification (EFTS phrase match) ─────────────────
    n6_going_concern_phrase: str = "substantial doubt about its ability to continue as a going concern"

    # ── N16 — auditor quality ─────────────────────────────────────────────────
    # Static maintained whitelist rather than a live PCAOB Form AP integration
    # (design-v2.md §Roadmap decision) — matches StructuralSignals.md's own
    # suggested approach ("a maintained whitelist ... needs periodic review").
    # Substring-matched, case-insensitive, against the firm name extracted from
    # the filing's EX-23.1 consent exhibit.
    auditor_whitelist: List[str] = field(
        default_factory=lambda: [
            "DELOITTE",
            "ERNST & YOUNG",
            "PRICEWATERHOUSECOOPERS",
            "KPMG",
            "BDO",
            "GRANT THORNTON",
            "MARCUM",
            "RSM US",
            "CROWE",
            "MOSS ADAMS",
            "WITHUMSMITH",
            "MAYER HOFFMAN MCCANN",
            "CHERRY BEKAERT",
            "PLANTE MORAN",
            "BAKER TILLY",
            "CBIZ",
            "ARMANINO",
            "FRAZIER & DEETER",
        ]
    )

    # ── N15 — recent IPO + micro float + FPI reporting ────────────────────────
    n15_ipo_window_months: int = 18
    n15_float_threshold_shares: float = 5_000_000.0

    # ── P8 — institutional 13D/G accumulation (presence proxy, magnitude-blind) ─
    # True P8 (new/increased positions, magnitude-aware) needs the filing's Item
    # 3/11 percent-of-class, which requires fetching+parsing each 13D/G document
    # — deferred. This proxy only asks "was any 13D/G filed against this CIK in
    # the trailing 2 quarters" (design-v2.md §Roadmap), which cannot distinguish
    # new/increased from decreased/exited — documented limitation.
    p8_lookback_quarters: int = 2

    # ── P11 — SI conditional on grade (context at C/D, conviction at A/B) ────
    p11_si_threshold: float = 0.20  # 20% of float
    p11_days_to_cover_threshold: float = 3.0
    p11_dilution_urgency_bump: float = 10.0  # added to dilution_urgency at C/D when squeeze conditions are met

    # ── Layer 0 cache cadence ────────────────────────────────────────────────
    profile_cache_ttl_days: int = 7  # weekly full refresh (spec §4.0)


@dataclass
class P19AlertConfig:
    """Alerting + dedup state (spec §14)."""

    enabled: bool = True
    telegram_enabled: bool = True
    email_enabled: bool = False
    daily_alert_cap: int = 20
    dedup_per_day: bool = True
    realert_on_escalation: bool = True  # re-alert only on a higher severity tier


@dataclass
class P19Config:
    """Complete P19 pipeline configuration."""

    filter_config: P19FilterConfig = field(default_factory=P19FilterConfig)
    feed_config: P19FeedConfig = field(default_factory=P19FeedConfig)
    trigger_config: P19TriggerConfig = field(default_factory=P19TriggerConfig)
    alert_config: P19AlertConfig = field(default_factory=P19AlertConfig)
    structural_config: P19StructuralConfig = field(default_factory=P19StructuralConfig)

    # Watchlist sources (spec §4.1)
    use_p17_watchlist: bool = True
    use_gappers: bool = True
    manual_pins: List[str] = field(default_factory=list)

    shadow_mode: bool = True  # Phase 1: log only, no alerts
    user_id: str | None = None  # scheduler-injected (alert recipient)

    @classmethod
    def create_default(cls) -> "P19Config":
        """Production defaults: shadow-mode on, alerting gated until calibrated."""
        return cls()

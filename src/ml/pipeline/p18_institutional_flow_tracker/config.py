"""
P18 Institutional Flow Tracker — Pipeline Configuration

All tunable thresholds live here so they can be adjusted without touching
pipeline logic.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class P18Config:
    """
    Configuration for the Institutional Flow Tracker pipeline.

    All monetary thresholds are in USD. All percentage thresholds are decimals
    (0.30 = 30 %).
    """

    # ------------------------------------------------------------------
    # Institution filter
    # ------------------------------------------------------------------
    min_aum_usd: int = 1_000_000_000
    """Minimum portfolio value (AUM proxy) to include an institution. Default $1B."""

    # ------------------------------------------------------------------
    # Exit screener
    # ------------------------------------------------------------------
    exit_threshold_pct: float = 0.30
    """Position must be reduced by at least this fraction to count as an exit."""

    min_position_pct_of_portfolio: float = 0.005
    """Minimum prior-quarter portfolio weight to count — filters out noise from tiny positions."""

    min_position_value_usd: int = 25_000_000
    """Minimum prior-quarter position value ($25M) as an alternative to portfolio-weight filter."""

    # ------------------------------------------------------------------
    # Consensus detector
    # ------------------------------------------------------------------
    consensus_min_institutions: int = 3
    """How many institutions must exit the same stock to raise a consensus signal."""

    # ------------------------------------------------------------------
    # Volume anomaly
    # ------------------------------------------------------------------
    volume_spike_multiplier: float = 3.5
    """Daily volume must exceed this multiple of the rolling average to flag."""

    volume_lookback_days: int = 20
    """Rolling window for average volume baseline."""

    volume_spike_recent_days: int = 5
    """Number of recent days to check for spikes."""

    # ------------------------------------------------------------------
    # Composite scorer
    # ------------------------------------------------------------------
    score_alert_threshold: int = 60
    """Minimum composite score to generate a Telegram alert."""

    # Binary signal weights. The large single-institution exit is NOT a flat
    # bonus here — it is graded by dollar value sold inside CompositeScorer
    # (_DEFAULT_LARGE_EXIT_TIERS_USD), and consensus breadth (institution count)
    # adds a separate graded bonus. This prevents every $500M+ exit from clearing
    # the alert threshold and saturating the results.
    signal_weights: Dict[str, int] = field(default_factory=lambda: {
        "consensus_exit_3plus": 40,
        "volume_spike_confirmed": 20,
        "form4_insider_sell": 10,
        "schedule_13dg_drop": 10,
        "seasonal_redemption_window": 5,
        "price_below_52w_high_15pct": 5,
    })

    # ------------------------------------------------------------------
    # Scheduler / pipeline behaviour
    # ------------------------------------------------------------------
    max_tickers_for_volume_check: int = 200
    """Cap on how many tickers are run through the volume anomaly detector per day."""

    results_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "results" / "p18_institutional_flow")
    """Directory where per-run CSVs are written."""

    summary_top_n: int = 10
    """How many of the highest-scoring tickers to record in run_summary.json."""

    @classmethod
    def create_default(cls) -> "P18Config":
        """Return a P18Config with all defaults applied."""
        return cls()

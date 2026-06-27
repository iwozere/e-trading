"""
P17 Scoring Agent

Computes the final composite score for each candidate:
  1. Normalise each sub-score to [0, 100] using fixed thresholds from the spec
  2. Apply weighted sum (weights sum to 1.0)
  3. Deduct dilution penalty (hard deduction, not weighted)
  4. Floor result at 0
  5. Assign tier: A | B | C | W
  6. Mark explosive_candidate flag
  7. Collect human-readable signals list

Sub-score thresholds (0 / 50 / 100):
  momentum      −20% / 0% / +50%  (20d return)
  volume        rvol 1.0 / 2.0 / 5.0
  technical     0 / 1 / 3+ signals
  fundamentals  declining / stable / rev accel >50%
  catalyst      0 (phase 1 placeholder)
  short_squeeze SI <5% / 15% / 25% float
  accumulation  0 / neutral / strong (7+ days)
"""

from pathlib import Path
import sys
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.ml.pipeline.p17_penny_stocks.config import P17ScoringConfig
from src.ml.pipeline.p17_penny_stocks.models.candidate import Candidate
from src.ml.pipeline.p17_penny_stocks.agents.short_squeeze_agent import ShortSqueezeAgent

_logger = setup_logger(__name__)


def _interp(value: float, low: float, mid: float, high: float) -> float:
    """Linear two-segment normalisation to [0, 100]."""
    if value <= low:
        return 0.0
    if value >= high:
        return 100.0
    if value <= mid:
        return 50.0 * (value - low) / (mid - low)
    return 50.0 + 50.0 * (value - mid) / (high - mid)


class ScoringAgent:
    """
    Stage 6: Compute composite scores and assign tiers.

    Each sub-score normalised to [0, 100] → weighted sum → dilution deduction.
    """

    def __init__(
        self,
        scoring_config: P17ScoringConfig,
        short_squeeze_agent: ShortSqueezeAgent,
    ) -> None:
        self.cfg = scoring_config
        self._ss_agent = short_squeeze_agent

    def run(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Score and tier all candidates.

        Args:
            candidates: List of fully enriched Candidate objects.

        Returns:
            Same list, sorted descending by final_score.
        """
        for c in candidates:
            self._score(c)

        candidates.sort(key=lambda c: c.final_score, reverse=True)

        tier_counts = {"A": 0, "B": 0, "C": 0, "W": 0}
        for c in candidates:
            tier_counts[c.tier] = tier_counts.get(c.tier, 0) + 1

        _logger.info(
            "Scoring complete: A=%d B=%d C=%d W=%d (total=%d)",
            tier_counts.get("A", 0), tier_counts.get("B", 0),
            tier_counts.get("C", 0), tier_counts.get("W", 0),
            len(candidates),
        )
        return candidates

    # ── Per-candidate scoring ──────────────────────────────────────────────

    def _score(self, c: Candidate) -> None:
        c.momentum_score = self._momentum_score(c)
        c.volume_score = self._volume_score(c)
        c.technical_score = self._technical_score(c)
        c.fundamentals_score = self._fundamentals_score(c)
        # c.catalyst_score is set upstream by the CatalystAgent (defaults to 0.0
        # when that stage is skipped or finds no catalyst); do not overwrite it.
        c.short_squeeze_score = self._ss_agent.compute_score(c)
        c.accumulation_score = self._accumulation_score(c)

        cfg = self.cfg
        c.weighted_score = (
            cfg.weight_momentum * c.momentum_score
            + cfg.weight_volume * c.volume_score
            + cfg.weight_technical * c.technical_score
            + cfg.weight_fundamentals * c.fundamentals_score
            + cfg.weight_catalyst * c.catalyst_score
            + cfg.weight_short_squeeze * c.short_squeeze_score
            + cfg.weight_accumulation * c.accumulation_score
        )

        c.final_score = max(0.0, c.weighted_score - c.dilution_penalty)

        c.tier = self._assign_tier(c)
        c.explosive_candidate = self._is_explosive(c)
        c.signals = self._collect_signals(c)

    # ── Sub-score computation ──────────────────────────────────────────────

    @staticmethod
    def _momentum_score(c: Candidate) -> float:
        ret = c.price_20d_return
        # >300% already capped in TechnicalAgent; score 0 if exactly at cap
        if ret >= 3.00:
            return 0.0
        return _interp(ret, -0.20, 0.0, 0.50)

    @staticmethod
    def _volume_score(c: Candidate) -> float:
        return _interp(c.relative_volume, 1.0, 2.0, 5.0)

    @staticmethod
    def _technical_score(c: Candidate) -> float:
        signals = 0
        if c.breakout_20d:
            signals += 2       # counts double — most important technical signal
        if c.breakout_50d:
            signals += 1
        if c.bb_squeeze:
            signals += 1
        if c.above_sma50:
            signals += 1
        # Normalise: 0 signals → 0, 1 → 25, 2 → 50, 3 → 75, 4+ → 100
        return min(100.0, signals * 20.0)

    @staticmethod
    def _fundamentals_score(c: Candidate) -> float:
        if c.revenue_growth_yoy is None:
            return 50.0  # unknown → neutral, not a penalty

        score = _interp(c.revenue_growth_yoy, -0.10, 0.25, 0.50)

        # Bonus: positive or improving operating cash flow
        if c.operating_cashflow is not None and c.operating_cashflow > 0:
            score = min(100.0, score + 10.0)

        # Penalty: gross margin < 0 (selling below cost)
        if c.gross_margin is not None and c.gross_margin < 0:
            score = max(0.0, score - 20.0)

        return score

    @staticmethod
    def _accumulation_score(c: Candidate) -> float:
        # OBV slope contribution (0–50)
        obv_score = min(50.0, max(0.0, c.obv_slope * 500.0)) if c.obv_slope > 0 else 0.0
        # Accumulation days contribution (0–50)
        day_score = min(50.0, c.accumulation_days * 7.0)
        return min(100.0, obv_score + day_score)

    # ── Tier assignment ────────────────────────────────────────────────────

    def _assign_tier(self, c: Candidate) -> str:
        score = c.final_score
        cfg = self.cfg

        if score >= cfg.tier_a_min_score:
            # Tier A requires strong technical structure in addition to high score
            if (c.relative_volume >= cfg.mandatory_rvol
                    and c.above_sma50
                    and (c.breakout_20d or c.breakout_50d)
                    and c.dilution_penalty <= cfg.mandatory_dilution_penalty_max):
                return "A"
            return "B"   # high score but mandatory conditions not met

        if score >= cfg.tier_b_min_score:
            return "B"

        if score >= cfg.tier_c_min_score:
            return "C"

        return "W"

    def _is_explosive(self, c: Candidate) -> bool:
        """Mark EXPLOSIVE_CANDIDATE per spec mandatory conditions."""
        cfg = self.cfg
        if c.relative_volume < cfg.mandatory_rvol:
            return False
        if not c.above_sma50:
            return False
        if not (c.breakout_20d or c.breakout_50d):
            return False
        if c.dilution_penalty > cfg.mandatory_dilution_penalty_max:
            return False

        # One of: revenue growth > 30%, strong catalyst, institutional accumulation
        rev_ok = c.revenue_growth_yoy is not None and c.revenue_growth_yoy > 0.30
        catalyst_ok = c.catalyst_score >= 50.0
        accum_ok = c.institutional_pct is not None and c.institutional_pct > 0.10
        return rev_ok or catalyst_ok or accum_ok

    # ── Signal collection ──────────────────────────────────────────────────

    @staticmethod
    def _collect_signals(c: Candidate) -> List[str]:
        signals: List[str] = []

        if c.breakout_20d:
            signals.append("breakout_20d")
        if c.breakout_50d:
            signals.append("breakout_50d")
        if c.relative_volume >= 3.0:
            signals.append(f"high_rvol_{c.relative_volume:.1f}x")
        if c.bb_squeeze:
            signals.append("bb_squeeze")
        if c.revenue_growth_yoy is not None and c.revenue_growth_yoy > 0.25:
            signals.append(f"rev_growth_{c.revenue_growth_yoy:.0%}")
        if c.operating_cashflow is not None and c.operating_cashflow > 0:
            signals.append("cashflow_positive")
        if c.short_interest_pct_float is not None and c.short_interest_pct_float > 0.15:
            signals.append(f"high_si_{c.short_interest_pct_float:.0%}")
        if c.accumulation_days >= 5:
            signals.append(f"accumulation_{c.accumulation_days}d")
        if c.catalyst_score > 0 and c.catalyst_signals:
            signals.extend(c.catalyst_signals)
        if c.dilution_penalty > 0:
            signals.extend(c.dilution_signals)
        if "halt_risk_high_si" in (c.signals or []):
            signals.append("halt_risk_high_si")

        return signals

"""
P22 — therapeutic-area demand features: matching targets to acquirers by therapeutic area (2026-11-09).

Block D's pairwise `fit()` scored only 5% of targets: its `ta_overlap` term needs a per-area pipeline
gap (revenue at risk minus Phase III count x assumed peak sales), whose inputs were never curated, so
only `modality_capability` was ever computable, and 92% of target assets have no classified modality.
About 60% of target assets do have a classified therapeutic area, so these features match on that alone:

- `ta_acquirer_demand` — acquirers in the universe on `as_of` with at least one Phase III asset in
  the target's lead therapeutic area (big pharma mostly buys inside the areas it already sells in).
- `ta_acquirer_phase3_assets` — those acquirers' Phase III assets in that area in total (how crowded
  the area is on the buy side).
- `ta_deal_heat_24m` — strategic acquisitions announced in the 24 months up to `as_of` whose target's
  lead area, as of its announcement, was the same (deals come in waves by area).
- `ta_acquirer_revenue_growth` — median latest annual revenue growth of the acquirers present in the
  area (Block A's pressure proxy, `block_a.revenue_growth_1y`): buyers whose growth is stalling.

Each is `None` when the target's lead area is unknown. Everything reads trials public by `as_of` (or
by the past deal's announcement date), and only deals announced by `as_of`, so a backtest snapshot
sees what was knowable then. Area classification itself is today's (it comes from trial conditions,
which do not change).
"""

from __future__ import annotations

import sys
from collections import Counter
from statistics import median
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.backtest.labels import add_months
from src.ml.pipeline.p22_biotech_ma.features import block_b
from src.ml.pipeline.p22_biotech_ma.features.context import FeatureContext
from src.ml.pipeline.p22_biotech_ma.ingest.deal_entry import POSITIVE_DEAL_TYPE
from src.ml.pipeline.p22_biotech_ma.ingest.therapeutic_area_classifier import UNCLASSIFIED

DEAL_HEAT_MONTHS = 24


def lead_therapeutic_area(trials: Sequence[Dict[str, Any]]) -> Optional[str]:
    """
    The lead asset's therapeutic area (`block_b.lead_asset`: furthest phase). When that one is
    unclassified, the most common classified area across the company's trials (ties alphabetical).
    `None` when no trial has a classified area.
    """
    lead = block_b.lead_asset(list(trials))
    area = lead.get("therapeutic_area") if lead else None
    if area and area != UNCLASSIFIED:
        return area
    counts = Counter(a for t in trials if (a := t.get("therapeutic_area")) and a != UNCLASSIFIED)
    if not counts:
        return None
    return min(counts, key=lambda a: (-counts[a], a))


@dataclass
class DealAreaHistory:
    """
    Past strategic acquisitions with each target's lead area as of its announcement date. One
    instance can serve every snapshot of a backtest: an area, once computed, is cached.
    """

    repo: Any
    deals: Sequence[Mapping[str, Any]]
    _areas: Dict[Tuple[int, date], Optional[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.deals = [d for d in self.deals if d.get("deal_type") == POSITIVE_DEAL_TYPE and d.get("announcement_date") is not None]

    def deal_area(self, deal: Mapping[str, Any]) -> Optional[str]:
        """The deal target's lead area from the trials public on the announcement date."""
        key = (int(deal["target_id"]), deal["announcement_date"])
        if key not in self._areas:
            ctx = FeatureContext(as_of=key[1], repo=self.repo)
            self._areas[key] = lead_therapeutic_area(ctx.get_trials_for_company(key[0]))
        return self._areas[key]

    def heat(self, area: Optional[str], as_of: date, months: int = DEAL_HEAT_MONTHS) -> Optional[float]:
        """Deals announced in `(as_of - months, as_of]` whose target's lead area was `area`."""
        if area is None:
            return None
        start = add_months(as_of, -months)
        return float(sum(1 for d in self.deals if start < d["announcement_date"] <= as_of and self.deal_area(d) == area))


def ta_demand_features(
    area: Optional[str],
    acquirer_phase3_by_area: Mapping[int, Mapping[str, int]],
    as_of: date,
    deal_history: Optional[DealAreaHistory],
    acquirer_growth: Optional[Mapping[int, Optional[float]]] = None,
) -> Dict[str, Optional[float]]:
    """
    The three features for one target whose lead area is `area`.

    Args:
        area: The target's lead therapeutic area, or None.
        acquirer_phase3_by_area: `{acquirer_id: {area: Phase III asset count}}` as of `as_of`.
        as_of: Snapshot date.
        deal_history: Past deals; None when not available (deal heat is then None).
        acquirer_growth: `{acquirer_id: latest annual revenue growth or None}` as of `as_of`.
    """
    if area is None:
        return {"ta_acquirer_demand": None, "ta_acquirer_phase3_assets": None, "ta_deal_heat_24m": None,
                "ta_acquirer_revenue_growth": None}
    present = [cid for cid, by_area in acquirer_phase3_by_area.items() if by_area.get(area, 0) > 0]
    growth = sorted(g for cid in present if (g := (acquirer_growth or {}).get(cid)) is not None)
    return {
        "ta_acquirer_demand": float(len(present)),
        "ta_acquirer_phase3_assets": float(sum(by_area.get(area, 0) for by_area in acquirer_phase3_by_area.values())),
        "ta_deal_heat_24m": deal_history.heat(area, as_of) if deal_history is not None else None,
        "ta_acquirer_revenue_growth": float(median(growth)) if growth else None,
    }

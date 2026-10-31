"""
P22 — walk-forward backtest harness (spec §0.3, §8.4, M6), 2026-10-30.

For each monthly snapshot `as_of`:

1. **Universe, point in time.** The latest landed DERA quarter ending on or before `as_of`
   (`entity_resolution.build_universe_history`: reporting in the lookback window, not a SPAC),
   mapped to `p22_company` by CIK. `p22_company.is_active` describes today and is ignored here —
   using it would drop every company later acquired, i.e. every positive.
2. **Same filters as live scoring** (`score_run.select_targets` with `require_active=False`): not an
   acquirer as of `as_of`, size floor and asset floor not known to fail. Companies whose own deal is
   already pending at `as_of` are removed first (`labels.DealLabels.pending`).
3. **Rankings:** the Phase 1 model (`score_universe` restricted to those targets), naive-informed and
   catalyst-proximity (`baselines`); random is reported as its exact expectation.
4. **Metrics** at each k, for all names and separately for Tier 0 names (spec §9: Tier 0 is the
   honest measure of Blocks A–F), pooled over snapshots (`metrics`).

Snapshots whose 24-month label window runs past `labels_complete_through` are skipped (censored).
The spec's 24-month embargo (§5.5, §8.4) separates training from test data; Phase 1 has nothing
fitted, so every snapshot is out-of-sample and no embargo applies until the M8 model.

Known lookahead gaps inherited from the feature store, not introduced here: `p22_trial` is not
bitemporal (phases and completion dates are today's), and the DERA quarter is used from its quarter
end although DERA publishes it some weeks later.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.backtest import baselines
from src.ml.pipeline.p22_biotech_ma.backtest.labels import CvrPolicy, DealLabels, add_months, is_censored, return_labels
from src.ml.pipeline.p22_biotech_ma.backtest.metrics import DEFAULT_KS, MetricsTable
from src.ml.pipeline.p22_biotech_ma.features import block_b, block_c
from src.ml.pipeline.p22_biotech_ma.features.context import FeatureContext
from src.ml.pipeline.p22_biotech_ma.ingest.acquirer_config import AcquirerConfig
from src.ml.pipeline.p22_biotech_ma.ingest.entity_resolution import UniverseCandidate, quarter_end_date
from src.ml.pipeline.p22_biotech_ma.scoring.score_run import resolve_acquirers, score_universe, select_targets
from src.ml.pipeline.p22_biotech_ma.scoring.weights_config import ScoringConfig
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

RANKINGS = ("model", "naive_informed", "catalyst_proximity")


@dataclass(frozen=True)
class BacktestWindow:
    """Snapshot dates and label coverage."""

    start: date
    end: date
    labels_complete_through: date
    step_months: int = 1
    ks: Sequence[int] = DEFAULT_KS


@dataclass
class BacktestResult:
    """Pooled metrics for both populations plus per-snapshot detail and return labels."""

    all_names: MetricsTable
    tier0: MetricsTable
    snapshots: List[Dict[str, Any]] = field(default_factory=list)
    return_rows: List[Dict[str, Any]] = field(default_factory=list)


def snapshot_dates(window: BacktestWindow) -> List[date]:
    """Month-stepped dates from `start` through `end`, dropping censored ones."""
    dates: List[date] = []
    current = window.start
    while current <= window.end:
        if not is_censored(current, window.labels_complete_through):
            dates.append(current)
        current = add_months(current, window.step_months)
    return dates


def universe_as_of(history: Mapping[str, Sequence[UniverseCandidate]], as_of: date) -> tuple[Optional[str], List[str]]:
    """
    CIKs eligible at `as_of`: the latest quarter ending on or before it, eligible by reporting and
    not flagged as a SPAC. `(None, [])` when no quarter is that old.
    """
    usable = [(end, q) for q in history if (end := quarter_end_date(q)) is not None and end <= as_of]
    if not usable:
        return None, []
    _, quarter = max(usable)
    return quarter, [c.cik for c in history[quarter] if c.eligible_reporting and not c.flagged_spac]


def baseline_features(company_ids: Sequence[int], as_of: date, ctx: FeatureContext) -> Dict[int, Dict[str, Optional[float]]]:
    """The three inputs the §0.3 baselines rank on."""
    features: Dict[int, Dict[str, Optional[float]]] = {}
    for company_id in company_ids:
        days = block_b.days_to_next_catalyst(company_id, as_of, ctx)
        features[company_id] = {
            "phase_max": block_b.phase_max(company_id, as_of, ctx),
            "cash_runway_months": block_c.cash_runway_months(company_id, as_of, ctx),
            "days_to_next_catalyst": float(days) if days is not None else None,
        }
    return features


def _add_rankings(table: MetricsTable, rankings: Mapping[str, List[int]], positives: set[int], universe_size: int) -> None:
    for name in RANKINGS:
        table.add(name, rankings[name], positives, universe_size)


def run_snapshot(
    repo: Any,
    as_of: date,
    cik_universe: Sequence[str],
    labels: DealLabels,
    scoring_config: ScoringConfig,
    acquirers: Sequence[AcquirerConfig],
) -> Dict[str, Any]:
    """
    Score one snapshot and build every ranking. Writes nothing.

    Returns:
        `{"targets", "rows", "rankings", "tier0_rankings", "tier0_ids", "positives", "pe_positives", "excluded"}`.
    """
    companies = repo.list_companies_full()
    by_cik = {c["cik"]: c for c in companies if c.get("cik")}
    pending = labels.pending(as_of)
    snapshot_companies = [by_cik[cik] for cik in cik_universe if cik in by_cik and by_cik[cik]["company_id"] not in pending]

    ctx = FeatureContext(as_of=as_of, repo=repo)
    acquirer_ids = {cid for _, cid in resolve_acquirers(acquirers, companies, as_of)}
    targets, excluded = select_targets(repo, ctx, snapshot_companies, acquirer_ids, as_of, require_active=False)
    excluded = {**excluded, "pending_deal": sum(1 for cik in cik_universe if cik in by_cik and by_cik[cik]["company_id"] in pending),
                "no_company_row": sum(1 for cik in cik_universe if cik not in by_cik)}

    result = score_universe(repo, as_of, scoring_config, acquirers, target_ids=targets)
    features = baseline_features(targets, as_of, ctx)
    tier0_ids = {r["company_id"] for r in result.rows if r["tier"] == 0}
    tier0_features = {cid: f for cid, f in features.items() if cid in tier0_ids}
    target_set = set(targets)
    return {
        "targets": targets,
        "rows": result.rows,
        "rankings": {
            "model": baselines.model_ranking(result.rows),
            "naive_informed": baselines.naive_informed(features),
            "catalyst_proximity": baselines.catalyst_proximity(features),
        },
        "tier0_rankings": {
            "model": baselines.model_ranking(result.rows, tier0_ids),
            "naive_informed": baselines.naive_informed(tier0_features),
            "catalyst_proximity": baselines.catalyst_proximity(tier0_features),
        },
        "tier0_ids": tier0_ids,
        "positives": labels.positives(as_of) & target_set,
        "pe_positives": labels.pe_positives(as_of) & target_set,
        "excluded": excluded,
    }


def run_backtest(
    repo: Any,
    history: Mapping[str, Sequence[UniverseCandidate]],
    window: BacktestWindow,
    labels: DealLabels,
    scoring_config: ScoringConfig,
    acquirers: Sequence[AcquirerConfig],
    cvr_policy: CvrPolicy,
    on_snapshot: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> BacktestResult:
    """
    Walk every snapshot in `window` and pool the metrics.

    Args:
        repo: A `P22Repo`-shaped object (read-only use).
        history: `build_universe_history` output, `{quarter: [UniverseCandidate]}`.
        window: Snapshot dates and label coverage.
        labels: The entered deals.
        scoring_config: Parsed `p22_weights.yaml`.
        acquirers: Parsed `p22_acquirers.yaml`.
        cvr_policy: Parsed `p22_cvr_policy.yaml`, for the return labels.
        on_snapshot: Called with each snapshot's summary row (progress logging).
    """
    result = BacktestResult(all_names=MetricsTable(ks=window.ks), tier0=MetricsTable(ks=window.ks))
    dates = snapshot_dates(window)
    if not dates:
        _logger.warning("No uncensored snapshots between %s and %s (labels complete through %s)",
                        window.start, window.end, window.labels_complete_through)
    for as_of in dates:
        quarter, ciks = universe_as_of(history, as_of)
        if not ciks:
            _logger.warning("No DERA quarter on or before %s — snapshot skipped", as_of)
            continue
        snap = run_snapshot(repo, as_of, ciks, labels, scoring_config, acquirers)
        targets, positives = snap["targets"], snap["positives"]
        tier0_ids = snap["tier0_ids"]
        _add_rankings(result.all_names, snap["rankings"], positives, len(targets))
        _add_rankings(result.tier0, snap["tier0_rankings"], positives & tier0_ids, len(tier0_ids))
        result.return_rows.extend(return_labels(repo, labels, sorted(positives), as_of, cvr_policy))

        top50 = set(snap["rankings"]["model"][:50])
        summary = {
            "as_of": as_of,
            "quarter": quarter,
            "universe": len(ciks),
            "targets": len(targets),
            "ranked_by_model": len(snap["rankings"]["model"]),
            "tier0": len(tier0_ids),
            "positives": len(positives),
            "pe_positives": len(snap["pe_positives"]),
            "model_hits_top50": len(top50 & positives),
            **{f"excluded_{k}": v for k, v in snap["excluded"].items()},
        }
        result.snapshots.append(summary)
        if on_snapshot is not None:
            on_snapshot(summary)
    return result

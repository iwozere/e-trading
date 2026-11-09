"""
P22 — rank the live universe with the learned model (2026-11-09).

The weekly backtest job trains the model (`backtest/learned.py`, `jobs/run_backtest.py`). Here each
scored row is flattened exactly as the backtest flattens a snapshot (`harness.score_rows_for_snapshot`
plus `harness.baseline_features`), so the model sees the same columns it was trained on, and gets
`rank_by_learned` plus `subscores["learned"]`. The composite and its blocks stay as they are: they
explain a name, the learned rank orders the watchlist.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.backtest.harness import baseline_features, score_rows_for_snapshot
from src.ml.pipeline.p22_biotech_ma.backtest.learned import SavedModel, rank_by_probability
from src.ml.pipeline.p22_biotech_ma.features.context import FeatureContext


def learned_ranking(repo: Any, as_of: date, rows: List[Dict[str, Any]], model: SavedModel) -> Dict[int, Tuple[float, int]]:
    """`{company_id: (P(acquired within 24 months), rank)}` for the scored `rows`, rank 1 best."""
    if not rows:
        return {}
    ids = [r["company_id"] for r in rows]
    ctx = FeatureContext(as_of=as_of, repo=repo)
    flat = score_rows_for_snapshot(as_of, rows, baseline_features(ids, as_of, ctx), set(), set())
    probabilities = dict(zip(ids, model.fitted.predict(flat)))
    order = rank_by_probability(probabilities, salt=as_of.isoformat())
    return {cid: (probabilities[cid], position) for position, cid in enumerate(order, start=1)}


def apply_learned(rows: List[Dict[str, Any]], ranking: Dict[int, Tuple[float, int]], model: SavedModel) -> None:
    """Add `rank_by_learned` and `subscores["learned"]` to every row in `ranking` (in place)."""
    for row in rows:
        entry = ranking.get(row["company_id"])
        if entry is None:
            row["rank_by_learned"] = None
            continue
        probability, rank = entry
        row["rank_by_learned"] = rank
        row["subscores"]["learned"] = {
            "p": round(probability, 6),
            "rank": rank,
            "labels_through": model.labels_through.isoformat(),
            "trained_at": model.trained_at.isoformat(),
            "train_rows": model.fitted.train_rows,
        }


def by_learned_rank(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """`rows` ordered by `rank_by_learned` (unranked last, then by composite rank)."""
    return sorted(rows, key=lambda r: (r.get("rank_by_learned") is None, r.get("rank_by_learned") or 0,
                                       r.get("rank_by_composite") is None, r.get("rank_by_composite") or 0))

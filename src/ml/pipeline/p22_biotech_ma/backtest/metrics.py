"""
P22 — ranking metrics for the walk-forward backtest (spec §0.3, §8.4), M6, 2026-10-30.

Per snapshot: `hits@k` among the first k ranked names, against the positives in that snapshot's
eligible universe. Across snapshots the harness pools counts (micro-averaging) rather than
averaging ratios, so a month with two positives does not weigh as much as one with twenty.

`precision@k` divides by k even when a ranking holds fewer than k names (the catalyst baseline only
ranks Phase II+), so a strategy that cannot fill k slots is not flattered. Random-ranking figures
are the exact expectations of a shuffle, not a sampled shuffle: precision = base rate, lift = 1,
recall = k / N.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_KS = (25, 50, 200)


@dataclass(frozen=True)
class SnapshotCounts:
    """One ranking's counts on one snapshot at one k."""

    k: int
    hits: int
    positives: int
    universe_size: int

    @property
    def base_rate(self) -> float:
        return self.positives / self.universe_size if self.universe_size else 0.0


def snapshot_counts(ranked: Sequence[int], positives: Iterable[int], universe_size: int, k: int) -> SnapshotCounts:
    """
    Count positives among the first `k` names of `ranked`.

    Args:
        ranked: Company ids, best first. May be shorter than `k`.
        positives: Company ids labelled positive in this snapshot (within the universe).
        universe_size: Number of companies eligible in this snapshot (N).
        k: Cut-off.
    """
    positive_set = set(positives)
    hits = sum(1 for company_id in ranked[:k] if company_id in positive_set)
    return SnapshotCounts(k=k, hits=hits, positives=len(positive_set), universe_size=universe_size)


@dataclass
class PooledMetrics:
    """Counts pooled over snapshots for one ranking at one k."""

    k: int
    snapshots: int = 0
    hits: int = 0
    slots: int = 0
    positives: int = 0
    expected_random_hits: float = 0.0
    random_recall_sum: float = 0.0
    snapshots_with_positives: int = 0

    def add(self, counts: SnapshotCounts) -> None:
        """Fold one snapshot's counts in."""
        self.snapshots += 1
        self.hits += counts.hits
        self.slots += counts.k
        self.positives += counts.positives
        self.expected_random_hits += counts.k * counts.base_rate
        if counts.universe_size:
            self.random_recall_sum += min(1.0, counts.k / counts.universe_size)
        if counts.positives:
            self.snapshots_with_positives += 1

    def as_dict(self) -> Dict[str, Any]:
        """`precision`, `lift`, `recall`, the random figures beside them, and the raw counts."""
        precision = self.hits / self.slots if self.slots else None
        random_precision = self.expected_random_hits / self.slots if self.slots else None
        return {
            "k": self.k,
            "snapshots": self.snapshots,
            "hits": self.hits,
            "positives": self.positives,
            "precision": precision,
            "random_precision": random_precision,
            "lift": self.hits / self.expected_random_hits if self.expected_random_hits else None,
            "recall": self.hits / self.positives if self.positives else None,
            "random_recall": self.random_recall_sum / self.snapshots if self.snapshots else None,
        }


@dataclass
class MetricsTable:
    """`{ranking name: {k: PooledMetrics}}` for one population (all names, or Tier 0)."""

    ks: Sequence[int] = DEFAULT_KS
    rankings: Dict[str, Dict[int, PooledMetrics]] = field(default_factory=dict)

    def add(self, name: str, ranked: Sequence[int], positives: Iterable[int], universe_size: int) -> None:
        """Score one ranking on one snapshot at every k."""
        positive_list = list(positives)
        per_k = self.rankings.setdefault(name, {k: PooledMetrics(k=k) for k in self.ks})
        for k in self.ks:
            per_k[k].add(snapshot_counts(ranked, positive_list, universe_size, k))

    def as_rows(self) -> List[Dict[str, Any]]:
        """One flat row per (ranking, k), for CSV and logging."""
        rows: List[Dict[str, Any]] = []
        for name, per_k in self.rankings.items():
            for k in self.ks:
                rows.append({"ranking": name, **per_k[k].as_dict()})
        return rows

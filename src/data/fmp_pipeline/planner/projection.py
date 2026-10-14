"""
Bandwidth and disk projection (spec §6.3.2, §6.8).

Finished work is counted with its **real** bytes from the manifest, follow-up pages and window halves included;
remaining work with the generator's estimate. Both counting hypotheses are kept side by side, because whether FMP's
50 GB cap counts wire (gzip) or decoded bytes is a ~7× question the paid probe answers (probe F10).

The schedule walks the remaining work in execution order (blocks, then intraday tiers) and fills months of
``safety × cap`` each. Month 1 has less room by what the rolling window already holds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from src.data.fmp_pipeline.manifest.manifest import FINAL_STATUSES
from src.data.fmp_pipeline.planner.catalog import BLOCKS
from src.data.fmp_pipeline.planner.generate import PlannedTask

GB = 1024**3


@dataclass
class Bucket:
    """Requests and bytes of one block or tier."""

    done: int = 0
    done_wire: int = 0
    done_decoded: int = 0
    planned: int = 0
    deferred: int = 0
    rows: float = 0.0
    wire: float = 0.0
    decoded: float = 0.0
    assumed: bool = False

    @property
    def remaining(self) -> int:
        """Requests still to make (written to the manifest or deferred)."""
        return self.planned + self.deferred

    def add(self, task: PlannedTask) -> None:
        """Count one unfinished planned task."""
        if task.deferred:
            self.deferred += 1
        else:
            self.planned += 1
        self.rows += task.rows
        self.wire += task.bytes_wire
        self.decoded += task.bytes_decoded
        self.assumed = self.assumed or not task.measured


@dataclass
class Projection:
    """Per-block and per-tier totals plus deferral reasons."""

    blocks: dict[int, Bucket] = field(default_factory=lambda: {b: Bucket() for b in BLOCKS})
    tiers: dict[str, Bucket] = field(default_factory=dict)
    bootstrap: Bucket = field(default_factory=Bucket)
    deferred_reasons: dict[str, int] = field(default_factory=dict)

    def add_done(self, priority: int, tier: Optional[str], status: str, count: int, wire: int,
                 decoded: int) -> None:
        """Count finished (or in-flight) manifest rows; only final statuses count as done."""
        if status not in FINAL_STATUSES:
            return
        buckets = [self.bootstrap if priority == 0 else self.blocks.setdefault(priority // 100, Bucket())]
        if priority // 100 == 6 and tier is not None:
            buckets.append(self.tiers.setdefault(tier, Bucket()))
        for bucket in buckets:
            bucket.done += count
            bucket.done_wire += wire
            bucket.done_decoded += decoded

    def add_planned(self, task: PlannedTask) -> None:
        """Count one generated task that is not finished yet."""
        self.blocks.setdefault(task.block, Bucket()).add(task)
        if task.block == 6 and task.spec.tier is not None:
            self.tiers.setdefault(task.spec.tier, Bucket()).add(task)
        if task.deferred:
            self.deferred_reasons[task.deferred] = self.deferred_reasons.get(task.deferred, 0) + 1

    def remaining_bytes(self) -> tuple[float, float]:
        """Return remaining ``(wire, decoded)`` bytes over all blocks."""
        return sum(b.wire for b in self.blocks.values()), sum(b.decoded for b in self.blocks.values())

    def schedule(self, budget_bytes: float, used_wire: int, used_decoded: int) -> list[tuple[str, int, int]]:
        """
        Return ``(label, month under wire counting, month under decoded counting)`` for each remaining step.

        Args:
            budget_bytes: ``safety × cap`` per 30-day window.
            used_wire: Wire bytes already in the current window.
            used_decoded: Decoded bytes already in the current window.
        """
        steps: list[tuple[str, float, float]] = []
        for block, bucket in sorted(self.blocks.items()):
            if block == 6 and self.tiers:
                steps += [(f"intraday tier {t}", b.wire, b.decoded) for t, b in sorted(self.tiers.items())]
            elif bucket.remaining:
                steps.append((BLOCKS.get(block, str(block)), bucket.wire, bucket.decoded))
        out = []
        total_wire, total_decoded = float(used_wire), float(used_decoded)
        for label, wire, decoded in steps:
            total_wire += wire
            total_decoded += decoded
            out.append((label, _month(total_wire, budget_bytes), _month(total_decoded, budget_bytes)))
        return out


def _month(cumulative: float, budget: float) -> int:
    return max(1, math.ceil(cumulative / budget)) if budget > 0 else 0


def disk_need(projection: Projection, overhead: float, vintage_gb: float, build_temp_gb: float) -> int:
    """
    Estimate the free space the rest of the dump needs on the data volume.

    The archive stores each response gzip-compressed, so its growth is about the remaining **wire** bytes times
    ``overhead`` (one gzip member per response compresses a little worse than FMP's stream). The vintage and the
    build's temporary space are the spec §6.8 estimates until ``fmp build`` measures them.
    """
    wire, _ = projection.remaining_bytes()
    return int(wire * overhead + (vintage_gb + build_temp_gb) * GB)

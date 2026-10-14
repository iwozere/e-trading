"""
``fmp run``: execute a planned dump block by block (spec §6.1, §6.3.5, §6.8).

Blocks run in the strict spec order (``block = priority // 100``); within a block the manifest's priority order
already puts delisted symbols first and intraday tiers in order. Before each block the free space on the archive
volume is checked against what the last ``fmp plan`` projected for the remaining blocks in scope (remaining wire
bytes × ``archive_overhead``, plus a 20% margin), so the run fails early instead of filling the disk (D43).

Every block uses one ``Downloader`` with the ``RawArchive`` sink, the planner's ``FollowUps`` reviewer (paging,
daily bounds, intraday truncation) and one shared ``BandwidthGuard`` (rolling-window hard stop plus the optional
per-run ``--max-gb`` limit). The first run-level stop (quota, auth, bandwidth, archive) ends the whole run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.data.fmp_pipeline.archive.archive import RawArchive
from src.data.fmp_pipeline.client.fmp_client import FmpClient
from src.data.fmp_pipeline.fsutil import ensure_free_space
from src.data.fmp_pipeline.manifest.manifest import Manifest
from src.data.fmp_pipeline.planner.catalog import BLOCKS
from src.data.fmp_pipeline.planner.config import PlanConfig
from src.data.fmp_pipeline.planner.follow_ups import FollowUps
from src.data.fmp_pipeline.planner.planner import bandwidth_guard, planned_remaining
from src.data.fmp_pipeline.planner.universe import TIERS
from src.data.fmp_pipeline.runner import GB, Downloader, RunSummary
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

INTRADAY_BLOCK = 6


@dataclass(frozen=True)
class RunScope:
    """Which part of the dump a run executes: one block (``--phase``), one intraday tier (``--tier``), or all."""

    phase: Optional[int] = None
    tier: Optional[str] = None

    def __post_init__(self) -> None:
        if self.phase is not None and self.phase not in BLOCKS:
            raise ValueError(f"--phase must be one of {sorted(BLOCKS)}")
        if self.tier is not None and self.tier not in TIERS:
            raise ValueError(f"--tier must be one of {', '.join(TIERS)}")
        if self.tier is not None and self.phase not in (None, INTRADAY_BLOCK):
            raise ValueError(f"--tier applies to intraday (phase {INTRADAY_BLOCK}) only")

    def blocks(self) -> list[int]:
        """Blocks to run, in execution order."""
        if self.tier is not None:
            return [INTRADAY_BLOCK]
        return [self.phase] if self.phase is not None else sorted(BLOCKS)


@dataclass
class RunReport:
    """What one ``fmp run`` did."""

    blocks: dict[int, RunSummary] = field(default_factory=dict)
    stop_reason: str = "done"
    run_bytes: int = 0

    @property
    def processed(self) -> int:
        """Tasks finished in this run."""
        return sum(s.processed for s in self.blocks.values())


class DumpExecutor:
    """Runs one dump's pending tasks within a scope."""

    def __init__(self, manifest: Manifest, archive: RawArchive, config: PlanConfig,
                 max_gb: Optional[float] = None, workers: Optional[int] = None) -> None:
        """
        Args:
            manifest: The dump's manifest (planned by ``fmp plan``).
            archive: The dump's raw archive.
            config: Pipeline settings (bandwidth cap, intraday row cap, workers, archive overhead).
            max_gb: Optional per-run limit, counted on the configured bandwidth basis.
            workers: Worker count; defaults to ``config.workers``.

        Raises:
            ValueError: If the manifest and config disagree on the dump id.
        """
        if manifest.dump_id != config.dump_id:
            raise ValueError(f"dump id mismatch: manifest {manifest.dump_id}, config {config.dump_id}")
        self.manifest = manifest
        self.archive = archive
        self.config = config
        self.workers = workers or config.workers
        self.guard = bandwidth_guard(config)
        if max_gb is not None:
            self.guard.run_limit_bytes = int(max_gb * GB)

    def remaining_wire(self) -> dict[int, int]:
        """
        Return the last plan's projected remaining wire bytes per block.

        Raises:
            RuntimeError: No ``fmp plan`` (without ``--dry-run``) has been run for this dump.
        """
        projected = planned_remaining(self.manifest)
        if projected is None:
            raise RuntimeError(f"dump {self.manifest.dump_id} has not been planned; run `fmp plan` first")
        return {block: wire for block, (wire, _) in projected.items()}

    def check_disk(self, blocks: list[int]) -> int:
        """
        Fail early if the archive volume cannot hold the remaining blocks in scope (need + 20%).

        Args:
            blocks: Blocks still to run in this scope.

        Returns:
            Free bytes on the volume.

        Raises:
            InsufficientDiskSpace: Not enough room.
        """
        projected = self.remaining_wire()
        need = int(sum(projected.get(b, 0) for b in blocks) * self.config.archive_overhead)
        return ensure_free_space(self.archive.raw_root, need)

    async def run(self, client: FmpClient, scope: RunScope) -> RunReport:
        """
        Run every block in scope, in order, until done or a run-level stop.

        Args:
            client: FMP client (its ``on_bytes`` hook is taken over by each block's ``Downloader``).
            scope: Blocks and tier to run.

        Returns:
            Per-block summaries and the stop reason.

        Raises:
            RuntimeError: The dump has not been planned.
            InsufficientDiskSpace: Free space is below the projection for the remaining blocks.
        """
        report = RunReport()
        blocks = scope.blocks()
        self._warn_on_skipped_order(blocks)
        reviewer = FollowUps(self.manifest, self.config)
        for i, block in enumerate(blocks):
            free = self.check_disk(blocks[i:])
            _logger.info("Block %d (%s): starting, %.1f GB free on the archive volume", block, BLOCKS[block],
                         free / GB)
            downloader = Downloader(self.manifest, client, sink=self.archive, workers=self.workers, guard=self.guard,
                                    reviewer=reviewer, min_priority=block * 100, max_priority=block * 100 + 99,
                                    tier=scope.tier)
            summary = await downloader.run()
            report.blocks[block] = summary
            if summary.stop_reason != "done":
                report.stop_reason = summary.stop_reason
                break
        report.run_bytes = self.guard.run_used
        _logger.info("Run stopped (%s): %d tasks, %.2f GB (%s) this run", report.stop_reason, report.processed,
                     report.run_bytes / GB, self.config.bandwidth.basis)
        return report

    def _warn_on_skipped_order(self, blocks: list[int]) -> None:
        """Warn if earlier blocks still have pending work: spec §6.3.5 wants them first."""
        earlier = min(blocks)
        pending = sum(n for priority, _, status, n, _, _ in self.manifest.totals()
                      if status == "pending" and priority // 100 < earlier)
        if pending:
            _logger.warning("%d tasks of earlier blocks are still pending; spec §6.3.5 runs blocks in order", pending)

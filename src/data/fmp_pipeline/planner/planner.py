"""
``fmp plan``: fetch the universe source lists, assemble the universe, generate tasks, project the budget.

Planning is **iterative and idempotent** (D36). Each run plans everything it can decide from what is downloaded so
far and defers the rest:

1. ``fetch_sources``: the bootstrap lists (``stock-list``, ``delisted-companies``, constituents, …) are ordinary
   priority-0 tasks of the dump. They go through the normal ``Downloader`` into the raw archive, paging included,
   so a later ``fmp run`` never fetches them again and a second ``fmp plan`` costs no calls.
2. ``plan``: reads those lists and any finished profiles back from the archive, assembles the universe, generates
   tasks in spec §6.3.5 order, writes the new ones to the manifest (unless dry run), marks the pending tasks of
   symbols that left the universe ``skipped``, and renders ``budget_plan.md`` and ``universe_report.md``.

The dump's ``as_of`` date is stored in the manifest by the first plan and reused afterwards: it is part of every
date window, so it must not drift between plans (task ids would change).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.data.fmp_pipeline.archive.archive import SUFFIX, RawArchive
from src.data.fmp_pipeline.archive.reader import SUCCESS_CLASSIFICATIONS, iter_records, latest_successful
from src.data.fmp_pipeline.archive.symbol_path import decode_symbol, endpoint_dir
from src.data.fmp_pipeline.client.fmp_client import FmpClient
from src.data.fmp_pipeline.fsutil import atomic_write_bytes, free_bytes
from src.data.fmp_pipeline.manifest.manifest import Manifest, TaskSpec, task_id
from src.data.fmp_pipeline.planner.catalog import BOOTSTRAP, DAILY_RAW
from src.data.fmp_pipeline.planner.config import PlanConfig
from src.data.fmp_pipeline.planner.follow_ups import FollowUps
from src.data.fmp_pipeline.planner.generate import bootstrap_specs, generate
from src.data.fmp_pipeline.planner.projection import GB, Projection, disk_need
from src.data.fmp_pipeline.planner.report import PlanContext, render_budget_plan, render_universe_report
from src.data.fmp_pipeline.planner.universe import Universe, assemble_universe
from src.data.fmp_pipeline.runner import BandwidthGuard, Downloader, RunSummary
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

INSERT_BATCH = 20_000
#: Manifest meta key: ``{block: {"wire": bytes, "decoded": bytes}}`` still to fetch, as projected by the last
#: non-dry-run plan (``fmp run`` checks disk space with it, ``fmp status`` estimates time).
REMAINING_KEY = "plan_remaining_bytes"


def planned_remaining(manifest: Manifest) -> Optional[dict[int, tuple[int, int]]]:
    """Return ``{block: (wire, decoded)}`` projected by the last plan, or ``None`` if the dump was never planned."""
    stored = manifest.meta(REMAINING_KEY)
    if stored is None:
        return None
    return {int(block): (int(v["wire"]), int(v["decoded"])) for block, v in json.loads(stored).items()}


@dataclass(frozen=True)
class PlanResult:
    """What ``Planner.plan`` produced."""

    universe: Universe
    projection: Projection
    context: PlanContext
    skipped: int
    budget_path: Path
    universe_path: Path


def bandwidth_guard(config: PlanConfig) -> BandwidthGuard:
    """Return the run's bandwidth hard stop built from the configuration."""
    bw = config.bandwidth
    return BandwidthGuard(cap_bytes=int(bw.cap_gb * GB), safety=bw.safety, basis=bw.basis,  # type: ignore[arg-type]
                          window_days=bw.window_days)


class Planner:
    """Plans one dump."""

    def __init__(self, manifest: Manifest, archive: RawArchive, config: PlanConfig, report_dir: Path,
                 as_of: Optional[date] = None) -> None:
        """
        Args:
            manifest: The dump's manifest.
            archive: The dump's raw archive (source of the bootstrap lists and profiles).
            config: Planner settings; ``config.dump_id`` must match the manifest's.
            report_dir: Where ``budget_plan.md`` and ``universe_report.md`` are written.
            as_of: Dump date for the **first** plan (default: today UTC); ignored once the manifest has one.

        Raises:
            ValueError: If the manifest, archive and config disagree on the dump id.
        """
        if not (manifest.dump_id == config.dump_id == archive.dump_dir.name):
            raise ValueError(f"dump id mismatch: manifest {manifest.dump_id}, config {config.dump_id}, "
                             f"archive {archive.dump_dir.name}")
        self.manifest = manifest
        self.archive = archive
        self.config = config
        self.report_dir = report_dir
        requested = as_of or datetime.now(timezone.utc).date()
        self.as_of = date.fromisoformat(manifest.meta_setdefault("as_of", requested.isoformat()))
        if as_of is not None and as_of != self.as_of:
            _logger.warning("Dump %s is fixed to as_of %s; ignoring requested %s", config.dump_id, self.as_of, as_of)

    async def fetch_sources(self, client: FmpClient) -> RunSummary:
        """
        Download the bootstrap lists (all pages) into the archive. Already finished ones are not fetched again.

        Args:
            client: FMP client.

        Returns:
            The download summary; ``stop_reason`` other than ``done`` means the lists may be incomplete.
        """
        self.manifest.add_tasks(bootstrap_specs(self.config))
        downloader = Downloader(self.manifest, client, sink=self.archive, workers=min(self.config.workers, 4),
                                guard=bandwidth_guard(self.config), reviewer=FollowUps(self.manifest, self.config),
                                max_priority=0)
        return await downloader.run()

    def plan(self, dry_run: bool = False) -> PlanResult:
        """
        Assemble the universe, generate and (unless ``dry_run``) store tasks, and write both reports.

        Args:
            dry_run: Project only; the manifest is not changed.

        Returns:
            The universe, the projection and the report paths.
        """
        sources, status = self._read_sources()
        universe = assemble_universe(sources, self._read_profiles(), self.config.universe, self.as_of)
        universe.source_status = status

        projection = Projection()
        for row in self.manifest.totals():
            projection.add_done(*row)
        finished = self.manifest.finished_ids()
        bounds = self.manifest.bounds()
        daily_done = self.manifest.symbols_finished(DAILY_RAW)
        added, batch = 0, []
        for planned in generate(universe, self.config, bounds, daily_done):
            spec = planned.spec
            if task_id(self.manifest.dump_id, spec.endpoint, spec.symbol, spec.params) in finished:
                continue
            projection.add_planned(planned)
            if planned.deferred is None and not dry_run:
                batch.append(spec)
                if len(batch) >= INSERT_BATCH:
                    added += self._insert(batch)
        if batch:
            added += self._insert(batch)

        skipped = 0
        if not dry_run:
            dropped = self.manifest.pending_symbols() - set(universe.securities)
            skipped = self.manifest.skip_symbols(sorted(dropped), "excluded from the universe by fmp plan")
            remaining = {str(block): {"wire": int(b.wire), "decoded": int(b.decoded)}
                         for block, b in projection.blocks.items()}
            self.manifest.meta_set(REMAINING_KEY, json.dumps(remaining, sort_keys=True))

        used_wire, used_decoded = self.manifest.bandwidth_used(self.config.bandwidth.window_days)
        need = disk_need(projection, self.config.archive_overhead, self.config.vintage_gb, self.config.build_temp_gb)
        context = PlanContext(self.config.dump_id, self.as_of, dry_run, added, used_wire, used_decoded,
                              free_bytes(self.archive.raw_root), need)
        budget_path = self.report_dir / "budget_plan.md"
        universe_path = self.report_dir / "universe_report.md"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(budget_path, render_budget_plan(projection, universe, self.config, context).encode())
        atomic_write_bytes(universe_path, render_universe_report(
            universe, self.config.universe.include_exchanges, self.config.dump_id).encode())
        _logger.info("Plan %s: %d resolved, %d unresolved, %d excluded; %d tasks added, %d skipped; reports in %s",
                     self.config.dump_id, len(universe.resolved), len(universe.unresolved),
                     sum(universe.excluded.values()), added, skipped, self.report_dir)
        return PlanResult(universe, projection, context, skipped, budget_path, universe_path)

    def _insert(self, batch: list[TaskSpec]) -> int:
        added = self.manifest.add_tasks(batch)
        batch.clear()
        return added

    # --- reading back from the archive -----------------------------------------------------------------------

    def _records(self, path: Path) -> Iterable[dict[str, Any]]:
        return iter_records(path, strict=False) if path.exists() else ()

    def _read_sources(self) -> tuple[dict[str, Optional[list[Any]]], dict[str, str]]:
        """Return ``{endpoint: all pages' rows or None}`` and a status text per bootstrap endpoint."""
        rows_by_endpoint: dict[str, list[str]] = {}
        for row in self.manifest.task_rows([e.endpoint for e in BOOTSTRAP]):
            rows_by_endpoint.setdefault(row["endpoint"], []).append(row["status"])
        sources: dict[str, Optional[list[Any]]] = {}
        status: dict[str, str] = {}
        for entry in BOOTSTRAP:
            statuses = rows_by_endpoint.get(entry.endpoint, [])
            latest = latest_successful(self._records(self.archive.path_for(entry.endpoint, None)))
            pages = sorted(latest.values(), key=lambda r: int(r["request"]["params"].get("page", 0)))
            body: Optional[list[Any]] = None
            previous: Any = None
            for record in pages:
                part = record.get("body")
                if body is not None and part == previous:
                    continue  # an endpoint that ignores `page` repeats itself (see FollowUps)
                previous = part
                body = (body or []) + (part if isinstance(part, list) else [part] if isinstance(part, dict) else [])
            sources[entry.endpoint] = body
            status[entry.endpoint] = _source_status(statuses)
        return sources, status

    def _read_profiles(self) -> dict[str, Optional[dict[str, Any]]]:
        """Return ``{symbol: profile row or None}`` for every symbol whose profile task is finished."""
        finished = self.manifest.symbols_finished("profile")
        profiles: dict[str, Optional[dict[str, Any]]] = dict.fromkeys(finished)
        folder = self.archive.dump_dir / endpoint_dir("profile")
        for path in sorted(folder.glob("*" + SUFFIX)) if folder.exists() else []:
            try:
                symbol = decode_symbol(path.name[: -len(SUFFIX)])
            except ValueError:
                _logger.warning("Ignoring archive file with a non-canonical name: %s", path)
                continue
            if symbol is None or symbol not in profiles:
                continue
            for record in latest_successful(self._records(path)).values():
                body = record.get("body")
                row = body[0] if isinstance(body, list) and body else body
                profiles[symbol] = row if isinstance(row, dict) and row else None
        return profiles


def _source_status(statuses: list[str]) -> str:
    if not statuses:
        return "not fetched"
    if all(s in SUCCESS_CLASSIFICATIONS for s in statuses):
        return f"ok ({len(statuses)} request{'s' if len(statuses) > 1 else ''})"
    if any(s in ("pending", "running") for s in statuses):
        return "incomplete"
    return ", ".join(sorted(set(statuses)))

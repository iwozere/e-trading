"""
``fmp verify --final`` (spec §14, gate G3): the checklist that must print ``READY TO CANCEL`` before the owner
cancels the FMP subscription.

1. V-COMPLETE passes and the plan's skips are listed (``budget_plan.md`` exists).
2. No FAIL validator; every WARN is in the report.
3. ``ARCHIVE.sha256`` and ``VINTAGE.sha256`` written (``sha256sum -c`` format, with file count and bytes); every
   gzip member decompressed; the copy on the backup volume (``--backup``: a packed ``fmp backup`` or a plain copy
   mirroring the ``data_root`` layout) matches; then the raw archive is made read-only; free disk space is adequate.
4. The reconciliation sample and the V-ADJ-RECON result exist.
5. V-REPRO passes.
6. The adapter's backward-compatibility and new-source tests pass.
7. Anything not downloaded is confirmed as postponed (``--confirm-postponed``), listed with estimated bytes.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.data.fmp_pipeline.fsutil import atomic_write_bytes, free_bytes
from src.data.fmp_pipeline.planner.config import PROJECT_ROOT, PlanConfig
from src.data.fmp_pipeline.planner.planner import planned_remaining
from src.data.fmp_pipeline.transfer.packing import MANIFEST, verify_packed
from src.data.fmp_pipeline.validators.checks_events import RECON_TABLE
from src.data.fmp_pipeline.validators.checksums import (
    ARCHIVE_FILE,
    VINTAGE_FILE,
    checksum_dir,
    gzip_test,
    read_summary,
    verify_checksums,
    write_checksums,
    write_summary,
)
from src.data.fmp_pipeline.validators.coverage import render_coverage
from src.data.fmp_pipeline.validators.framework import FAIL, WARN, ValidationContext, ValidationReport, run_checks
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

GB = 1024**3
ADAPTER_TESTS = ("src/data/fmp_pipeline/tests/test_adapter.py", "src/data/fmp_pipeline/tests/test_datamanager_hook.py",
                 "src/data/fmp_pipeline/tests/test_increments.py")
READY = "READY TO CANCEL"


@dataclass
class Item:
    """One checklist line."""

    number: int
    title: str
    ok: bool
    detail: str = ""

    def line(self) -> str:
        """``[PASS] 3. …`` text."""
        return f"[{'PASS' if self.ok else 'FAIL'}] {self.number}. {self.title}" + (f" — {self.detail}" if self.detail
                                                                                   else "")


def write_reports(ctx: ValidationContext, report: ValidationReport) -> list[Path]:
    """Write ``validation_report.md/.json`` and ``coverage_report.md``; return the paths."""
    paths = list(report.write(ctx.report_dir))
    manifest = ctx.manifest()
    if manifest is not None:
        with manifest:
            coverage = ctx.report_dir / "coverage_report.md"
            atomic_write_bytes(coverage, render_coverage(manifest).encode("utf-8"))
            paths.append(coverage)
    return paths


def _run_pytest(tests: tuple[str, ...]) -> tuple[bool, str]:
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests], cwd=PROJECT_ROOT,
                         capture_output=True, text=True, timeout=1800, check=False)
    return run.returncode == 0, (run.stdout.strip().splitlines() or [""])[-1]


def make_read_only(root: Path) -> int:
    """Clear the write bits of every file under ``root`` (best effort on a NAS); return how many were changed."""
    changed = 0
    for path in root.rglob("*"):
        if path.is_file():
            mode = path.stat().st_mode
            os.chmod(path, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
            changed += 1
    return changed


class FinalVerifier:
    """Runs the spec §14 checklist for the active vintage's dump."""

    def __init__(self, data_root: Path, config: PlanConfig, backup: Optional[Path] = None,
                 confirm_postponed: bool = False, legacy_cache_dir: Optional[Path] = None, sample_size: int = 100,
                 run_tests: bool = True) -> None:
        """
        Args:
            data_root: Pipeline data root.
            config: Pipeline config.
            backup: Root of the backup copy (same layout: ``raw/``, ``vintages/``, ``checksums/``).
            confirm_postponed: The owner confirms that what was not downloaded is postponed or abandoned.
            legacy_cache_dir: Legacy yfinance cache for V-YF-XCHECK.
            sample_size: Cross-source sample size.
            run_tests: Run the unit and adapter test suites (items 2 and 6); tests of this module turn it off.
        """
        self.data_root = data_root
        self.config = config
        self.backup = backup
        self.confirm_postponed = confirm_postponed
        self.ctx = ValidationContext(data_root, config, sample_size, legacy_cache_dir=legacy_cache_dir)
        self.run_tests = run_tests

    def run(self) -> tuple[list[Item], ValidationReport]:
        """Run every item; return the checklist and the validation report (also written to disk)."""
        ids = None if self.run_tests else [i for i in _all_ids() if i != "V-ADJ-UNIT"]
        report = run_checks(self.ctx, ids)
        write_reports(self.ctx, report)
        items = [self._complete(report), self._validators(report), self._archive(), self._recon(report),
                 self._repro(report), self._adapter(), self._postponed()]
        for item in items:
            (_logger.info if item.ok else _logger.error)("%s", item.line())
        _logger.info("%s", READY if all(i.ok for i in items) else "NOT READY: keep the subscription")
        return items, report

    # --- items ------------------------------------------------------------------------------------------------

    def _complete(self, report: ValidationReport) -> Item:
        result = report.result("V-COMPLETE")
        plan = self.ctx.report_dir / "budget_plan.md"
        ok = result is not None and result.status != FAIL and plan.exists()
        skipped = result.details.get("skipped_by_plan", {}) if result is not None else {}
        detail = f"{result.violations if result else '?'} open tasks; skipped by plan: {sum(skipped.values())}" \
            + ("" if plan.exists() else "; budget_plan.md missing")
        return Item(1, "No pending, running or unexplained failed tasks", ok, detail)

    def _validators(self, report: ValidationReport) -> Item:
        failed = [r.id for r in report.failed]
        warned = [r.id for r in report.results if r.status == WARN]
        return Item(2, "All FAIL-severity validators pass", not failed,
                    f"FAIL: {', '.join(failed) or 'none'}; WARN for review: {', '.join(warned) or 'none'}")

    def _archive(self) -> Item:
        dump = self.ctx.store.vintage
        directory = checksum_dir(self.data_root, dump)
        archive = write_checksums(self.ctx.dump_dir, directory / ARCHIVE_FILE)
        vintage = write_checksums(self.ctx.vintage_dir, directory / VINTAGE_FILE)
        write_summary(directory, {"archive": archive, "vintage": vintage})
        _logger.info("%s: %d files, %.2f GB, file sha256 %s", ARCHIVE_FILE, archive.files, archive.bytes / GB,
                     archive.checksum_file_sha256)
        _logger.info("%s: %d files, %.2f GB, file sha256 %s", VINTAGE_FILE, vintage.files, vintage.bytes / GB,
                     vintage.checksum_file_sha256)
        files, members, gzip_errors = gzip_test(self.ctx.dump_dir)
        problems = gzip_errors[:3]
        free = free_bytes(self.data_root)
        if free is not None and free < self.config.build_temp_gb * GB:
            problems.append(f"only {free / GB:.1f} GB free (need {self.config.build_temp_gb:g} GB to rebuild)")
        if self.backup is None:
            problems.append(f"no --backup: copy raw/{dump}, vintages/{dump} and checksums/{dump} to the backup "
                            "volume, then re-run with --backup PATH")
        else:
            problems += self._backup_errors(dump)[:5]
        if not problems:
            changed = make_read_only(self.ctx.dump_dir)
            _logger.info("Raw archive of %s marked read-only (%d files)", dump, changed)
        outcome = "; ".join(problems) or "backup matches; archive now read-only"
        return Item(3, "Archive integrity, checksums and verified backup", not problems,
                    f"{files} files / {members} gzip members read; {outcome}")

    def _backup_errors(self, dump: str) -> list[str]:
        assert self.backup is not None
        packed = self.backup / dump
        if (packed / MANIFEST).exists():  # written by `fmp backup`: check every tar member, no unpacking
            originals_dir = checksum_dir(self.data_root, dump)
            return ([f"backup raw: {e}" for e in verify_packed(packed, "raw", originals_dir / ARCHIVE_FILE)]
                    + [f"backup vintage: {e}" for e in verify_packed(packed, "vintage", originals_dir / VINTAGE_FILE)])
        originals = read_summary(checksum_dir(self.data_root, dump))
        copied = checksum_dir(self.backup, dump)
        errors = []
        for name, tree, key in ((ARCHIVE_FILE, self.backup / "raw" / dump, "archive"),
                                (VINTAGE_FILE, self.backup / "vintages" / dump, "vintage")):
            errors += [f"backup {name}: {e}" for e in verify_checksums(tree, copied / name, originals.get(key))]
        return errors

    def _recon(self, report: ValidationReport) -> Item:
        result = report.result("V-ADJ-RECON")
        exists = (self.ctx.vintage_dir / RECON_TABLE).exists()
        ok = exists and result is not None and result.status != FAIL
        return Item(4, "Reconciliation sample and V-ADJ-RECON report exist", ok,
                    result.message if result is not None else "V-ADJ-RECON did not run")

    def _repro(self, report: ValidationReport) -> Item:
        result = report.result("V-REPRO")
        ok = result is not None and result.status != FAIL and result.examined > 0
        return Item(5, "Rebuild from the archive reproduces identical hashes (V-REPRO)", ok,
                    f"{result.examined} files rebuilt, {result.violations} differ" if result else "did not run")

    def _adapter(self) -> Item:
        if not self.run_tests:
            return Item(6, "DataManager adapter backward-compatibility and new-source tests", True, "not run")
        ok, tail = _run_pytest(ADAPTER_TESTS)
        return Item(6, "DataManager adapter backward-compatibility and new-source tests", ok, tail)

    def _postponed(self) -> Item:
        manifest = self.ctx.manifest()
        remaining = {}
        if manifest is not None:
            with manifest:
                remaining = planned_remaining(manifest) or {}
        basis = 0 if self.config.bandwidth.basis == "wire" else 1
        open_blocks = {block: v[basis] for block, v in remaining.items() if v[basis] > 0}
        listing = ", ".join(f"block {b}: {n / GB:.2f} GB" for b, n in sorted(open_blocks.items())) or "nothing"
        ok = not open_blocks or self.confirm_postponed
        detail = f"not downloaded per the last plan: {listing}" + (
            "" if ok else " — re-run with --confirm-postponed once you accept this")
        return Item(7, "Everything not downloaded is confirmed as postponed or abandoned", ok, detail)


def _all_ids() -> list[str]:
    from src.data.fmp_pipeline.validators.checks import ORDER

    return list(ORDER)

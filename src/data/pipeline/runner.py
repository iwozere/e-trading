"""
Standalone "download everything in scope" runner.

Iterates `registry.PLUGIN_REGISTRY` and runs each plugin's script as a
subprocess, mirroring the exact invocation shape
`SchedulerService._execute_data_processing_job` uses in production
(`src/scheduler/scheduler_service.py`) — same allowlist validation
(`PluginSpec.validate`), same `python <script_path> <script_args>` command,
same `__SCHEDULER_RESULT__:{json}` stdout convention, same per-job timeout.
This is what makes a scheduled-cron run and a manual `runner.py` run of the
same plugin behave identically.

Purpose: ad hoc full-cache rebuilds (new machine bootstrap, disaster
recovery) or re-running one category on demand — NOT a replacement for the
per-plugin cron schedules in `job_schedules`, which keep running independently
via APScheduler regardless of whether this is ever invoked.

Usage:
    # List what's in scope without running anything.
    python -m src.data.pipeline.runner --scope all --dry-run

    # Run everything, sequentially, in registry order.
    python -m src.data.pipeline.runner --scope all

    # Run one category or one named plugin.
    python -m src.data.pipeline.runner --scope p22
    python -m src.data.pipeline.runner --scope "P22 Daily Price Ingest"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.data.pipeline.base_plugin import PROJECT_ROOT, PluginSpec
from src.data.pipeline.registry import PLUGIN_REGISTRY, get_by_category, get_by_name
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass(frozen=True)
class PluginRunResult:
    """Outcome of running one `PluginSpec`'s script."""

    name: str
    success: bool
    exit_code: Optional[int]
    duration_seconds: float
    script_result: Dict[str, Any]
    error: Optional[str] = None


def _resolve_scope(scope: str) -> List[PluginSpec]:
    if scope == "all":
        return list(PLUGIN_REGISTRY)
    by_category = get_by_category(scope)
    if by_category:
        return by_category
    by_name = get_by_name(scope)
    if by_name:
        return [by_name]
    raise ValueError(
        f"Unknown scope '{scope}' — not 'all', not a registered category "
        f"({sorted({s.category for s in PLUGIN_REGISTRY})}), and not a registered plugin name."
    )


def _parse_script_output(stdout: str) -> Dict[str, Any]:
    """Same convention as `SchedulerService._parse_script_output`."""
    for line in stdout.splitlines():
        if line.startswith("__SCHEDULER_RESULT__:"):
            json_str = line.split("__SCHEDULER_RESULT__:", 1)[1].strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                return {"parse_error": str(e), "raw": json_str[:200]}
    return {}


def run_one(spec: PluginSpec) -> PluginRunResult:
    """Run a single plugin's script as a subprocess and capture its result."""
    spec.validate()
    cmd = [sys.executable, str(spec.resolved_script_path()), *spec.script_args]
    _logger.info("Running plugin '%s': %s", spec.name, " ".join(cmd))

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=spec.timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        duration = time.monotonic() - start
        _logger.error("Plugin '%s' timed out after %ds", spec.name, spec.timeout_seconds)
        return PluginRunResult(
            name=spec.name, success=False, exit_code=None, duration_seconds=duration,
            script_result={}, error=f"timed out after {spec.timeout_seconds}s",
        )

    duration = time.monotonic() - start
    if proc.stderr:
        single_line = " | ".join(line for line in proc.stderr.splitlines() if line.strip())
        _logger.warning("Plugin '%s' stderr: %s", spec.name, single_line[-4000:])

    script_result = _parse_script_output(proc.stdout)
    success = proc.returncode == 0
    if not success:
        _logger.error("Plugin '%s' exited %d", spec.name, proc.returncode)

    return PluginRunResult(
        name=spec.name, success=success, exit_code=proc.returncode, duration_seconds=duration,
        script_result=script_result, error=None if success else f"exit code {proc.returncode}",
    )


def run_scope(scope: str) -> List[PluginRunResult]:
    """Run every plugin matching `scope`, sequentially, in registry order."""
    specs = _resolve_scope(scope)
    results: List[PluginRunResult] = []
    for spec in specs:
        results.append(run_one(spec))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scope", default="all", help="'all', a category (e.g. 'p20'), or an exact plugin name.")
    parser.add_argument("--dry-run", action="store_true", help="List what would run; execute nothing.")
    args = parser.parse_args()

    try:
        specs = _resolve_scope(args.scope)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    if args.dry_run:
        print(f"\n{len(specs)} plugin(s) in scope '{args.scope}':\n")
        for spec in specs:
            print(f"  {spec.name}  [{spec.category}]  cron={spec.cron}  script={spec.script_path}")
        print()
        return

    results = run_scope(args.scope)
    failed = [r for r in results if not r.success]

    print(f"\n{len(results)} plugin(s) run, {len(results) - len(failed)} succeeded, {len(failed)} failed.\n")
    for r in results:
        status = "OK" if r.success else "FAIL"
        print(f"  [{status}] {r.name}  ({r.duration_seconds:.1f}s)" + (f"  — {r.error}" if r.error else ""))
    print()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

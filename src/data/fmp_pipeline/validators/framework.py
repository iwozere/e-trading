"""
The validation framework (spec §11): checks, results, the shared context and the two report files.

A check is a function ``(ValidationContext) -> CheckResult`` registered under its spec ID with a default severity.
A result's ``status`` is ``PASS``, ``INFO``, ``WARN`` or ``FAIL``; only ``FAIL`` blocks gate G3. Every result carries
how many items were examined, how many violated, and at most ``MAX_EXAMPLES`` example offenders.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

from src.data.fmp_pipeline.datamanager_adapter.fmp_store import FmpStore
from src.data.fmp_pipeline.fsutil import atomic_write_bytes
from src.data.fmp_pipeline.manifest.manifest import Manifest
from src.data.fmp_pipeline.planner.config import PlanConfig
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

PASS, INFO, WARN, FAIL = "PASS", "INFO", "WARN", "FAIL"
STATUS_ORDER = {PASS: 0, INFO: 1, WARN: 2, FAIL: 3}
MAX_EXAMPLES = 20
#: Named symbols always in the cross-source sample, so runs are comparable (spec §11).
FIXED_SAMPLE = ("AAPL", "MSFT", "TSLA", "AMZN", "NVDA", "GOOGL", "JPM", "XOM", "KO", "SPY")


@dataclass
class CheckResult:
    """The outcome of one check."""

    id: str
    status: str
    examined: int = 0
    violations: int = 0
    examples: list[str] = field(default_factory=list)
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def add_example(self, text: str) -> None:
        """Record an offender (only the first ``MAX_EXAMPLES`` are kept)."""
        if len(self.examples) < MAX_EXAMPLES:
            self.examples.append(text)


@dataclass(frozen=True)
class CheckSpec:
    """A registered check."""

    id: str
    severity: str
    title: str
    func: Callable[["ValidationContext"], CheckResult]


CHECKS: dict[str, CheckSpec] = {}


def check(check_id: str, severity: str, title: str) -> Callable[[Callable[["ValidationContext"], CheckResult]],
                                                                Callable[["ValidationContext"], CheckResult]]:
    """Register a check function under its spec ID."""
    def register(func: Callable[["ValidationContext"], CheckResult]) -> Callable[["ValidationContext"], CheckResult]:
        CHECKS[check_id] = CheckSpec(check_id, severity, title, func)
        return func
    return register


class ValidationContext:
    """Everything the checks read, opened once and shared (the price scan is computed lazily, once)."""

    def __init__(self, data_root: Path, config: PlanConfig, sample_size: int = 100, seed: int = 0,
                 legacy_cache_dir: Optional[Path] = None, manifest_path: Optional[Path] = None) -> None:
        """
        Args:
            data_root: Pipeline data root (``current.txt``, ``vintages/``, ``raw/``, ``manifests/``, ``reports/``).
            config: Pipeline config; ``config.dump_id`` names the dump whose manifest and archive are checked.
            sample_size: Random symbols for the expensive cross-source checks (plus ``FIXED_SAMPLE``).
            seed: Random seed of the sample (fixed: runs are comparable).
            legacy_cache_dir: ``DATA_CACHE_DIR`` of the legacy yfinance cache (V-YF-XCHECK); ``None`` skips it.
            manifest_path: Override of ``<data_root>/manifests/<dump_id>.sqlite``.
        """
        self.data_root = data_root
        self.config = config
        self.store = FmpStore(data_root)
        self.vintage_dir = self.store.dir
        self.dump_dir = data_root / "raw" / self.store.vintage
        self.report_dir = data_root / "reports" / self.store.vintage
        self.sample_size = sample_size
        self.seed = seed
        self.legacy_cache_dir = legacy_cache_dir
        self.manifest_path = manifest_path or data_root / "manifests" / f"{self.store.vintage}.sqlite"

    @cached_property
    def scan(self) -> Any:
        """The one pass over every price file (``scan.PriceScan``)."""
        from src.data.fmp_pipeline.validators.scan import scan_prices

        return scan_prices(self.store)

    def manifest(self) -> Optional[Manifest]:
        """The dump's manifest opened read-only, or ``None`` if it does not exist."""
        if not self.manifest_path.exists():
            return None
        return Manifest(self.manifest_path, self.store.vintage, recover=False)

    @cached_property
    def sample(self) -> list[int]:
        """``security_id``s of the cross-source sample: the fixed symbols present, then a seeded random draw."""
        with_daily = [s for s in self.store.securities() if s.first_price_date is not None]
        chosen: dict[int, None] = {}
        for symbol in FIXED_SAMPLE:
            info = next((s for s in with_daily if s.symbol == symbol), None)
            if info is not None:
                chosen[info.security_id] = None
        rest = [s.security_id for s in with_daily if s.security_id not in chosen]
        random.Random(self.seed).shuffle(rest)
        for sid in rest[: max(0, self.sample_size)]:
            chosen[sid] = None
        return list(chosen)


@dataclass
class ValidationReport:
    """All results of one ``fmp validate`` run."""

    vintage: str
    results: list[CheckResult]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def failed(self) -> list[CheckResult]:
        """Results with status FAIL."""
        return [r for r in self.results if r.status == FAIL]

    def result(self, check_id: str) -> Optional[CheckResult]:
        """The result of one check, if it ran."""
        return next((r for r in self.results if r.id == check_id), None)

    def to_json(self) -> str:
        """``validation_report.json``."""
        return json.dumps({"vintage": self.vintage, "created_at": self.created_at,
                           "summary": self.summary(), "results": [asdict(r) for r in self.results]},
                          indent=1, default=str)

    def summary(self) -> dict[str, int]:
        """Count of results per status."""
        counts = {status: 0 for status in STATUS_ORDER}
        for r in self.results:
            counts[r.status] += 1
        return counts

    def to_markdown(self) -> str:
        """``validation_report.md``: summary table, then each WARN/FAIL with its examples."""
        lines = [f"# Validation report — vintage {self.vintage}", "", f"Generated {self.created_at}.", "",
                 "| Status | Count |", "|---|---|"]
        lines += [f"| {status} | {n} |" for status, n in self.summary().items()]
        lines += ["", "| ID | Status | Examined | Violations | Note |", "|---|---|---|---|---|"]
        for r in self.results:
            lines.append(f"| {r.id} | {r.status} | {r.examined:,} | {r.violations:,} | {r.message.replace('|', '/')} |")
        for r in self.results:
            if r.status in (WARN, FAIL) and (r.examples or r.details):
                lines += ["", f"## {r.id} — {r.status}", "", r.message, ""]
                lines += [f"- {e}" for e in r.examples]
                if r.details:
                    lines += ["", "```json", json.dumps(r.details, indent=1, default=str)[:4000], "```"]
        return "\n".join(lines) + "\n"

    def write(self, report_dir: Path) -> tuple[Path, Path]:
        """Write both report files; return their paths."""
        md, js = report_dir / "validation_report.md", report_dir / "validation_report.json"
        atomic_write_bytes(md, self.to_markdown().encode("utf-8"))
        atomic_write_bytes(js, self.to_json().encode("utf-8"))
        return md, js


def run_checks(context: ValidationContext, ids: Optional[Iterable[str]] = None) -> ValidationReport:
    """
    Run the registered checks (all, or ``ids``) in spec order.

    A check that raises is reported as FAIL with the exception, so one broken check never hides the others.
    """
    from src.data.fmp_pipeline.validators.checks import ORDER  # importing it registers every check

    ordered = [i for i in ORDER if i in CHECKS] + [i for i in CHECKS if i not in ORDER]
    wanted = ordered if ids is None else [i for i in ordered if i in set(ids)]
    unknown = sorted(set(ids or ()) - set(CHECKS))
    if unknown:
        raise ValueError(f"unknown check id(s): {', '.join(unknown)}")
    results = []
    for check_id in wanted:
        spec = CHECKS[check_id]
        try:
            result = spec.func(context)
        except Exception as exc:  # noqa: BLE001 - a crashing check must not hide the others
            _logger.exception("Check %s crashed", check_id)
            result = CheckResult(check_id, FAIL, message=f"check crashed: {type(exc).__name__}: {exc}")
        _logger.info("%s %s: %d examined, %d violations", check_id, result.status, result.examined, result.violations)
        results.append(result)
    return ValidationReport(context.store.vintage, results)

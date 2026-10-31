"""
P22 — spec §8.3 lookahead audit against real data, 2026-10-30.

`lookahead_audit.py` holds the pure sampling and assertion logic; this module wires it to the store.
For each sampled `(company, as_of)` it runs every registered feature, Block G and the size floor
through a `RecordingRepo`, which keeps every row a repo call returns. The checks then run on what
scoring actually read, not on what the tables happen to contain:

- every row carrying `known_from` satisfies `known_from <= as_of` (the core §8.3 assertion);
- every 13D/G position and 8-K process event read has `known_from` on its filing date;
- no unverified process event or partnership structure reached a feature (the §8.3 regression test).

Sampled `as_of` dates sit around each source row's own `known_from`: half the day before (the row
must not be visible yet) and half up to a year after.

Not covered: reads whose rows carry no `known_from` (prices returned as `{trade_date, close_raw}`,
trials, which are not bitemporal). They are counted per method in the report as unaudited reads.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.features import block_a, block_b, block_c, block_e  # noqa: F401  (registers features)
from src.ml.pipeline.p22_biotech_ma.features.block_g import build_block_g
from src.ml.pipeline.p22_biotech_ma.features.context import FeatureContext
from src.ml.pipeline.p22_biotech_ma.features.lookahead_audit import AuditSample, stratified_sample
from src.ml.pipeline.p22_biotech_ma.features.registry import get_feature, list_features
from src.ml.pipeline.p22_biotech_ma.ingest.size_floor import evaluate_size_floor
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_FILING_DATE_FIELD = {"get_verified_activist_positions": "filed_date", "get_verified_process_events": "event_date"}
_VERIFIED_METHODS = ("get_verified_process_events", "get_verified_partnership_structures")
_MAX_DAYS_AFTER = 365


class RecordingRepo:
    """Proxy that forwards every call to `repo` and keeps each returned row (a dict, or dicts in a list)."""

    def __init__(self, repo: Any) -> None:
        self._repo = repo
        self.rows: List[Dict[str, Any]] = []
        self.unaudited_calls: Dict[str, int] = {}

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._repo, name)
        if not callable(attr):
            return attr

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = attr(*args, **kwargs)
            self._record(name, result)
            return result

        return wrapper

    def _record(self, method: str, result: Any) -> None:
        items = result if isinstance(result, list) else [result]
        audited = False
        for item in items:
            if isinstance(item, dict) and "known_from" in item:
                self.rows.append({"method": method, **item})
                audited = True
        if not audited and result not in (None, [], {}):
            self.unaudited_calls[method] = self.unaudited_calls.get(method, 0) + 1


@dataclass
class AuditReport:
    """Everything the audit found; `passed` is the build gate."""

    samples: int = 0
    rows_checked: int = 0
    rows_by_method: Dict[str, int] = field(default_factory=dict)
    unaudited_calls: Dict[str, int] = field(default_factory=dict)
    samples_by_category: Dict[str, int] = field(default_factory=dict)
    lookahead_violations: List[Dict[str, Any]] = field(default_factory=list)
    filing_date_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    unverified_reads: List[Dict[str, Any]] = field(default_factory=list)
    feature_errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not (self.lookahead_violations or self.filing_date_mismatches or self.unverified_reads or self.feature_errors)


def sample_as_of_dates(population: Sequence[Dict[str, Any]], rng: random.Random) -> List[AuditSample]:
    """One `AuditSample` per population row: the day before its `known_from`, or up to a year after."""
    samples: List[AuditSample] = []
    for row in population:
        known = row["known_from"].date()
        as_of = known - timedelta(days=1) if rng.random() < 0.5 else known + timedelta(days=rng.randint(0, _MAX_DAYS_AFTER))
        samples.append(AuditSample(company_id=row["company_id"], as_of=as_of, source_category=row["category"]))
    return samples


def _known_date(row: Dict[str, Any]) -> Optional[date]:
    known = row.get("known_from")
    return known.date() if known is not None else None


def check_rows(rows: Sequence[Dict[str, Any]], sample: AuditSample, report: AuditReport) -> None:
    """Apply the three row checks to everything recorded for one sample."""
    for row in rows:
        report.rows_checked += 1
        report.rows_by_method[row["method"]] = report.rows_by_method.get(row["method"], 0) + 1
        context = {"company_id": sample.company_id, "as_of": sample.as_of, "method": row["method"]}
        known = _known_date(row)
        if known is not None and known > sample.as_of:
            report.lookahead_violations.append({**context, "known_from": row["known_from"]})
        filing_field = _FILING_DATE_FIELD.get(row["method"])
        if filing_field is not None and known is not None and known != row.get(filing_field):
            report.filing_date_mismatches.append({**context, "known_from": row["known_from"], filing_field: row.get(filing_field)})
        if row["method"] in _VERIFIED_METHODS and row.get("is_verified") is not True:
            report.unverified_reads.append(context)


def audit_sample(repo: Any, sample: AuditSample, report: AuditReport,
                 features: Optional[Dict[str, Callable[..., Any]]] = None) -> None:
    """Run every feature, Block G and the size floor for one sample through a `RecordingRepo` and check what was read."""
    recorder = RecordingRepo(repo)
    ctx = FeatureContext(as_of=sample.as_of, repo=recorder)
    feature_fns = features if features is not None else {name: get_feature(name) for name in list_features()}
    for name, fn in feature_fns.items():
        try:
            fn(sample.company_id, sample.as_of, ctx)
        except Exception as exc:  # an audit must report a crashing feature, not stop on it
            report.feature_errors.append({"company_id": sample.company_id, "as_of": sample.as_of, "feature": name, "error": repr(exc)})
    build_block_g(sample.company_id, sample.as_of, ctx)
    evaluate_size_floor(recorder, sample.company_id, sample.as_of)
    check_rows(recorder.rows, sample, report)
    for method, count in recorder.unaudited_calls.items():
        report.unaudited_calls[method] = report.unaudited_calls.get(method, 0) + count


def run_audit(repo: Any, total: int = 200, min_per_category: int = 20, seed: Optional[int] = None,
              per_category_population: int = 1000) -> AuditReport:
    """Draw the stratified sample from the store and audit it. Read-only."""
    rng = random.Random(seed)
    population = sample_as_of_dates(repo.lookahead_audit_population(per_category_population), rng)
    samples = stratified_sample(population, total=total, min_per_high_risk_category=min_per_category, rng=rng)
    report = AuditReport(samples=len(samples))
    for i, sample in enumerate(samples, 1):
        report.samples_by_category[sample.source_category] = report.samples_by_category.get(sample.source_category, 0) + 1
        audit_sample(repo, sample, report)
        if i % 50 == 0:
            _logger.info("Lookahead audit progress: %d/%d samples", i, len(samples))
    return report

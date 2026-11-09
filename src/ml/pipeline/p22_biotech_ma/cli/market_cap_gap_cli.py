"""
P22 — why backtest rows have no market cap, and how many a 10-K public float would cover (2026-11-09).

Backtest rows with no `size_band` were acquired about twice as often as the rest, because the price
history mostly lacks companies that later delisted. This reads a backtest's `scores.csv`, samples the
rows with no `size_band`, and for each one re-runs `compute_market_cap` as of that snapshot to record
the reason (`no_price`, `stale_price`, `no_shares_outstanding`, ..., or `no_cash` when the market cap
exists but enterprise value needs a cash figure). It also checks whether a `public_float` fact
(10-K cover, `dei:EntityPublicFloat`) was known then and how old its measurement date was.

    python -m src.ml.pipeline.p22_biotech_ma.cli.market_cap_gap_cli \
        --scores results/p22_biotech_ma/2026-11-09/backtest/scores.csv --sample 3000

Read-only; writes nothing.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.ml.pipeline.p22_biotech_ma.features.context import FeatureContext
from src.ml.pipeline.p22_biotech_ma.ingest.market_cap import compute_market_cap
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Public-float age buckets (days from its measurement date to the snapshot).
FLOAT_AGE_EDGES = (365, 548, 730)


@dataclass(frozen=True)
class GapRecord:
    """One sampled row: its label, why market cap / EV was missing, and the public float known then."""

    positive: int
    reason: str
    float_age_days: Optional[int]


def float_age_bucket(age_days: Optional[int]) -> str:
    """`none`, or the first `FLOAT_AGE_EDGES` bound the age falls under (`<=365d`, ...), else `>730d`."""
    if age_days is None:
        return "none"
    for edge in FLOAT_AGE_EDGES:
        if age_days <= edge:
            return f"<={edge}d"
    return f">{FLOAT_AGE_EDGES[-1]}d"


def summarize(records: Sequence[GapRecord]) -> List[Dict[str, Any]]:
    """One row per reason: count, positives, positive rate, and the count per public-float age bucket."""
    groups: Dict[str, List[GapRecord]] = defaultdict(list)
    for record in records:
        groups[record.reason].append(record)
    buckets = ["none"] + [f"<={e}d" for e in FLOAT_AGE_EDGES] + [f">{FLOAT_AGE_EDGES[-1]}d"]
    out: List[Dict[str, Any]] = []
    for reason, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        positives = sum(r.positive for r in items)
        row: Dict[str, Any] = {"reason": reason, "rows": len(items), "positives": positives,
                               "positive_rate": round(positives / len(items), 4)}
        for bucket in buckets:
            row[f"float_{bucket}"] = sum(1 for r in items if float_age_bucket(r.float_age_days) == bucket)
        out.append(row)
    return out


def _gap_record(repo: Any, company_id: int, as_of: date, positive: int) -> GapRecord:
    result, reason = compute_market_cap(repo, company_id, as_of)
    if result is not None:
        cash = FeatureContext(as_of=as_of, repo=repo).get_latest_fact(company_id, "cash_and_equivalents")
        reason = "no_cash" if not cash else "has_ev"
    floats = repo.get_financial_facts_as_of(company_id, "public_float", as_of)
    age: Optional[int] = None
    if floats and floats[0].get("value") and floats[0].get("period_end") is not None:
        age = (as_of - floats[0]["period_end"]).days
    return GapRecord(positive=positive, reason=reason or "unknown", float_age_days=age)


def main(argv: Optional[List[str]] = None) -> int:
    """Sample, classify and log the summary; returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, required=True, help="A backtest scores.csv")
    parser.add_argument("--sample", type=int, default=3000, help="Rows to sample from those with no size_band")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    with args.scores.open(encoding="utf-8") as fh:
        missing = [r for r in csv.DictReader(fh) if not r.get("f_size_band")]
    sample = random.Random(args.seed).sample(missing, min(args.sample, len(missing)))
    _logger.info("%d rows have no size_band; sampling %d", len(missing), len(sample))

    with DatabaseService().uow() as uow:
        records = [_gap_record(uow.p22, int(r["company_id"]), date.fromisoformat(r["as_of"]), int(r["positive"])) for r in sample]
    for row in summarize(records):
        _logger.info("%s", row)
    return 0


if __name__ == "__main__":
    sys.exit(main())

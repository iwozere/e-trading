"""
P21 Momentum — Data quality gates (docs/pipeline-specification.md §13).

Every job begins with validation. An ``ABORT`` halts the job, leaves state
unchanged, sends an alert, and leaves the portfolio untouched until
intervention. ``WARN`` results are collected into the report rather than
stopping the run. ``HOLD`` is the one non-boolean outcome (§GSPC/§VIX
unavailable -> retain the prior regime scalar) — modeled as a third
``GateOutcome`` value rather than a boolean bolted on as a special case.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from src.ml.pipeline.p21_momentum.config import (
    MAX_POSITION_PCT,
    MAX_POSITION_TOLERANCE_USD,
    MIN_CONSTITUENTS,
    MIN_POSITIONS_ABORT,
    MIN_POSITIONS_WARN,
    MIN_PRICE_COVERAGE_PCT,
    NAV_TOTAL_USD,
    TARGET_WEIGHT_SUM_TOLERANCE_USD,
)


class GateOutcome(str, Enum):
    """Result of a single §13 gate check."""

    PASS = "PASS"
    WARN = "WARN"
    ABORT = "ABORT"
    HOLD = "HOLD"


class PipelineAbort(Exception):
    """
    Raised when an ABORT-level §13 gate fails.

    Every job's main() catches this, alerts via
    NotificationServiceClient.send_to_admins(), and exits non-zero without
    touching results/ for that run. Carries full context for reproduction,
    per spec §13: "Every ABORT carries full context sufficient for
    reproduction."
    """

    def __init__(self, check: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.check = check
        self.context = context or {}
        super().__init__(f"[{check}] {message} | context={self.context}")


@dataclass(slots=True)
class GateResult:
    """One row of the §13 gate table, evaluated."""

    check: str
    outcome: GateOutcome
    detail: str
    context: Dict[str, Any]


def run_gates(results: List[GateResult]) -> List[GateResult]:
    """
    Raise PipelineAbort on the first ABORT-level result; otherwise return all results.

    WARN/HOLD results are returned (not raised) so the caller can include
    them in the run's report. This function does not itself evaluate any
    check — individual job modules build the GateResult list by calling the
    specific check functions below and pass the combined list here as the
    final step.

    Args:
        results: All GateResult rows evaluated so far, in the order they
            were checked (spec §13 table order).

    Returns:
        The same list, unchanged, if no ABORT is present.

    Raises:
        PipelineAbort: on the first ABORT-level result found.
    """
    for r in results:
        if r.outcome == GateOutcome.ABORT:
            raise PipelineAbort(check=r.check, message=r.detail, context=r.context)
    return results


# ---------------------------------------------------------------------------
# Individual §13 checks. Each returns a GateResult; callers assemble the
# list they pass to run_gates() in the order the job actually evaluates them
# (which — per §5 "cheap filters first" and §13's own table order — is
# usually the table's own order, but job scripts are not required to call
# every check every run: e.g. daily_mark never evaluates the target-weight
# checks, which only apply to monthly_rebalance).
# ---------------------------------------------------------------------------


def check_constituent_count(count: int, min_count: int = MIN_CONSTITUENTS) -> GateResult:
    """§13: 'Constituent list loaded' >= min_count -> ABORT on failure."""
    outcome = GateOutcome.PASS if count >= min_count else GateOutcome.ABORT
    return GateResult(
        check="Constituent list loaded",
        outcome=outcome,
        detail=f"{count} constituents loaded (minimum {min_count})",
        context={"count": count, "min_count": min_count},
    )


def check_price_coverage(
    non_empty_count: int, requested_count: int, min_pct: float = MIN_PRICE_COVERAGE_PCT
) -> GateResult:
    """§13: 'Tickers with complete price data' >= min_pct of universe -> ABORT on failure."""
    coverage = non_empty_count / requested_count if requested_count else 0.0
    outcome = GateOutcome.PASS if coverage >= min_pct else GateOutcome.ABORT
    return GateResult(
        check="Tickers with complete price data",
        outcome=outcome,
        detail=f"Coverage {coverage:.1%} (minimum {min_pct:.1%})",
        context={"non_empty_count": non_empty_count, "requested_count": requested_count},
    )


def check_regime_inputs_available(gspc_available: bool, vix_available: bool) -> GateResult:
    """§13: '^GSPC and ^VIX available' -> HOLD (retain prior scalar), not ABORT."""
    both_available = gspc_available and vix_available
    outcome = GateOutcome.PASS if both_available else GateOutcome.HOLD
    return GateResult(
        check="^GSPC and ^VIX available",
        outcome=outcome,
        detail="Both available" if both_available else "Retaining prior regime scalar",
        context={"gspc_available": gspc_available, "vix_available": vix_available},
    )


def check_signal_date_is_trading_day(is_trading_day: bool) -> GateResult:
    """§13: 'Signal date is a trading day' -> ABORT on failure."""
    outcome = GateOutcome.PASS if is_trading_day else GateOutcome.ABORT
    return GateResult(
        check="Signal date is a trading day",
        outcome=outcome,
        detail="Signal date confirmed as an NYSE trading day" if is_trading_day else "Signal date is not a trading day",
        context={},
    )


def check_target_weight_sum(
    target_weight_sum_usd: float,
    sleeve_usd: float,
    tolerance_usd: float = TARGET_WEIGHT_SUM_TOLERANCE_USD,
) -> GateResult:
    """§13: 'Sum of target weights' == sleeve_usd +/- $1 -> ABORT on failure."""
    diff = abs(target_weight_sum_usd - sleeve_usd)
    outcome = GateOutcome.PASS if diff <= tolerance_usd else GateOutcome.ABORT
    return GateResult(
        check="Sum of target weights",
        outcome=outcome,
        detail=f"Sum={target_weight_sum_usd:.2f} vs sleeve_usd={sleeve_usd:.2f} (tolerance ${tolerance_usd})",
        context={"target_weight_sum_usd": target_weight_sum_usd, "sleeve_usd": sleeve_usd},
    )


def check_no_weight_exceeds_cap(
    max_weight_usd: float,
    nav_total: float = NAV_TOTAL_USD,
    max_position_pct: float = MAX_POSITION_PCT,
    tolerance_usd: float = MAX_POSITION_TOLERANCE_USD,
) -> GateResult:
    """§13: 'No weight exceeds cap' <= cap_usd + $1 -> ABORT on failure."""
    cap_usd = nav_total * max_position_pct
    outcome = GateOutcome.PASS if max_weight_usd <= cap_usd + tolerance_usd else GateOutcome.ABORT
    return GateResult(
        check="No weight exceeds cap",
        outcome=outcome,
        detail=f"Max weight ${max_weight_usd:.2f} vs cap ${cap_usd:.2f} (tolerance ${tolerance_usd})",
        context={"max_weight_usd": max_weight_usd, "cap_usd": cap_usd},
    )


def check_position_count(
    count: int,
    min_abort: int = MIN_POSITIONS_ABORT,
    min_warn: int = MIN_POSITIONS_WARN,
) -> GateResult:
    """§13: 'Positions in target portfolio' 8-20; <8 -> ABORT, 8-19 -> WARN."""
    if count < min_abort:
        outcome = GateOutcome.ABORT
    elif count < min_warn:
        outcome = GateOutcome.WARN
    else:
        outcome = GateOutcome.PASS
    return GateResult(
        check="Positions in target portfolio",
        outcome=outcome,
        detail=f"{count} positions (abort below {min_abort}, warn below {min_warn})",
        context={"count": count},
    )


def check_cash_after_execution(cash: float) -> GateResult:
    """§13: 'Cash after execution' >= 0 -> ABORT on failure."""
    outcome = GateOutcome.PASS if cash >= 0 else GateOutcome.ABORT
    return GateResult(
        check="Cash after execution",
        outcome=outcome,
        detail=f"Cash={cash:.2f}",
        context={"cash": cash},
    )


def check_daily_price_change(ticker: str, change_pct: float, threshold_pct: float) -> GateResult:
    """
    §13: 'Daily price change' > threshold -> exclude ticker, flag (not ABORT).

    Per §10.1/§11: because the series is pre-adjusted, a jump this size is
    never a real split masquerading as an anomaly — every trigger here is a
    genuine price event or a data error.
    """
    breach = abs(change_pct) > threshold_pct
    outcome = GateOutcome.WARN if breach else GateOutcome.PASS
    return GateResult(
        check="Daily price change",
        outcome=outcome,
        detail=f"{ticker}: {change_pct:.1%} change" + (" — excluded" if breach else ""),
        context={"ticker": ticker, "change_pct": change_pct},
    )

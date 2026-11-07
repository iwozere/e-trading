"""
Signal Ledger — exit policies the paper books are simulated under.

Every stream's signals are replayed once per policy, so exit rules are compared on
identical entries. A policy combines a time cap with optional price rules:

- ``horizon``   — the stream's own holding period (``horizon_days``; the book's original rule);
- ``fixed_N``   — sell at the close N sessions after entry;
- ``bracket``   — fixed stop-loss and profit target, with a time cap;
- ``trail_atr`` — trailing stop ``TRAIL_ATR_MULT`` x ATR below the highest close, with a time cap;
- ``native``    — the pipeline's own exit: P22 tier drop / P24 paper exit (``meta["exit_signal_date"]``),
                  or for streams that republish their picks, "no longer signalled" (``DECAY_SESSIONS``).

``native`` only exists for streams that have such an exit (see ``policies_for``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.ml.pipeline.signal_ledger.config import (
    BRACKET_MAX_HOLD,
    BRACKET_STOP_PCT,
    BRACKET_TARGET_PCT,
    DECAY_SESSIONS,
    MAX_POLICY_HOLD,
    META_EXIT_SOURCES,
    TRAIL_ATR_MULT,
)

# Why a paper trade ended.
EXIT_TIME = "time"
EXIT_STOP = "stop"
EXIT_TARGET = "target"
EXIT_TRAIL = "trail"
EXIT_NATIVE = "native"
EXIT_DECAY = "decay"
EXIT_DATA = "data"  # suspected bad price (unadjusted split): closed at the last good close
EXIT_OPEN = "open"  # still held at the last session


@dataclass(frozen=True)
class ExitPolicy:
    """
    One exit rule set.

    Attributes:
        name: Policy id stored with NAV and trade rows.
        max_hold: Time cap in sessions; None = the signal's ``horizon_days``.
        stop_pct: Stop-loss distance below the entry price (0.15 = -15 %).
        target_pct: Profit target above the entry price (0.30 = +30 %).
        trail_atr: Trailing stop distance in ATRs below the highest close.
        native: Use the pipeline's own exit (meta exit date or signal decay).
        description: Human-readable summary for the API/UI.
    """

    name: str
    max_hold: Optional[int] = None
    stop_pct: Optional[float] = None
    target_pct: Optional[float] = None
    trail_atr: Optional[float] = None
    native: bool = False
    description: str = ""


DEFAULT_POLICY = ExitPolicy("horizon", description="Stream default holding period")

POLICIES: Tuple[ExitPolicy, ...] = (
    DEFAULT_POLICY,
    *(ExitPolicy(f"fixed_{n}", max_hold=n, description=f"Sell at the close {n} sessions after entry") for n in (20, 60, 120, 250)),
    ExitPolicy(
        "bracket", max_hold=BRACKET_MAX_HOLD, stop_pct=BRACKET_STOP_PCT, target_pct=BRACKET_TARGET_PCT,
        description=f"Stop -{BRACKET_STOP_PCT:.0%}, target +{BRACKET_TARGET_PCT:.0%}, max {BRACKET_MAX_HOLD} sessions",
    ),
    ExitPolicy(
        "trail_atr", max_hold=MAX_POLICY_HOLD, trail_atr=TRAIL_ATR_MULT,
        description=f"Trailing stop {TRAIL_ATR_MULT:g} x ATR below the highest close, max {MAX_POLICY_HOLD} sessions",
    ),
    ExitPolicy("native", max_hold=MAX_POLICY_HOLD, native=True,
               description=f"Pipeline's own exit (tier drop, paper exit or no longer signalled), max {MAX_POLICY_HOLD} sessions"),
)


def has_native_exit(source: str) -> bool:
    """True if the stream has a pipeline-native exit (a meta exit date or a decay window)."""
    return source in META_EXIT_SOURCES or source in DECAY_SESSIONS


def policies_for(source: str) -> List[ExitPolicy]:
    """
    Return the policies a stream's book is simulated under.

    Args:
        source: Signal source.

    Returns:
        All generic policies, plus ``native`` when the stream has a native exit.
    """
    return [p for p in POLICIES if not p.native or has_native_exit(source)]


def policy_by_name(name: str) -> Optional[ExitPolicy]:
    """Look up a policy by name (None if unknown)."""
    return next((p for p in POLICIES if p.name == name), None)

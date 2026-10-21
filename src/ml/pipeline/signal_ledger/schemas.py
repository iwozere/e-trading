"""Signal Ledger — record types shared by harvesters, evaluator and repository."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SignalRecord:
    """
    One buy (or sell) recommendation emitted by a pipeline.

    Attributes:
        pipeline: Short pipeline id (``"p05"``).
        source: Signal stream within the pipeline (``"p05_top_picks"``); unique key part.
        signal_date: Date the pipeline labels its output with (results folder date).
            Entry is assumed at the open of the first session strictly after it.
        ticker: Upper-case symbol.
        side: ``"long"`` or ``"short"`` — what the pipeline recommends, not what we think of it.
        rank: 1-based rank within the day, if the pipeline ranks (lower = better).
        score: Pipeline score / confidence, if any.
        horizon_days: Holding period the paper book uses for this source.
        meta: Small extra fields (tier, confidence, alert price...).
    """

    pipeline: str
    source: str
    signal_date: date
    ticker: str
    side: str = "long"
    rank: Optional[int] = None
    score: Optional[float] = None
    horizon_days: int = 20
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        """Return the record as a ``signal_ledger`` insert row."""
        return {
            "pipeline": self.pipeline,
            "source": self.source,
            "signal_date": self.signal_date,
            "ticker": self.ticker,
            "side": self.side,
            "rank": self.rank,
            "score": self.score,
            "horizon_days": self.horizon_days,
            "meta": self.meta or None,
        }

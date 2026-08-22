"""
P21 Momentum — Data contracts.

One dataclass per ``docs/pipeline-specification.md`` §3 JSON shape. Plain
dataclasses, not pydantic — this repo's convention reserves pydantic for
schemas crossing an API boundary; file-I/O-only models use dataclasses plus
``dataclasses.asdict()`` (see ``project_pyright_cleanup`` era convention).

``LedgerEntry.reason`` is a ``Literal`` of the nine permitted values from
spec §3 — note ``CORPORATE_ACTION_SPLIT`` is deliberately absent (§10.1: an
adjusted-close-only series never produces a split discontinuity to log).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

LedgerReason = Literal[
    "ENTRY_RANK_n",
    "EXIT_RANK_DROP",
    "EXIT_SECTOR_CAP",
    "EXIT_FILTER_FAIL",
    "EXIT_DELISTED",
    "EXIT_CATASTROPHIC_STOP",
    "REBAL_TRIM",
    "REBAL_ADD",
    "REGIME_SCALE_DOWN",
    "REGIME_SCALE_UP",
]
# "ENTRY_RANK_n" above is a placeholder for the pattern ENTRY_RANK_<int>;
# callers should format the actual reason string as f"ENTRY_RANK_{rank}"
# and Literal membership is not statically enforced for that one value.


@dataclass(slots=True)
class SignalRow:
    """One row of ``signals.json`` — the full ranked signal table."""

    ticker: str
    raw_return: float
    vol: float
    signal: float
    rank: int
    sector: Optional[str] = None
    filters_passed: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SignalRow":
        return cls(**d)


@dataclass(slots=True)
class TargetPosition:
    """One row of ``targets.json`` — the pre-execution target portfolio."""

    ticker: str
    target_weight_pct: float
    target_usd: float
    rank: int
    sector: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TargetPosition":
        return cls(**d)


@dataclass(slots=True)
class Position:
    """One holding — backs ``_state/current_positions.json`` and ``positions.json``."""

    ticker: str
    shares: float
    avg_cost: float
    entry_date: str
    entry_rank: int
    current_rank: Optional[int]
    sector: Optional[str]
    target_weight_pct: float
    high_water_price: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Position":
        return cls(**d)


@dataclass(slots=True)
class LedgerEntry:
    """One line of ``_state/ledger.jsonl`` — append-only, one per simulated trade, ever."""

    ts: str
    track: str
    ticker: str
    side: Literal["BUY", "SELL"]
    shares: float
    ref_open: float
    fill_price: float
    slippage_bps: float
    commission_usd: float
    gross_usd: float
    net_usd: float
    reason: str  # see LedgerReason for the permitted-value catalogue

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LedgerEntry":
        return cls(**d)


@dataclass(slots=True)
class RegimeState:
    """One entry of ``_state/regime_history.json`` — appended once per month."""

    date: str
    spx_12m_return: float
    spx_vs_200dma: float
    vix_20d_avg: float
    bear: bool
    high_vol: bool
    scalar_raw: float
    scalar_applied: float
    months_at_state: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegimeState":
        return cls(**d)


@dataclass(slots=True)
class DailyMarkSnapshot:
    """``daily_mark.json`` — today's NAV/high-water/anomaly snapshot."""

    as_of: str
    nav: Dict[str, float]  # track letter -> NAV
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    catastrophic_stops: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DailyMarkSnapshot":
        return cls(**d)


@dataclass(slots=True)
class NavRow:
    """One row of ``_state/nav_daily.csv`` — append-only, daily NAV for all 5 tracks."""

    date: str
    nav_a: float
    nav_b: float
    nav_c: float
    nav_d: float
    nav_e: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NavRow":
        return cls(**d)

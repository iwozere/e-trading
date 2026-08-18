"""P19 StructuralProfile model (spec v2 §4.0)."""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional


@dataclass
class StructuralProfile:
    """
    Layer 0 cap-structure integrity profile for one ticker (spec v2 §4.0).

    Computed pre-market from EDGAR filings (never intraday — decision #5), cached
    per ticker, and denormalised as a point-in-time snapshot onto every shadow row
    (spec §12.1) so a later join against a mutable profile table can never leak
    future information into calibration.

    Every field beyond ``ticker``/``cik``/``as_of``/``grade``/``coverage`` is
    nullable by design — an unresolved signal must never be silently treated as
    clean (StructuralSignals.md §1 rule 2).
    """

    ticker: str
    cik: Optional[str] = None
    as_of: Optional[date] = None
    grade: str = "C"  # A / B / C / D — unknown defaults to C, never A (N17)

    dilution_urgency: float = 0.0  # 0–100
    insider_conviction: float = 0.0  # 0–100

    # ── Component evidence (nullable — coverage tracked explicitly) ─────────
    reverse_splits_24m: int = 0
    reverse_split_ever: bool = False
    share_count_cagr_8q: Optional[float] = None
    shares_outstanding: Optional[float] = None
    cash: Optional[float] = None
    quarterly_burn: Optional[float] = None
    runway_quarters: Optional[float] = None
    shelf_active: Optional[bool] = None
    shelf_capacity_pct_mcap: Optional[float] = None
    days_since_last_offering: Optional[int] = None
    atm_used_last_2q: Optional[bool] = None
    floating_convert_flag: Optional[bool] = None  # N5 — EFTS phrase match, latest annual+interim
    going_concern_flag: Optional[bool] = None  # N6 — EFTS phrase match, latest annual
    exchange_deficiency_flag: Optional[bool] = None
    warrant_overhang_pct_float: Optional[float] = None  # N12 — still deferred, no safe fallback (see config.py)
    insider_buys_90d: int = 0
    distinct_insider_buyers_90d: int = 0
    insider_sells_90d: int = 0
    inst_holders_delta_2q: Optional[int] = None  # true magnitude-aware P8 — still deferred, see inst_13dg_activity_2q
    inst_13dg_activity_2q: Optional[bool] = None  # P8 proxy — any 13D/G filed against this CIK, trailing 2q
    recent_ipo_months: Optional[int] = None
    is_fpi: Optional[bool] = None  # N15 + StructuralSignals.md §2 — track FPIs separately in calibration
    auditor_name: Optional[str] = None  # N16 — extracted from the EX-23.1 consent exhibit
    auditor_whitelisted: Optional[bool] = None  # N16
    no_debt_maturity_24m: Optional[bool] = None  # P9
    short_interest_pct_float: Optional[float] = None  # P11 evidence (yfinance)
    days_to_cover: Optional[float] = None  # P11 evidence (yfinance)

    coverage: float = 0.0  # fraction of Tier 1+2 fields resolved (Phase 1.5 scope)
    disqualifiers: List[str] = field(default_factory=list)  # human-readable, surfaced in alert copy
    # signal id -> "C" | "D", the grading input the free-text disqualifiers list can't
    # be safely re-derived from (design-v2.md §2)
    disqualifier_severities: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Flat dict suitable for JSON caching / shadow-row denormalisation."""
        return {
            "ticker": self.ticker,
            "cik": self.cik,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "grade": self.grade,
            "dilution_urgency": round(self.dilution_urgency, 2),
            "insider_conviction": round(self.insider_conviction, 2),
            "reverse_splits_24m": self.reverse_splits_24m,
            "reverse_split_ever": self.reverse_split_ever,
            "share_count_cagr_8q": self.share_count_cagr_8q,
            "shares_outstanding": self.shares_outstanding,
            "cash": self.cash,
            "quarterly_burn": self.quarterly_burn,
            "runway_quarters": self.runway_quarters,
            "shelf_active": self.shelf_active,
            "shelf_capacity_pct_mcap": self.shelf_capacity_pct_mcap,
            "days_since_last_offering": self.days_since_last_offering,
            "atm_used_last_2q": self.atm_used_last_2q,
            "floating_convert_flag": self.floating_convert_flag,
            "going_concern_flag": self.going_concern_flag,
            "exchange_deficiency_flag": self.exchange_deficiency_flag,
            "warrant_overhang_pct_float": self.warrant_overhang_pct_float,
            "insider_buys_90d": self.insider_buys_90d,
            "distinct_insider_buyers_90d": self.distinct_insider_buyers_90d,
            "insider_sells_90d": self.insider_sells_90d,
            "inst_holders_delta_2q": self.inst_holders_delta_2q,
            "inst_13dg_activity_2q": self.inst_13dg_activity_2q,
            "recent_ipo_months": self.recent_ipo_months,
            "is_fpi": self.is_fpi,
            "auditor_name": self.auditor_name,
            "auditor_whitelisted": self.auditor_whitelisted,
            "no_debt_maturity_24m": self.no_debt_maturity_24m,
            "short_interest_pct_float": self.short_interest_pct_float,
            "days_to_cover": self.days_to_cover,
            "coverage": round(self.coverage, 3),
            "disqualifiers": list(self.disqualifiers),
            "disqualifier_severities": dict(self.disqualifier_severities),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "StructuralProfile":
        """Rehydrate from a cached dict (mirror of to_dict); tolerant of missing keys."""
        as_of_str = d.get("as_of")
        return cls(
            ticker=d["ticker"],
            cik=d.get("cik"),
            as_of=date.fromisoformat(as_of_str) if as_of_str else None,
            grade=d.get("grade", "C"),
            dilution_urgency=float(d.get("dilution_urgency", 0.0) or 0.0),
            insider_conviction=float(d.get("insider_conviction", 0.0) or 0.0),
            reverse_splits_24m=int(d.get("reverse_splits_24m", 0) or 0),
            reverse_split_ever=bool(d.get("reverse_split_ever", False)),
            share_count_cagr_8q=d.get("share_count_cagr_8q"),
            shares_outstanding=d.get("shares_outstanding"),
            cash=d.get("cash"),
            quarterly_burn=d.get("quarterly_burn"),
            runway_quarters=d.get("runway_quarters"),
            shelf_active=d.get("shelf_active"),
            shelf_capacity_pct_mcap=d.get("shelf_capacity_pct_mcap"),
            days_since_last_offering=d.get("days_since_last_offering"),
            atm_used_last_2q=d.get("atm_used_last_2q"),
            floating_convert_flag=d.get("floating_convert_flag"),
            going_concern_flag=d.get("going_concern_flag"),
            exchange_deficiency_flag=d.get("exchange_deficiency_flag"),
            warrant_overhang_pct_float=d.get("warrant_overhang_pct_float"),
            insider_buys_90d=int(d.get("insider_buys_90d", 0) or 0),
            distinct_insider_buyers_90d=int(d.get("distinct_insider_buyers_90d", 0) or 0),
            insider_sells_90d=int(d.get("insider_sells_90d", 0) or 0),
            inst_holders_delta_2q=d.get("inst_holders_delta_2q"),
            inst_13dg_activity_2q=d.get("inst_13dg_activity_2q"),
            recent_ipo_months=d.get("recent_ipo_months"),
            is_fpi=d.get("is_fpi"),
            auditor_name=d.get("auditor_name"),
            auditor_whitelisted=d.get("auditor_whitelisted"),
            no_debt_maturity_24m=d.get("no_debt_maturity_24m"),
            short_interest_pct_float=d.get("short_interest_pct_float"),
            days_to_cover=d.get("days_to_cover"),
            coverage=float(d.get("coverage", 0.0) or 0.0),
            disqualifiers=list(d.get("disqualifiers", []) or []),
            disqualifier_severities=dict(d.get("disqualifier_severities", {}) or {}),
        )

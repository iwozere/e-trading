"""
P19 Layer 0 — grading (spec v2 §7, §7.5, StructuralSignals.md).

Pure evaluation logic: given already-fetched raw data for one ticker (XBRL
company facts, recent filings, Form 4 transactions, split history), produces a
``StructuralProfile`` with grade, ``dilution_urgency``, ``insider_conviction``,
disqualifiers, and per-signal coverage. No network calls here — that's
``profiler.py``'s job.

**Scope**: Phase 1.5 added N1,N2,N3,N4,N7,N8,N9(sub-$75M float only),N10,N11,
N13,N14 / P1,P2,P3,P4,P5,P6,P7. Phase 3 added N5,N6,N15,N16 / P8,P9,P11.
**Still not evaluated**: N9 above $75M float, N12, P10 — their profile fields
stay unresolved (``None``) and correctly depress ``coverage`` rather than
being silently assumed clean (StructuralSignals.md §1 rule 2 / N17). See
config.py's ``P19StructuralConfig`` docstring for why each is still deferred.

**Rule that governs everything below** (StructuralSignals.md §1 rule 1): no
positive signal ever overrides a D disqualifier. Grade assignment checks D
first, then C, and only then considers ``insider_conviction``/``dilution_urgency``.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.ml.pipeline.p19_penny_intraday.config import P19StructuralConfig
from src.ml.pipeline.p19_penny_intraday.models.structural_profile import StructuralProfile
from src.ml.pipeline.p19_penny_intraday.structural import xbrl_facts

# Offering-document form types (N8) — the takedown/marketed-deal documents
# themselves, distinct from S-3 (the shelf *registration*, used for N9/shelf_active).
_OFFERING_FORMS = frozenset({"424B5", "424B4", "424B3", "424B2", "424B1", "S-1", "S-1/A"})
_SHELF_REGISTRATION_FORMS = frozenset({"S-3", "S-3ASR"})
_SHELF_VALIDITY_DAYS = 3 * 365  # S-3 shelf registrations are typically valid ~3 years

_BUY_CODE = "P"
_SALE_CODES = frozenset({"S", "S-"})


@dataclass
class GradingInputs:
    """Raw, already-fetched data for one ticker — everything grading needs."""

    ticker: str
    cik: Optional[str]
    as_of: date
    company_facts: Optional[Dict[str, Any]]  # None if EDGAR companyfacts unresolved
    filings: Optional[List[Dict[str, Any]]]  # None if submissions unresolved; [] is a valid "none found"
    splits: Optional[List[Tuple[date, float]]]  # None if the yfinance fetch failed
    form4_rows: Optional[List[Dict[str, Any]]]  # None if the daily cache read failed
    market_cap: Optional[float] = None
    float_shares: Optional[float] = None
    prior_close: Optional[float] = None

    # ── Phase 3 additions — all None means "not fetched / EFTS unavailable",
    # which correctly depresses coverage rather than being read as "no match" ──
    floating_convert_hit: Optional[bool] = None  # N5 — EFTS phrase match, latest annual+interim
    going_concern_hit: Optional[bool] = None  # N6 — EFTS phrase match, latest annual, affirmative form
    auditor_name: Optional[str] = None  # N16 — extracted from EX-23.1 consent exhibit
    listing_date: Optional[date] = None  # N15 — earliest filing date, IPO-recency proxy
    is_fpi: Optional[bool] = None  # N15 — 20-F/6-K filer with no 10-K/10-Q
    dg_activity_2q: Optional[bool] = None  # P8 — any 13D/G filed against this CIK, trailing 2 quarters
    debt_maturity_near_term: Optional[bool] = None  # P9 raw signal — True = near-term maturity tagged
    short_interest_pct_float: Optional[float] = None  # P11
    days_to_cover: Optional[float] = None  # P11


@dataclass
class _SignalResult:
    resolved: bool
    fires: bool = False
    severity: Optional[str] = None  # "C" | "D"
    label: str = ""  # human-readable disqualifier text, if fires


def _months_ago(as_of: date, months: int) -> date:
    # Approximate (30-day months) — fine at this granularity, avoids a calendar dep.
    return as_of - timedelta(days=months * 30)


def _filings_by_form(filings: List[Dict[str, Any]], forms: frozenset) -> List[Tuple[date, Dict[str, Any]]]:
    out = []
    for f in filings:
        form = str(f.get("form") or "").upper().strip()
        if form not in forms:
            continue
        d = _parse_filing_date(f.get("filingDate", ""))
        if d is not None:
            out.append((d, f))
    return sorted(out)


def _parse_filing_date(s: str) -> Optional[date]:
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


def _has_item(filing: Dict[str, Any], item: str) -> bool:
    items = str(filing.get("items") or "")
    return item in [i.strip() for i in items.split(",")]


# ── N1/N2 — reverse splits ───────────────────────────────────────────────────


def _eval_reverse_splits(inputs: GradingInputs, cfg: P19StructuralConfig) -> Tuple[_SignalResult, int, bool]:
    if inputs.splits is None:
        return _SignalResult(resolved=False), 0, False
    reverse = [(d, r) for d, r in inputs.splits if r < 1.0]
    ever = len(reverse) > 0
    window_start = _months_ago(inputs.as_of, cfg.reverse_split_window_months)
    count_recent = sum(1 for d, _ in reverse if d >= window_start)
    if count_recent > 0:
        return (
            _SignalResult(True, fires=True, severity="D", label=f"reverse split within {cfg.reverse_split_window_months}mo (N1)"),
            count_recent,
            ever,
        )
    if ever:
        return _SignalResult(True, fires=True, severity="C", label="reverse split in listed history (N2)"), 0, ever
    return _SignalResult(True, fires=False), 0, ever


# ── N3/N4/P3 — share-count CAGR (split-adjusted) ─────────────────────────────


def _eval_share_cagr(inputs: GradingInputs, cfg: P19StructuralConfig) -> Tuple[_SignalResult, Optional[float]]:
    if inputs.company_facts is None or inputs.splits is None:
        return _SignalResult(resolved=False), None
    raw_series = xbrl_facts.shares_outstanding_series(inputs.company_facts)
    if not raw_series:
        return _SignalResult(resolved=False), None
    adjusted = xbrl_facts.split_adjust(raw_series, inputs.splits)
    cagr_8q = xbrl_facts.cagr(adjusted, lookback_quarters=8)
    if cagr_8q is None:
        return _SignalResult(resolved=False), None
    if cagr_8q > cfg.share_cagr_d_threshold:
        return _SignalResult(True, fires=True, severity="D", label=f"share count CAGR {cagr_8q:.0%} > {cfg.share_cagr_d_threshold:.0%} (N3)"), cagr_8q
    if cagr_8q > cfg.share_cagr_c_threshold:
        return _SignalResult(True, fires=True, severity="C", label=f"share count CAGR {cagr_8q:.0%} (N4)"), cagr_8q
    return _SignalResult(True, fires=False), cagr_8q


# ── N7 — exchange deficiency notice (8-K item 3.01) ──────────────────────────


def _eval_deficiency_notice(inputs: GradingInputs, cfg: P19StructuralConfig) -> _SignalResult:
    if inputs.filings is None:
        return _SignalResult(resolved=False)
    window_start = inputs.as_of - timedelta(days=cfg.deficiency_notice_lookback_days)
    hits = [
        (d, f)
        for f in inputs.filings
        if str(f.get("form") or "").upper().strip() == "8-K" and _has_item(f, "3.01")
        for d in [_parse_filing_date(f.get("filingDate", ""))]
        if d is not None and d >= window_start
    ]
    # NB: no compliance-regained detection (StructuralSignals.md N7 failure mode
    # — those disclosures are inconsistently filed). A stale-open notice near a
    # much-higher current price is a known false positive, deferred to Phase 3.
    if hits:
        return _SignalResult(True, fires=True, severity="D", label="active exchange deficiency notice, 8-K item 3.01 (N7)")
    return _SignalResult(True, fires=False)


# ── N8 — offering document filed recently ────────────────────────────────────


def _last_offering_date(filings: List[Dict[str, Any]]) -> Optional[date]:
    hits = _filings_by_form(filings, _OFFERING_FORMS)
    return hits[-1][0] if hits else None


def _eval_offering_recency(inputs: GradingInputs, cfg: P19StructuralConfig) -> Tuple[_SignalResult, Optional[int]]:
    if inputs.filings is None:
        return _SignalResult(resolved=False), None
    last = _last_offering_date(inputs.filings)
    if last is None:
        return _SignalResult(True, fires=False), None
    days_since = (inputs.as_of - last).days
    if days_since <= cfg.offering_window_d_days:
        return _SignalResult(True, fires=True, severity="D", label=f"offering document filed {days_since}d ago (N8)"), days_since
    if days_since <= cfg.offering_window_c_days:
        return _SignalResult(True, fires=True, severity="C", label=f"offering document filed {days_since}d ago (N8)"), days_since
    return _SignalResult(True, fires=False), days_since


# ── N9 — baby-shelf remaining capacity (§5, sub-$75M float only) ────────────


def _shelf_active(filings: List[Dict[str, Any]], as_of: date) -> bool:
    hits = _filings_by_form(filings, _SHELF_REGISTRATION_FORMS)
    if not hits:
        return False
    return (as_of - hits[-1][0]).days <= _SHELF_VALIDITY_DAYS


def _eval_shelf_capacity(
    inputs: GradingInputs,
    cfg: P19StructuralConfig,
    proceeds_trailing_12m: Optional[float],
) -> Tuple[_SignalResult, Optional[bool], Optional[float]]:
    if inputs.filings is None:
        return _SignalResult(resolved=False), None, None
    shelf_active = _shelf_active(inputs.filings, inputs.as_of)
    float_value = None
    if inputs.float_shares and inputs.prior_close:
        float_value = inputs.float_shares * inputs.prior_close
    if not shelf_active:
        return _SignalResult(True, fires=False), False, None
    if float_value is None or float_value >= cfg.baby_shelf_float_threshold_usd:
        # Above the baby-shelf threshold, exact capacity needs prospectus text
        # parsing (Phase 3, StructuralSignals.md §3 N9) — leave unresolved.
        return _SignalResult(resolved=False), True, None
    annual_capacity = float_value / 3.0
    used = proceeds_trailing_12m or 0.0
    remaining = max(0.0, annual_capacity - used)
    market_cap = inputs.market_cap or float_value
    capacity_pct = (remaining / market_cap) if market_cap > 0 else None
    if capacity_pct is not None and capacity_pct > 0.30:
        return (
            _SignalResult(True, fires=True, severity="C", label=f"active S-3 shelf, remaining capacity {capacity_pct:.0%} of mcap (N9)"),
            True,
            capacity_pct,
        )
    return _SignalResult(True, fires=False), True, capacity_pct


# ── N10 — ATM usage in either of the last 2 quarters ─────────────────────────


def _eval_atm_usage(inputs: GradingInputs, cfg: P19StructuralConfig) -> Tuple[_SignalResult, Optional[bool]]:
    if inputs.company_facts is None or inputs.filings is None:
        return _SignalResult(resolved=False), None
    proceeds = xbrl_facts.proceeds_from_issuance(inputs.company_facts)
    if not proceeds:
        return _SignalResult(True, fires=False), False
    market_cap = inputs.market_cap or 0.0
    threshold = market_cap * cfg.atm_proceeds_pct_mcap_threshold if market_cap > 0 else 0.0
    offering_dates = {d for d, _ in _filings_by_form(inputs.filings, _OFFERING_FORMS)}
    recent = proceeds[-cfg.atm_lookback_quarters :]
    for end, value in recent:
        if value <= threshold:
            continue
        # A discrete marketed offering in the same ~quarter is a different
        # animal from dribble-out ATM sales (StructuralSignals.md N10 note).
        quarter_has_marketed_deal = any(abs((end - od).days) <= 100 for od in offering_dates)
        if not quarter_has_marketed_deal:
            return _SignalResult(True, fires=True, severity="C", label="ATM used in a recent quarter (N10)"), True
    return _SignalResult(True, fires=False), False


# ── N11/P5/P6 — cash runway / operating cash flow ────────────────────────────


def _eval_runway(
    inputs: GradingInputs, cfg: P19StructuralConfig, shelf_active: Optional[bool]
) -> Tuple[_SignalResult, Optional[float], Optional[float]]:
    if inputs.company_facts is None:
        return _SignalResult(resolved=False), None, None
    cash, _, runway = xbrl_facts.cash_and_burn(inputs.company_facts)
    if cash is None:
        return _SignalResult(resolved=False), None, None
    if runway is None:
        # Positive OCF (or no burn data) — not a runway disqualifier; P5/P6 cover this.
        return _SignalResult(True, fires=False), cash, None
    if runway < cfg.runway_d_threshold_quarters and shelf_active:
        return (
            _SignalResult(True, fires=True, severity="D", label=f"cash runway {runway:.1f}q with an active shelf (N11)"),
            cash,
            runway,
        )
    if runway < cfg.runway_c_threshold_quarters:
        return _SignalResult(True, fires=True, severity="C", label=f"cash runway {runway:.1f} quarters (N11)"), cash, runway
    return _SignalResult(True, fires=False), cash, runway


def _eval_positive_ocf(inputs: GradingInputs) -> _SignalResult:
    """P5 — positive operating cash flow, latest discrete quarter."""
    if inputs.company_facts is None:
        return _SignalResult(resolved=False)
    ocf = xbrl_facts.operating_cash_flow_quarterly(inputs.company_facts)
    if not ocf:
        return _SignalResult(resolved=False)
    return _SignalResult(True, fires=ocf[-1][1] > 0)


def _eval_long_runway_or_net_cash(inputs: GradingInputs, cfg: P19StructuralConfig, runway: Optional[float]) -> _SignalResult:
    """
    P6 — runway > 8 quarters or net cash positive. True total-debt data is out
    of Phase 1.5 scope, so "net cash positive" is approximated by positive
    operating cash flow (a strictly narrower, more conservative reading than
    the spec's cash-vs-total-debt test — documented simplification).
    """
    if inputs.company_facts is None:
        return _SignalResult(resolved=False)
    if runway is not None and runway > cfg.runway_normalize_quarters:
        return _SignalResult(True, fires=True)
    ocf_result = _eval_positive_ocf(inputs)
    if not ocf_result.resolved:
        return _SignalResult(resolved=runway is not None)
    return _SignalResult(True, fires=ocf_result.fires)


# ── N13 — 8-K item 3.02 (unregistered equity sale) ──────────────────────────


def _eval_unregistered_sale(inputs: GradingInputs, cfg: P19StructuralConfig) -> _SignalResult:
    if inputs.filings is None:
        return _SignalResult(resolved=False)
    window_start = inputs.as_of - timedelta(days=cfg.unregistered_sale_lookback_days)
    hits = [
        f
        for f in inputs.filings
        if str(f.get("form") or "").upper().strip() == "8-K"
        and _has_item(f, "3.02")
        and (_parse_filing_date(f.get("filingDate", "")) or date.min) >= window_start
    ]
    return _SignalResult(True, fires=bool(hits), severity="C" if hits else None, label="unregistered equity sale, 8-K item 3.02 (N13)")


# ── N14/P1/P2 — Form 4 transactions ──────────────────────────────────────────


def _form4_window(rows: List[Dict[str, Any]], as_of: date, days: int) -> List[Dict[str, Any]]:
    start = as_of - timedelta(days=days)
    out = []
    for r in rows:
        d = _parse_filing_date(str(r.get("filed_date", "")))
        if d is not None and d >= start:
            out.append(r)
    return out


def _mechanical_sale_pairs(rows: List[Dict[str, Any]]) -> set:
    """(insider_name, filed_date) pairs with a same-day code-M exercise — an
    M+S pair is a mechanical exercise-and-sell, not an information-bearing
    disposal (StructuralSignals.md N14 failure mode)."""
    return {(r.get("insider_name"), r.get("filed_date")) for r in rows if r.get("transaction_code") == "M"}


def _eval_insider_sells(inputs: GradingInputs, cfg: P19StructuralConfig) -> _SignalResult:
    if inputs.form4_rows is None:
        return _SignalResult(resolved=False)
    window = _form4_window(inputs.form4_rows, inputs.as_of, cfg.insider_sell_window_days)
    mechanical = _mechanical_sale_pairs(window)
    sellers = {
        r.get("insider_name")
        for r in window
        if r.get("transaction_code") in _SALE_CODES
        and not r.get("is_10b5_1_plan")
        and (r.get("insider_name"), r.get("filed_date")) not in mechanical
    }
    fires = len(sellers) >= cfg.n14_min_distinct_sellers
    return _SignalResult(True, fires=fires, severity="C" if fires else None, label=f"{len(sellers)} distinct insider sellers, 90d (N14)")


def _eval_insider_buys(
    inputs: GradingInputs, cfg: P19StructuralConfig
) -> Tuple[_SignalResult, _SignalResult, int, int]:
    """Returns (P1 result, P2 result, insider_buys_90d, distinct_buyers_90d)."""
    if inputs.form4_rows is None:
        empty = _SignalResult(resolved=False)
        return empty, empty, 0, 0

    window_90 = _form4_window(inputs.form4_rows, inputs.as_of, cfg.insider_buy_window_days)
    buys_90 = [r for r in window_90 if r.get("transaction_code") == _BUY_CODE and not r.get("is_10b5_1_plan")]
    distinct_90 = {r.get("insider_name") for r in buys_90}

    window_30 = _form4_window(inputs.form4_rows, inputs.as_of, cfg.insider_buy_cluster_window_days)
    buys_30 = [r for r in window_30 if r.get("transaction_code") == _BUY_CODE and not r.get("is_10b5_1_plan")]
    distinct_30 = {r.get("insider_name") for r in buys_30}

    p1 = _SignalResult(True, fires=len(distinct_30) >= cfg.p1_min_distinct_buyers)
    p2 = _SignalResult(True, fires=len(distinct_90) >= 1)
    return p1, p2, len(buys_90), len(distinct_90)


# ── P4 — executed buyback ─────────────────────────────────────────────────


def _eval_buyback(inputs: GradingInputs) -> _SignalResult:
    if inputs.company_facts is None:
        return _SignalResult(resolved=False)
    series = xbrl_facts.buybacks_quarterly(inputs.company_facts)
    if not series:
        return _SignalResult(resolved=False)
    return _SignalResult(True, fires=any(v > 0 for _, v in series))


# ── P7 — no dilution event in 24 months ──────────────────────────────────────


def _eval_no_dilution_event(inputs: GradingInputs, days_since_offering: Optional[int]) -> _SignalResult:
    if inputs.filings is None:
        return _SignalResult(resolved=False)
    no_3_02_result = _eval_unregistered_sale(
        inputs,
        # 24 months, not the N13 30-day window -- construct a throwaway cfg-shaped lookback.
        P19StructuralConfig(unregistered_sale_lookback_days=730),
    )
    no_recent_offering = days_since_offering is None or days_since_offering > 730
    return _SignalResult(True, fires=no_recent_offering and not no_3_02_result.fires)


# ── N5 — floating-rate/toxic convertible (EFTS phrase match) ────────────────


def _eval_floating_convert(inputs: GradingInputs, cfg: P19StructuralConfig) -> _SignalResult:
    if inputs.floating_convert_hit is None:
        return _SignalResult(resolved=False)
    return _SignalResult(
        True,
        fires=inputs.floating_convert_hit,
        severity=cfg.n5_severity if inputs.floating_convert_hit else None,
        label="floating-rate/toxic convertible language matched, latest annual+interim (N5)",
    )


# ── N6 — going-concern qualification (EFTS phrase match) ────────────────────


def _eval_going_concern(inputs: GradingInputs) -> _SignalResult:
    if inputs.going_concern_hit is None:
        return _SignalResult(resolved=False)
    return _SignalResult(
        True,
        fires=inputs.going_concern_hit,
        severity="D" if inputs.going_concern_hit else None,
        label="going-concern qualification, latest annual (N6)",
    )


# ── N15 — recent IPO + micro float + FPI reporting ───────────────────────────


def _eval_recent_ipo_fpi(inputs: GradingInputs, cfg: P19StructuralConfig) -> Tuple[_SignalResult, Optional[int]]:
    if inputs.listing_date is None or inputs.is_fpi is None or inputs.float_shares is None:
        return _SignalResult(resolved=False), None
    months_since_listing = int((inputs.as_of - inputs.listing_date).days / 30.44)
    fires = (
        months_since_listing <= cfg.n15_ipo_window_months
        and inputs.float_shares < cfg.n15_float_threshold_shares
        and inputs.is_fpi
    )
    return (
        _SignalResult(True, fires=fires, severity="C" if fires else None, label="recent IPO + micro float + FPI reporting (N15)"),
        months_since_listing,
    )


# ── N16 — auditor quality ─────────────────────────────────────────────────


def _eval_auditor(inputs: GradingInputs, cfg: P19StructuralConfig) -> Tuple[_SignalResult, Optional[bool]]:
    if inputs.auditor_name is None:
        return _SignalResult(resolved=False), None
    normalized = inputs.auditor_name.strip().upper()
    whitelisted = any(w in normalized for w in cfg.auditor_whitelist)
    return (
        _SignalResult(
            True,
            fires=not whitelisted,
            severity="C" if not whitelisted else None,
            label=f"auditor '{inputs.auditor_name}' not on the reputable-auditor whitelist (N16)",
        ),
        whitelisted,
    )


# ── P8 — institutional 13D/G accumulation (presence proxy) ──────────────────


def _eval_inst_accumulation(inputs: GradingInputs) -> _SignalResult:
    if inputs.dg_activity_2q is None:
        return _SignalResult(resolved=False)
    return _SignalResult(True, fires=inputs.dg_activity_2q)


# ── P9 — no debt maturity within 24 months ───────────────────────────────────


def _eval_debt_maturity(inputs: GradingInputs) -> Tuple[_SignalResult, Optional[bool]]:
    if inputs.debt_maturity_near_term is None:
        return _SignalResult(resolved=False), None
    no_near_term = not inputs.debt_maturity_near_term
    return _SignalResult(True, fires=no_near_term), no_near_term


# ── P11 — SI conditional on grade (StructuralSignals.md §4, the highest-value
# conditionality in the model: identical raw number, opposite meaning) ──────


def _eval_short_interest_conditional(
    inputs: GradingInputs, cfg: P19StructuralConfig, pre_grade_is_c_or_d: bool
) -> Tuple[_SignalResult, float]:
    """
    Returns (P11 result for insider_conviction weighting, dilution_urgency bump).

    At grade A/B: high SI + rising days-to-cover is squeeze fuel, feeds
    ``insider_conviction`` like any other resolved P-signal.

    At grade C/D: the same numbers are distribution fuel — an issuer with an
    active ATM can supply every share the shorts need. P11 is deliberately
    returned as ``resolved=False`` here so it is excluded from
    ``insider_conviction``'s renormalisation (never silently zero-scored, per
    the "absence of data" trap — this isn't absence, it's a real exclusion);
    instead it nudges ``dilution_urgency`` up via the returned bump.
    """
    if inputs.short_interest_pct_float is None:
        return _SignalResult(resolved=False), 0.0
    high_si = inputs.short_interest_pct_float >= cfg.p11_si_threshold
    rising_dtc = inputs.days_to_cover is not None and inputs.days_to_cover >= cfg.p11_days_to_cover_threshold
    squeeze_conditions_met = high_si and rising_dtc
    if pre_grade_is_c_or_d:
        return _SignalResult(resolved=False), (cfg.p11_dilution_urgency_bump if squeeze_conditions_met else 0.0)
    return _SignalResult(True, fires=squeeze_conditions_met), 0.0


# ── dilution_urgency (§7.4) ───────────────────────────────────────────────


def _dilution_urgency(
    cfg: P19StructuralConfig,
    runway: Optional[float],
    capacity_pct: Optional[float],
    recent_usage_count: int,
    cagr_8q: Optional[float],
) -> float:
    """
    Weighted sum, each term clipped to [0, 1] independently before weighting,
    then the whole sum clipped to [0, 1] (spec §7.4). Missing terms are
    treated as 0 contribution — this is a deliberate under-estimate (better
    to understate urgency on thin data than fabricate a score), distinct from
    the grade/coverage logic which never treats "unknown" as "clean".
    """
    runway_pressure = 1.0 - min(1.0, max(0.0, (runway or cfg.runway_normalize_quarters) / cfg.runway_normalize_quarters))
    shelf_term = min(1.0, max(0.0, capacity_pct or 0.0))
    usage_term = min(1.0, max(0.0, recent_usage_count / cfg.recent_usage_lookback_quarters))
    history_term = min(1.0, max(0.0, (cagr_8q or 0.0) / cfg.dilution_history_cagr_clip))
    total = (
        cfg.dilution_urgency_w_runway * runway_pressure
        + cfg.dilution_urgency_w_shelf_capacity * shelf_term
        + cfg.dilution_urgency_w_recent_usage * usage_term
        + cfg.dilution_urgency_w_dilution_history * history_term
    )
    return 100.0 * min(1.0, max(0.0, total))


# ── insider_conviction (renormalised over resolved signals only) ────────────


def _insider_conviction(cfg: P19StructuralConfig, results: Dict[str, _SignalResult]) -> Tuple[float, float]:
    """Returns (insider_conviction, fraction of P-signal weight resolved)."""
    weights = cfg.insider_conviction_weights
    available = 0.0
    earned = 0.0
    for sig_id, w in weights.items():
        r = results.get(sig_id)
        if r is None or not r.resolved:
            continue
        available += w
        if r.fires:
            earned += w
    if available <= 0:
        return 0.0, 0.0
    return 100.0 * earned / available, available / sum(weights.values())


# ── Top-level orchestration ──────────────────────────────────────────────


def grade_ticker(inputs: GradingInputs, cfg: Optional[P19StructuralConfig] = None) -> StructuralProfile:
    """
    Evaluate every Phase-1.5 N/P signal for one ticker and return its
    ``StructuralProfile``. Never raises on missing data — an unresolved signal
    just doesn't contribute to coverage or to any score (StructuralSignals.md
    §1 rule 2: unknown grades C, never A).
    """
    cfg = cfg or P19StructuralConfig()

    splits_r, splits_24m, splits_ever = _eval_reverse_splits(inputs, cfg)
    cagr_r, cagr_8q = _eval_share_cagr(inputs, cfg)
    deficiency_r = _eval_deficiency_notice(inputs, cfg)
    offering_r, days_since_offering = _eval_offering_recency(inputs, cfg)

    proceeds_series = xbrl_facts.proceeds_from_issuance(inputs.company_facts) if inputs.company_facts else []
    twelve_months_ago = inputs.as_of - timedelta(days=365)
    proceeds_ttm = sum(v for d, v in proceeds_series if d >= twelve_months_ago and v > 0)
    shelf_r, shelf_active, capacity_pct = _eval_shelf_capacity(inputs, cfg, proceeds_ttm)

    atm_r, atm_used = _eval_atm_usage(inputs, cfg)
    runway_r, cash, runway = _eval_runway(inputs, cfg, shelf_active)
    ocf_r = _eval_positive_ocf(inputs)
    net_cash_r = _eval_long_runway_or_net_cash(inputs, cfg, runway)
    unreg_r = _eval_unregistered_sale(inputs, cfg)
    sells_r = _eval_insider_sells(inputs, cfg)
    p1_r, p2_r, insider_buys_90d, distinct_buyers_90d = _eval_insider_buys(inputs, cfg)
    buyback_r = _eval_buyback(inputs)
    no_dilution_r = _eval_no_dilution_event(inputs, days_since_offering)
    # P3 shares N3/N4's resolved-ness (same underlying split-adjusted series) but
    # is the OPPOSITE polarity: it's a positive signal firing on flat/declining
    # share count, not a disqualifier on high CAGR. Must be its own object --
    # aliasing cagr_r directly would both invert P3's meaning and wrongly add a
    # "P3" entry to disqualifier_severities whenever N3/N4 fires.
    p3_r = _SignalResult(resolved=cagr_r.resolved, fires=bool(cagr_r.resolved and cagr_8q is not None and cagr_8q <= 0))

    convert_r = _eval_floating_convert(inputs, cfg)
    going_concern_r = _eval_going_concern(inputs)
    ipo_fpi_r, recent_ipo_months = _eval_recent_ipo_fpi(inputs, cfg)
    auditor_r, auditor_whitelisted = _eval_auditor(inputs, cfg)
    inst_accum_r = _eval_inst_accumulation(inputs)
    debt_maturity_r, no_debt_maturity = _eval_debt_maturity(inputs)

    signal_results: Dict[str, _SignalResult] = {
        "N1/N2": splits_r,
        "N3/N4": cagr_r,
        "N5": convert_r,
        "N6": going_concern_r,
        "N7": deficiency_r,
        "N8": offering_r,
        "N9": shelf_r,
        "N10": atm_r,
        "N11": runway_r,
        "N13": unreg_r,
        "N14": sells_r,
        "N15": ipo_fpi_r,
        "N16": auditor_r,
        "P1": p1_r,
        "P2": p2_r,
        "P3": p3_r,
        "P4": buyback_r,
        "P5": ocf_r,
        "P6": net_cash_r,
        "P7": no_dilution_r,
        "P8": inst_accum_r,
        "P9": debt_maturity_r,
    }

    disqualifiers: List[str] = []
    disqualifier_severities: Dict[str, str] = {}
    for sig_id, r in signal_results.items():
        if r.resolved and r.fires and r.severity and r.label:
            disqualifiers.append(r.label)
            disqualifier_severities[sig_id] = r.severity

    has_d = any(sev == "D" for sev in disqualifier_severities.values())
    has_c = any(sev == "C" for sev in disqualifier_severities.values())

    resolved_count = sum(1 for r in signal_results.values() if r.resolved)
    coverage = resolved_count / len(signal_results)

    # P11 needs to know whether the disqualifier-driven grade is C/D *before*
    # insider_conviction is finalised (StructuralSignals.md §4 — conditional on
    # grade, so this can't wait for the final A/B/C/D value below).
    pre_grade_is_c_or_d = has_d or has_c or coverage < cfg.coverage_c_threshold
    si_r, dilution_urgency_si_bump = _eval_short_interest_conditional(inputs, cfg, pre_grade_is_c_or_d)
    conviction_results = dict(signal_results)
    conviction_results["P11"] = si_r

    dilution_urgency = min(
        100.0,
        _dilution_urgency(cfg, runway, capacity_pct, 1 if atm_used else 0, cagr_8q) + dilution_urgency_si_bump,
    )
    insider_conviction, _ = _insider_conviction(cfg, conviction_results)

    if has_d:
        grade = "D"
    elif has_c or coverage < cfg.coverage_c_threshold:
        grade = "C"
    elif insider_conviction >= cfg.insider_conviction_a_threshold and dilution_urgency < cfg.dilution_urgency_a_threshold:
        grade = "A"
    else:
        grade = "B"

    return StructuralProfile(
        ticker=inputs.ticker,
        cik=inputs.cik,
        as_of=inputs.as_of,
        grade=grade,
        dilution_urgency=dilution_urgency,
        insider_conviction=insider_conviction,
        reverse_splits_24m=splits_24m,
        reverse_split_ever=splits_ever,
        share_count_cagr_8q=cagr_8q,
        shares_outstanding=inputs.float_shares,
        cash=cash,
        quarterly_burn=(cash / runway) if (cash is not None and runway) else None,
        runway_quarters=runway,
        shelf_active=shelf_active,
        shelf_capacity_pct_mcap=capacity_pct,
        days_since_last_offering=days_since_offering,
        atm_used_last_2q=atm_used,
        insider_buys_90d=insider_buys_90d,
        distinct_insider_buyers_90d=distinct_buyers_90d,
        insider_sells_90d=len(_form4_window(inputs.form4_rows or [], inputs.as_of, cfg.insider_sell_window_days))
        if inputs.form4_rows is not None
        else 0,
        floating_convert_flag=inputs.floating_convert_hit,
        going_concern_flag=inputs.going_concern_hit,
        inst_13dg_activity_2q=inputs.dg_activity_2q,
        recent_ipo_months=recent_ipo_months,
        is_fpi=inputs.is_fpi,
        auditor_name=inputs.auditor_name,
        auditor_whitelisted=auditor_whitelisted,
        no_debt_maturity_24m=no_debt_maturity,
        short_interest_pct_float=inputs.short_interest_pct_float,
        days_to_cover=inputs.days_to_cover,
        coverage=coverage,
        disqualifiers=disqualifiers,
        disqualifier_severities=disqualifier_severities,
    )

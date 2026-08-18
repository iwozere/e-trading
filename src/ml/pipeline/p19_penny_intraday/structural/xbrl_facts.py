"""
P19 Layer 0 — XBRL fact extraction (pure functions, no network calls).

Operates on an already-loaded ``companyfacts`` JSON blob (from
``EdgarDownloader.load_company_facts``). Two defects fixed here are the ones
StructuralSignals.md §7 calls the most dangerous in the whole Layer 0 design:

1. **Split-adjustment** (`split_adjust`) — an unadjusted share-count series makes
   a reverse split look like *falling* share count, i.e. the most-diluted names
   in the universe would score cleanest (N3/N4/P3). Fixed centrally here, once,
   rather than per-signal.
2. **De-cumulation** (`decumulate_quarterly`) — many filers report cash-flow
   figures as fiscal-year-to-date cumulative (a Q3 10-Q covers 9 months), which
   without correction makes every filer's Q4 look like a spike (N10/N11).

Every function here is pure and takes already-loaded data — no rate limiting, no
caching, no EDGAR client. That lives in ``EdgarDownloader`` (fetch) and
``profiler.py`` (orchestration).
"""

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

# Tag fallback chains, preference order (StructuralSignals.md notes each one).
_SHARES_TAGS: List[Tuple[str, str]] = [
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstandingPeriod"),
]
_CASH_TAGS: List[Tuple[str, str]] = [
    ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
    ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
]
_OCF_TAGS: List[Tuple[str, str]] = [
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
]
_PROCEEDS_TAGS: List[Tuple[str, str]] = [
    ("us-gaap", "ProceedsFromIssuanceOfCommonStock"),
    ("us-gaap", "ProceedsFromIssuanceOfCommonStockNetOfIssuanceCosts"),
]

# De-cumulation: a duration fact spanning this many days or fewer is treated as
# already discrete (a real quarter is ~90 days; leave headroom for short/long
# fiscal quarters and reporting-date drift).
_DISCRETE_DURATION_DAYS = 100


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _instant_entries(cf_json: Dict[str, Any], taxonomy: str, tag: str) -> List[Tuple[date, float, str]]:
    """
    Return (end_date, value, filed_date) for an XBRL instant fact (balance-sheet
    style — shares outstanding, cash), deduplicated per end date keeping the
    latest-filed value (restatements supersede the original).
    """
    fact = cf_json.get("facts", {}).get(taxonomy, {}).get(tag)
    if not fact:
        return []
    best_by_end: Dict[date, Tuple[float, str]] = {}
    for unit_entries in fact.get("units", {}).values():
        for e in unit_entries:
            end = _parse_date(e.get("end") or e.get("instant"))
            val = e.get("val")
            if end is None or val is None:
                continue
            filed = str(e.get("filed", ""))
            prev = best_by_end.get(end)
            if prev is None or filed >= prev[1]:
                best_by_end[end] = (float(val), filed)
    return sorted((end, val, filed) for end, (val, filed) in best_by_end.items())


def _duration_entries(cf_json: Dict[str, Any], taxonomy: str, tag: str) -> List[Tuple[date, date, float, str]]:
    """
    Return (start, end, value, filed_date) for an XBRL duration fact (income /
    cash-flow statement style — may be a discrete quarter or a cumulative
    year-to-date figure; see ``decumulate_quarterly``).
    """
    fact = cf_json.get("facts", {}).get(taxonomy, {}).get(tag)
    if not fact:
        return []
    out: List[Tuple[date, date, float, str]] = []
    for unit_entries in fact.get("units", {}).values():
        for e in unit_entries:
            start = _parse_date(e.get("start"))
            end = _parse_date(e.get("end"))
            val = e.get("val")
            if start is None or end is None or val is None:
                continue
            out.append((start, end, float(val), str(e.get("filed", ""))))
    return out


def _first_available_instant(cf_json: Dict[str, Any], tags: List[Tuple[str, str]]) -> List[Tuple[date, float, str]]:
    for taxonomy, tag in tags:
        entries = _instant_entries(cf_json, taxonomy, tag)
        if entries:
            return entries
    return []


def _first_available_duration(cf_json: Dict[str, Any], tags: List[Tuple[str, str]]) -> List[Tuple[date, date, float, str]]:
    for taxonomy, tag in tags:
        entries = _duration_entries(cf_json, taxonomy, tag)
        if entries:
            return entries
    return []


# ── Public API ────────────────────────────────────────────────────────────


def shares_outstanding_series(cf_json: Dict[str, Any]) -> List[Tuple[date, float]]:
    """
    Cover-page shares-outstanding time series, one point per filing date,
    de-duplicated to the latest-filed value per end date. **Not split-adjusted**
    — pass through ``split_adjust`` before computing any CAGR (N3/N4/P3).
    """
    return [(end, val) for end, val, _ in _first_available_instant(cf_json, _SHARES_TAGS)]


def split_adjust(series: List[Tuple[date, float]], splits: List[Tuple[date, float]]) -> List[Tuple[date, float]]:
    """
    Adjust a share-count series for splits so historical points are comparable
    to today's share count (StructuralSignals.md §7 trap #1 — the single most
    dangerous defect in Layer 0 if skipped).

    Args:
        series: (date, shares) points, ascending or unordered.
        splits: (effective_date, ratio) — ratio < 1.0 is a reverse split (e.g.
            a 1-for-10 reverse split is ratio 0.1), ratio > 1.0 is a forward
            split. This is exactly the shape of ``yf.Ticker(t).splits``.

    Returns:
        (date, adjusted_shares) points. A point dated before a split gets
        multiplied by that split's ratio (and every later split's ratio too),
        so a pre-split count reads as its post-split equivalent.
    """
    adjusted = []
    for d, v in series:
        factor = 1.0
        for split_date, ratio in splits:
            if split_date > d:
                factor *= ratio
        adjusted.append((d, v * factor))
    return sorted(adjusted)


def cagr(series: List[Tuple[date, float]], lookback_quarters: int = 8) -> Optional[float]:
    """
    Compound annual growth rate over the trailing ``lookback_quarters`` points
    of an (already split-adjusted, for share counts) series. None if fewer than
    2 points exist in the window, or the earliest value is not positive.
    """
    if len(series) < 2:
        return None
    window = sorted(series)[-lookback_quarters:]
    if len(window) < 2:
        return None
    (start_date, start_val), (end_date, end_val) = window[0], window[-1]
    if start_val <= 0 or end_val <= 0:
        return None
    years = (end_date - start_date).days / 365.25
    if years <= 0:
        return None
    return (end_val / start_val) ** (1.0 / years) - 1.0


def decumulate_quarterly(entries: List[Tuple[date, date, float]]) -> List[Tuple[date, float]]:
    """
    De-cumulate XBRL duration facts into discrete quarterly values
    (StructuralSignals.md §7 trap #2 — affects N10 and N11 identically).

    Many filers report cash-flow-statement facts as fiscal-year-to-date
    cumulative (a Q3 10-Q covers 9 months, not 3). A fact whose own duration is
    already close to one quarter (<= 100 days) is trusted as discrete as-is.
    Longer-duration facts are grouped by their shared ``start`` date (every
    cumulative fact within one fiscal year shares the same fiscal-year-start),
    sorted by end date, and de-cumulated by subtracting the prior cumulative
    value within that group — seeded from the fiscal year's own discrete Q1
    (which shares that same start date) when one is present, not from zero,
    otherwise Q2's YTD figure is mistaken for a discrete Q2 value.

    Args:
        entries: (start, end, value) duration-fact tuples.

    Returns:
        (end, discrete_value) points, one per quarter-end, ascending.
    """
    discrete: Dict[date, float] = {}
    # Exact-match on start date; a filer whose Q1 start date drifts slightly
    # from its YTD figures' shared start would miss this seed and fall back to
    # a zero baseline — acceptable for Phase 1.5's deterministic-signals scope.
    short_by_start: Dict[date, float] = {}
    groups: Dict[date, List[Tuple[date, float]]] = {}
    for start, end, value in entries:
        if (end - start).days <= _DISCRETE_DURATION_DAYS:
            discrete[end] = value
            short_by_start[start] = value
        else:
            groups.setdefault(start, []).append((end, value))

    for start, points in groups.items():
        points.sort(key=lambda p: p[0])
        prev_cum = short_by_start.get(start, 0.0)
        for end, cum_value in points:
            if end in discrete:
                # A genuinely discrete fact for this exact quarter-end already
                # exists and is more precise than one derived by subtraction.
                prev_cum = cum_value
                continue
            discrete[end] = cum_value - prev_cum
            prev_cum = cum_value

    return sorted(discrete.items())


def cash_and_burn(
    cf_json: Dict[str, Any],
    trailing_quarters: int = 4,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Latest cash balance, average quarterly operating cash burn (trailing
    ``trailing_quarters`` **discrete**, de-cumulated quarters), and the
    resulting runway in quarters (N11).

    Returns:
        (cash, quarterly_burn, runway_quarters) — each None if unresolved.
        ``quarterly_burn`` is None both when operating cash flow was positive
        every quarter in the window (a *good* sign) and when no OCF data
        resolved at all — these are different things for P5/P6 and the caller
        must tell them apart via ``operating_cash_flow_quarterly`` directly
        rather than inferring it from this tuple.
    """
    cash_entries = _first_available_instant(cf_json, _CASH_TAGS)
    cash = cash_entries[-1][1] if cash_entries else None

    ocf_discrete = operating_cash_flow_quarterly(cf_json)
    window = ocf_discrete[-trailing_quarters:]
    burns = [-v for _, v in window if v < 0]
    quarterly_burn = (sum(burns) / len(burns)) if burns else None

    if cash is None or not quarterly_burn or quarterly_burn <= 0:
        return cash, quarterly_burn, None
    return cash, quarterly_burn, cash / quarterly_burn


def operating_cash_flow_quarterly(cf_json: Dict[str, Any]) -> List[Tuple[date, float]]:
    """De-cumulated discrete quarterly operating cash flow (P5/P6; feeds ``cash_and_burn``)."""
    raw = _first_available_duration(cf_json, _OCF_TAGS)
    return decumulate_quarterly([(s, e, v) for s, e, v, _ in raw])


def proceeds_from_issuance(cf_json: Dict[str, Any]) -> List[Tuple[date, float]]:
    """
    De-cumulated discrete quarterly ``ProceedsFromIssuanceOfCommonStock`` (N10).

    Caller must still apply the magnitude threshold from StructuralSignals.md's
    N10 failure-mode note (the tag also captures option/warrant exercises) —
    that's a grading decision (config-driven threshold), not an extraction one.
    """
    raw = _first_available_duration(cf_json, _PROCEEDS_TAGS)
    return decumulate_quarterly([(s, e, v) for s, e, v, _ in raw])


def buybacks_quarterly(cf_json: Dict[str, Any]) -> List[Tuple[date, float]]:
    """De-cumulated discrete quarterly ``PaymentsForRepurchaseOfCommonStock`` (P4)."""
    raw = _first_available_duration(cf_json, [("us-gaap", "PaymentsForRepurchaseOfCommonStock")])
    return decumulate_quarterly([(s, e, v) for s, e, v, _ in raw])


# Both instant-style facts (verified live, 2026-08-18, CIK 0000320193 Apple):
# ``end``, no ``start`` — the current-portion balance and the explicit
# next-12-months maturity schedule line, in fallback-chain order.
_DEBT_MATURITY_TAGS: List[Tuple[str, str]] = [
    ("us-gaap", "LongTermDebtCurrent"),
    ("us-gaap", "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths"),
]


def has_near_term_debt_maturity(cf_json: Dict[str, Any]) -> Optional[bool]:
    """
    P9 raw signal: is there a tagged near-term (next 12 months) debt maturity?

    A proxy for the spec's 24-month window (StructuralSignals.md P9) — the
    next-12-months tag is what's reliably reported; a filer with debt due in
    months 13-24 but nothing due in the next 12 would read as "no near-term
    maturity" here, a documented approximation, not the full 24-month test.

    Requires an explicit tag to resolve — absence of *any* debt tag on a
    filer's companyfacts is ambiguous (many microcaps simply don't carry debt
    to tag, but "not tagged" and "zero" aren't distinguishable without it), so
    this returns None (unresolved) rather than assuming debt-free.

    Returns:
        True if a positive near-term maturity value is tagged, False if the
        tag resolves to zero/negative, None if no such tag was found at all.
    """
    entries = _first_available_instant(cf_json, _DEBT_MATURITY_TAGS)
    if not entries:
        return None
    return entries[-1][1] > 0

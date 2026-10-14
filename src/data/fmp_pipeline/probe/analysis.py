"""
Pure analysis helpers for the free-key probe report (spec §4.2.5, §4.2.6, §4.2.8). No I/O.

Outputs are derived facts (ratios, counts, date ranges, field names), never raw price rows, so they can be
written into the committed report without redistributing FMP data (D5).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class DividendBasis:
    """Which ``dividends`` field matches a known raw-basis amount on a known ex-date."""

    ex_date: str
    expected: float
    found: bool
    fields_matching: tuple[str, ...]
    values: dict[str, Optional[float]]


@dataclass(frozen=True)
class SplitCheck:
    """Close-to-close ratio across a split boundary for one price endpoint."""

    endpoint: str
    pre_date: Optional[str]
    post_date: Optional[str]
    close_ratio: Optional[float]
    verdict: str


@dataclass(frozen=True)
class ReusedTicker:
    """A ticker with evidence of belonging to more than one company."""

    symbol: str
    reason: str
    entries: tuple[dict[str, Any], ...]
    latest_date: str


def dividend_basis(rows: Iterable[dict[str, Any]], ex_date: str, expected: float, tol: float = 1e-6) -> DividendBasis:
    """
    Find which numeric dividend fields equal ``expected`` on ``ex_date``.

    Args:
        rows: Records from the ``dividends`` endpoint.
        ex_date: Ex-date to look up (``YYYY-MM-DD``), matched against the ``date`` field.
        expected: Known raw-basis amount.
        tol: Absolute tolerance.

    Returns:
        The matching fields and every numeric field's value on that date (fields named like dividends only).
    """
    for row in rows:
        if str(row.get("date", ""))[:10] != ex_date:
            continue
        values = {k: (float(v) if isinstance(v, (int, float)) else None) for k, v in row.items()
                  if "dividend" in k.lower()}
        matching = tuple(k for k, v in values.items() if v is not None and abs(v - expected) <= tol)
        return DividendBasis(ex_date, expected, True, matching, values)
    return DividendBasis(ex_date, expected, False, (), {})


def split_check(
    endpoint: str, rows: Iterable[dict[str, Any]], split_date: str, ratio: float, tol: float = 0.15
) -> SplitCheck:
    """
    Classify a daily price endpoint as raw or split-adjusted from the close ratio across a split.

    ``close(last bar before split) / close(first bar on or after split)`` is near ``ratio`` for raw prices and near
    1 for split-adjusted prices; ``tol`` absorbs the ordinary one-day move.

    Args:
        endpoint: Endpoint name, for labelling.
        rows: Daily records with ``date`` and ``close``, or ``adjClose`` when there is no ``close`` (the
            ``non-split-adjusted`` and ``dividend-adjusted`` endpoints only carry ``adj*`` price fields).
        split_date: Split effective date (``YYYY-MM-DD``).
        ratio: Split ratio (numerator / denominator).
        tol: Relative tolerance for the classification.

    Returns:
        The check result; ``verdict`` is ``raw``, ``split_adjusted``, ``unclear`` or ``no_data``.
    """
    rows = list(rows)
    field = "close" if any("close" in r for r in rows) else "adjClose"
    bars = sorted((str(r["date"])[:10], float(r[field])) for r in rows if r.get(field) is not None)
    before = [b for b in bars if b[0] < split_date]
    after = [b for b in bars if b[0] >= split_date]
    if not before or not after:
        return SplitCheck(endpoint, None, None, None, "no_data")
    value = before[-1][1] / after[0][1]
    if abs(value / ratio - 1) <= tol:
        verdict = "raw"
    elif abs(value - 1) <= tol:
        verdict = "split_adjusted"
    else:
        verdict = "unclear"
    return SplitCheck(endpoint, before[-1][0], after[0][0], round(value, 4), verdict)


def date_range(rows: Iterable[dict[str, Any]], key: str = "date") -> tuple[Optional[str], Optional[str]]:
    """Return the min and max of ``key`` (first 10 characters) across records, or ``(None, None)``."""
    dates = sorted(str(r[key])[:10] for r in rows if r.get(key))
    return (dates[0], dates[-1]) if dates else (None, None)


def find_reused_tickers(
    delisted: Iterable[dict[str, Any]],
    active_symbols: Iterable[str],
    symbol_changes: Iterable[dict[str, Any]],
    limit: int = 5,
) -> list[ReusedTicker]:
    """
    Find tickers that plausibly belong to more than one company.

    Evidence, strongest first: a symbol delisted more than once with different names or IPO dates; a symbol that
    is both delisted and actively trading today (reassigned); a symbol that a ``symbol-change`` both frees (as an
    old symbol) and takes (as a new symbol) for different companies. The most recent candidates come first, since
    the free key only reaches recent history.

    Args:
        delisted: ``delisted-companies`` records.
        active_symbols: Symbols from ``actively-trading-list``.
        symbol_changes: ``symbol-change`` records (``oldSymbol``, ``newSymbol``, ``companyName``, ``date``).
        limit: Maximum number of candidates returned.

    Returns:
        Up to ``limit`` candidates.
    """
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in delisted:
        if row.get("symbol"):
            by_symbol[str(row["symbol"])].append(row)
    active = set(active_symbols)

    candidates: list[ReusedTicker] = []
    for symbol, rows in by_symbol.items():
        identities = {(r.get("companyName"), r.get("ipoDate")) for r in rows}
        latest = max(str(r.get("delistedDate") or "") for r in rows)
        if len(rows) > 1 and len(identities) > 1:
            candidates.append(ReusedTicker(symbol, "delisted more than once", tuple(rows), latest))
        elif symbol in active:
            candidates.append(ReusedTicker(symbol, "delisted and actively trading", tuple(rows), latest))

    olds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    news: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for change in symbol_changes:
        olds[str(change.get("oldSymbol"))].append(change)
        news[str(change.get("newSymbol"))].append(change)
    seen = {c.symbol for c in candidates}
    for symbol in set(olds) & set(news):
        names = {c.get("companyName") for c in olds[symbol] + news[symbol]}
        if symbol not in seen and len(names) > 1:
            rows = olds[symbol] + news[symbol]
            latest = max(str(r.get("date") or "") for r in rows)
            candidates.append(ReusedTicker(symbol, "freed and retaken via symbol-change", tuple(rows), latest))

    rank = {"delisted more than once": 0, "delisted and actively trading": 1, "freed and retaken via symbol-change": 2}
    # Strongest reason first; within a reason, most recent first (two stable sorts).
    candidates.sort(key=lambda c: c.latest_date, reverse=True)
    candidates.sort(key=lambda c: rank[c.reason])
    return candidates[:limit]

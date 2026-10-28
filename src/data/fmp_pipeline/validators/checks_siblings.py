"""
Same-company listing check (V-SIBLING-BASIS): listings that share a CIK and disagree by a split factor.

A company renamed in a reverse merger often has two securities: the old ticker, which ends at the rename, and the
new one, whose history FMP back-fills over the old ticker's lifetime. That back-filled history can be
split-adjusted while the old ticker's bars are raw (vintage 2026-10: KLRS over ALVR, KRRO over FREQ, ACGN over IDRA).
The new listing then shows no jump at the split and often has no split row either, so ``V-SPLIT-CONSISTENCY``
cannot see it. Comparing the two listings on the dates both have can: their closes differ by a constant
whole-number factor (25, 1/50, ...) until the split, and agree after it.

Nothing is corrected (D11). The check writes ``sibling_basis_candidates.csv``: proposed ``overrides/price_basis.csv``
rows with an empty ``evidence`` column, so an unreviewed row copied over is rejected by the store. Which listing is
wrong is a guess: the one with a reviewed override is right; otherwise the listing that continues after the other
ends is taken to carry the adjusted back-fill. Apply the reviewed rows and rerun: the store applies them, so a
corrected pair passes, and a chain of renames (ARPO → AADI → WHWK) resolves one link per round.
"""

from __future__ import annotations

import csv
import io
import itertools
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.fmp_pipeline.builders.events import SPLITS
from src.data.fmp_pipeline.builders.validate import SECURITIES
from src.data.fmp_pipeline.datamanager_adapter.price_overrides import COLUMNS
from src.data.fmp_pipeline.fsutil import atomic_write_bytes
from src.data.fmp_pipeline.validators.framework import PASS, WARN, CheckResult, ValidationContext, check

SIBLING_CANDIDATES = "sibling_basis_candidates.csv"
#: Instrument types compared. Warrants, units and preferreds are mostly typed ``stock`` too; the symbol and
#: whole-factor rules below drop them.
LISTING_TYPES = ("stock", "adr")
MIN_SHARED_DAYS = 20
#: A run of days on one factor shorter than this is noise (a stale print, a bad bar), not a basis.
MIN_RUN_DAYS = 10
#: Share of the shared days that must fall in runs: below it the two closes do not move together at all.
MIN_COVERED = 0.8
#: A daily close ratio within this band of 1 means the listings agree.
SAME_BAND = (0.8, 1.25)
#: Relative distance from a whole number for a ratio to count as a split factor.
FACTOR_TOLERANCE = 0.03
#: Largest spread of the log close ratio inside one run (about 3%).
RUN_SPREAD = math.log(1.03)
#: Factors closer than this are one basis: at large factors the nearest whole number wanders (149999 / 150000).
SAME_FACTOR = 1.15
DIAGNOSTIC_COLUMNS = ("cik", "sibling_id", "sibling_symbol", "shared_days", "first_shared", "last_shared", "factor",
                      "run_days", "ex_date_basis", "direction", "confirmed_by", "confidence")
CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass(frozen=True)
class FactorRun:
    """Consecutive shared days on which ``wrong / right`` closes sit on one split factor."""

    start: date
    end: date
    factor: float
    days: int


@dataclass(frozen=True)
class Listing:
    """One security of a CIK group."""

    security_id: int
    symbol: str
    last_price_date: Optional[date]


def split_factor(ratio: float) -> float:
    """
    The whole-number split factor a close ratio sits on.

    Returns:
        ``1.0`` inside ``SAME_BAND``; ``n`` or ``1/n`` for a ratio within ``FACTOR_TOLERANCE`` of a whole number
        ``n >= 2`` (or of its reciprocal); ``nan`` for anything else.
    """
    if not math.isfinite(ratio) or ratio <= 0:
        return math.nan
    if SAME_BAND[0] <= ratio <= SAME_BAND[1]:
        return 1.0
    big = max(ratio, 1 / ratio)
    whole = round(big)
    if whole < 2 or abs(big - whole) / whole > FACTOR_TOLERANCE:
        return math.nan
    return float(whole) if ratio > 1 else 1 / whole


def _same_factor(a: float, b: float) -> bool:
    return not (math.isnan(a) or math.isnan(b)) and max(a / b, b / a) < SAME_FACTOR


def _confidence(ratio: float, confirmed: list[str]) -> str:
    """``high``: a split row on the ex-date; ``medium``: the step is a plain n-for-1 or 1-for-n; ``low``: neither."""
    if confirmed:
        return "high"
    step = split_factor(ratio)
    return "medium" if not math.isnan(step) and step != 1.0 else "low"


def factor_runs(wrong: pd.Series, right: pd.Series) -> tuple[list[FactorRun], int]:
    """
    Split the dates both listings have into runs on one split factor of ``wrong / right``.

    A run is a stretch where the daily ratio stays within ``RUN_SPREAD`` of its first day: two listings of one
    basis error are the same bars scaled, so the ratio is constant, while two different instruments drift. A run
    counts when it lasts ``MIN_RUN_DAYS`` and its median ratio is a split factor (``split_factor``); adjacent runs
    on factors within ``SAME_FACTOR`` are merged.

    Args:
        wrong: Daily closes of the listing suspected of carrying adjusted bars.
        right: Daily closes of its sibling.

    Returns:
        The runs in date order, and the number of shared days.
    """
    both = pd.concat([wrong.rename("w"), right.rename("r")], axis=1, join="inner").dropna()
    both = both[(both["w"] > 0) & (both["r"] > 0)]
    if both.empty:
        return [], 0
    logs = np.log((both["w"] / both["r"]).to_numpy(dtype=float))
    dates = [ts.date() for ts in pd.DatetimeIndex(both.index)]
    runs: list[FactorRun] = []
    start = 0
    for i in range(1, len(logs) + 1):
        if i < len(logs) and abs(logs[i] - logs[start]) <= RUN_SPREAD:
            continue
        factor = split_factor(float(np.exp(np.median(logs[start:i]))))
        if i - start >= MIN_RUN_DAYS and not math.isnan(factor):
            if runs and _same_factor(runs[-1].factor, factor):
                last = runs.pop()
                runs.append(FactorRun(last.start, dates[i - 1], last.factor, last.days + i - start))
            else:
                runs.append(FactorRun(dates[start], dates[i - 1], factor, i - start))
        start = i
    return runs, len(both)


def _related(a: str, b: str) -> bool:
    """Share classes, warrants, units and notes of one issuer: same symbol root, or a class separator."""
    return a[:3] == b[:3] or any(sep in a + b for sep in "-.")


def _first_bar_on_or_after(closes: pd.Series, when: date) -> Optional[date]:
    index = pd.DatetimeIndex(closes.index)
    later = index[index >= pd.Timestamp(when)]
    return later[0].date() if len(later) else None


def _split_near(splits: pd.DataFrame, security_id: int, when: date, ratio: float, days: int = 5) -> bool:
    if splits.empty:
        return False
    rows = splits[splits["security_id"] == security_id]
    near = rows[(pd.to_datetime(rows["date"]) - pd.Timestamp(when)).abs() <= pd.Timedelta(days=days)]
    return bool(((near["ratio"] >= 0.7 * ratio) & (near["ratio"] <= 1.4 * ratio)).any())


def propose_rows(vintage: str, cik: str, wrong: Listing, right: Listing, wrong_closes: pd.Series,
                 runs: list[FactorRun], shared: int, direction: str, splits: pd.DataFrame) -> list[dict[str, object]]:
    """
    Turn the factor runs of one pair into override rows for ``wrong``, one per change of factor.

    On a run with factor ``m`` the wrong bars are ``m`` times the raw ones, so they need a cumulative multiplier
    of ``1/m``. Overrides stack (each multiplies every bar before its ex-date), so the row at a change from ``m_i``
    to ``m_next`` carries ``m_next / m_i``; after the last run the listings agree (``m = 1``). When the last run is
    not ``1`` the split lies after the shared dates end: the ex-date is then the wrong listing's next bar, to verify.
    """
    rows: list[dict[str, object]] = []
    for i, run in enumerate(runs):
        if run.factor == 1.0:
            continue
        ex_date: Optional[date]
        if i + 1 < len(runs):
            following = runs[i + 1]
            ex_date, basis, nxt = following.start, "factor change", following.factor
        else:
            ex_date, basis, nxt = _first_bar_on_or_after(wrong_closes, run.end + timedelta(days=1)), "overlap end", 1.0
        if ex_date is None:
            continue
        ratio = nxt / run.factor
        confirmed = [name for name, sid in (("own split row", wrong.security_id), ("sibling split row", right.security_id))
                     if _split_near(splits, sid, ex_date, ratio)]
        rows.append({
            "vintage": vintage, "security_id": wrong.security_id, "symbol": wrong.symbol,
            "ex_date": ex_date.isoformat(), "ratio": round(ratio, 6), "from_date": "",
            "add_split": "false" if "own split row" in confirmed else "true", "evidence": "",
            "cik": cik, "sibling_id": right.security_id, "sibling_symbol": right.symbol, "shared_days": shared,
            "first_shared": runs[0].start.isoformat(), "last_shared": runs[-1].end.isoformat(),
            "factor": round(run.factor, 6), "run_days": run.days, "ex_date_basis": basis, "direction": direction,
            "confirmed_by": "; ".join(confirmed) or "none", "confidence": _confidence(ratio, confirmed),
        })
    return rows


def _orient(a: Listing, b: Listing, overridden: set[int]) -> tuple[Listing, Listing, str]:
    """``(wrong, right, why)``: a reviewed override marks the right one, else the continuing listing is wrong."""
    if (a.security_id in overridden) != (b.security_id in overridden):
        return (b, a, "sibling has a reviewed override") if a.security_id in overridden \
            else (a, b, "sibling has a reviewed override")
    a_last, b_last = a.last_price_date or date.min, b.last_price_date or date.min
    return (a, b, "continues after the sibling") if a_last >= b_last else (b, a, "continues after the sibling")


def _groups(ctx: ValidationContext) -> dict[str, list[Listing]]:
    columns = ["security_id", "symbol", "cik", "instrument_type", "first_price_date", "last_price_date"]
    ref = pq.read_table(ctx.vintage_dir / SECURITIES, columns=columns).to_pandas()
    ref = ref[ref["cik"].notna() & ref["instrument_type"].isin(LISTING_TYPES) & ref["first_price_date"].notna()]
    out: dict[str, list[Listing]] = {}
    for row in ref.itertuples(index=False):
        last = pd.Timestamp(row.last_price_date).date() if pd.notna(row.last_price_date) else None
        out.setdefault(str(row.cik), []).append(Listing(int(row.security_id), str(row.symbol), last))
    return {cik: listings for cik, listings in out.items() if len(listings) > 1}


@check("V-SIBLING-BASIS", WARN, "Listings sharing a CIK do not differ by a split factor on their shared dates")
def sibling_basis(ctx: ValidationContext) -> CheckResult:
    """
    Compare every pair of same-CIK listings and propose override rows where they sit on a split factor.

    Each wrong listing is proposed against one sibling: one that is not itself proposed as wrong if there is one,
    then the one with the most shared days (two siblings would propose the same correction twice).
    """
    result = CheckResult("V-SIBLING-BASIS", PASS)
    overridden = set(ctx.store.overrides.rows)
    splits_path = ctx.vintage_dir / SPLITS.path
    splits = pq.read_table(splits_path).to_pandas() if splits_path.exists() else pd.DataFrame()
    closes: dict[int, pd.Series] = {}

    def closes_of(security_id: int) -> pd.Series:
        if security_id not in closes:
            prices = ctx.store.prices(security_id, "1d")
            closes[security_id] = prices["close"] if prices is not None else pd.Series(dtype=float)
        return closes[security_id]

    found: dict[int, list[tuple[Listing, list[FactorRun], int, str, str]]] = {}
    for cik, listings in _groups(ctx).items():
        for a, b in itertools.combinations(listings, 2):
            if _related(a.symbol, b.symbol):
                continue
            wrong, right, direction = _orient(a, b, overridden)
            runs, shared = factor_runs(closes_of(wrong.security_id), closes_of(right.security_id))
            if shared < MIN_SHARED_DAYS:
                continue
            result.examined += 1
            covered = sum(r.days for r in runs)
            if covered < MIN_COVERED * shared or all(r.factor == 1.0 for r in runs):
                continue
            found.setdefault(wrong.security_id, []).append((right, runs, shared, direction, cik))
    wrong_ids = set(found)
    rows: list[dict[str, object]] = []
    for sid, candidates in sorted(found.items()):
        right, runs, shared, direction, cik = min(candidates, key=lambda c: (c[0].security_id in wrong_ids, -c[2]))
        info = ctx.store.security(sid)
        wrong = Listing(sid, info.symbol, info.last_price_date)
        proposed = propose_rows(ctx.store.vintage, cik, wrong, right, closes_of(sid), runs, shared, direction, splits)
        if proposed:
            result.violations += 1
            result.add_example(f"{wrong.symbol} ({sid}) vs {right.symbol} ({right.security_id}): "
                               + ", ".join(f"x{r['ratio']:g} before {r['ex_date']}" for r in proposed))
            rows.extend(proposed)
    rows.sort(key=lambda r: (CONFIDENCE_ORDER[str(r["confidence"])], str(r["symbol"]), str(r["ex_date"])))
    buffer = io.StringIO()
    fieldnames: list[str] = [*COLUMNS, *DIAGNOSTIC_COLUMNS]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    ctx.report_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(ctx.report_dir / SIBLING_CANDIDATES, buffer.getvalue().encode("utf-8"))
    result.status = WARN if result.violations else PASS
    result.message = (f"{result.violations} listing(s) sit on a split factor against a same-CIK sibling "
                      f"({result.examined} pairs compared); {len(rows)} proposed row(s) in {SIBLING_CANDIDATES}")
    result.details = {SIBLING_CANDIDATES: len(rows),
                      **{level: sum(1 for r in rows if r["confidence"] == level) for level in CONFIDENCE_ORDER}}
    return result

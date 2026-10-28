"""Corporate-event and cross-source checks (spec §11): V-SPLIT-KNOWN, V-SPLIT-CONSISTENCY, V-OVERRIDES, V-DIV-SANITY,
V-ADJ-RECON, V-YF-XCHECK."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from datetime import date, datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.fmp_pipeline.builders.events import DIVIDENDS, SPLITS
from src.data.fmp_pipeline.datamanager_adapter.adapter import FmpOhlcvAdapter
from src.data.fmp_pipeline.datamanager_adapter.price_overrides import OVERRIDES_FILE
from src.data.fmp_pipeline.fsutil import atomic_write_bytes
from src.data.fmp_pipeline.validators.framework import FAIL, INFO, PASS, WARN, CheckResult, ValidationContext, check

#: Spec §8.3 test vectors: (symbol, effective date, ratio).
KNOWN_SPLITS = (("AAPL", date(2020, 8, 31), 4.0), ("AAPL", date(2014, 6, 9), 7.0), ("TSLA", date(2020, 8, 31), 5.0),
                ("TSLA", date(2022, 8, 25), 3.0), ("AMZN", date(2022, 6, 6), 20.0), ("GOOGL", date(2022, 7, 18), 20.0),
                ("NVDA", date(2024, 6, 10), 10.0))
#: (symbol, ex-date, raw amount).
KNOWN_DIVIDENDS = (("AAPL", date(2020, 8, 7), 0.82),)
RECON_TABLE = "prices/recon/daily_div_adjusted_sample.parquet"
SPLIT_CANDIDATES = "split_without_jump_candidates.csv"
#: A raw close move inside this band means "no split happened in the prices".
FLAT_BAND = (0.7, 1.4)
EARLY = datetime(1900, 1, 1)
LATE = datetime(2100, 1, 1)


def _events_frame(ctx: ValidationContext, rel: str) -> pd.DataFrame:
    path = ctx.vintage_dir / rel
    return pq.read_table(path).to_pandas() if path.exists() else pd.DataFrame()


@check("V-SPLIT-KNOWN", FAIL, "The spec §8.3 test-vector splits and dividend exist with the right ratios and dates")
def split_known(ctx: ValidationContext) -> CheckResult:
    """Look up each test vector in the events tables."""
    result = CheckResult("V-SPLIT-KNOWN", PASS, examined=len(KNOWN_SPLITS) + len(KNOWN_DIVIDENDS))
    splits, dividends = _events_frame(ctx, SPLITS.path), _events_frame(ctx, DIVIDENDS.path)
    for symbol, when, ratio in KNOWN_SPLITS:
        info = ctx.store.resolve(symbol)
        rows = splits[(splits["security_id"] == info.security_id) & (splits["date"] == when)] \
            if info is not None and not splits.empty else splits.iloc[:0]
        if rows.empty or not np.isclose(rows["ratio"], ratio).any():
            result.violations += 1
            result.add_example(f"{symbol} {ratio:g}:1 on {when}: {'missing' if rows.empty else 'wrong ratio'}")
    for symbol, when, amount in KNOWN_DIVIDENDS:
        info = ctx.store.resolve(symbol)
        rows = dividends[(dividends["security_id"] == info.security_id) & (dividends["ex_date"] == when)] \
            if info is not None and not dividends.empty else dividends.iloc[:0]
        if rows.empty or not np.isclose(rows["amount_raw"], amount, atol=0.005).any():
            result.violations += 1
            result.add_example(f"{symbol} dividend {amount} ex {when}: {'missing' if rows.empty else 'wrong amount'}")
    result.status = FAIL if result.violations else PASS
    return result


def _closes(ctx: ValidationContext, security_id: int) -> pd.Series:
    prices = ctx.store.prices(security_id, "1d")
    # The empty fallback keeps a DatetimeIndex: a RangeIndex cannot be compared with the event dates
    return prices["close"] if prices is not None else pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def _by_security(frame: pd.DataFrame) -> Iterator[tuple[int, pd.DataFrame]]:
    if frame.empty:
        return
    for sid, group in frame[frame["security_id"].notna()].groupby("security_id"):
        yield int(group["security_id"].iloc[0]), group


def _verdict(ratio: float, observed: float) -> str:
    """``flat``: the raw closes do not move although the split is large (bars probably already adjusted)."""
    split_is_large = not FLAT_BAND[0] <= ratio <= FLAT_BAND[1]
    return "flat" if split_is_large and FLAT_BAND[0] <= observed <= FLAT_BAND[1] else "mismatch"


@check("V-SPLIT-CONSISTENCY", WARN, "Raw close ratio across each split within 0.7–1.4 × the declared ratio (≥ 95%)")
def split_consistency(ctx: ValidationContext) -> CheckResult:
    """
    For each split, compare the raw price move across its date with the declared ratio.

    Every inconsistent split goes to ``split_without_jump_candidates.csv`` for review (D11: flag, never alter).
    ``flat`` rows are the probable price-basis errors; a reviewed one is corrected with a row in
    ``overrides/price_basis.csv``, after which it passes here (the store applies the override).
    """
    result = CheckResult("V-SPLIT-CONSISTENCY", PASS)
    rows: list[list[object]] = []
    for sid, group in _by_security(_events_frame(ctx, SPLITS.path)):
        closes = _closes(ctx, sid)
        for _, row in group[group["ratio"].notna()].iterrows():
            when = pd.Timestamp(row["date"])
            before, after = closes[closes.index < when], closes[closes.index >= when]
            if before.empty or after.empty:
                continue
            result.examined += 1
            observed = float(before.iloc[-1] / after.iloc[0])
            if not 0.7 * row["ratio"] <= observed <= 1.4 * row["ratio"]:
                result.violations += 1
                result.add_example(f"{row['symbol']} {when.date()}: declared {row['ratio']:g}, raw closes moved "
                                   f"{observed:.3f}x")
                gap = (after.index[0] - before.index[-1]).days
                rows.append([row["symbol"], sid, when.date().isoformat(), round(float(row["ratio"]), 6),
                             round(observed, 6), _verdict(float(row["ratio"]), observed), float(before.iloc[-1]),
                             float(after.iloc[0]), gap])
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["symbol", "security_id", "date", "ratio", "observed", "verdict", "close_before", "close_after",
                     "gap_days"])
    rows.sort(key=lambda r: (str(r[0]), str(r[2])))
    writer.writerows(rows)
    ctx.report_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(ctx.report_dir / SPLIT_CANDIDATES, buffer.getvalue().encode("utf-8"))
    share = 1 - result.violations / result.examined if result.examined else 1.0
    flat = sum(1 for r in rows if r[5] == "flat")
    result.status = WARN if share < 0.95 else PASS
    result.message = f"{share:.1%} of splits consistent with the raw prices; {flat} flat (see {SPLIT_CANDIDATES})"
    result.details = {SPLIT_CANDIDATES: len(rows), "flat": flat}
    return result


@check("V-OVERRIDES", WARN, "Price-basis overrides: every row valid; none left from another vintage")
def overrides(ctx: ValidationContext) -> CheckResult:
    """Report the overrides the store applies, rows it skipped, and rows reviewed against another vintage."""
    found = ctx.store.overrides
    result = CheckResult("V-OVERRIDES", PASS, examined=len(found) + len(found.problems) + found.other_vintages)
    if not (ctx.data_root / OVERRIDES_FILE).exists():
        result.status, result.message = INFO, f"no {OVERRIDES_FILE}"
        return result
    for problem in found.problems:
        result.violations += 1
        result.add_example(problem)
    if found.other_vintages:
        result.violations += found.other_vintages
        result.add_example(f"{found.other_vintages} row(s) reviewed against another vintage: review them again for "
                           f"{ctx.store.vintage} or remove them")
    result.status = WARN if result.violations else PASS
    result.message = f"{len(found)} override(s) applied for {len(found.rows)} securities"
    result.details = {"applied": [f"{o.symbol} ({o.security_id}) {o.ex_date} x{o.ratio:g}"
                                  f"{' +split' if o.add_split else ''}: {o.evidence}"
                                  for sid in sorted(found.rows) for o in found.rows[sid]]}
    return result


@check("V-DIV-SANITY", WARN, "Dividends positive and below half the previous raw close")
def dividend_sanity(ctx: ValidationContext) -> CheckResult:
    """Flag negative dividends and ``amount_raw / C_prev >= 0.5`` (review list; ``adjust()`` checks large ones against the closes)."""
    result = CheckResult("V-DIV-SANITY", PASS)
    for sid, group in _by_security(_events_frame(ctx, DIVIDENDS.path)):
        closes = _closes(ctx, sid)
        for _, row in group[group["amount_raw"].notna() & group["ex_date"].notna()].iterrows():
            result.examined += 1
            amount = float(row["amount_raw"])
            prior = closes[closes.index < pd.Timestamp(row["ex_date"])]
            ratio = amount / float(prior.iloc[-1]) if not prior.empty and prior.iloc[-1] > 0 else None
            if amount < 0 or (ratio is not None and ratio >= 0.5):
                result.violations += 1
                result.add_example(f"{row['symbol']} ex {row['ex_date']}: {amount:g}"
                                   + (f" = {ratio:.0%} of the previous close" if ratio is not None else ""))
    result.status = WARN if result.violations else PASS
    return result


@check("V-ADJ-RECON", FAIL, "Our adjClose vs FMP dividend-adjusted on the reconciliation sample")
def adj_recon(ctx: ValidationContext) -> CheckResult:
    """Median relative difference per sampled symbol; FAIL if > 5% of symbols exceed 1%, WARN above 0.1%."""
    result = CheckResult("V-ADJ-RECON", PASS)
    path = ctx.vintage_dir / RECON_TABLE
    if not path.exists():
        result.status, result.message = WARN, f"{RECON_TABLE} missing: no reconciliation sample was downloaded"
        return result
    recon = pq.read_table(path).to_pandas()
    if "adjClose" not in recon.columns:
        result.status, result.message = WARN, "reconciliation sample has no adjClose column"
        return result
    adapter = FmpOhlcvAdapter(ctx.store)
    stats = {}
    for sid, group in _by_security(recon):
        ours = adapter.get_ohlcv(ctx.store.security(sid).symbol, "1d", EARLY, LATE, "total")
        if ours is None or ours.empty:
            continue
        theirs = group.set_index(pd.DatetimeIndex(pd.to_datetime(group["date"])))["adjClose"].astype(float)
        joined = pd.concat([ours["close"], theirs.rename("fmp")], axis=1, join="inner").dropna()
        if joined.empty:
            continue
        diff = (joined["close"] / joined["fmp"] - 1).abs()
        median, p99, worst = float(diff.median()), float(diff.quantile(0.99)), float(diff.max())
        symbol = str(group["symbol"].iloc[0])
        stats[symbol] = {"median": round(median, 6), "p99": round(p99, 6), "max": round(worst, 6), "days": len(diff)}
        result.examined += 1
        if median > 0.001:
            result.violations += 1
            result.add_example(f"{symbol}: median {median:.3%}, p99 {p99:.3%}, max {worst:.3%}")
    over_one = sum(1 for s in stats.values() if s["median"] > 0.01)
    if result.examined and over_one / result.examined > 0.05:
        result.status = FAIL
    elif result.violations:
        result.status = WARN
    result.message = f"{result.examined} symbols compared; {over_one} with a median difference above 1%"
    result.details = {"per_symbol": dict(sorted(stats.items(), key=lambda kv: -kv[1]["median"])[:50])}
    return result


@check("V-YF-XCHECK", WARN, "Our total-adjusted close vs the legacy yfinance cache (Finding A: total, not split)")
def yf_xcheck(ctx: ValidationContext) -> CheckResult:
    """Share of overlapping days with a relative difference above 0.5%, per sampled symbol."""
    result = CheckResult("V-YF-XCHECK", PASS)
    if ctx.legacy_cache_dir is None:
        result.status, result.message = INFO, "no legacy cache configured (--cache-dir)"
        return result
    from src.data.cache.unified_cache import UnifiedCache

    cache = UnifiedCache(str(ctx.legacy_cache_dir))
    adapter = FmpOhlcvAdapter(ctx.store)
    shares = {}
    for sid in ctx.sample:
        info = ctx.store.security(sid)
        legacy = cache.get(info.yf_symbol or info.symbol, "1d", EARLY, LATE)
        ours = adapter.get_ohlcv(info.symbol, "1d", EARLY, LATE, "total")
        if legacy is None or legacy.empty or ours is None or ours.empty:
            continue
        legacy_index = pd.DatetimeIndex(legacy.index).normalize()
        joined = pd.concat([ours["close"], pd.Series(legacy["close"].to_numpy(dtype=float), index=legacy_index,
                                                     name="legacy")], axis=1, join="inner").dropna()
        if joined.empty:
            continue
        result.examined += 1
        share = float(((joined["close"] / joined["legacy"] - 1).abs() > 0.005).mean())
        shares[info.symbol] = round(share, 4)
        if share > 0.05:
            result.violations += 1
            result.add_example(f"{info.symbol}: {share:.1%} of {len(joined)} days differ by more than 0.5%")
    result.status = WARN if result.violations else PASS
    result.details = {"share_of_days_off": dict(sorted(shares.items(), key=lambda kv: -kv[1])[:50])}
    result.message = (f"{result.examined} sampled symbols overlap the legacy cache. A constant offset is expected: "
                      "legacy prices are adjusted as of their download date (Finding A).")
    return result

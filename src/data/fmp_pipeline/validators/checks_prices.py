"""Price-table checks (spec §11): V-SCHEMA, V-DUP, V-ORDER, V-OHLC, V-CAL, V-FLAGS, V-INTRADAY-ALIGN, V-INTRADAY-DAILY."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.fmp_pipeline.builders.builder import HISTORY, HISTORY_SCHEMA, SECURITIES_SCHEMA
from src.data.fmp_pipeline.builders.events import DIVIDENDS, DIVIDENDS_SCHEMA, SPLITS, SPLITS_SCHEMA
from src.data.fmp_pipeline.builders.validate import SECURITIES, validate_structure
from src.data.fmp_pipeline.validators.framework import FAIL, INFO, PASS, WARN, CheckResult, ValidationContext, check
from src.data.fmp_pipeline.validators.scan import SeriesStats

TIER_STRICT = ("A", "B")
DECLARED = {SECURITIES: SECURITIES_SCHEMA, HISTORY: HISTORY_SCHEMA, SPLITS.path: SPLITS_SCHEMA,
            DIVIDENDS.path: DIVIDENDS_SCHEMA}
#: Columns that must never be null, per table.
KEY_COLUMNS = {SECURITIES: ("security_id", "symbol"), HISTORY: ("security_id", "symbol"), SPLITS.path: ("date",),
               DIVIDENDS.path: ("ex_date",)}


@check("V-SCHEMA", FAIL, "Tables match their declared schemas; no nulls in key columns")
def schema(ctx: ValidationContext) -> CheckResult:
    """Every recorded output opens with its row count; declared tables keep their schema; keys are non-null."""
    outputs = {rel: int(o["rows"]) for rel, o in ctx.store.build_info.get("outputs", {}).items()}
    result = CheckResult("V-SCHEMA", PASS, examined=len(outputs))
    for error in validate_structure(ctx.vintage_dir, outputs) if outputs else ["build_info.json lists no outputs"]:
        result.violations += 1
        result.add_example(error)
    for rel, declared in DECLARED.items():
        path = ctx.vintage_dir / rel
        if not path.exists():
            result.violations += 1
            result.add_example(f"{rel}: missing")
            continue
        table = pq.read_table(path)
        if not table.schema.equals(declared):
            result.violations += 1
            result.add_example(f"{rel}: schema differs from the declared one")
        for column in KEY_COLUMNS[rel]:
            if table.column(column).null_count:
                result.violations += 1
                result.add_example(f"{rel}: {table.column(column).null_count} null {column}")
    result.status = FAIL if result.violations else PASS
    result.message = f"{len(outputs):,} files and {len(DECLARED)} declared tables checked"
    return result


def _series(ctx: ValidationContext) -> list[SeriesStats]:
    return list(ctx.scan.daily.values()) + ctx.scan.intraday_series()


@check("V-DUP", FAIL, "No duplicate (security_id, date) / (security_id, ts_utc)")
def duplicates(ctx: ValidationContext) -> CheckResult:
    """Count duplicate time keys inside each price file."""
    result = CheckResult("V-DUP", PASS)
    for stats in _series(ctx):
        result.examined += stats.rows
        if stats.duplicates:
            result.violations += stats.duplicates
            result.add_example(f"{stats.symbol} ({stats.kind}): {stats.duplicates} duplicate keys")
    result.status = FAIL if result.violations else PASS
    return result


@check("V-ORDER", FAIL, "Timestamps strictly increasing within each security")
def order(ctx: ValidationContext) -> CheckResult:
    """Count non-increasing steps inside each price file."""
    result = CheckResult("V-ORDER", PASS)
    for stats in _series(ctx):
        result.examined += stats.rows
        if stats.order_violations or stats.null_keys or stats.foreign_rows:
            result.violations += stats.order_violations + stats.null_keys + stats.foreign_rows
            result.add_example(f"{stats.symbol} ({stats.kind}): {stats.order_violations} out of order, "
                               f"{stats.null_keys} null keys, {stats.foreign_rows} rows of another security")
    result.status = FAIL if result.violations else PASS
    return result


@check("V-OHLC", FAIL, "OHLC consistency, prices > 0, volume >= 0 (WARN > 0.1%, FAIL > 5% of a symbol's bars)")
def ohlc(ctx: ValidationContext) -> CheckResult:
    """Per symbol, the share of bars that break OHLC consistency."""
    result = CheckResult("V-OHLC", PASS)
    worst = PASS
    for stats in _series(ctx):
        result.examined += stats.rows
        if not stats.rows or not stats.ohlc_violations:
            continue
        result.violations += stats.ohlc_violations
        rate = stats.ohlc_violations / stats.rows
        level = FAIL if rate > 0.05 else WARN if rate > 0.001 else PASS
        if level != PASS:
            worst = FAIL if FAIL in (worst, level) else WARN
            result.add_example(f"{stats.symbol} ({stats.kind}): {stats.ohlc_violations}/{stats.rows} = {rate:.2%}")
    result.status = worst
    return result


@check("V-CAL", FAIL, "Daily bars on expected NYSE sessions (WARN > 1% missing; FAIL > 10% for tier A/B)")
def calendar(ctx: ValidationContext) -> CheckResult:
    """Missing-session ratio per security between its first and last bar."""
    result = CheckResult("V-CAL", PASS)
    worst = PASS
    ratios = {}
    for stats in ctx.scan.daily.values():
        if not stats.expected_days:
            continue
        result.examined += 1
        rate = stats.missing_days / stats.expected_days
        ratios[stats.symbol] = round(rate, 4)
        if rate > 0.01:
            result.violations += 1
            strict = stats.tier in TIER_STRICT and rate > 0.10
            worst = FAIL if strict or worst == FAIL else WARN
            result.add_example(f"{stats.symbol} (tier {stats.tier}): {stats.missing_days}/{stats.expected_days} "
                               f"sessions missing ({rate:.1%})")
    result.status = worst
    result.details = {"worst": dict(sorted(ratios.items(), key=lambda kv: -kv[1])[:20])}
    return result


@check("V-FLAGS", WARN, "Flag rates per tier, top flagged symbols; WARN > 0.5% for tier A/B")
def flags(ctx: ValidationContext) -> CheckResult:
    """Spec §7.5 reporting: flag rate per tier and the 50 most flagged symbols."""
    result = CheckResult("V-FLAGS", INFO)
    per_tier: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    shares = []
    for stats in ctx.scan.daily.values():
        result.examined += stats.rows
        per_tier[stats.tier][0] += stats.flagged
        per_tier[stats.tier][1] += stats.rows
        if stats.rows:
            shares.append((stats.flagged / stats.rows, stats.symbol, stats.tier))
        if stats.tier in TIER_STRICT and stats.rows and stats.flagged / stats.rows > 0.005:
            result.violations += 1
            counts = {k: v for k, v in stats.flag_counts.items() if v}
            result.add_example(f"{stats.symbol} (tier {stats.tier}): {stats.flagged}/{stats.rows} bars flagged {counts}")
    candidates = ctx.report_dir / "missing_splits_candidates.csv"
    result.details = {
        "flag_rate_by_tier": {t: round(f / n, 5) if n else 0.0 for t, (f, n) in sorted(per_tier.items())},
        "top_flagged": [f"{s} ({t}): {share:.2%}" for share, s, t in sorted(shares, reverse=True)[:50] if share],
        "missing_splits_candidates.csv": candidates.exists(),
    }
    if result.violations or not candidates.exists():
        result.status = WARN
    result.message = "" if candidates.exists() else "missing_splits_candidates.csv not found"
    return result


@check("V-INTRADAY-ALIGN", FAIL, "Intraday bars: a full session per regular day (fewer on early closes), first bar "
       "09:30 ET, none outside; FAIL for tier A/B above 1% of days")
def intraday_align(ctx: ValidationContext) -> CheckResult:
    """Session alignment of every intraday series, every interval (FAIL for tier A/B, WARN otherwise)."""
    result = CheckResult("V-INTRADAY-ALIGN", PASS)
    if not ctx.scan.intraday:
        result.status, result.message = INFO, "no intraday data in the vintage"
        return result
    worst = PASS
    per_timeframe: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for stats in sorted(ctx.scan.intraday_series(), key=lambda st: (st.tier, st.symbol, st.kind)):
        result.examined += stats.days
        per_timeframe[stats.kind][1] += stats.days
        bad_days = max(stats.bad_first_bar_days, stats.bad_bar_count_days)
        per_timeframe[stats.kind][0] += bad_days
        if not bad_days and not stats.outside_session_bars:
            continue
        result.violations += stats.bad_first_bar_days + stats.bad_bar_count_days + stats.outside_session_bars
        strict = stats.tier in TIER_STRICT and stats.days and bad_days / stats.days > 0.01
        worst = FAIL if strict or worst == FAIL else WARN
        result.add_example(f"{stats.symbol} ({stats.kind}, tier {stats.tier}): {stats.bad_first_bar_days} days not "
                           f"starting 09:30 ET, {stats.bad_bar_count_days} days with an unexpected bar count, "
                           f"{stats.outside_session_bars} bars outside the session")
    result.status = worst
    result.details = {"share_of_days_off_by_timeframe": {tf: round(bad / days, 4) if days else 0.0
                                                         for tf, (bad, days) in sorted(per_timeframe.items())}}
    result.message = ", ".join(f"{tf}: {len(by_sid):,} series" for tf, by_sid in sorted(ctx.scan.intraday.items()))
    return result


@check("V-INTRADAY-DAILY", WARN, "Intraday aggregated to days vs daily raw: OHLC within 0.5%, volume within 5%")
def intraday_daily(ctx: ValidationContext) -> CheckResult:
    """Compare intraday bars rolled up per exchange-local day with the daily raw series (sample, every interval)."""
    result = CheckResult("V-INTRADAY-DAILY", PASS)
    if not ctx.scan.intraday:
        result.status, result.message = INFO, "no intraday data in the vintage"
        return result
    volume_ratios: dict[str, list[float]] = defaultdict(list)
    for timeframe, by_sid in sorted(ctx.scan.intraday.items()):
        sids = [sid for sid in ctx.sample if sid in by_sid] or list(by_sid)[: ctx.sample_size]
        for sid in sids:
            bars = pq.read_table(ctx.store.price_path(sid, timeframe)).to_pandas()  # type: ignore[arg-type]
            daily = ctx.store.prices(sid, "1d")
            if daily is None or bars.empty:
                continue
            rolled = bars.sort_values("ts_utc").groupby("date_et").agg(
                open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                volume=("volume", "sum"))
            rolled.index = pd.DatetimeIndex(pd.to_datetime(rolled.index))
            joined = rolled.join(daily[["open", "high", "low", "close", "volume"]], rsuffix="_d", how="inner")
            if joined.empty:
                continue
            result.examined += len(joined)
            price_bad = np.zeros(len(joined), dtype=bool)
            for column in ("open", "high", "low", "close"):
                price_bad |= (joined[column] / joined[f"{column}_d"] - 1).abs().to_numpy() > 0.005
            ratio = (joined["volume"] / joined["volume_d"].replace(0, np.nan)).to_numpy(dtype=float)
            volume_ratios[timeframe] += [r for r in ratio if np.isfinite(r)]
            bad = int((price_bad | (np.abs(ratio - 1) > 0.05)).sum())
            if bad:
                result.violations += bad
                result.add_example(f"{ctx.store.security(sid).symbol} ({timeframe}): {bad}/{len(joined)} days differ")
    result.details = {"intraday_to_daily_volume_ratio": {
        tf: dict(zip(("p01", "median", "p99"), (round(float(q), 4) for q in np.quantile(r, [0.01, 0.5, 0.99]))))
        for tf, r in sorted(volume_ratios.items()) if r}}
    result.status = WARN if result.violations else PASS
    return result

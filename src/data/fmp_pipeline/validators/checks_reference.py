"""Reference-data checks (spec §11): V-DELIST, V-UNIVERSE, V-IDENT, V-INDEX, V-FUND-PIT, V-FUND-COV."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta
from typing import Any, Optional

import pandas as pd
import pyarrow.parquet as pq

from src.data.fmp_pipeline.builders.identity import normalize_name
from src.data.fmp_pipeline.datamanager_adapter.fmp_store import AmbiguousSymbol
from src.data.fmp_pipeline.validators.framework import FAIL, INFO, PASS, WARN, CheckResult, ValidationContext, check

#: (symbol, expected delisting year, soft): spec §11 V-DELIST.
KNOWN_DELISTED = (("TWTR", 2022, False), ("SIVB", 2023, False), ("BBBY", 2023, False), ("ATVI", 2023, False),
                  ("LEH", 2008, True))
INDEX_DATES = (date(2008, 1, 2), date(2012, 1, 3), date(2016, 1, 4), date(2020, 1, 2), date(2024, 1, 2))
STATEMENTS = ("fundamentals/income_statement.parquet", "fundamentals/balance_sheet.parquet",
              "fundamentals/cash_flow.parquet")


def _table(ctx: ValidationContext, rel: str) -> Optional[pd.DataFrame]:
    path = ctx.vintage_dir / rel
    return pq.read_table(path).to_pandas() if path.exists() else None


def _extra(row: Any, key: str) -> Any:
    raw = row.get("_extra") if hasattr(row, "get") else None
    if not isinstance(raw, str):
        return None
    return json.loads(raw).get(key)


@check("V-DELIST", WARN, "Known delisted symbols present with a delisting date and bars up to about then")
def delist(ctx: ValidationContext) -> CheckResult:
    """Coverage of a few well-known delistings (findings about FMP coverage, not bugs)."""
    result = CheckResult("V-DELIST", PASS, examined=len(KNOWN_DELISTED))
    securities = ctx.scan.securities
    for symbol, year, soft in KNOWN_DELISTED:
        rows = securities[securities["symbol"] == symbol]
        delisted = rows[rows["delisted_date"].notna()]
        problem = None
        if delisted.empty:
            problem = "not in securities" if rows.empty else "no delisting date"
        else:
            row = delisted.iloc[-1]
            when, last = row["delisted_date"], row["last_price_date"]
            if when.year != year:
                problem = f"delisted {when}, expected {year}"
            elif last is None or pd.isna(last) or (when - last).days > 60:
                problem = f"last bar {last}, delisted {when}"
        if problem:
            result.violations += 1
            result.add_example(f"{symbol}: {problem}" + (" (soft check)" if soft else ""))
    result.status = WARN if result.violations else PASS
    return result


@check("V-UNIVERSE", INFO, "Universe counts by exchange and type; share without daily bars; list overlaps")
def universe(ctx: ValidationContext) -> CheckResult:
    """Counts for the report; WARN if more than 10% of securities have no daily bar."""
    securities = ctx.scan.securities
    result = CheckResult("V-UNIVERSE", INFO, examined=len(securities))
    no_bars = int(securities["first_price_date"].isna().sum())
    share = no_bars / len(securities) if len(securities) else 0.0
    combos = Counter(",".join(sorted(src for src in str(f).split(",")
                                     if src in ("stock-list", "actively-trading-list", "delisted-companies")))
                     or "other" for f in securities["source_flags"])
    result.details = {
        "by_exchange": dict(Counter(securities["exchange"].fillna("?")).most_common()),
        "by_instrument_type": dict(Counter(securities["instrument_type"].fillna("?")).most_common()),
        "by_tier": dict(sorted(Counter(securities["tier"].fillna("?")).items())),
        "without_daily_bars": no_bars,
        "list_overlaps": dict(combos.most_common()),
    }
    result.violations = no_bars
    if share > 0.10:
        result.status = WARN
    result.message = f"{len(securities):,} securities; {no_bars:,} ({share:.1%}) without daily bars"
    return result


@check("V-IDENT", WARN, "Identity risks: gaps > 180 days, name mismatches, IPO after first bar, identity_uncertain")
def ident(ctx: ValidationContext) -> CheckResult:
    """Spec §7.3 rule 3: flag and report, no repair."""
    securities = ctx.scan.securities.set_index("security_id")
    result = CheckResult("V-IDENT", PASS, examined=len(securities))
    counts: Counter[str] = Counter()
    for sid, stats in ctx.scan.daily.items():
        if stats.max_gap_days > 180:
            counts["gap>180d"] += 1
            result.add_example(f"{stats.symbol}: {stats.max_gap_days}-day gap after {stats.max_gap_at}")
        ipo = securities.at[sid, "ipo_date"] if sid in securities.index else None
        if ipo is not None and not pd.isna(ipo) and stats.first is not None and ipo > stats.first:
            counts["ipo_after_first_bar"] += 1
            result.add_example(f"{stats.symbol}: IPO {ipo} after first bar {stats.first}")
    uncertain = securities[securities["identity_uncertain"].fillna(False)]
    counts["identity_uncertain"] = len(uncertain)
    for _, row in uncertain.head(5).iterrows():
        result.add_example(f"{row['symbol']}: {row['uncertain_reason']}")
    delisted, profiles = _table(ctx, "reference/delisted.parquet"), _table(ctx, "reference/profiles.parquet")
    if delisted is not None and profiles is not None:
        names = {s: n for s, n in zip(profiles["symbol"], profiles["name"]) if isinstance(n, str)}
        for _, row in delisted.iterrows():
            profile_name, delisted_name = names.get(row["symbol"]), row["name"]
            if (profile_name and isinstance(delisted_name, str) and delisted_name
                    and normalize_name(profile_name) != normalize_name(delisted_name)):
                counts["name_mismatch"] += 1
                result.add_example(f"{row['symbol']}: delisted as {row['name']!r}, profile {profile_name!r}")
    result.violations = sum(counts.values())
    result.details = dict(counts)
    result.status = WARN if result.violations else PASS
    return result


def _members_as_of(current: set[str], changes: pd.DataFrame, as_of: date) -> set[str]:
    """Roll the change log backwards from the current list to ``as_of``."""
    members = set(current)
    for _, row in changes.sort_values("date", ascending=False).iterrows():
        if row["date"] is None or pd.isna(row["date"]) or row["date"] <= as_of:
            break
        added, removed = row.get("symbol"), _extra(row, "removedTicker")
        if isinstance(added, str) and added:
            members.discard(added)
        if isinstance(removed, str) and removed:
            members.add(removed)
    return members


@check("V-INDEX", FAIL, "S&P 500 rebuilt as of five dates: 500–505 members, all known, ≥ 97% priced that day")
def index(ctx: ValidationContext) -> CheckResult:
    """Roll the S&P 500 change log backwards and check size, identity and price coverage."""
    result = CheckResult("V-INDEX", PASS)
    changes, current = _table(ctx, "index_membership/index_changes.parquet"), \
        _table(ctx, "index_membership/index_members_current.parquet")
    if changes is None or current is None:
        result.status, result.message = WARN, "index membership tables missing"
        return result
    changes = changes[changes["index"] == "sp500"]
    members_now = set(current.loc[current["index"] == "sp500", "symbol"].dropna())
    if not members_now:
        result.status, result.message = WARN, "no current S&P 500 list to roll back from"
        return result
    worst = PASS
    sizes = {}
    for as_of in INDEX_DATES:
        members = _members_as_of(members_now, changes, as_of)
        result.examined += len(members)
        sizes[str(as_of)] = len(members)
        if not 500 <= len(members) <= 505:
            worst = WARN if worst == PASS else worst
            result.add_example(f"{as_of}: {len(members)} members")
        known = priced = 0
        for symbol in members:
            try:
                info = ctx.store.resolve(symbol, as_of)
            except AmbiguousSymbol:
                info = None
            if info is None:
                result.violations += 1
                continue
            known += 1
            prices = ctx.store.prices(info.security_id, "1d")
            if prices is not None:
                window = prices.index[(prices.index <= pd.Timestamp(as_of))
                                      & (prices.index >= pd.Timestamp(as_of - timedelta(days=5)))]
                priced += int(len(window) > 0)
        share = priced / len(members) if members else 0.0
        if share < 0.97:
            worst = FAIL
            result.add_example(f"{as_of}: {known}/{len(members)} known, {share:.1%} priced")
    result.status = worst
    result.details = {"sizes": sizes}
    return result


@check("V-FUND-PIT", WARN, "Statements: filing/accepted date on ≥ 95% of rows and on or after the period end")
def fund_pit(ctx: ValidationContext) -> CheckResult:
    """Point-in-time usability of the statement tables (spec §5.4a)."""
    result = CheckResult("V-FUND-PIT", PASS)
    present = {}
    for rel in STATEMENTS:
        table = _table(ctx, rel)
        if table is None or table.empty:
            continue
        rows = len(table)
        result.examined += rows
        has = table["filing_date"].notna() | table["accepted_date"].notna()
        present[rel] = round(float(has.mean()), 4)
        early = table[table["filing_date"].notna() & table["date"].notna() & (table["filing_date"] < table["date"])]
        result.violations += int((~has).sum()) + len(early)
        if has.mean() < 0.95:
            result.add_example(f"{rel}: only {has.mean():.1%} of rows carry a filing or accepted date")
        for _, row in early.head(5).iterrows():
            result.add_example(f"{rel}: {row['symbol']} period {row['date']} filed {row['filing_date']}")
    if not present:
        result.status, result.message = INFO, "no statement tables in the vintage"
        return result
    result.details = {"share_with_dates": present}
    result.status = WARN if any(v < 0.95 for v in present.values()) or result.violations else PASS
    return result


@check("V-FUND-COV", INFO, "Share of common stocks with at least one annual statement")
def fund_cov(ctx: ValidationContext) -> CheckResult:
    """Coverage of the income statement against the stock universe and FMP's own symbol list."""
    securities = ctx.scan.securities
    stocks = set(securities.loc[(securities["instrument_type"] == "stock")
                                & securities["first_price_date"].notna(), "symbol"])
    result = CheckResult("V-FUND-COV", INFO, examined=len(stocks))
    income = _table(ctx, STATEMENTS[0])
    listed = _table(ctx, "reference/statement_symbols.parquet")
    with_annual = set(income.loc[income["period"] == "FY", "symbol"]) if income is not None else set()
    covered = len(stocks & with_annual)
    result.violations = len(stocks) - covered
    result.details = {"stocks": len(stocks), "with_annual_statement": covered,
                      "in_financial_statement_symbol_list": len(stocks & set(listed["symbol"]))
                      if listed is not None else None}
    result.message = f"{covered:,}/{len(stocks):,} priced common stocks have an annual statement" \
        + (f" ({covered / len(stocks):.1%})" if stocks else "")
    return result

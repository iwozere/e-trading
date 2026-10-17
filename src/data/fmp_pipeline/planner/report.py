"""Markdown reports of ``fmp plan``: ``budget_plan.md`` (spec §6.3.2) and ``universe_report.md`` (spec §7.2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

from src.data.fmp_pipeline.planner.catalog import BLOCKS
from src.data.fmp_pipeline.planner.config import IntradayConfig, PlanConfig
from src.data.fmp_pipeline.planner.projection import GB, Bucket, Projection
from src.data.fmp_pipeline.planner.universe import RULES, TIERS, Universe, normalize_exchange

RULE_TEXT = {
    "otc": "OTC / pink-sheet venue (`include_otc: false`)",
    "exchange": "exchange not in `include_exchanges` (non-US, crypto, forex, …)",
    "fund": "mutual fund",
    "no_profile": "exchange unknown and the profile returned nothing (`keep_unresolved: false`)",
}


@dataclass(frozen=True)
class PlanContext:
    """Facts about the run that the budget report prints alongside the projection."""

    dump_id: str
    as_of: date
    dry_run: bool
    added: int
    used_wire: int
    used_decoded: int
    free_bytes: Optional[int]
    need_bytes: int


def _gb(value: float) -> str:
    return f"{value / GB:,.2f}"


def _row(label: str, b: Bucket) -> str:
    flag = " *" if b.assumed and b.remaining else ""
    return (f"| {label} | {b.done:,} | {b.planned:,} | {b.deferred:,} | {_gb(b.done_wire)} | {_gb(b.done_decoded)} "
            f"| {_gb(b.wire)}{flag} | {_gb(b.decoded)}{flag} |")


_HEADER = ("| {0} | Done | Planned | Deferred | Done wire GB | Done decoded GB | Remaining wire GB | "
           "Remaining decoded GB |\n|---|---|---|---|---|---|---|---|")


def _intraday_line(intraday: IntradayConfig) -> str:
    """The budget plan's input line for one intraday interval and the tiers it covers."""
    return (f"- Intraday `{intraday.interval}`: row cap "
            + (f"**{intraday.max_rows}** (configured)" if intraday.max_rows is not None
               else f"**unknown** — windows sized with an assumed {intraday.assumed_max_rows} rows; no task of this "
                    f"interval is written until the paid probe measures it")
            + f", {intraday.session_bars} bars/day, {intraday.decoded_bytes_per_row:g} B/row decoded and "
              f"{intraday.wire_bytes_per_row:g} B/row wire (assumed). Tiers: {', '.join(intraday.tiers)}.")


def render_budget_plan(projection: Projection, universe: Universe, config: PlanConfig, ctx: PlanContext) -> str:
    """Return ``budget_plan.md``."""
    budget = config.bandwidth.limit_bytes
    lines = [
        f"# FMP budget plan — dump `{ctx.dump_id}`",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `fmp plan{' --dry-run' if ctx.dry_run else ''}`."
        f" As-of date **{ctx.as_of}** (fixed for the dump). "
        + ("Dry run: nothing was added to the manifest." if ctx.dry_run
           else f"**{ctx.added:,}** new tasks added to the manifest."),
        "",
        "## Inputs",
        "",
        f"- Universe: **{len(universe.resolved):,}** securities resolved, **{len(universe.unresolved):,}** candidates "
        f"waiting for their profile, **{sum(universe.excluded.values()):,}** excluded (see `universe_report.md`).",
        f"- Priorities: {', '.join(config.priorities)}. Daily history from {config.daily_from}, "
        f"{config.daily_window_years}-year windows. Page size {config.page_size}.",
        *[_intraday_line(config.intraday.for_interval(i)) for i in config.intraday.intervals],
        f"- Bandwidth cap: {config.bandwidth.cap_gb:g} GB per {config.bandwidth.window_days} days, hard stop at "
        f"{config.bandwidth.safety:.0%} = **{_gb(budget)} GB**, counted on `{config.bandwidth.basis}` bytes"
        + (f" (FMP counts {config.bandwidth.scale:g} bytes per local byte)" if config.bandwidth.scale != 1 else "")
        + ". Used in "
        f"the current window: {_gb(ctx.used_wire)} GB wire, {_gb(ctx.used_decoded)} GB decoded.",
        "",
        "## Projection per block (spec §6.3.5 order)",
        "",
        "Remaining = planned + deferred tasks at their estimated size. `*` marks totals that rest at least partly on "
        "assumed sizes (see the catalogue in `planner/catalog.py`); re-running `fmp plan` after a run replaces "
        "estimates with measured bytes.",
        "",
        _HEADER.format("Block"),
        _row("bootstrap lists", projection.bootstrap),
    ]
    lines += [_row(f"{n} {BLOCKS.get(n, '')}", b) for n, b in sorted(projection.blocks.items())]
    total = Bucket()
    for b in [projection.bootstrap, *projection.blocks.values()]:
        total.done += b.done
        total.done_wire += b.done_wire
        total.done_decoded += b.done_decoded
        total.planned += b.planned
        total.deferred += b.deferred
        total.wire += b.wire
        total.decoded += b.decoded
        total.assumed = total.assumed or b.assumed
    lines.append(_row("**total**", total))
    lines += ["", "## Intraday per tier (spec §6.3.3)", "", _HEADER.format("Tier")]
    lines += [_row(t, projection.tiers.get(t, Bucket())) for t in TIERS]
    lines += ["", "## Schedule", "",
              f"Month in which each step finishes if every month uses the full {_gb(budget)} GB "
              "(month 1 minus what the window already holds).", "",
              "| Step | Month if the cap counts wire bytes | Month if it counts decoded bytes |", "|---|---|---|"]
    lines += [f"| {label} | {wire} | {decoded} |"
              for label, wire, decoded in projection.schedule(budget, ctx.used_wire, ctx.used_decoded)]
    free = "unknown (data root's drive not found)" if ctx.free_bytes is None else f"{_gb(ctx.free_bytes)} GB"
    verdict = "" if ctx.free_bytes is None else (
        " — **enough**" if ctx.free_bytes >= ctx.need_bytes * 1.2 else " — **NOT enough (need + 20% margin)**")
    lines += ["", "## Disk", "",
              f"Remaining archive growth (wire × {config.archive_overhead:g}) plus vintage ({config.vintage_gb:g} GB) "
              f"and build temp ({config.build_temp_gb:g} GB), spec §6.8: **{_gb(ctx.need_bytes)} GB**. "
              f"Free on the data volume: **{free}**{verdict}.", ""]
    if projection.deferred_reasons:
        lines += ["## Deferred tasks", "", "Counted above, not yet in the manifest. Re-run `fmp plan` once the "
                  "reason is gone.", "", "| Reason | Tasks |", "|---|---|"]
        lines += [f"| {reason} | {n:,} |" for reason, n in sorted(projection.deferred_reasons.items())]
        lines.append("")
    return "\n".join(lines)


def render_universe_report(universe: Universe, include_exchanges: tuple[str, ...], dump_id: str) -> str:
    """Return ``universe_report.md``: sources, exchanges seen, exclusions by rule and tiers."""
    included = {normalize_exchange(x) for x in include_exchanges}
    lines = [
        f"# FMP universe — dump `{dump_id}`",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. As-of {universe.as_of}.",
        "",
        "## Sources",
        "",
        "| Endpoint | Status | Rows |",
        "|---|---|---|",
    ]
    lines += [f"| `{ep}` | {universe.source_status.get(ep, 'not fetched')} | {universe.source_rows.get(ep, 0):,} |"
              for ep in sorted(set(universe.source_status) | set(universe.source_rows))]
    lines += ["", "## Exchanges seen", "", "Normalized codes (upper case, no spaces/dashes). Edit "
              "`universe.include_exchanges` in `config/fmp/fmp.yaml` if a US venue is missing.", "",
              "| Exchange | Candidates | Included |", "|---|---|---|"]
    lines += [f"| {ex} | {n:,} | {'yes' if ex in included else 'no'} |" for ex, n in universe.exchanges.most_common()]
    lines += ["", "## Exclusions", "", "| Rule | Removed | Examples |", "|---|---|---|"]
    lines += [f"| {RULE_TEXT[r]} | {universe.excluded.get(r, 0):,} | "
              f"{', '.join(universe.excluded_examples.get(r, []))} |" for r in RULES]
    tiers = universe.tier_counts()
    lines += ["", "## Included", "",
              f"**{len(universe.resolved):,}** resolved, **{len(universe.unresolved):,}** waiting for their profile "
              "(only `profile` is planned for those; the next `fmp plan` decides).", "",
              "| Tier | Securities |", "|---|---|"]
    lines += [f"| {t} | {tiers.get(t, 0):,} |" for t in TIERS]
    statement = ("not fetched — fundamentals cover all stocks" if universe.statement_symbols is None
                 else f"{len(universe.statement_symbols):,} symbols")
    lines += ["", f"`financial-statement-symbol-list`: {statement}.", ""]
    return "\n".join(lines)

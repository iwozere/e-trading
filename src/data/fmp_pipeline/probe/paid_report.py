"""
``docs/fmp_paid_probe_report.md`` (spec §4.4, gate G2): what the paid probe measured, which settings ``fmp plan``
will now use, and the answer to every "Requires paid probe" item of the free-key audit. Counts, dates, field
names and verdicts only: no prices or response bodies (D5).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from src.data.fmp_pipeline.probe.paid_probe import PaidFindings, PaidRun

MARKER = "<!-- AUTO-GENERATED BELOW by `fmp probe-paid`; everything after this line is overwritten -->"
TITLE = "FMP paid probe report (Phase 2)"
MB = 1024**2


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def _answers(f: PaidFindings) -> list[tuple[str, str]]:
    """One answer per "Requires paid probe" item of ``docs/fmp_probe_report.md``."""
    c, cap = f.conventions, f.cap
    served = [i for i, s in f.intervals.items() if s == "ok"]
    cap_text = ("no cap seen; largest full response " + str(cap.largest_full) + " rows" if cap and not cap.capped
                else f"{cap.max_rows} rows, keeps the {cap.keeps or 'unclear'} end" if cap else "not measured")
    conv_text = (f"`date` is {c.timezone} time labelling the bar {c.label}; {c.bars_per_day} bars per regular day; "
                 f"extended hours {_yes(c.extended_hours)}" if c and c.conclusive else
                 "inconclusive: no single convention puts every first bar at 09:30 ET" if c else "not measured")
    rows = [
        ("§3.1 / A1 intraday intervals", ", ".join(served) or "none"),
        ("§3.2 / A1 depth of `1hour`", f"earliest month with bars: {f.earliest_hourly}" if f.earliest_hourly
         else "; ".join(n for n in f.notes if "hourly" in n) or "not measured"),
        ("§3.3 row cap / maximum window", cap_text),
        ("§3.4 / A2 bandwidth basis", "measured by `fmp calibrate-usage` (dashboard vs local counters)"),
        ("§3.5 / A4 delisted symbols in prices", ", ".join(f"{s}: {v}" for s, v in f.delisted.items())),
        ("§3.5 / A5 constituent endpoints", ", ".join(f"{e}: {v['status']} ({v['rows']} rows)"
                                                      for e, v in f.lists.items() if "constituent" in e)),
        ("§5.3 / A5 index prices", ", ".join(f"{s}: {f.coverage.get(s, '?')}"
                                             for s in ("^GSPC", "^IXIC", "^DJI", "^VIX", "^RUT"))),
        ("§3.6 intraday split adjustment", f.intraday_split or "not measured"),
        ("§3.7 / A9 / A11 label, time zone, extended hours, DST", conv_text),
        ("A6 calendar `from`/`to` maximum window",
         ", ".join(f"{e}: {d} days" if d else f"{e}: none complete" for e, d in f.calendar_windows.items())),
        ("A7 class-share symbol", f"BRK-B: {f.coverage.get('BRK-B')}, BRK.B: {f.coverage.get('BRK.B')}"),
        ("A7 exchange codes (stock-list, available-exchanges)", ", ".join(list(f.exchanges)[:40]) or "none"),
        ("A10 ticker reuse", "; ".join(f"{s} — {v}" for s, v in f.reuse.items()) or "no candidates found"),
        ("§5 page sizes", ", ".join(f"{e}: cap {n}" for e, n in f.page_sizes.items())
         + (" | full pages up to: " + ", ".join(f"{e}: {n}" for e, n in f.page_full.items()) if f.page_full else "")),
        ("§5.4a market-cap date range", f"complete for a {f.market_cap_years}-year window" if f.market_cap_years
         else "no window was complete"),
        ("economic-indicators names", ", ".join(f"{n}: {r}" for n, r in f.indicators.items())),
        ("Coverage (US/UK/Canada, unknown symbol)", ", ".join(f"{s}: {v}" for s, v in f.coverage.items())),
        ("Rate-limit headers", json.dumps(f.rate_headers) if f.rate_headers else "none"),
    ]
    return rows


def render_paid_report(run: PaidRun, calls_used: int, budget: int) -> str:
    """Render the generated part of the report (starting with ``MARKER``)."""
    f = run.findings
    live = [r.response for r in run.results if r.response is not None]
    wire = sum(r.bytes_wire for r in live)
    decoded = sum(r.bytes_decoded for r in live)
    lines = [MARKER, "",
             f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. Calls used: **{calls_used} / {budget}**; "
             f"this run's responses: {wire / MB:.1f} MB wire, {decoded / MB:.1f} MB decoded."
             + (f" **Stopped early: {run.stopped}**" if run.stopped else ""), "",
             "## Settings `fmp plan` and `fmp build` now use", "",
             "Written to `<data_root>/probe-paid/probe_results.json` and applied over `config/fmp/fmp.yaml` "
             "(set `fmp.use_probe_results: false` to ignore):", "", "```json",
             json.dumps(f.settings, indent=1, sort_keys=True), "```", "",
             "## Answers to the paid-probe items", "", "| Item | Finding |", "|---|---|"]
    lines += [f"| {item} | {answer.replace('|', '/')} |" for item, answer in _answers(f)]
    if f.cap_probes:
        lines += ["", "## Row-cap requests (AAPL `1hour`)", "", "| Window | Rows | Expected | First | Last |",
                  "|---|---|---|---|---|"]
        lines += [f"| {p.start} … {p.end} | {p.rows:,} | {p.expected:,} | {p.first} | {p.last} |"
                  for p in f.cap_probes]
    if f.conventions is not None:
        c = f.conventions
        lines += ["", "## Hourly conventions", "", f"Days examined: {c.days}. First bar per day (as interpreted): "
                  f"{json.dumps(c.first_bars)}. Early-close days: {json.dumps(c.early_close_bars)}."]
    lines += ["", "## Reference lists", "", "| Endpoint | Status | Rows | Fields |", "|---|---|---|---|"]
    lines += [f"| `{e}` | {v['status']} | {v['rows']:,} | {', '.join(v['fields'])} |" for e, v in f.lists.items()]
    if f.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in f.notes]
    lines += ["", "## Calls", "", "| Group | Label | HTTP | Class | Rows |", "|---|---|---|---|---|"]
    for result in run.results:
        response = result.response
        rows: Any = len(run.rows(result.call.label)) if response is not None else "—"
        lines.append(f"| {result.call.group} | `{result.call.label}` | {response.status if response else '—'} | "
                     f"{response.classification if response else result.skipped[:40]} | {rows} |")
    return "\n".join(lines) + "\n"

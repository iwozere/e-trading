"""
``docs/fmp_paid_probe_report.md`` (spec §4.4, gate G2): what the paid probe measured, which settings ``fmp plan``
will now use, and the answer to every "Requires paid probe" item of the free-key audit. Counts, dates, field
names and verdicts only: no prices or response bodies (D5).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from src.data.fmp_pipeline.probe.paid_probe import IntradayFindings, PaidFindings, PaidRun

MARKER = "<!-- AUTO-GENERATED BELOW by `fmp probe-paid`; everything after this line is overwritten -->"
TITLE = "FMP paid probe report (Phase 2)"
MB = 1024**2


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def _cap_text(m: IntradayFindings) -> str:
    cap = m.cap
    if cap is None:
        return "not measured"
    return (f"no cap seen; largest full response {cap.largest_full} rows" if not cap.capped
            else f"{cap.max_rows} rows, keeps the {cap.keeps or 'unclear'} end")


def _conventions_text(m: IntradayFindings) -> str:
    c = m.conventions
    if c is None:
        return "not measured"
    if not c.conclusive:
        return "inconclusive: no single convention puts every first bar at 09:30 ET"
    return (f"`date` is {c.timezone} time labelling the bar {c.label}; {c.bars_per_day} bars per regular day; "
            f"extended hours {_yes(c.extended_hours)}")


def _bytes_text(m: IntradayFindings) -> str:
    if not m.decoded_bytes_per_row:
        return "not measured"
    return f"{m.wire_bytes_per_row:g} B/row wire, {m.decoded_bytes_per_row:g} B/row decoded"


def _per_interval(f: PaidFindings, item: str, text: Any) -> list[tuple[str, str]]:
    """One answer row per measured interval (configured one first); ``not measured`` when none was."""
    if not f.intraday:
        return [(item, "not measured")]
    return [(f"{item} (`{i}`)", text(m)) for i, m in f.intraday.items()]


def _answers(f: PaidFindings) -> list[tuple[str, str]]:
    """One answer per "Requires paid probe" item of ``docs/fmp_probe_report.md``."""
    served = [i for i, s in f.intervals.items() if s == "ok"]

    def depth(m: IntradayFindings) -> str:
        return (f"earliest month with bars: {m.earliest}" if m.earliest
                else "; ".join(n for n in f.notes if n.startswith(f"{m.interval} ") or f" {m.interval} " in n)
                or "not measured")

    rows = [("§3.1 / A1 intraday intervals", ", ".join(served) or "none")]
    rows += _per_interval(f, "§3.2 / A1 intraday depth", depth)
    rows += _per_interval(f, "§3.3 row cap / maximum window", _cap_text)
    rows += _per_interval(f, "Intraday bytes per row", _bytes_text)
    rows += [
        ("§3.4 / A2 bandwidth basis", "measured by `fmp calibrate-usage` (dashboard vs local counters)"),
        ("§3.5 / A4 delisted symbols in prices", ", ".join(f"{s}: {v}" for s, v in f.delisted.items())),
        ("§3.5 / A5 constituent endpoints", ", ".join(f"{e}: {v['status']} ({v['rows']} rows)"
                                                      for e, v in f.lists.items() if "constituent" in e)),
        ("§5.3 / A5 index prices", ", ".join(f"{s}: {f.coverage.get(s, '?')}"
                                             for s in ("^GSPC", "^IXIC", "^DJI", "^VIX", "^RUT"))),
    ]
    rows += _per_interval(f, "§3.6 intraday split adjustment", lambda m: m.split or "not measured")
    rows += _per_interval(f, "§3.7 / A9 / A11 label, time zone, extended hours, DST", _conventions_text)
    rows += [
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


def _interval_sections(f: PaidFindings) -> list[str]:
    """Row-cap requests and conventions detail per measured interval."""
    lines: list[str] = []
    for interval, m in f.intraday.items():
        if m.cap_probes:
            lines += ["", f"## Row-cap requests (AAPL `{interval}`)", "", "| Window | Rows | Expected | First | Last |",
                      "|---|---|---|---|---|"]
            lines += [f"| {p.start} … {p.end} | {p.rows:,} | {p.expected:,} | {p.first} | {p.last} |"
                      for p in m.cap_probes]
        if m.conventions is not None:
            c = m.conventions
            lines += ["", f"## `{interval}` conventions", "", f"Days examined: {c.days}. First bar per day (as "
                      f"interpreted): {json.dumps(c.first_bars)}. Early-close days: {json.dumps(c.early_close_bars)}."]
    return lines


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
             f"## Settings `fmp plan` and `fmp build` now use (intraday `{f.interval}`)", "",
             "Written to `<data_root>/probe-paid/probe_results.json` and applied over `config/fmp/fmp.yaml` "
             "(set `fmp.use_probe_results: false` to ignore):", "", "```json",
             json.dumps(f.settings, indent=1, sort_keys=True), "```", ""]
    others = {i: v for i, v in f.settings_by_interval.items() if i != f.interval}
    if others:
        lines += ["Also measured; applied instead when `intraday.interval` is set to one of these:", "", "```json",
                  json.dumps(others, indent=1, sort_keys=True), "```", ""]
    lines += ["## Answers to the paid-probe items", "", "| Item | Finding |", "|---|---|"]
    lines += [f"| {item} | {answer.replace('|', '/')} |" for item, answer in _answers(f)]
    lines += _interval_sections(f)
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

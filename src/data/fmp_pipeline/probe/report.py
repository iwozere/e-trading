"""
Render ``docs/fmp_probe_report.md`` and the synthetic fixtures from a ``ProbeRun`` (plan WP-1 §4.3).

The report holds schemas, counts, byte sizes, ratios and error messages, never price rows or other response
bodies (D5). Anything above ``AUTO_MARKER`` in an existing report is hand-written interpretation and is kept on
regeneration.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.data.fmp_pipeline.probe.analysis import date_range, dividend_basis, split_check
from src.data.fmp_pipeline.probe.free_probe import (
    DAILY_PRICE_ENDPOINTS,
    DIVIDEND_FACTS,
    SPLIT_WINDOWS,
    ProbeResult,
    ProbeRun,
)
from src.data.fmp_pipeline.probe.probe_client import ProbeResponse, row_count
from src.data.fmp_pipeline.probe.schema import infer_schema, synthesize

AUTO_MARKER = "<!-- AUTO-GENERATED BELOW by `fmp probe-free`; everything after this line is overwritten -->"
_MESSAGE_LIMIT = 200


def _fmt_params(params: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items()) or "—"


def _per_row(total: int, rows: int) -> str:
    return f"{total / rows:.0f}" if rows else "—"


def _status(result: ProbeResult) -> tuple[str, str]:
    if result.response is None:
        return "—", f"skipped ({result.skipped})"
    return str(result.response.status), result.response.classification


def _body_shape(response: ProbeResponse) -> str:
    """Describe an error body without reproducing data: JSON keys plus a truncated message, or truncated text."""
    body = response.json()
    if isinstance(body, dict):
        message = next((str(v) for k, v in body.items() if "error" in k.lower() or "message" in k.lower()), "")
        return f"object keys {sorted(body)}" + (f": `{message[:_MESSAGE_LIMIT]}`" if message else "")
    if isinstance(body, list):
        return f"list of {len(body)}"
    text = re.sub(r"\s+", " ", response.body).strip()
    return f"text: `{text[:_MESSAGE_LIMIT]}`" if text else "empty body"


def _endpoint_table(results: list[ProbeResult]) -> list[str]:
    lines = ["| Label | Params | HTTP | Class | Rows | Wire B | Decoded B | Wire B/row | Decoded B/row | Encoding |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for result in results:
        status, cls = _status(result)
        r = result.response
        if r is None:
            lines.append(f"| `{result.call.label}` | {_fmt_params(result.call.params)} | {status} | {cls} "
                         "| | | | | | |")
            continue
        rows = row_count(r.json())
        lines.append(
            f"| `{result.call.label}` | {_fmt_params(result.call.params)} | {status} | {cls} | {rows} "
            f"| {r.bytes_wire} | {r.bytes_decoded} | {_per_row(r.bytes_wire, rows)} "
            f"| {_per_row(r.bytes_decoded, rows)} | {r.headers.get('content-encoding', 'none')} |"
        )
    return lines


def _schema_section(run: ProbeRun) -> list[str]:
    lines = ["## Recorded schemas", "",
             "Field names, JSON types (with counts), null rate and string format per endpoint. No values.", ""]
    seen: set[str] = set()
    for result in run.results:
        r = result.response
        if r is None or r.classification != "ok" or r.endpoint in seen:
            continue
        seen.add(r.endpoint)
        schema = infer_schema(r.json())
        wrapper = f", wrapped under `{schema.wrapper_key}`" if schema.wrapper_key else ""
        lines += [f"### `{r.endpoint}`", "", f"Container: {schema.container}{wrapper}; {schema.records} records "
                  f"in the sample (`{result.call.label}`).", "", "| Field | Types | Null rate | String format |",
                  "|---|---|---|---|"]
        for f in schema.fields.values():
            types = ", ".join(f"{t}×{n}" for t, n in f.types.most_common())
            formats = ", ".join(f"{t}×{n}" for t, n in f.formats.most_common()) or "—"
            lines.append(f"| `{f.name}` | {types} | {f.null_rate(schema.records):.0%} | {formats} |")
        lines.append("")
    return lines


def _adjustment_section(run: ProbeRun) -> list[str]:
    lines = ["## Dividend basis (§4.2.5)", "", "| Symbol | Ex-date | Known raw amount | Row found | Fields equal to it "
             "| All dividend fields |", "|---|---|---|---|---|---|"]
    for symbol, ex_date, amount in DIVIDEND_FACTS:
        response = run.by_label(f"dividends:{symbol}")
        if response is None or response.classification != "ok":
            cls = response.classification if response else "skipped"
            lines.append(f"| {symbol} | {ex_date} | {amount} | {cls} | | |")
            continue
        basis = dividend_basis(run.rows(f"dividends:{symbol}"), ex_date, amount)
        matching = ", ".join(basis.fields_matching) or "none"
        lines.append(f"| {symbol} | {ex_date} | {amount} | {basis.found} | {matching} | {basis.values} |")
    first, last = date_range(run.rows("dividends:AAPL"))
    lines += ["", f"AAPL `dividends` history spans {first} … {last}.", "",
              "## Daily endpoints across known splits (§4.2.6)", "",
              "`close ratio` = last close before the split ÷ first close on/after it: ≈ split ratio means raw, "
              "≈ 1 means split-adjusted. Ratios only; no prices are recorded here.", "",
              "| Split | Endpoint | Class | Pre bar | Post bar | Close ratio | Verdict |",
              "|---|---|---|---|---|---|---|"]
    for label, _symbol, split_date, ratio, _window in SPLIT_WINDOWS:
        for endpoint in DAILY_PRICE_ENDPOINTS:
            key = f"split:{label}:{endpoint}"
            response = run.by_label(key)
            cls = response.classification if response else "skipped"
            check = split_check(endpoint, run.rows(key), split_date, ratio)
            lines.append(f"| {label} | `{endpoint}` | {cls} | {check.pre_date} | {check.post_date} "
                         f"| {check.close_ratio} | {check.verdict} |")
    for symbol in ("AAPL", "NVDA"):
        splits = [(str(r.get("date")), r.get("numerator"), r.get("denominator")) for r in run.rows(f"splits:{symbol}")]
        lines.append(f"\n`splits` for {symbol}: {splits or 'none returned'}")
    return lines + [""]


def _reuse_section(run: ProbeRun) -> list[str]:
    lines = ["## Ticker reuse (§4.2.8)", ""]
    if not run.reused:
        return lines + ["No candidates found in the sampled `delisted-companies` pages and `symbol-change`.", ""]
    keep = ("symbol", "companyName", "exchange", "ipoDate", "delistedDate", "oldSymbol", "newSymbol", "date")
    for candidate in run.reused:
        lines += [f"### {candidate.symbol} — {candidate.reason}", ""]
        for entry in candidate.entries:
            lines.append("- " + ", ".join(f"{k}={entry[k]}" for k in keep if entry.get(k) is not None))
        prices = run.by_label(f"reuse:{candidate.symbol}:prices")
        profile = run.by_label(f"reuse:{candidate.symbol}:profile")
        if prices is not None:
            first, last = date_range(run.rows(f"reuse:{candidate.symbol}:prices"))
            lines.append(f"- prices: {prices.classification}, {row_count(prices.json())} rows, {first} … {last}")
        if profile is not None:
            rows = run.rows(f"reuse:{candidate.symbol}:profile")
            name = rows[0].get("companyName") if rows else None
            lines.append(f"- profile today: {profile.classification}, companyName={name}")
        lines.append("")
    return lines


def _error_section(run: ProbeRun) -> list[str]:
    lines = ["## Error shapes (§4.2.4)", "", "| Case | HTTP | Class | Body shape |", "|---|---|---|---|"]
    labels = [r.call.label for r in run.results if r.call.group == "errors"]
    labels += [r.call.label for r in run.results if r.call.group == "entitlement"
               and r.response is not None and r.response.classification != "ok"][:2]
    for label in labels:
        response = run.by_label(label)
        if response is None:
            lines.append(f"| `{label}` | — | skipped | |")
        else:
            lines.append(f"| `{label}` | {response.status} | {response.classification} | {_body_shape(response)} |")
    return lines + [""]


def _headers_section(run: ProbeRun) -> list[str]:
    seen: Counter[str] = Counter()
    for result in run.results:
        if result.response is not None:
            seen.update(k for k in result.response.headers if "limit" in k or "retry" in k or "quota" in k)
    found = ", ".join(f"`{k}`×{n}" for k, n in seen.most_common()) or "none"
    return ["## Rate-limit and quota headers", "", f"Headers observed: {found}.", ""]


def render_report(run: ProbeRun, calls_used: int, budget: int) -> str:
    """
    Render the auto-generated part of the probe report.

    Args:
        run: Output of ``FreeProbe.run``.
        calls_used: Live calls recorded in the ledger across all runs.
        budget: The call budget.

    Returns:
        Markdown text starting with ``AUTO_MARKER``.
    """
    live = sum(1 for r in run.results if r.response is not None and not r.response.from_cache)
    skipped = sum(1 for r in run.results if r.response is None)
    entitlement = [r for r in run.results if r.call.group == "entitlement"]
    lines = [
        AUTO_MARKER, "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. Live calls used in total: **{calls_used} / "
        f"{budget}** ({live} live in the last run, {skipped} skipped). Primary symbol: **{run.primary_symbol}**.",
    ]
    if run.stopped:
        lines.append(f"\n**Run stopped early:** {run.stopped}")
    lines += ["", "## Entitlement sweep (Finding C)", "",
              f"Entitled on the free key: {', '.join(run.entitled) or 'none'}.", "",
              "| Symbol | HTTP | Class |", "|---|---|---|"]
    for result in entitlement:
        status, cls = _status(result)
        lines.append(f"| {result.call.params['symbol']} | {status} | {cls} |")

    for title, group in (("Endpoint sweep (§4.2.1, §4.2.2)", "schema"), ("Parameter checks", "params"),
                         ("Pagination (§4.2.3)", "pagination")):
        group_results = [r for r in run.results if r.call.group == group]
        lines += ["", f"## {title}", ""] + _endpoint_table(group_results)
        if group in ("params", "pagination"):
            lines += ["", "| Label | Date range returned |", "|---|---|"]
            for result in group_results:
                first, last = date_range(run.rows(result.call.label))
                if first:
                    lines.append(f"| `{result.call.label}` | {first} … {last} |")
    lines += [""] + _error_section(run) + _adjustment_section(run) + _reuse_section(run) + _headers_section(run)
    lines += _schema_section(run)
    return "\n".join(lines).rstrip() + "\n"


def write_report(path: Path, body: str) -> None:
    """
    Write the report, keeping any hand-written text above ``AUTO_MARKER`` in an existing file.

    Args:
        path: Report path.
        body: Output of ``render_report``.
    """
    header = "# FMP free-key probe report\n\nHand-written findings go above the marker.\n\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if AUTO_MARKER in existing:
            header = existing.split(AUTO_MARKER, 1)[0]
    path.write_text(header + body, encoding="utf-8")


def fixture_name(endpoint: str) -> str:
    """Map an endpoint path to a fixture file name, e.g. ``historical-chart/1hour`` -> ``historical-chart__1hour``."""
    return endpoint.replace("/", "__") + ".json"


def write_fixtures(run: ProbeRun, fixtures_dir: Path) -> list[Path]:
    """
    Write one synthetic fixture per endpoint that returned data, plus the observed error shapes.

    Values are fake; only field names, types and container shapes come from the probe (D5).

    Args:
        run: Output of ``FreeProbe.run``.
        fixtures_dir: Target directory, created if missing.

    Returns:
        Paths written.
    """
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    seen: set[str] = set()
    errors: dict[str, dict[str, Any]] = {}
    for result in run.results:
        r = result.response
        if r is None:
            continue
        if r.classification == "ok" and r.endpoint not in seen:
            seen.add(r.endpoint)
            path = fixtures_dir / fixture_name(r.endpoint)
            path.write_text(json.dumps(synthesize(infer_schema(r.json())), indent=2) + "\n", encoding="utf-8")
            written.append(path)
        elif r.classification not in ("ok", "ok_empty") and r.classification not in errors:
            body: Optional[Any] = r.json()
            errors[r.classification] = {"status": r.status, "body": body if body is not None else r.body[:500]}
    path = fixtures_dir / "_error_shapes.json"
    path.write_text(json.dumps(errors, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return written + [path]

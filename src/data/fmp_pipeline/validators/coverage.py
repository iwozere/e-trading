"""``coverage_report.md`` (spec §11): per endpoint, symbols requested and tasks per final status, with bytes."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.data.fmp_pipeline.manifest.manifest import Manifest

STATUSES = ("ok", "ok_empty", "plan_restricted", "api_error", "failed", "possibly_truncated", "skipped", "pending",
            "running")
GB = 1024**3


def render_coverage(manifest: Manifest) -> str:
    """
    Render the coverage report of one dump.

    Args:
        manifest: The dump's manifest (read-only use).

    Returns:
        Markdown text.
    """
    rows: dict[str, dict[str, Any]] = defaultdict(lambda: {"symbols": set(), "wire": 0, "decoded": 0,
                                                           **{s: 0 for s in STATUSES}})
    for task in manifest.task_rows():
        row = rows[task["endpoint"]]
        if task["symbol"]:
            row["symbols"].add(task["symbol"])
        row[task["status"]] = row.get(task["status"], 0) + 1
        row["wire"] += task["bytes_wire"] or 0
        row["decoded"] += task["bytes_decoded"] or 0
    header = ["Endpoint", "Symbols", *STATUSES, "GB wire", "GB decoded"]
    lines = [f"# Coverage report — dump {manifest.dump_id}", "", "| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for endpoint, row in sorted(rows.items()):
        cells = [endpoint, f"{len(row['symbols']):,}", *(f"{row[s]:,}" for s in STATUSES),
                 f"{row['wire'] / GB:.3f}", f"{row['decoded'] / GB:.3f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"

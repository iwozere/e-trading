"""
P22 — the weekly label-review digest (2026-11-09).

The backtest's labels are only complete up to the oldest deal candidate still waiting for review
(`backtest.labels.labels_complete_through`), so this lists what is pending: the deal candidates
queued in the last week, the oldest ones holding the label cut-off back, and the depth of the
other review queues.
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.backtest.labels import labels_complete_through
from src.ml.pipeline.p22_biotech_ma.outputs.email_notify import html_table, link

DEAL_CANDIDATE = "deal_candidate"
MAX_LISTED = 100

_REVIEW_HELP = (
    "Review with <code>python -m src.ml.pipeline.p22_biotech_ma.cli.review_queue_cli list --item-type deal_candidate</code>, "
    "then <code>show ITEM_ID</code> and <code>confirm</code> / <code>reject</code>; enter confirmed deals with "
    "<code>cli.deal_entry_cli</code>."
)


def _filed(item: Mapping[str, Any]) -> str:
    return str((item.get("payload") or {}).get("filed_date") or "")


def _row(item: Mapping[str, Any], companies: Mapping[int, Mapping[str, Any]]) -> List[Any]:
    payload = item.get("payload") or {}
    company_id = payload.get("company_id")
    company: Mapping[str, Any] = companies.get(int(company_id), {}) if company_id is not None else {}
    others = [n for n in payload.get("all_names_in_filing") or [] if company.get("name") and company["name"].lower() not in n.lower()]
    return [item.get("item_id"), company.get("ticker") or "", company.get("name") or payload.get("company_id"),
            payload.get("form_type"), _filed(item), "; ".join(others[:3]),
            link(item.get("evidence_url"), "filing")]


def build_digest(pending: Sequence[Mapping[str, Any]], companies: Mapping[int, Mapping[str, Any]], now: datetime,
                 days: int) -> Tuple[str, str, str, Dict[str, Any]]:
    """
    `(title, plain text, html, stats)` for the digest.

    Args:
        pending: Every pending review item (any type), with `created_at` (tz-aware) and `payload`.
        companies: `{company_id: company row}` for tickers and names.
        now: Run time (tz-aware).
        days: How far back "new" reaches.
    """
    deal_items = [i for i in pending if i.get("item_type") == DEAL_CANDIDATE]
    since = now - timedelta(days=days)
    new = [i for i in deal_items if (created := i.get("created_at")) is not None and created >= since]
    cutoff = labels_complete_through(deal_items, now.date())
    other_queues = Counter(str(i.get("item_type")) for i in pending if i.get("item_type") != DEAL_CANDIDATE)
    stats = {"deal_candidates_pending": len(deal_items), "deal_candidates_new": len(new),
             "labels_complete_through": cutoff.isoformat(), "other_pending": dict(other_queues)}

    title = f"P22 label review: {len(new)} new deal candidates, {len(deal_items)} pending (labels complete through {cutoff})"
    headers = ["item", "ticker", "company", "form", "filed", "other parties", "evidence"]
    oldest = sorted(deal_items, key=_filed)[:MAX_LISTED]
    body = (
        f"<h3>{title}</h3>"
        f"<p>The backtest only uses snapshots whose 24-month window ends by the label cut-off "
        f"({cutoff}): each candidate reviewed from the oldest moves it forward. {_REVIEW_HELP}</p>"
        f"<h4>New in the last {days} days ({len(new)})</h4>"
        + (html_table(headers, [_row(i, companies) for i in sorted(new, key=_filed)]) if new else "<p>None.</p>")
        + f"<h4>Oldest pending ({len(oldest)} of {len(deal_items)})</h4>"
        + (html_table(headers, [_row(i, companies) for i in oldest]) if oldest else "<p>None.</p>")
        + "<h4>Other review queues</h4>"
        + (html_table(["item type", "pending"], sorted(other_queues.items())) if other_queues else "<p>Empty.</p>")
    )
    text_lines = [title, ""] + [f"{r[0]}  {r[1]:6s} {r[2]}  {r[3]} filed {r[4]}" for r in (_row(i, companies) for i in new)]
    text_lines += ["", f"Other queues: {dict(other_queues) or 'empty'}"]
    return title, "\n".join(text_lines), body, stats


def should_send(stats: Mapping[str, Any]) -> bool:
    """Send only when something waits for review."""
    return bool(stats["deal_candidates_pending"] or stats["other_pending"])


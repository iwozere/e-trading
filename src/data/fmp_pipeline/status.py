"""``fmp status`` (spec §6.1): progress per block and tier, rolling-window bandwidth, failures, a rough ETA."""

from __future__ import annotations

from collections import defaultdict

from src.data.fmp_pipeline.manifest.manifest import Manifest
from src.data.fmp_pipeline.planner.catalog import BLOCKS
from src.data.fmp_pipeline.planner.config import PlanConfig
from src.data.fmp_pipeline.planner.planner import planned_remaining
from src.data.fmp_pipeline.planner.universe import TIERS
from src.data.fmp_pipeline.runner import GB

COLUMNS = ("pending", "running", "ok", "ok_empty", "plan_restricted", "api_error", "failed", "possibly_truncated",
           "skipped")
HEADERS = ("pending", "running", "ok", "ok_empty", "restricted", "api_error", "failed", "truncated", "skipped")


def _labels() -> list[str]:
    labels = ["bootstrap"]
    for block in sorted(BLOCKS):
        labels.append(f"{block} {BLOCKS[block]}")
        if block == 6:
            labels += [f"  tier {t}" for t in TIERS]
    return labels


def render_status(manifest: Manifest, config: PlanConfig) -> str:
    """
    Return a plain-text status report of one dump.

    The ETA gives two lower bounds; the real time is the larger: the pending requests at the configured rate, and
    the bytes the last ``fmp plan`` projected as remaining, in bandwidth windows of ``safety × cap`` on the
    configured basis.
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for priority, tier, status, n, _, _ in manifest.totals():
        block = priority // 100
        counts["bootstrap" if priority == 0 else f"{block} {BLOCKS.get(block, '?')}"][status] += n
        if block == 6 and tier is not None:
            counts[f"  tier {tier}"][status] += n
    labels = [label for label in _labels() if label in counts]
    width = max([len(label) for label in labels] + [10])
    lines = [f"FMP dump {manifest.dump_id} (as_of {manifest.meta('as_of') or 'not planned'})", "",
             " " * width + " " + " ".join(f"{h:>10}" for h in HEADERS)]
    lines += [f"{label:{width}} " + " ".join(f"{counts[label].get(c, 0):>10,}" for c in COLUMNS) for label in labels]

    wire, decoded = manifest.bandwidth_used(config.bandwidth.window_days)
    used = wire if config.bandwidth.basis == "wire" else decoded
    limit = config.bandwidth.limit_bytes
    lines += ["", f"Bandwidth, last {config.bandwidth.window_days} days: {wire / GB:.2f} GB wire, "
                  f"{decoded / GB:.2f} GB decoded; hard stop at {limit / GB:.2f} GB {config.bandwidth.basis} "
                  f"({used / limit:.0%} used)."]

    pending = sum(c.get("pending", 0) + c.get("running", 0) for label, c in counts.items() if not label.startswith(" "))
    hours = pending / config.rate_limit_per_min / 60
    eta = f"{pending:,} tasks pending: at least {hours:.1f} h at {config.rate_limit_per_min} requests/min"
    projected = planned_remaining(manifest)
    if projected is not None:
        index = 0 if config.bandwidth.basis == "wire" else 1
        remaining = sum(v[index] for v in projected.values())
        eta += (f"; the last plan projected {remaining / GB:.2f} GB ({config.bandwidth.basis}) still to fetch, "
                f"about {remaining / limit:.1f} bandwidth windows of {config.bandwidth.window_days} days")
    lines.append(eta + ".")

    events = manifest.events()[-3:]
    if events:
        lines += ["", "Recent run events:"] + [f"  {ts[:19]}  {event}  {detail}" for ts, event, detail in events]
    failures = manifest.failures()
    if failures:
        lines += ["", "Recent failures (`fmp retry-failed` re-queues them):"]
        lines += [f"  {endpoint} {symbol or '-'}: {error[:100]}" for endpoint, symbol, error in failures]
    return "\n".join(lines)

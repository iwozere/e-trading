"""
Run-time planning: the ``Reviewer`` the ``Downloader`` calls with every final response (see ``runner.py``).

- **Paging** (spec §5): a non-empty page of a paged endpoint plans the next page; the list ends at the first empty
  page. "Non-empty", not "full", because FMP caps ``limit`` per plan (100 on the free key), so a short page proves
  nothing. Two guards stop runaway paging on an endpoint that ignores ``page``: ``max_pages``, and a page whose
  body is identical to the previous page of the same request.
- **Daily bounds**: an ``ok`` daily raw response widens the symbol's first/last bar dates in the manifest; the next
  ``fmp plan`` windows intraday history by them (spec §6.5.5).
- **Intraday truncation** (spec §6.5.3): a window at ≥ 95% of the row cap is relabelled ``possibly_truncated`` (so
  the archive and the builder never treat it as complete) and its two halves are planned. A one-day window that
  still hits the cap cannot be split and is reported in its error text.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date
from typing import Any

from src.data.fmp_pipeline.client.fmp_client import FetchResult
from src.data.fmp_pipeline.manifest.manifest import Manifest, Task, TaskSpec
from src.data.fmp_pipeline.planner.catalog import CATALOG, DAILY_RAW
from src.data.fmp_pipeline.planner.config import PlanConfig
from src.data.fmp_pipeline.planner.windowing import is_possibly_truncated, split_window
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

PAGED = frozenset(e.endpoint for e in CATALOG if e.kind == "paged")


def _dates(body: Any) -> list[date]:
    out = []
    for row in body if isinstance(body, list) else []:
        value = row.get("date") if isinstance(row, dict) else None
        if isinstance(value, str) and len(value) >= 10:
            try:
                out.append(date.fromisoformat(value[:10]))
            except ValueError:
                continue
    return out


class FollowUps:
    """Reviewer for the ``Downloader``: paging, daily bounds and intraday truncation."""

    def __init__(self, manifest: Manifest, config: PlanConfig) -> None:
        """
        Args:
            manifest: Receives daily bounds.
            config: Supplies the page cap, the intraday interval and its row cap.
        """
        self._manifest = manifest
        self._config = config
        self._intraday = f"historical-chart/{config.intraday.interval}"
        self._page_hashes: dict[tuple[str, str], tuple[int, str]] = {}

    def __call__(self, task: Task, result: FetchResult) -> tuple[FetchResult, list[TaskSpec]]:
        """Return the (possibly relabelled) result and the tasks it implies."""
        if result.classification != "ok":
            return result, []
        if task.endpoint in PAGED:
            return result, self._next_page(task, result)
        if task.endpoint == DAILY_RAW and task.symbol is not None:
            days = _dates(result.json())
            if days:
                self._manifest.update_bounds(task.symbol, min(days), max(days))
            return result, []
        if task.endpoint == self._intraday and self._config.intraday.max_rows is not None:
            return self._check_truncation(task, result, self._config.intraday.max_rows)
        return result, []

    def _next_page(self, task: Task, result: FetchResult) -> list[TaskSpec]:
        page = int(task.params.get("page", 0))
        rest = {k: v for k, v in task.params.items() if k != "page"}
        key = (task.endpoint, json.dumps(rest, sort_keys=True))
        digest = hashlib.sha256(result.body).hexdigest()
        previous = self._page_hashes.get(key)
        self._page_hashes[key] = (page, digest)
        if previous is not None and previous == (page - 1, digest):
            _logger.warning("%s page %d repeats page %d; the endpoint seems to ignore 'page', stopping",
                            task.endpoint, page, page - 1)
            return []
        if page + 1 >= self._config.max_pages:
            _logger.warning("%s reached max_pages=%d; stopping pagination", task.endpoint, self._config.max_pages)
            return []
        return [TaskSpec(task.endpoint, task.symbol, {**task.params, "page": page + 1}, task.priority, task.tier)]

    def _check_truncation(self, task: Task, result: FetchResult,
                          max_rows: int) -> tuple[FetchResult, list[TaskSpec]]:
        if not is_possibly_truncated(result.rows, max_rows, self._config.intraday.truncation_ratio):
            return result, []
        start, end = date.fromisoformat(task.params["from"]), date.fromisoformat(task.params["to"])
        halves = split_window(start, end)
        message = f"possibly truncated: {result.rows} rows >= {self._config.intraday.truncation_ratio:.0%} of " \
                  f"{max_rows}"
        if halves is None:
            _logger.error("%s %s %s: one-day window hit the row cap", task.endpoint, task.symbol, start)
            return replace(result, classification="possibly_truncated", error=message + "; cannot split"), []
        _logger.warning("%s %s %s..%s %s; re-planning as two halves", task.endpoint, task.symbol, start, end,
                        message)
        specs = [TaskSpec(task.endpoint, task.symbol, {**task.params, "from": lo.isoformat(), "to": hi.isoformat()},
                          task.priority, task.tier) for lo, hi in halves]
        return replace(result, classification="possibly_truncated", error=message), specs

"""Write synthetic responses straight into a raw archive (builder tests; no server, no real FMP data)."""

from __future__ import annotations

import json
from typing import Any, Optional

from src.data.fmp_pipeline.archive.archive import RawArchive
from src.data.fmp_pipeline.client.fmp_client import FetchResult
from src.data.fmp_pipeline.manifest.manifest import Task, task_id

FETCHED_AT = "2026-10-14T10:00:00+00:00"


def put(archive: RawArchive, endpoint: str, symbol: Optional[str], body: Any, params: Optional[dict[str, Any]] = None,
        fetched_at: str = FETCHED_AT, classification: str = "ok", status: int = 200) -> None:
    """
    Append one synthetic response to ``archive``.

    Args:
        archive: The archive.
        endpoint: Endpoint path.
        symbol: Symbol, or ``None`` for global endpoints.
        body: JSON body.
        params: Request parameters other than ``symbol``.
        fetched_at: Fetch timestamp (ISO, UTC).
        classification: Response classification.
        status: HTTP status.
    """
    params = dict(params or {})
    task = Task(task_id(archive.dump_dir.name, endpoint, symbol, params), archive.dump_dir.name, endpoint, symbol,
                params, 1, None, 1)
    raw = json.dumps(body).encode("utf-8")
    rows = len(body) if isinstance(body, list) else 1
    archive.append(task, FetchResult(endpoint, task.request_params, status, classification, raw, len(raw), len(raw),
                                     rows, 1, fetched_at))


def daily_rows(symbol: str, closes: dict[str, float], volume: int = 1000) -> list[dict[str, Any]]:
    """Rows shaped like ``historical-price-eod/non-split-adjusted`` (raw prices under ``adj*`` names, probe F3)."""
    return [{"symbol": symbol, "date": day, "adjOpen": close, "adjHigh": close * 1.01, "adjLow": close * 0.99,
             "adjClose": close, "volume": volume} for day, close in sorted(closes.items(), reverse=True)]

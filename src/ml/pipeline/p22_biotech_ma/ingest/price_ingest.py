"""
P22 — daily current-price ingest normalization (spec §2.0.7, M3, 2026-09-01).

Turns yfinance's narrow trailing-window bars (`ingest/yfinance_client.py`)
into `p22_price_daily`/`p22_corporate_action` rows, for the ONGOING/current-
price role — NOT the deep-history backfill role, which is FMP's job
(`ingest/fmp_backfill.py`). See `yfinance_client.py`'s docstring for the
retroactive-split-adjustment trap this two-source split deliberately avoids.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def write_daily_bars(
    company_id: int, bars: List[Dict[str, Any]], repo: Any, *, known_from: Optional[datetime] = None
) -> Dict[str, int]:
    """
    Write yfinance daily-bar dicts (`ingest.yfinance_client.fetch_recent_daily_bars`)
    for one company via `P22Repo.upsert_price_daily`/`upsert_corporate_action`.

    Args:
        company_id: The company these bars belong to.
        bars: Daily bar dicts.
        repo: A `P22Repo`-shaped object.
        known_from: Defaults to "now" — spec §2.0.7's `VENDOR_PRICE_LAG_DAYS
        = 0` documents same-day publication as the default assumption for
            price data (unlike fundamentals' 45-day lag).

    Returns:
        `{"prices_written": int, "actions_written": int}`. A bar with no
        `close` (e.g. a still-pending/incomplete trading day) is skipped, not
        written as a zero.
    """
    known_from = known_from or datetime.now(timezone.utc)
    prices_written = 0
    actions_written = 0

    for bar in bars:
        if bar.get("close") is None:
            _logger.debug("Skipping bar with no close for %s on %s", bar.get("ticker"), bar.get("date"))
            continue
        trade_date = date.fromisoformat(bar["date"])

        repo.upsert_price_daily(
            company_id=company_id,
            trade_date=trade_date,
            vendor="yfinance",
            open_raw=bar.get("open"),
            high_raw=bar.get("high"),
            low_raw=bar.get("low"),
            close_raw=bar.get("close"),
            volume_raw=bar.get("volume"),
            known_from=known_from,
        )
        prices_written += 1

        split_ratio = bar.get("stock_splits") or 0.0
        if split_ratio:
            repo.upsert_corporate_action(
                company_id=company_id,
                ex_date=trade_date,
                # yfinance's own convention already matches price_archive.py's documented one
                # ("4.0 for 4:1 fwd, 0.05 for 1:20 reverse") — no ratio translation needed.
                action_type="split" if split_ratio >= 1 else "reverse_split",
                ratio=split_ratio,
                source="yfinance",
                is_verified=False,
                known_from=known_from,
            )
            actions_written += 1

        dividend = bar.get("dividends") or 0.0
        if dividend:
            repo.upsert_corporate_action(
                company_id=company_id,
                ex_date=trade_date,
                action_type="dividend",
                cash_amount=dividend,
                source="yfinance",
                is_verified=False,
                known_from=known_from,
            )
            actions_written += 1

    return {"prices_written": prices_written, "actions_written": actions_written}

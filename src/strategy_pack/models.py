"""Typed records for trading-strategy-pack signal audit trail."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any, Dict

from pydantic import BaseModel, Field, model_validator


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_idempotency_key(strategy_id: str, symbol: str, bar_close_ts: str, signal: str, variant: str = "") -> str:
    raw = f"{strategy_id}|{variant}|{symbol}|{bar_close_ts}|{signal}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


class PackSignal(BaseModel):
    """One decision row for JSONL audit + optional notification."""

    strategy_id: str = Field(..., description="Pack strategy id, e.g. SP-2")
    variant: str = ""
    symbol: str
    signal: str
    ts_utc: str = Field(default_factory=utc_now_iso)
    bar_timeframe: str = ""
    bar_close_ts: str = ""
    price: float = 0.0
    reason_code: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = ""
    notify_recommended: bool = True

    @model_validator(mode="after")
    def _fill_idempotency(self) -> PackSignal:
        if not self.idempotency_key:
            self.idempotency_key = make_idempotency_key(
                self.strategy_id,
                self.symbol,
                self.bar_close_ts or self.ts_utc,
                self.signal,
                self.variant,
            )
        return self

    def to_jsonl_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

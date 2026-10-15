"""A small synthetic vintage built by the real builder (adapter and increment tests; no real data)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.fmp_pipeline.archive.archive import RawArchive
from src.data.fmp_pipeline.builders.builder import VintageBuilder
from src.data.fmp_pipeline.builders.prices import DAILY_ENDPOINT
from src.data.fmp_pipeline.planner.config import BuildConfig, PlanConfig, UniverseConfig
from src.data.fmp_pipeline.tests.archive_fixtures import daily_rows, put

AS_OF = date(2026, 10, 14)
DAYS = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2025-01-02", "2025-02-28")]
SPLIT_DAY = "2025-02-03"
DIVIDEND_DAY = "2025-01-21"


def _profile(symbol: str, name: str, **extra: Any) -> list[dict[str, Any]]:
    return [{"symbol": symbol, "companyName": name, "exchange": "NYSE", "isEtf": False, "isFund": False,
             "isAdr": False, "isActivelyTrading": True, "ipoDate": "2010-01-04", **extra}]


def make_vintage(data_root: Path, segment: bool = False) -> PlanConfig:
    """Archive + build: AAA (4:1 split, one dividend, 1h bars), BRK.B, a renamed symbol, a reused ticker."""
    config = PlanConfig(dump_id="v", universe=UniverseConfig(tier_a=(), indices=()),
                        build=BuildConfig(segment_reused_tickers=segment))
    archive = RawArchive(data_root / "raw", "v")
    put(archive, "stock-list", None, [{"symbol": s, "exchange": "NYSE"} for s in ("AAA", "BRK.B", "NEWC", "XYZ")])
    put(archive, "actively-trading-list", None, [{"symbol": s} for s in ("AAA", "BRK.B", "NEWC", "XYZ")])
    put(archive, "delisted-companies", None, [{"symbol": "XYZ", "companyName": "Old Widgets", "exchange": "NYSE",
                                               "ipoDate": "1999-01-04", "delistedDate": "2024-12-31"}],
        {"page": 0, "limit": 100})
    put(archive, "symbol-change", None, [{"date": "2025-02-10", "oldSymbol": "OLDC", "newSymbol": "NEWC"}],
        {"page": 0, "limit": 100})
    put(archive, "profile", "XYZ", _profile("XYZ", "New Robotics", ipoDate="2025-01-02"))
    closes = {d: (400.0 if d < SPLIT_DAY else 100.0) + i for i, d in enumerate(DAYS)}
    for symbol in ("AAA", "BRK.B", "NEWC"):
        put(archive, DAILY_ENDPOINT, symbol, daily_rows(symbol, closes), {"from": "2025-01-01", "to": "2025-12-31"})
    put(archive, DAILY_ENDPOINT, "XYZ", daily_rows("XYZ", {"2024-12-30": 5.0, "2025-01-02": 7.0}),
        {"from": "2020-01-01", "to": "2025-12-31"})
    put(archive, "splits", "AAA", [{"symbol": "AAA", "date": SPLIT_DAY, "numerator": 4, "denominator": 1}])
    put(archive, "dividends", "AAA", [{"symbol": "AAA", "date": DIVIDEND_DAY, "dividend": 2.0, "adjDividend": 0.5,
                                       "recordDate": "", "paymentDate": "", "declarationDate": ""}])
    hourly = []
    for day in ("2025-01-31", SPLIT_DAY):
        for hour in range(9, 16):
            price = 400.0 if day < SPLIT_DAY else 100.0
            hourly.append({"date": f"{day} {hour:02d}:30:00", "open": price, "high": price, "low": price,
                           "close": price, "volume": 10})
    put(archive, "historical-chart/1hour", "AAA", hourly, {"from": "2025-01-01", "to": "2025-03-31"})
    VintageBuilder(data_root, config, AS_OF).build()
    return config

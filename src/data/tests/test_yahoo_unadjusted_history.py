"""Tests for ``YahooDataDownloader.get_unadjusted_history`` (FMP increment layer, spec §10.2.2). No network."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import pytest
from yfinance.exceptions import YFRateLimitError

from src.data.downloader import yahoo_data_downloader as module
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader

START, END = datetime(2025, 3, 3), datetime(2025, 3, 6)
FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]


def _batch(symbols: list[str]) -> pd.DataFrame:
    index = pd.DatetimeIndex(["2025-03-03", "2025-03-04", "2025-03-05"]).tz_localize("America/New_York")
    columns = pd.MultiIndex.from_product([symbols, FIELDS])
    frame = pd.DataFrame(1.0, index=index, columns=columns)
    frame.loc[index[0], (symbols[-1], slice(None))] = float("nan")  # a symbol with a missing first bar
    return frame


@pytest.fixture
def downloader() -> YahooDataDownloader:
    return YahooDataDownloader.__new__(YahooDataDownloader)  # no session setup needed: yf.download is faked


def test_requests_unadjusted_prices_with_actions(downloader: YahooDataDownloader,
                                                 monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {}

    def fake_download(symbols: list[str], **kwargs: Any) -> pd.DataFrame:
        seen.update(kwargs, symbols=symbols)
        return _batch(["AAA", "BBB"])

    monkeypatch.setattr(module.yf, "download", fake_download)
    result = downloader.get_unadjusted_history(["AAA", "BBB", "MISSING"], "1d", START, END)
    assert seen["auto_adjust"] is False and seen["actions"] is True and seen["group_by"] == "ticker"
    assert seen["ignore_tz"] is False and seen["progress"] is False
    assert list(result) == ["AAA", "BBB"]
    assert list(result["AAA"].columns) == FIELDS and len(result["AAA"]) == 3 and len(result["BBB"]) == 2
    assert pd.DatetimeIndex(result["AAA"].index).tz is not None  # exchange time kept for the caller's conversion


def test_empty_failures_and_rate_limits(downloader: YahooDataDownloader, monkeypatch: pytest.MonkeyPatch):
    assert downloader.get_unadjusted_history([], "1d", START, END) == {}
    monkeypatch.setattr(module.yf, "download", lambda *a, **k: pd.DataFrame())
    assert downloader.get_unadjusted_history(["AAA"], "1d", START, END) == {}

    def broken(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise RuntimeError("boom")

    monkeypatch.setattr(module.yf, "download", broken)
    assert downloader.get_unadjusted_history(["AAA"], "1d", START, END) == {}

    def throttled(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise YFRateLimitError()

    monkeypatch.setattr(module.yf, "download", throttled)
    with pytest.raises(YFRateLimitError):
        downloader.get_unadjusted_history(["AAA"], "1d", START, END)

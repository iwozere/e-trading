"""Tests for P17 MarketAgent OHLCV freshness (no stale bars, no look-ahead)."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p17_penny_stocks.agents import market_agent as ma
from src.ml.pipeline.p17_penny_stocks.config import P17FilterConfig


def _bars(end: str, n: int = 40, tz: Optional[str] = None) -> pd.DataFrame:
    idx = pd.bdate_range(end=end, periods=n, tz=tz)
    close = np.linspace(2.0, 3.0, n)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": np.full(n, 1e6)},
        index=idx,
    )


# ── Calendar helpers ───────────────────────────────────────────────────────────


def test_last_session_weekday_is_itself():
    assert ma.last_session_on_or_before(pd.Timestamp("2026-11-05")) == pd.Timestamp("2026-11-05")


def test_last_session_weekend_rolls_back_to_friday():
    assert ma.last_session_on_or_before(pd.Timestamp("2026-11-08")) == pd.Timestamp("2026-11-06")


def test_last_session_holiday_rolls_back():
    # Thanksgiving 2026-11-26 (Thursday) is an NYSE holiday
    assert ma.last_session_on_or_before(pd.Timestamp("2026-11-26")) == pd.Timestamp("2026-11-25")


def test_last_bar_date_and_truncate_handle_tz_aware_index():
    df = _bars("2026-11-06", tz="America/New_York")
    assert ma.last_bar_date(df) == pd.Timestamp("2026-11-06")
    cut = ma.truncate_to(df, pd.Timestamp("2026-11-04"))
    assert ma.last_bar_date(cut) == pd.Timestamp("2026-11-04")


# ── _download_ohlcv ────────────────────────────────────────────────────────────


class _FakeCache:
    def __init__(self, data: Dict[str, pd.DataFrame]) -> None:
        self.data = data
        self.puts: List[str] = []

    def get(self, ticker: str, *_: Any, **__: Any) -> Optional[pd.DataFrame]:
        return self.data.get(ticker)

    def put(self, df: pd.DataFrame, ticker: str, *_: Any, **__: Any) -> None:
        self.puts.append(ticker)


def _agent(monkeypatch: pytest.MonkeyPatch, cache: _FakeCache, batch: Dict[str, pd.DataFrame], target: str) -> Any:
    monkeypatch.setattr(ma, "YahooDataDownloader", MagicMock())
    monkeypatch.setattr(ma, "FundamentalsCache", MagicMock())
    monkeypatch.setattr(ma, "get_unified_cache", lambda: cache)
    agent = ma.MarketAgent(P17FilterConfig(), Path("."), target)
    agent._downloader = MagicMock()
    agent._downloader.get_ohlcv_batch.return_value = {
        t: df.rename_axis("timestamp").reset_index() for t, df in batch.items()
    }
    return agent


def test_download_wins_over_partial_cached_bar(monkeypatch: pytest.MonkeyPatch):
    """Regression: a partial intraday bar in the shared cache must not be scored."""
    partial = _bars("2026-11-05")
    partial.loc[partial.index[-1], "volume"] = 1.0
    full = _bars("2026-11-05")
    cache = _FakeCache({"ANGI": partial})
    agent = _agent(monkeypatch, cache, {"ANGI": full}, "2026-11-05")
    out = agent._download_ohlcv(["ANGI"], force_refresh=False)
    assert out["ANGI"]["Volume"].iloc[-1] == 1e6
    assert cache.puts == ["ANGI"]  # fresh bars written back over the partial one


def test_fallback_uses_cache_only_when_fresh(monkeypatch: pytest.MonkeyPatch):
    """Download failed → cache fallback, but the old 5-day tolerance is gone."""
    cache = _FakeCache({"FRESH": _bars("2026-11-05"), "STALE": _bars("2026-11-04")})
    agent = _agent(monkeypatch, cache, {}, "2026-11-05")
    out = agent._download_ohlcv(["FRESH", "STALE"], force_refresh=False)
    assert set(out) == {"FRESH"}


def test_weekend_target_accepts_friday_bar(monkeypatch: pytest.MonkeyPatch):
    cache = _FakeCache({"FRI": _bars("2026-11-06")})
    agent = _agent(monkeypatch, cache, {}, "2026-11-08")  # Sunday target (Monday run), download failed
    out = agent._download_ohlcv(["FRI"], force_refresh=False)
    assert ma.last_bar_date(out["FRI"]) == pd.Timestamp("2026-11-06")


def test_bars_after_target_date_are_dropped(monkeypatch: pytest.MonkeyPatch):
    agent = _agent(monkeypatch, _FakeCache({}), {"LATER": _bars("2026-11-06")}, "2026-11-04")
    out = agent._download_ohlcv(["LATER"], force_refresh=False)
    assert ma.last_bar_date(out["LATER"]) == pd.Timestamp("2026-11-04")


def test_lagging_download_is_kept(monkeypatch: pytest.MonkeyPatch):
    """A halted name whose fresh download still ends early stays usable (and is logged as lagging)."""
    agent = _agent(monkeypatch, _FakeCache({}), {"HALT": _bars("2026-11-02")}, "2026-11-05")
    out = agent._download_ohlcv(["HALT"], force_refresh=False)
    assert ma.last_bar_date(out["HALT"]) == pd.Timestamp("2026-11-02")

"""Tests for Stage1Prefilter."""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ml.pipeline.p05_ai_selector.config import STAGE1_TOP_N
from src.ml.pipeline.p05_ai_selector.stages.stage1_prefilter import Stage1Prefilter


def _make_ohlcv(last_close: float = 10.0, avg_volume: float = 1_000_000.0, n: int = 65) -> pd.DataFrame:
    closes = [last_close] * n
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [avg_volume] * n,
        }
    )


class TestStage1Prefilter:
    def test_price_filter_removes_penny_stocks(self, tmp_path):
        """Tickers with last close < $2 are excluded."""
        prefilter = Stage1Prefilter(cache_dir=tmp_path)
        ohlcv_cheap = _make_ohlcv(last_close=1.50)
        ohlcv_ok = _make_ohlcv(last_close=50.0)

        with patch("src.ml.pipeline.p05_ai_selector.stages.stage1_prefilter.DataManager") as MockDM:
            dm_instance = MagicMock()
            MockDM.return_value = dm_instance
            dm_instance.get_ohlcv_batch.side_effect = lambda tickers, *a, **kw: {
                t: (ohlcv_cheap if t == "PENNYSTOCK" else ohlcv_ok) for t in tickers
            }
            result = prefilter.run(["PENNYSTOCK", "AAPL"], date(2026, 6, 14))

        assert "PENNYSTOCK" not in result["ticker"].values
        assert "AAPL" in result["ticker"].values

    def test_volume_filter_applied(self, tmp_path):
        """Equities with avg_vol_usd < $5M are excluded."""
        prefilter = Stage1Prefilter(cache_dir=tmp_path)
        # price=$10, volume=1 → avg_vol_usd = $10 < $5M
        ohlcv_low_vol = _make_ohlcv(last_close=10.0, avg_volume=1.0)
        ohlcv_ok = _make_ohlcv(last_close=100.0, avg_volume=100_000.0)

        with patch("src.ml.pipeline.p05_ai_selector.stages.stage1_prefilter.DataManager") as MockDM:
            dm_instance = MagicMock()
            MockDM.return_value = dm_instance
            dm_instance.get_ohlcv_batch.side_effect = lambda tickers, *a, **kw: {
                t: (ohlcv_low_vol if t == "LOWVOL" else ohlcv_ok) for t in tickers
            }
            result = prefilter.run(["LOWVOL", "AAPL"], date(2026, 6, 14))

        assert "LOWVOL" not in result["ticker"].values

    def test_output_capped_at_top_n(self, tmp_path):
        """Result is capped at STAGE1_TOP_N rows."""
        prefilter = Stage1Prefilter(cache_dir=tmp_path)
        tickers = [f"TICK{i}" for i in range(STAGE1_TOP_N + 50)]
        ohlcv_ok = _make_ohlcv(last_close=50.0, avg_volume=200_000.0)

        with patch("src.ml.pipeline.p05_ai_selector.stages.stage1_prefilter.DataManager") as MockDM:
            dm_instance = MagicMock()
            MockDM.return_value = dm_instance
            dm_instance.get_ohlcv_batch.side_effect = lambda tickers, *a, **kw: {t: ohlcv_ok for t in tickers}
            result = prefilter.run(tickers, date(2026, 6, 14))

        assert len(result) <= STAGE1_TOP_N

    def test_cache_hit_skips_computation(self, tmp_path):
        """When today's cache exists, DataManager is never called."""
        prefilter = Stage1Prefilter(cache_dir=tmp_path)
        ref_date = date(2026, 6, 14)
        cached_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "asset_type": ["equity"],
                "last_price": [195.0],
                "avg_vol_usd": [50_000_000.0],
                "momentum_score": [30.0],
                "volume_surge_ratio": [1.2],
                "signal_breakdown": ["{}"],
            }
        )
        cache_file = tmp_path / f"{ref_date}.csv.gz"
        cached_df.to_csv(cache_file, index=False, compression="gzip")

        with patch("src.ml.pipeline.p05_ai_selector.stages.stage1_prefilter.DataManager") as MockDM:
            result = prefilter.run(["AAPL"], ref_date)
            MockDM.assert_not_called()

        assert len(result) == 1

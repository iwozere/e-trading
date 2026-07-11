"""
Regression tests for PackSignal JSON serialization.

SP-2/SP-3 crashed nightly from 2026-07-06 with
PydanticSerializationError: Unable to serialize unknown type: <class 'numpy.bool'>
because pandas boolean-Series elements (numpy.bool_) reached PackSignal.metadata.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.strategy_pack.models import PackSignal
from src.strategy_pack.strategies import RunContext, run_strategy_2, run_strategy_3


class _FakeDataManager:
    """Returns a fixed OHLCV frame for any symbol/timeframe."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def get_ohlcv(self, symbol, timeframe, start_date=None, end_date=None, force_refresh=False):
        return self._df


def _make_ohlcv(rows: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    index = pd.date_range(end="2026-07-09", periods=rows, freq="D")
    close = pd.Series(100.0 + np.arange(rows) * 0.05 + rng.normal(0, 0.3, rows), index=index)
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": rng.integers(1_000_000, 2_000_000, rows).astype("int64"),
        },
        index=index,
    )


def _ctx(strategy_key: str, **cfg) -> RunContext:
    config = {strategy_key: {"symbols": ["TEST"], **cfg}}
    return RunContext(
        dm=_FakeDataManager(_make_ohlcv()),  # type: ignore[arg-type]  # lightweight test double
        end=datetime(2026, 7, 9),
        config=config,
    )


def test_strategy_2_signals_are_json_serializable():
    signals = run_strategy_2(_ctx("strategy_2", lookback_days=400))
    assert signals, "expected SP-2 to emit a signal"
    for s in signals:
        d = s.to_jsonl_dict()
        json.dumps(d)  # must not raise
        assert type(d["metadata"]["in_uptrend"]) is bool
        assert type(d["metadata"]["cross_up"]) is bool
        assert type(d["metadata"]["cross_down"]) is bool


def test_strategy_3_signals_are_json_serializable():
    signals = run_strategy_3(_ctx("strategy_3", lookback_days=800, weekly_sma=26))
    assert signals, "expected SP-3 to emit a signal"
    for s in signals:
        d = s.to_jsonl_dict()
        json.dumps(d)
        assert type(d["metadata"]["in_uptrend"]) is bool


def test_pack_signal_fallback_coerces_numpy_scalars():
    sig = PackSignal(
        strategy_id="SP-X",
        symbol="TEST",
        signal="STATUS",
        metadata={"flag": np.bool_(True), "value": np.float64(1.5), "count": np.int64(3)},
    )
    d = sig.to_jsonl_dict()
    json.dumps(d)
    assert d["metadata"] == {"flag": True, "value": 1.5, "count": 3}

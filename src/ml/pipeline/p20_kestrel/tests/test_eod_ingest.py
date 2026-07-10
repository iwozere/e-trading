"""
Tests for eod_ingest signal computation.

Regression test: signal values derived from pandas reductions (e.g. adv_20d)
were numpy.float64 scalars; under numpy 2.x psycopg2 rendered them as
"np.float64(...)" in SQL, failing the k20_signals upsert with
InvalidSchemaName: schema "np" does not exist.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("talib")

from src.ml.pipeline.p20_kestrel.ingest.eod_ingest import _compute_signals_for_ticker


def _make_ohlcv(rows: int = 250) -> pd.DataFrame:
    """Build a synthetic upward-trending OHLCV frame."""
    rng = np.random.default_rng(42)
    index = pd.bdate_range(end="2026-07-08", periods=rows)
    close = pd.Series(100.0 + np.arange(rows) * 0.1 + rng.normal(0, 0.5, rows), index=index)
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, rows).astype("int64"),
        },
        index=index,
    )


def test_compute_signals_returns_expected_types():
    rows = _compute_signals_for_ticker("TEST", _make_ohlcv(), date(2026, 7, 8))

    assert rows, "expected signal rows for a 250-bar frame"
    signal_types = {r["signal_type"] for r in rows}
    assert {"close", "adv_20d", "return_3m", "return_6m", "two_yr_high"} <= signal_types

    for row in rows:
        # Must be builtin float, not numpy.float64 — psycopg2 binds float
        # subclasses via repr(), which is not a SQL literal under numpy 2.x.
        assert type(row["value"]) is float, f"{row['signal_type']} value is {type(row['value'])!r}"


def test_compute_signals_short_history_returns_empty():
    assert _compute_signals_for_ticker("TEST", _make_ohlcv(10), date(2026, 7, 8)) == []

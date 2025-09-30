# ---------------------------------------------------------------------------
# src/indicators/adapters/ta_lib_adapter.py
# TA-Lib adapter with canonical → TA-Lib param translation
# + strict required-input validation (raise KeyError if missing)
# ---------------------------------------------------------------------------
from __future__ import annotations

from typing import Dict, Any, Optional, Iterable
import pandas as pd
import numpy as np

try:
    import talib  # type: ignore
except Exception as e:
    raise ImportError(
        "TA-Lib is required for TaLibAdapter. Install with `pip install TA-Lib`."
    ) from e

from src.indicators.adapters.base import BaseAdapter  # optional base


# ---- Required inputs per indicator (strict; from inputs dict only) ----------
_REQUIRED_INPUTS: dict[str, Iterable[str]] = {
    # single series
    "rsi": {"close"},
    "ema": {"close"},
    "sma": {"close"},
    "obv": {"close", "volume"},
    # hlc trio
    "atr": {"high", "low", "close"},
    "adx": {"high", "low", "close"},
    "plus_di": {"high", "low", "close"},
    "minus_di": {"high", "low", "close"},
    "macd": {"close"},
    "bbands": {"close"},
    "stoch": {"high", "low", "close"},
}


def _ensure_series(x, index: pd.Index) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.reindex(index)
    return pd.Series(np.asarray(x), index=index)


def _validate_and_collect_inputs(indicator: str, inputs: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    needed = _REQUIRED_INPUTS.get(indicator, set())
    missing = [k for k in needed if k not in inputs]
    if missing:
        raise KeyError(f"{indicator}: missing required input(s): {', '.join(missing)}")

    # also sanity: ensure all requested keys are Series and aligned
    out: Dict[str, pd.Series] = {}
    for k in needed:
        s = inputs[k]
        if not isinstance(s, pd.Series):
            raise TypeError(f"{indicator}: input '{k}' must be pandas.Series")
        out[k] = s
    return out


# ---- Canonical → TA-Lib argument maps ---------------------------------------
_CANONICAL_TO_TALIB: dict[str, dict[str, str]] = {
    "ema": {"length": "timeperiod"},
    "sma": {"length": "timeperiod"},
    "rsi": {"length": "timeperiod"},
    "atr": {"length": "timeperiod"},
    "adx": {"length": "timeperiod"},
    "plus_di": {"length": "timeperiod"},
    "minus_di": {"length": "timeperiod"},

    "macd": {"fast": "fastperiod", "slow": "slowperiod", "signal": "signalperiod"},

    "stoch": {
        "k": "fastk_period",
        "d": "slowd_period",
        "smooth_k": "slowk_period",
        # k_ma/d_ma handled separately to matype ints
    },

    "bbands": {"length": "timeperiod", "std_up": "nbdevup", "std_down": "nbdevdn"},
}

_MATYPE: dict[str, int] = {
    "sma": 0, "ema": 1, "wma": 2, "dema": 3, "tema": 4,
    "trima": 5, "kama": 6, "mama": 7, "t3": 8,
}


def _xlate(indicator: str, params: Optional[dict]) -> dict:
    if not params:
        return {}
    mapping = _CANONICAL_TO_TALIB.get(indicator, {})
    out: dict = {}
    for k, v in params.items():
        out[mapping.get(k, k)] = v

    if indicator == "bbands":
        if "std" in params:
            out.setdefault("nbdevup", params["std"])
            out.setdefault("nbdevdn", params["std"])

    if indicator == "stoch":
        k_ma = params.get("k_ma")
        d_ma = params.get("d_ma")
        if isinstance(k_ma, str):
            out["slowk_matype"] = _MATYPE.get(k_ma.lower(), 0)
        if isinstance(d_ma, str):
            out["slowd_matype"] = _MATYPE.get(d_ma.lower(), 0)

    return out


class TaLibAdapter(BaseAdapter):
    """Adapter using TA-Lib backend with strict input validation."""
    _map = {
        "rsi": talib.RSI,
        "ema": talib.EMA,
        "sma": talib.SMA,
        "atr": talib.ATR,
        "adx": talib.ADX,
        "plus_di": talib.PLUS_DI,
        "minus_di": talib.MINUS_DI,
        "obv": talib.OBV,
        "bbands": talib.BBANDS,
        "macd": talib.MACD,
        "stoch": talib.STOCH,
    }

    def supports(self, name: str) -> bool:
        return name in self._map

    def compute(
        self,
        name: str,
        df: pd.DataFrame,
        inputs: Dict[str, pd.Series],
        params: Optional[dict]
    ) -> Dict[str, pd.Series]:
        src = _validate_and_collect_inputs(name, inputs)
        p = _xlate(name, params)
        fn = self._map[name]

        if name == "bbands":
            u, m, l = fn(src["close"].values.astype(float), **p)
            return {"upper": _ensure_series(u, df.index),
                    "middle": _ensure_series(m, df.index),
                    "lower": _ensure_series(l, df.index)}

        if name == "macd":
            macd, signal, hist = fn(src["close"].values.astype(float), **p)
            return {"macd": _ensure_series(macd, df.index),
                    "signal": _ensure_series(signal, df.index),
                    "hist": _ensure_series(hist, df.index)}

        if name == "stoch":
            k, d = fn(
                src["high"].values.astype(float),
                src["low"].values.astype(float),
                src["close"].values.astype(float),
                **p,
            )
            return {"k": _ensure_series(k, df.index),
                    "d": _ensure_series(d, df.index)}

        if name in ("atr", "adx", "plus_di", "minus_di"):
            v = fn(
                src["high"].values.astype(float),
                src["low"].values.astype(float),
                src["close"].values.astype(float),
                **p,
            )
            return {"value": _ensure_series(v, df.index)}

        if name == "obv":
            v = fn(
                src["close"].values.astype(float),
                src["volume"].values.astype(float),
                **(p or {}),
            )
            return {"value": _ensure_series(v, df.index)}

        # default single-series (close)
        v = fn(src["close"].values.astype(float), **p)
        return {"value": _ensure_series(v, df.index)}

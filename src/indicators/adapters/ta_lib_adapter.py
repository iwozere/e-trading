# ---------------------------------------------------------------------------
# src/indicators/adapters/ta_lib_adapter.py
# TA-Lib adapter with canonical → TA-Lib param translation
# + strict required-input validation (raise KeyError if missing)
# ---------------------------------------------------------------------------
from __future__ import annotations

from typing import Dict, Optional
import pandas as pd
import numpy as np

try:
    import talib  # type: ignore
except Exception as e:
    raise ImportError(
        "TA-Lib is required for TaLibAdapter. Install with `pip install TA-Lib`."
    ) from e

from src.indicators.adapters.base import BaseAdapter  # optional base
from src.indicators.registry import INDICATOR_META

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


def _validate_and_collect_inputs(indicator: str, inputs: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    # Use registry instead of local _REQUIRED_INPUTS
    meta = INDICATOR_META.get(indicator)
    if not meta:
        raise ValueError(f"Unknown indicator: {indicator}")

    needed = set(meta.inputs)  # Use from registry
    missing = [k for k in needed if k not in inputs]
    if missing:
        raise KeyError(f"{indicator}: missing required input(s): {', '.join(missing)}")

    out: Dict[str, pd.Series] = {}
    for k in needed:
        s = inputs[k]
        if not isinstance(s, pd.Series):
            raise TypeError(f"{indicator}: input '{k}' must be pandas.Series")
        out[k] = s
    return out

def _ensure_series(x, index: pd.Index) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.reindex(index)
    return pd.Series(np.asarray(x), index=index)


# ---- Canonical → TA-Lib argument maps ---------------------------------------
_CANONICAL_TO_TALIB: dict[str, dict[str, str]] = {
    "ema": {"length": "timeperiod"},
    "sma": {"length": "timeperiod"},
    "rsi": {"length": "timeperiod"},
    "atr": {"length": "timeperiod"},
    "adx": {"length": "timeperiod"},
    "plus_di": {"length": "timeperiod"},
    "minus_di": {"length": "timeperiod"},
    "cci": {"length": "timeperiod"},
    "roc": {"length": "timeperiod"},
    "mfi": {"length": "timeperiod"},
    "williams_r": {"length": "timeperiod"},
    "aroon": {"length": "timeperiod"},
    "adr": {"length": "timeperiod"},

    "macd": {"fast": "fastperiod", "slow": "slowperiod", "signal": "signalperiod"},

    "stoch": {
        "k": "fastk_period",
        "d": "slowd_period",
        "smooth_k": "slowk_period",
        # k_ma/d_ma handled separately to matype ints
    },

    "bbands": {"length": "timeperiod", "std_up": "nbdevup", "std_down": "nbdevdn"},

    "sar": {"acceleration": "acceleration", "maximum": "maximum"},

    "adosc": {"fast": "fastperiod", "slow": "slowperiod"},

    # Custom indicators
    "ichimoku": {"tenkan": "tenkan", "kijun": "kijun", "senkou": "senkou"},
    "super_trend": {"length": "length", "multiplier": "multiplier"},
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
        # Basic indicators
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
        # Additional technical indicators
        "cci": talib.CCI,
        "roc": talib.ROC,
        "mfi": talib.MFI,
        "williams_r": talib.WILLR,
        "aroon": talib.AROON,
        "sar": talib.SAR,
        "ad": talib.AD,
        "adosc": talib.ADOSC,
        "bop": talib.BOP,
        # Custom implementations for indicators not directly in TA-Lib
        "adr": "_calculate_adr",
        "ichimoku": "_calculate_ichimoku",
        "super_trend": "_calculate_super_trend",
    }

    def supports(self, name: str) -> bool:
        return name in self._map

    async def compute(
        self,
        name: str,
        df: pd.DataFrame,
        inputs: Dict[str, pd.Series],
        params: Optional[dict]
    ) -> Dict[str, pd.Series]:
        """Wrap synchronous TA-Lib calls in async context"""
        # TA-Lib is CPU-bound, run in thread pool to not block event loop
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._compute_sync,
            name, df, inputs, params
        )

    def _compute_sync(
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

        if name == "aroon":
            aroon_down, aroon_up = fn(
                src["high"].values.astype(float),
                src["low"].values.astype(float),
                **p,
            )
            return {"aroon_up": _ensure_series(aroon_up, df.index),
                    "aroon_down": _ensure_series(aroon_down, df.index)}

        if name == "sar":
            v = fn(
                src["high"].values.astype(float),
                src["low"].values.astype(float),
                **p,
            )
            return {"value": _ensure_series(v, df.index)}

        if name in ("cci", "williams_r"):
            v = fn(
                src["high"].values.astype(float),
                src["low"].values.astype(float),
                src["close"].values.astype(float),
                **p,
            )
            return {"value": _ensure_series(v, df.index)}

        if name == "mfi":
            v = fn(
                src["high"].values.astype(float),
                src["low"].values.astype(float),
                src["close"].values.astype(float),
                src["volume"].values.astype(float),
                **p,
            )
            return {"value": _ensure_series(v, df.index)}

        if name in ("ad", "bop"):
            if name == "ad":
                v = fn(
                    src["high"].values.astype(float),
                    src["low"].values.astype(float),
                    src["close"].values.astype(float),
                    src["volume"].values.astype(float),
                    **(p or {}),
                )
            else:  # bop
                v = fn(
                    src["open"].values.astype(float),
                    src["high"].values.astype(float),
                    src["low"].values.astype(float),
                    src["close"].values.astype(float),
                    **(p or {}),
                )
            return {"value": _ensure_series(v, df.index)}

        if name == "adosc":
            v = fn(
                src["high"].values.astype(float),
                src["low"].values.astype(float),
                src["close"].values.astype(float),
                src["volume"].values.astype(float),
                **p,
            )
            return {"value": _ensure_series(v, df.index)}

        # Handle custom implementations
        if isinstance(fn, str) and fn.startswith("_calculate_"):
            return getattr(self, fn)(src, df.index, p)

        # default single-series (close)
        v = fn(src["close"].values.astype(float), **p)
        return {"value": _ensure_series(v, df.index)}

    def _calculate_adr(self, src: Dict[str, pd.Series], index: pd.Index, params: dict) -> Dict[str, pd.Series]:
        """Calculate Average Daily Range."""
        timeperiod = params.get("timeperiod", 14)
        daily_range = src["high"] - src["low"]
        adr = daily_range.rolling(window=timeperiod).mean()
        return {"value": _ensure_series(adr, index)}

    def _calculate_ichimoku(self, src: Dict[str, pd.Series], index: pd.Index, params: dict) -> Dict[str, pd.Series]:
        """Calculate Ichimoku Cloud components."""
        tenkan_period = params.get("tenkan", 9)
        kijun_period = params.get("kijun", 26)
        senkou_period = params.get("senkou", 52)

        high = src["high"]
        low = src["low"]
        close = src["close"]

        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=tenkan_period).max()
        tenkan_low = low.rolling(window=tenkan_period).min()
        tenkan = (tenkan_high + tenkan_low) / 2

        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=kijun_period).max()
        kijun_low = low.rolling(window=kijun_period).min()
        kijun = (kijun_high + kijun_low) / 2

        # Senkou Span A (Leading Span A)
        senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)

        # Senkou Span B (Leading Span B)
        senkou_high = high.rolling(window=senkou_period).max()
        senkou_low = low.rolling(window=senkou_period).min()
        senkou_b = ((senkou_high + senkou_low) / 2).shift(kijun_period)

        # Chikou Span (Lagging Span)
        chikou = close.shift(-kijun_period)

        return {
            "tenkan": _ensure_series(tenkan, index),
            "kijun": _ensure_series(kijun, index),
            "senkou_a": _ensure_series(senkou_a, index),
            "senkou_b": _ensure_series(senkou_b, index),
            "chikou": _ensure_series(chikou, index)
        }

    def _calculate_super_trend(self, src: Dict[str, pd.Series], index: pd.Index, params: dict) -> Dict[str, pd.Series]:
        """Calculate Super Trend indicator."""
        length = params.get("length", 10)
        multiplier = params.get("multiplier", 3.0)

        high = src["high"]
        low = src["low"]
        close = src["close"]

        # Calculate ATR
        atr = talib.ATR(high.values.astype(float), low.values.astype(float), close.values.astype(float), timeperiod=length)
        atr_series = pd.Series(atr, index=index)

        # Calculate HL2 (median price)
        hl2 = (high + low) / 2

        # Calculate upper and lower bands
        upper_band = hl2 + (multiplier * atr_series)
        lower_band = hl2 - (multiplier * atr_series)

        # Initialize super trend
        super_trend = pd.Series(index=index, dtype=float)
        trend = pd.Series(index=index, dtype=int)

        for i in range(1, len(close)):
            # Upper band calculation
            if upper_band.iloc[i] < upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]:
                upper_band.iloc[i] = upper_band.iloc[i]
            else:
                upper_band.iloc[i] = upper_band.iloc[i-1]

            # Lower band calculation
            if lower_band.iloc[i] > lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]:
                lower_band.iloc[i] = lower_band.iloc[i]
            else:
                lower_band.iloc[i] = lower_band.iloc[i-1]

            # Super trend calculation
            if i == 1:
                super_trend.iloc[i] = upper_band.iloc[i]
                trend.iloc[i] = 1
            else:
                if super_trend.iloc[i-1] == upper_band.iloc[i-1] and close.iloc[i] <= upper_band.iloc[i]:
                    super_trend.iloc[i] = upper_band.iloc[i]
                    trend.iloc[i] = 1
                elif super_trend.iloc[i-1] == upper_band.iloc[i-1] and close.iloc[i] > upper_band.iloc[i]:
                    super_trend.iloc[i] = lower_band.iloc[i]
                    trend.iloc[i] = -1
                elif super_trend.iloc[i-1] == lower_band.iloc[i-1] and close.iloc[i] >= lower_band.iloc[i]:
                    super_trend.iloc[i] = lower_band.iloc[i]
                    trend.iloc[i] = -1
                elif super_trend.iloc[i-1] == lower_band.iloc[i-1] and close.iloc[i] < lower_band.iloc[i]:
                    super_trend.iloc[i] = upper_band.iloc[i]
                    trend.iloc[i] = 1

        return {
            "value": _ensure_series(super_trend, index),
            "trend": _ensure_series(trend, index)
        }

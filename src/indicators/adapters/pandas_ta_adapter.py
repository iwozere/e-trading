# ---------------------------------------------------------------------------
# src/indicators/adapters/pandas_ta_adapter.py
# pandas-ta adapter with canonical param normalization
# + strict required-input validation (raise KeyError if missing)
# ---------------------------------------------------------------------------
from __future__ import annotations

from typing import Dict, Optional
import pandas as pd

try:
    import pandas_ta as pta  # type: ignore
except Exception as e:
    raise ImportError(
        "pandas-ta is required for PandasTaAdapter. Install with `pip install pandas_ta`."
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

def _norm_params(indicator: str, params: Optional[dict]) -> dict:
    """Canonical → pandas_ta param normalization (mostly pass-through)."""
    p = dict(params or {})

    if indicator == "bbands":
        std_up = p.pop("std_up", None)
        std_dn = p.pop("std_down", None)
        if "std" not in p and std_up is not None and std_dn is not None and std_up == std_dn:
            p["std"] = std_up

    if indicator in ("ema", "sma", "rsi", "atr", "adx", "plus_di", "minus_di"):
        if "length" not in p and "timeperiod" in p:
            p["length"] = p.pop("timeperiod")

    if indicator == "macd":
        if "fast" not in p and "fastperiod" in p:
            p["fast"] = p.pop("fastperiod")
        if "slow" not in p and "slowperiod" in p:
            p["slow"] = p.pop("slowperiod")
        if "signal" not in p and "signalperiod" in p:
            p["signal"] = p.pop("signalperiod")

    if indicator == "stoch":
        if "k" not in p and "fastk_period" in p:
            p["k"] = p.pop("fastk_period")
        if "d" not in p and "slowd_period" in p:
            p["d"] = p.pop("slowd_period")
        if "smooth_k" not in p and "slowk_period" in p:
            p["smooth_k"] = p.pop("slowk_period")
        # 'mamode' may be provided; pass-through

    return p


def _as_series(x, name: str, index: pd.Index) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.rename(name).reindex(index)
    if hasattr(x, "reindex"):
        # DataFrame: take first column by convention
        s = x.iloc[:, 0]
        return s.rename(name).reindex(index)
    return pd.Series(x, index=index, name=name)


class PandasTaAdapter(BaseAdapter):
    """Adapter using pandas-ta backend with strict input validation."""

    def supports(self, name: str) -> bool:
        return hasattr(pta, name) or name in ("bbands", "stoch", "plus_di", "minus_di")

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
        p = _norm_params(name, params)

        close = src.get("close")
        high = src.get("high")
        low = src.get("low")
        volume = src.get("volume")

        # Structured outputs
        if name == "macd":
            macd_df = getattr(pta, "macd")(close, **p)
            cols = {c.lower(): c for c in macd_df.columns}
            macd_col = next((c for k, c in cols.items() if "macd" in k and "h" not in k and "s" not in k), macd_df.columns[0])
            signal_col = next((c for k, c in cols.items() if "macds" in k or "signal" in k), macd_df.columns[-1])
            hist_col = next((c for k, c in cols.items() if "macdh" in k or "hist" in k),
                            macd_df.columns[1] if macd_df.shape[1] > 1 else macd_df.columns[0])
            return {
                "macd": macd_df[macd_col].rename("macd").reindex(df.index),
                "signal": macd_df[signal_col].rename("signal").reindex(df.index),
                "hist": macd_df[hist_col].rename("hist").reindex(df.index),
            }

        if name == "bbands":
            bb = getattr(pta, "bbands")(close, **p)  # often [BBL, BBM, BBU]
            if bb.shape[1] >= 3:
                lower = bb.iloc[:, 0]; middle = bb.iloc[:, 1]; upper = bb.iloc[:, 2]
                return {
                    "upper": upper.rename("upper").reindex(df.index),
                    "middle": middle.rename("middle").reindex(df.index),
                    "lower": lower.rename("lower").reindex(df.index),
                }
            cols = {c.lower(): c for c in bb.columns}
            return {
                "upper": bb[cols.get("bbu", bb.columns[-1])].rename("upper").reindex(df.index),
                "middle": bb[cols.get("bbm", bb.columns[0])].rename("middle").reindex(df.index),
                "lower": bb[cols.get("bbl", bb.columns[0])].rename("lower").reindex(df.index),
            }

        if name == "stoch":
            st = getattr(pta, "stoch")(high=high, low=low, close=close, **p)  # returns k,d
            k = st.iloc[:, 0].rename("k").reindex(df.index)
            d = st.iloc[:, 1].rename("d").reindex(df.index)
            return {"k": k, "d": d}

        if name == "plus_di":
            if hasattr(pta, "plus_di"):
                s = getattr(pta, "plus_di")(high=high, low=low, close=close, **p)
                return {"value": _as_series(s, "value", df.index)}
            if hasattr(pta, "di"):
                di_df = getattr(pta, "di")(high=high, low=low, close=close, **p)
                col = next((c for c in di_df.columns if "DMP" in c or "+DI" in c or "DIp" in c), di_df.columns[0])
                return {"value": di_df[col].rename("value").reindex(df.index)}
            adx_df = getattr(pta, "adx")(high=high, low=low, close=close, **p)
            col = next((c for c in adx_df.columns if "DMP" in c or "+DI" in c or "DIp" in c), adx_df.columns[0])
            return {"value": adx_df[col].rename("value").reindex(df.index)}

        if name == "minus_di":
            if hasattr(pta, "minus_di"):
                s = getattr(pta, "minus_di")(high=high, low=low, close=close, **p)
                return {"value": _as_series(s, "value", df.index)}
            if hasattr(pta, "di"):
                di_df = getattr(pta, "di")(high=high, low=low, close=close, **p)
                col = next((c for c in di_df.columns if "DMN" in c or "-DI" in c or "DIn" in c), di_df.columns[-1])
                return {"value": di_df[col].rename("value").reindex(df.index)}
            adx_df = getattr(pta, "adx")(high=high, low=low, close=close, **p)
            col = next((c for c in adx_df.columns if "DMN" in c or "-DI" in c or "DIn" in c), adx_df.columns[-1])
            return {"value": adx_df[col].rename("value").reindex(df.index)}

        # Single-series
        if name == "ema":
            s = getattr(pta, "ema")(close, **p)
            return {"value": _as_series(s, "value", df.index)}

        if name == "sma":
            s = getattr(pta, "sma")(close, **p)
            return {"value": _as_series(s, "value", df.index)}

        if name == "rsi":
            s = getattr(pta, "rsi")(close, **p)
            return {"value": _as_series(s, "value", df.index)}

        if name == "atr":
            s = getattr(pta, "atr")(high=high, low=low, close=close, **p)
            return {"value": _as_series(s, "value", df.index)}

        if name == "adx":
            adx_df = getattr(pta, "adx")(high=high, low=low, close=close, **p)
            col = next((c for c in adx_df.columns if "ADX" in c.upper()), adx_df.columns[0])
            return {"value": adx_df[col].rename("value").reindex(df.index)}

        if name == "obv":
            s = getattr(pta, "obv")(close=close, volume=volume)
            return {"value": _as_series(s, "value", df.index)}

        if name == "super_trend":
            st = getattr(pta, "supertrend")(high=high, low=low, close=close, **p)
            # SuperTrend typically returns [SUPERT, SUPERTd, SUPERTl, SUPERTs]
            # where SUPERT is the main line and SUPERTd is the direction
            if st.shape[1] >= 2:
                value = st.iloc[:, 0].rename("value").reindex(df.index)
                trend = st.iloc[:, 1].rename("trend").reindex(df.index)
                return {"value": value, "trend": trend}
            else:
                return {"value": st.iloc[:, 0].rename("value").reindex(df.index)}

        # Fallback: try function by name
        if hasattr(pta, name):
            fn = getattr(pta, name)
            res = fn(close=close, high=high, low=low, volume=volume, **p)
            if isinstance(res, pd.DataFrame):
                return {"value": res.iloc[:, 0].rename("value").reindex(df.index)}
            return {"value": _as_series(res, "value", df.index)}

        raise NotImplementedError(f"pandas_ta cannot compute '{name}'")

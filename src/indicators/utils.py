# ---------------------------------------------------------------------------
# utils.py
# ---------------------------------------------------------------------------
import pandas as pd

def coerce_ohlcv(df: pd.DataFrame, input_map=None) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    mapping = input_map or {}
    for k,v in mapping.items():
        if v in out.columns:
            out.rename(columns={v:k}, inplace=True)
    for c in ["open","high","low","close","volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def resample_df(df, timeframe: str | None):
    if not timeframe: return df
    ohlc = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    return df.resample(timeframe).agg(ohlc).dropna(how="all")

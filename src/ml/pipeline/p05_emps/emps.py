# emps.py
"""
EMPS — Explosive Move Probability Score module

Provides:
- component detectors:
    * volume z-score (intraday)
    * vwap deviation
    * realized volatility short vs long
    * breakout detector (multi-day)
    * liquidity/float score
    * social proxy (stocktwits message count)
- normalization and weighted combination into emps_score (0..1)
- flags: explosion_flag (soft), hard_flag (strict)
- helper: scan_and_score(ticker, fetch_intraday_fn, meta) to run end-to-end for one ticker

Dependencies: pandas, numpy, requests, yfinance(optional for sample fetch)
"""
from typing import Callable, Dict, Any, Optional, Tuple
import math
import numpy as np
import pandas as pd
import requests
from datetime import timedelta

# ------------------------- Defaults / thresholds -------------------------
DEFAULTS = {
    # lookbacks in bars for intraday series (e.g. 5m bars)
    'vol_lookback': 60, # ~5 hours with 5m bars
    'vwap_lookback': 60,
    'rv_short_window': 15,
    'rv_long_window': 120,
    # thresholds for component interpretation
    'vol_zscore_thresh': 4.0,
    'vwap_dev_thresh': 0.03, # 3%
    'rv_ratio_thresh': 1.8,
    # combined threshold for soft alert (score 0..1)
    'combined_score_thresh': 0.6,
    # weights for combining components
    'weights': {
        'vol': 0.45,
        'vwap': 0.25,
        'rv': 0.25,
        'liquidity': 0.05, # small weight; liquidity is a gating factor
        # social optional and treated separately
    },
    # hard flag requires each component to exceed these tighter thresholds
    'hard': {
        'vol_zscore': 6.0,
        'vwap_dev': 0.05,
        'rv_ratio': 2.2,
    }
}

# ------------------------- Utilities -------------------------
def safe_div(a, b):
    try:
        return a / b
    except Exception:
        return float('nan')

def fetch_stocktwits_count(ticker: str, timeout: float = 6.0) -> Optional[int]:
    """Simple Stocktwits messages count as social proxy. Returns count or None."""
    try:
        url = f'https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json'
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        j = r.json()
        msgs = j.get('messages')
        return len(msgs) if msgs is not None else None
    except Exception:
        return None

# ------------------------- Feature calculators -------------------------
def compute_volume_zscore(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Volume z-score: (V - mean)/std over rolling lookback. Fill NaN with 0."""
    vol = df['Volume'].astype(float)
    mean = vol.rolling(window=lookback, min_periods=max(5, lookback//6)).mean()
    std = vol.rolling(window=lookback, min_periods=max(5, lookback//6)).std().replace(0, np.nan)
    z = (vol - mean) / std
    return z.fillna(0.0)

def compute_vwap_deviation(df: pd.DataFrame, lookback_vwap: int) -> pd.Series:
    """Rolling VWAP deviation: (Close - rollingVWAP) / rollingVWAP."""
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    pv = tp * df['Volume']
    rolling_pv = pv.rolling(window=lookback_vwap, min_periods=max(3, lookback_vwap//10)).sum()
    rolling_vol = df['Volume'].rolling(window=lookback_vwap, min_periods=max(3, lookback_vwap//10)).sum().replace(0, np.nan)
    rolling_vwap = rolling_pv / rolling_vol
    dev = (df['Close'] - rolling_vwap) / rolling_vwap
    return dev.fillna(0.0)

def realized_volatility(series_close: pd.Series, window: int) -> pd.Series:
    """Realized volatility (annualized) from log returns over 'window' bars.
    Annualization uses approximate bars-per-day inferred from median index diff.
    """
    if series_close.index.nlevels == 0:
        # estimate minutes per bar
        diffs = series_close.index.to_series().diff().dropna().dt.total_seconds() / 60.0
        median_min = float(diffs.median()) if not diffs.empty else 5.0
    else:
        diffs = series_close.index.to_series().diff().dropna().dt.total_seconds() / 60.0
        median_min = float(diffs.median()) if not diffs.empty else 5.0

    bars_per_day = max(1.0, 390.0 / median_min) # 6.5h trading day ≈ 390 minutes
    annual_factor = math.sqrt(252.0 * bars_per_day)
    logret = np.log(series_close).diff()
    rv = logret.rolling(window=window, min_periods=max(5, window//6)).std() * annual_factor
    return rv.fillna(0.0)

def breakout_detector(df_daily: pd.DataFrame, lookback_days: int = 20) -> Tuple[pd.Series, pd.Series]:
    """
    Simple breakout detector on daily bars:
    - breakout_high: current close > max(high[-lookback_days:-1])
    - breakout_vol_spike: today's volume > avg(volume[-lookback_days:-1]) * 2
    Returns two boolean series aligned to df_daily index.
    """
    high_max = df_daily['High'].rolling(window=lookback_days, min_periods=5).max().shift(1)
    breakout_high = df_daily['Close'] > high_max
    vol_avg = df_daily['Volume'].rolling(window=lookback_days, min_periods=5).mean().shift(1)
    breakout_vol_spike = df_daily['Volume'] > (vol_avg * 2.0)
    return breakout_high.fillna(False), breakout_vol_spike.fillna(False)

# ------------------------- Normalizers / scorers -------------------------
def map_vol_zscore_to_score(z: float, thresh: float = 4.0) -> float:
    """
    Map z-score to 0..1 with sigmoid-ish curve.
    z near thresh -> ~0.5, higher -> approaches 1.
    """
    try:
        return float(1.0 / (1.0 + math.exp(-(z - thresh/2.0))))
    except Exception:
        return 0.0

def map_vwap_dev_to_score(dev: float, sat: float = 0.10) -> float:
    """Map absolute VWAP deviation to 0..1, saturation at sat (e.g., 10%)."""
    try:
        return float(min(1.0, abs(dev) / sat))
    except Exception:
        return 0.0

def map_rv_ratio_to_score(ratio: float, low: float = 1.0, high: float = 3.0) -> float:
    """Map rv_ratio into 0..1 where low->0, high->1 (clipped)."""
    try:
        if math.isnan(ratio) or ratio <= low:
            return 0.0
        return float(min(1.0, (ratio - low) / (high - low)))
    except Exception:
        return 0.0

def liquidity_score(market_cap: Optional[float], float_shares: Optional[float], avg_volume: Optional[float]) -> float:
    """
    Small score 0..1 that is high when the ticker is in 'liquidity sweet spot':
    - not too big (<= 5B) and not too small (market cap > e.g. 20M),
    - float below a cap (e.g., 50M),
    - avg_volume in 0.5M..20M band
    """
    s = 0.0
    try:
        if market_cap is None or float_shares is None or avg_volume is None:
            return 0.0
        # market cap factor
        if 20_000_000 <= market_cap <= 5_000_000_000:
            s += 0.4
        # float factor
        if float_shares <= 50_000_000:
            s += 0.4
        # avg volume factor
        if 500_000 <= avg_volume <= 20_000_000:
            s += 0.2
        return min(1.0, s)
    except Exception:
        return 0.0

# ------------------------- Main EMPS combo -------------------------
def compute_emps_from_intraday(
    df_intraday: pd.DataFrame,
    *,
    market_cap: Optional[float] = None,
    float_shares: Optional[float] = None,
    avg_volume: Optional[float] = None,
    ticker: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Given intraday dataframe with columns ['Open','High','Low','Close','Volume'] and
    optional meta (market_cap, float_shares, avg_volume), compute EMPS series.
    Returns copy of df_intraday with added columns:
      - vol_zscore, vol_score
      - vwap_dev, vwap_score
      - rv_short, rv_long, rv_ratio, rv_score
      - liquidity_score
      - emps_score (0..1)
      - explosion_flag (bool), hard_flag (bool)
    The final row contains the most recent EMPS.
    """
    if params is None:
        params = DEFAULTS

    p = {**DEFAULTS, **params} # override defaults if provided
    weights = p['weights']

    df = df_intraday.copy()
    # ensure numeric columns
    for col in ['Open','High','Low','Close','Volume']:
        if col not in df.columns:
            raise ValueError(f"Intraday DF must contain column: {col}")
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0.0)
    # compute components
    df['vol_zscore'] = compute_volume_zscore(df, lookback=p['vol_lookback'])
    df['vol_score'] = df['vol_zscore'].apply(lambda z: map_vol_zscore_to_score(z, p['vol_zscore_thresh']))
    df['vwap_dev'] = compute_vwap_deviation(df, lookback_vwap=p['vwap_lookback'])
    df['vwap_score'] = df['vwap_dev'].apply(lambda d: map_vwap_dev_to_score(d, sat=0.10))
    df['rv_short'] = realized_volatility(df['Close'], window=p['rv_short_window'])
    df['rv_long'] = realized_volatility(df['Close'], window=p['rv_long_window'])
    df['rv_ratio'] = df['rv_short'] / df['rv_long'].replace(0, np.nan)
    df['rv_ratio'] = df['rv_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df['rv_score'] = df['rv_ratio'].apply(lambda r: map_rv_ratio_to_score(r, low=1.0, high=3.0))

    # liquidity score is constant per-ticker, copied across rows
    liq = liquidity_score(market_cap, float_shares, avg_volume)
    df['liquidity_score'] = liq

    # combine with weights
    df['emps_score'] = (
        weights['vol'] * df['vol_score']
        + weights['vwap'] * df['vwap_score']
        + weights['rv'] * df['rv_score']
        + weights.get('liquidity', 0.0) * df['liquidity_score']
    )

    # optional social signal adds a small boost if present (we don't fetch inside this function)
    # df['social_score'] can be set externally and added to emps_score if desired

    # flags
    df['explosion_flag'] = df['emps_score'] >= p['combined_score_thresh']
    df['hard_flag'] = (
        (df['vol_zscore'] >= p['hard']['vol_zscore']) &
        (df['vwap_dev'].abs() >= p['hard']['vwap_dev']) &
        (df['rv_ratio'] >= p['hard']['rv_ratio'])
    )

    # mark recent timestamp convenience columns
    df['emps_timestamp'] = pd.Timestamp.utcnow()
    df['ticker'] = ticker

    return df

# ------------------------- End-to-end helper -------------------------
def scan_and_score(
    ticker: str,
    fetch_intraday_fn: Callable[[str, Dict[str, Any]], pd.DataFrame],
    meta: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    fetch_intraday_fn(ticker, fetch_kwargs) -> DataFrame (intraday)
    meta: dict with keys market_cap, float_shares, avg_volume, optional social_count
    Returns dataframe with EMPS columns.
    """
    df = fetch_intraday_fn(ticker, {}) # fetch_kwargs can be passed if needed
    if df is None or df.empty:
        raise RuntimeError(f"No intraday data for {ticker}")
    df_scored = compute_emps_from_intraday(
        df,
        market_cap=meta.get('market_cap'),
        float_shares=meta.get('float_shares'),
        avg_volume=meta.get('avg_volume'),
        ticker=ticker,
        params=params
    )
    # optionally add social_score boost (if provided)
    social = meta.get('social_count')
    if social is not None:
        social_norm = min(1.0, social / 50.0)
        # small additive boost: up to +0.05 to emps_score (tunable)
        df_scored['emps_score'] = df_scored['emps_score'] + 0.05 * social_norm
        # re-evaluate explosion_flag
        combined_thresh = (params or DEFAULTS)['combined_score_thresh'] if params else DEFAULTS['combined_score_thresh']
        df_scored['explosion_flag'] = df_scored['emps_score'] >= combined_thresh

    return df_scored

# ------------------------- CLI example (if run as script) -------------------------
if __name__ == "__main__":
    import argparse
    import yfinance as yf

    def yf_fetch_intraday(ticker: str, fetch_kwargs: Dict[str, Any]) -> pd.DataFrame:
        # default example: 7 days, 5m bars
        period = fetch_kwargs.get('period', '7d')
        interval = fetch_kwargs.get('interval', '5m')
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval=interval, auto_adjust=False)
        return hist

    parser = argparse.ArgumentParser(description="EMPS tester CLI")
    parser.add_argument('ticker', type=str)
    parser.add_argument('--period', type=str, default='7d')
    parser.add_argument('--interval', type=str, default='5m')
    parser.add_argument('--market-cap', type=float, default=None)
    parser.add_argument('--float-shares', type=float, default=None)
    parser.add_argument('--avg-volume', type=float, default=None)
    args = parser.parse_args()

    kwargs = {'period': args.period, 'interval': args.interval}
    df_intraday = yf_fetch_intraday(args.ticker, kwargs)
    if df_intraday.empty:
        print("No intraday data. Try different period/interval or data source.")
        exit(1)

    meta = {'market_cap': args.market_cap, 'float_shares': args.float_shares, 'avg_volume': args.avg_volume}
    df_scored = scan_and_score(args.ticker, yf_fetch_intraday, meta)
    # print last few rows with important columns
    cols = ['Close','Volume','vol_zscore','vwap_dev','rv_short','rv_long','rv_ratio','vol_score','vwap_score','rv_score','emps_score','explosion_flag','hard_flag']
    print(df_scored[cols].tail(20).to_string())
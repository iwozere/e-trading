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
from pathlib import Path
import sys

# Setup logging
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from src.notification.logger import setup_logger

logger = setup_logger(__name__)

# ------------------------- Defaults / thresholds -------------------------
DEFAULTS = {
    # lookbacks in bars for intraday series (e.g. 5m bars)
    'vol_lookback': 60,  # ~5 hours with 5m bars
    'vwap_lookback': 60,
    'rv_short_window': 15,
    'rv_long_window': 120,

    # thresholds for component interpretation
    'vol_zscore_thresh': 4.0,
    'vwap_dev_thresh': 0.03,  # 3%
    'rv_ratio_thresh': 1.8,

    # combined threshold for soft alert (score 0..1)
    'combined_score_thresh': 0.6,

    # weights for combining components
    'weights': {
        'vol': 0.45,
        'vwap': 0.25,
        'rv': 0.25,
        'liquidity': 0.05,  # small weight; liquidity is a gating factor
    },

    # hard flag requires each component to exceed these tighter thresholds
    'hard': {
        'vol_zscore': 6.0,
        'vwap_dev': 0.05,
        'rv_ratio': 2.2,
    },

    # social score parameters (moved from hardcoded values)
    'social': {
        'normalization_factor': 50.0,  # messages count for 1.0 score
        'weight_boost': 0.05,  # max boost to emps_score
    },

    # liquidity score thresholds (moved from hardcoded values)
    'liquidity_thresholds': {
        'min_market_cap': 20_000_000,  # $20M
        'max_market_cap': 5_000_000_000,  # $5B
        'max_float_shares': 50_000_000,  # 50M shares
        'min_avg_volume': 500_000,  # 500K shares/day
        'max_avg_volume': 20_000_000,  # 20M shares/day
        'weights': {
            'market_cap': 0.4,
            'float': 0.4,
            'volume': 0.2,
        }
    },

    # sigmoid parameters for volume z-score mapping
    'vol_zscore_sigmoid': {
        'center_scale': 0.5,  # multiplier for threshold to find sigmoid center
        'steepness': 1.0,  # controls sigmoid steepness
    },

    # breakout detector parameters
    'breakout': {
        'lookback_days': 20,
        'volume_multiplier': 2.0,
    }
}

# ------------------------- Utilities -------------------------
def safe_div(a, b):
    """Safe division that returns NaN on error."""
    try:
        return a / b if b != 0 else float('nan')
    except (ZeroDivisionError, TypeError, ValueError):
        return float('nan')


def fetch_stocktwits_count(ticker: str, timeout: float = 6.0) -> Optional[int]:
    """
    Fetch Stocktwits messages count as social proxy.

    Args:
        ticker: Stock ticker symbol
        timeout: Request timeout in seconds

    Returns:
        Message count or None if failed
    """
    try:
        url = f'https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json'
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            logger.warning("Stocktwits API returned status %d for %s", r.status_code, ticker)
            return None
        j = r.json()
        msgs = j.get('messages')
        return len(msgs) if msgs is not None else None
    except requests.exceptions.Timeout:
        logger.warning("Stocktwits API timeout for %s", ticker)
        return None
    except requests.exceptions.RequestException as e:
        logger.warning("Stocktwits API request failed for %s: %s", ticker, e)
        return None
    except Exception as e:
        logger.exception("Unexpected error fetching Stocktwits data for %s", ticker)
        return None

# ------------------------- Feature calculators -------------------------
def compute_volume_zscore(df: pd.DataFrame, lookback: int) -> pd.Series:
    """
    Compute volume z-score: (V - mean)/std over rolling lookback.

    Args:
        df: DataFrame with 'Volume' column
        lookback: Rolling window size

    Returns:
        Series of z-scores (NaN for insufficient data, not filled)
    """
    vol = pd.to_numeric(df['Volume'], errors='coerce')
    min_periods = max(10, lookback // 4)  # Require at least 25% of lookback period

    mean = vol.rolling(window=lookback, min_periods=min_periods).mean()
    std = vol.rolling(window=lookback, min_periods=min_periods).std()

    # Avoid division by zero; keep as NaN if std is zero or very small
    z = np.where(std > 1e-8, (vol - mean) / std, np.nan)
    return pd.Series(z, index=df.index)

def compute_vwap_deviation(df: pd.DataFrame, lookback_vwap: int) -> pd.Series:
    """
    Compute rolling VWAP deviation: (Close - rollingVWAP) / rollingVWAP.

    Args:
        df: DataFrame with OHLCV columns
        lookback_vwap: Rolling window size

    Returns:
        Series of VWAP deviations (NaN for insufficient data)
    """
    min_periods = max(5, lookback_vwap // 10)

    # Typical price
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    pv = tp * df['Volume']

    rolling_pv = pv.rolling(window=lookback_vwap, min_periods=min_periods).sum()
    rolling_vol = df['Volume'].rolling(window=lookback_vwap, min_periods=min_periods).sum()

    # Vectorized division with NaN handling
    rolling_vwap = np.where(rolling_vol > 0, rolling_pv / rolling_vol, np.nan)
    rolling_vwap = pd.Series(rolling_vwap, index=df.index)

    dev = np.where(rolling_vwap > 0, (df['Close'] - rolling_vwap) / rolling_vwap, np.nan)
    return pd.Series(dev, index=df.index)

def realized_volatility(series_close: pd.Series, window: int) -> pd.Series:
    """
    Compute realized volatility (annualized) from log returns over 'window' bars.
    Annualization uses bars-per-day inferred from median time diff.

    Args:
        series_close: Series of closing prices with datetime index
        window: Rolling window size

    Returns:
        Series of annualized realized volatility (NaN for insufficient data)
    """
    # Estimate minutes per bar from index
    diffs = series_close.index.to_series().diff().dropna().dt.total_seconds() / 60.0
    median_min = float(diffs.median()) if not diffs.empty else 5.0

    bars_per_day = max(1.0, 390.0 / median_min)  # 6.5h trading day ≈ 390 minutes
    annual_factor = math.sqrt(252.0 * bars_per_day)

    min_periods = max(10, window // 4)
    logret = np.log(series_close).diff()
    rv = logret.rolling(window=window, min_periods=min_periods).std() * annual_factor

    return rv

def breakout_detector(df_daily: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Simple breakout detector on daily bars:
    - breakout_high: current close > max(high[-lookback_days:-1])
    - breakout_vol_spike: today's volume > avg(volume[-lookback_days:-1]) * multiplier

    Args:
        df_daily: DataFrame with daily OHLCV data
        params: Optional dict with 'lookback_days' and 'volume_multiplier'

    Returns:
        Tuple of (breakout_high, breakout_vol_spike) boolean Series
    """
    if params is None:
        params = DEFAULTS['breakout']

    lookback = params.get('lookback_days', 20)
    vol_mult = params.get('volume_multiplier', 2.0)

    min_periods = max(5, lookback // 4)

    high_max = df_daily['High'].rolling(window=lookback, min_periods=min_periods).max().shift(1)
    breakout_high = df_daily['Close'] > high_max

    vol_avg = df_daily['Volume'].rolling(window=lookback, min_periods=min_periods).mean().shift(1)
    breakout_vol_spike = df_daily['Volume'] > (vol_avg * vol_mult)

    return breakout_high.fillna(False), breakout_vol_spike.fillna(False)


# ------------------------- Normalizers / scorers (VECTORIZED) -------------------------
def map_vol_zscore_to_score_vectorized(z: pd.Series, params: Dict[str, Any]) -> pd.Series:
    """
    Map z-score Series to 0..1 with sigmoid curve (vectorized).

    Args:
        z: Series of z-scores
        params: Dict with 'vol_zscore_thresh' and 'vol_zscore_sigmoid' config

    Returns:
        Series of scores in [0, 1]
    """
    thresh = params.get('vol_zscore_thresh', 4.0)
    sigmoid_params = params.get('vol_zscore_sigmoid', {'center_scale': 0.5, 'steepness': 1.0})

    center = thresh * sigmoid_params['center_scale']
    steepness = sigmoid_params['steepness']

    # Vectorized sigmoid: 1 / (1 + exp(-steepness * (z - center)))
    scores = 1.0 / (1.0 + np.exp(-steepness * (z - center)))
    return pd.Series(scores, index=z.index).fillna(0.0)


def map_vwap_dev_to_score_vectorized(dev: pd.Series, sat: float = 0.10) -> pd.Series:
    """
    Map absolute VWAP deviation Series to 0..1, saturation at sat (vectorized).

    Args:
        dev: Series of VWAP deviations
        sat: Saturation level (default 10%)

    Returns:
        Series of scores in [0, 1]
    """
    scores = np.abs(dev) / sat
    return pd.Series(np.minimum(scores, 1.0), index=dev.index).fillna(0.0)


def map_rv_ratio_to_score_vectorized(ratio: pd.Series, low: float = 1.0, high: float = 3.0) -> pd.Series:
    """
    Map RV ratio Series into 0..1 where low->0, high->1 (clipped, vectorized).

    Args:
        ratio: Series of RV ratios
        low: Lower bound (maps to 0)
        high: Upper bound (maps to 1)

    Returns:
        Series of scores in [0, 1]
    """
    scores = (ratio - low) / (high - low)
    scores = np.maximum(scores, 0.0)  # Floor at 0
    scores = np.minimum(scores, 1.0)  # Cap at 1
    return pd.Series(scores, index=ratio.index).fillna(0.0)


def liquidity_score(market_cap: Optional[float], float_shares: Optional[float],
                   avg_volume: Optional[float], params: Optional[Dict[str, Any]] = None) -> float:
    """
    Calculate liquidity score 0..1 for 'liquidity sweet spot' detection.

    Args:
        market_cap: Market capitalization
        float_shares: Float shares outstanding
        avg_volume: Average daily volume
        params: Optional dict with 'liquidity_thresholds' config

    Returns:
        Score in [0, 1]
    """
    if params is None:
        params = DEFAULTS

    liq_config = params.get('liquidity_thresholds', DEFAULTS['liquidity_thresholds'])

    try:
        if market_cap is None or float_shares is None or avg_volume is None:
            return 0.0

        s = 0.0
        weights = liq_config['weights']

        # Market cap factor
        if liq_config['min_market_cap'] <= market_cap <= liq_config['max_market_cap']:
            s += weights['market_cap']

        # Float factor
        if float_shares <= liq_config['max_float_shares']:
            s += weights['float']

        # Avg volume factor
        if liq_config['min_avg_volume'] <= avg_volume <= liq_config['max_avg_volume']:
            s += weights['volume']

        return min(1.0, s)

    except (TypeError, ValueError, KeyError) as e:
        logger.warning("Error calculating liquidity score: %s", e)
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

    # Validate input
    df = df_intraday.copy()
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Intraday DF must contain column: {col}")

    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # Ensure numeric columns
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Volume'] = df['Volume'].fillna(0.0)

    logger.info("Computing EMPS for %s with %d bars", ticker or 'unknown', len(df))

    # Compute components (vectorized)
    df['vol_zscore'] = compute_volume_zscore(df, lookback=p['vol_lookback'])
    df['vol_score'] = map_vol_zscore_to_score_vectorized(df['vol_zscore'], p)

    df['vwap_dev'] = compute_vwap_deviation(df, lookback_vwap=p['vwap_lookback'])
    df['vwap_score'] = map_vwap_dev_to_score_vectorized(df['vwap_dev'], sat=0.10)

    df['rv_short'] = realized_volatility(df['Close'], window=p['rv_short_window'])
    df['rv_long'] = realized_volatility(df['Close'], window=p['rv_long_window'])

    # Compute RV ratio with safe division
    df['rv_ratio'] = np.where(df['rv_long'] > 1e-8, df['rv_short'] / df['rv_long'], np.nan)
    df['rv_ratio'] = pd.Series(df['rv_ratio'], index=df.index).replace([np.inf, -np.inf], np.nan)
    df['rv_score'] = map_rv_ratio_to_score_vectorized(df['rv_ratio'], low=1.0, high=3.0)

    # Liquidity score is constant per-ticker
    liq = liquidity_score(market_cap, float_shares, avg_volume, params=p)
    df['liquidity_score'] = liq

    # Combine with weights (vectorized)
    df['emps_score'] = (
        weights['vol'] * df['vol_score']
        + weights['vwap'] * df['vwap_score']
        + weights['rv'] * df['rv_score']
        + weights.get('liquidity', 0.0) * df['liquidity_score']
    )

    # Ensure score is in [0, 1] range before social boost
    df['emps_score'] = df['emps_score'].clip(0.0, 1.0)

    # Flags (vectorized boolean operations)
    df['explosion_flag'] = df['emps_score'] >= p['combined_score_thresh']
    df['hard_flag'] = (
        (df['vol_zscore'] >= p['hard']['vol_zscore']) &
        (df['vwap_dev'].abs() >= p['hard']['vwap_dev']) &
        (df['rv_ratio'] >= p['hard']['rv_ratio'])
    )

    # Use actual bar timestamps (from index) instead of current time
    if isinstance(df.index, pd.DatetimeIndex):
        df['emps_timestamp'] = df.index
    else:
        df['emps_timestamp'] = pd.Timestamp.utcnow()

    df['ticker'] = ticker

    logger.info("EMPS computed: latest score=%.3f, explosion=%s, hard=%s",
                df['emps_score'].iloc[-1] if not df.empty else 0.0,
                df['explosion_flag'].iloc[-1] if not df.empty else False,
                df['hard_flag'].iloc[-1] if not df.empty else False)

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
    # Optionally add social_score boost (if provided)
    social = meta.get('social_count')
    if social is not None and social > 0:
        p = {**DEFAULTS, **(params or {})}
        social_config = p['social']

        social_norm = min(1.0, social / social_config['normalization_factor'])
        boost = social_config['weight_boost'] * social_norm

        df_scored['social_boost'] = boost
        df_scored['emps_score'] = (df_scored['emps_score'] + boost).clip(0.0, 1.0)

        # Re-evaluate explosion_flag
        combined_thresh = p['combined_score_thresh']
        df_scored['explosion_flag'] = df_scored['emps_score'] >= combined_thresh

        logger.info("Added social boost: count=%d, normalized=%.3f, boost=%.3f",
                    social, social_norm, boost)

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
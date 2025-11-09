"""
Data validation utilities.

This module provides validation functions for OHLCV data, timestamps,
and other data quality checks.
"""

import pandas as pd
from typing import Optional, List, Tuple, Dict, Any, Union
from datetime import timedelta
import logging

_logger = logging.getLogger(__name__)


def validate_ohlcv_data(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_data_points: int = 10,
    allow_duplicate_timestamps: bool = False,
    require_volume_data: bool = True,
    price_sanity_check: bool = True,
    max_price_change_pct: float = 50.0,
    symbol: Optional[str] = None,
    interval: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """
    Validate OHLCV data for quality and consistency.

    Args:
        df: DataFrame containing OHLCV data
        required_columns: List of required column names
        min_data_points: Minimum number of data points required
        allow_duplicate_timestamps: Whether to allow duplicate timestamps
        require_volume_data: Whether volume data is required
        price_sanity_check: Whether to perform price sanity checks
        max_price_change_pct: Maximum allowed price change percentage

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    if required_columns is None:
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    # Check if DataFrame is empty
    if df.empty:
        errors.append("DataFrame is empty")
        return False, errors

    # Check required columns (handle both column and index cases)
    missing_columns = []
    for col in required_columns:
        if col == 'timestamp':
            # Check if timestamp is either a column or the index
            if col not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                missing_columns.append(col)
        else:
            if col not in df.columns:
                missing_columns.append(col)

    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    # Check minimum data points
    if len(df) < min_data_points:
        errors.append(f"Insufficient data points: {len(df)} < {min_data_points}")

    # Check for duplicate timestamps (handle both column and index cases)
    if 'timestamp' in df.columns and not allow_duplicate_timestamps:
        duplicate_timestamps = df['timestamp'].duplicated().sum()
        if duplicate_timestamps > 0:
            errors.append(f"Found {duplicate_timestamps} duplicate timestamps")
    elif isinstance(df.index, pd.DatetimeIndex) and not allow_duplicate_timestamps:
        duplicate_timestamps = df.index.duplicated().sum()
        if duplicate_timestamps > 0:
            errors.append(f"Found {duplicate_timestamps} duplicate timestamps")

    # Check for missing values
    for col in required_columns:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                errors.append(f"Column '{col}' has {missing_count} missing values")

    # Check timestamp format and order (handle both column and index cases)
    if 'timestamp' in df.columns:
        # Determine appropriate gap tolerance based on symbol, interval, and data frequency
        gap_tolerance = _determine_smart_gap_tolerance(df['timestamp'], symbol, interval)
        timestamp_errors = validate_timestamps(df['timestamp'], max_gap_hours=gap_tolerance)
        if timestamp_errors:
            errors.extend(timestamp_errors)
    elif isinstance(df.index, pd.DatetimeIndex):
        # Determine appropriate gap tolerance based on symbol, interval, and data frequency
        gap_tolerance = _determine_smart_gap_tolerance(df.index, symbol, interval)
        timestamp_errors = validate_timestamps(df.index, max_gap_hours=gap_tolerance)
        if timestamp_errors:
            errors.extend(timestamp_errors)

    # Check price data consistency
    if price_sanity_check and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        price_errors = validate_price_consistency(df, max_price_change_pct)
        if price_errors:
            errors.extend(price_errors)

    # Check volume data
    if require_volume_data and 'volume' in df.columns:
        volume_errors = validate_volume_data(df['volume'])
        if volume_errors:
            errors.extend(volume_errors)

    return len(errors) == 0, errors


def validate_timestamps(
    timestamps: pd.Series,
    check_order: bool = True,
    check_gaps: bool = True,
    max_gap_hours: int = 24,
    timezone_aware: bool = False
) -> List[str]:
    """
    Validate timestamp data for consistency and quality.

    Args:
        timestamps: Series of timestamps
        check_order: Whether to check if timestamps are in ascending order
        check_gaps: Whether to check for large gaps in timestamps
        max_gap_hours: Maximum allowed gap in hours
        timezone_aware: Whether timestamps should be timezone-aware

    Returns:
        List of validation errors
    """
    errors = []

    # Check if timestamps are datetime objects
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        errors.append("Timestamps must be datetime objects")
        return errors

    # Check for None/NaT values
    if timestamps.isna().any():
        errors.append("Timestamps contain None/NaT values")

    # Check timezone awareness - handle both Series and DatetimeIndex
    try:
        if timezone_aware:
            if not timestamps.dt.tz:
                errors.append("Timestamps must be timezone-aware")
        else:
            if timestamps.dt.tz:
                errors.append("Timestamps must be timezone-naive")
    except AttributeError:
        # If .dt accessor is not available, assume timezone-naive
        if timezone_aware:
            errors.append("Timestamps must be timezone-aware")

    # Check order
    if check_order and len(timestamps) > 1:
        if not timestamps.is_monotonic_increasing:
            errors.append("Timestamps are not in ascending order")

    # Check for gaps
    if check_gaps and len(timestamps) > 1:
        gaps = timestamps.diff().dropna()
        large_gaps = gaps[gaps > timedelta(hours=max_gap_hours)]
        if len(large_gaps) > 0:
            # For daily data, allow weekend gaps (48+ hours)
            if max_gap_hours == 24:
                # Filter out weekend gaps (48-72 hours) for daily data
                weekend_gaps = large_gaps[(large_gaps >= timedelta(hours=48)) & (large_gaps <= timedelta(hours=72))]
                non_weekend_gaps = large_gaps[~((large_gaps >= timedelta(hours=48)) & (large_gaps <= timedelta(hours=72)))]

                if len(non_weekend_gaps) > 0:
                    errors.append(f"Found {len(non_weekend_gaps)} non-weekend gaps larger than {max_gap_hours} hours")
            else:
                errors.append(f"Found {len(large_gaps)} gaps larger than {max_gap_hours} hours")

    return errors


def validate_price_consistency(
    df: pd.DataFrame,
    max_price_change_pct: float = 50.0,
    min_price: float = 0.0001,
    max_price: float = 1000000.0
) -> List[str]:
    """
    Validate price data for consistency and sanity.

    Args:
        df: DataFrame with OHLC columns
        max_price_change_pct: Maximum allowed price change percentage
        min_price: Minimum allowed price
        max_price: Maximum allowed price

    Returns:
        List of validation errors
    """
    errors = []

    # Check price ranges
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            # Check for negative prices
            negative_prices = (df[col] < 0).sum()
            if negative_prices > 0:
                errors.append(f"Column '{col}' has {negative_prices} negative prices")

            # Check for extremely low/high prices
            too_low = (df[col] < min_price).sum()
            too_high = (df[col] > max_price).sum()
            if too_low > 0:
                errors.append(f"Column '{col}' has {too_low} prices below {min_price}")
            if too_high > 0:
                errors.append(f"Column '{col}' has {too_high} prices above {max_price}")

    # Check OHLC consistency
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        # High should be >= open, close, low
        invalid_high = ((df['high'] < df['open']) |
                       (df['high'] < df['close']) |
                       (df['high'] < df['low'])).sum()
        if invalid_high > 0:
            errors.append(f"Found {invalid_high} rows where high price is not the highest")

        # Low should be <= open, close, high
        invalid_low = ((df['low'] > df['open']) |
                      (df['low'] > df['close']) |
                      (df['low'] > df['high'])).sum()
        if invalid_low > 0:
            errors.append(f"Found {invalid_low} rows where low price is not the lowest")

    # Check for extreme price changes
    if 'close' in df.columns and len(df) > 1:
        price_changes = df['close'].pct_change().dropna() * 100
        extreme_changes = price_changes[abs(price_changes) > max_price_change_pct]
        if len(extreme_changes) > 0:
            errors.append(f"Found {len(extreme_changes)} price changes > {max_price_change_pct}%")

    return errors


def validate_volume_data(volumes: pd.Series) -> List[str]:
    """
    Validate volume data for consistency.

    Args:
        volumes: Series of volume data

    Returns:
        List of validation errors
    """
    errors = []

    # Check for negative volumes
    negative_volumes = (volumes < 0).sum()
    if negative_volumes > 0:
        errors.append(f"Volume data has {negative_volumes} negative values")

    # Check for extremely large volumes (potential data corruption)
    if volumes.max() > 1e15:  # 1 quadrillion
        errors.append("Volume data contains extremely large values (potential corruption)")

    # Check for all-zero volumes (might indicate missing data)
    zero_volumes = (volumes == 0).sum()
    if zero_volumes == len(volumes):
        errors.append("All volume values are zero (potential missing data)")

    return errors


def validate_data_gaps(
    timestamps: pd.Series,
    expected_interval: str,
    tolerance_minutes: int = 5
) -> List[str]:
    """
    Validate that data follows expected interval with minimal gaps.

    Args:
        timestamps: Series of timestamps
        expected_interval: Expected interval (e.g., '1m', '1h', '1d')
        tolerance_minutes: Tolerance for gaps in minutes

    Returns:
        List of validation errors
    """
    errors = []

    if len(timestamps) < 2:
        return errors

    # Parse expected interval
    interval_minutes = parse_interval_to_minutes(expected_interval)
    if interval_minutes is None:
        errors.append(f"Invalid interval format: {expected_interval}")
        return errors

    # Calculate expected gaps
    expected_gap = timedelta(minutes=interval_minutes)
    tolerance_gap = timedelta(minutes=tolerance_minutes)

    # Check actual gaps
    gaps = timestamps.diff().dropna()
    expected_gaps = gaps[(gaps >= expected_gap - tolerance_gap) &
                         (gaps <= expected_gap + tolerance_gap)]
    unexpected_gaps = gaps[~((gaps >= expected_gap - tolerance_gap) &
                             (gaps <= expected_gap + tolerance_gap))]

    if len(unexpected_gaps) > 0:
        errors.append(f"Found {len(unexpected_gaps)} gaps that don't match expected interval {expected_interval}")

    return errors


def parse_interval_to_minutes(interval: str) -> Optional[int]:
    """
    Parse interval string to minutes.

    Args:
        interval: Interval string (e.g., '1m', '1h', '1d')

    Returns:
        Number of minutes or None if invalid
    """
    interval = interval.lower()

    if interval.endswith('m'):
        try:
            return int(interval[:-1])
        except ValueError:
            return None
    elif interval.endswith('h'):
        try:
            return int(interval[:-1]) * 60
        except ValueError:
            return None
    elif interval.endswith('d'):
        try:
            return int(interval[:-1]) * 24 * 60
        except ValueError:
            return None
    elif interval.endswith('w'):
        try:
            return int(interval[:-1]) * 7 * 24 * 60
        except ValueError:
            return None
    elif interval.endswith('mo') or interval.endswith('m'):
        try:
            return int(interval[:-2]) * 30 * 24 * 60
        except ValueError:
            return None
    else:
        return None


def get_data_quality_score(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate a comprehensive data quality score.

    Args:
        df: DataFrame to evaluate
        required_columns: List of required columns

    Returns:
        Dictionary with quality metrics and score
    """
    if required_columns is None:
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    # Run validation
    is_valid, errors = validate_ohlcv_data(df, required_columns)

    # Calculate completeness
    total_cells = len(df) * len(required_columns)
    missing_cells = sum(df[col].isna().sum() for col in required_columns if col in df.columns)
    completeness = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0

    # Calculate consistency score
    consistency_score = 1.0 - (len(errors) / 10)  # Normalize to 0-1
    consistency_score = max(0.0, min(1.0, consistency_score))

    # Overall quality score
    quality_score = (completeness + consistency_score) / 2

    return {
        'is_valid': is_valid,
        'quality_score': quality_score,
        'completeness': completeness,
        'consistency_score': consistency_score,
        'total_rows': len(df),
        'missing_cells': missing_cells,
        'errors': errors,
        'error_count': len(errors)
    }


def _determine_smart_gap_tolerance(
    timestamps: Union[pd.Series, pd.DatetimeIndex],
    symbol: Optional[str] = None,
    interval: Optional[str] = None
) -> int:
    """
    Determine appropriate gap tolerance based on symbol type, interval, and timestamp frequency.

    Args:
        timestamps: Series or DatetimeIndex of timestamps
        symbol: Symbol name (e.g., 'BTCUSDT', 'AAPL')
        interval: Time interval (e.g., '5m', '1h', '1d')

    Returns:
        Gap tolerance in hours
    """
    if len(timestamps) < 2:
        return 24  # Default tolerance

    # Determine asset type
    is_crypto = _is_crypto_symbol(symbol)

    # Calculate typical gaps
    gaps = timestamps.diff().dropna()
    if len(gaps) == 0:
        return 24

    # Convert to hours
    gap_hours = gaps / pd.Timedelta(hours=1)

    # Find the most common gap (mode)
    try:
        # Convert to Series to use mode()
        gap_series = pd.Series(gap_hours)
        mode_gap = gap_series.mode().iloc[0] if not gap_series.mode().empty else 24
    except:
        # Fallback to median if mode fails
        mode_gap = gap_hours.median() if not gap_hours.empty else 24

    # Set tolerance based on asset type, interval, and frequency
    if is_crypto:
        # Crypto markets are 24/7, but allow 24h gaps for any timeframe < 1d
        if interval in ['5m', '15m', '1h', '4h']:
            return 24  # Allow 24 hour gaps for intraday crypto
        elif interval == '1d':
            return 24  # Allow 1 day gaps for daily crypto
        else:
            # Fallback based on frequency
            if mode_gap <= 1:
                return 24
            elif mode_gap <= 24:
                return 24
            else:
                return 48
    else:
        # Stock markets have weekends and holidays
        if interval in ['5m', '15m']:
            return 72  # Allow 3 days (weekend + holiday) for intraday stocks
        elif interval in ['1h', '4h']:
            return 168  # Allow 7 days for hourly stocks
        elif interval == '1d':
            return 168  # Allow 7 days for daily stocks (weekends + holidays)
        else:
            # Fallback based on frequency
            if mode_gap <= 1:
                return 72
            elif mode_gap <= 24:
                return 168
            else:
                return 336


def _is_crypto_symbol(symbol: Optional[str]) -> bool:
    """
    Determine if a symbol is a cryptocurrency.

    Args:
        symbol: Symbol name

    Returns:
        True if crypto, False if stock/other
    """
    if not symbol:
        return False

    # Common crypto patterns
    crypto_patterns = [
        'BTC', 'ETH', 'LTC', 'XRP', 'ADA', 'DOT', 'LINK', 'UNI', 'AAVE', 'COMP',
        'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'GUSD', 'FRAX', 'LUSD'
    ]

    symbol_upper = symbol.upper()

    # Check if symbol contains crypto patterns
    for pattern in crypto_patterns:
        if pattern in symbol_upper:
            return True

    # Check if symbol ends with common crypto pairs
    crypto_suffixes = ['USDT', 'USDC', 'BUSD', 'BTC', 'ETH', 'BNB']
    for suffix in crypto_suffixes:
        if symbol_upper.endswith(suffix):
            return True

    return False


def _determine_gap_tolerance(timestamps: Union[pd.Series, pd.DatetimeIndex]) -> int:
    """
    Legacy function - use _determine_smart_gap_tolerance instead.

    Determine appropriate gap tolerance based on timestamp frequency.

    Args:
        timestamps: Series or DatetimeIndex of timestamps

    Returns:
        Gap tolerance in hours
    """
    return _determine_smart_gap_tolerance(timestamps, None, None)

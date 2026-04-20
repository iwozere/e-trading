"""
Price fetcher.

Thin wrapper around `DataManager.get_ohlcv` that returns the most recent daily
close for a set of symbols and is resilient to per-symbol failures.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def _build_data_manager() -> Any:
    """
    Construct a default `DataManager`. Imported lazily to avoid pulling the
    whole data stack into unrelated tests.
    """
    from src.data.data_manager import DataManager
    return DataManager()


def fetch_latest_closes(
    symbols: Iterable[str],
    data_manager: Optional[Any] = None,
    lookback_days: int = 7,
) -> Dict[str, float]:
    """
    Fetch the most recent daily close for each symbol.

    Per-symbol fetch errors are logged and the symbol is simply omitted from
    the returned dict; the caller should treat absent symbols as "no price".

    Args:
        symbols: Iterable of ticker symbols.
        data_manager: Optional `DataManager` instance. A default one is built
            if not provided. Accepting it keeps the function testable.
        lookback_days: How many calendar days back to fetch to ensure at least
            one valid daily bar is returned (covers weekends / holidays).

    Returns:
        Mapping `{symbol: latest_close_price}` for every symbol that produced
        a usable bar.
    """
    dm = data_manager if data_manager is not None else _build_data_manager()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(2, lookback_days))

    symbol_list = list(symbols)
    prices: Dict[str, float] = {}
    for symbol in symbol_list:
        try:
            df = dm.get_ohlcv(symbol, "1d", start, end)
        except Exception:
            _logger.exception("Failed to fetch OHLCV for %s", symbol)
            continue

        if df is None or df.empty or "close" not in df.columns:
            _logger.warning("No usable OHLCV returned for %s", symbol)
            continue

        try:
            last_close = float(df["close"].iloc[-1])
        except (IndexError, ValueError, TypeError):
            _logger.exception("Could not extract last close for %s", symbol)
            continue

        if last_close <= 0:
            _logger.warning("Ignoring non-positive last close for %s: %s", symbol, last_close)
            continue

        prices[symbol] = last_close

    _logger.info("Fetched latest closes for %d/%d symbols", len(prices), len(symbol_list))
    return prices

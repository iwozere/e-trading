"""
Data downloader implementation for Yahoo Finance, fetching historical market data for analysis and backtesting.

This module provides the YahooDataDownloader class for downloading historical OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance. It supports fetching, saving, loading, and updating data for single or multiple symbols, and is suitable for both research and production trading workflows.

Main Features:
- Download historical data for any stock or ticker from Yahoo Finance
- Save and load data as CSV files
- Update existing data files with new data
- Download data for multiple symbols in batch
- Inherits common logic from BaseDataDownloader for file management

Valid values:
- interval: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
- period: Any string like '1d', '7d', '1mo', '3mo', '6mo', '1y', '2y', etc. (used to calculate start_date/end_date)

Classes:
- YahooDataDownloader: Main class for interacting with Yahoo Finance and managing data downloads
"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

import pandas as pd
import yfinance as yf

from src.data.downloader.base_data_downloader import BaseDataDownloader
from src.model.schemas import OptionalFundamentals, Fundamentals
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class YahooDataDownloader(BaseDataDownloader):
    """
    Yahoo Finance data downloader with support for batch operations.

    Features:
    - Single ticker downloads with rate limiting
    - Batch downloads for multiple tickers (OHLCV and fundamentals)
    - Automatic rate limiting and error handling
    - Support for all Yahoo Finance intervals and periods
    """

    def __init__(self):
        """Initialize Yahoo Finance data downloader."""
        super().__init__()

    def _convert_debt_to_equity_ratio(self, value) -> Optional[float]:
        """Convert Yahoo Finance debt/equity percentage to ratio format.

        Yahoo Finance returns debt/equity as percentage (e.g., 47.997)
        but we want it as ratio (e.g., 0.47997) to match FMP format.
        """
        if value is None:
            return None
        try:
            float_value = float(value)
            # Convert percentage to ratio (divide by 100)
            return float_value / 100.0
        except (ValueError, TypeError):
            return None

    def get_supported_intervals(self) -> List[str]:
        """Return list of supported intervals for Yahoo Finance."""
        return ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

    def get_ohlcv(self, symbol: str, interval: str, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """
        Download OHLCV data for a single symbol with automatic batching for large date ranges.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Data interval (e.g., '1d', '1h')
            start_date: Start date as datetime.datetime
            end_date: End date as datetime.datetime

        Returns:
            pd.DataFrame: OHLCV data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        try:
            # Determine if we need to use batching based on interval
            max_period = self._get_max_period_for_interval(interval)
            date_range = end_date - start_date

            # Convert max_period to timedelta for comparison
            period_to_days = {
                '1d': 1, '7d': 7, '60d': 60, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730
            }
            max_days = period_to_days.get(max_period, 730)
            max_timedelta = timedelta(days=max_days)

            # If date range is within limits, download normally
            if date_range <= max_timedelta:
                return self._download_ohlcv_single(symbol, interval, start_date, end_date)
            else:
                # Use batching for large date ranges
                _logger.info("Date range %s exceeds yfinance limit for %s interval. Using batching.", date_range, interval)
                return self._download_ohlcv_batched(symbol, interval, start_date, end_date, max_period)

        except Exception:
            _logger.exception("Error downloading OHLCV data for %s:", symbol)
            raise

    def _get_max_period_for_interval(self, interval: str) -> str:
        """
        Get the maximum supported period for a given interval.

        yfinance has different limits for different intervals:
        - 1m: 7 days
        - 2m, 5m, 15m, 30m, 60m, 90m: 60 days
        - 1h: 730 days (2 years)
        - 1d, 5d, 1wk, 1mo, 3mo: unlimited (use 2y as reasonable limit)
        """
        interval_limits = {
            '1m': '7d',
            '2m': '60d', '5m': '60d', '15m': '60d', '30m': '60d', '60m': '60d', '90m': '60d',
            '1h': '2y',
            '1d': '2y', '5d': '2y', '1wk': '2y', '1mo': '2y', '3mo': '2y'
        }
        return interval_limits.get(interval, '2y')

    def _download_ohlcv_single(self, symbol: str, interval: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Download OHLCV data for a single request (no batching)."""
        _logger.debug("Downloading OHLCV data for %s (%s to %s)", symbol, start_date, end_date)

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)

        if df.empty:
            _logger.warning("No data returned for %s", symbol)
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Convert to standard format (lowercase columns, timestamp index to column)
        df = df.reset_index()
        # Lowercase
        df.columns = [str(c).lower() for c in df.columns]
        # Map yfinance names to standard
        rename_map = {
            'date': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'adj close': 'close',
            'volume': 'volume'
        }
        df = df.rename(columns=rename_map)
        # Keep only required columns if present
        cols = [c for c in ['timestamp','open','high','low','close','volume'] if c in df.columns]
        df = df[cols]
        # Ensure timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        _logger.debug("Downloaded %d rows for %s", len(df), symbol)
        return df

    def _download_ohlcv_batched(self, symbol: str, interval: str, start_date: datetime, end_date: datetime, max_period: str) -> pd.DataFrame:
        """Download OHLCV data using batching for large date ranges."""
        batches = self._calculate_batch_dates(start_date, end_date, max_period)

        all_data = []

        for i, (batch_start, batch_end) in enumerate(batches):
            _logger.debug("Downloading batch %d/%d for %s: %s to %s",
                         i + 1, len(batches), symbol, batch_start, batch_end)

            try:
                batch_df = self._download_ohlcv_single(symbol, interval, batch_start, batch_end)
                if not batch_df.empty:
                    all_data.append(batch_df)
            except Exception:
                _logger.exception("Error downloading batch %d for %s:", i + 1, symbol)
                continue

        if not all_data:
            _logger.warning("No data downloaded for %s in any batch", symbol)
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Combine all batches
        combined_df = pd.concat(all_data, ignore_index=True)

        # Remove duplicates and sort by timestamp
        combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

        _logger.info("Downloaded %d total rows for %s across %d batches", len(combined_df), symbol, len(batches))
        return combined_df

    def get_ohlcv_batch(self, symbols: List[str], interval: str, start_date: datetime, end_date: datetime, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Download OHLCV data for multiple symbols in a single batch request.

        Args:
            symbols: List of stock symbols
            interval: Data interval (e.g., '1d', '1h')
            start_date: Start date as datetime.datetime
            end_date: End date as datetime.datetime

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to OHLCV DataFrames
        """
        try:
            _logger.info("Downloading batch OHLCV data for %d symbols (%s to %s)", len(symbols), start_date, end_date)

            # Use yf.download for batch operation
            df_batch = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                interval=interval,
                group_by='ticker'
            )

            if df_batch.empty:
                _logger.warning("No data returned for batch download")
                return {symbol: pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']) for symbol in symbols}

            results = {}

            # Process multi-level DataFrame
            if len(symbols) == 1:
                # Single ticker case
                df = df_batch.reset_index()

                # Handle different column structures from YFinance
                expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                actual_columns = list(df.columns)

                _logger.debug("YFinance batch returned columns for %s: %s", symbols[0], actual_columns)

                # If we have more columns than expected, handle it gracefully
                if len(actual_columns) > len(expected_columns):
                    # YFinance sometimes returns additional columns like 'Adj Close', 'Dividends'
                    # We'll take the first 6 columns and rename them
                    df = df.iloc[:, :6]  # Take first 6 columns
                    df.columns = expected_columns
                elif len(actual_columns) == len(expected_columns):
                    # Exact match, just rename
                    df.columns = expected_columns
                else:
                    # Fewer columns than expected, this is an error
                    _logger.error("Unexpected column count for %s: expected %d, got %d", symbols[0], len(expected_columns), len(actual_columns))
                    raise ValueError(f"Unexpected column structure for {symbols[0]}: {actual_columns}")

                df['timestamp'] = pd.to_datetime(df['timestamp'])
                results[symbols[0]] = df
            else:
                # Multiple tickers case
                for symbol in symbols:
                    try:
                        # Extract data for this symbol
                        if symbol in df_batch.columns.get_level_values(0):
                            df_symbol = df_batch[symbol].reset_index()

                            # Handle different column structures from YFinance
                            expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                            actual_columns = list(df_symbol.columns)

                            _logger.debug("YFinance batch returned columns for %s: %s", symbol, actual_columns)

                            # If we have more columns than expected, handle it gracefully
                            if len(actual_columns) > len(expected_columns):
                                # YFinance sometimes returns additional columns like 'Adj Close', 'Dividends'
                                # We'll take the first 6 columns and rename them
                                df_symbol = df_symbol.iloc[:, :6]  # Take first 6 columns
                                df_symbol.columns = expected_columns
                            elif len(actual_columns) == len(expected_columns):
                                # Exact match, just rename
                                df_symbol.columns = expected_columns
                            else:
                                # Fewer columns than expected, this is an error
                                _logger.error("Unexpected column count for %s: expected %d, got %d", symbol, len(expected_columns), len(actual_columns))
                                raise ValueError(f"Unexpected column structure for {symbol}: {actual_columns}")

                            df_symbol['timestamp'] = pd.to_datetime(df_symbol['timestamp'])
                            results[symbol] = df_symbol
                        else:
                            _logger.warning("No data found for %s in batch download", symbol)
                            results[symbol] = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    except Exception:
                        _logger.exception("Error processing %s from batch download:", symbol)
                        results[symbol] = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            _logger.info("Batch download completed. %d/%d symbols processed successfully",
                        len([df for df in results.values() if not df.empty]), len(symbols))
            return results

        except Exception:
            _logger.exception("Error in batch OHLCV download:")
            # Fallback to individual downloads
            _logger.info("Falling back to individual downloads")
            return self._fallback_individual_downloads(symbols, interval, start_date, end_date)

    def _fallback_individual_downloads(self, symbols: List[str], interval: str, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fallback method for individual downloads when batch fails."""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_ohlcv(symbol, interval, start_date, end_date)
            except Exception:
                _logger.exception("Fallback download failed for %s:", symbol)
                results[symbol] = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return results

    def get_fundamentals(self, symbol: str) -> OptionalFundamentals:
        """
        Get comprehensive fundamental data for a given stock using Yahoo Finance.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Fundamentals: Comprehensive fundamental data for the stock
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                _logger.error("No data returned from yfinance for ticker %s", symbol)
                return Fundamentals(
                    ticker=symbol.upper(),
                    company_name="Unknown",
                    current_price=0.0,
                    market_cap=0.0,
                    pe_ratio=0.0,
                    forward_pe=0.0,
                    dividend_yield=0.0,
                    earnings_per_share=0.0,
                    data_source="Yahoo Finance",
                    last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )

            _logger.debug("Retrieved fundamentals for %s: %s", symbol, info.get('shortName', 'Unknown'))

            fundamentals = Fundamentals(
                ticker=symbol.upper(),
                company_name=info.get("longName", "Unknown"),
                current_price=info.get("regularMarketPrice", 0.0),
                market_cap=info.get("marketCap", 0.0),
                pe_ratio=info.get("trailingPE", 0.0),
                forward_pe=info.get("forwardPE", 0.0),
                dividend_yield=info.get("dividendYield", 0.0),
                earnings_per_share=info.get("trailingEps", 0.0),
                # Additional fields
                price_to_book=info.get("priceToBook", None),
                return_on_equity=info.get("returnOnEquity", None),
                return_on_assets=info.get("returnOnAssets", None),
                debt_to_equity=self._convert_debt_to_equity_ratio(info.get("debtToEquity", None)),
                current_ratio=info.get("currentRatio", None),
                quick_ratio=info.get("quickRatio", None),
                revenue=info.get("totalRevenue", None),
                revenue_growth=info.get("revenueGrowth", None),
                net_income=info.get("netIncomeToCommon", None),
                net_income_growth=info.get("netIncomeGrowth", None),
                free_cash_flow=info.get("freeCashflow", None),
                operating_margin=info.get("operatingMargins", None),
                profit_margin=info.get("profitMargins", None),
                beta=info.get("beta", None),
                sector=info.get("sector", None),
                industry=info.get("industry", None),
                country=info.get("country", None),
                exchange=info.get("exchange", None),
                currency=info.get("currency", None),
                shares_outstanding=info.get("sharesOutstanding", None),
                float_shares=info.get("floatShares", None),
                short_ratio=info.get("shortRatio", None),
                payout_ratio=info.get("payoutRatio", None),
                peg_ratio=info.get("pegRatio", None),
                price_to_sales=info.get("priceToSalesTrailing12Months", None),
                enterprise_value=info.get("enterpriseValue", None),
                enterprise_value_to_ebitda=info.get("enterpriseToEbitda", None),
                data_source="Yahoo Finance",
                last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

            return fundamentals

        except Exception:
            _logger.exception("Error getting fundamentals for %s:", symbol)
            raise

    def get_fundamentals_batch(self, symbols: List[str]) -> Dict[str, Fundamentals]:
        """
        Get fundamental data for multiple symbols using optimized batch request.
        This method minimizes individual API calls by using batch operations where possible.

        Args:
            symbols: List of stock symbols

        Returns:
            Dict[str, Fundamentals]: Dictionary mapping symbols to Fundamentals objects
        """
        try:
            _logger.info("Downloading batch fundamentals for %d symbols", len(symbols))

            # Use yf.Tickers for batch operation - this is the most efficient method
            tickers_str = " ".join(symbols)
            tickers_obj = yf.Tickers(tickers_str)

            # Get info for all tickers - access individual ticker info
            info_batch = {}
            for symbol in symbols:
                try:
                    ticker_obj = tickers_obj.tickers[symbol]
                    info_batch[symbol] = ticker_obj.info
                except Exception as e:
                    _logger.warning("Failed to get info for %s: %s", symbol, e)
                    info_batch[symbol] = None

            results = {}

            for symbol in symbols:
                try:
                    if symbol in info_batch:
                        info = info_batch[symbol]

                        if not info:
                            _logger.warning("No fundamental data for %s", symbol)
                            results[symbol] = self._create_default_fundamentals(symbol)
                            continue

                        # Create Fundamentals object with only the data available from batch call
                        # Avoid individual API calls for financial statements unless absolutely necessary
                        results[symbol] = Fundamentals(
                            ticker=symbol.upper(),
                            company_name=info.get("longName", "Unknown"),
                            current_price=info.get("regularMarketPrice", 0.0),
                            market_cap=info.get("marketCap", 0.0),
                            pe_ratio=info.get("trailingPE", 0.0),
                            forward_pe=info.get("forwardPE", 0.0),
                            dividend_yield=info.get("dividendYield", 0.0),
                            earnings_per_share=info.get("trailingEps", 0.0),
                            # Additional fields from batch call
                            price_to_book=info.get("priceToBook", None),
                            return_on_equity=info.get("returnOnEquity", None),
                            return_on_assets=info.get("returnOnAssets", None),
                            debt_to_equity=self._convert_debt_to_equity_ratio(info.get("debtToEquity", None)),
                            current_ratio=info.get("currentRatio", None),
                            quick_ratio=info.get("quickRatio", None),
                            revenue=info.get("totalRevenue", None),
                            revenue_growth=info.get("revenueGrowth", None),
                            net_income=info.get("netIncomeToCommon", None),
                            net_income_growth=info.get("netIncomeGrowth", None),
                            free_cash_flow=info.get("freeCashflow", None),
                            operating_margin=info.get("operatingMargins", None),
                            profit_margin=info.get("profitMargins", None),
                            beta=info.get("beta", None),
                            sector=info.get("sector", None),
                            industry=info.get("industry", None),
                            country=info.get("country", None),
                            exchange=info.get("exchange", None),
                            currency=info.get("currency", None),
                            shares_outstanding=info.get("sharesOutstanding", None),
                            float_shares=info.get("floatShares", None),
                            short_ratio=info.get("shortRatio", None),
                            payout_ratio=info.get("payoutRatio", None),
                            peg_ratio=info.get("pegRatio", None),
                            price_to_sales=info.get("priceToSalesTrailing12Months", None),
                            enterprise_value=info.get("enterpriseValue", None),
                            enterprise_value_to_ebitda=info.get("enterpriseToEbitda", None),
                            data_source="Yahoo Finance",
                            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                    else:
                        _logger.warning("Symbol %s not found in batch response", symbol)
                        results[symbol] = self._create_default_fundamentals(symbol)

                except Exception:
                    _logger.exception("Error processing fundamentals for %s:", symbol)
                    results[symbol] = self._create_default_fundamentals(symbol)

            _logger.info("Batch fundamentals download completed. %d/%d symbols processed successfully",
                        len([f for f in results.values() if f.company_name != "Unknown"]), len(symbols))
            return results

        except Exception:
            _logger.exception("Error in batch fundamentals download:")
            # Fallback to individual downloads
            _logger.info("Falling back to individual fundamental downloads")
            return self._fallback_individual_fundamentals(symbols)

    def get_fundamentals_batch_optimized(self, symbols: List[str], include_financials: bool = False) -> Dict[str, Fundamentals]:
        """
        Get fundamental data for multiple symbols using the most optimized batch approach.
        This method uses only batch operations and avoids individual API calls entirely.

        Args:
            symbols: List of stock symbols
            include_financials: Whether to include detailed financial statements (may require individual calls)

        Returns:
            Dict[str, Fundamentals]: Dictionary mapping symbols to Fundamentals objects
        """
        try:
            _logger.info("Downloading optimized batch fundamentals for %d symbols", len(symbols))

            # Method 1: Use yf.download for basic info (most efficient)
            # This gets basic price and volume data in a single call
            try:
                # Get basic market data for all symbols
                basic_data = yf.download(symbols, period="1d", progress=False)
                _logger.debug("Retrieved basic market data for %d symbols", len(symbols))
            except Exception as e:
                _logger.warning("Failed to get basic market data: %s", e)
                basic_data = None

            # Method 2: Use yf.Tickers for comprehensive info
            tickers_str = " ".join(symbols)
            tickers_obj = yf.Tickers(tickers_str)

            # Get comprehensive info for all tickers - access individual ticker info
            info_batch = {}
            for symbol in symbols:
                try:
                    ticker_obj = tickers_obj.tickers[symbol]
                    info_batch[symbol] = ticker_obj.info
                except Exception as e:
                    _logger.warning("Failed to get info for %s: %s", symbol, e)
                    info_batch[symbol] = None

            results = {}

            for symbol in symbols:
                try:
                    if symbol in info_batch:
                        info = info_batch[symbol]

                        if not info:
                            _logger.warning("No fundamental data for %s", symbol)
                            results[symbol] = self._create_default_fundamentals(symbol)
                            continue

                        # Get current price from basic data if available
                        current_price = info.get("regularMarketPrice", 0.0)
                        if basic_data is not None and symbol in basic_data.columns.get_level_values(0):
                            try:
                                # Extract current price from basic data
                                symbol_data = basic_data[symbol]
                                if not symbol_data.empty:
                                    current_price = symbol_data['Close'].iloc[-1]
                            except Exception:
                                pass  # Use price from info if basic data extraction fails

                        # Create Fundamentals object with all available data
                        results[symbol] = Fundamentals(
                            ticker=symbol.upper(),
                            company_name=info.get("longName", "Unknown"),
                            current_price=current_price,
                            market_cap=info.get("marketCap", 0.0),
                            pe_ratio=info.get("trailingPE", 0.0),
                            forward_pe=info.get("forwardPE", 0.0),
                            dividend_yield=info.get("dividendYield", 0.0),
                            earnings_per_share=info.get("trailingEps", 0.0),
                            # Additional fields from batch call
                            price_to_book=info.get("priceToBook", None),
                            return_on_equity=info.get("returnOnEquity", None),
                            return_on_assets=info.get("returnOnAssets", None),
                            debt_to_equity=self._convert_debt_to_equity_ratio(info.get("debtToEquity", None)),
                            current_ratio=info.get("currentRatio", None),
                            quick_ratio=info.get("quickRatio", None),
                            revenue=info.get("totalRevenue", None),
                            revenue_growth=info.get("revenueGrowth", None),
                            net_income=info.get("netIncomeToCommon", None),
                            net_income_growth=info.get("netIncomeGrowth", None),
                            free_cash_flow=info.get("freeCashflow", None),
                            operating_margin=info.get("operatingMargins", None),
                            profit_margin=info.get("profitMargins", None),
                            beta=info.get("beta", None),
                            sector=info.get("sector", None),
                            industry=info.get("industry", None),
                            country=info.get("country", None),
                            exchange=info.get("exchange", None),
                            currency=info.get("currency", None),
                            shares_outstanding=info.get("sharesOutstanding", None),
                            float_shares=info.get("floatShares", None),
                            short_ratio=info.get("shortRatio", None),
                            payout_ratio=info.get("payoutRatio", None),
                            peg_ratio=info.get("pegRatio", None),
                            price_to_sales=info.get("priceToSalesTrailing12Months", None),
                            enterprise_value=info.get("enterpriseValue", None),
                            enterprise_value_to_ebitda=info.get("enterpriseToEbitda", None),
                            data_source="Yahoo Finance",
                            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                    else:
                        _logger.warning("Symbol %s not found in batch response", symbol)
                        results[symbol] = self._create_default_fundamentals(symbol)

                except Exception:
                    _logger.exception("Error processing fundamentals for %s:", symbol)
                    results[symbol] = self._create_default_fundamentals(symbol)

            _logger.info("Optimized batch fundamentals download completed. %d/%d symbols processed successfully",
                        len([f for f in results.values() if f.company_name != "Unknown"]), len(symbols))
            return results

        except Exception:
            _logger.exception("Error in optimized batch fundamentals download:")
            # Fallback to regular batch method
            _logger.info("Falling back to regular batch method")
            return self.get_fundamentals_batch(symbols)

    def _create_default_fundamentals(self, symbol: str) -> Fundamentals:
        """Create a default Fundamentals object for failed downloads."""
        return Fundamentals(
            ticker=symbol.upper(),
            company_name="Unknown",
            current_price=0.0,
            market_cap=0.0,
            pe_ratio=0.0,
            forward_pe=0.0,
            dividend_yield=0.0,
            earnings_per_share=0.0,
            data_source="Yahoo Finance",
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    def _fallback_individual_fundamentals(self, symbols: List[str]) -> Dict[str, Fundamentals]:
        """Fallback method for individual fundamental downloads when batch fails."""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_fundamentals(symbol)
            except Exception:
                _logger.exception("Fallback fundamental download failed for %s:", symbol)
                results[symbol] = self._create_default_fundamentals(symbol)
        return results

    def download_multiple_symbols(
        self, symbols: List[str], interval: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, str]:
        """
        Download data for multiple symbols with rate limiting.

        Args:
            symbols: List of stock symbols
            interval: Data interval
            start_date: Start date as datetime.datetime
            end_date: End date as datetime.datetime

        Returns:
            Dict[str, str]: Dictionary mapping symbols to file paths
        """
        def download_func(symbol, interval, start_date, end_date):
            return self.get_ohlcv(symbol, interval, start_date, end_date)

        # Override the base method to add rate limiting between symbols
        results = {}
        for symbol in symbols:
            try:
                _logger.info("Processing symbol %s (%d/%d)", symbol, len(results) + 1, len(symbols))

                df = download_func(symbol, interval, start_date, end_date)
                filepath = self.save_data(df, symbol, interval, start_date, end_date)
                results[symbol] = filepath

                # Process next symbol

            except Exception:
                _logger.exception("Error processing %s:", symbol)
                continue
        return results

    def download_multiple_symbols_batch(
        self, symbols: List[str], interval: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, str]:
        """
        Download data for multiple symbols using batch operations.

        Args:
            symbols: List of stock symbols
            interval: Data interval
            start_date: Start date as datetime.datetime
            end_date: End date as datetime.datetime

        Returns:
            Dict[str, str]: Dictionary mapping symbols to file paths
        """
        try:
            _logger.info("Starting batch download for %d symbols", len(symbols))

            # Use batch OHLCV download
            ohlcv_data = self.get_ohlcv_batch(symbols, interval, start_date, end_date)

            results = {}
            for symbol, df in ohlcv_data.items():
                try:
                    filepath = self.save_data(df, symbol, interval, start_date, end_date)
                    results[symbol] = filepath
                except Exception:
                    _logger.exception("Error saving data for %s:", symbol)
                    continue

            _logger.info("Batch download completed. %d/%d symbols saved successfully", len(results), len(symbols))
            return results

        except Exception:
            _logger.exception("Error in batch download:")
            # Fallback to individual downloads
            return self.download_multiple_symbols(symbols, interval, start_date, end_date)

    def get_periods(self) -> list:
        return ['1d', '7d', '1mo', '3mo', '6mo', '1y', '2y']

    def get_intervals(self) -> list:
        return ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

    def _calculate_batch_dates(self, start_date: datetime, end_date: datetime, max_period: str = "2y") -> List[tuple]:
        """
        Calculate batch dates to respect yfinance period limits.

        yfinance has different limits for different intervals:
        - 1m: 7 days
        - 2m, 5m, 15m, 30m, 60m, 90m: 60 days
        - 1h: 730 days (2 years)
        - 1d, 5d, 1wk, 1mo, 3mo: unlimited

        Args:
            start_date: Start date
            end_date: End date
            max_period: Maximum period to use for batching (default: "2y")

        Returns:
            List of (batch_start, batch_end) tuples
        """
        # Convert max_period to timedelta
        period_to_days = {
            '1d': 1, '7d': 7, '60d': 60, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730
        }

        max_days = period_to_days.get(max_period, 730)  # default to 2y
        max_timedelta = timedelta(days=max_days)

        batches = []
        current_start = start_date

        while current_start < end_date:
            current_end = min(current_start + max_timedelta, end_date)
            batches.append((current_start, current_end))
            current_start = current_end

        return batches

    def is_valid_period_interval(self, period, interval) -> bool:
        """
        Validate period/interval combination for yfinance.

        Since we now support batching for large periods, we accept any period
        that can be converted to a date range, and let the batching logic handle
        the actual limits during download.
        """
        # First check if interval is supported
        if interval not in self.get_intervals():
            return False

        # For periods, we now accept any period that can be parsed
        # The batching logic will handle the actual limits during download
        try:
            # Try to parse the period to see if it's valid

            if period.endswith("y"):
                years = int(period[:-1])
                if years > 0:
                    return True
            elif period.endswith("mo"):
                months = int(period[:-2])
                if months > 0:
                    return True
            elif period.endswith("w"):
                weeks = int(period[:-1])
                if weeks > 0:
                    return True
            elif period.endswith("d"):
                days = int(period[:-1])
                if days > 0:
                    return True
            else:
                # Check if it's in our supported periods list
                return period in self.get_periods()

            return True
        except (ValueError, TypeError):
            return False

    def test_connection(self) -> bool:
        """
        Test connection to Yahoo Finance by fetching data for a well-known ticker.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            _logger.info("Testing Yahoo Finance connection...")

            # Try to fetch recent data for a well-known ticker
            test_ticker = yf.Ticker("AAPL")
            df = test_ticker.history(period="1d", interval="1d")

            if df is not None and not df.empty:
                _logger.info("Yahoo Finance connection test successful")
                return True
            else:
                _logger.error("Yahoo Finance connection test failed: No data returned")
                return False

        except Exception as e:
            _logger.error("Yahoo Finance connection test failed: %s", e)
            return False

    def load_universe_from_screener(self, criteria: Optional[Dict[str, any]] = None) -> List[str]:
        """
        Load universe from screener criteria.

        Note: Yahoo Finance doesn't have a built-in screener API.
        This method returns a curated fallback list of liquid tickers suitable for trading.

        Args:
            criteria: Screening criteria (not used, provided for interface compatibility)

        Returns:
            List of ticker symbols
        """
        _logger.warning(
            "Yahoo Finance does not support screener functionality. "
            "Returning curated list of liquid tickers suitable for EMPS analysis."
        )

        # Return a curated list of liquid, volatile tickers suitable for EMPS
        fallback_tickers = [
            # Large cap tech (high liquidity)
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD',

            # Volatile mid-caps
            'PLTR', 'SOFI', 'HOOD', 'COIN', 'RIVN', 'LCID', 'PLUG', 'ROKU', 'UBER', 'LYFT',

            # Meme stocks (high volatility)
            'GME', 'AMC', 'BBBY',

            # Biotech (volatile sector)
            'MRNA', 'BNTX', 'NVAX', 'PFE', 'JNJ',

            # Energy/EV (explosive moves common)
            'FCEL', 'BLNK', 'CHPT', 'QS', 'HYLN',

            # Growth/Fintech
            'DKNG', 'UPST', 'AFRM', 'SQ', 'SHOP', 'PYPL',

            # Additional high-volume stocks
            'SPY', 'QQQ', 'IWM',  # ETFs for market exposure
            'BA', 'DIS', 'BABA', 'NIO', 'INTC', 'SNAP', 'TWTR', 'PINS',
        ]

        _logger.info("Returning %d curated tickers for Yahoo Finance", len(fallback_tickers))
        return fallback_tickers

    def get_company_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get company profile information from Yahoo Finance.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with company profile data or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                _logger.warning("No profile data for %s", symbol)
                return None

            # Map Yahoo Finance fields to common profile format
            profile = {
                'symbol': symbol.upper(),
                'companyName': info.get('longName') or info.get('shortName'),
                'exchange': info.get('exchange'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'mktCap': info.get('marketCap'),
                'marketCap': info.get('marketCap'),
                'floatShares': info.get('floatShares'),
                'volAvg': info.get('averageVolume'),
                'avgVolume': info.get('averageVolume'),
                'website': info.get('website'),
                'description': info.get('longBusinessSummary'),
                'country': info.get('country'),
                'currency': info.get('currency'),
            }

            _logger.info("Retrieved profile for %s", symbol)
            return profile

        except Exception as e:
            _logger.error("Error fetching profile for %s: %s", symbol, e)
            return None

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
import os
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
import yfinance as yf

from src.notification.logger import setup_logger
from src.model.telegram_bot import Fundamentals
from src.data.base_data_downloader import BaseDataDownloader

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

    def __init__(self, data_dir: str = "data"):
        super().__init__(data_dir)
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

    def _rate_limit(self):
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def get_ohlcv(self, symbol: str, interval: str, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """
        Download OHLCV data for a single symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Data interval (e.g., '1d', '1h')
            start_date: Start date as datetime.datetime
            end_date: End date as datetime.datetime

        Returns:
            pd.DataFrame: OHLCV data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        try:
            # Apply rate limiting
            self._rate_limit()

            _logger.debug("Downloading OHLCV data for %s (%s to %s)", symbol, start_date, end_date)

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                _logger.warning("No data returned for %s", symbol)
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Convert to standard format
            df = df.reset_index()
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            _logger.debug("Downloaded %d rows for %s", len(df), symbol)
            return df

        except Exception as e:
            _logger.exception("Error downloading OHLCV data for %s: %s", symbol, str(e))
            raise

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
            # Apply rate limiting
            self._rate_limit()

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
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                results[symbols[0]] = df
            else:
                # Multiple tickers case
                for symbol in symbols:
                    try:
                        # Extract data for this symbol
                        if symbol in df_batch.columns.get_level_values(0):
                            df_symbol = df_batch[symbol].reset_index()
                            df_symbol.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                            df_symbol['timestamp'] = pd.to_datetime(df_symbol['timestamp'])
                            results[symbol] = df_symbol
                        else:
                            _logger.warning("No data found for %s in batch download", symbol)
                            results[symbol] = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    except Exception as e:
                        _logger.error("Error processing %s from batch download: %s", symbol, str(e))
                        results[symbol] = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            _logger.info("Batch download completed. %d/%d symbols processed successfully",
                        len([df for df in results.values() if not df.empty]), len(symbols))
            return results

        except Exception as e:
            _logger.exception("Error in batch OHLCV download: %s", str(e))
            # Fallback to individual downloads
            _logger.info("Falling back to individual downloads")
            return self._fallback_individual_downloads(symbols, interval, start_date, end_date)

    def _fallback_individual_downloads(self, symbols: List[str], interval: str, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fallback method for individual downloads when batch fails."""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_ohlcv(symbol, interval, start_date, end_date)
            except Exception as e:
                _logger.error("Fallback download failed for %s: %s", symbol, str(e))
                results[symbol] = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return results

    def get_fundamentals(self, symbol: str) -> Fundamentals:
        """
        Get comprehensive fundamental data for a given stock using Yahoo Finance.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Fundamentals: Comprehensive fundamental data for the stock
        """
        try:
            # Apply rate limiting
            self._rate_limit()

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

            return Fundamentals(
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
                debt_to_equity=info.get("debtToEquity", None),
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

        except Exception as e:
            _logger.exception("Error getting fundamentals for %s: %s", symbol, str(e))
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
            # Apply rate limiting
            self._rate_limit()

            _logger.info("Downloading batch fundamentals for %d symbols", len(symbols))

            # Use yf.Tickers for batch operation - this is the most efficient method
            tickers_str = " ".join(symbols)
            tickers_obj = yf.Tickers(tickers_str)

            # Get info for all tickers in a single batch call
            info_batch = tickers_obj.info

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
                            debt_to_equity=info.get("debtToEquity", None),
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

                except Exception as e:
                    _logger.error("Error processing fundamentals for %s: %s", symbol, str(e))
                    results[symbol] = self._create_default_fundamentals(symbol)

            _logger.info("Batch fundamentals download completed. %d/%d symbols processed successfully",
                        len([f for f in results.values() if f.company_name != "Unknown"]), len(symbols))
            return results

        except Exception as e:
            _logger.exception("Error in batch fundamentals download: %s", str(e))
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
            # Apply rate limiting
            self._rate_limit()

            _logger.info("Downloading optimized batch fundamentals for %d symbols", len(symbols))

            # Method 1: Use yf.download for basic info (most efficient)
            # This gets basic price and volume data in a single call
            try:
                # Get basic market data for all symbols
                basic_data = yf.download(symbols, period="1d", progress=False)
                _logger.debug("Retrieved basic market data for %d symbols", len(symbols))
            except Exception as e:
                _logger.warning("Failed to get basic market data: %s", str(e))
                basic_data = None

            # Method 2: Use yf.Tickers for comprehensive info
            tickers_str = " ".join(symbols)
            tickers_obj = yf.Tickers(tickers_str)

            # Get comprehensive info for all tickers in a single batch call
            info_batch = tickers_obj.info

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
                            debt_to_equity=info.get("debtToEquity", None),
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

                except Exception as e:
                    _logger.error("Error processing fundamentals for %s: %s", symbol, str(e))
                    results[symbol] = self._create_default_fundamentals(symbol)

            _logger.info("Optimized batch fundamentals download completed. %d/%d symbols processed successfully",
                        len([f for f in results.values() if f.company_name != "Unknown"]), len(symbols))
            return results

        except Exception as e:
            _logger.exception("Error in optimized batch fundamentals download: %s", str(e))
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
            except Exception as e:
                _logger.error("Fallback fundamental download failed for %s: %s", symbol, str(e))
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

                # Rate limiting between symbols (already handled in get_ohlcv, but extra safety)
                if len(results) < len(symbols):  # Don't sleep after the last symbol
                    self._rate_limit()

            except Exception as e:
                _logger.exception("Error processing %s: %s", symbol, str(e))
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
                except Exception as e:
                    _logger.error("Error saving data for %s: %s", symbol, str(e))
                    continue

            _logger.info("Batch download completed. %d/%d symbols saved successfully", len(results), len(symbols))
            return results

        except Exception as e:
            _logger.exception("Error in batch download: %s", str(e))
            # Fallback to individual downloads
            return self.download_multiple_symbols(symbols, interval, start_date, end_date)

    def get_periods(self) -> list:
        return ['1d', '7d', '1mo', '3mo', '6mo', '1y', '2y']

    def get_intervals(self) -> list:
        return ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

    def is_valid_period_interval(self, period, interval) -> bool:
        return interval in self.get_intervals() and period in self.get_periods()

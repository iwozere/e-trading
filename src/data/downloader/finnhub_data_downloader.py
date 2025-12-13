from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import requests
import aiohttp
import asyncio
import time
from src.notification.logger import setup_logger
from src.model.schemas import OptionalFundamentals, Fundamentals, SentimentData
from src.data.downloader.base_data_downloader import BaseDataDownloader

_logger = setup_logger(__name__)

"""
Data downloader implementation for Finnhub, fetching historical market data for analysis and backtesting.

This module provides the FinnhubDataDownloader class for downloading historical OHLCV (Open, High, Low, Close, Volume) data from Finnhub. It supports fetching, saving, loading, and updating data for single or multiple symbols, and is suitable for both research and production trading workflows.

Main Features:
- Download historical data for any stock or ticker from Finnhub (free tier)
- Save and load data as CSV files
- Update existing data files with new data
- Download data for multiple symbols in batch
- Inherits common logic from BaseDataDownloader for file management

Valid values:
- interval: '1m', '5m', '15m', '1h', '1d' (Finnhub free tier: 1m for 30 days, 1d for 5 years; other intervals are resampled)
- period: '1d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y'

API limits (free tier):
- 60 requests per minute
- 30 days 1m data, 5 years 1d data
- If you exceed the free API limit or request unsupported data, a clear error will be raised.

Classes:
- FinnhubDataDownloader: Main class for interacting with Finnhub and managing data downloads
"""

class FinnhubDataDownloader(BaseDataDownloader):
    """
    A class to download historical data from Finnhub.

    This class provides methods to:
    1. Download historical OHLCV data for a given symbol
    2. Save data to CSV files
    3. Load data from CSV files
    4. Update existing data files with new data
    5. Get comprehensive fundamental data for stocks

    **Fundamental Data Capabilities:**
    - ✅ PE Ratio (trailing and forward)
    - ✅ Financial Ratios (P/B, ROE, ROA, debt/equity, current ratio, quick ratio)
    - ✅ Growth Metrics (revenue growth, net income growth)
    - ✅ Company Information (name, sector, industry, country, exchange)
    - ✅ Market Data (market cap, current price, shares outstanding)
    - ✅ Profitability Metrics (operating margin, profit margin, free cash flow)
    - ✅ Valuation Metrics (beta, PEG ratio, price-to-sales, enterprise value)

    **Data Quality:** High - Finnhub provides comprehensive fundamental data
    **Rate Limits:** 60 API calls per minute (free tier)
    **Coverage:** Global stocks and ETFs

    Parameters:
    -----------
    api_key : str
        Finnhub API key
    data_dir : str
        Directory to store downloaded data files

    Example:
    --------
    >>> from datetime import datetime
    >>> downloader = FinnhubDataDownloader("YOUR_API_KEY")
    >>> df = downloader.get_ohlcv("AAPL", "1d", datetime(2023, 1, 1), datetime(2023, 12, 31))
    >>> # Get fundamental data
    >>> fundamentals = downloader.get_fundamentals("AAPL")
    >>> print(f"PE Ratio: {fundamentals.pe_ratio}")
    >>> print(f"Market Cap: ${fundamentals.market_cap:,.0f}")
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        # Get API key from parameter or config
        self.api_key = api_key or self._get_config_value('FINNHUB_KEY', 'FINNHUB_KEY')
        self.base_url = "https://finnhub.io/api/v1/stock/candle"

        if not self.api_key:
            raise ValueError("Finnhub API key is required. Get one at: https://finnhub.io/")

    def get_ohlcv(self, symbol: str, interval: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Download historical data for a given symbol from Finnhub.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Data interval ('1m', '5m', '15m', '1h', '1d')
            start_date: Start date as datetime.datetime
            end_date: End date as datetime.datetime

        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        try:
            # Validate interval and period
            if not self.is_valid_period_interval('1d', interval):
                raise ValueError(f"Unsupported interval: {interval}")
            # Finnhub supports: 1, 5, 15, 30, 60 minute, daily, weekly, monthly
            interval_map = {
                '1m': '1', '5m': '5', '15m': '15', '30m': '30', '1h': '60', '1d': 'D'
            }
            finnhub_interval = interval_map.get(interval, 'D')
            # Convert dates to UNIX timestamps (seconds)
            start_unix = int(start_date.timestamp())
            end_unix = int(end_date.timestamp())
            params = {
                'symbol': symbol,
                'resolution': finnhub_interval,
                'from': start_unix,
                'to': end_unix,
                'token': self.api_key
            }
            response = requests.get(self.base_url, params=params)
            if response.status_code == 429:
                raise RuntimeError("Finnhub API rate limit exceeded (free tier: 60 requests/minute)")
            if response.status_code != 200:
                raise RuntimeError(f"Finnhub API error: {response.status_code} {response.text}")
            data = response.json()
            if data.get('s') != 'ok':
                raise ValueError(f"No results in Finnhub response: {data}")
            # Finnhub returns: t (timestamp), o (open), h (high), l (low), c (close), v (volume)
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(data['t'], unit='s'),
                'open': data['o'],
                'high': data['h'],
                'low': data['l'],
                'close': data['c'],
                'volume': data['v']
            })
            # Resample if needed
            if interval == '5m':
                df = df.set_index('timestamp').resample('5T').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
            elif interval == '15m':
                df = df.set_index('timestamp').resample('15T').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
            elif interval == '1h':
                df = df.set_index('timestamp').resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
            # For '1m' and '1d', no resampling needed
            return df
        except Exception as e:
            _logger.exception("Error downloading data for %s: %s", symbol, str(e))
            raise

    def get_periods(self) -> list:
        return ['1d', '7d', '1mo', '3mo', '6mo', '1y', '2y', '5y']

    def get_intervals(self) -> list:
        return ['1m', '5m', '15m', '1h', '1d']

    def is_valid_period_interval(self, period, interval) -> bool:
        return interval in self.get_intervals() and period in self.get_periods()

    def get_fundamentals(self, symbol: str) -> OptionalFundamentals:
        """
        Get comprehensive fundamental data for a given stock using Finnhub.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Fundamentals: Comprehensive fundamental data for the stock
        """
        try:
            # Get company profile
            profile_url = "https://finnhub.io/api/v1/stock/profile2"
            profile_params = {
                'symbol': symbol,
                'token': self.api_key
            }

            profile_response = requests.get(profile_url, params=profile_params)
            if profile_response.status_code != 200:
                raise RuntimeError(f"Finnhub API error: {profile_response.status_code}")

            profile_data = profile_response.json()

            if not profile_data:
                _logger.error("No data returned from Finnhub for ticker %s", symbol)
                return Fundamentals(
                    ticker=symbol.upper(),
                    company_name="Unknown",
                    current_price=0.0,
                    market_cap=0.0,
                    pe_ratio=0.0,
                    forward_pe=0.0,
                    dividend_yield=0.0,
                    earnings_per_share=0.0,
                    data_source="Finnhub",
                    last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )

            # Get current price
            quote_url = "https://finnhub.io/api/v1/quote"
            quote_params = {
                'symbol': symbol,
                'token': self.api_key
            }

            quote_response = requests.get(quote_url, params=quote_params)
            quote_data = quote_response.json() if quote_response.status_code == 200 else {}
            current_price = quote_data.get('c', 0.0) if quote_data else 0.0

            # Get financial metrics
            metrics_url = "https://finnhub.io/api/v1/stock/metric"
            metrics_params = {
                'symbol': symbol,
                'metric': 'all',
                'token': self.api_key
            }

            metrics_response = requests.get(metrics_url, params=metrics_params)
            metrics_data = metrics_response.json() if metrics_response.status_code == 200 else {}
            metrics = metrics_data.get('metric', {}) if metrics_data else {}

            _logger.debug("Retrieved fundamentals for %s: %s", symbol, profile_data.get('name', 'Unknown'))

            # Extract volume data (in millions, convert to shares)
            avg_vol_millions = metrics.get("10DayAverageTradingVolume", 0)
            avg_volume = avg_vol_millions * 1_000_000 if avg_vol_millions else 0.0

            return Fundamentals(
                ticker=symbol.upper(),
                company_name=profile_data.get("name", "Unknown"),
                current_price=current_price,
                market_cap=float(profile_data.get("marketCapitalization", 0)) if profile_data.get("marketCapitalization") else 0.0,
                pe_ratio=float(metrics.get("peNormalizedAnnual", 0)) if metrics.get("peNormalizedAnnual") else 0.0,
                forward_pe=float(metrics.get("peForwardAnnual", 0)) if metrics.get("peForwardAnnual") else 0.0,
                dividend_yield=float(metrics.get("dividendYieldIndicatedAnnual", 0)) if metrics.get("dividendYieldIndicatedAnnual") else 0.0,
                earnings_per_share=float(metrics.get("epsTTM", 0)) if metrics.get("epsTTM") else 0.0,
                # Additional fields
                price_to_book=float(metrics.get("pbAnnual", 0)) if metrics.get("pbAnnual") else None,
                return_on_equity=float(metrics.get("roeRfy", 0)) if metrics.get("roeRfy") else None,
                return_on_assets=float(metrics.get("roaRfy", 0)) if metrics.get("roaRfy") else None,
                debt_to_equity=float(metrics.get("debtToEquityAnnual", 0)) if metrics.get("debtToEquityAnnual") else None,
                current_ratio=float(metrics.get("currentRatioAnnual", 0)) if metrics.get("currentRatioAnnual") else None,
                quick_ratio=float(metrics.get("quickRatioAnnual", 0)) if metrics.get("quickRatioAnnual") else None,
                revenue=float(metrics.get("revenuePerShareAnnual", 0)) if metrics.get("revenuePerShareAnnual") else None,
                revenue_growth=float(metrics.get("revenueGrowthAnnual", 0)) if metrics.get("revenueGrowthAnnual") else None,
                net_income=float(metrics.get("netIncomeGrowthAnnual", 0)) if metrics.get("netIncomeGrowthAnnual") else None,
                net_income_growth=float(metrics.get("netIncomeGrowthAnnual", 0)) if metrics.get("netIncomeGrowthAnnual") else None,
                free_cash_flow=float(metrics.get("freeCashFlowPerShareTTM", 0)) if metrics.get("freeCashFlowPerShareTTM") else None,
                operating_margin=float(metrics.get("operatingMarginTTM", 0)) if metrics.get("operatingMarginTTM") else None,
                profit_margin=float(metrics.get("netProfitMarginTTM", 0)) if metrics.get("netProfitMarginTTM") else None,
                beta=float(metrics.get("beta", 0)) if metrics.get("beta") else None,
                sector=profile_data.get("finnhubIndustry", None),
                industry=profile_data.get("finnhubIndustry", None),
                country=profile_data.get("country", None),
                exchange=profile_data.get("exchange", None),
                currency=profile_data.get("currency", None),
                shares_outstanding=float(profile_data.get("shareOutstanding", 0)) if profile_data.get("shareOutstanding") else None,
                float_shares=None,  # Finnhub doesn't provide float shares
                avg_volume=avg_volume,  # 10-day average trading volume
                short_ratio=float(metrics.get("shortInterestRatioAnnual", 0)) if metrics.get("shortInterestRatioAnnual") else None,
                payout_ratio=float(metrics.get("payoutRatioAnnual", 0)) if metrics.get("payoutRatioAnnual") else None,
                peg_ratio=float(metrics.get("pegAnnual", 0)) if metrics.get("pegAnnual") else None,
                price_to_sales=float(metrics.get("psTTM", 0)) if metrics.get("psTTM") else None,
                enterprise_value=float(metrics.get("enterpriseValueAnnual", 0)) if metrics.get("enterpriseValueAnnual") else None,
                enterprise_value_to_ebitda=float(metrics.get("evToEbitdaAnnual", 0)) if metrics.get("evToEbitdaAnnual") else None,
                data_source="Finnhub",
                last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

        except Exception as e:
            _logger.exception("Failed to get fundamentals for %s: %s", symbol, str(e))
            return Fundamentals(
                ticker=symbol.upper(),
                company_name="Unknown",
                current_price=0.0,
                market_cap=0.0,
                pe_ratio=0.0,
                forward_pe=0.0,
                dividend_yield=0.0,
                earnings_per_share=0.0,
                data_source="Finnhub",
                last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

    def download_multiple_symbols(
        self, symbols: List[str], interval: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, str]:
        def download_func(symbol, interval, start_date, end_date):
            return self.get_ohlcv(symbol, interval, start_date, end_date)
        return super().download_multiple_symbols(
            symbols, download_func, interval, start_date, end_date
        )

    def get_supported_intervals(self) -> List[str]:
        """Return list of supported intervals for Finnhub."""
        return ['1m', '5m', '15m', '30m', '1h', '1d']

    # Short Squeeze Detection Pipeline Extensions

    def get_sentiment_data(self, symbol: str, hours_back: int = 24) -> Optional[Dict[str, Any]]:
        """
        Get sentiment data for a symbol from news and social media.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            hours_back: Number of hours to look back for sentiment data (default: 24)

        Returns:
            Dictionary with sentiment data or None if failed
        """
        try:
            from datetime import datetime, timedelta

            # Calculate date range for sentiment data
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=hours_back)

            # Convert to required format (YYYY-MM-DD)
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            # Get news sentiment
            news_url = "https://finnhub.io/api/v1/news-sentiment"
            news_params = {
                'symbol': symbol,
                'token': self.api_key
            }

            _logger.debug("Fetching sentiment data for %s", symbol)

            response = requests.get(news_url, params=news_params, timeout=30)

            if response.status_code == 429:
                _logger.warning("Finnhub API rate limit exceeded for sentiment data")
                return None

            if response.status_code != 200:
                _logger.warning("Finnhub sentiment API error: %s", response.status_code)
                return None

            data = response.json()

            if not data:
                _logger.warning("No sentiment data found for %s", symbol)
                return None

            # Extract sentiment metrics
            sentiment_data = {
                'symbol': symbol,
                'sentiment_score': data.get('sentiment', {}).get('bearishPercent', 0),
                'bullish_percent': data.get('sentiment', {}).get('bullishPercent', 0),
                'bearish_percent': data.get('sentiment', {}).get('bearishPercent', 0),
                'buzz_articles_in_last_week': data.get('buzz', {}).get('articlesInLastWeek', 0),
                'buzz_weekly_average': data.get('buzz', {}).get('weeklyAverage', 0),
                'company_news_score': data.get('companyNewsScore', 0),
                'sector_average_bullish': data.get('sectorAverageBullishPercent', 0),
                'sector_average_news_score': data.get('sectorAverageNewsScore', 0),
                'timestamp': datetime.now().isoformat(),
                'hours_back': hours_back
            }

            _logger.debug("Retrieved sentiment data for %s: bullish=%s%%, bearish=%s%%",
                        symbol, sentiment_data.get('bullish_percent', 'N/A'),
                        sentiment_data.get('bearish_percent', 'N/A'))

            return sentiment_data

        except Exception as e:
            _logger.error("Error getting sentiment data for %s: %s", symbol, e)
            return None

    def get_options_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get options data for a symbol to calculate call/put ratios.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Dictionary with options data or None if failed
        """
        try:
            # Get options chain data
            options_url = "https://finnhub.io/api/v1/stock/option-chain"
            options_params = {
                'symbol': symbol,
                'token': self.api_key
            }

            _logger.debug("Fetching options data for %s", symbol)

            response = requests.get(options_url, params=options_params, timeout=30)

            if response.status_code == 429:
                _logger.warning("Finnhub API rate limit exceeded for options data")
                return None

            if response.status_code != 200:
                _logger.warning("Finnhub options API error: %s", response.status_code)
                return None

            data = response.json()

            if not data or 'data' not in data:
                _logger.warning("No options data found for %s", symbol)
                return None

            options_chain = data.get('data', [])

            # Calculate call/put metrics
            total_call_volume = 0
            total_put_volume = 0
            total_call_oi = 0
            total_put_oi = 0

            for option in options_chain:
                option_type = option.get('type', '').lower()
                volume = option.get('volume', 0) or 0
                open_interest = option.get('openInterest', 0) or 0

                if option_type == 'call':
                    total_call_volume += volume
                    total_call_oi += open_interest
                elif option_type == 'put':
                    total_put_volume += volume
                    total_put_oi += open_interest

            # Calculate ratios
            call_put_volume_ratio = (total_call_volume / total_put_volume) if total_put_volume > 0 else None
            call_put_oi_ratio = (total_call_oi / total_put_oi) if total_put_oi > 0 else None

            options_data = {
                'symbol': symbol,
                'total_call_volume': total_call_volume,
                'total_put_volume': total_put_volume,
                'total_call_open_interest': total_call_oi,
                'total_put_open_interest': total_put_oi,
                'call_put_volume_ratio': call_put_volume_ratio,
                'call_put_oi_ratio': call_put_oi_ratio,
                'total_options_count': len(options_chain),
                'timestamp': datetime.now().isoformat()
            }

            _logger.debug("Retrieved options data for %s: C/P volume ratio=%s, C/P OI ratio=%s",
                        symbol, call_put_volume_ratio, call_put_oi_ratio)

            return options_data

        except Exception as e:
            _logger.error("Error getting options data for %s: %s", symbol, e)
            return None

    def get_borrow_rates_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get stock lending/borrow rates data for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Dictionary with borrow rates data or None if failed
        """
        try:
            # Note: Finnhub's stock lending data might be limited or require premium subscription
            # This is a placeholder implementation that attempts to get the data

            lending_url = "https://finnhub.io/api/v1/stock/lending-rate"
            lending_params = {
                'symbol': symbol,
                'token': self.api_key
            }

            _logger.debug("Fetching borrow rates data for %s", symbol)

            response = requests.get(lending_url, params=lending_params, timeout=30)

            if response.status_code == 429:
                _logger.warning("Finnhub API rate limit exceeded for borrow rates data")
                return None

            if response.status_code != 200:
                _logger.warning("Finnhub borrow rates API error: %s (may require premium subscription)", response.status_code)
                return None

            data = response.json()

            if not data:
                _logger.warning("No borrow rates data found for %s", symbol)
                return None

            borrow_data = {
                'symbol': symbol,
                'borrow_fee_rate': data.get('fee', None),
                'available_shares': data.get('available', None),
                'fee_rate_percentage': data.get('feeRate', None),
                'timestamp': datetime.now().isoformat()
            }

            _logger.debug("Retrieved borrow rates data for %s: fee rate=%s%%",
                        symbol, borrow_data.get('fee_rate_percentage', 'N/A'))

            return borrow_data

        except Exception as e:
            _logger.error("Error getting borrow rates data for %s: %s", symbol, e)
            return None

    def calculate_call_put_ratio(self, options_data: Dict[str, Any]) -> Optional[float]:
        """
        Calculate call-to-put ratio from options data.

        Args:
            options_data: Options data dictionary from get_options_data()

        Returns:
            Call-to-put ratio or None if calculation not possible
        """
        try:
            if not options_data:
                return None

            call_volume = options_data.get('total_call_volume', 0)
            put_volume = options_data.get('total_put_volume', 0)

            if put_volume == 0:
                return None

            ratio = call_volume / put_volume
            _logger.debug("Calculated call/put ratio for %s: %s",
                        options_data.get('symbol', 'Unknown'), ratio)

            return ratio

        except Exception:
            _logger.exception("Error calculating call/put ratio:")
            return None

    def aggregate_24h_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Aggregate sentiment data over the last 24 hours.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Dictionary with aggregated 24-hour sentiment data or None if failed
        """
        try:
            # Get 24-hour sentiment data
            sentiment_data = self.get_sentiment_data(symbol, hours_back=24)

            if not sentiment_data:
                return None

            # Calculate aggregated sentiment score
            bullish_pct = sentiment_data.get('bullish_percent', 0)
            bearish_pct = sentiment_data.get('bearish_percent', 0)

            # Normalize sentiment score to -1 to 1 scale
            # Positive values indicate bullish sentiment, negative values indicate bearish
            if bullish_pct + bearish_pct > 0:
                sentiment_score = (bullish_pct - bearish_pct) / (bullish_pct + bearish_pct)
            else:
                sentiment_score = 0.0

            # Calculate buzz intensity
            articles_this_week = sentiment_data.get('buzz_articles_in_last_week', 0)
            weekly_average = sentiment_data.get('buzz_weekly_average', 1)
            buzz_intensity = articles_this_week / weekly_average if weekly_average > 0 else 0

            aggregated_data = {
                'symbol': symbol,
                'sentiment_score_24h': sentiment_score,
                'bullish_percent_24h': bullish_pct,
                'bearish_percent_24h': bearish_pct,
                'buzz_intensity': buzz_intensity,
                'articles_count_24h': articles_this_week,
                'company_news_score': sentiment_data.get('company_news_score', 0),
                'vs_sector_bullish': bullish_pct - sentiment_data.get('sector_average_bullish', 0),
                'vs_sector_news_score': sentiment_data.get('company_news_score', 0) - sentiment_data.get('sector_average_news_score', 0),
                'timestamp': datetime.now().isoformat()
            }

            _logger.debug("Aggregated 24h sentiment for %s: score=%s, buzz intensity=%s",
                        symbol, sentiment_score, buzz_intensity)

            return aggregated_data

        except Exception as e:
            _logger.error("Error aggregating 24h sentiment for %s: %s", symbol, e)
            return None

    def get_short_squeeze_batch_data(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get short squeeze relevant data for multiple tickers in batch.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker symbols to their short squeeze data
        """
        try:
            batch_data = {}
            total_tickers = len(tickers)

            _logger.info("Fetching Finnhub short squeeze data for %d tickers", total_tickers)

            for i, ticker in enumerate(tickers, 1):
                try:
                    # Get sentiment data
                    sentiment = self.aggregate_24h_sentiment(ticker)

                    # Get options data
                    options = self.get_options_data(ticker)

                    # Get borrow rates data
                    borrow_rates = self.get_borrow_rates_data(ticker)

                    # Calculate call/put ratio
                    call_put_ratio = self.calculate_call_put_ratio(options) if options else None

                    # Combine all data
                    ticker_data = {
                        'symbol': ticker,
                        'sentiment_24h': sentiment,
                        'options_data': options,
                        'borrow_rates': borrow_rates,
                        'call_put_ratio': call_put_ratio,
                        'timestamp': datetime.now().isoformat()
                    }

                    batch_data[ticker] = ticker_data

                    if i % 25 == 0:  # Log progress every 25 tickers (lower due to rate limits)
                        _logger.info("Processed %d/%d tickers", i, total_tickers)

                    # Add small delay to respect rate limits (60 calls/minute = 1 call/second)
                    time.sleep(1.1)

                except Exception as e:
                    _logger.warning("Failed to get Finnhub data for ticker %s: %s", ticker, e)
                    continue

            _logger.info("Successfully retrieved Finnhub data for %d/%d tickers", len(batch_data), total_tickers)
            return batch_data

        except Exception:
            _logger.exception("Error in Finnhub batch data retrieval:")
            return {}

    # ========================================================================
    # ASYNC SENTIMENT DATA METHODS (New Implementation)
    # ========================================================================

    async def get_news_sentiment(self, symbol: str) -> Optional[SentimentData]:
        """
        Get news sentiment data for a symbol (async).

        This method fetches sentiment data from Finnhub's news-sentiment API,
        which provides bullish/bearish percentages, buzz metrics, and sector comparisons.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            SentimentData object with news sentiment metrics, or None if failed
        """
        try:
            url = "https://finnhub.io/api/v1/news-sentiment"
            params = {
                'symbol': symbol.upper(),
                'token': self.api_key
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 429:
                        _logger.warning("Finnhub API rate limit exceeded for news sentiment")
                        return None

                    if response.status != 200:
                        _logger.warning("Finnhub news sentiment API error: %s", response.status)
                        return None

                    data = await response.json()

                    if not data:
                        _logger.warning("No news sentiment data found for %s", symbol)
                        return None

                    # Extract sentiment metrics
                    sentiment = data.get('sentiment', {})
                    buzz = data.get('buzz', {})

                    bullish_pct = sentiment.get('bullishPercent', 0)
                    bearish_pct = sentiment.get('bearishPercent', 0)

                    # Calculate normalized sentiment score (-1 to 1)
                    if bullish_pct + bearish_pct > 0:
                        sentiment_score = (bullish_pct - bearish_pct) / 100.0
                    else:
                        sentiment_score = 0.0

                    # Create SentimentData object
                    sentiment_data = SentimentData(
                        symbol=symbol.upper(),
                        provider='finnhub',
                        timestamp=datetime.now().isoformat(),
                        sentiment_score=sentiment_score,
                        bullish_score=bullish_pct / 100.0 if bullish_pct else None,
                        bearish_score=bearish_pct / 100.0 if bearish_pct else None,
                        article_count=buzz.get('articlesInLastWeek', 0),
                        buzz_ratio=buzz.get('buzz', None),
                        sector_comparison={
                            'sector_average_bullish': sentiment.get('sectorAverageBullishPercent', 0) / 100.0,
                            'vs_sector_bullish': (bullish_pct - sentiment.get('sectorAverageBullishPercent', 0)) / 100.0,
                            'company_news_score': data.get('companyNewsScore', 0),
                            'sector_average_news_score': data.get('sectorAverageNewsScore', 0)
                        },
                        raw_data=data,
                        data_source='finnhub_news_sentiment_api'
                    )

                    _logger.debug("Retrieved news sentiment for %s: score=%.3f, articles=%d",
                                symbol, sentiment_score, buzz.get('articlesInLastWeek', 0))

                    return sentiment_data

        except asyncio.TimeoutError:
            _logger.error("Timeout getting news sentiment for %s", symbol)
            return None
        except Exception as e:
            _logger.error("Error getting news sentiment for %s: %s", symbol, e)
            return None

    async def get_social_sentiment(self, symbol: str, days_back: int = 7) -> Optional[SentimentData]:
        """
        Get social media sentiment data for a symbol (async).

        This method fetches sentiment from Reddit, Twitter, and other social platforms
        via Finnhub's social-sentiment API, providing historical daily data.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            days_back: Number of days to look back (default: 7, max: 30)

        Returns:
            SentimentData object with social sentiment metrics, or None if failed
        """
        try:
            url = "https://finnhub.io/api/v1/stock/social-sentiment"

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            params = {
                'symbol': symbol.upper(),
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': self.api_key
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 429:
                        _logger.warning("Finnhub API rate limit exceeded for social sentiment")
                        return None

                    if response.status != 200:
                        _logger.warning("Finnhub social sentiment API error: %s", response.status)
                        return None

                    data = await response.json()

                    if not data:
                        _logger.warning("No social sentiment data found for %s", symbol)
                        return None

                    # Process Reddit data
                    reddit_data = None
                    reddit_mentions = 0
                    reddit_score = 0.0

                    if 'reddit' in data and data['reddit']:
                        reddit_entries = data['reddit']
                        # Get most recent entry or aggregate
                        if reddit_entries:
                            latest_reddit = reddit_entries[-1]  # Most recent
                            reddit_mentions = latest_reddit.get('mention', 0)
                            reddit_score = latest_reddit.get('score', 0.0)
                            reddit_data = {
                                'mentions': reddit_mentions,
                                'positive_mentions': latest_reddit.get('positiveMention', 0),
                                'negative_mentions': latest_reddit.get('negativeMention', 0),
                                'score': reddit_score,
                                'positive_score': latest_reddit.get('positiveScore', 0),
                                'negative_score': latest_reddit.get('negativeScore', 0),
                                'time': latest_reddit.get('atTime', '')
                            }

                    # Process Twitter data
                    twitter_data = None
                    twitter_mentions = 0
                    twitter_score = 0.0

                    if 'twitter' in data and data['twitter']:
                        twitter_entries = data['twitter']
                        if twitter_entries:
                            latest_twitter = twitter_entries[-1]  # Most recent
                            twitter_mentions = latest_twitter.get('mention', 0)
                            twitter_score = latest_twitter.get('score', 0.0)
                            twitter_data = {
                                'mentions': twitter_mentions,
                                'positive_mentions': latest_twitter.get('positiveMention', 0),
                                'negative_mentions': latest_twitter.get('negativeMention', 0),
                                'score': twitter_score,
                                'positive_score': latest_twitter.get('positiveScore', 0),
                                'negative_score': latest_twitter.get('negativeScore', 0),
                                'time': latest_twitter.get('atTime', '')
                            }

                    # Calculate overall sentiment score (weighted average)
                    total_mentions = reddit_mentions + twitter_mentions
                    if total_mentions > 0:
                        overall_score = (reddit_score * reddit_mentions + twitter_score * twitter_mentions) / total_mentions
                    else:
                        overall_score = 0.0

                    # Create SentimentData object
                    sentiment_data = SentimentData(
                        symbol=symbol.upper(),
                        provider='finnhub',
                        timestamp=datetime.now().isoformat(),
                        sentiment_score=overall_score,
                        mention_count=total_mentions,
                        reddit_data=reddit_data,
                        twitter_data=twitter_data,
                        sources={
                            'reddit': reddit_data is not None,
                            'twitter': twitter_data is not None
                        },
                        raw_data=data,
                        data_source='finnhub_social_sentiment_api'
                    )

                    _logger.debug("Retrieved social sentiment for %s: score=%.3f, mentions=%d (reddit=%d, twitter=%d)",
                                symbol, overall_score, total_mentions, reddit_mentions, twitter_mentions)

                    return sentiment_data

        except asyncio.TimeoutError:
            _logger.error("Timeout getting social sentiment for %s", symbol)
            return None
        except Exception as e:
            _logger.error("Error getting social sentiment for %s: %s", symbol, e)
            return None

    async def get_combined_sentiment(self, symbol: str) -> Optional[SentimentData]:
        """
        Get aggregated sentiment data from both news and social sources (async).
        """
        try:
            # Fetch both concurrently
            news_task = self.get_news_sentiment(symbol)
            social_task = self.get_social_sentiment(symbol)

            results = await asyncio.gather(news_task, social_task, return_exceptions=True)

            news_data = results[0] if not isinstance(results[0], Exception) else None
            social_data = results[1] if not isinstance(results[1], Exception) else None

            if not news_data and not social_data:
                _logger.warning("Both sentiment sources failed for %s", symbol)
                return None

            # Combine the data
            if news_sentiment and social_sentiment:
                # Both available - average the scores
                combined_score = (news_sentiment.sentiment_score + social_sentiment.sentiment_score) / 2.0

                combined_data = SentimentData(
                    symbol=symbol.upper(),
                    provider='finnhub',
                    timestamp=datetime.now().isoformat(),
                    sentiment_score=combined_score,
                    bullish_score=news_sentiment.bullish_score,
                    bearish_score=news_sentiment.bearish_score,
                    mention_count=social_sentiment.mention_count,
                    buzz_ratio=news_sentiment.buzz_ratio,
                    article_count=news_sentiment.article_count,
                    reddit_data=social_sentiment.reddit_data,
                    twitter_data=social_sentiment.twitter_data,
                    sector_comparison=news_sentiment.sector_comparison,
                    sources={
                        'news': True,
                        'reddit': social_sentiment.reddit_data is not None,
                        'twitter': social_sentiment.twitter_data is not None
                    },
                    raw_data={
                        'news': news_sentiment.raw_data,
                        'social': social_sentiment.raw_data
                    },
                    data_source='finnhub_combined_sentiment_api'
                )

                _logger.debug("Retrieved combined sentiment for %s: score=%.3f", symbol, combined_score)
                return combined_data

            elif news_sentiment:
                # Only news available
                _logger.debug("Using news sentiment only for %s", symbol)
                return news_sentiment

            else:
                # Only social available
                _logger.debug("Using social sentiment only for %s", symbol)
                return social_sentiment

        except Exception as e:
            _logger.error("Error getting combined sentiment for %s: %s", symbol, e)
            return None

#!/usr/bin/env python3
"""
Enhanced Screener Module

This module provides advanced screening capabilities that combine fundamental
and technical analysis based on JSON configuration.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import yfinance as yf
import pandas as pd
from src.model.telegram_bot import Fundamentals, ScreenerResult, DCFResult, ScreenerReport
from src.util.tickers_list import (
    get_us_small_cap_tickers,
    get_us_medium_cap_tickers,
    get_us_large_cap_tickers,
    get_six_tickers
)

# Use optimized batch fundamentals download for maximum performance
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.indicators.service import IndicatorService
from src.indicators.models import TickerIndicatorsRequest
from src.telegram.screener.screener_config_parser import (ScreenerConfig, FundamentalCriteria, TechnicalCriteria)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class EnhancedScreener:
    """
    Enhanced screener that combines fundamental and technical analysis
    based on JSON configuration.
    """

    def __init__(self, indicator_service: IndicatorService = None):
        """Initialize the enhanced screener."""
        self.risk_free_rate = 0.04  # 4% risk-free rate (can be made configurable)
        self.indicator_service = indicator_service or IndicatorService()

    def load_ticker_list(self, list_type: str) -> List[str]:
        """Load ticker list based on the specified type."""
        try:
            if list_type == 'us_small_cap':
                return get_us_small_cap_tickers()
            elif list_type == 'us_medium_cap':
                return get_us_medium_cap_tickers()
            elif list_type == 'us_large_cap':
                return get_us_large_cap_tickers()
            elif list_type == 'swiss_shares':
                return get_six_tickers()
            elif list_type == 'custom_list':
                # For custom lists, we'll need to implement storage/retrieval
                # For now, return an empty list
                _logger.warning("Custom list support not yet implemented")
                return []
            else:
                _logger.error("Unknown list type: %s", list_type)
                return []
        except Exception as e:
            _logger.error("Error loading ticker list %s: %s", list_type, e)
            return []

    async def run_enhanced_screener(self, config: ScreenerConfig) -> ScreenerReport:
        """Run enhanced screener with FMP integration and comprehensive analysis."""
        try:
            _logger.info("Starting enhanced screener for %s", config.list_type)

            # Stage 1: FMP Pre-filtering (if FMP criteria provided)
            tickers = []
            fmp_results = {}

            if config.fmp_criteria or config.fmp_strategy:
                _logger.info("Running FMP pre-filtering")
                tickers, fmp_results = self._run_fmp_screening(config)

                if not tickers:
                    _logger.warning("FMP screening returned no results, falling back to traditional screening")
                    tickers = self._load_ticker_list(config.list_type)
                    fmp_results = {}
            else:
                # Traditional screening without FMP
                tickers = self._load_ticker_list(config.list_type)

            if not tickers:
                return ScreenerReport(
                    list_type=config.list_type,
                    total_tickers_processed=0,
                    total_tickers_with_data=0,
                    top_results=[],
                    error="No tickers found for screening"
                )

            # Stage 2: Enhanced Analysis (Only for FMP-filtered tickers)
            # Always collect detailed fundamentals from YFinance using optimized batch operations
            if config.screener_type in ["fundamental", "hybrid"]:
                # Use YFinance's optimized batch download for all tickers (FMP-filtered or traditional)
                fundamentals_data = self.collect_fundamentals(tickers)
                _logger.info("Collected detailed fundamental data from YFinance for %d tickers", len(fundamentals_data))
            else:
                fundamentals_data = {}

            if config.screener_type in ["technical", "hybrid"]:
                technical_data = await self.collect_technical_data(tickers, config.period, config.interval, "yf")
            else:
                technical_data = {}

            # Stage 3: Apply screening criteria and generate results
            results = self.apply_enhanced_screening_criteria(
                config, fundamentals_data, technical_data
            )

            # Stage 4: Generate comprehensive report
            report = self.generate_enhanced_report(config, results, len(tickers), fmp_results)

            _logger.info("Enhanced screener completed successfully. Found %d stocks from %d FMP-filtered tickers",
                        len(results), len(tickers))

            return report

        except Exception as e:
            _logger.exception("Error running enhanced screener")
            return ScreenerReport(
                list_type=config.list_type,
                total_tickers_processed=0,
                total_tickers_with_data=0,
                top_results=[],
                error=f"Error running enhanced screener: {e}"
            )

    def _run_fmp_screening(self, config: ScreenerConfig) -> Tuple[List[str], Dict[str, Any]]:
        """Run FMP screening to get initial list of tickers."""
        try:
            from src.telegram.screener.fmp_integration import run_fmp_screening

            # Convert ScreenerConfig to dictionary for FMP integration
            screener_config = {
                "screener_type": config.screener_type,
                "list_type": config.list_type,
                "fmp_criteria": getattr(config, 'fmp_criteria', None),
                "fmp_strategy": getattr(config, 'fmp_strategy', None)
            }

            # Run FMP screening
            ticker_list, fmp_results = run_fmp_screening(screener_config)

            return ticker_list, fmp_results

        except Exception:
            _logger.exception("Error in FMP screening")
            return [], {}

    def collect_fundamentals(self, tickers: List[str]) -> Dict[str, Fundamentals]:
        """Collect fundamental data for a list of tickers using DataManager with file caching."""
        if not tickers:
            return {}

        _logger.info("Starting fundamental data collection for %d tickers using DataManager cache", len(tickers))

        try:
            # Use DataManager for cached fundamentals retrieval
            from src.data.data_manager import get_data_manager
            dm = get_data_manager()

            valid_fundamentals = {}
            for ticker in tickers:
                try:
                    # Get fundamentals from DataManager (uses file cache)
                    fundamentals_dict = dm.get_fundamentals(ticker, force_refresh=False)

                    if fundamentals_dict:
                        # Convert dict to Fundamentals object
                        fundamentals = self._dict_to_fundamentals(ticker, fundamentals_dict)

                        if self._validate_fundamental_data(fundamentals):
                            valid_fundamentals[ticker] = fundamentals
                        else:
                            _logger.warning("Insufficient fundamental data for %s, skipping", ticker)
                    else:
                        _logger.warning("No fundamental data returned for %s", ticker)

                except Exception as e:
                    _logger.warning("Failed to get fundamentals for %s: %s", ticker, e)
                    continue

            _logger.info("Fundamental data collection completed. %d/%d tickers processed successfully",
                        len(valid_fundamentals), len(tickers))
            return valid_fundamentals

        except Exception as e:
            _logger.error("Error in DataManager fundamental collection: %s", str(e))
            _logger.info("Falling back to direct downloader collection")
            return self._collect_fundamentals_fallback(tickers)

    def _dict_to_fundamentals(self, ticker: str, data_dict: Dict[str, Any]) -> Fundamentals:
        """Convert DataManager dict to Fundamentals object."""
        return Fundamentals(
            ticker=ticker.upper(),
            company_name=data_dict.get("company_name"),
            current_price=data_dict.get("current_price"),
            market_cap=data_dict.get("market_cap"),
            pe_ratio=data_dict.get("pe_ratio"),
            forward_pe=data_dict.get("forward_pe"),
            dividend_yield=data_dict.get("dividend_yield"),
            earnings_per_share=data_dict.get("earnings_per_share"),
            price_to_book=data_dict.get("price_to_book"),
            return_on_equity=data_dict.get("return_on_equity"),
            return_on_assets=data_dict.get("return_on_assets"),
            debt_to_equity=data_dict.get("debt_to_equity"),
            current_ratio=data_dict.get("current_ratio"),
            quick_ratio=data_dict.get("quick_ratio"),
            revenue=data_dict.get("revenue"),
            revenue_growth=data_dict.get("revenue_growth"),
            net_income=data_dict.get("net_income"),
            net_income_growth=data_dict.get("net_income_growth"),
            free_cash_flow=data_dict.get("free_cash_flow"),
            operating_margin=data_dict.get("operating_margin"),
            profit_margin=data_dict.get("profit_margin"),
            beta=data_dict.get("beta"),
            sector=data_dict.get("sector"),
            industry=data_dict.get("industry"),
            country=data_dict.get("country"),
            exchange=data_dict.get("exchange"),
            currency=data_dict.get("currency"),
            shares_outstanding=data_dict.get("shares_outstanding"),
            float_shares=data_dict.get("float_shares"),
            short_ratio=data_dict.get("short_ratio"),
            payout_ratio=data_dict.get("payout_ratio"),
            peg_ratio=data_dict.get("peg_ratio"),
            price_to_sales=data_dict.get("price_to_sales"),
            enterprise_value=data_dict.get("enterprise_value"),
            enterprise_value_to_ebitda=data_dict.get("enterprise_value_to_ebitda"),
            data_source=data_dict.get("data_source", "DataManager"),
            last_updated=data_dict.get("last_updated")
        )

    def _collect_fundamentals_fallback(self, tickers: List[str]) -> Dict[str, Fundamentals]:
        """Fallback method using direct Yahoo downloader when DataManager fails."""
        try:
            downloader = YahooDataDownloader()
            fundamentals_data = downloader.get_fundamentals_batch_optimized(tickers, include_financials=False)

            # Filter out invalid data
            valid_fundamentals = {}
            for ticker, fundamentals in fundamentals_data.items():
                if self._validate_fundamental_data(fundamentals):
                    valid_fundamentals[ticker] = fundamentals
                else:
                    _logger.warning("Insufficient fundamental data for %s, skipping", ticker)

            return valid_fundamentals

        except Exception as e:
            _logger.error("Fallback fundamental collection also failed: %s", str(e))
            return {}

    async def collect_technical_data(self, tickers: List[str], period: str, interval: str, provider: str) -> Dict[str, Dict[str, Any]]:
        """Collect technical data for a list of tickers using IndicatorService."""
        if not tickers:
            return {}

        _logger.info("Starting technical data collection for %d tickers using IndicatorService", len(tickers))

        # Define the technical indicators we need for screening (using registry names)
        indicators = ["rsi", "macd", "sma", "bbands", "stoch", "adx", "obv"]

        technical_data = {}
        for ticker in tickers:
            try:
                _logger.debug("Processing technical indicators for %s", ticker)

                # Create request for IndicatorService
                request = TickerIndicatorsRequest(
                    ticker=ticker,
                    timeframe=interval,
                    period=period,
                    provider=provider,
                    indicators=indicators
                )

                # Get indicators from service with error handling
                try:
                    result_set = await self.indicator_service.compute_for_ticker(request)
                except ValueError as e:
                    _logger.warning("Validation error calculating indicators for %s: %s", ticker, e)
                    continue
                except (RuntimeError, ConnectionError, TimeoutError) as e:
                    _logger.error("Service error calculating indicators for %s: %s", ticker, e)
                    continue
                except Exception as e:
                    _logger.exception("Unexpected error calculating indicators for %s: %s", ticker, e)
                    continue

                # Convert IndicatorResultSet to the expected format for backward compatibility
                technical_indicators = self._convert_indicator_result_to_technicals(result_set)
                technical_data[ticker] = technical_indicators

                _logger.debug("Successfully calculated technical indicators for %s", ticker)

            except Exception as e:
                _logger.error("Error calculating technical indicators for %s: %s", ticker, e)
                continue

        _logger.info("Technical data collection completed. %d/%d tickers processed successfully",
                    len(technical_data), len(tickers))
        return technical_data

    def _convert_indicator_result_to_technicals(self, result_set) -> Dict[str, Any]:
        """Convert IndicatorResultSet to the format expected by screener logic."""
        from src.model.telegram_bot import Technicals

        # Extract technical indicator values
        technical_values = {}

        # Map IndicatorService results to expected field names
        for name, indicator_value in result_set.technical.items():
            value = indicator_value.value

            # Map indicator names to expected format
            if name == "rsi":
                technical_values['rsi'] = value
            elif name == "macd":
                technical_values['macd'] = value
            elif name == "sma":
                technical_values['sma'] = value
            elif name == "bbands_upper":
                technical_values['bb_upper'] = value
            elif name == "bbands_middle":
                technical_values['bb_middle'] = value
            elif name == "bbands_lower":
                technical_values['bb_lower'] = value
            elif name == "stoch_k":
                technical_values['stoch_k'] = value
            elif name == "stoch_d":
                technical_values['stoch_d'] = value
            elif name == "adx":
                technical_values['adx'] = value
            elif name == "obv":
                technical_values['obv'] = value

        # Create Technicals object with default values
        technicals = Technicals(
            rsi=technical_values.get('rsi'),
            sma_fast=technical_values.get('sma'),
            sma_slow=technical_values.get('sma'),
            macd=technical_values.get('macd'),
            macd_signal=None,
            macd_histogram=None,
            stoch_k=technical_values.get('stoch_k'),
            stoch_d=technical_values.get('stoch_d'),
            adx=technical_values.get('adx'),
            plus_di=None,
            minus_di=None,
            obv=technical_values.get('obv'),
            adr=None,
            avg_adr=None,
            trend='NEUTRAL',
            bb_upper=technical_values.get('bb_upper'),
            bb_middle=technical_values.get('bb_middle'),
            bb_lower=technical_values.get('bb_lower'),
            bb_width=None,
            ema_fast=None,
            ema_slow=None,
            cci=None,
            roc=None,
            mfi=None,
            williams_r=None,
            atr=None,
            recommendations={}
        )

        return technicals

    def _get_ticker_fundamentals(self, ticker: str) -> Optional[Fundamentals]:
        """Get fundamental data for a single ticker using yfinance."""
        try:
            stock = yf.Ticker(ticker)

            # Get basic info
            info = stock.info

            # Get financial statements
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow

            # Extract key metrics
            fundamentals = Fundamentals(
                ticker=ticker,
                company_name=info.get('longName', ticker),
                current_price=info.get('regularMarketPrice'),
                pe_ratio=info.get('trailingPE'),
                forward_pe=info.get('forwardPE'),
                price_to_book=info.get('priceToBook'),
                price_to_sales=info.get('priceToSalesTrailing12Months'),
                peg_ratio=info.get('pegRatio'),
                debt_to_equity=info.get('debtToEquity'),
                current_ratio=info.get('currentRatio'),
                quick_ratio=info.get('quickRatio'),
                return_on_equity=info.get('returnOnEquity'),
                return_on_assets=info.get('returnOnAssets'),
                operating_margin=info.get('operatingMargins'),
                profit_margin=info.get('profitMargins'),
                revenue_growth=info.get('revenueGrowth'),
                net_income_growth=info.get('earningsGrowth'),
                free_cash_flow=info.get('freeCashflow'),
                dividend_yield=info.get('dividendYield'),
                payout_ratio=info.get('payoutRatio'),
                market_cap=info.get('marketCap'),
                enterprise_value=info.get('enterpriseValue'),
                sector=info.get('sector'),
                industry=info.get('industry'),
                country=info.get('country'),
                description=info.get('longBusinessSummary')
            )

            return fundamentals

        except Exception as e:
            _logger.error("Error getting fundamentals for %s: %s", ticker, e)
            return None

    def _validate_fundamental_data(self, fundamentals: Fundamentals) -> bool:
        """Validate that fundamental data has sufficient information."""
        # Check if we have at least some key metrics
        key_metrics = [
            fundamentals.pe_ratio,
            fundamentals.price_to_book,
            fundamentals.return_on_equity,
            fundamentals.market_cap
        ]

        return any(metric is not None and not pd.isna(metric) for metric in key_metrics)

    def apply_enhanced_screening_criteria(self, config: ScreenerConfig,
                                        fundamentals_data: Dict[str, Fundamentals],
                                        technical_data: Dict[str, Dict[str, Any]]) -> List[ScreenerResult]:
        """Apply enhanced screening criteria combining fundamental and technical analysis."""
        results = []

        # For fundamental/hybrid screening: use fundamental data for filtering
        # Technical data is only used for additional info when available
        if config.screener_type in ["fundamental", "hybrid"]:
            all_tickers = set(fundamentals_data.keys())
        elif config.screener_type == "technical":
            all_tickers = set(technical_data.keys())
        else:
            all_tickers = set()

        _logger.info("Starting criteria evaluation for %d tickers", len(all_tickers))
        _logger.info("Fundamental data available for: %s", list(fundamentals_data.keys()))
        _logger.info("Technical data available for: %s", list(technical_data.keys()))

        for ticker in all_tickers:
            try:
                _logger.debug("Processing ticker: %s", ticker)

                # Calculate fundamental score
                fundamental_score = 0.0
                fundamental_analysis = {}

                if config.screener_type in ["fundamental", "hybrid"] and ticker in fundamentals_data:
                    fundamental_score, fundamental_analysis = self._calculate_fundamental_score(
                        config.fundamental_criteria, fundamentals_data[ticker]
                    )
                    _logger.debug("Ticker %s - Fundamental score: %.2f", ticker, fundamental_score)

                # Calculate technical score (only if data available)
                technical_score = 0.0
                technical_analysis = {}

                if config.screener_type in ["technical", "hybrid"] and ticker in technical_data:
                    technical_score, technical_analysis = self._calculate_technical_score(
                        config.technical_criteria, technical_data[ticker]
                    )
                    _logger.debug("Ticker %s - Technical score: %.2f", ticker, technical_score)

                # Calculate composite score
                composite_score = self._calculate_composite_score(
                    config, fundamental_score, technical_score
                )
                _logger.debug("Ticker %s - Composite score: %.2f (min required: %.2f)",
                            ticker, composite_score, config.min_score)

                # Check if meets minimum score requirement
                if composite_score >= config.min_score:
                    _logger.info("Ticker %s PASSED - Score: %.2f", ticker, composite_score)

                    # Get current price
                    current_price = None
                    if ticker in fundamentals_data:
                        current_price = fundamentals_data[ticker].current_price
                    elif ticker in technical_data:
                        # Try to get price from yfinance
                        try:
                            stock = yf.Ticker(ticker)
                            current_price = stock.info.get('regularMarketPrice')
                        except:
                            pass

                    # Create screener result
                    result = ScreenerResult(
                        ticker=ticker,
                        fundamentals=fundamentals_data.get(ticker),
                        technicals=technical_data.get(ticker) if ticker in technical_data else None,
                        composite_score=composite_score,
                        dcf_valuation=self._calculate_dcf_valuation(fundamentals_data.get(ticker)),
                        recommendation=self._get_recommendation(composite_score)
                    )

                    results.append(result)
                else:
                    _logger.debug("Ticker %s FAILED - Score: %.2f < %.2f",
                                ticker, composite_score, config.min_score)

            except Exception as e:
                _logger.error("Error processing ticker %s: %s", ticker, e)
                continue

        # Sort by composite score (descending) and limit results
        results.sort(key=lambda x: x.composite_score, reverse=True)
        _logger.info("Final results: %d stocks passed criteria", len(results))
        return results[:config.max_results]

    def _calculate_fundamental_score(self, criteria: List[FundamentalCriteria],
                                   fundamentals: Fundamentals) -> Tuple[float, Dict[str, Any]]:
        """Calculate fundamental score based on criteria."""
        if not criteria:
            return 0.0, {}

        total_score = 0.0
        total_weight = 0.0
        analysis = {}

        for criterion in criteria:
            indicator_value = self._get_fundamental_value(fundamentals, criterion.indicator)

            _logger.debug("Evaluating %s: value=%s, criterion=%s",
                         criterion.indicator, indicator_value, criterion)

            if indicator_value is not None and not pd.isna(indicator_value):
                score = self._evaluate_fundamental_criterion(criterion, indicator_value)
                weighted_score = score * criterion.weight

                total_score += weighted_score
                total_weight += criterion.weight

                analysis[criterion.indicator] = {
                    'value': indicator_value,
                    'score': score,
                    'weighted_score': weighted_score,
                    'criterion': criterion
                }

                _logger.debug("  %s: value=%s, score=%.2f, weighted=%.2f",
                             criterion.indicator, indicator_value, score, weighted_score)
            elif criterion.required:
                # If required criterion is missing, return 0 score
                _logger.debug("  %s: MISSING REQUIRED VALUE - returning 0 score", criterion.indicator)
                return 0.0, {}
            else:
                _logger.debug("  %s: MISSING VALUE (not required) - skipping", criterion.indicator)

        # Normalize score to 0-10 scale
        final_score = (total_score / total_weight * 10) if total_weight > 0 else 0.0
        _logger.debug("Final fundamental score: %.2f (total=%.2f, weight=%.2f)",
                     final_score, total_score, total_weight)
        return final_score, analysis

    def _get_fundamental_value(self, fundamentals: Fundamentals, indicator: str) -> Optional[float]:
        """Get fundamental value for a specific indicator."""
        indicator_mapping = {
            'PE': fundamentals.pe_ratio,
            'Forward_PE': fundamentals.forward_pe,
            'PB': fundamentals.price_to_book,
            'PS': fundamentals.price_to_sales,
            'PEG': fundamentals.peg_ratio,
            'Debt_Equity': fundamentals.debt_to_equity,
            'Current_Ratio': fundamentals.current_ratio,
            'Quick_Ratio': fundamentals.quick_ratio,
            'ROE': fundamentals.return_on_equity,
            'ROA': fundamentals.return_on_assets,
            'Operating_Margin': fundamentals.operating_margin,
            'Profit_Margin': fundamentals.profit_margin,
            'Revenue_Growth': fundamentals.revenue_growth,
            'Net_Income_Growth': fundamentals.net_income_growth,
            'Free_Cash_Flow': fundamentals.free_cash_flow,
            'Dividend_Yield': fundamentals.dividend_yield,
            'Payout_Ratio': fundamentals.payout_ratio
        }

        return indicator_mapping.get(indicator)

    def _evaluate_fundamental_criterion(self, criterion: FundamentalCriteria, value: float) -> float:
        """Evaluate a fundamental criterion and return a score (0-1)."""
        if criterion.operator == "max":
            if value <= criterion.value:
                return 1.0
            else:
                # Linear penalty for exceeding max
                penalty = min(1.0, (value - criterion.value) / criterion.value)
                return max(0.0, 1.0 - penalty)

        elif criterion.operator == "min":
            if value >= criterion.value:
                return 1.0
            else:
                # Linear penalty for being below min
                penalty = min(1.0, (criterion.value - value) / criterion.value)
                return max(0.0, 1.0 - penalty)

        elif criterion.operator == "range":
            if isinstance(criterion.value, dict):
                min_val = criterion.value.get("min")
                max_val = criterion.value.get("max")

                if min_val is not None and max_val is not None:
                    if min_val <= value <= max_val:
                        return 1.0
                    else:
                        # Penalty for being outside range
                        if value < min_val:
                            penalty = min(1.0, (min_val - value) / min_val)
                        else:
                            penalty = min(1.0, (value - max_val) / max_val)
                        return max(0.0, 1.0 - penalty)

        return 0.0

    def _calculate_technical_score(self, criteria: List[TechnicalCriteria],
                                 technical_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calculate technical score based on criteria."""
        if not criteria:
            return 0.0, {}

        total_score = 0.0
        total_weight = 0.0
        analysis = {}

        for criterion in criteria:
            score = self._evaluate_technical_criterion(criterion, technical_data)
            weighted_score = score * criterion.weight

            total_score += weighted_score
            total_weight += criterion.weight

            analysis[criterion.indicator] = {
                'score': score,
                'weighted_score': weighted_score,
                'criterion': criterion
            }

        # Normalize score to 0-10 scale
        final_score = (total_score / total_weight * 10) if total_weight > 0 else 0.0
        return final_score, analysis

    def _evaluate_technical_criterion(self, criterion: TechnicalCriteria,
                                    technical_data: Dict[str, Any]) -> float:
        """Evaluate a technical criterion and return a score (0-1)."""
        try:
            # technical_data is a Technicals object, not a dictionary
            technicals = technical_data

            # Get the indicator value from the Technicals object
            indicator_name = criterion.indicator.lower()

            if not hasattr(technicals, indicator_name):
                _logger.warning("Indicator %s not found in technicals data", indicator_name)
                return 0.0

            current_value = getattr(technicals, indicator_name)

            if current_value is None or pd.isna(current_value):
                return 0.0

            # Evaluate condition
            condition = criterion.condition
            operator = condition.get("operator")

            if operator == "<":
                threshold = condition.get("value")
                return 1.0 if current_value < threshold else 0.0

            elif operator == ">":
                threshold = condition.get("value")
                return 1.0 if current_value > threshold else 0.0

            elif operator == "range":
                min_val = condition.get("min")
                max_val = condition.get("max")
                if min_val is not None and max_val is not None:
                    return 1.0 if min_val <= current_value <= max_val else 0.0

            elif operator == "above":
                # For moving averages, check if price is above MA
                # We need to get the current price from OHLCV data
                # For now, we'll use a simple comparison
                return 1.0 if current_value > 0 else 0.0

            elif operator == "above_signal":
                # For MACD, check if MACD is above signal line
                if indicator_name == "macd" and hasattr(technicals, "macd_signal"):
                    return 1.0 if current_value > technicals.macd_signal else 0.0
                return 0.0

            elif operator == "below_lower_band":
                # For Bollinger Bands
                if indicator_name == "bb_lower" and hasattr(technicals, "bb_upper"):
                    # We need current price to compare with lower band
                    # For now, return 0.0 as we don't have current price in this context
                    return 0.0

            elif operator == "not_above_upper_band":
                # For Bollinger Bands
                if indicator_name == "bb_upper":
                    # We need current price to compare with upper band
                    # For now, return 0.0 as we don't have current price in this context
                    return 0.0

            elif operator == "between_bands":
                # For Bollinger Bands
                if hasattr(technicals, "bb_upper") and hasattr(technicals, "bb_lower"):
                    # We need current price to compare with bands
                    # For now, return 0.0 as we don't have current price in this context
                    return 0.0

        except Exception as e:
            _logger.error("Error evaluating technical criterion %s: %s", criterion.indicator, e)
            return 0.0

        return 0.0

    def _calculate_composite_score(self, config: ScreenerConfig,
                                 fundamental_score: float,
                                 technical_score: float) -> float:
        """Calculate composite score based on screener type."""
        if config.screener_type == "fundamental":
            return fundamental_score
        elif config.screener_type == "technical":
            return technical_score
        elif config.screener_type == "hybrid":
            # Weight fundamental analysis more heavily in hybrid mode
            fundamental_weight = 0.7 if config.include_fundamental_analysis else 0.0
            technical_weight = 0.3 if config.include_technical_analysis else 0.0

            # Normalize weights
            total_weight = fundamental_weight + technical_weight
            if total_weight > 0:
                fundamental_weight /= total_weight
                technical_weight /= total_weight

            return (fundamental_score * fundamental_weight +
                   technical_score * technical_weight)
        else:
            return 0.0

    def _get_recommendation(self, composite_score: float) -> str:
        """Get recommendation based on composite score."""
        if composite_score >= 8.0:
            return "STRONG_BUY"
        elif composite_score >= 7.0:
            return "BUY"
        elif composite_score >= 6.0:
            return "HOLD"
        elif composite_score >= 5.0:
            return "WEAK_HOLD"
        else:
            return "SELL"

    def _calculate_dcf_valuation(self, fundamentals: Optional[Fundamentals]) -> Optional[DCFResult]:
        """Calculate DCF valuation if fundamental data is available."""
        if not fundamentals:
            _logger.debug("DCF: No fundamentals data available")
            return None

        try:
            # Simple DCF calculation (can be enhanced)
            free_cash_flow = fundamentals.free_cash_flow
            if free_cash_flow is None or free_cash_flow <= 0:
                _logger.debug("DCF for %s: Free cash flow missing or <= 0 (value: %s)",
                            fundamentals.ticker, free_cash_flow)
                return None

            # Assume 5% growth rate and 10% discount rate
            growth_rate = 0.05
            discount_rate = 0.10

            # Terminal value calculation
            terminal_value = free_cash_flow * (1 + growth_rate) / (discount_rate - growth_rate)

            # Present value of terminal value
            present_value = terminal_value / ((1 + discount_rate) ** 5)

            # Fair value per share (assuming market cap is available)
            if not fundamentals.market_cap:
                _logger.debug("DCF for %s: Market cap missing", fundamentals.ticker)
                return None

            if not fundamentals.current_price:
                _logger.debug("DCF for %s: Current price missing", fundamentals.ticker)
                return None

            shares_outstanding = fundamentals.market_cap / fundamentals.current_price
            if shares_outstanding <= 0:
                _logger.debug("DCF for %s: Invalid shares outstanding calculation (market_cap: %s, current_price: %s)",
                            fundamentals.ticker, fundamentals.market_cap, fundamentals.current_price)
                return None

            fair_value_per_share = present_value / shares_outstanding

            _logger.debug("DCF for %s: Successfully calculated (FCF: %s, Fair Value: %s)",
                        fundamentals.ticker, free_cash_flow, fair_value_per_share)

            return DCFResult(
                ticker=fundamentals.ticker,
                fair_value=fair_value_per_share,
                assumptions={
                    'growth_rate': growth_rate,
                    'discount_rate': discount_rate,
                    'free_cash_flow': free_cash_flow
                }
            )

        except Exception as e:
            _logger.exception("Error calculating DCF valuation for %s: %s", fundamentals.ticker, e)

        return None

    def generate_enhanced_report(self, config: ScreenerConfig,
                               results: List[ScreenerResult],
                               total_tickers: int,
                               fmp_results: Dict[str, Any] = None) -> ScreenerReport:
        """Generate enhanced screener report with optional FMP information."""
        # Add FMP information to the report if available
        report = ScreenerReport(
            list_type=config.list_type,
            total_tickers_processed=total_tickers,
            total_tickers_with_data=len(results),
            top_results=results,
            error=None
        )

        # Store FMP results in the report for later use
        if fmp_results:
            report.fmp_results = fmp_results

        return report

    def format_enhanced_telegram_message(self, report: ScreenerReport, config: ScreenerConfig) -> str:
        """Format enhanced screener report for Telegram."""
        if report.error:
            return f"❌ **Screener Error**\n\n{report.error}"

        if not report.top_results:
            return "📊 **Enhanced Screener Results**\n\nNo stocks found matching your criteria."

        message = "📊 **Enhanced Screener Results**\n\n"
        message += f"🔍 **List Type**: {config.list_type.replace('_', ' ').title()}\n"
        message += f"📈 **Screener Type**: {config.screener_type.title()}\n"

        # Add FMP information if available
        if hasattr(report, 'fmp_results') and report.fmp_results:
            fmp_criteria = report.fmp_results.get('fmp_criteria', {})
            fmp_results = report.fmp_results.get('fmp_results', [])
            message += f"🚀 **FMP Pre-filtered**: {len(fmp_results)} stocks\n"
            if fmp_criteria:
                message += f"🎯 **FMP Criteria**: {', '.join(fmp_criteria.keys())}\n"

        message += f"📊 **Processed**: {report.total_tickers_processed} tickers\n"
        message += f"✅ **Found**: {len(report.top_results)} matching stocks\n"
        message += f"🎯 **Min Score**: {config.min_score}/10\n\n"

        message += "🏆 **Top Results**:\n\n"

        for i, result in enumerate(report.top_results[:10], 1):
            message += f"{i}. **{result.ticker}** "

            if result.fundamentals and result.fundamentals.current_price:
                message += f"(${result.fundamentals.current_price:.2f}) "

            message += f"Score: {result.composite_score:.1f}/10 "

            # Add recommendation emoji
            if result.recommendation == "STRONG_BUY":
                message += "🟢 "
            elif result.recommendation == "BUY":
                message += "🟢 "
            elif result.recommendation == "HOLD":
                message += "🟡 "
            elif result.recommendation == "WEAK_HOLD":
                message += "🟠 "
            else:
                message += "🔴 "

            message += f"({result.recommendation.replace('_', ' ')})"

            # Add fundamental score if available
            if config.screener_type in ["fundamental", "hybrid"] and hasattr(result, 'fundamental_score') and result.fundamental_score > 0:
                message += f"\n   📊 Fundamental: {result.fundamental_score:.1f}/10"

            # Add technical score if available
            if config.screener_type in ["technical", "hybrid"] and hasattr(result, 'technical_score') and result.technical_score > 0:
                message += f"\n   📈 Technical: {result.technical_score:.1f}/10"

            # Add DCF valuation if available
            if result.dcf_valuation and result.dcf_valuation.fair_value and result.fundamentals and result.fundamentals.current_price:
                current_price = result.fundamentals.current_price
                fair_value = result.dcf_valuation.fair_value
                upside = ((fair_value - current_price) / current_price) * 100
                if upside > 0:
                    message += f"\n   💰 DCF Upside: +{upside:.1f}%"
                else:
                    message += f"\n   💰 DCF Downside: {upside:.1f}%"

            message += "\n\n"

        # Add configuration summary
        message += "⚙️ **Configuration Summary**:\n"
        if config.fundamental_criteria:
            message += f"📊 Fundamental Criteria: {len(config.fundamental_criteria)} indicators\n"
        if config.technical_criteria:
            message += f"📈 Technical Criteria: {len(config.technical_criteria)} indicators\n"
        message += f"⏰ Period: {config.period}, Interval: {config.interval}\n"
        message += f"🎯 Max Results: {config.max_results}\n"

        return message

    def _convert_fmp_to_fundamentals(self, fmp_results: List[Dict[str, Any]]) -> Dict[str, Fundamentals]:
        """Convert FMP stock screener results to Fundamentals objects with basic data only."""
        fundamentals_data = {}

        for stock in fmp_results:
            try:
                ticker = stock.get('symbol', '').upper()
                if not ticker:
                    continue

                # FMP stock screener only provides basic data, not comprehensive fundamentals
                # We'll use this for initial filtering but need YFinance for detailed analysis
                fundamentals = Fundamentals(
                    ticker=ticker,
                    company_name=stock.get('companyName', 'Unknown'),
                    current_price=stock.get('price', 0.0),
                    market_cap=stock.get('marketCap', 0.0),
                    beta=stock.get('beta', 0.0),
                    sector=stock.get('sector', 'Unknown'),
                    industry=stock.get('industry', 'Unknown'),
                    country=stock.get('country', 'Unknown'),
                    exchange=stock.get('exchange', 'Unknown'),
                    # Calculate dividend yield from lastAnnualDividend and price
                    dividend_yield=(stock.get('lastAnnualDividend', 0.0) / stock.get('price', 1.0)) if stock.get('price', 0) > 0 else 0.0,
                    # Set other fields to None as FMP doesn't provide them
                    pe_ratio=None,
                    forward_pe=None,
                    earnings_per_share=None,
                    price_to_book=None,
                    return_on_equity=None,
                    return_on_assets=None,
                    debt_to_equity=None,
                    current_ratio=None,
                    quick_ratio=None,
                    revenue=None,
                    revenue_growth=None,
                    net_income=None,
                    net_income_growth=None,
                    free_cash_flow=None,
                    operating_margin=None,
                    profit_margin=None,
                    currency='USD',
                    shares_outstanding=None,
                    float_shares=None,
                    short_ratio=None,
                    payout_ratio=None,
                    peg_ratio=None,
                    price_to_sales=None,
                    enterprise_value=None,
                    enterprise_value_to_ebitda=None,
                    data_source="Financial Modeling Prep (Basic)",
                    last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )

                fundamentals_data[ticker] = fundamentals
                _logger.debug("Converted basic FMP data for %s", ticker)

            except Exception as e:
                _logger.exception("Error converting FMP data for %s: %s", stock.get('symbol', 'Unknown'), e)
                continue

        _logger.info("Converted basic FMP data for %d/%d stocks (detailed fundamentals will be fetched from YFinance)",
                    len(fundamentals_data), len(fmp_results))
        return fundamentals_data

    def _load_ticker_list(self, list_type: str) -> List[str]:
        """Load ticker list based on the specified type."""
        try:
            if list_type == 'us_small_cap':
                return get_us_small_cap_tickers()
            elif list_type == 'us_medium_cap':
                return get_us_medium_cap_tickers()
            elif list_type == 'us_large_cap':
                return get_us_large_cap_tickers()
            elif list_type == 'swiss_shares':
                # For Swiss shares, we should use FMP with exchange=SIX
                # This is a fallback in case FMP is not available
                _logger.warning("Swiss shares should be handled via FMP with exchange=SIX. Using CSV fallback.")
                return get_six_tickers()
            elif list_type == 'custom_list':
                # For custom lists, we'll need to implement storage/retrieval
                # For now, return an empty list
                _logger.warning("Custom list support not yet implemented")
                return []
            else:
                _logger.error("Unknown list type: %s", list_type)
                return []
        except Exception as e:
            _logger.exception("Error loading ticker list %s: %s", list_type, e)
            return []

    def _normalize_dividend_yield(self, dividend_yield_value) -> Optional[float]:
        """
        Normalize dividend yield value to percentage format.

        This method is kept for compatibility with other data sources that might
        return dividend yield in different formats.

        Args:
            dividend_yield_value: Raw dividend yield value from data source

        Returns:
            Normalized dividend yield as percentage (e.g., 4.61 for 4.61%) or None if invalid
        """
        if dividend_yield_value is None:
            return None

        try:
            dividend_yield = float(dividend_yield_value)

            # Handle extreme values that are clearly wrong
            if dividend_yield > 1000.0:  # More than 1000% dividend yield is impossible
                _logger.warning("Extreme dividend yield value detected: %s, setting to 0", dividend_yield_value)
                return 0.0

            # Ensure the result is reasonable (between 0 and 100)
            if dividend_yield < 0 or dividend_yield > 100:
                _logger.warning("Unreasonable dividend yield value: %s, setting to 0", dividend_yield)
                return 0.0

            return dividend_yield

        except (ValueError, TypeError):
            _logger.exception("Invalid dividend yield value: %s", dividend_yield_value)
            return None


# Global enhanced screener instance
enhanced_screener = EnhancedScreener(indicator_service=IndicatorService())

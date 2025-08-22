#!/usr/bin/env python3
"""
Enhanced Screener Module

This module provides advanced screening capabilities that combine fundamental
and technical analysis based on JSON configuration.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import yfinance as yf
import pandas as pd
import numpy as np
from src.model.telegram_bot import Fundamentals, ScreenerResult, DCFResult, ScreenerReport
from src.util.tickers_list import (
    get_us_small_cap_tickers,
    get_us_medium_cap_tickers,
    get_us_large_cap_tickers,
    get_six_tickers
)
from src.common import get_ohlcv
from src.common.technicals import calculate_technicals_from_df
from src.frontend.telegram.screener.screener_config_parser import (
    ScreenerConfig, FundamentalCriteria, TechnicalCriteria
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class EnhancedScreener:
    """
    Enhanced screener that combines fundamental and technical analysis
    based on JSON configuration.
    """

    def __init__(self):
        """Initialize the enhanced screener."""
        self.risk_free_rate = 0.04  # 4% risk-free rate (can be made configurable)

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

    def run_enhanced_screener(self, config: ScreenerConfig) -> ScreenerReport:
        """Run enhanced screener with combined fundamental and technical analysis."""
        _logger.info(f"Starting enhanced screener for {config.list_type}")

        try:
            # Load ticker list
            tickers = self.load_ticker_list(config.list_type)
            if not tickers:
                return ScreenerReport(
                    list_type=config.list_type,
                    total_tickers_processed=0,
                    total_tickers_with_data=0,
                    top_results=[],
                    error="No tickers found for the specified list type"
                )

            # Collect data based on screener type
            if config.screener_type in ["fundamental", "hybrid"]:
                fundamentals_data = self.collect_fundamentals(tickers)
            else:
                fundamentals_data = {}

            if config.screener_type in ["technical", "hybrid"]:
                technical_data = self.collect_technical_data(tickers, config.period, config.interval, config.provider)
            else:
                technical_data = {}

            # Apply screening criteria
            results = self.apply_enhanced_screening_criteria(
                config, fundamentals_data, technical_data
            )

            # Generate report
            report = self.generate_enhanced_report(config, results, len(tickers))

            _logger.info(f"Enhanced screener completed successfully. Found {len(results)} stocks")
            return report

        except Exception as e:
            _logger.error(f"Error running enhanced screener: {e}")
            return ScreenerReport(
                list_type=config.list_type,
                total_tickers_processed=0,
                total_tickers_with_data=0,
                top_results=[],
                error=str(e)
            )

    def collect_fundamentals(self, tickers: List[str]) -> Dict[str, Fundamentals]:
        """Collect fundamental data for a list of tickers."""
        fundamentals_data = {}
        total_tickers = len(tickers)

        _logger.info("Starting fundamental data collection for %d tickers", total_tickers)

        for i, ticker in enumerate(tickers, 1):
            try:
                _logger.info("Processing %s (%d/%d)", ticker, i, total_tickers)

                # Get fundamental data using yfinance
                fundamentals = self._get_ticker_fundamentals(ticker)

                if fundamentals and self._validate_fundamental_data(fundamentals):
                    fundamentals_data[ticker] = fundamentals
                    _logger.info("Successfully collected fundamental data for %s", ticker)
                else:
                    _logger.warning("Insufficient fundamental data for %s, skipping", ticker)

                # Rate limiting - sleep between requests
                time.sleep(0.1)  # 100ms delay between requests

            except Exception as e:
                _logger.error("Error collecting fundamental data for %s: %s", ticker, e)
                continue

        _logger.info("Fundamental data collection completed. %d/%d tickers processed successfully",
                    len(fundamentals_data), total_tickers)
        return fundamentals_data

    def collect_technical_data(self, tickers: List[str], period: str, interval: str, provider: str) -> Dict[str, Dict[str, Any]]:
        """Collect technical data for a list of tickers."""
        technical_data = {}
        total_tickers = len(tickers)

        _logger.info("Starting technical data collection for %d tickers", total_tickers)

        for i, ticker in enumerate(tickers, 1):
            try:
                _logger.info("Processing %s (%d/%d)", ticker, i, total_tickers)

                # Get OHLCV data
                df = get_ohlcv(ticker, interval, period, provider)

                if df is not None and not df.empty:
                    # Calculate technical indicators
                    df_with_technicals, technicals = calculate_technicals_from_df(df)

                    if technicals:
                        technical_data[ticker] = {
                            'ohlcv': df_with_technicals,
                            'technicals': technicals,
                            'current_price': df['close'].iloc[-1] if not df.empty else None
                        }
                        _logger.info("Successfully collected technical data for %s", ticker)
                    else:
                        _logger.warning("No technical indicators calculated for %s", ticker)
                else:
                    _logger.warning("No OHLCV data available for %s", ticker)

                # Rate limiting - sleep between requests
                time.sleep(0.1)  # 100ms delay between requests

            except Exception as e:
                _logger.error("Error collecting technical data for %s: %s", ticker, e)
                continue

        _logger.info("Technical data collection completed. %d/%d tickers processed successfully",
                    len(technical_data), total_tickers)
        return technical_data

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

        # Get all tickers that have data
        all_tickers = set()
        if config.screener_type in ["fundamental", "hybrid"]:
            all_tickers.update(fundamentals_data.keys())
        if config.screener_type in ["technical", "hybrid"]:
            all_tickers.update(technical_data.keys())

        for ticker in all_tickers:
            try:
                # Calculate fundamental score
                fundamental_score = 0.0
                fundamental_analysis = {}

                if config.screener_type in ["fundamental", "hybrid"] and ticker in fundamentals_data:
                    fundamental_score, fundamental_analysis = self._calculate_fundamental_score(
                        config.fundamental_criteria, fundamentals_data[ticker]
                    )

                # Calculate technical score
                technical_score = 0.0
                technical_analysis = {}

                if config.screener_type in ["technical", "hybrid"] and ticker in technical_data:
                    technical_score, technical_analysis = self._calculate_technical_score(
                        config.technical_criteria, technical_data[ticker]
                    )

                # Calculate composite score
                composite_score = self._calculate_composite_score(
                    config, fundamental_score, technical_score
                )

                # Check if meets minimum score requirement
                if composite_score >= config.min_score:
                    # Get current price
                    current_price = None
                    if ticker in technical_data:
                        current_price = technical_data[ticker].get('current_price')
                    elif ticker in fundamentals_data:
                        # Try to get price from yfinance
                        try:
                            stock = yf.Ticker(ticker)
                            current_price = stock.info.get('regularMarketPrice')
                        except:
                            pass

                    # Create screener result
                    result = ScreenerResult(
                        ticker=ticker,
                        current_price=current_price,
                        fundamental_score=fundamental_score,
                        technical_score=technical_score,
                        composite_score=composite_score,
                        fundamental_analysis=fundamental_analysis,
                        technical_analysis=technical_analysis,
                        recommendation=self._get_recommendation(composite_score),
                        dcf_valuation=self._calculate_dcf_valuation(fundamentals_data.get(ticker))
                    )

                    results.append(result)

            except Exception as e:
                _logger.error("Error processing ticker %s: %s", ticker, e)
                continue

        # Sort by composite score (descending) and limit results
        results.sort(key=lambda x: x.composite_score, reverse=True)
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
            elif criterion.required:
                # If required criterion is missing, return 0 score
                return 0.0, {}

        # Normalize score to 0-10 scale
        final_score = (total_score / total_weight * 10) if total_weight > 0 else 0.0
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
            technicals = technical_data.get('technicals', {})
            indicator_data = technicals.get(criterion.indicator)

            if indicator_data is None:
                return 0.0

            # Get the latest value
            if isinstance(indicator_data, pd.Series):
                current_value = indicator_data.iloc[-1]
            elif isinstance(indicator_data, dict):
                # For indicators that return multiple values (like Bollinger Bands)
                current_value = indicator_data
            else:
                current_value = indicator_data

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
                if isinstance(current_value, dict) and 'close' in technical_data.get('ohlcv', {}):
                    close_price = technical_data['ohlcv']['close'].iloc[-1]
                    return 1.0 if close_price > current_value else 0.0

            elif operator == "below_lower_band":
                # For Bollinger Bands
                if isinstance(current_value, dict) and 'lower' in current_value:
                    close_price = technical_data['ohlcv']['close'].iloc[-1]
                    return 1.0 if close_price < current_value['lower'].iloc[-1] else 0.0

            elif operator == "not_above_upper_band":
                # For Bollinger Bands
                if isinstance(current_value, dict) and 'upper' in current_value:
                    close_price = technical_data['ohlcv']['close'].iloc[-1]
                    return 1.0 if close_price <= current_value['upper'].iloc[-1] else 0.0

            elif operator == "between_bands":
                # For Bollinger Bands
                if isinstance(current_value, dict) and 'upper' in current_value and 'lower' in current_value:
                    close_price = technical_data['ohlcv']['close'].iloc[-1]
                    upper = current_value['upper'].iloc[-1]
                    lower = current_value['lower'].iloc[-1]
                    return 1.0 if lower <= close_price <= upper else 0.0

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
            return None

        try:
            # Simple DCF calculation (can be enhanced)
            free_cash_flow = fundamentals.free_cash_flow
            if free_cash_flow is None or free_cash_flow <= 0:
                return None

            # Assume 5% growth rate and 10% discount rate
            growth_rate = 0.05
            discount_rate = 0.10

            # Terminal value calculation
            terminal_value = free_cash_flow * (1 + growth_rate) / (discount_rate - growth_rate)

            # Present value of terminal value
            present_value = terminal_value / ((1 + discount_rate) ** 5)

            # Fair value per share (assuming market cap is available)
            if fundamentals.market_cap:
                shares_outstanding = fundamentals.market_cap / fundamentals.current_price if fundamentals.current_price else None
                if shares_outstanding:
                    fair_value_per_share = present_value / shares_outstanding

                    return DCFResult(
                        fair_value=fair_value_per_share,
                        upside_potential=((fair_value_per_share - fundamentals.current_price) / fundamentals.current_price * 100) if fundamentals.current_price else None,
                        assumptions={
                            'growth_rate': growth_rate,
                            'discount_rate': discount_rate,
                            'free_cash_flow': free_cash_flow
                        }
                    )

        except Exception as e:
            _logger.error("Error calculating DCF valuation: %s", e)

        return None

    def generate_enhanced_report(self, config: ScreenerConfig,
                               results: List[ScreenerResult],
                               total_tickers: int) -> ScreenerReport:
        """Generate enhanced screener report."""
        return ScreenerReport(
            list_type=config.list_type,
            total_tickers_processed=total_tickers,
            total_tickers_with_data=len(results),
            top_results=results,
            error=None
        )

    def format_enhanced_telegram_message(self, report: ScreenerReport, config: ScreenerConfig) -> str:
        """Format enhanced screener report for Telegram."""
        if report.error:
            return f"❌ **Screener Error**\n\n{report.error}"

        if not report.top_results:
            return f"📊 **Enhanced Screener Results**\n\nNo stocks found matching your criteria."

        message = f"📊 **Enhanced Screener Results**\n\n"
        message += f"🔍 **List Type**: {config.list_type.replace('_', ' ').title()}\n"
        message += f"📈 **Screener Type**: {config.screener_type.title()}\n"
        message += f"📊 **Processed**: {report.total_tickers_processed} tickers\n"
        message += f"✅ **Found**: {len(report.top_results)} matching stocks\n"
        message += f"🎯 **Min Score**: {config.min_score}/10\n\n"

        message += "🏆 **Top Results**:\n\n"

        for i, result in enumerate(report.top_results[:10], 1):
            message += f"{i}. **{result.ticker}** "

            if result.current_price:
                message += f"(${result.current_price:.2f}) "

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
            if config.screener_type in ["fundamental", "hybrid"] and result.fundamental_score > 0:
                message += f"\n   📊 Fundamental: {result.fundamental_score:.1f}/10"

            # Add technical score if available
            if config.screener_type in ["technical", "hybrid"] and result.technical_score > 0:
                message += f"\n   📈 Technical: {result.technical_score:.1f}/10"

            # Add DCF valuation if available
            if result.dcf_valuation and result.dcf_valuation.upside_potential:
                upside = result.dcf_valuation.upside_potential
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


# Global enhanced screener instance
enhanced_screener = EnhancedScreener()

"""
Fundamental Screener Module

This module provides comprehensive fundamental screening capabilities for stocks,
including valuation analysis, financial health assessment, and DCF calculations.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import yfinance as yf
import pandas as pd
import numpy as np
from src.model.telegram_bot import Fundamentals, ScreenerResult, DCFResult, ScreenerReport
from src.indicators.service import IndicatorService
from src.indicators.models import TickerIndicatorsRequest
from src.util.tickers_list import (
    get_us_small_cap_tickers,
    get_us_medium_cap_tickers,
    get_us_large_cap_tickers,
    get_six_tickers
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class FundamentalScreener:
    """Core fundamental screener for undervalued stocks."""

    def __init__(self, indicator_service: IndicatorService = None):
        """Initialize the fundamental screener."""
        self.indicator_service = indicator_service or IndicatorService()
        self.screening_thresholds = {
            # Valuation ratios
            'pe_ratio': {'max': 15, 'weight': 0.2},
            'price_to_book': {'max': 1.5, 'weight': 0.15},
            'price_to_sales': {'max': 1.0, 'weight': 0.1},
            'peg_ratio': {'max': 1.5, 'weight': 0.1},

            # Financial health
            'debt_to_equity': {'max': 0.5, 'weight': 0.1},
            'current_ratio': {'min': 1.5, 'weight': 0.1},
            'quick_ratio': {'min': 1.0, 'weight': 0.05},

            # Profitability
            'return_on_equity': {'min': 15, 'weight': 0.1},
            'return_on_assets': {'min': 5, 'weight': 0.05},
            'operating_margin': {'min': 10, 'weight': 0.05},
            'profit_margin': {'min': 5, 'weight': 0.05},

            # Growth
            'revenue_growth': {'min': 5, 'weight': 0.05},
            'net_income_growth': {'min': 5, 'weight': 0.05},

            # Cash flow
            'free_cash_flow': {'min': 0, 'weight': 0.1},  # Must be positive
        }

        self.risk_free_rate = 0.04  # 4% risk-free rate (can be made configurable)
        self.max_results = 10  # Top 10 undervalued stocks

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
        except Exception:
            _logger.exception("Error loading ticker list %s:", list_type)
            return []

    async def collect_fundamentals(self, tickers: List[str]) -> Dict[str, Fundamentals]:
        """Collect fundamental data for a list of tickers using IndicatorService."""
        fundamentals_data = {}
        total_tickers = len(tickers)

        _logger.info("Starting fundamental data collection for %d tickers using IndicatorService", total_tickers)

        # Define fundamental indicators we need (using registry names)
        fundamental_indicators = ["pe", "pb", "ps", "roe", "roa", "de_ratio", "current_ratio", "quick_ratio",
                                "dividend_yield", "market_cap", "revenue_growth", "net_income_growth",
                                "free_cash_flow", "operating_margin", "profit_margin"]

        for i, ticker in enumerate(tickers, 1):
            try:
                _logger.info("Processing %s (%d/%d)", ticker, i, total_tickers)

                # Try to get fundamental data using IndicatorService first
                fundamentals = None
                try:
                    request = TickerIndicatorsRequest(
                        ticker=ticker,
                        timeframe="1d",
                        period="1y",
                        indicators=fundamental_indicators
                    )

                    result_set = await self.indicator_service.compute_for_ticker(request)
                    fundamentals = self._convert_indicator_result_to_fundamentals(ticker, result_set)

                except Exception as service_error:
                    _logger.warning("IndicatorService failed for %s: %s, falling back to yfinance", ticker, service_error)
                    fundamentals = None

                # Fallback to yfinance if IndicatorService fails or returns insufficient data
                if not fundamentals or not self._validate_fundamental_data(fundamentals):
                    _logger.info("Falling back to yfinance for %s", ticker)
                    fundamentals = self._get_ticker_fundamentals(ticker)

                if fundamentals and self._validate_fundamental_data(fundamentals):
                    fundamentals_data[ticker] = fundamentals
                    _logger.info("Successfully collected data for %s", ticker)
                else:
                    _logger.warning("Insufficient data for %s, skipping", ticker)

                # Rate limiting - sleep between requests
                time.sleep(0.1)  # 100ms delay between requests

            except Exception:
                _logger.exception("Error collecting data for %s:", ticker)
                continue

        _logger.info("Fundamental data collection completed. %d/%d tickers processed successfully", len(fundamentals_data), total_tickers)
        return fundamentals_data

    async def _get_fundamentals_from_service(self, ticker: str, indicators: List[str]) -> Optional[Fundamentals]:
        """Get fundamental data using IndicatorService."""
        try:
            request = TickerIndicatorsRequest(
                ticker=ticker,
                timeframe="1d",
                period="1y",
                indicators=indicators
            )

            result_set = await self.indicator_service.compute_for_ticker(request)
            return self._convert_indicator_result_to_fundamentals(ticker, result_set)

        except Exception:
            _logger.exception("Error getting fundamentals from service for %s:", ticker)
            return None

    def _convert_indicator_result_to_fundamentals(self, ticker: str, result_set) -> Optional[Fundamentals]:
        """Convert IndicatorResultSet to Fundamentals object."""
        try:
            # Extract fundamental values from result set
            fundamental_values = {}

            for name, indicator_value in result_set.fundamental.items():
                value = indicator_value.value

                # Map indicator names to Fundamentals fields
                if name == "pe":
                    fundamental_values['pe_ratio'] = value
                elif name == "pb":
                    fundamental_values['price_to_book'] = value
                elif name == "ps":
                    fundamental_values['price_to_sales'] = value
                elif name == "roe":
                    fundamental_values['return_on_equity'] = value
                elif name == "roa":
                    fundamental_values['return_on_assets'] = value
                elif name == "de_ratio":
                    fundamental_values['debt_to_equity'] = value
                elif name == "current_ratio":
                    fundamental_values['current_ratio'] = value
                elif name == "quick_ratio":
                    fundamental_values['quick_ratio'] = value
                elif name == "dividend_yield":
                    fundamental_values['dividend_yield'] = value
                elif name == "market_cap":
                    fundamental_values['market_cap'] = value
                elif name == "revenue_growth":
                    fundamental_values['revenue_growth'] = value
                elif name == "net_income_growth":
                    fundamental_values['net_income_growth'] = value
                elif name == "free_cash_flow":
                    fundamental_values['free_cash_flow'] = value
                elif name == "operating_margin":
                    fundamental_values['operating_margin'] = value
                elif name == "profit_margin":
                    fundamental_values['profit_margin'] = value

            # Create Fundamentals object with available data
            fundamentals = Fundamentals(
                ticker=ticker,
                company_name=ticker,  # Will be filled by yfinance fallback if needed
                current_price=None,
                market_cap=fundamental_values.get('market_cap'),
                pe_ratio=fundamental_values.get('pe_ratio'),
                forward_pe=None,
                dividend_yield=fundamental_values.get('dividend_yield'),
                earnings_per_share=None,
                price_to_book=fundamental_values.get('price_to_book'),
                return_on_equity=fundamental_values.get('return_on_equity'),
                return_on_assets=fundamental_values.get('return_on_assets'),
                debt_to_equity=fundamental_values.get('debt_to_equity'),
                current_ratio=fundamental_values.get('current_ratio'),
                quick_ratio=fundamental_values.get('quick_ratio'),
                revenue=None,
                revenue_growth=fundamental_values.get('revenue_growth'),
                net_income=None,
                net_income_growth=fundamental_values.get('net_income_growth'),
                free_cash_flow=fundamental_values.get('free_cash_flow'),
                operating_margin=fundamental_values.get('operating_margin'),
                profit_margin=fundamental_values.get('profit_margin'),
                beta=None,
                sector=None,
                industry=None,
                country=None,
                exchange=None,
                currency=None,
                shares_outstanding=None,
                float_shares=None,
                short_ratio=None,
                payout_ratio=None,
                peg_ratio=None,
                price_to_sales=fundamental_values.get('price_to_sales'),
                enterprise_value=None,
                enterprise_value_to_ebitda=None,
                data_source='IndicatorService',
                last_updated=datetime.now().isoformat()
            )

            return fundamentals

        except Exception:
            _logger.exception("Error converting indicator result to fundamentals for %s:", ticker)
            return None

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

            # Create Fundamentals object
            fundamentals = Fundamentals(
                ticker=ticker,
                company_name=info.get('longName', ticker),
                current_price=info.get('currentPrice'),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                forward_pe=info.get('forwardPE'),
                dividend_yield=info.get('dividendYield'),
                earnings_per_share=info.get('trailingEps'),
                price_to_book=info.get('priceToBook'),
                return_on_equity=info.get('returnOnEquity'),
                return_on_assets=info.get('returnOnAssets'),
                debt_to_equity=info.get('debtToEquity'),
                current_ratio=info.get('currentRatio'),
                quick_ratio=info.get('quickRatio'),
                revenue=info.get('totalRevenue'),
                revenue_growth=info.get('revenueGrowth'),
                net_income=info.get('netIncomeToCommon'),
                net_income_growth=info.get('earningsGrowth'),
                free_cash_flow=info.get('freeCashflow'),
                operating_margin=info.get('operatingMargins'),
                profit_margin=info.get('profitMargins'),
                beta=info.get('beta'),
                sector=info.get('sector'),
                industry=info.get('industry'),
                country=info.get('country'),
                exchange=info.get('exchange'),
                currency=info.get('currency'),
                shares_outstanding=info.get('sharesOutstanding'),
                float_shares=info.get('floatShares'),
                short_ratio=info.get('shortRatio'),
                payout_ratio=info.get('payoutRatio'),
                peg_ratio=info.get('pegRatio'),
                price_to_sales=info.get('priceToSalesTrailing12Months'),
                enterprise_value=info.get('enterpriseValue'),
                enterprise_value_to_ebitda=info.get('enterpriseToEbitda'),
                data_source='yfinance',
                last_updated=datetime.now().isoformat()
            )

            # Calculate additional metrics if financial statements are available
            if not financials.empty and not balance_sheet.empty and not cash_flow.empty:
                fundamentals = self._calculate_additional_metrics(fundamentals, financials, balance_sheet, cash_flow)

            return fundamentals

        except Exception:
            _logger.exception("Error getting fundamentals for %s:", ticker)
            return None

    def _calculate_additional_metrics(self, fundamentals: Fundamentals,
                                    financials: pd.DataFrame,
                                    balance_sheet: pd.DataFrame,
                                    cash_flow: pd.DataFrame) -> Fundamentals:
        """Calculate additional metrics from financial statements."""
        try:
            # Calculate revenue growth if not available
            if fundamentals.revenue_growth is None and not financials.empty:
                if 'Total Revenue' in financials.index:
                    revenue_data = financials.loc['Total Revenue']
                    if len(revenue_data) >= 2:
                        current_revenue = revenue_data.iloc[0]
                        previous_revenue = revenue_data.iloc[1]
                        if previous_revenue and previous_revenue > 0:
                            fundamentals.revenue_growth = (current_revenue - previous_revenue) / previous_revenue * 100

            # Calculate net income growth if not available
            if fundamentals.net_income_growth is None and not financials.empty:
                if 'Net Income' in financials.index:
                    net_income_data = financials.loc['Net Income']
                    if len(net_income_data) >= 2:
                        current_ni = net_income_data.iloc[0]
                        previous_ni = net_income_data.iloc[1]
                        if previous_ni and previous_ni > 0:
                            fundamentals.net_income_growth = (current_ni - previous_ni) / previous_ni * 100

            # Calculate free cash flow if not available
            if fundamentals.free_cash_flow is None and not cash_flow.empty:
                if 'Free Cash Flow' in cash_flow.index:
                    fcf_data = cash_flow.loc['Free Cash Flow']
                    if len(fcf_data) > 0:
                        fundamentals.free_cash_flow = fcf_data.iloc[0]

        except Exception as e:
            _logger.exception("Error calculating additional metrics: %s", e)

        return fundamentals

    def _validate_fundamental_data(self, fundamentals: Fundamentals) -> bool:
        """Validate that fundamental data has sufficient information for screening."""
        required_fields = [
            'current_price', 'market_cap', 'pe_ratio', 'price_to_book',
            'return_on_equity', 'debt_to_equity', 'current_ratio'
        ]

        # Check if at least 70% of required fields are present
        present_fields = sum(1 for field in required_fields
                           if getattr(fundamentals, field) is not None)

        return present_fields >= len(required_fields) * 0.7

    def apply_screening_criteria(self, fundamentals_data: Dict[str, Fundamentals]) -> List[ScreenerResult]:
        """Apply screening criteria to fundamental data and return top results."""
        screener_results = []

        _logger.info("Applying screening criteria to %d tickers", len(fundamentals_data))

        for ticker, fundamentals in fundamentals_data.items():
            try:
                # Apply screening criteria
                screening_status = self._apply_screening_criteria(fundamentals)

                # Calculate composite score
                composite_score = self._calculate_composite_score(fundamentals, screening_status)

                # Calculate DCF valuation
                dcf_result = self._calculate_dcf_valuation(fundamentals)

                # Generate recommendation
                recommendation, reasoning = self._generate_recommendation(composite_score, fundamentals, screening_status)

                # Create screener result
                result = ScreenerResult(
                    ticker=ticker,
                    fundamentals=fundamentals,
                    dcf_valuation=dcf_result,
                    composite_score=composite_score,
                    screening_status=screening_status,
                    recommendation=recommendation,
                    reasoning=reasoning
                )

                screener_results.append(result)

            except Exception as e:
                _logger.exception("Error screening %s: %s", ticker, e)
                result = ScreenerResult(
                    ticker=ticker,
                    error=str(e)
                )
                screener_results.append(result)

        # Sort by composite score (descending) and return top results
        valid_results = [r for r in screener_results if r.composite_score is not None]
        valid_results.sort(key=lambda x: x.composite_score, reverse=True)

        return valid_results[:self.max_results]

    def _apply_screening_criteria(self, fundamentals: Fundamentals) -> Dict[str, bool]:
        """Apply individual screening criteria to fundamentals."""
        screening_status = {}

        for criterion, config in self.screening_thresholds.items():
            value = getattr(fundamentals, criterion, None)

            if value is None:
                screening_status[criterion] = False
                continue

            if 'max' in config:
                screening_status[criterion] = value <= config['max']
            elif 'min' in config:
                screening_status[criterion] = value >= config['min']
            else:
                screening_status[criterion] = True

        return screening_status

    def _calculate_composite_score(self, fundamentals: Fundamentals,
                                 screening_status: Dict[str, bool]) -> float:
        """Calculate composite score (0-10) based on screening criteria."""
        total_score = 0
        total_weight = 0

        for criterion, config in self.screening_thresholds.items():
            weight = config['weight']
            total_weight += weight

            if screening_status.get(criterion, False):
                total_score += weight

        # Normalize to 0-10 scale
        if total_weight > 0:
            return (total_score / total_weight) * 10
        else:
            return 0

    def _calculate_dcf_valuation(self, fundamentals: Fundamentals) -> DCFResult:
        """Calculate DCF valuation for a ticker."""
        try:
            # Get required data
            current_price = fundamentals.current_price
            free_cash_flow = fundamentals.free_cash_flow
            beta = fundamentals.beta or 1.0
            revenue_growth = fundamentals.revenue_growth or 0.05
            net_income_growth = fundamentals.net_income_growth or 0.05

            if not current_price or not free_cash_flow or free_cash_flow <= 0:
                return DCFResult(
                    ticker=fundamentals.ticker,
                    error="Insufficient data for DCF calculation"
                )

            # Calculate discount rate (CAPM)
            market_risk_premium = 0.06  # 6% market risk premium
            discount_rate = self.risk_free_rate + (beta * market_risk_premium)

            # Estimate growth rate (average of revenue and net income growth)
            growth_rate = (revenue_growth + net_income_growth) / 2 / 100  # Convert to decimal

            # Cap growth rate at reasonable levels
            growth_rate = min(growth_rate, 0.15)  # Max 15% growth
            growth_rate = max(growth_rate, 0.02)  # Min 2% growth

            # Calculate terminal value
            terminal_growth_rate = 0.02  # 2% terminal growth
            terminal_value = free_cash_flow * (1 + growth_rate) / (discount_rate - terminal_growth_rate)

            # Calculate present value of future cash flows (5 years)
            present_value = 0
            for year in range(1, 6):
                future_fcf = free_cash_flow * (1 + growth_rate) ** year
                present_value += future_fcf / (1 + discount_rate) ** year

            # Add terminal value
            terminal_pv = terminal_value / (1 + discount_rate) ** 5
            total_present_value = present_value + terminal_pv

            # Calculate fair value per share
            shares_outstanding = fundamentals.shares_outstanding
            if shares_outstanding:
                fair_value = total_present_value / shares_outstanding
            else:
                fair_value = total_present_value

            # Determine confidence level
            confidence_level = self._assess_dcf_confidence(fundamentals)

            return DCFResult(
                ticker=fundamentals.ticker,
                fair_value=fair_value,
                growth_rate=growth_rate,
                discount_rate=discount_rate,
                terminal_value=terminal_value,
                assumptions={
                    'risk_free_rate': self.risk_free_rate,
                    'market_risk_premium': market_risk_premium,
                    'terminal_growth_rate': terminal_growth_rate,
                    'forecast_period': 5
                },
                confidence_level=confidence_level
            )

        except Exception:
            _logger.exception("Error calculating DCF for %s", fundamentals.ticker)
            return None

    def _assess_dcf_confidence(self, fundamentals: Fundamentals) -> str:
        """Assess confidence level of DCF calculation."""
        confidence_score = 0

        # Check data quality
        if fundamentals.free_cash_flow and fundamentals.free_cash_flow > 0:
            confidence_score += 1
        if fundamentals.revenue_growth is not None:
            confidence_score += 1
        if fundamentals.net_income_growth is not None:
            confidence_score += 1
        if fundamentals.beta is not None:
            confidence_score += 1
        if fundamentals.shares_outstanding is not None:
            confidence_score += 1

        if confidence_score >= 4:
            return "HIGH"
        elif confidence_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_recommendation(self, composite_score: float,
                               fundamentals: Fundamentals,
                               screening_status: Dict[str, bool]) -> Tuple[str, str]:
        """Generate buy/sell/hold recommendation with reasoning."""
        if composite_score >= 7.0:
            recommendation = "BUY"
            reasoning = "Strong undervaluation signals with good financial health and profitability metrics."
        elif composite_score >= 5.0:
            recommendation = "HOLD"
            reasoning = "Fair valuation with mixed signals. Monitor for improvement in key metrics."
        else:
            recommendation = "SELL"
            reasoning = "Poor valuation metrics and financial health indicators."

        # Add specific reasoning based on screening results
        failed_criteria = [k for k, v in screening_status.items() if not v]
        if failed_criteria:
            reasoning += f" Failed criteria: {', '.join(failed_criteria[:3])}."

        return recommendation, reasoning

    def generate_report(self, list_type: str, results: List[ScreenerResult],
                       total_processed: int) -> ScreenerReport:
        """Generate comprehensive screener report."""
        try:
            # Calculate summary statistics
            summary_stats = self._calculate_summary_stats(results)

            # Create report
            report = ScreenerReport(
                list_type=list_type,
                total_tickers_processed=total_processed,
                total_tickers_with_data=len(results),
                top_results=results,
                summary_stats=summary_stats,
                generated_at=datetime.now().isoformat()
            )

            return report

        except Exception as e:
            _logger.exception("Error generating report: %s", e)
            return ScreenerReport(
                list_type=list_type,
                total_tickers_processed=0,
                total_tickers_with_data=0,
                top_results=[],
                error=f"Error generating report: {e}"
            )

    def _calculate_summary_stats(self, results: List[ScreenerResult]) -> Dict[str, Any]:
        """Calculate summary statistics for the screener results."""
        if not results:
            return {}

        # Calculate averages
        avg_score = np.mean([r.composite_score for r in results if r.composite_score])
        avg_pe = np.mean([r.fundamentals.pe_ratio for r in results
                         if r.fundamentals and r.fundamentals.pe_ratio])
        avg_pb = np.mean([r.fundamentals.price_to_book for r in results
                         if r.fundamentals and r.fundamentals.price_to_book])
        avg_roe = np.mean([r.fundamentals.return_on_equity for r in results
                          if r.fundamentals and r.fundamentals.return_on_equity])

        # Count recommendations
        buy_count = sum(1 for r in results if r.recommendation == "BUY")
        hold_count = sum(1 for r in results if r.recommendation == "HOLD")
        sell_count = sum(1 for r in results if r.recommendation == "SELL")

        # Sector distribution
        sectors = {}
        for r in results:
            if r.fundamentals and r.fundamentals.sector:
                sector = r.fundamentals.sector
                sectors[sector] = sectors.get(sector, 0) + 1

        return {
            'average_composite_score': round(avg_score, 2) if avg_score else None,
            'average_pe_ratio': round(avg_pe, 2) if avg_pe else None,
            'average_price_to_book': round(avg_pb, 2) if avg_pb else None,
            'average_roe': round(avg_roe, 2) if avg_roe else None,
            'recommendations': {
                'buy': buy_count,
                'hold': hold_count,
                'sell': sell_count
            },
            'sector_distribution': sectors
        }

    def format_telegram_message(self, report: ScreenerReport) -> str:
        """Format screener report for Telegram message."""
        if report.error:
            return f"❌ Screener Error: {report.error}"

        if not report.top_results:
            return "📊 No undervalued stocks found in the screening criteria."

        # Header
        message = "📊 **Fundamental Screener Report**\n"
        message += f"🔍 **List Type**: {report.list_type.replace('_', ' ').title()}\n"
        message += f"📈 **Processed**: {report.total_tickers_processed} tickers\n"
        message += f"✅ **With Data**: {report.total_tickers_with_data} tickers\n"
        message += f"🏆 **Top Results**: {len(report.top_results)} undervalued stocks\n"
        message += f"⏰ **Generated**: {datetime.fromisoformat(report.generated_at).strftime('%Y-%m-%d %H:%M UTC')}\n\n"

        # Summary statistics
        if report.summary_stats:
            stats = report.summary_stats
            message += "📊 **Summary Statistics**\n"
            if stats.get('average_composite_score'):
                message += f"• Average Score: {stats['average_composite_score']}/10\n"
            if stats.get('average_pe_ratio'):
                message += f"• Average P/E: {stats['average_pe_ratio']}\n"
            if stats.get('average_price_to_book'):
                message += f"• Average P/B: {stats['average_price_to_book']}\n"
            if stats.get('average_roe'):
                message += f"• Average ROE: {stats['average_roe']}%\n"

            recs = stats.get('recommendations', {})
            message += f"• Recommendations: {recs.get('buy', 0)} BUY, {recs.get('hold', 0)} HOLD, {recs.get('sell', 0)} SELL\n\n"

        # Top results summary
        message += "🏆 **Top Undervalued Stocks**\n"
        for i, result in enumerate(report.top_results, 1):
            if result.error:
                message += f"{i}. {result.ticker}: ❌ Error - {result.error}\n"
                continue

            fundamentals = result.fundamentals
            score = result.composite_score
            recommendation = result.recommendation

            # Emoji for recommendation
            rec_emoji = "🟢" if recommendation == "BUY" else "🟡" if recommendation == "HOLD" else "🔴"

            message += f"{i}. {rec_emoji} **{result.ticker}** - {fundamentals.company_name}\n"
            message += f"   💯 Score: {score:.1f}/10 | {recommendation}\n"
            message += f"   💰 Price: ${fundamentals.current_price:.2f}\n"

            if fundamentals.pe_ratio:
                message += f"   📊 P/E: {fundamentals.pe_ratio:.1f}"
            if fundamentals.price_to_book:
                message += f" | P/B: {fundamentals.price_to_book:.2f}"
            if fundamentals.return_on_equity:
                message += f" | ROE: {fundamentals.return_on_equity:.1f}%"
            message += "\n"

            if fundamentals.sector:
                message += f"   🏭 Sector: {fundamentals.sector}\n"

            # DCF valuation if available
            if result.dcf_valuation and result.dcf_valuation.fair_value:
                dcf = result.dcf_valuation
                current_price = fundamentals.current_price
                upside = ((dcf.fair_value - current_price) / current_price) * 100
                message += f"   💎 DCF Fair Value: ${dcf.fair_value:.2f} ({upside:+.1f}%)\n"

            message += "\n"

        return message

    async def run_screener(self, list_type: str, max_results: int = 10, min_score: float = 7.0) -> ScreenerReport:
        """Run the fundamental screener for the specified list type."""
        _logger.info("Starting fundamental screener for %s", list_type)

        try:
            # Load ticker list
            tickers = self.load_ticker_list(list_type)
            if not tickers:
                return ScreenerReport(
                    list_type=list_type,
                    total_tickers_processed=0,
                    total_tickers_with_data=0,
                    top_results=[],
                    error="No tickers found for screening"
                )

            # Collect fundamental data
            fundamentals_data = await self.collect_fundamentals(tickers)

            # Apply screening criteria
            results = self.apply_screening_criteria(fundamentals_data)

            # Generate report
            report = self.generate_report(list_type, results, len(tickers))

            _logger.info("Screener completed successfully. Found %d undervalued stocks", len(results))
            return report

        except Exception as e:
            _logger.exception("Error running screener")
            return ScreenerReport(
                list_type=list_type,
                total_tickers_processed=0,
                total_tickers_with_data=0,
                top_results=[],
                error=f"Error running screener: {e}"
            )


# Global screener instance
screener = FundamentalScreener(indicator_service=IndicatorService())

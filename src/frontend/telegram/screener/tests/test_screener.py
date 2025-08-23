"""
Test script for the fundamental and enhanced screener functionality.
This script can be run independently to test the screeners without the full bot.
"""

import json
import time
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

# Import the enhanced screener and related modules
from src.frontend.telegram.screener.enhanced_screener import EnhancedScreener
from src.frontend.telegram.screener.screener_config_parser import (
    parse_screener_config,
    validate_screener_config
)
from src.notification.logger import setup_logger

logger = setup_logger(__name__)

# Setup logging
def test_comprehensive_fmp_analysis():
    """
    Basic FMP Analysis Test - Executes FMP query, selects top X undervalued shares,
    and provides detailed fundamental/technical indicators with recommendations.
    """
    print("\n🎯 BASIC FMP ANALYSIS TEST")
    print("=" * 70)

    # Configuration for mid-cap stocks ($200M - $2B market cap)
    config = {
        "screener_type": "hybrid",
        "list_type": "us_medium_cap",
        "fmp_criteria": {
            "marketCapMoreThan": 200000000,   # $200M+ market cap
            "marketCapLowerThan": 2000000000, # $2B- market cap
            "peRatioLessThan": 20,            # P/E < 20
            "returnOnEquityMoreThan": 0.10,   # ROE > 10%
            "isEtf": False,                   # Exclude ETFs
            "isFund": False,                  # Exclude mutual funds
            "limit": 20                        # Get up to 20 stocks from FMP
        },
        "fundamental_criteria": [
            # Only YFinance-only criteria that cannot be filtered by FMP
            {
                "indicator": "Revenue_Growth",
                "operator": "min",
                "value": 0.05,       # Revenue growth > 5% (0.05 in decimal)
                "weight": 1.0,
                "required": False   # Not required since FMP already filtered
            },
            {
                "indicator": "Operating_Margin",
                "operator": "min",
                "value": 0.10,       # Operating margin > 10% (0.10 in decimal)
                "weight": 1.0,
                "required": False   # Not required since FMP already filtered
            }
        ],
        "technical_criteria": [
            {
                "indicator": "RSI",
                "parameters": {"period": 14},
                "condition": {"operator": "<", "value": 70},
                "weight": 0.6,
                "required": False
            },
            {
                "indicator": "MACD",
                "parameters": {"fast": 12, "slow": 26, "signal": 9},
                "condition": {"operator": "above_signal"},
                "weight": 0.5,
                "required": False
            }
        ],
        "max_results": 5,   # Show top 5 results
        "min_score": 2.0    # Lower minimum score to see results
    }

    try:
        print("🔧 Configuration:")
        print(f"   FMP Criteria: {len(config['fmp_criteria'])} criteria")
        print(f"   Fundamental Criteria: {len(config['fundamental_criteria'])} indicators")
        print(f"   Technical Criteria: {len(config['technical_criteria'])} indicators")
        print(f"   Max Results: {config['max_results']}")
        print(f"   Min Score: {config['min_score']}/10")

        # Validate and parse configuration
        config_json = json.dumps(config)
        is_valid, errors = validate_screener_config(config_json)

        if not is_valid:
            print(f"❌ Configuration validation failed: {errors}")
            return

        screener_config = parse_screener_config(config_json)
        print("✅ Configuration validated and parsed")

        # Run the enhanced screener with FMP integration
        print("\n🚀 Executing FMP Query and Analysis...")
        start_time = time.time()

        enhanced_screener = EnhancedScreener()
        report = enhanced_screener.run_enhanced_screener(screener_config)

        total_time = time.time() - start_time
        print(f"✅ Analysis completed in {total_time:.2f} seconds")

        if report.error:
            print(f"❌ Analysis error: {report.error}")
            return

        print(f"✅ Analysis completed successfully!")
        print(f"   FMP Pre-filtered: {len(report.fmp_results.get('fmp_results', [])) if hasattr(report, 'fmp_results') and report.fmp_results else 'N/A'} stocks")
        print(f"   Processed: {report.total_tickers_processed} tickers")
        print(f"   Found: {len(report.top_results)} matching stocks")

        # Debug: Show why no stocks were found
        if len(report.top_results) == 0:
            print("\n🔍 DEBUG: Why no stocks found?")
            print("   - Check if FMP returned any stocks")
            print("   - Check if fundamental criteria are too strict (PE < 15, ROE > 12%)")
            print("   - Check if technical criteria are too strict (RSI < 70, MACD above signal)")
            print("   - Check if min_score (6.0) is too high")
            print("   - Consider relaxing criteria for mid-cap stocks")

        if not report.top_results:
            print("❌ No stocks found matching criteria")
            return

        # Display results
        print("\n" + "=" * 70)
        print("📊 ANALYSIS RESULTS")
        print("=" * 70)

        for i, result in enumerate(report.top_results, 1):
            print(f"\n🏆 #{i}: {result.ticker}")
            print("-" * 50)

            # Overall Score
            print(f"📈 Overall Score: {result.composite_score:.1f}/10")
            print(f"   Recommendation: {result.recommendation}")

            if result.fundamentals and result.fundamentals.current_price:
                print(f"   Current Price: ${result.fundamentals.current_price:.2f}")

            # Fundamental Analysis
            if result.fundamentals:
                print(f"\n📊 Fundamental Analysis:")
                if result.fundamentals.pe_ratio:
                    print(f"   • PE Ratio: {result.fundamentals.pe_ratio:.2f}")
                if result.fundamentals.return_on_equity:
                    print(f"   • ROE: {result.fundamentals.return_on_equity:.2%}")
                if result.fundamentals.revenue_growth:
                    print(f"   • Revenue Growth: {result.fundamentals.revenue_growth:.2%}")
                if result.fundamentals.operating_margin:
                    print(f"   • Operating Margin: {result.fundamentals.operating_margin:.2%}")
                if result.fundamentals.market_cap:
                    print(f"   • Market Cap: ${result.fundamentals.market_cap:,.0f}")

            # Technical Analysis
            if result.technicals:
                print(f"\n📈 Technical Analysis:")
                print(f"   • RSI: {result.technicals.rsi:.2f}")
                print(f"   • MACD: {result.technicals.macd:.4f}")
                print(f"   • MACD Signal: {result.technicals.macd_signal:.4f}")
                print(f"   • SMA 50: {result.technicals.sma_50:.2f}")
                print(f"   • SMA 200: {result.technicals.sma_200:.2f}")
                print(f"   • Bollinger Upper: {result.technicals.bb_upper:.2f}")
                print(f"   • Bollinger Lower: {result.technicals.bb_lower:.2f}")
                print(f"   • ADX: {result.technicals.adx:.2f}")
                print(f"   • Stochastic K: {result.technicals.stoch_k:.2f}")
                print(f"   • Stochastic D: {result.technicals.stoch_d:.2f}")

                # Technical recommendations if available
                if result.technicals.recommendations:
                    print(f"\n   📋 Technical Recommendations:")
                    for indicator, rec in result.technicals.recommendations.items():
                        if isinstance(rec, dict) and 'signal' in rec:
                            signal = rec['signal']
                            reason = rec.get('reason', 'No reason provided')
                            print(f"   • {indicator.upper()}: {signal} - {reason}")
            else:
                print(f"\n📈 Technical Analysis: No technical data available")

            # DCF Analysis
            if result.dcf_valuation and result.dcf_valuation.fair_value:
                dcf = result.dcf_valuation
                if result.fundamentals and result.fundamentals.current_price:
                    current_price = result.fundamentals.current_price
                    fair_value = dcf.fair_value
                    upside = ((fair_value - current_price) / current_price) * 100
                    print(f"\n💰 DCF Analysis:")
                    print(f"   • Fair Value: ${fair_value:.2f}")
                    print(f"   • Current Price: ${current_price:.2f}")
                    print(f"   • Upside Potential: {upside:+.1f}%")

            print("\n" + "-" * 50)

        # Summary Statistics
        print(f"\n📋 SUMMARY")
        print("-" * 30)

        if report.top_results:
            scores = [r.composite_score for r in report.top_results]
            recommendations = [r.recommendation for r in report.top_results]

            # Count all recommendation types
            strong_buy_count = recommendations.count('STRONG_BUY')
            buy_count = recommendations.count('BUY')
            hold_count = recommendations.count('HOLD')
            weak_hold_count = recommendations.count('WEAK_HOLD')
            sell_count = recommendations.count('SELL')

            print(f"   Average Score: {sum(scores)/len(scores):.1f}/10")
            print(f"   Recommendations:")
            if strong_buy_count > 0:
                print(f"     • STRONG_BUY: {strong_buy_count}")
            if buy_count > 0:
                print(f"     • BUY: {buy_count}")
            if hold_count > 0:
                print(f"     • HOLD: {hold_count}")
            if weak_hold_count > 0:
                print(f"     • WEAK_HOLD: {weak_hold_count}")
            if sell_count > 0:
                print(f"     • SELL: {sell_count}")

            # Also show total buy signals (STRONG_BUY + BUY)
            total_buy_signals = strong_buy_count + buy_count
            if total_buy_signals > 0:
                print(f"   Total Buy Signals: {total_buy_signals}")

        print(f"\n✅ Basic FMP Analysis completed successfully!")

    except Exception as e:
        print(f"❌ Error during FMP analysis: {e}")
        logger.exception("FMP analysis test failed")


if __name__ == "__main__":
    print("🚀 Starting Basic FMP Analysis Test")
    print("=" * 70)

    # Run only the basic FMP analysis test
    test_comprehensive_fmp_analysis()

    print("\n🎉 Test completed!")


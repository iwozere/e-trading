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

    # Simple configuration for basic analysis
    config = {
        "screener_type": "hybrid",
        "list_type": "us_medium_cap",
        "fmp_criteria": {
            "marketCapMoreThan": 2000000000,  # $2B+ market cap
            "peRatioLessThan": 20,            # P/E < 20
            "returnOnEquityMoreThan": 0.12,   # ROE > 12%
            "limit": 20                        # Get up to 20 stocks from FMP
        },
        "fundamental_criteria": [
            {
                "indicator": "PE",
                "operator": "max",
                "value": 15,
                "weight": 1.0,
                "required": True
            },
            {
                "indicator": "ROE",
                "operator": "min",
                "value": 15,
                "weight": 1.0,
                "required": True
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
        "min_score": 6.0    # Minimum composite score
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
            print(f"   Fundamental Score: {result.fundamental_score:.1f}/10")
            print(f"   Technical Score: {result.technical_score:.1f}/10")
            print(f"   Recommendation: {result.recommendation}")

            if result.current_price:
                print(f"   Current Price: ${result.current_price:.2f}")

            # Fundamental Analysis
            if hasattr(result, 'fundamental_analysis') and result.fundamental_analysis:
                print(f"\n📊 Fundamental Analysis:")
                for indicator, analysis in result.fundamental_analysis.items():
                    if isinstance(analysis, dict) and 'value' in analysis and 'recommendation' in analysis:
                        value = analysis['value']
                        recommendation = analysis['recommendation']
                        print(f"   • {indicator}: {value:.2f} - {recommendation}")

            # Technical Analysis
            if hasattr(result, 'technical_analysis') and result.technical_analysis:
                print(f"\n📈 Technical Analysis:")
                for indicator, analysis in result.technical_analysis.items():
                    if isinstance(analysis, dict) and 'value' in analysis and 'recommendation' in analysis:
                        value = analysis['value']
                        recommendation = analysis['recommendation']
                        print(f"   • {indicator}: {value:.2f} - {recommendation}")

            # DCF Analysis
            if hasattr(result, 'dcf_analysis') and result.dcf_analysis:
                dcf = result.dcf_analysis
                if 'fair_value' in dcf and 'current_price' in dcf:
                    fair_value = dcf['fair_value']
                    current_price = dcf['current_price']
                    if current_price > 0:
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
            buy_count = recommendations.count('BUY')
            hold_count = recommendations.count('HOLD')
            sell_count = recommendations.count('SELL')

            print(f"   Average Score: {sum(scores)/len(scores):.1f}/10")
            print(f"   Recommendations: {buy_count} BUY, {hold_count} HOLD, {sell_count} SELL")

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


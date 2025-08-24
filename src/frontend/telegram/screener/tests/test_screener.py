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

def test_mid_cap_screener():
    """Screen 100 mid-cap tickers ($200M - $2B market cap)"""
    print("\n🎯 MID-CAP SCREENER (200M - 2B)")
    print("=" * 70)

    config = {
        "screener_type": "hybrid",
        "list_type": "us_medium_cap",
        "fmp_criteria": {
            "marketCapMoreThan": 200000000,   # $200M+ market cap
            "marketCapLowerThan": 2000000000, # $2B- market cap
            "peRatioLessThan": 25,            # P/E < 25
            "returnOnEquityMoreThan": 0.08,   # ROE > 8%
            "isEtf": False,                   # Exclude ETFs
            "isActivelyTrading": True,        # Only actively trading stocks
            "limit": 50                        # Get up to 50 stocks from FMP
        },
        "fundamental_criteria": [
            {
                "indicator": "PE",
                "operator": "max",
                "value": 25,         # P/E < 25
                "weight": 1.0,
                "required": False
            },
            {
                "indicator": "PB",
                "operator": "max",
                "value": 3.0,        # P/B < 3.0
                "weight": 1.0,
                "required": False
            },
            {
                "indicator": "ROE",
                "operator": "min",
                "value": 0.10,       # ROE > 10%
                "weight": 1.0,
                "required": False
            },
            {
                "indicator": "Revenue_Growth",
                "operator": "min",
                "value": 0.03,       # Revenue growth > 3%
                "weight": 0.5,
                "required": False
            },
            {
                "indicator": "Operating_Margin",
                "operator": "min",
                "value": 0.05,       # Operating margin > 5%
                "weight": 0.5,
                "required": False
            }
        ],
        "technical_criteria": [
            {
                "indicator": "RSI",
                "parameters": {"period": 14},
                "condition": {"operator": "<", "value": 80},
                "weight": 0.5,
                "required": False
            }
        ],
        "max_results": 50,
        "min_score": 0.5
    }

    return run_screener_test("Mid-Cap", config)

def test_large_cap_screener():
    """Screen 100 large-cap tickers ($2B - $200B market cap)"""
    print("\n🎯 LARGE-CAP SCREENER (2B - 200B)")
    print("=" * 70)

    config = {
        "screener_type": "hybrid",
        "list_type": "us_large_cap",
        "fmp_criteria": {
            "marketCapMoreThan": 2000000000,   # $2B+ market cap
            "marketCapLowerThan": 200000000000, # $200B- market cap
            "peRatioLessThan": 30,             # P/E < 30
            "returnOnEquityMoreThan": 0.10,    # ROE > 10%
            "isEtf": False,                    # Exclude ETFs
            "isActivelyTrading": True,         # Only actively trading stocks
            "limit": 50
        },
        "fundamental_criteria": [
            {
                "indicator": "PE",
                "operator": "max",
                "value": 30,         # P/E < 30
                "weight": 1.0,
                "required": False
            },
            {
                "indicator": "PB",
                "operator": "max",
                "value": 4.0,        # P/B < 4.0
                "weight": 1.0,
                "required": False
            },
            {
                "indicator": "ROE",
                "operator": "min",
                "value": 0.12,       # ROE > 12%
                "weight": 1.0,
                "required": False
            },
            {
                "indicator": "Revenue_Growth",
                "operator": "min",
                "value": 0.05,       # Revenue growth > 5%
                "weight": 0.5,
                "required": False
            },
            {
                "indicator": "Operating_Margin",
                "operator": "min",
                "value": 0.08,       # Operating margin > 8%
                "weight": 0.5,
                "required": False
            }
        ],
        "technical_criteria": [
            {
                "indicator": "RSI",
                "parameters": {"period": 14},
                "condition": {"operator": "<", "value": 75},
                "weight": 0.5,
                "required": False
            }
        ],
        "max_results": 50,
        "min_score": 0.5
    }

    return run_screener_test("Large-Cap", config)

def test_super_large_cap_screener():
    """Screen 100 super large-cap tickers ($200B+ market cap)"""
    print("\n🎯 SUPER LARGE-CAP SCREENER (200B+)")
    print("=" * 70)

    config = {
        "screener_type": "hybrid",
        "list_type": "us_large_cap",
        "fmp_criteria": {
            "marketCapMoreThan": 200000000000, # $200B+ market cap
            "peRatioLessThan": 35,             # P/E < 35
            "returnOnEquityMoreThan": 0.12,    # ROE > 12%
            "isEtf": False,                    # Exclude ETFs
            "isActivelyTrading": True,         # Only actively trading stocks
            "limit": 50
        },
        "fundamental_criteria": [
            {
                "indicator": "PE",
                "operator": "max",
                "value": 35,         # P/E < 35
                "weight": 1.0,
                "required": False
            },
            {
                "indicator": "PB",
                "operator": "max",
                "value": 5.0,        # P/B < 5.0
                "weight": 1.0,
                "required": False
            },
            {
                "indicator": "ROE",
                "operator": "min",
                "value": 0.15,       # ROE > 15%
                "weight": 1.0,
                "required": False
            },
            {
                "indicator": "Revenue_Growth",
                "operator": "min",
                "value": 0.08,       # Revenue growth > 8%
                "weight": 0.5,
                "required": False
            },
            {
                "indicator": "Operating_Margin",
                "operator": "min",
                "value": 0.12,       # Operating margin > 12%
                "weight": 0.5,
                "required": False
            }
        ],
        "technical_criteria": [
            {
                "indicator": "RSI",
                "parameters": {"period": 14},
                "condition": {"operator": "<", "value": 70},
                "weight": 0.5,
                "required": False
            }
        ],
        "max_results": 50,
        "min_score": 0.5
    }

    return run_screener_test("Super Large-Cap", config)

def test_swiss_screener():
    """Screen 100 Swiss undervalued tickers"""
    print("\n🎯 SWISS SCREENER")
    print("=" * 70)

    config = {
        "screener_type": "hybrid",
        "list_type": "swiss_shares",
        "fmp_criteria": {
            "exchange": "SWX",       # Swiss Exchange
            "peRatioLessThan": 20,   # P/E < 20
            "returnOnEquityMoreThan": 0.08, # ROE > 8%
            "isEtf": False,          # Exclude ETFs
            "isActivelyTrading": True, # Only actively trading stocks
            "limit": 50
        },
        "fundamental_criteria": [
            {
                "indicator": "PE",
                "operator": "max",
                "value": 25,         # P/E < 25
                "weight": 1.0,
                "required": False
            },
            {
                "indicator": "PB",
                "operator": "max",
                "value": 3.0,        # P/B < 3.0
                "weight": 1.0,
                "required": False
            },
            {
                "indicator": "ROE",
                "operator": "min",
                "value": 0.10,       # ROE > 10%
                "weight": 1.0,
                "required": False
            },
            {
                "indicator": "Revenue_Growth",
                "operator": "min",
                "value": 0.02,       # Revenue growth > 2%
                "weight": 0.5,
                "required": False
            },
            {
                "indicator": "Operating_Margin",
                "operator": "min",
                "value": 0.05,       # Operating margin > 5%
                "weight": 0.5,
                "required": False
            }
        ],
        "technical_criteria": [
            {
                "indicator": "RSI",
                "parameters": {"period": 14},
                "condition": {"operator": "<", "value": 80},
                "weight": 0.5,
                "required": False
            }
        ],
        "max_results": 50,
        "min_score": 0.5
    }

    return run_screener_test("Swiss", config)

def run_screener_test(screener_name, config):
    """Run a screener test and return results"""
    try:
        print(f"🔧 {screener_name} Configuration:")
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
            return None

        screener_config = parse_screener_config(config_json)
        print("✅ Configuration validated and parsed")

        # Run the enhanced screener with FMP integration
        print(f"\n🚀 Executing {screener_name} FMP Query and Analysis...")
        start_time = time.time()

        enhanced_screener = EnhancedScreener()
        report = enhanced_screener.run_enhanced_screener(screener_config)

        total_time = time.time() - start_time
        print(f"✅ Analysis completed in {total_time:.2f} seconds")

        if report.error:
            print(f"❌ Analysis error: {report.error}")
            return None

        print(f"✅ {screener_name} Analysis completed successfully!")
        print(f"   FMP Pre-filtered: {len(report.fmp_results.get('fmp_results', [])) if hasattr(report, 'fmp_results') and report.fmp_results else 'N/A'} stocks")
        print(f"   Processed: {report.total_tickers_processed} tickers")
        print(f"   Found: {len(report.top_results)} matching stocks")

        if not report.top_results:
            print(f"❌ No {screener_name} stocks found matching criteria")
            return None

        # Display summary
        print(f"\n📋 {screener_name} SUMMARY")
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

            total_buy_signals = strong_buy_count + buy_count
            if total_buy_signals > 0:
                print(f"   Total Buy Signals: {total_buy_signals}")

        return report

    except Exception as e:
        print(f"❌ Error during {screener_name} analysis: {e}")
        logger.exception(f"{screener_name} analysis test failed")
        return None

def send_screener_email(screener_name, report):
    """Send screener results via email"""
    try:
        from src.notification.emailer import EmailNotifier
        from src.frontend.telegram import db

        # Get user email from database (assuming we have a default user or test user)
        # For testing purposes, we'll use a hardcoded email or get from environment
        import os
        user_email = os.getenv('TEST_EMAIL', 'akossyrev@gmail.com')

        # Try to get email from database if available
        try:
            # Get the first verified user from database
            users = db.get_all_users()
            for user in users:
                if user.get('verified') and user.get('email'):
                    user_email = user['email']
                    break
        except Exception:
            pass  # Use default email if database lookup fails

        # Create a mock config for the email function
        class MockConfig:
            def __init__(self, list_type):
                self.list_type = list_type

        # Use the same email function as the Telegram bot for consistency
        from src.frontend.telegram.screener.notifications import send_screener_email as bot_send_email
        mock_config = MockConfig(screener_name.lower().replace(' ', '_').replace('-', '_'))
        bot_send_email(user_email, report, mock_config)

        print(f"✅ {screener_name} results sent via email to {user_email}")

    except Exception as e:
        print(f"❌ Error sending {screener_name} email: {e}")
        logger.exception(f"Email sending failed for {screener_name}")

def run_all_screeners():
    """Run all four screeners and send results via email"""
    print("🚀 STARTING COMPREHENSIVE SCREENER ANALYSIS")
    print("=" * 70)
    print("This will screen:")
    print("• 50 Mid-Cap stocks ($200M - $2B)")
    print("• 50 Large-Cap stocks ($2B - $200B)")
    print("• 50 Super Large-Cap stocks ($200B+)")
    print("• 50 Swiss undervalued stocks")
    print("=" * 70)

    results = {}

    # Run all screeners
    screeners = [
        ("Mid-Cap", test_mid_cap_screener),
        ("Large-Cap", test_large_cap_screener),
        ("Super Large-Cap", test_super_large_cap_screener),
        ("Swiss", test_swiss_screener)
    ]

    for screener_name, screener_func in screeners:
        print(f"\n{'='*70}")
        print(f"🎯 RUNNING {screener_name.upper()} SCREENER")
        print(f"{'='*70}")

        report = screener_func()
        if report:
            results[screener_name] = report
            # Send email for each screener
            send_screener_email(screener_name, report)
        else:
            print(f"❌ {screener_name} screener failed or returned no results")

        # Add delay between screeners to avoid rate limiting
        time.sleep(2)

    # Final summary
    print(f"\n{'='*70}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*70}")

    total_stocks = 0
    for screener_name, report in results.items():
        stock_count = len(report.top_results) if report else 0
        total_stocks += stock_count
        print(f"   {screener_name}: {stock_count} stocks")

        print(f"\n   Total Stocks Found: {total_stocks}")
    print(f"   Screeners Completed: {len(results)}/4")
    print(f"   Email Reports Sent: {len(results)}")

    print(f"\n✅ Comprehensive screener analysis completed!")
    print("📧 Check your email for detailed results!")
    print("📊 Total potential stocks: 200 (50 per screener)")

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
            "isActivelyTrading": True,        # Only actively trading stocks
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
    print("🚀 Starting Comprehensive Screener Analysis")
    print("=" * 70)
    print("Choose your option:")
    print("1. Run basic FMP analysis test (5 stocks)")
    print("2. Run all screeners (200 stocks total) with email")
    print("3. Run individual screeners")
    print("=" * 70)

    try:
        choice = input("Enter your choice (1-3): ").strip()

        if choice == "1":
            print("\n🎯 Running Basic FMP Analysis Test")
            test_comprehensive_fmp_analysis()
        elif choice == "2":
            print("\n🎯 Running All Screeners with Email")
            run_all_screeners()
        elif choice == "3":
            print("\n🎯 Running Individual Screeners")
            print("1. Mid-Cap Screener")
            print("2. Large-Cap Screener")
            print("3. Super Large-Cap Screener")
            print("4. Swiss Screener")

            sub_choice = input("Enter your choice (1-4): ").strip()

            if sub_choice == "1":
                report = test_mid_cap_screener()
                if report:
                    send_screener_email("Mid-Cap", report)
            elif sub_choice == "2":
                report = test_large_cap_screener()
                if report:
                    send_screener_email("Large-Cap", report)
            elif sub_choice == "3":
                report = test_super_large_cap_screener()
                if report:
                    send_screener_email("Super Large-Cap", report)
            elif sub_choice == "4":
                report = test_swiss_screener()
                if report:
                    send_screener_email("Swiss", report)
            else:
                print("❌ Invalid choice")
        else:
            print("❌ Invalid choice")

    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    print("\n�� Test completed!")


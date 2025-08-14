#!/usr/bin/env python3
"""
Simple test to check multiple ticker processing.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[0]
sys.path.append(str(project_root))

def test_command_parsing():
    """Test command parsing for multiple tickers."""
    try:
        from src.frontend.telegram.command_parser import parse_command

        # Test command
        test_command = "/report AAPL MSFT VT"
        print(f"Testing command: {test_command}")

        # Parse the command
        parsed = parse_command(test_command)
        print(f"Parsed command: {parsed}")
        print(f"Parsed args: {parsed.args}")
        print(f"Parsed positionals: {parsed.positionals}")

        # Check if tickers are properly extracted
        tickers = parsed.args.get("tickers")
        print(f"Tickers from args: {tickers}")
        print(f"Type of tickers: {type(tickers)}")

        if isinstance(tickers, list):
            print(f"Number of tickers: {len(tickers)}")
            for i, ticker in enumerate(tickers):
                print(f"  Ticker {i+1}: {ticker}")
        else:
            print("Tickers is not a list!")

    except Exception as e:
        print(f"Error in command parsing: {e}")
        import traceback
        traceback.print_exc()

def test_business_logic():
    """Test business logic for multiple tickers."""
    try:
        from src.frontend.telegram.command_parser import parse_command
        from src.frontend.telegram.screener.business_logic import handle_command

        # Test command
        test_command = "/report AAPL MSFT VT"
        print(f"\nTesting business logic for: {test_command}")

        # Parse the command
        parsed = parse_command(test_command)
        parsed.args["telegram_user_id"] = "test_user_123"

        # Handle the command
        result = handle_command(parsed)

        print(f"Result status: {result.get('status')}")
        print(f"Result title: {result.get('title')}")

        if "reports" in result:
            reports = result["reports"]
            print(f"Number of reports: {len(reports)}")
            for i, report in enumerate(reports):
                print(f"Report {i+1}:")
                print(f"  Ticker: {report.get('ticker')}")
                print(f"  Error: {report.get('error')}")
                print(f"  Message length: {len(report.get('message', ''))}")
        else:
            print("No reports in result")
            print(f"Result message: {result.get('message')}")

    except Exception as e:
        print(f"Error in business logic: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== Testing Command Parsing ===")
    test_command_parsing()

    print("\n=== Testing Business Logic ===")
    test_business_logic()

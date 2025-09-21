#!/usr/bin/env python3
"""
Test script for the updated /schedules add_json functionality.
Tests both single ticker and multiple ticker report configurations.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.telegram.command_parser import parse_command

def test_add_json_schedules():
    """Test that the /schedules add_json command is parsed correctly for both single and multiple tickers."""
    print("Testing /schedules add_json command parsing...")

    # Test single ticker configuration
    result1 = parse_command('/schedules add_json {"type":"report","ticker":"AAPL","scheduled_time":"09:00","period":"1y","interval":"1d","email":true}')
    print(f"Single ticker command: {result1.command}")
    print(f"Action: {result1.positionals[0] if result1.positionals else 'None'}")
    print(f"Config: {result1.positionals[1] if len(result1.positionals) > 1 else 'None'}")
    print(f"Args: {result1.args}")

    # Test multiple ticker configuration
    result2 = parse_command('/schedules add_json {"type":"report","tickers":["AAPL","MSFT","GOOGL"],"scheduled_time":"09:00","period":"1y","interval":"1d","indicators":"RSI,MACD","email":true}')
    print(f"\nMultiple ticker command: {result2.command}")
    print(f"Action: {result2.positionals[0] if result2.positionals else 'None'}")
    print(f"Config: {result2.positionals[1] if len(result2.positionals) > 1 else 'None'}")
    print(f"Args: {result2.args}")

    # Test enhanced screener configuration
    result3 = parse_command('/schedules add_json {"type":"enhanced_screener","screener_type":"hybrid","list_type":"us_medium_cap","fmp_criteria":{"marketCapMoreThan":2000000000,"peRatioLessThan":20},"max_results":10,"min_score":7.0}')
    print(f"\nEnhanced screener command: {result3.command}")
    print(f"Action: {result3.positionals[0] if result3.positionals else 'None'}")
    print(f"Config: {result3.positionals[1] if len(result3.positionals) > 1 else 'None'}")
    print(f"Args: {result3.args}")

    print("\n✅ Add JSON schedules command parsing test completed!")

if __name__ == "__main__":
    test_add_json_schedules()

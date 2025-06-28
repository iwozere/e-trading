#!/usr/bin/env python3
"""
Test script for screener bot components
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

def test_database():
    """Test database functionality"""
    print("Testing database...")
    try:
        from src.screener.telegram.screener_db import (
            init_db, get_or_create_user, add_ticker, 
            list_tickers, all_tickers_for_status
        )
        
        # Test user creation
        test_user_id = "12345"
        user_id = get_or_create_user(test_user_id)
        print(f"✓ User created/retrieved: {user_id}")
        
        # Test adding ticker
        add_ticker(test_user_id, "yf", "AAPL")
        print("✓ Ticker added")
        
        # Test listing tickers
        tickers = list_tickers(test_user_id)
        print(f"✓ Tickers listed: {tickers}")
        
        # Test status tickers
        status_tickers = all_tickers_for_status(test_user_id)
        print(f"✓ Status tickers: {status_tickers}")
        
        return True
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

def test_technical_analysis():
    """Test technical analysis functionality"""
    print("\nTesting technical analysis...")
    try:
        from src.screener.telegram.technicals import calculate_technicals
        from src.screener.telegram.chart import generate_enhanced_chart
        
        # Test technical analysis
        technicals = calculate_technicals("AAPL")
        print(f"✓ Technical analysis completed: {len(technicals)} indicators")
        
        # Test chart generation
        chart_data = generate_enhanced_chart("AAPL")
        print(f"✓ Chart generated: {len(chart_data)} bytes")
        
        return True
    except Exception as e:
        print(f"✗ Technical analysis test failed: {e}")
        return False

def test_email():
    """Test email functionality"""
    print("\nTesting email functionality...")
    try:
        from src.notification.emailer import EmailNotifier
        
        notifier = EmailNotifier()
        print("✓ Email notifier created")
        
        return True
    except Exception as e:
        print(f"✗ Email test failed: {e}")
        return False

def test_bot_imports():
    """Test bot imports"""
    print("\nTesting bot imports...")
    try:
        from src.screener.telegram.bot import bot, dp
        print("✓ Bot imports successful")
        return True
    except Exception as e:
        print(f"✗ Bot import test failed: {e}")
        return False

def main():
    print("Screener Bot Component Tests")
    print("=" * 40)
    
    tests = [
        test_database,
        test_technical_analysis,
        test_email,
        test_bot_imports
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'=' * 40}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Bot should work correctly.")
    else:
        print("✗ Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
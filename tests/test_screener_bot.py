#!/usr/bin/env python3
"""
Enhanced test script for screener bot logic (no Telegram required)
"""
import sys
import os
sys.path.append(os.path.abspath('.'))

import tempfile
from pprint import pprint
from datetime import datetime, timedelta, timezone
import random
import shutil

# TODO: Update imports after ticker management functions are implemented
# from src.data.db.services import telegram_service as db
from src.common.technicals import format_technical_analysis
from src.common.ticker_chart import generate_chart
from src.notification.async_notification_manager import initialize_notification_manager, NotificationType, NotificationPriority
from src.common.ticker_analyzer import analyze_ticker, format_comprehensive_analysis

import yfinance as yf
import pandas as pd

import unittest
from unittest.mock import patch, MagicMock

# TODO: Implement parse_analyze_parameters function
# from src.screener.telegram.telegram_screener_bot import parse_analyze_parameters


TEST_USER_ID = "test_user"
TEST_SMTP_USER = "test@example.com"

# Helper to simulate /my-list

def test_my_list(user_id=TEST_USER_ID):
    print("\n=== /my-list ===")
    tickers = list_tickers(user_id)
    if not tickers:
        print("No tickers found.")
    else:
        for provider, tlist in tickers.items():
            print(f"Provider: {provider}")
            for ticker in tlist:
                print(f"  - {ticker}")

# Helper to simulate /my-status

def test_my_status(user_id=TEST_USER_ID, provider_filter=None, email=None):
    print("\n=== /my-status ===")
    if provider_filter:
        pairs = all_tickers_with_providers_for_status(user_id, provider_filter)
    else:
        pairs = []
    if not pairs:
        print("No tickers found. Use /my-add to add tickers first.")
        return
    email_body = []
    chart_files = []
    for prov, ticker in pairs:
        print(f"\n--- {prov}:{ticker} ---")
        try:
            result = analyze_ticker(ticker, provider=prov)
            if prov.lower() == "yf":
                text = format_comprehensive_analysis(ticker, result.technicals, result.fundamentals)
            else:
                t = result.technicals
                text = f"{ticker} (Binance)\nClose: {getattr(t, 'sma_fast', 0):.2f}, RSI: {getattr(t, 'rsi', 0):.2f}, MACD: {getattr(t, 'macd', 0):.2f}, BB High: {getattr(t, 'bb_upper', 0):.2f}, BB Low: {getattr(t, 'bb_lower', 0):.2f}"
            print(text)
            if email:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"{ticker}_") as temp_file:
                    temp_file.write(result.chart_image)
                    temp_file.flush()
                    chart_files.append(temp_file.name)
            email_body.append(text.replace("*", ""))
        except Exception as e:
            print(f"Error analyzing {prov}:{ticker}: {e}")
    # Send email if requested
    if email and email_body:
        import asyncio
        async def send_email_async():
            notification_manager = await initialize_notification_manager(
                email_sender=TEST_SMTP_USER,
                email_receiver=email
            )
            await notification_manager.send_notification(
                notification_type=NotificationType.INFO,
                title=f"Your Screener Status Report - {len(pairs)} Tickers Analyzed",
                message=f"<h2>Your Screener Status Report</h2><hr>" + "<br><br>".join(email_body),
                priority=NotificationPriority.NORMAL,
                data={},
                source="test_screener_bot",
                channels=["email"],
            )
        asyncio.run(send_email_async())
        print(f"Email sent to {email} with {len(chart_files)} charts.")
        for chart_file in chart_files:
            try:
                os.unlink(chart_file)
            except:
                pass

# Helper to simulate /my-analyze

def test_my_analyze(user_id=TEST_USER_ID, provider="yf", ticker=None, email=None):
    print("\n=== /my-analyze ===")
    if not ticker:
        print("Ticker is required.")
        return
    try:
        result = analyze_ticker(ticker, provider=provider)
        if provider == "yf":
            text = format_comprehensive_analysis(ticker, result.technicals, result.fundamentals)
        else:
            t = result.technicals
            text = f"{ticker} (Binance)\nClose: {getattr(t, 'sma_fast', 0):.2f}, RSI: {getattr(t, 'rsi', 0):.2f}, MACD: {getattr(t, 'macd', 0):.2f}, BB High: {getattr(t, 'bb_upper', 0):.2f}, BB Low: {getattr(t, 'bb_lower', 0):.2f}"
        print(text)
        if email:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"{ticker}_") as temp_file:
                temp_file.write(result.chart_image)
                temp_file.flush()
                chart_file = temp_file.name
            import asyncio
            async def send_email_async():
                notification_manager = await initialize_notification_manager(
                    email_sender=TEST_SMTP_USER,
                    email_receiver=email
                )
                await notification_manager.send_notification(
                    notification_type=NotificationType.INFO,
                    title=f"Comprehensive Analysis for {ticker}",
                    message=text.replace("*", "").replace("\n", "<br>"),
                    priority=NotificationPriority.NORMAL,
                    data={},
                    source="test_screener_bot",
                    channels=["email"],
                )
            asyncio.run(send_email_async())
            print(f"Email sent to {email} with chart for {ticker}.")
            os.unlink(chart_file)
    except Exception as e:
        print(f"Error analyzing {provider}:{ticker}: {e}")

# Helper to simulate simple ticker input

def test_simple_ticker_input():
    ticker = "AAPL"
    print(f"\n=== Simple ticker input: {ticker} ===")
    try:
        result = analyze_ticker(ticker, provider="yf")
        print(format_comprehensive_analysis(ticker, result.technicals, result.fundamentals))
        print(f"Chart generated for {ticker}, {len(result.chart_image)} bytes.")
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")

# --- DB TESTS ---
def test_db_user_registration():
    print("\n=== DB: User Registration ===")
    # Register email
    id = get_or_create_user(TEST_USER_ID)
    email = "testuser@example.com"
    code = f"{random.randint(100000, 999999)}"
    sent_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    set_user_email(TEST_USER_ID, email, code, sent_time)
    # Check email
    stored_email = get_user_email(TEST_USER_ID)
    print(f"Stored email: {stored_email}")
    assert stored_email == email
    # Check verification status
    status = get_user_verification_status(TEST_USER_ID)
    print(f"Verification status: {status}")
    assert status["email"] == email
    assert status["verification_sent"] == sent_time
    assert status["verification_received"] is None
    # Check code
    db_code, db_sent = get_user_verification_code(TEST_USER_ID)
    print(f"Verification code: {db_code}, sent: {db_sent}")
    assert db_code == code
    # Set verified
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    set_user_verified(TEST_USER_ID, now)
    status2 = get_user_verification_status(TEST_USER_ID)
    print(f"After verify: {status2}")
    assert status2["verification_received"] == now
    print("DB user registration/verification: OK")

def test_db_ticker_ops():
    print("\n=== DB: Ticker Operations ===")
    # Clean up first
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM tickers WHERE user_id = (SELECT id FROM users WHERE telegram_user_id=?)", (TEST_USER_ID,))
    conn.commit()
    # Add
    add_ticker(TEST_USER_ID, "yf", "AAPL")
    add_ticker(TEST_USER_ID, "yf", "MSFT")
    add_ticker(TEST_USER_ID, "bnc", "BTCUSDT")
    # List
    tickers = list_tickers(TEST_USER_ID)
    print(f"Tickers: {tickers}")
    assert "yf" in tickers and any(t['ticker'] == 'AAPL' for t in tickers["yf"])
    # Delete
    delete_ticker(TEST_USER_ID, "yf", "AAPL")
    tickers2 = list_tickers(TEST_USER_ID)
    print(f"After delete: {tickers2}")
    assert not any(t['ticker'] == 'AAPL' for t in tickers2.get("yf", []))
    print("DB ticker ops: OK")

# --- BOT LOGIC TESTS ---
def test_my_register_and_verify():
    print("\n=== /my-register and /my-verify ===")
    email = "testuser@example.com"
    code = f"{random.randint(100000, 999999)}"
    sent_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    set_user_email(TEST_USER_ID, email, code, sent_time)
    # Simulate /my-verify
    db_code, db_sent = get_user_verification_code(TEST_USER_ID)
    assert db_code == code
    # Not expired
    sent_dt = datetime.strptime(db_sent, "%Y-%m-%d %H:%M:%S")
    if datetime.now(timezone.utc) > sent_dt + timedelta(hours=1):
        print("Verification code expired!")
    else:
        set_user_verified(TEST_USER_ID, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
        print("Email verified!")
    status = get_user_verification_status(TEST_USER_ID)
    print(f"/my-info: {status}")
    assert status["verification_received"] is not None
    print("Register/verify/info: OK")

def test_my_add_list_delete():
    print("\n=== /my-add, /my-list, /my-delete ===")
    # Clean up
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM tickers WHERE user_id = (SELECT id FROM users WHERE telegram_user_id=?)", (TEST_USER_ID,))
    conn.commit()
    # Add
    add_ticker(TEST_USER_ID, "yf", "AAPL")
    add_ticker(TEST_USER_ID, "bnc", "BTCUSDT")
    # List
    tickers = list_tickers(TEST_USER_ID)
    print(f"Tickers: {tickers}")
    # Delete
    delete_ticker(TEST_USER_ID, "yf", "AAPL")
    tickers2 = list_tickers(TEST_USER_ID)
    print(f"After delete: {tickers2}")
    print("Add/list/delete: OK")

def test_my_status_and_analyze():
    print("\n=== /my-status and /my-analyze ===")
    # Add tickers
    add_ticker(TEST_USER_ID, "yf", "AAPL")
    add_ticker(TEST_USER_ID, "bnc", "BTCUSDT")
    # Simulate verified email
    email = "testuser@example.com"
    code = f"{random.randint(100000, 999999)}"
    sent_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    # /my-status with -email
    pairs = []
    email_body = []
    chart_files = []
    for prov, ticker in pairs:
        print(f"Analyzing {prov}:{ticker}")
        try:
            result = analyze_ticker(ticker, provider=prov)
            if prov == "yf":
                text = format_comprehensive_analysis(ticker, result.technicals, result.fundamentals)
            else:
                t = result.technicals
                text = f"{ticker} (Binance)\nClose: {getattr(t, 'sma_fast', 0):.2f}, RSI: {getattr(t, 'rsi', 0):.2f}, MACD: {getattr(t, 'macd', 0):.2f}, BB High: {getattr(t, 'bb_upper', 0):.2f}, BB Low: {getattr(t, 'bb_lower', 0):.2f}"
            print(text)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"{ticker}_") as temp_file:
                temp_file.write(result.chart_image)
                temp_file.flush()
                chart_files.append(temp_file.name)
            email_body.append(text.replace("*", ""))
        except Exception as e:
            print(f"Error analyzing {prov}:{ticker}: {e}")
    # Send email
    import asyncio
    async def send_email_async():
        notification_manager = await initialize_notification_manager(
            email_sender=TEST_SMTP_USER,
            email_receiver=email
        )
        await notification_manager.send_notification(
            notification_type=NotificationType.INFO,
            title=f"Your Screener Status Report - {len(pairs)} Tickers Analyzed",
            message=f"<h2>Your Screener Status Report</h2><hr>" + "<br><br>".join(email_body),
            priority=NotificationPriority.NORMAL,
            data={},
            source="test_screener_bot",
            channels=["email"],
        )
    asyncio.run(send_email_async())
    print(f"Email sent to {email} with {len(chart_files)} charts.")
    for chart_file in chart_files:
        try:
            os.unlink(chart_file)
        except:
            pass
    print("/my-status and /my-analyze: OK")

def cleanup_test_user():
    print("\n=== Cleanup test user ===")
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM tickers WHERE user_id = (SELECT id FROM users WHERE telegram_user_id=?)", (TEST_USER_ID,))
    cur.execute("DELETE FROM users WHERE telegram_user_id=?", (TEST_USER_ID,))
    conn.commit()
    print("Test user cleaned up.")

# Example usage
if __name__ == "__main__":
    test_db_user_registration()
    #test_db_ticker_ops()
    #test_my_register_and_verify()
    test_my_add_list_delete()
    test_my_status_and_analyze()
    cleanup_test_user()
    print("\nAll tests completed.")

# TODO: Implement parse_analyze_parameters function and uncomment these tests
# class TestParseAnalyzeParameters(unittest.TestCase):
#     def setUp(self):
#         self.user_id = 'test_user'
#         # Patch all_tickers_with_providers_for_status globally for the test class
#         patcher = patch('src.screener.telegram.telegram_screener_bot.all_tickers_with_providers_for_status')
#         self.mock_all_tickers = patcher.start()
#         self.addCleanup(patcher.stop)
#
#     def test_valid_single_ticker(self):
#         def fake_get_ticker_settings(user_id, provider, ticker):
#             if provider == 'YF' and ticker == 'AAPL':
#                 return ('2y', '1d')
#             return (None, None)
#         pairs, telegram_actions, email_flag = parse_analyze_parameters(
#             self.user_id, '/analyze -yf AAPL', fake_get_ticker_settings
#         )
#         self.assertEqual(len(pairs), 1)
#         self.assertEqual(pairs[0][:2], ('YF', 'AAPL'))
#         self.assertEqual(telegram_actions, [])
#         self.assertFalse(email_flag)
#
#     def test_valid_provider_only(self):
#         self.mock_all_tickers.return_value = [('yf', 'AAPL'), ('yf', 'MSFT')]
#         def fake_get_ticker_settings(user_id, provider, ticker):
#             return ('2y', '1d')
#         pairs, telegram_actions, email_flag = parse_analyze_parameters(
#             self.user_id, '/analyze -yf', fake_get_ticker_settings
#         )
#         self.assertEqual(len(pairs), 2)
#         self.assertEqual(pairs[0][:2], ('yf', 'AAPL'))
#         self.assertEqual(telegram_actions, [])
#         self.assertFalse(email_flag)
#
#     def test_missing_ticker_and_provider(self):
#         self.mock_all_tickers.return_value = []
#         def fake_get_ticker_settings(user_id, provider, ticker):
#             return (None, None)
#         pairs, telegram_actions, email_flag = parse_analyze_parameters(
#             self.user_id, '/analyze', fake_get_ticker_settings
#         )
#         self.assertEqual(pairs, [])
#         self.assertTrue(any('No tickers found' in a['content'] for a in telegram_actions))
#         self.assertFalse(email_flag)
#
#     def test_invalid_flag(self):
#         def fake_get_ticker_settings(user_id, provider, ticker):
#             return (None, None)
#         pairs, telegram_actions, email_flag = parse_analyze_parameters(
#             self.user_id, '/analyze -foo', fake_get_ticker_settings
#         )
#         self.assertEqual(pairs, [1])
#         self.assertTrue(any('Usage' in a['content'] for a in telegram_actions))
#         self.assertFalse(email_flag)
#
#     def test_email_flag(self):
#         def fake_get_ticker_settings(user_id, provider, ticker):
#             if provider == 'YF' and ticker == 'AAPL':
#                 return ('2y', '1d')
#             return (None, None)
#         pairs, telegram_actions, email_flag = parse_analyze_parameters(
#             self.user_id, '/analyze -yf AAPL -email', fake_get_ticker_settings
#         )
#         self.assertEqual(len(pairs), 1)
#         self.assertTrue(email_flag)

if __name__ == '__main__':
    unittest.main()
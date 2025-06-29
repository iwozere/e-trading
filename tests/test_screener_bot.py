#!/usr/bin/env python3
"""
Enhanced test script for screener bot logic (no Telegram required)
"""
import sys
import os
sys.path.append(os.path.abspath('.'))

import tempfile
from pprint import pprint
from datetime import datetime, timedelta
import random
import shutil

from src.screener.telegram.screener_db import (
    add_ticker, delete_ticker, list_tickers, all_tickers_for_status, all_tickers_with_providers_for_status,
    set_user_email, get_user_email, get_user_verification_status, get_user_verification_code, set_user_verified, get_conn, get_ticker_settings
)
from src.screener.telegram.technicals import calculate_technicals, format_technical_analysis
from src.screener.telegram.chart import generate_enhanced_chart, generate_binance_chart
from src.notification.async_notification_manager import initialize_notification_manager, NotificationType, NotificationPriority
from src.screener.telegram.combine import analyze_ticker, format_comprehensive_analysis

import yfinance as yf
import pandas as pd
import ta

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
        pairs = all_tickers_for_status(user_id)
    if not pairs:
        print("No tickers found. Use /my-add to add tickers first.")
        return
    email_body = []
    chart_files = []
    for prov, ticker in pairs:
        print(f"\n--- {prov}:{ticker} ---")
        if prov.lower() == "yf":
            result = analyze_ticker(ticker)
            technicals = result.technicals
            fundamentals = result.fundamentals
            text = format_technical_analysis(ticker, technicals)
            comprehensive_text = format_comprehensive_analysis(ticker, technicals, fundamentals)
            # Generate chart
            chart_data = generate_enhanced_chart(ticker)
            if email:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"{ticker}_") as temp_file:
                    temp_file.write(chart_data)
                    temp_file.flush()
                    chart_files.append(temp_file.name)
            # Add to email body
            email_body.append(text.replace("*", ""))
        elif prov.lower() == "bnc":
            # Download Binance data
            symbol = ticker.upper()
            url = f"https://api.binance.com/api/v3/klines"
            params = {"symbol": symbol, "interval": "1d", "limit": 365}
            import requests
            resp = requests.get(url, params=params)
            if resp.status_code != 200:
                print(f"Could not fetch data for '{ticker}' from Binance.")
                continue
            data = resp.json()
            if not data:
                print(f"No price data found for '{ticker}' on Binance.")
                continue
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df = df.dropna()
            # Calculate technicals (simple)
            df["RSI"] = ta.momentum.RSIIndicator(df["close"]).rsi()
            df["MACD"] = ta.trend.MACD(df["close"]).macd()
            bb = ta.volatility.BollingerBands(df["close"])
            df["BB_High"] = bb.bollinger_hband()
            df["BB_Low"] = bb.bollinger_lband()
            latest = df.iloc[-1]
            print(f"Close: {latest['close']:.2f}, RSI: {latest['RSI']:.2f}, MACD: {latest['MACD']:.2f}, BB High: {latest['BB_High']:.2f}, BB Low: {latest['BB_Low']:.2f}")
            # Generate chart
            chart_data = generate_binance_chart(ticker, df)
            if email:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"{ticker}_") as temp_file:
                    temp_file.write(chart_data)
                    temp_file.flush()
                    chart_files.append(temp_file.name)
            # Add to email body
            email_body.append(f"{ticker} (Binance)\nClose: {latest['close']:.2f}, RSI: {latest['RSI']:.2f}, MACD: {latest['MACD']:.2f}, BB High: {latest['BB_High']:.2f}, BB Low: {latest['BB_Low']:.2f}")
        else:
            print(f"Unknown provider '{prov}' for ticker '{ticker}'.")
    # Send email if requested
    if email and email_body:
        # Use async notification manager for email
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
    if provider == "yf":
        technicals_data = calculate_technicals(ticker)
        if not technicals_data:
            print(f"Failed to calculate technicals for {ticker}")
            return
        print(format_technical_analysis(ticker, technicals_data))
        chart_data = generate_enhanced_chart(ticker)
        if email:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"{ticker}_") as temp_file:
                temp_file.write(chart_data)
                temp_file.flush()
                chart_file = temp_file.name
            # Use async notification manager for email
            import asyncio
            async def send_email_async():
                notification_manager = await initialize_notification_manager(
                    email_sender=TEST_SMTP_USER,
                    email_receiver=email
                )
                await notification_manager.send_notification(
                    notification_type=NotificationType.INFO,
                    title=f"Comprehensive Analysis for {ticker}",
                    message=format_technical_analysis(ticker, technicals_data).replace("*", "").replace("\n", "<br>"),
                    priority=NotificationPriority.NORMAL,
                    data={},
                    source="test_screener_bot",
                    channels=["email"],
                )
            asyncio.run(send_email_async())
            print(f"Email sent to {email} with chart for {ticker}.")
            os.unlink(chart_file)
    elif provider == "bnc":
        # Download Binance data
        symbol = ticker.upper()
        url = f"https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": "1d", "limit": 365}
        import requests
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            print(f"Could not fetch data for '{ticker}' from Binance.")
            return
        data = resp.json()
        if not data:
            print(f"No price data found for '{ticker}' on Binance.")
            return
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df.dropna()
        # Calculate technicals (simple)
        df["RSI"] = ta.momentum.RSIIndicator(df["close"]).rsi()
        df["MACD"] = ta.trend.MACD(df["close"]).macd()
        bb = ta.volatility.BollingerBands(df["close"])
        df["BB_High"] = bb.bollinger_hband()
        df["BB_Low"] = bb.bollinger_lband()
        latest = df.iloc[-1]
        print(f"Close: {latest['close']:.2f}, RSI: {latest['RSI']:.2f}, MACD: {latest['MACD']:.2f}, BB High: {latest['BB_High']:.2f}, BB Low: {latest['BB_Low']:.2f}")
        chart_data = generate_binance_chart(ticker, df)
        if email:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"{ticker}_") as temp_file:
                temp_file.write(chart_data)
                temp_file.flush()
                chart_file = temp_file.name
            # Use async notification manager for email
            import asyncio
            async def send_email_async():
                notification_manager = await initialize_notification_manager(
                    email_sender=TEST_SMTP_USER,
                    email_receiver=email
                )
                await notification_manager.send_notification(
                    notification_type=NotificationType.INFO,
                    title=f"Comprehensive Analysis for {ticker}",
                    message=f"{ticker} (Binance)<br>Close: {latest['close']:.2f}, RSI: {latest['RSI']:.2f}, MACD: {latest['MACD']:.2f}, BB High: {latest['BB_High']:.2f}, BB Low: {latest['BB_Low']:.2f}",
                    priority=NotificationPriority.NORMAL,
                    data={},
                    source="test_screener_bot",
                    channels=["email"],
                )
            asyncio.run(send_email_async())
            print(f"Email sent to {email} with chart for {ticker}.")
            os.unlink(chart_file)
    else:
        print(f"Unknown provider '{provider}' for ticker '{ticker}'.")

# Helper to simulate simple ticker input

def test_simple_ticker_input(ticker):
    print(f"\n=== Simple ticker input: {ticker} ===")
    technicals_data = calculate_technicals(ticker)
    if not technicals_data:
        print(f"Failed to calculate technicals for {ticker}")
        return
    print(format_technical_analysis(ticker, technicals_data))
    chart_data = generate_enhanced_chart(ticker)
    print(f"Chart generated for {ticker}, {len(chart_data)} bytes.")

# --- DB TESTS ---
def test_db_user_registration():
    print("\n=== DB: User Registration ===")
    # Register email
    email = "testuser@example.com"
    code = f"{random.randint(100000, 999999)}"
    sent_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
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
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
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
    assert "yf" in tickers and "AAPL" in tickers["yf"]
    # Delete
    delete_ticker(TEST_USER_ID, "yf", "AAPL")
    tickers2 = list_tickers(TEST_USER_ID)
    print(f"After delete: {tickers2}")
    assert "AAPL" not in tickers2.get("yf", [])
    print("DB ticker ops: OK")

# --- BOT LOGIC TESTS ---
def test_my_register_and_verify():
    print("\n=== /my-register and /my-verify ===")
    email = "testuser@example.com"
    code = f"{random.randint(100000, 999999)}"
    sent_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    set_user_email(TEST_USER_ID, email, code, sent_time)
    # Simulate /my-verify
    db_code, db_sent = get_user_verification_code(TEST_USER_ID)
    assert db_code == code
    # Not expired
    sent_dt = datetime.strptime(db_sent, "%Y-%m-%d %H:%M:%S")
    if datetime.utcnow() > sent_dt + timedelta(hours=1):
        print("Verification code expired!")
    else:
        set_user_verified(TEST_USER_ID, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
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
    sent_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    set_user_email(TEST_USER_ID, email, code, sent_time)
    set_user_verified(TEST_USER_ID, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    # /my-status with -email
    pairs = all_tickers_for_status(TEST_USER_ID)
    email_body = []
    chart_files = []
    for prov, ticker in pairs:
        print(f"Analyzing {prov}:{ticker}")
        if prov == "yf":
            technicals_data = calculate_technicals(ticker)
            print(format_technical_analysis(ticker, technicals_data))
            chart_data = generate_enhanced_chart(ticker)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"{ticker}_") as temp_file:
                temp_file.write(chart_data)
                temp_file.flush()
                chart_files.append(temp_file.name)
            email_body.append(format_technical_analysis(ticker, technicals_data).replace("*", ""))
        elif prov == "bnc":
            # Download Binance data
            symbol = ticker.upper()
            url = f"https://api.binance.com/api/v3/klines"
            params = {"symbol": symbol, "interval": "1d", "limit": 365}
            import requests
            resp = requests.get(url, params=params)
            if resp.status_code != 200:
                print(f"Could not fetch data for '{ticker}' from Binance.")
                continue
            data = resp.json()
            if not data:
                print(f"No price data found for '{ticker}' on Binance.")
                continue
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df = df.dropna()
            df["RSI"] = ta.momentum.RSIIndicator(df["close"]).rsi()
            df["MACD"] = ta.trend.MACD(df["close"]).macd()
            bb = ta.volatility.BollingerBands(df["close"])
            df["BB_High"] = bb.bollinger_hband()
            df["BB_Low"] = bb.bollinger_lband()
            latest = df.iloc[-1]
            print(f"Close: {latest['close']:.2f}, RSI: {latest['RSI']:.2f}, MACD: {latest['MACD']:.2f}, BB High: {latest['BB_High']:.2f}, BB Low: {latest['BB_Low']:.2f}")
            chart_data = generate_binance_chart(ticker, df)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"{ticker}_") as temp_file:
                temp_file.write(chart_data)
                temp_file.flush()
                chart_files.append(temp_file.name)
            email_body.append(f"{ticker} (Binance)\nClose: {latest['close']:.2f}, RSI: {latest['RSI']:.2f}, MACD: {latest['MACD']:.2f}, BB High: {latest['BB_High']:.2f}, BB Low: {latest['BB_Low']:.2f}")
    # Send email
    # Use async notification manager for email
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
    test_db_ticker_ops()
    test_my_register_and_verify()
    test_my_add_list_delete()
    test_my_status_and_analyze()
    cleanup_test_user()
    print("\nAll tests completed.") 
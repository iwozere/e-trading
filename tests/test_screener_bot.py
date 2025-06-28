#!/usr/bin/env python3
"""
Enhanced test script for screener bot logic (no Telegram required)
"""
import sys
import os
sys.path.append(os.path.abspath('.'))

import tempfile
from pprint import pprint

from src.screener.telegram.screener_db import (
    add_ticker, list_tickers, all_tickers_for_status, all_tickers_with_providers_for_status
)
from src.screener.telegram.technicals import calculate_technicals, format_technical_analysis
from src.screener.telegram.chart import generate_enhanced_chart, generate_binance_chart
from src.notification.emailer import EmailNotifier

import yfinance as yf
import pandas as pd
import ta

TEST_USER_ID = "test_user"

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
            technicals_data = calculate_technicals(ticker)
            if not technicals_data:
                print(f"Failed to calculate technicals for {ticker}")
                continue
            print(format_technical_analysis(ticker, technicals_data))
            # Generate chart
            chart_data = generate_enhanced_chart(ticker)
            if email:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"{ticker}_") as temp_file:
                    temp_file.write(chart_data)
                    temp_file.flush()
                    chart_files.append(temp_file.name)
            # Add to email body
            email_body.append(format_technical_analysis(ticker, technicals_data).replace("*", ""))
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
        notifier = EmailNotifier()
        email_content = f"<h2>Your Screener Status Report</h2><hr>" + "<br><br>".join(email_body)
        notifier.send_email(
            to_addr=email,
            subject=f"Your Screener Status Report - {len(pairs)} Tickers Analyzed",
            body=email_content,
            attachments=chart_files
        )
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
            notifier = EmailNotifier()
            notifier.send_email(
                to_addr=email,
                subject=f"Comprehensive Analysis for {ticker}",
                body=format_technical_analysis(ticker, technicals_data).replace("*", "").replace("\n", "<br>"),
                attachments=[chart_file]
            )
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
            notifier = EmailNotifier()
            notifier.send_email(
                to_addr=email,
                subject=f"Comprehensive Analysis for {ticker}",
                body=f"{ticker} (Binance)<br>Close: {latest['close']:.2f}, RSI: {latest['RSI']:.2f}, MACD: {latest['MACD']:.2f}, BB High: {latest['BB_High']:.2f}, BB Low: {latest['BB_Low']:.2f}",
                attachments=[chart_file]
            )
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

# Example usage
if __name__ == "__main__":
    # Add some test tickers
    add_ticker(TEST_USER_ID, "yf", "AAPL")
    add_ticker(TEST_USER_ID, "yf", "YQ")
    add_ticker(TEST_USER_ID, "bnc", "BTCUSDT")
    add_ticker(TEST_USER_ID, "bnc", "LTCUSDC")

    test_my_list()
    test_my_status(email=None)  # Set email to test email sending
    test_my_analyze(provider="yf", ticker="AAPL", email=None)
    test_my_analyze(provider="bnc", ticker="BTCUSDT", email=None)
    test_simple_ticker_input("AAPL") 
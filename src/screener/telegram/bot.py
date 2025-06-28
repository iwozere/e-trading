"""
Screener Telegram Bot

Features:
- Per-user ticker management (add, delete, list tickers)
- Analyze all user tickers on command, with BUY/SELL/HOLD signals
- Detailed TA info for BUY/SELL, minimal info for HOLD
- Chart sent only on explicit request
- All actions and errors logged to logs/log/my_screener.log
- User tickers stored in config/screener/my_screener.json (auto-created if missing)
- Optionally email analysis results by providing an email address as the last argument to /my-status or /my-analyze

Commands:
  /my-add -PROVIDER TICKER      Add ticker to your provider list (provider mandatory)
  /my-delete -PROVIDER TICKER   Remove ticker from your provider list (provider mandatory)
  /my-list                      Show all your tickers (all providers)
  /my-list -PROVIDER            Show your tickers for a provider (provider optional)
  /my-status [-PROVIDER] [EMAIL]         Analyze your tickers (optionally for a provider) and optionally email results
  /my-analyze -PROVIDER TICKER [EMAIL]   Analyze ticker and optionally email results + chart

Note: Uses analyze_ticker for all analysis. Respects yfinance rate limits.
"""
# ticker_bot/bot.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import asyncio
import tempfile
import yfinance as yf
import re
import ta
import pandas as pd
from binance.client import Client
import requests
from datetime import datetime, timedelta
import random

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import FSInputFile, Message
from src.notification.logger import setup_logger
from src.screener.telegram.combine import analyze_ticker, format_comprehensive_analysis
from src.screener.telegram.screener_db import add_ticker, delete_ticker, list_tickers, all_tickers_for_status, all_tickers_with_providers_for_status, set_user_email, get_user_email, get_user_verification_status, get_user_verification_code, set_user_verified
from src.screener.telegram.technicals import calculate_technicals
from src.notification.emailer import EmailNotifier
from src.screener.telegram.chart import generate_enhanced_chart, generate_binance_chart
from src.screener.telegram.models import Fundamentals, Technicals

from config.donotshare.donotshare import TELEGRAM_BOT_TOKEN

# Set up logger using the telegram_bot configuration
logger = setup_logger("telegram_bot")

if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN environment variable is not set")
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is not set")

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()


@dp.message(Command("start", "help"))
async def send_welcome(message: Message):
    logger.info(f"User {message.from_user.id} started the bot")
    await message.reply(
        "<b>Welcome to the e-Trading Screener Bot!</b>\n\n"
        "Send a ticker symbol (e.g., AAPL, TSLA, BTCUSDT), and I'll analyze it for you.\n\n"
        "<b>Available providers:</b> yf (yfinance) and bnc (binance for crypto pairs)\n\n"
        "<b>Key commands:</b>\n"
        "/my-register email@example.com Register or update your email for reports\n"
        "/my-verify CODE               Verify your email with the code sent\n"
        "/my-info                      Show your registered email and verification status\n\n"
        "/my-add -PROVIDER TICKER      Add ticker to your provider list. Supported providers are YF and BNC.\n"
        "/my-delete -PROVIDER TICKER   Remove ticker from your provider list\n"
        "/my-list                      Show all your tickers\n"
        "/my-list -PROVIDER            Show your tickers for a provider\n"
        "/my-status [-PROVIDER] [-email]         Analyze your tickers (optionally for a provider), use -email to send to your verified email\n"
        "/my-analyze -PROVIDER TICKER [-email]   Analyze ticker, use -email to send to your verified email\n"
        "<b>Email flow:</b>\n"
        "1. Register your email with /my-register email@example.com\n"
        "2. Check your inbox for a 6-digit code\n"
        "3. Verify with /my-verify CODE\n"
        "4. Use -email flag with /my-status or /my-analyze to receive reports by email (only if verified)\n\n"
        "All actions and errors are logged. For help, contact the admin."
        , parse_mode="HTML"
    )

def is_valid_ticker(text):
    return re.match(r"^[A-Za-z0-9.\-]{1,15}$", text.strip()) is not None

def is_email(s):
    return re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", s) is not None

@dp.message(lambda message: message.text and is_valid_ticker(message.text.strip()))
async def handle_ticker(message: Message):
    ticker = message.text.strip().upper()
    logger.info(f"User {message.from_user.id} requested analysis for {ticker}")
    await message.reply(f"🔍 Analyzing {ticker}...")

    try:
        result = analyze_ticker(ticker)
        logger.info(f"Successfully analyzed {ticker}")

        # Format response text with enhanced information
        technicals = result.technicals
        fundamentals = result.fundamentals
        text = (
            f"📈 <b>{result.ticker}</b> - {fundamentals.company_name or 'Unknown'}\n\n"
            f"💵 Price: ${fundamentals.current_price or 0.0:.2f}\n"
            f"🏦 P/E: {fundamentals.pe_ratio or 0.0:.2f}, Forward P/E: {fundamentals.forward_pe or 0.0:.2f}\n"
            f"💸 Market Cap: ${(fundamentals.market_cap or 0.0)/1e9:.2f}B\n"
            f"📊 EPS: ${fundamentals.earnings_per_share or 0.0:.2f}, Div Yield: {(fundamentals.dividend_yield or 0.0)*100:.2f}%\n\n"
            f"📉 Technical Analysis:\n"
            f"RSI: {technicals.rsi:.2f}\n"
            f"MA(50): ${technicals.sma_50:.2f}\n"
            f"MA(200): ${technicals.sma_200:.2f}\n"
            f"MACD Signal: {technicals.macd_signal:.2f}\n"
            f"Trend: {technicals.trend}\n\n"
            f"📊 Bollinger Bands:\n"
            f"Upper: ${technicals.bb_upper:.2f}\n"
            f"Middle: ${technicals.bb_middle:.2f}\n"
            f"Lower: ${technicals.bb_lower:.2f}\n"
            f"Width: {technicals.bb_width:.4f}\n\n"
            f"🎯 Recommendation: {result.recommendation}"
        )

        # Save chart to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(result.chart_image)
            temp_file.flush()

            await bot.send_photo(
                chat_id=message.chat.id,
                photo=FSInputFile(temp_file.name),
                caption=text,
                parse_mode="HTML",
            )

        # Clean up the temporary file
        os.unlink(temp_file.name)

    except Exception as e:
        logger.error(f"Error analyzing {ticker}", exc_info=e)
        await message.reply(
            f"⚠️ Error analyzing {ticker}:\n"
            f"Please check if the ticker symbol is correct and try again."
        )

@dp.message(Command("my-add"))
async def my_add(message: Message):
    print("[DEBUG] Entered my_add handler", flush=True)
    logger.info("Entered my_add handler")
    user_id = str(message.from_user.id)
    args = message.text.split()
    print(f"[DEBUG] my_add args: {args}", flush=True)
    logger.info(f"my_add args: {args}")
    try:
        if len(args) < 3 or not args[1].startswith('-'):
            await message.reply("Usage: /my-add -PROVIDER TICKER1[,TICKER2,...] (e.g., /my-add -yf AAPL,MSFT,TSLA)")
            print("[DEBUG] my_add: invalid arguments", flush=True)
            logger.warning("my_add: invalid arguments")
            return
        provider = args[1][1:].lower()
        tickers = args[2].upper().split(',')
        added = []
        for ticker in tickers:
            ticker = ticker.strip()
            if ticker:
                try:
                    add_ticker(user_id, provider, ticker)
                    added.append(ticker)
                    print(f"[DEBUG] Added {ticker} to {provider}", flush=True)
                    logger.info(f"User {user_id} added {ticker} to {provider}")
                except Exception as e:
                    print(f"[ERROR] Exception adding {ticker}: {e}", flush=True)
                    logger.error(f"Exception adding {ticker}: {e}", exc_info=e)
        if added:
            await message.reply(f"✅ Added to your {provider} list: {', '.join(added)}.")
            print(f"[DEBUG] my_add: Successfully added: {added}", flush=True)
            logger.info(f"my_add: Successfully added: {added}")
        else:
            await message.reply("No valid tickers provided.")
            print("[DEBUG] my_add: No valid tickers provided", flush=True)
            logger.warning("my_add: No valid tickers provided")
    except Exception as e:
        print(f"[ERROR] Exception in my_add handler: {e}", flush=True)
        logger.error(f"Exception in my_add handler: {e}", exc_info=e)
        await message.reply(f"Error in /my-add: {e}")

@dp.message(Command("my-delete"))
async def my_delete(message: Message):
    print("[DEBUG] Entered my_delete handler", flush=True)
    logger.info("Entered my_delete handler")
    user_id = str(message.from_user.id)
    # Split on whitespace, remove empty strings, and strip each arg
    args = [a.strip() for a in re.split(r'\s+', message.text) if a.strip()]
    print(f"[DEBUG] my_delete args: {args}", flush=True)
    logger.info(f"my_delete args: {args}")
    try:
        if len(args) < 3 or not args[1].startswith('-'):
            await message.reply("Usage: /my-delete -PROVIDER TICKER1[,TICKER2,...] (e.g., /my-delete -yf AAPL,MSFT,TSLA)")
            print("[DEBUG] my_delete: invalid arguments", flush=True)
            logger.warning("my_delete: invalid arguments")
            return
        provider = args[1][1:].lower()
        tickers = args[2].upper().split(',')
        deleted = []
        for ticker in tickers:
            ticker = ticker.strip()
            if ticker:
                try:
                    delete_ticker(user_id, provider, ticker)
                    deleted.append(ticker)
                    print(f"[DEBUG] Deleted {ticker} from {provider}", flush=True)
                    logger.info(f"User {user_id} deleted {ticker} from {provider}")
                except Exception as e:
                    print(f"[ERROR] Exception deleting {ticker}: {e}", flush=True)
                    logger.error(f"Exception deleting {ticker}: {e}", exc_info=e)
        if deleted:
            await message.reply(f"❌ Removed from your {provider} list: {', '.join(deleted)}.")
            print(f"[DEBUG] my_delete: Successfully deleted: {deleted}", flush=True)
            logger.info(f"my_delete: Successfully deleted: {deleted}")
        else:
            await message.reply("No valid tickers provided.")
            print("[DEBUG] my_delete: No valid tickers provided", flush=True)
            logger.warning("my_delete: No valid tickers provided")
    except Exception as e:
        print(f"[ERROR] Exception in my_delete handler: {e}", flush=True)
        logger.error(f"Exception in my_delete handler: {e}", exc_info=e)
        await message.reply(f"Error in /my-delete: {e}")

@dp.message(Command("my-list"))
async def my_list(message: Message):
    print("[DEBUG] Entered my_list handler", flush=True)
    logger.info("Entered my_list handler")
    user_id = str(message.from_user.id)
    args = message.text.split()
    print(f"[DEBUG] my_list args: {args}", flush=True)
    logger.info(f"my_list args: {args}")
    provider = None
    if len(args) == 2 and args[1].startswith('-'):
        provider = args[1][1:].lower()
    try:
        tickers_by_provider = list_tickers(user_id, provider)
        if not tickers_by_provider or all(not v for v in tickers_by_provider.values()):
            await message.reply("Your ticker list is empty. Use /my-add -PROVIDER TICKER to add one.")
            print("[DEBUG] my_list: ticker list is empty", flush=True)
            logger.info("my_list: ticker list is empty")
            return
        lines = []
        for prov, tickers in tickers_by_provider.items():
            lines.append(f"{prov.upper()}: " + ", ".join(tickers))
        await message.reply("Your tickers:\n" + "\n".join(lines))
        print(f"[DEBUG] my_list: Successfully listed: {lines}", flush=True)
        logger.info(f"my_list: Successfully listed: {lines}")
    except Exception as e:
        print(f"[ERROR] Exception in my_list handler: {e}", flush=True)
        logger.error(f"Exception in my_list handler: {e}", exc_info=e)
        await message.reply(f"Error in /my-list: {e}")

@dp.message(Command("my-status"))
async def my_status(message: Message):
    user_id = str(message.from_user.id)
    args = message.text.split()
    email_flag = False
    provider_filter = None
    
    logger.info(f"my_status called by user {user_id} with args: {args}")
    
    # Parse arguments
    for arg in args[1:]:
        if arg == "-email":
            email_flag = True
        elif arg.startswith("-"):
            provider_filter = arg[1:].lower()

    await message.reply("🔍 Analyzing your tickers...")
    logger.info("Sent initial response")
    
    # Get user's tickers
    if provider_filter:
        pairs = all_tickers_with_providers_for_status(user_id, provider_filter)
        logger.info(f"Got {len(pairs)} tickers for provider {provider_filter}")
    else:
        pairs = all_tickers_for_status(user_id)
        logger.info(f"Got {len(pairs)} total tickers")
    
    if not pairs:
        await message.reply("❌ No tickers found. Use /my-add to add tickers first.")
        logger.info("No tickers found for user")
        return

    email_body = []
    chart_files = []
    
    for prov, ticker in pairs:
        logger.info(f"Processing {prov}:{ticker}")
        try:
            if prov.lower() == "yf":
                # Use enhanced analysis for Yahoo Finance tickers
                try:
                    # Calculate technicals first
                    technicals_data = calculate_technicals(ticker)
                    if not technicals_data:
                        await message.reply(f"❌ Failed to calculate technical indicators for {ticker}")
                        continue
                    
                    # Generate chart
                    chart_data = generate_enhanced_chart(ticker)
                    
                    # Get fundamentals from yfinance
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    fundamentals_data = Fundamentals(
                        current_price=info.get('currentPrice', 0.0),
                        company_name=info.get('longName', ticker),
                        market_cap=info.get('marketCap', 0.0),
                        pe_ratio=info.get('trailingPE', 0.0),
                        forward_pe=info.get('forwardPE', 0.0),
                        earnings_per_share=info.get('trailingEps', 0.0),
                        dividend_yield=info.get('dividendYield', 0.0) if info.get('dividendYield') else 0.0
                    )
                    
                    # Format comprehensive analysis for Telegram (plain text)
                    comprehensive_text = format_comprehensive_analysis(ticker, technicals_data, fundamentals_data)
                    telegram_text = comprehensive_text.replace('<br>', '\n').replace('<b>', '').replace('</b>', '')
                    await message.reply(telegram_text)
                    
                    # Format comprehensive analysis for email
                    email_body.append(comprehensive_text)
                    
                    # Get recommendation
                    recommendation = technicals_data.recommendations.overall.signal if technicals_data.recommendations else 'HOLD'
                    
                    # Save chart for email attachment
                    if email_flag:
                        chart_filename = f"{ticker}_analysis.png"
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"{ticker}_") as temp_file:
                            temp_file.write(chart_data)
                            temp_file.flush()
                            chart_files.append(temp_file.name)
                            logger.info(f"Saved chart for {ticker}")
                    
                    # Send chart to Telegram
                    chart_temp_file = None
                    try:
                        chart_temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                        chart_temp_file.write(chart_data)
                        chart_temp_file.flush()
                        
                        await bot.send_photo(
                            chat_id=message.chat.id,
                            photo=FSInputFile(chart_temp_file.name),
                            caption=f"📊 {ticker} Analysis\n🎯 {recommendation}",
                            parse_mode="HTML",
                        )
                        logger.info(f"Sent chart for {ticker} to Telegram")
                        
                    except Exception as e:
                        logger.error(f"Error analyzing YF ticker {ticker}: {e}")
                        await message.reply(f"❌ Error analyzing {ticker}: {str(e)}")
                    finally:
                        # Clean up temp file only after message is delivered
                        if chart_temp_file:
                            try:
                                os.unlink(chart_temp_file.name)
                                chart_temp_file.close()
                            except Exception as cleanup_error:
                                logger.warning(f"Failed to cleanup chart temp file for {ticker}: {cleanup_error}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing YF ticker {ticker}: {e}")
                    await message.reply(f"❌ Error analyzing {ticker}: {str(e)}")
                    continue

            elif prov.lower() == "bnc":
                # Enhanced Binance analysis with chart generation
                symbol = ticker.upper()
                url = f"https://api.binance.com/api/v3/klines"
                params = {"symbol": symbol, "interval": "1d", "limit": 365}
                resp = requests.get(url, params=params)
                if resp.status_code != 200:
                    await message.reply(f"❌ Could not fetch data for '{ticker}' from Binance.")
                    continue
                data = resp.json()
                if not data:
                    await message.reply(f"❌ No price data found for '{ticker}' on Binance.")
                    continue
                
                df = pd.DataFrame(data, columns=[
                    "timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                ])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float)
                df = df.dropna()
                
                # Calculate technical indicators
                df["RSI"] = ta.momentum.RSIIndicator(df["close"]).rsi()
                df["MACD"] = ta.trend.MACD(df["close"]).macd()
                df["MACD_Signal"] = ta.trend.MACD(df["close"]).macd_signal()
                df["MACD_Hist"] = ta.trend.MACD(df["close"]).macd_diff()
                df["SMA_20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
                df["EMA_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
                bb = ta.volatility.BollingerBands(df["close"])
                df["BB_High"] = bb.bollinger_hband()
                df["BB_Middle"] = bb.bollinger_mavg()
                df["BB_Low"] = bb.bollinger_lband()
                
                latest = df.iloc[-1]
                
                # Generate per-indicator recommendations
                rsi_val = latest['RSI']
                if rsi_val < 30:
                    rsi_rec = "BUY: Oversold condition"
                elif rsi_val > 70:
                    rsi_rec = "SELL: Overbought condition"
                else:
                    rsi_rec = "HOLD: Neutral condition"
                
                macd_val = latest['MACD']
                macd_signal = latest['MACD_Signal']
                macd_hist = latest['MACD_Hist']
                if macd_val > macd_signal and macd_hist > 0:
                    macd_rec = "BUY: Bullish MACD crossover"
                elif macd_val < macd_signal and macd_hist < 0:
                    macd_rec = "SELL: Bearish MACD crossover"
                else:
                    macd_rec = "HOLD: MACD neutral"
                
                close_val = latest['close']
                bb_high = latest['BB_High']
                bb_middle = latest['BB_Middle']
                bb_low = latest['BB_Low']
                if close_val <= bb_low:
                    bb_rec = "BUY: Price at lower Bollinger band"
                elif close_val >= bb_high:
                    bb_rec = "SELL: Price at upper Bollinger band"
                elif close_val < bb_middle:
                    bb_rec = "BUY: Price below middle band"
                else:
                    bb_rec = "HOLD: Price above middle band"
                
                sma_20 = latest['SMA_20']
                ema_20 = latest['EMA_20']
                if close_val > sma_20 and close_val > ema_20:
                    ma_rec = "BUY: Price above moving averages"
                elif close_val < sma_20 and close_val < ema_20:
                    ma_rec = "SELL: Price below moving averages"
                else:
                    ma_rec = "HOLD: Mixed moving average signals"
                
                # Overall recommendation
                buy_count = sum(1 for rec in [rsi_rec, macd_rec, bb_rec, ma_rec] if rec.startswith("BUY"))
                sell_count = sum(1 for rec in [rsi_rec, macd_rec, bb_rec, ma_rec] if rec.startswith("SELL"))
                
                if buy_count > sell_count and buy_count >= 2:
                    overall_rec = "BUY: Multiple bullish signals"
                elif sell_count > buy_count and sell_count >= 2:
                    overall_rec = "SELL: Multiple bearish signals"
                else:
                    overall_rec = "HOLD: Mixed signals"
                
                text = (
                    f"<b>{ticker}</b> (Binance)\n\n"
                    f"📊 <b>Price Analysis:</b>\n"
                    f"Latest Close: {latest['close']:.2f}\n\n"
                    f"📈 <b>Technical Indicators:</b>\n"
                    f"RSI ({latest['RSI']:.2f}): {rsi_rec}\n"
                    f"MACD ({latest['MACD']:.2f}): {macd_rec}\n"
                    f"Bollinger Bands: {bb_rec}\n"
                    f"Moving Averages: {ma_rec}\n\n"
                    f"🎯 <b>Overall Recommendation:</b> {overall_rec}"
                )
                
                await message.reply(text, parse_mode="HTML")
                
                # Format for email (with fancy icons and HTML tags)
                email_text = (
                    f"<b>📊 {ticker} (Binance)</b><br><br>"
                    f"💰 <b>Latest Close:</b> {latest['close']:.2f}<br>"
                    f"<b>📈 Technical Indicators:</b><br>"
                    f"🔸 <b>RSI</b> ({latest['RSI']:.2f}): {rsi_rec}<br>"
                    f"🔸 <b>MACD</b> ({latest['MACD']:.2f}): {macd_rec}<br>"
                    f"🔸 <b>Bollinger Bands:</b> {bb_rec}<br>"
                    f"🔸 <b>Moving Averages:</b> {ma_rec}<br><br>"
                    f"🎯 <b>Overall Recommendation:</b> {overall_rec}"
                )
                email_body.append(email_text)
                
                # Generate and send chart for BNC tickers
                chart_temp_file = None
                try:
                    chart_data = generate_binance_chart(ticker, df)
                    
                    # Save chart for email attachment
                    if email_flag:
                        chart_filename = f"{ticker}_analysis.png"
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"{ticker}_") as temp_file:
                            temp_file.write(chart_data)
                            temp_file.flush()
                            chart_files.append(temp_file.name)
                            logger.info(f"Saved chart for {ticker}")
                    
                    # Send chart to Telegram
                    chart_temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    chart_temp_file.write(chart_data)
                    chart_temp_file.flush()
                    
                    await bot.send_photo(
                        chat_id=message.chat.id,
                        photo=FSInputFile(chart_temp_file.name),
                        caption=f"📊 {ticker} Analysis (Binance)\n🎯 {overall_rec}",
                        parse_mode="HTML",
                    )
                    logger.info(f"Sent BNC chart for {ticker} to Telegram")
                    
                except Exception as chart_error:
                    logger.error(f"Failed to generate chart for BNC ticker {ticker}: {chart_error}", exc_info=e)
                    await message.reply(f"⚠️ Analysis completed but chart generation failed for {ticker}")
                finally:
                    # Clean up temp file only after message is delivered
                    if chart_temp_file:
                        try:
                            os.unlink(chart_temp_file.name)
                            chart_temp_file.close()
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to cleanup chart temp file for {ticker}: {cleanup_error}")
                
            else:
                await message.reply(f"Unknown provider '{prov}' for ticker '{ticker}'.")
                
        except Exception as e:
            await message.reply(f"Error analyzing {ticker}: {e}")
            logger.error(f"User {user_id} error {ticker}: {e}")
    
    # Send comprehensive email if requested
    if email_flag:
        # Check verification
        status = get_user_verification_status(user_id)
        if not status or not status["email"]:
            await message.reply("No email registered. Use /my-register to set your email.")
            return
        if not status["verification_received"]:
            await message.reply("Your email is not verified. Use /my-verify CODE to verify.")
            return
        email = status["email"]
        logger.info(f"Sending email to {email} with {len(chart_files)} charts")
        try:
            notifier = EmailNotifier()
            email_content = f"""
            <h2>📊 Your Screener Status Report</h2>
            <p>Analysis completed for {len(pairs)} ticker(s)</p>
            <hr>
            """
            email_content += "<br><br>".join(email_body)
            notifier.send_email(
                to_addr=email,
                subject=f"Your Screener Status Report - {len(pairs)} Tickers Analyzed",
                body=email_content,
                attachments=chart_files
            )
            await message.reply(f"📧 Status report sent to {email} with {len(chart_files)} charts")
            logger.info("Email sent successfully")
            for chart_file in chart_files:
                try:
                    os.unlink(chart_file)
                except:
                    pass
        except Exception as e:
            await message.reply(f"❌ Failed to send email: {e}")
            logger.error(f"Email send error: {e}")
    elif any(arg == "-email" for arg in args[1:]):
        await message.reply("❌ No analysis data to send to your email.")

@dp.message(Command("my-analyze"))
async def my_analyze(message: Message):
    user_id = str(message.from_user.id)
    args = message.text.split()
    email_flag = False
    provider = None
    ticker = None
    for arg in args[1:]:
        if arg == "-email":
            email_flag = True
        elif arg.startswith("-") and provider is None:
            provider = arg[1:].lower()
        elif ticker is None:
            ticker = arg.upper()
    if not provider or not ticker:
        await message.reply("Usage: /my-analyze -PROVIDER TICKER [-email]")
        return
    try:
        result = analyze_ticker(ticker)
        
        # Enhanced text with comprehensive analysis
        technicals = result.technicals
        fundamentals = result.fundamentals
        text = format_comprehensive_analysis(ticker, technicals, fundamentals)
        
        chart_temp_file = None
        try:
            chart_temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            chart_temp_file.write(result.chart_image)
            chart_temp_file.flush()
            
            await bot.send_photo(
                chat_id=message.chat.id,
                photo=FSInputFile(chart_temp_file.name),
                caption=f"📊 {ticker} Analysis\n🎯 {result.recommendation}",
                parse_mode="HTML",
            )
            
            # Email if requested
            if email_flag:
                status = get_user_verification_status(user_id)
                if not status or not status["email"]:
                    await message.reply("No email registered. Use /my-register to set your email.")
                    return
                if not status["verification_received"]:
                    await message.reply("Your email is not verified. Use /my-verify CODE to verify.")
                    return
                email = status["email"]
                notifier = EmailNotifier()
                notifier.send_email(
                    to_addr=email,
                    subject=f"Comprehensive Analysis for {ticker}",
                    body=text,
                    attachments=[chart_temp_file.name]
                )
                await message.reply(f"📧 Analysis for {ticker} sent to {email}")
                
        except Exception as e:
            await message.reply(f"Error analyzing {ticker}: {e}")
            logger.error(f"User {user_id} error analyze {ticker}: {e}")
        finally:
            # Clean up temp file only after message is delivered
            if chart_temp_file:
                try:
                    os.unlink(chart_temp_file.name)
                    chart_temp_file.close()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup chart temp file for {ticker}: {cleanup_error}")
                    
        logger.info(f"User {user_id} analyzed {ticker}")
    except Exception as e:
        await message.reply(f"Error analyzing {ticker}: {e}")
        logger.error(f"User {user_id} error analyze {ticker}: {e}")

@dp.message(Command("my-register"))
async def my_register(message: Message):
    args = message.text.split()
    if len(args) != 2 or not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", args[1]):
        await message.reply("Usage: /my-register email@example.com")
        return
    email = args[1].strip()
    telegram_id = str(message.from_user.id)
    code = f"{random.randint(100000, 999999)}"
    sent_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    set_user_email(telegram_id, email, code, sent_time)
    # Send verification email
    subject = "e-Trading: email verification"
    body = f"""
    <h2>e-Trading Email Verification</h2>
    <p>Your verification code is: <b>{code}</b></p>
    <p>Enter this code in Telegram using <b>/my-verify {code}</b> within 1 hour to verify your email.</p>
    """
    try:
        EmailNotifier().send_email(email, subject, body)
        await message.reply(f"Verification code sent to {email}. Please check your inbox and use /my-verify CODE in Telegram.")
    except Exception as e:
        await message.reply(f"Failed to send verification email: {e}")

@dp.message(Command("my-verify"))
async def my_verify(message: Message):
    args = message.text.split()
    if len(args) != 2 or not args[1].isdigit():
        await message.reply("Usage: /my-verify CODE (6 digits)")
        return
    code = args[1]
    telegram_id = str(message.from_user.id)
    db_code, sent_time = get_user_verification_code(telegram_id)
    if not db_code or not sent_time:
        await message.reply("No verification code found. Please register your email first with /my-register.")
        return
    # Check code and expiry
    sent_dt = datetime.strptime(sent_time, "%Y-%m-%d %H:%M:%S")
    if code != db_code:
        await message.reply("Invalid verification code.")
        return
    if datetime.utcnow() > sent_dt + timedelta(hours=1):
        await message.reply("Verification code expired. Please re-register your email.")
        return
    set_user_verified(telegram_id, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    await message.reply("✅ Email verified successfully! You can now use -email flag to receive reports by email.")

@dp.message(Command("my-info"))
async def my_info(message: Message):
    telegram_id = str(message.from_user.id)
    status = get_user_verification_status(telegram_id)
    if not status or not status["email"]:
        await message.reply("No email registered. Use /my-register email@example.com to set your email.")
        return
    verified = bool(status["verification_received"])
    reply = f"<b>Email:</b> {status['email']}\n"
    reply += f"<b>Verified:</b> {'✅' if verified else '❌'}\n"
    reply += f"<b>Verification sent:</b> {status['verification_sent']}\n"
    reply += f"<b>Verification received:</b> {status['verification_received'] or '-'}"
    await message.reply(reply, parse_mode="HTML")

async def main():
    logger.info("Starting ticker analyzer bot")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

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
from src.screener.telegram.screener_db import (
    add_ticker, delete_ticker, list_tickers, all_tickers_for_status, all_tickers_with_providers_for_status,
    set_user_email, get_user_email, get_user_verification_status, get_user_verification_code, set_user_verified,
    get_ticker_settings, update_ticker_settings
)
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

DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"

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

# Helper to parse period/interval flags from args
def parse_period_interval(args):
    period = None
    interval = None
    for arg in args:
        if arg.startswith('-period='):
            period = arg.split('=', 1)[1]
        elif arg.startswith('-interval='):
            interval = arg.split('=', 1)[1]
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
        logger.error(f"Error analyzing {ticker}", exc_info=True)
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
                    # Parse -period= and -interval= flags per ticker, store in DB
                    period, interval = get_ticker_settings(ticker)
                    if not period:
                        period = DEFAULT_PERIOD
                    if not interval:
                        interval = DEFAULT_INTERVAL
                    add_ticker(user_id, provider, ticker, period, interval)
                    added.append(ticker)
                    print(f"[DEBUG] Added {ticker} to {provider}", flush=True)
                    logger.info(f"User {user_id} added {ticker} to {provider}")
                except Exception as e:
                    print(f"[ERROR] Exception adding {ticker}: {e}", flush=True)
                    logger.error(f"Exception adding {ticker}: {e}", exc_info=True)
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
        logger.error(f"Exception in my_add handler: {e}", exc_info=True)
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
                    logger.error(f"Exception deleting {ticker}: {e}", exc_info=True)
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
        logger.error(f"Exception in my_delete handler: {e}", exc_info=True)
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
        logger.error(f"Exception in my_list handler: {e}", exc_info=True)
        await message.reply(f"Error in /my-list: {e}")

@dp.message(Command("analyze"))
async def analyze_command(message: Message):
    user_id = str(message.from_user.id)
    args = message.text.split()
    # Parse flags and arguments
    ticker = None
    provider = None
    email_flag = False
    period = None
    interval = None
    for arg in args[1:]:
        if arg == "-email":
            email_flag = True
        elif arg.startswith("-") and arg not in ("-email",):
            if arg[1:] in ["yf", "bnc"]:
                provider = arg[1:].lower()
            elif arg.startswith("-period="):
                period = arg.split("=", 1)[1]
            elif arg.startswith("-interval="):
                interval = arg.split("=", 1)[1]
        elif ticker is None:
            ticker = arg.upper()
    # If ticker is given, analyze just that ticker
    if ticker:
        # If provider not given, try to infer from DB
        if not provider:
            for prov in ["yf", "bnc"]:
                p, i = get_ticker_settings(user_id, prov, ticker)
                if p or i:
                    provider = prov
                    break
        if not provider:
            provider = "yf"  # Default to yf if not found
        # Get period/interval from DB if not provided
        db_period, db_interval = get_ticker_settings(user_id, provider, ticker)
        if not period:
            period = db_period or DEFAULT_PERIOD
        if not interval:
            interval = db_interval or DEFAULT_INTERVAL
        await message.reply(f"🔍 Analyzing {ticker} (provider={provider}, period={period}, interval={interval})...")
        result = analyze_ticker(ticker, period=period, interval=interval)
        technicals = result.technicals
        fundamentals = result.fundamentals
        text = format_comprehensive_analysis(ticker, technicals, fundamentals)
        await message.reply(text.replace('<br>', '\n').replace('<b>', '').replace('</b>', ''))
        # Chart
        chart_data = generate_enhanced_chart(ticker, technicals, period=period, interval=interval)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"{ticker}_") as temp_file:
            temp_file.write(chart_data)
            temp_file.flush()
            chart_file = temp_file.name
        await bot.send_photo(
            chat_id=message.chat.id,
            photo=FSInputFile(chart_file),
            caption=f"📊 {ticker} Analysis\n🎯 {technicals.recommendations.get('overall', {}).get('signal', 'HOLD') if technicals.recommendations else 'HOLD'}",
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
                attachments=[chart_file]
            )
            await message.reply(f"📧 Analysis for {ticker} sent to {email}")
        try:
            os.unlink(chart_file)
        except:
            pass
        return
    # If no ticker, analyze all tickers in user's list (optionally filtered by provider)
    pairs = all_tickers_with_providers_for_status(user_id, provider)
    if not pairs:
        await message.reply("No tickers found. Use /my-add to add tickers first.")
        return
    email_body = []
    chart_files = []
    for prov, ticker in pairs:
        p, i = get_ticker_settings(user_id, prov, ticker)
        use_period = period or p or DEFAULT_PERIOD
        use_interval = interval or i or DEFAULT_INTERVAL
        await message.reply(f"🔍 Analyzing {ticker} (provider={prov}, period={use_period}, interval={use_interval})...")
        result = analyze_ticker(ticker, period=use_period, interval=use_interval)
        technicals = result.technicals
        fundamentals = result.fundamentals
        text = format_comprehensive_analysis(ticker, technicals, fundamentals)
        await message.reply(text.replace('<br>', '\n').replace('<b>', '').replace('</b>', ''))
        chart_data = generate_enhanced_chart(ticker, technicals, period=use_period, interval=use_interval)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"{ticker}_") as temp_file:
            temp_file.write(chart_data)
            temp_file.flush()
            chart_files.append(temp_file.name)
        email_body.append(text)
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
        for chart_file in chart_files:
            try:
                os.unlink(chart_file)
            except:
                pass

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

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

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import FSInputFile, Message
from src.notification.logger import setup_logger
from src.screener.telegram.combine import analyze_ticker, format_comprehensive_analysis
from src.screener.telegram.screener_db import add_ticker, delete_ticker, list_tickers, all_tickers_for_status, all_tickers_with_providers_for_status
from src.screener.telegram.technicals import calculate_technicals
from src.notification.emailer import EmailNotifier

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
        "📊 Send a ticker symbol (e.g., AAPL, TSLA, BTC-USD), and I'll analyze it for you.\n\n"
        "Available providers: yf (yfinance) and bnc (binance for crypto pairs)\n"
        "Available commands:\n"
        "/start or /help - Show this message\n"
        "/my-add -PROVIDER TICKER      Add ticker to your provider list (provider mandatory)\n"
        "/my-delete -PROVIDER TICKER   Remove ticker from your provider list (provider mandatory)\n"
        "/my-list                      Show all your tickers (all providers)\n"
        "/my-list -PROVIDER            Show your tickers for a provider (provider optional)\n"
        "/my-status [-PROVIDER] [EMAIL]         Analyze your tickers (optionally for a provider) and optionally email results\n"
        "/my-analyze -PROVIDER TICKER [EMAIL]   Analyze ticker and optionally email results + chart\n"
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
        text = (
            f"📈 <b>{result.ticker}</b> - {result.fundamentals.company_name or 'Unknown'}\n\n"
            f"💵 Price: ${result.fundamentals.current_price or 0.0:.2f}\n"
            f"🏦 P/E: {result.fundamentals.pe_ratio or 0.0:.2f}, Forward P/E: {result.fundamentals.forward_pe or 0.0:.2f}\n"
            f"💸 Market Cap: ${(result.fundamentals.market_cap or 0.0)/1e9:.2f}B\n"
            f"📊 EPS: ${result.fundamentals.earnings_per_share or 0.0:.2f}, Div Yield: {(result.fundamentals.dividend_yield or 0.0)*100:.2f}%\n\n"
            f"📉 Technical Analysis:\n"
            f"RSI: {result.technicals.rsi:.2f}\n"
            f"MA(50): ${result.technicals.sma_50:.2f}\n"
            f"MA(200): ${result.technicals.sma_200:.2f}\n"
            f"MACD Signal: {result.technicals.macd_signal:.2f}\n"
            f"Trend: {result.technicals.trend}\n\n"
            f"📊 Bollinger Bands:\n"
            f"Upper: ${result.technicals.bb_upper:.2f}\n"
            f"Middle: ${result.technicals.bb_middle:.2f}\n"
            f"Lower: ${result.technicals.bb_lower:.2f}\n"
            f"Width: {result.technicals.bb_width:.4f}\n\n"
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
    email = None
    provider_filter = None
    
    logger.info(f"my_status called by user {user_id} with args: {args}")
    
    # Parse arguments
    if len(args) == 3 and is_email(args[2]):
        provider_filter = args[1][1:].lower()
        email = args[2]
        logger.info(f"Parsed: provider_filter={provider_filter}, email={email}")
    elif len(args) == 2:
        if is_email(args[1]):
            email = args[1]
            logger.info(f"Parsed: email={email}")
        else:
            provider_filter = args[1][1:].lower()
            logger.info(f"Parsed: provider_filter={provider_filter}")
    elif len(args) > 3:
        await message.reply("Usage: /my-status [-PROVIDER] [EMAIL] (e.g., /my-status -yf user@email.com)")
        return

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
                result = analyze_ticker(ticker)
                logger.info(f"Successfully analyzed {ticker}")
                
                # Format comprehensive analysis for email
                technicals_data = calculate_technicals(ticker)
                fundamentals_data = {
                    'current_price': result.fundamentals.current_price,
                    'company_name': result.fundamentals.company_name,
                    'market_cap': result.fundamentals.market_cap,
                    'pe_ratio': result.fundamentals.pe_ratio,
                    'forward_pe': result.fundamentals.forward_pe,
                    'earnings_per_share': result.fundamentals.earnings_per_share,
                    'dividend_yield': result.fundamentals.dividend_yield
                }
                
                comprehensive_text = format_comprehensive_analysis(ticker, technicals_data, fundamentals_data)
                email_body.append(comprehensive_text)
                
                # Save chart for email attachment
                if email:
                    chart_filename = f"{ticker}_analysis.png"
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"{ticker}_") as temp_file:
                        temp_file.write(result.chart_image)
                        temp_file.flush()
                        chart_files.append(temp_file.name)
                        logger.info(f"Saved chart for {ticker}")
                
                # Send chart to Telegram
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_file.write(result.chart_image)
                    temp_file.flush()
                    
                    await bot.send_photo(
                        chat_id=message.chat.id,
                        photo=FSInputFile(temp_file.name),
                        caption=f"📊 {ticker} Analysis\n🎯 {result.recommendation}",
                        parse_mode="HTML",
                    )
                    logger.info(f"Sent chart for {ticker} to Telegram")
                    
                    # Clean up temp file
                    os.unlink(temp_file.name)

            elif prov.lower() == "bnc":
                # Enhanced Binance analysis
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
                df["SMA_20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
                df["EMA_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
                bb = ta.volatility.BollingerBands(df["close"])
                df["BB_High"] = bb.bollinger_hband()
                df["BB_Low"] = bb.bollinger_lband()
                
                latest = df.iloc[-1]
                
                # Generate recommendation
                rsi_val = latest['RSI']
                if rsi_val < 30:
                    recommendation = "BUY: Oversold condition"
                elif rsi_val > 70:
                    recommendation = "SELL: Overbought condition"
                else:
                    recommendation = "HOLD: Neutral condition"
                
                text = (
                    f"<b>{ticker}</b> (Binance)\n"
                    f"Latest Close: {latest['close']:.2f}\n"
                    f"RSI: {latest['RSI']:.2f}\n"
                    f"MACD: {latest['MACD']:.2f}\n"
                    f"SMA 20: {latest['SMA_20']:.2f}\n"
                    f"EMA 20: {latest['EMA_20']:.2f}\n"
                    f"Bollinger High: {latest['BB_High']:.2f}\n"
                    f"Bollinger Low: {latest['BB_Low']:.2f}\n"
                    f"🎯 Recommendation: {recommendation}"
                )
                
                await message.reply(text, parse_mode="HTML")
                email_body.append(text.replace('<b>', '').replace('</b>', ''))
                
            else:
                await message.reply(f"Unknown provider '{prov}' for ticker '{ticker}'.")
                
        except Exception as e:
            await message.reply(f"Error analyzing {ticker}: {e}")
            logger.error(f"User {user_id} error {ticker}: {e}")
    
    # Send comprehensive email if requested
    if email and email_body:
        logger.info(f"Sending email to {email} with {len(chart_files)} charts")
        try:
            notifier = EmailNotifier()
            
            # Create comprehensive email body
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
            
            # Clean up chart files
            for chart_file in chart_files:
                try:
                    os.unlink(chart_file)
                except:
                    pass
                    
        except Exception as e:
            await message.reply(f"❌ Failed to send email: {e}")
            logger.error(f"Email send error: {e}")
    elif email:
        logger.warning(f"Email requested but no email_body generated")
        await message.reply(f"❌ No analysis data to send to {email}")

@dp.message(Command("my-analyze"))
async def my_analyze(message: Message):
    user_id = str(message.from_user.id)
    args = message.text.split()
    email = None
    if len(args) == 4 and is_email(args[3]):
        provider = args[1][1:].lower()
        ticker = args[2].upper()
        email = args[3]
    elif len(args) == 3 and is_email(args[2]):
        provider = args[1][1:].lower()
        ticker = None
        email = args[2]
    elif len(args) == 3:
        provider = args[1][1:].lower()
        ticker = args[2].upper()
    else:
        await message.reply("Usage: /my-analyze -PROVIDER TICKER [EMAIL] (e.g., /my-analyze -yf AAPL user@email.com)")
        return
    if not ticker:
        await message.reply("Ticker is required.")
        return
    try:
        result = analyze_ticker(ticker)
        
        # Enhanced text with comprehensive analysis
        technicals_data = calculate_technicals(ticker)
        fundamentals_data = {
            'current_price': result.fundamentals.current_price,
            'company_name': result.fundamentals.company_name,
            'market_cap': result.fundamentals.market_cap,
            'pe_ratio': result.fundamentals.pe_ratio,
            'forward_pe': result.fundamentals.forward_pe,
            'earnings_per_share': result.fundamentals.earnings_per_share,
            'dividend_yield': result.fundamentals.dividend_yield
        }
        
        comprehensive_text = format_comprehensive_analysis(ticker, technicals_data, fundamentals_data)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(result.chart_image)
            temp_file.flush()
            await bot.send_photo(
                chat_id=message.chat.id,
                photo=FSInputFile(temp_file.name),
                caption=f"📊 {ticker} Analysis\n🎯 {result.recommendation}",
                parse_mode="HTML",
            )
            # Email if requested
            if email:
                notifier = EmailNotifier()
                notifier.send_email(
                    to_addr=email,
                    subject=f"Comprehensive Analysis for {ticker}",
                    body=comprehensive_text.replace('<b>', '').replace('</b>', '').replace('\n', '<br>'),
                    attachments=[temp_file.name]
                )
                await message.reply(f"📧 Analysis for {ticker} sent to {email}")
        os.unlink(temp_file.name)
        logger.info(f"User {user_id} analyzed {ticker}")
    except Exception as e:
        await message.reply(f"Error analyzing {ticker}: {e}")
        logger.error(f"User {user_id} error analyze {ticker}: {e}")

async def main():
    logger.info("Starting ticker analyzer bot")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

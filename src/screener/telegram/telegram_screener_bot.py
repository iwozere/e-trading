"""
Screener Telegram Bot

Features:
- Per-user ticker management (add, delete, list tickers)
- Analyze all user tickers on command, with BUY/SELL/HOLD signals
- Detailed TA info for BUY/SELL, minimal info for HOLD
- Chart sent only on explicit request
- All actions and errors logged to logs/log/my_screener.log
- User tickers stored in config/screener/my_screener.json (auto-created if missing)
- Optionally email analysis results by providing an email address as the last argument to /analyze

Commands:
  /add -PROVIDER TICKER      Add ticker to your provider list (provider mandatory)
  /delete -PROVIDER TICKER   Remove ticker from your provider list (provider mandatory)
  /list                      Show all your tickers (all providers)
  /list -PROVIDER            Show your tickers for a provider (provider optional)
  /analyze [-PROVIDER] [TICKER] [-email]   Analyze ticker and optionally send to email results + chart

Note: Uses analyze_ticker for all analysis. Respects yfinance rate limits.
"""
# ticker_bot/bot.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import asyncio
import re
from datetime import datetime, timedelta, timezone
import random
import functools
import tempfile

from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import FSInputFile, Message
from src.notification.logger import setup_logger
from src.screener.telegram.combine import analyze_ticker
from src.screener.telegram.screener_db import (
    add_ticker, delete_ticker, list_tickers, all_tickers_with_providers_for_status,
    get_user_verification_status, get_user_verification_code, set_user_verified, get_ticker_settings, set_user_email
)
from src.notification.async_notification_manager import initialize_notification_manager, NotificationType, NotificationPriority

from config.donotshare.donotshare import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SMTP_USER, SMTP_PASSWORD

# Set up logger using the telegram_bot configuration
logger = setup_logger("telegram_bot")

notification_manager = None  # global

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

async def main():
    global notification_manager
    notification_manager = await initialize_notification_manager(
        telegram_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID,
        email_api_key=SMTP_PASSWORD,
        email_sender=SMTP_USER,
        email_receiver=SMTP_USER  # or dummy
    )
    logger.info("Starting ticker analyzer bot")
    await dp.start_polling(bot)

DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"

@dp.message(Command("start", "help"))
async def send_welcome(message: Message):
    logger.info("User %s started the bot", message.from_user.id)
    await message.reply(
        "<b>Welcome to the e-Trading Screener Bot!</b>\n\n"
        "Send a ticker symbol (e.g., AAPL, TSLA, BTCUSDT), and I'll analyze it for you.\n\n"
        "<b>Available providers:</b> yf (yfinance) and bnc (binance for crypto pairs)\n\n"
        "<b>Key commands:</b>\n"
        "/register email@example.com Register or update your email for reports\n"
        "/verify CODE               Verify your email with the code sent\n"
        "/info                      Show your registered email and verification status\n\n"
        "/add -PROVIDER TICKER      Add ticker to your provider list. Supported providers are YF and BNC.\n"
        "/delete -PROVIDER TICKER   Remove ticker from your provider list\n"
        "/list                      Show all your tickers\n"
        "/list [-PROVIDER]          Show your tickers for a provider\n"
        "/analyze [-PROVIDER] [TICKER] [-email]   Analyze ticker, use -email to send to your verified email\n"
        "period: 1d, 5d, 1mo or 1m, 3mo or 3m, 6mo or 6m, 1y, 2y, 5y, 10y\n"
        "interval:  1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo\n"
        "<b>Email flow:</b>\n"
        "1. Register your email with /register email@example.com\n"
        "2. Check your inbox for a 6-digit code\n"
        "3. Verify with /verify CODE\n"
        "4. Use -email flag with /analyze to receive reports by email (only if verified)\n\n"
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

def generate_recommendation(recommendations: dict, indicator : str) -> str:
    """Generate trading recommendation based on technical indicators."""
    if not recommendations or indicator == "overall" and "overall" not in recommendations:
        return "Unable to generate recommendation - insufficient data"
    rec = recommendations.get(indicator, {})
    signal = rec.get("signal", "HOLD")
    reason = rec.get("reason", "No reason available")
    return f"{signal}: {reason}"

def format_analysis_text(result, technicals, fundamentals):
    recs = technicals.recommendations
    text = (f"📈 <b>{result.ticker}</b>")
    if fundamentals:
        text += (
            f" - {getattr(fundamentals, 'company_name', 'Unknown')}\n\n"
            f"💵 Price: ${getattr(fundamentals, 'current_price', 0.0):.2f}\n"
            f"🏦 P/E: {getattr(fundamentals, 'pe_ratio', 0.0):.2f}, Forward P/E: {getattr(fundamentals, 'forward_pe', 0.0):.2f}\n"
            f"💸 Market Cap: ${(getattr(fundamentals, 'market_cap', 0.0)/1e9):.2f}B\n"
            f"📊 EPS: ${getattr(fundamentals, 'earnings_per_share', 0.0):.2f}, Div Yield: {(getattr(fundamentals, 'dividend_yield', 0.0)*100):.2f}%\n\n"
        )
    else:
        text += "\n"

    text += (
        f"📉 Technical Analysis:\n"
        f"RSI: {getattr(technicals, 'rsi', 0.0):.2f}{generate_recommendation(recs, 'rsi')}\n"
        f"Stochastic %K: {getattr(technicals, 'stoch_k', 0.0):.2f}, %D: {getattr(technicals, 'stoch_d', 0.0):.2f} {generate_recommendation(recs, 'stochastic')}\n"
        f"ADX: {getattr(technicals, 'adx', 0.0):.2f}, +DI: {getattr(technicals, 'plus_di', 0.0):.2f}, -DI: {getattr(technicals, 'minus_di', 0.0):.2f} {generate_recommendation(recs, 'adx')}\n"
        f"OBV: {getattr(technicals, 'obv', 0.0):.0f} {generate_recommendation(recs, 'obv')}\n"
        f"ADR: {getattr(technicals, 'adr', 0.0):.2f}, Avg ADR: {getattr(technicals, 'avg_adr', 0.0):.2f} {generate_recommendation(recs, 'adr')}\n"
        f"MA(50): ${getattr(technicals, 'sma_50', 0.0):.2f}\n"
        f"MA(200): ${getattr(technicals, 'sma_200', 0.0):.2f}\n"
        f"MACD: {getattr(technicals, 'macd', 0.0):.4f}, Signal: {getattr(technicals, 'macd_signal', 0.0):.4f}, Hist: {getattr(technicals, 'macd_histogram', 0.0):.4f} {generate_recommendation(recs, 'macd')}\n"
        f"Trend: {getattr(technicals, 'trend', '-')}\n\n"
        f"📊 Bollinger Bands: {generate_recommendation(recs, 'bollinger')}\n"
        f"Upper: ${getattr(technicals, 'bb_upper', 0.0):.2f}\n"
        f"Middle: ${getattr(technicals, 'bb_middle', 0.0):.2f}\n"
        f"Lower: ${getattr(technicals, 'bb_lower', 0.0):.2f}\n"
        f"Width: {getattr(technicals, 'bb_width', 0.0):.4f}\n\n"
        f"🎯 Overall: {generate_recommendation(recs, 'overall')}"
    )
    return text

# Helper to analyze a ticker and return telegram action and email info (or error action)
def analyze_and_format_ticker(user_id, ticker, provider, period, interval, email_flag, get_ticker_settings, get_user_verification_status):
    telegram_actions = []
    email_info = None
    try:
        result = analyze_ticker(ticker, period=period, interval=interval, provider=provider)
        technicals = result.technicals
        fundamentals = result.fundamentals
        text = format_analysis_text(result, technicals, fundamentals)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"{ticker}_") as temp_file:
            temp_file.write(result.chart_image)
            temp_file.flush()
            chart_file = temp_file.name
        # Send the full analysis text as a separate message
        telegram_actions.append({"type": "text", "content": text})
        # Send the chart with a short caption
        short_caption = f"📊 {ticker} Analysis\n🎯 {generate_recommendation(technicals.recommendations, 'overall')}"
        telegram_actions.append({"type": "photo", "file": chart_file, "caption": short_caption, "parse_mode": "HTML"})
        if email_flag:
            status = get_user_verification_status(user_id)
            if status and status["email"] and status["verification_received"]:
                email_info = {
                    "ticker": ticker,
                    "html": text.replace('\n', '<br>'),
                    "chart_file": chart_file
                }
    except Exception as e:
        telegram_actions.append({"type": "text", "content": f"⚠️ Error analyzing {ticker}:\nPlease check if the ticker symbol is correct and try again.\nReason: {e}"})
    return telegram_actions, email_info

@dp.message(lambda message: message.text and is_valid_ticker(message.text.strip()))
async def handle_ticker(message: Message):
    ticker = message.text.strip().upper()
    email_flag = "-email" in message.text
    user_id = str(message.from_user.id)
    # Get period/interval for this ticker
    period, interval = get_ticker_settings(ticker)
    period = period or DEFAULT_PERIOD
    interval = interval or DEFAULT_INTERVAL
    # Build the info string
    info_parts = [ticker]
    if period:
        info_parts.append(f"-period={period}")
    if interval:
        info_parts.append(f"-interval={interval}")
    info_str = " ".join(info_parts)
    await message.reply(info_str)
    await message.reply(f"🔍 Analyzing {ticker}...")
    try:
        result = analyze_ticker(ticker)
        technicals = result.technicals
        fundamentals = result.fundamentals
        text = format_analysis_text(result, technicals, fundamentals)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"{ticker}_") as temp_file:
            temp_file.write(result.chart_image)
            temp_file.flush()
            chart_file = temp_file.name
        await message.reply_photo(FSInputFile(chart_file), caption=text, parse_mode="HTML")
        os.unlink(chart_file)
        if email_flag:
            status = get_user_verification_status(user_id)
            if status and status["email"] and status["verification_received"]:
                email_content = text.replace('\n', '<br>')
                notification_manager.send_notification(
                    notification_type="INFO",
                    title=f"Comprehensive Analysis for {ticker}",
                    message=email_content,
                    priority="CRITICAL",
                    data={},
                    source="telegram_screener_bot",
                    channels=["email"],
                    email_receiver=status["email"]
                )
                await message.reply(f"📧 Analysis for {ticker} sent to {status['email']}")
    except Exception as e:
        await message.reply(
            f"⚠️ Error analyzing {ticker}:\nPlease check if the ticker symbol is correct and try again.\nReason: {e}"
        )

# --- Command Logic Functions ---
def handle_add(user_id, args):
    # Logic for adding tickers, returns (success, message)
    if len(args) < 3 or not args[1].startswith('-'):
        return False, "Usage: /add -PROVIDER TICKER1[,TICKER2,...] (e.g., /add -yf AAPL,MSFT,TSLA)"
    provider = args[1][1:].lower()
    tickers = args[2].upper().split(',')
    added = []
    for ticker in tickers:
        ticker = ticker.strip()
        if ticker:
            try:
                period, interval = get_ticker_settings(ticker)
                if not period:
                    period = DEFAULT_PERIOD
                if not interval:
                    interval = DEFAULT_INTERVAL
                add_ticker(user_id, provider, ticker, period, interval)
                added.append(ticker)
            except Exception as e:
                continue
    if added:
        return True, f"✅ Added to your {provider} list: {', '.join(added)}."
    else:
        return False, "No valid tickers provided."

def handle_delete(user_id, args):
    # Logic for deleting tickers, returns (success, message)
    if len(args) < 3 or not args[1].startswith('-'):
        return False, "Usage: /delete -PROVIDER TICKER1[,TICKER2,...] (e.g., /delete -yf AAPL,MSFT,TSLA)"
    provider = args[1][1:].lower()
    tickers = args[2].upper().split(',')
    deleted = []
    for ticker in tickers:
        ticker = ticker.strip()
        if ticker:
            try:
                delete_ticker(user_id, provider, ticker)
                deleted.append(ticker)
            except Exception as e:
                continue
    if deleted:
        return True, f"❌ Removed from your {provider} list: {', '.join(deleted)}."
    else:
        return False, "No valid tickers provided."

def handle_list(user_id, provider=None):
    tickers_by_provider = list_tickers(user_id, provider)
    if not tickers_by_provider or all(not v for v in tickers_by_provider.values()):
        return False, "Your ticker list is empty. Use /add -PROVIDER TICKER to add one."
    lines = []
    for prov, tickers in tickers_by_provider.items():
        for t in tickers:
            period = t.get('period') or DEFAULT_PERIOD
            interval = t.get('interval') or DEFAULT_INTERVAL
            lines.append(f"{prov.upper()}: {t.get('ticker')} {interval} {period}")
    return True, "Your tickers:\n" + "\n".join(lines)

def handle_register(telegram_id, email):
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return False, "Usage: /register email@example.com"
    code = f"{random.randint(100000, 999999)}"
    sent_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    set_user_email(telegram_id, email, code, sent_time)
    subject = "e-Trading Email Verification"
    body = f"""
    <h2>e-Trading Email Verification</h2>
    <p>Your verification code is: <b>{code}</b></p>
    <p>Enter this code in Telegram using <b>/verify {code}</b> within 1 hour to verify your email.</p>
    """
    return True, (email, code, subject, body)

def handle_info(telegram_id):
    status = get_user_verification_status(telegram_id)
    if not status or not status["email"]:
        return False, "No email registered. Use /register email@example.com to set your email."
    verified = bool(status["verification_received"])
    reply = f"<b>Email:</b> {status['email']}\n"
    reply += f"<b>Verified:</b> {'✅' if verified else '❌'}\n"
    reply += f"<b>Verification sent:</b> {status['verification_sent']}\n"
    reply += f"<b>Verification received:</b> {status['verification_received'] or '-'}"
    return True, reply

def handle_verify(telegram_id, code):
    db_code, sent_time = get_user_verification_code(telegram_id)
    if not db_code or not sent_time:
        return False, "No verification code found. Please register your email first with /register."
    sent_dt = datetime.strptime(sent_time, "%Y-%m-%d %H:%M:%S")
    # Make sent_dt timezone-aware if naive
    if sent_dt.tzinfo is None:
        sent_dt = sent_dt.replace(tzinfo=timezone.utc)
    if code != db_code:
        return False, "Invalid verification code."
    if datetime.now(timezone.utc) > sent_dt + timedelta(hours=1):
        return False, "Verification code expired. Please re-register your email."
    set_user_verified(telegram_id, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    return True, "✅ Email verified successfully! You can now use -email flag to receive reports by email."

def parse_analyze_parameters(user_id, message_text, get_ticker_settings):
    """
    Parses the analyze command parameters and returns:
    - pairs: list of (provider, ticker, period, interval)
    - telegram_actions: list of usage/help messages if parsing fails or no tickers are found
    If pairs is not empty, telegram_actions is empty, and vice versa.
    """
    args = message_text.upper().split()
    ticker = None
    provider = None
    email_flag = False
    period = None
    interval = None
    telegram_actions = []
    pairs = []

    for arg in args[1:]:
        if arg == "-EMAIL":
            email_flag = True
        elif arg.startswith("-") and arg.upper() not in ("-EMAIL",):
            prov_candidate = arg[1:].upper()
            if prov_candidate in ["YF", "BNC"]:
                provider = prov_candidate
            elif arg.upper().startswith("-PERIOD="):
                period = arg.split("=", 1)[1].lower()
            elif arg.upper().startswith("-INTERVAL="):
                interval = arg.split("=", 1)[1].lower()
            else:
                telegram_actions.append({"type": "text", "content": "Usage: /analyze [-PROVIDER] [TICKER] [-email]"})
                return [], telegram_actions, email_flag
        elif ticker is None:
            ticker = arg.upper()

    # Build pairs to process
    if ticker:
        # If provider is not specified, try to infer it (case-insensitive)
        if not provider:
            found = False
            for prov in ["YF", "BNC"]:
                p, i = get_ticker_settings(user_id, prov, ticker)
                if p or i:
                    provider = prov
                    found = True
                    break
            if not found:
                telegram_actions.append({"type": "text", "content": "Usage: /analyze [-PROVIDER [TICKER]] [-email]"})
                return [], telegram_actions, email_flag
        if not provider:
            provider = "yf"
        pairs = [(provider, ticker, period, interval)]
    else:
        if provider:
            provider = provider.lower()
        # all_tickers_with_providers_for_status returns list of (provider, ticker)
        all_pairs = all_tickers_with_providers_for_status(user_id, provider)
        if not all_pairs:
            telegram_actions.append({"type": "text", "content": "No tickers found. Use /add to add tickers first."})
            return [], telegram_actions, email_flag
        # For each, get period/interval
        for prov, tick in all_pairs:
            p, i = get_ticker_settings(user_id, prov, tick)
            pairs.append((prov, tick, period or p, interval or i))

    return pairs, telegram_actions, email_flag

def analyze_command_core(
    user_id,
    message_text,
    get_ticker_settings,
    get_user_verification_status
):
    try:
        pairs, telegram_actions, email_flag = parse_analyze_parameters(user_id, message_text, get_ticker_settings)
        email_tickers = []
        if telegram_actions:
            return {"telegram_actions": telegram_actions, "email_info": None}
        for prov, tick, period, interval in pairs:
            # Determine period/interval with fallback
            p, i = get_ticker_settings(user_id, prov, tick)
            use_period = period or p or DEFAULT_PERIOD
            use_interval = interval or i or DEFAULT_INTERVAL
            telegram_actions.append({"type": "text", "content": f"🔍 Analyzing {tick} (provider={prov}, period={use_period}, interval={use_interval})..."})
            actions, email_ticker = analyze_and_format_ticker(
                user_id, tick, prov, use_period, use_interval, email_flag, get_ticker_settings, get_user_verification_status
            )
            telegram_actions.extend(actions)
            if email_flag and email_ticker:
                email_tickers.append(email_ticker)
        email_info = None
        if email_flag and email_tickers:
            status = get_user_verification_status(user_id)
            if status and status["email"] and status["verification_received"]:
                subj = f"Comprehensive Analysis for {pairs[0][1]}" if len(pairs) == 1 else f"Your Screener Status Report - {len(email_tickers)} Tickers Analyzed"
                email_info = {
                    "to": status["email"],
                    "subject": subj,
                    "tickers": email_tickers
                }
        logger.info(f"Telegram actions to send: {telegram_actions}")
        return {"telegram_actions": telegram_actions, "email_info": email_info}
    except Exception as e:
        logger.error("Internal error: %s", e, exc_info=True)
        telegram_actions = [{"type": "text", "content": f"Internal error: {e}"}]
        return {"telegram_actions": telegram_actions, "email_info": None}

@dp.message(Command("analyze"))
async def analyze_command(message: Message):
    logger.info("Entered analyze handler")

    user_id = str(message.from_user.id)
    args = message.text.split()

    logger.info("analyze args: %s", args)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            functools.partial(
                analyze_command_core,
                user_id,
                message.text,
                get_ticker_settings,
                get_user_verification_status
            )
        )
        files_to_delete = []
        for action in result["telegram_actions"]:
            if action["type"] == "text":
                await message.reply(sanitize_for_telegram(action["content"]))
            elif action["type"] == "photo":
                await bot.send_photo(
                    chat_id=message.chat.id,
                    photo=FSInputFile(action["file"]),
                    caption=action["caption"],
                    parse_mode="HTML",
                )
                files_to_delete.append(action["file"])
        if result["email_info"]:
            email_body = "<h2>📊 Your Screener Status Report</h2>"
            attachments = []
            for ticker_info in result["email_info"]["tickers"]:
                email_body += f"<h3>{ticker_info['ticker']}</h3>{ticker_info['html']}<br><br>"
                attachments.append(ticker_info["chart_file"])
            await notification_manager.send_notification(
                notification_type="INFO",
                title=result["email_info"]["subject"],
                message=email_body,
                priority="NORMAL",
                data={},
                source="telegram_screener_bot",
                channels=["email"],
                attachments=attachments,
                email_receiver=result["email_info"]["to"]
            )
            await message.reply(f"📧 Status report sent to {result['email_info']['to']}")
            files_to_delete.extend(attachments)
        for f in set(files_to_delete):
            try:
                os.unlink(f)
            except Exception:
                pass
    except Exception as e:
        logger.error("Error in analyze_command: %s", e, exc_info=True)
        await message.reply(f"Error: {e}")

@dp.message(Command("add"))
async def my_add(message: Message):
    logger.info("Entered add handler")
    user_id = str(message.from_user.id)
    args = message.text.split()
    logger.info("add args: %s", args)
    success, reply = handle_add(user_id, args)
    await message.reply(reply)

@dp.message(Command("delete"))
async def my_delete(message: Message):
    logger.info("Entered delete handler")
    user_id = str(message.from_user.id)
    args = [a.strip() for a in re.split(r'\s+', message.text) if a.strip()]
    logger.info("delete args: %s", args)
    success, reply = handle_delete(user_id, args)
    await message.reply(reply)

@dp.message(Command("list"))
async def my_list(message: Message):
    logger.info("Entered list handler")
    user_id = str(message.from_user.id)
    args = message.text.split()
    logger.info("list args: %s", args)
    provider = None
    if len(args) == 2 and args[1].startswith('-'):
        provider = args[1][1:].lower()
    success, reply = handle_list(user_id, provider)
    await message.reply(reply)

@dp.message(Command("register"))
async def my_register(message: Message):
    logger.info("Entered register handler")
    args = message.text.split()
    if len(args) != 2:
        await message.reply("Usage: /register email@example.com")
        return
    email = args[1].strip()
    telegram_id = str(message.from_user.id)
    success, result = handle_register(telegram_id, email)
    if not success:
        await message.reply(result)
        return
    email, code, subject, body = result
    try:
        if notification_manager:
            await notification_manager.send_notification(
                notification_type=NotificationType.INFO,
                title=subject,
                message=body,
                priority=NotificationPriority.NORMAL,
                data={},
                source="telegram_screener_bot",
                channels=["email"],
                email_receiver=email
            )
        await message.reply(f"Verification code sent to {email}. Please check your inbox and use /verify CODE in Telegram.")
    except Exception as e:
        logger.error("Failed to send verification email: %s", e, exc_info=True)
        await message.reply(f"Failed to send verification email: {e}")

@dp.message(Command("info"))
async def my_info(message: Message):
    logger.info("Entered info handler")
    telegram_id = str(message.from_user.id)
    success, reply = handle_info(telegram_id)
    await message.reply(reply, parse_mode="HTML")

@dp.message(Command("verify"))
async def my_verify(message: Message):
    logger.info("Entered verify handler")
    args = message.text.split()
    if len(args) != 2 or not args[1].isdigit():
        await message.reply("Usage: /verify CODE (6 digits)")
        return
    code = args[1]
    telegram_id = str(message.from_user.id)
    success, reply = handle_verify(telegram_id, code)
    await message.reply(reply)

# Fallback handler for unknown commands
@dp.message(lambda message: message.text and message.text.startswith("/"))
async def unknown_command(message: Message):
    await send_welcome(message)

# Ensure handle_analyze is defined at the module level and importable
__all__ = [
    'handle_add', 'handle_delete', 'handle_list',
    'handle_register', 'handle_info', 'handle_verify', 'analyze_command_core',
    # ... other exports ...
]

def sanitize_for_telegram(html: str) -> str:
    html = html.replace('<h2>', '<b>').replace('</h2>', '</b>')
    html = html.replace('<h3>', '<b>').replace('</h3>', '</b>')
    html = html.replace('<br>', '\n')
    # Optionally strip any other unsupported tags
    return html

if __name__ == "__main__":
    asyncio.run(main())

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
import tempfile
import re
from datetime import datetime, timedelta, timezone
import random
import functools

from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import FSInputFile, Message
from src.notification.logger import setup_logger
from src.screener.telegram.combine import analyze_ticker, format_comprehensive_analysis
from src.screener.telegram.screener_db import (
    add_ticker, delete_ticker, list_tickers, all_tickers_with_providers_for_status, set_user_email,
    get_user_verification_status, get_user_verification_code, set_user_verified, get_ticker_settings
)
from src.notification.async_notification_manager import initialize_notification_manager, NotificationType, NotificationPriority
from src.screener.telegram.chart import generate_chart
from src.screener.telegram.technicals import calculate_technicals_from_df
from src.screener.telegram.technicals import format_technical_analysis

from config.donotshare.donotshare import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SMTP_USER

# Set up logger using the telegram_bot configuration
logger = setup_logger("telegram_bot")

# Initialize async notification manager (for email/telegram notifications)
notification_manager = None
try:
    notification_manager = asyncio.get_event_loop().run_until_complete(
        initialize_notification_manager(
            telegram_token=TELEGRAM_BOT_TOKEN,
            telegram_chat_id=TELEGRAM_CHAT_ID,
            email_sender=SMTP_USER,
            email_receiver=None  # Will be set per user
        )
    )
except Exception as e:
    logger.error("Notification manager not initialized: %s", e, exc_info=True)

if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN environment variable is not set")
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is not set")

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

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
        "<b>Email flow:</b>\n"
        "1. Register your email with /my-register email@example.com\n"
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

@dp.message(lambda message: message.text and is_valid_ticker(message.text.strip()))
async def handle_ticker(message: Message):
    ticker = message.text.strip().upper()
    logger.info("User %s requested analysis for %s", message.from_user.id, ticker)
    await message.reply(f"🔍 Analyzing {ticker}...")

    try:
        result = analyze_ticker(ticker)
        logger.info("Successfully analyzed %s", ticker)

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
        logger.error("Error analyzing %s", ticker, exc_info=True)
        await message.reply(
            f"⚠️ Error analyzing {ticker}:\n"
            f"Please check if the ticker symbol is correct and try again."
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
    import re
    from src.screener.telegram.telegram_screener_bot import get_user_verification_code, set_user_email
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
        from datetime import timezone
        sent_dt = sent_dt.replace(tzinfo=timezone.utc)
    if code != db_code:
        return False, "Invalid verification code."
    if datetime.now(timezone.utc) > sent_dt + timedelta(hours=1):
        return False, "Verification code expired. Please re-register your email."
    set_user_verified(telegram_id, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    return True, "✅ Email verified successfully! You can now use -email flag to receive reports by email."

def analyze_command_core(
    user_id,
    message_text,
    notification_manager,
    get_ticker_settings,
    format_comprehensive_analysis,
    get_user_verification_status,
    bot=None
):
    import re
    import tempfile
    import asyncio
    args = message_text.split()
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
    actions = []
    if ticker:
        if not provider:
            for prov in ["yf", "bnc"]:
                p, i = get_ticker_settings(user_id, prov, ticker)
                if p or i:
                    provider = prov
                    break
        if not provider:
            provider = "yf"
        db_period, db_interval = get_ticker_settings(user_id, provider, ticker)
        if not period:
            period = db_period or DEFAULT_PERIOD
        if not interval:
            interval = db_interval or DEFAULT_INTERVAL
        actions.append({"type": "text", "content": f"🔍 Analyzing {ticker} (provider={provider}, period={period}, interval={interval})..."})
        try:
            result = analyze_ticker(ticker, period=period, interval=interval, provider=provider)
            text = format_comprehensive_analysis(ticker, result.technicals, result.fundamentals)
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"{ticker}_") as temp_file:
                    temp_file.write(result.chart_image)
                    temp_file.flush()
                    chart_file = temp_file.name
                actions.append({"type": "photo", "file": chart_file, "caption": f"📊 {ticker} Analysis\n🎯 {result.recommendation}"})
            except Exception as e:
                actions.append({"type": "text", "content": f"No chart available for {ticker}. Reason: {e}"})
            if email_flag:
                status = get_user_verification_status(user_id)
                if not status or not status["email"]:
                    actions.append({"type": "text", "content": "No email registered. Use /register to set your email."})
                    return actions
                if not status["verification_received"]:
                    actions.append({"type": "text", "content": "Your email is not verified. Use /verify CODE to verify."})
                    return actions
                email = status["email"]
                email_content = text
                if notification_manager:
                    notification_manager.send_notification(
                        notification_type="INFO",
                        title=f"Comprehensive Analysis for {ticker}",
                        message=email_content,
                        priority="NORMAL",
                        data={},
                        source="telegram_screener_bot",
                        channels=["email"],
                    )
                actions.append({"type": "text", "content": f"📧 Analysis for {ticker} sent to {email}"})
        except Exception as e:
            actions.append({"type": "text", "content": f"⚠️ Error analyzing {ticker}:\nPlease check if the ticker symbol is correct and try again.\nReason: {e}"})
        return actions
    # If no ticker, analyze all tickers in user's list (optionally filtered by provider)
    pairs = all_tickers_with_providers_for_status(user_id, provider)
    if not pairs:
        actions.append({"type": "text", "content": "No tickers found. Use /add to add tickers first."})
        return actions
    email_body = []
    chart_files = []
    for prov, ticker in pairs:
        p, i = get_ticker_settings(user_id, prov, ticker)
        use_period = period or p or DEFAULT_PERIOD
        use_interval = interval or i or DEFAULT_INTERVAL
        actions.append({"type": "text", "content": f"🔍 Analyzing {ticker} (provider={prov}, period={use_period}, interval={use_interval})..."})
        try:
            result = analyze_ticker(ticker, period=use_period, interval=use_interval, provider=prov)
            text = format_comprehensive_analysis(ticker, result.technicals, result.fundamentals)
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix=f"{ticker}_") as temp_file:
                    temp_file.write(result.chart_image)
                    temp_file.flush()
                    chart_files.append(temp_file.name)
                actions.append({"type": "photo", "file": chart_files[-1], "caption": f"📊 {ticker} Analysis\n🎯 {result.recommendation}"})
            except Exception as e:
                actions.append({"type": "text", "content": f"No chart available for {ticker}. Reason: {e}"})
            email_body.append(text)
        except Exception as e:
            actions.append({"type": "text", "content": f"⚠️ Error analyzing {ticker}:\nPlease check if the ticker symbol is correct and try again.\nReason: {e}"})
    if email_flag:
        status = get_user_verification_status(user_id)
        if not status or not status["email"]:
            actions.append({"type": "text", "content": "No email registered. Use /register to set your email."})
            return actions
        if not status["verification_received"]:
            actions.append({"type": "text", "content": "Your email is not verified. Use /verify CODE to verify."})
            return actions
        email = status["email"]
        email_content = f"""
        <h2>📊 Your Screener Status Report</h2>
        <p>Analysis completed for {len(pairs)} ticker(s)</p>
        <hr>
        """
        email_content += "<br><br>".join(email_body)
        if notification_manager:
            notification_manager.send_notification(
                notification_type="INFO",
                title=f"Your Screener Status Report - {len(pairs)} Tickers Analyzed",
                message=email_content,
                priority="NORMAL",
                data={},
                source="telegram_screener_bot",
                channels=["email"],
            )
        actions.append({"type": "text", "content": f"📧 Status report sent to {email} with {len(chart_files)} charts"})
    return actions

@dp.message(Command("analyze"))
async def analyze_command(message: Message):
    user_id = str(message.from_user.id)
    loop = asyncio.get_event_loop()
    actions = await loop.run_in_executor(
        None,
        functools.partial(
            analyze_command_core,
            user_id,
            message.text,
            notification_manager,
            get_ticker_settings,
            format_comprehensive_analysis,
            get_user_verification_status,
            bot
        )
    )
    for action in actions:
        if action["type"] == "text":
            await message.reply(action["content"])
        elif action["type"] == "photo":
            await bot.send_photo(
                chat_id=message.chat.id,
                photo=FSInputFile(action["file"]),
                caption=action["caption"],
                parse_mode="HTML",
            )
            import os
            os.unlink(action["file"])

@dp.message(Command("add"))
async def my_add(message: Message):
    print("[DEBUG] Entered add handler", flush=True)
    logger.info("Entered add handler")
    user_id = str(message.from_user.id)
    args = message.text.split()
    print(f"[DEBUG] add args: {args}", flush=True)
    logger.info("add args: %s", args)
    success, reply = handle_add(user_id, args)
    await message.reply(reply)

@dp.message(Command("delete"))
async def my_delete(message: Message):
    print("[DEBUG] Entered delete handler", flush=True)
    logger.info("Entered delete handler")
    user_id = str(message.from_user.id)
    args = [a.strip() for a in re.split(r'\s+', message.text) if a.strip()]
    print(f"[DEBUG] delete args: {args}", flush=True)
    logger.info("delete args: %s", args)
    success, reply = handle_delete(user_id, args)
    await message.reply(reply)

@dp.message(Command("list"))
async def my_list(message: Message):
    print("[DEBUG] Entered list handler", flush=True)
    logger.info("Entered list handler")
    user_id = str(message.from_user.id)
    args = message.text.split()
    print(f"[DEBUG] list args: {args}", flush=True)
    logger.info("list args: %s", args)
    provider = None
    if len(args) == 2 and args[1].startswith('-'):
        provider = args[1][1:].lower()
    success, reply = handle_list(user_id, provider)
    await message.reply(reply)

@dp.message(Command("register"))
async def my_register(message: Message):
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
            )
        await message.reply(f"Verification code sent to {email}. Please check your inbox and use /verify CODE in Telegram.")
    except Exception as e:
        logger.error("Failed to send verification email: %s", e, exc_info=True)
        await message.reply(f"Failed to send verification email: {e}")

@dp.message(Command("info"))
async def my_info(message: Message):
    telegram_id = str(message.from_user.id)
    success, reply = handle_info(telegram_id)
    await message.reply(reply, parse_mode="HTML")

@dp.message(Command("verify"))
async def my_verify(message: Message):
    args = message.text.split()
    if len(args) != 2 or not args[1].isdigit():
        await message.reply("Usage: /verify CODE (6 digits)")
        return
    code = args[1]
    telegram_id = str(message.from_user.id)
    success, reply = handle_verify(telegram_id, code)
    await message.reply(reply)

async def main():
    logger.info("Starting ticker analyzer bot")
    await dp.start_polling(bot)

# Ensure handle_analyze is defined at the module level and importable
__all__ = [
    'handle_add', 'handle_delete', 'handle_list',
    'handle_register', 'handle_info', 'handle_verify', 'analyze_command_core',
    # ... other exports ...
]

if __name__ == "__main__":
    asyncio.run(main())

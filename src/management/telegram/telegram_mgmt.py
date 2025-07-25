import importlib
import os

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from config.donotshare.donotshare import TELEGRAM_BOT_TOKEN
from src.model.config_models import TradingBotConfig, BrokerType, DataSourceType, StrategyType

"""
Telegram Bot Management Module
-----------------------------

This module implements a Telegram bot for managing trading bots via chat commands. It allows users to start, stop, monitor, and backtest trading strategies directly from Telegram. The bot dynamically loads strategy classes and manages running bots in memory.

Main Features:
- Start and stop trading bots with /start_bot and /stop_bot
- Check status of all running bots with /status
- View recent logs for a strategy with /log
- Run backtests with /backtest
- Integrates with the trading bot infrastructure for real-time control

Commands:
- /start_bot <strategy>
- /stop_bot <strategy>
- /status
- /log <strategy>
- /backtest <strategy> <ticker> <tf>
"""

# In-memory registry of running bots
running_bots = {}

# Set up logging
_logger.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=_logger.INFO
)


async def start_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 1:
        await update.message.reply_text("Usage: /start_bot <strategy>")
        return
    strategy_name = context.args[0]
    if strategy_name in running_bots:
        await update.message.reply_text(f"Bot for {strategy_name} is already running.")
        return
    try:
        # Create a simple config using the new Pydantic models
        config = TradingBotConfig(
            bot_id=f"telegram_{strategy_name}",
            symbol="BTCUSDT",
            broker_type=BrokerType.BINANCE_PAPER,
            data_source=DataSourceType.BINANCE,
            strategy_type=StrategyType.CUSTOM,
            description=f"Telegram-managed bot for {strategy_name}",
            initial_balance=1000.0,
            risk_per_trade=0.01,
            max_open_trades=3
        )

        # Dynamically import the bot class
        bot_module = importlib.import_module(f"src.trading.{strategy_name}_bot")
        bot_class = getattr(
            bot_module,
            "".join([w.capitalize() for w in strategy_name.split("_")]) + "Bot",
        )

        # Create bot instance with config file name
        config_filename = f"telegram_{strategy_name}.json"
        bot_instance = bot_class(config_filename)
        running_bots[strategy_name] = bot_instance

        # Start the bot in a background thread
        import threading
        t = threading.Thread(target=bot_instance.run, daemon=True)
        t.start()

        await update.message.reply_text(f"Started bot for {strategy_name} with config: {config.bot_id}")
    except Exception as e:
        _logger.error("Failed to start bot: %s", e, exc_info=True)
        await update.message.reply_text(f"Failed to start bot: {e}")


async def stop_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 1:
        await update.message.reply_text("Usage: /stop_bot <strategy>")
        return
    strategy_name = context.args[0]
    bot = running_bots.get(strategy_name)
    if not bot:
        await update.message.reply_text(f"No running bot for {strategy_name}.")
        return
    try:
        bot.stop()
        del running_bots[strategy_name]
        await update.message.reply_text(f"Stopped bot for {strategy_name}.")
    except Exception as e:
        _logger.error("Failed to stop bot: %s", e, exc_info=True)
        await update.message.reply_text(f"Failed to stop bot: {e}")


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not running_bots:
        await update.message.reply_text("No bots are currently running.")
        return
    status_lines = [f"{name}: running" for name in running_bots.keys()]
    await update.message.reply_text("\n".join(status_lines))


async def log(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 1:
        await update.message.reply_text("Usage: /log <strategy>")
        return
    strategy_name = context.args[0]
    log_file = f"logs/{strategy_name}.log"
    if not os.path.exists(log_file):
        await update.message.reply_text(f"No log file found for {strategy_name}.")
        return
    with open(log_file, "r") as f:
        lines = f.readlines()[-20:]
    await update.message.reply_text("".join(lines) or "No recent logs.")


async def backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 3:
        await update.message.reply_text("Usage: /backtest <strategy> <ticker> <tf>")
        return
    strategy, ticker, tf = context.args[:3]
    # Stub: Replace with actual backtest logic
    await update.message.reply_text(
        f"Backtesting {strategy} on {ticker} ({tf})... [stub]"
    )


def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start_bot", start_bot))
    app.add_handler(CommandHandler("stop_bot", stop_bot))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("log", log))
    app.add_handler(CommandHandler("backtest", backtest))
    print("Telegram management bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()

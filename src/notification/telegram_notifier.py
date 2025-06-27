"""
Sends trade and error notifications to Telegram using a bot, for real-time trading alerts.

This module defines TelegramNotifier, which can send trade entries, updates, and error alerts to a configured Telegram chat for trading bots.
"""

import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from datetime import datetime
from typing import Any, Dict, Optional

from src.notification.logger import _logger
from telegram import Bot
from telegram.error import TelegramError

from config.donotshare.donotshare import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        """
        Initialize the Telegram trade notifier.

        Args:
            token (str): Telegram bot token
            chat_id (str): Chat ID to send notifications to
        """
        self.logger = _logger
        self.bot = Bot(token=token)
        self.chat_id = chat_id
        # self.logger.info("Telegram notifier initialized")

    def send_trade_notification(self, trade_data: Dict[str, Any]) -> bool:
        """
        Send a trade notification to Telegram (synchronous).

        Args:
            trade_data (Dict[str, Any]): Dictionary containing trade information
                Required keys:
                - symbol: Trading pair symbol
                - side: 'BUY' or 'SELL'
                - entry_price: Entry price
                - tp_price: Take profit price
                - sl_price: Stop loss price
                - quantity: Trade quantity
                - timestamp: Trade timestamp
                Optional keys:
                - reason: Reason for the trade
                - rsi: RSI value
                - bb_position: Position relative to Bollinger Bands

        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        try:
            # Format the message
            message = self._format_trade_message(trade_data)

            # Send the message
            self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode="HTML")

            self.logger.info(
                f"Trade notification sent successfully for {trade_data['symbol']}"
            )
            return True

        except TelegramError as e:
            self.logger.error(f"Failed to send trade notification: {e}")
            return False

    def send_trade_update(self, trade_data: Dict[str, Any]) -> bool:
        """
        Send a trade update notification (TP/SL hit, trade closed) to Telegram (synchronous).

        Args:
            trade_data (Dict[str, Any]): Dictionary containing trade update information
                Required keys:
                - symbol: Trading pair symbol
                - side: 'BUY' or 'SELL'
                - entry_price: Entry price
                - exit_price: Exit price
                - pnl: Profit/Loss percentage
                - exit_type: 'TP' or 'SL'
                - timestamp: Exit timestamp

        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        try:
            # Format the message
            message = self._format_trade_update_message(trade_data)

            # Send the message
            self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode="HTML")

            self.logger.info(
                f"Trade update notification sent successfully for {trade_data['symbol']}"
            )
            return True

        except TelegramError as e:
            self.logger.error(f"Failed to send trade update notification: {e}")
            return False

    def send_error_notification(self, error_message: str) -> bool:
        """
        Send an error notification to Telegram (synchronous).

        Args:
            error_message (str): Error message to send

        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        try:
            message = f"⚠️ <b>Error Alert</b>\n\n{error_message}"

            self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode="HTML")

            self.logger.info("Error notification sent successfully")
            return True

        except TelegramError as e:
            self.logger.error(f"Failed to send error notification: {e}")
            return False

    def _format_trade_message(self, trade_data: Dict[str, Any]) -> str:
        """Format trade entry message"""
        message = [
            f"🔔 <b>New Trade Alert</b>",
            f"Symbol: {trade_data['symbol']}",
            f"Side: {'🟢 BUY' if trade_data['side'] == 'BUY' else '🔴 SELL'}",
            f"Entry Price: {trade_data['entry_price']:.8f}",
            f"Take Profit: {trade_data['tp_price']:.8f}",
            f"Stop Loss: {trade_data['sl_price']:.8f}",
            f"Quantity: {trade_data['quantity']:.8f}",
            f"Time: {trade_data['timestamp']}",
        ]

        # Add optional information if available
        if "reason" in trade_data:
            message.append(f"Reason: {trade_data['reason']}")
        if "rsi" in trade_data:
            message.append(f"RSI: {trade_data['rsi']:.2f}")
        if "bb_position" in trade_data:
            message.append(f"BB Position: {trade_data['bb_position']}")

        return "\n".join(message)

    def _format_trade_update_message(self, trade_data: Dict[str, Any]) -> str:
        """Format trade update message, robust to missing 'side' and 'symbol'"""
        pnl = trade_data.get("pnl", 0)
        pnl_emoji = "🟢" if pnl > 0 else "🔴"
        # Try to get 'side', or infer from 'type' or 'direction'
        side = trade_data.get("side")
        if not side:
            t = (
                trade_data.get("type", "").lower()
                or trade_data.get("direction", "").lower()
            )
            if t == "long":
                side = "BUY"
            elif t == "short":
                side = "SELL"
            else:
                side = "UNKNOWN"
        side_str = (
            "🟢 BUY"
            if side == "BUY"
            else ("🔴 SELL" if side == "SELL" else "❓ UNKNOWN")
        )
        message = [
            f"📊 <b>Trade Update</b>",
            f"Symbol: {trade_data.get('symbol', 'UNKNOWN')}",
            f"Side: {side_str}",
            f"Entry Price: {trade_data.get('entry_price', 0):.8f}",
            f"Exit Price: {trade_data.get('exit_price', 0):.8f}",
            f"Exit Type: {trade_data.get('exit_type', 'UNKNOWN')}",
            f"PnL: {pnl_emoji} {pnl:.2f}%",
            f"Time: {trade_data.get('timestamp', 'UNKNOWN')}",
        ]
        return "\n".join(message)

    async def send_trade_notification_async(self, trade_data: Dict[str, Any]) -> bool:
        """
        Asynchronously send a trade notification to Telegram.
        Args:
            trade_data (Dict[str, Any]): Dictionary containing trade information
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        try:
            message = self._format_trade_message(trade_data)
            await self.bot.send_message(
                chat_id=self.chat_id, text=message, parse_mode="HTML"
            )
            self.logger.info(
                f"Trade notification sent successfully for {trade_data['symbol']}"
            )
            return True
        except TelegramError as e:
            self.logger.error(f"Failed to send trade notification (async): {e}")
            return False

    async def send_trade_update_async(self, trade_data: Dict[str, Any]) -> bool:
        """
        Asynchronously send a trade update notification to Telegram.
        Args:
            trade_data (Dict[str, Any]): Dictionary containing trade update information
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        try:
            message = self._format_trade_update_message(trade_data)
            await self.bot.send_message(
                chat_id=self.chat_id, text=message, parse_mode="HTML"
            )
            self.logger.info(
                f"Trade update notification sent successfully for {trade_data.get('symbol', 'UNKNOWN')}"
            )
            return True
        except TelegramError as e:
            self.logger.error(f"Failed to send trade update notification (async): {e}")
            return False

    async def send_error_notification_async(self, error_message: str) -> bool:
        """
        Asynchronously send an error notification to Telegram.
        Args:
            error_message (str): Error message to send
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        try:
            message = f"⚠️ <b>Error Alert</b>\n\n{error_message}"
            await self.bot.send_message(
                chat_id=self.chat_id, text=message, parse_mode="HTML"
            )
            self.logger.info("Error notification sent successfully (async)")
            return True
        except TelegramError as e:
            self.logger.error(f"Failed to send error notification (async): {e}")
            return False

    async def send_message_async(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Asynchronously send a message to the configured chat.

        Args:
            message: The message to send
            parse_mode: The parse mode for the message (HTML or Markdown)

        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        try:
            if not self.bot or not self.chat_id:
                self.logger.warning("Telegram bot not configured")
                return False

            # Add timestamp to message
            timestamp = datetime.now()
            formatted_message = (
                f"🤖 Trading Bot\n"
                f"⏰ {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{message}"
            )

            await self.bot.send_message(
                chat_id=self.chat_id, text=formatted_message, parse_mode=parse_mode
            )
            return True

        except Exception as e:
            self.logger.error(f"Error sending Telegram message (async): {str(e)}")
            return False


def create_notifier() -> Optional[TelegramNotifier]:
    """
    Create a TelegramNotifier instance using environment variables.

    Returns:
        Optional[TelegramNotifier]: TelegramNotifier instance if credentials are available,
                               None otherwise
    """
    token = TELEGRAM_BOT_TOKEN
    chat_id = TELEGRAM_CHAT_ID
    logger = _logger

    if not token or not chat_id:
        logger.warning("Telegram credentials not found in environment variables")
        return None

    try:
        return TelegramNotifier(token=token, chat_id=chat_id)
    except Exception as e:
        logger.error(f"Failed to create TelegramNotifier: {e}")
        return None


def send_telegram_alert(message: str):
    try:
        notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        notifier.send_error_notification(message)
        _logger.info(f"System alert sent to Telegram: {message}")
    except Exception as e:
        _logger.error(f"Failed to send system alert to Telegram: {e}")


if __name__ == "__main__":
    n = create_notifier()
    asyncio.run(
        n.send_trade_notification_async(
            {
                "rsi": 50,
                "symbol": "BTCUSDT",
                "side": "BUY",
                "entry_price": 1000,
                "tp_price": 2000,
                "sl_price": 500,
                "quantity": 1,
                "timestamp": datetime.now(),
            }
        )
    )

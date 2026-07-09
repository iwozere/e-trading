"""
Strategy Instance
-----------------
Represents a single running instance of a trading strategy with its own
broker connection, data feed, and Backtrader engine.
"""

import asyncio
import threading
import time
from datetime import UTC, datetime
from typing import Optional, Any, Dict

from src.data.db.services.trading_service import trading_service
from src.data.feed.data_feed_factory import DataFeedFactory
from src.notification.logger import setup_logger
from src.notification.service.client import MessagePriority, MessageType, NotificationServiceClient
from src.trading.base_trading_bot import BaseTradingBot, _run_async, is_file_based_simulation_config
from src.trading.broker.backtrader_availability import BACKTRADER_AVAILABLE
from src.trading.broker.broker_factory import get_broker
from src.trading.strategy_handler import strategy_handler
from src.trading.trading_notification_recipient import get_trading_bot_notification_recipient

_logger = setup_logger(__name__)


class StrategyInstance:
    """Represents a single strategy instance with its own configuration and broker."""

    def __init__(
        self,
        instance_id: str,
        config: Dict[str, Any],
        notification_client: NotificationServiceClient | None = None,
        trade_repository: Optional[Any] = None,
    ):
        """Initialize strategy instance."""
        self.instance_id = instance_id
        self.config = config
        self.name = config.get("name", f"Strategy_{instance_id}")
        self.notification_client = notification_client
        self.trade_repository = trade_repository

        # Core components
        self.broker = None
        self.trading_bot: Optional[BaseTradingBot] = None
        self.data_feed: Optional[Any] = None
        self.cerebro: Optional[Any] = None

        # Status tracking
        self.status = "stopped"
        self.start_time: Optional[datetime] = None
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.is_running = False
        self.should_stop = False

        # Heartbeat tracking
        self.last_heartbeat = None
        self.heartbeat_interval = 60  # seconds

        # Threading for monitoring
        self.monitor_thread = None
        self.heartbeat_thread = None

    async def start(self) -> bool:
        """Start the strategy instance with full Backtrader integration."""
        try:
            _logger.info("Starting strategy instance: %s", self.name)

            # Create broker
            broker_config = self.config["broker"]
            broker_type = broker_config.get("type", "").lower()

            if broker_type == "backtrader":
                _logger.info("Using Backtrader's built-in broker (no custom broker needed)")
                self.broker = None
            else:
                self.broker = get_broker(broker_config)
                if not self.broker:
                    raise Exception("Failed to create broker")

                _logger.info(
                    "Created broker for %s: %s (mode: %s)",
                    self.name,
                    broker_config.get("type"),
                    broker_config.get("trading_mode"),
                )

            # Get strategy class
            strategy_config = self.config["strategy"]
            strategy_class = self._get_strategy_class(strategy_config["type"])
            _logger.info("Loaded strategy class: %s", strategy_class.__name__)

            # Create data feed
            if not self._create_data_feed():
                raise RuntimeError("Failed to create data feed")

            # Create the trading bot
            if not await self._create_trading_bot():
                raise RuntimeError("Failed to create trading bot")

            # Setup Backtrader
            if not self._setup_backtrader(strategy_class):
                raise RuntimeError("Failed to setup Backtrader")

            # Start monitoring threads
            self._start_threads()

            # Set status
            self.status = "running"
            self.start_time = datetime.now(UTC)
            self.is_running = True
            self.error_count = 0
            self.last_error = None

            # Update DB
            try:
                trading_service.update_bot_status(self.instance_id, "running", started_at=self.start_time)
            except Exception as e:
                _logger.warning("Failed to update bot status in DB: %s", e)

            # Start processing loops.
            # When cerebro is active, Backtrader drives all signal generation.
            # Signals are forwarded to execute_trade() synchronously inside the
            # Backtrader thread via _bt_signal_execute (set as on_signal_callback
            # in _setup_backtrader). Starting _start_trading_bot_loop() alongside
            # cerebro would:
            #   1. Process an always-empty signal queue with a 1 s sleep, wasting CPU.
            #   2. Race on active_positions if strategy_class.get_signals() exists.
            #   3. Duplicate DB heartbeat writes from both loop and _heartbeat_loop.
            # When cerebro is None (pure live-data bot, no Backtrader), the
            # BaseTradingBot loop is the sole execution path and must run.
            asyncio.create_task(self._run_backtrader_async())
            if self.cerebro is None:
                asyncio.create_task(self._start_trading_bot_loop())

            _logger.info("✅ Strategy instance %s started successfully", self.name)
            return True

        except Exception as e:
            self.status = "error"
            self.error_count += 1
            self.last_error = str(e)
            _logger.exception("❌ Failed to start strategy instance %s:", self.name)

            try:
                trading_service.update_bot_status(self.instance_id, "error", error_message=str(e))
                await self._send_error_notification(f"Failed to start bot: {str(e)}", error_type="START_ERROR")
            except Exception:
                pass
            return False

    def _start_threads(self):
        """Start monitoring and heartbeat threads."""
        self.monitor_thread = threading.Thread(target=self._monitor_data_feed, daemon=True, name=f"Monitor-{self.name}")
        self.monitor_thread.start()

        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name=f"Heartbeat-{self.name}"
        )
        self.heartbeat_thread.start()

    async def stop(self) -> bool:
        """Stop the strategy instance gracefully."""
        try:
            _logger.info("Stopping strategy instance: %s", self.name)
            self.should_stop = True
            self.is_running = False

            if self.data_feed:
                try:
                    self.data_feed.stop()
                    _logger.info("Data feed stopped for %s", self.name)
                except Exception:
                    _logger.exception("Error stopping data feed for %s:", self.name)

            await self._stop_trading_bot()

            if self.broker:
                try:
                    await self.broker.disconnect()
                    _logger.info("Broker disconnected for %s", self.name)
                except Exception:
                    _logger.exception("Error disconnecting broker for %s:", self.name)

            self.status = "stopped"
            try:
                trading_service.update_bot_status(self.instance_id, "stopped")
                _logger.info("Bot status updated to stopped for %s", self.name)
            except Exception:
                _logger.exception("Error updating bot status for %s:", self.name)

            _logger.info("✅ Strategy instance %s stopped successfully", self.name)
            return True
        except Exception:
            self.status = "error"
            _logger.exception("❌ Failed to stop strategy instance %s: %s", self.name)
            return False

    async def restart(self) -> bool:
        """Restart the strategy instance."""
        await self.stop()
        await asyncio.sleep(2)
        return await self.start()

    def get_status(self) -> Dict[str, Any]:
        """Get strategy instance status summary."""
        uptime = 0.0
        if self.start_time:
            uptime = (datetime.now(UTC) - self.start_time).total_seconds()

        heartbeat_age = None
        if self.last_heartbeat:
            heartbeat_age = (datetime.now(UTC) - self.last_heartbeat).total_seconds()

        return {
            "instance_id": self.instance_id,
            "name": self.name,
            "status": self.status,
            "uptime_seconds": uptime,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "broker_type": self.config["broker"].get("type"),
            "trading_mode": self.config["broker"].get("trading_mode"),
            "symbol": self.config.get("symbol"),
            "strategy_type": self.config["strategy"].get("type"),
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "heartbeat_age_seconds": heartbeat_age,
            "is_healthy": heartbeat_age < (self.heartbeat_interval * 2) if heartbeat_age else False,
        }

    def _get_strategy_class(self, strategy_type: str):
        """Get strategy class based on type using StrategyHandler."""
        return strategy_handler.get_strategy_class(strategy_type)

    def _build_bot_config(self) -> Dict[str, Any]:
        """Build configuration for BaseTradingBot."""
        return {
            "trading_pair": self.config.get("symbol", "BTCUSDT"),
            "initial_balance": self.config["broker"].get("cash", 10000.0),
            "notifications": self.config.get("notifications", {}),
            "user_id": self.config.get("user_id"),
            "risk_management": self.config.get("risk_management", {}),
            "logging": self.config.get("logging", {}),
            "data": self.config.get("data", {}),
            "trading": self.config.get("trading", {}),
        }

    async def _create_trading_bot(self) -> bool:
        """Instantiate the BaseTradingBot for this instance."""
        try:
            bot_config = self._build_bot_config()
            strategy_class = self._get_strategy_class(self.config["strategy"]["type"])
            parameters = self.config["strategy"].get("parameters", {})
            paper = self.config["broker"].get("trading_mode", "paper") == "paper"

            trade_hook = (
                None if is_file_based_simulation_config(self.config) else self._schedule_user_trade_notification
            )
            self.trading_bot = BaseTradingBot(
                config=bot_config,
                strategy_class=strategy_class,
                parameters=parameters,
                broker=self.broker,
                paper_trading=paper,
                bot_id=str(self.instance_id),
                trade_repository=self.trade_repository,
                trade_notification_hook=trade_hook,
            )
            return True
        except Exception:
            _logger.exception("Error creating BaseTradingBot for %s:", self.name)
            return False

    async def _start_trading_bot_loop(self):
        """Run the BaseTradingBot heartbeat/execution loop."""
        if not self.trading_bot:
            _logger.warning("Trading bot not found for %s", self.name)
            return
        _logger.info("Starting trading bot loop for %s", self.name)
        try:
            await asyncio.get_event_loop().run_in_executor(None, self.trading_bot.run)
        except Exception:
            _logger.exception("Error in BaseTradingBot loop for %s:", self.name)
            self.status = "error"

    async def _stop_trading_bot(self):
        """Stop the trading bot gracefully."""
        if self.trading_bot:
            try:
                self.trading_bot.stop()
                _logger.info("Trading bot stopped for %s", self.name)
            except Exception:
                _logger.exception("Error stopping trading bot for %s:", self.name)

    def _create_data_feed(self) -> bool:
        """Create and initialize the data feed."""
        try:
            data_config = dict(
                self.config.get(
                    "data",
                    {
                        "data_source": self.config["broker"].get("type", "binance"),
                        "symbol": self.config.get("symbol", "BTCUSDT"),
                        "interval": "1h",
                        "lookback_bars": 500,
                    },
                )
            )

            # Ensure required fields for DataFeedFactory when config.data is partial.
            if not data_config.get("data_source"):
                data_config["data_source"] = self.config["broker"].get("type", "binance")
            if not data_config.get("symbol"):
                data_config["symbol"] = self.config.get("symbol", "BTCUSDT")
            if not data_config.get("interval"):
                data_config["interval"] = "1h"
            if data_config.get("lookback_bars") is None:
                data_config["lookback_bars"] = 500

            def on_new_bar(symbol, timestamp, data):
                _logger.debug("New %s bar for %s", symbol, self.name)

            data_config["on_new_bar"] = on_new_bar
            self.data_feed = DataFeedFactory.create_data_feed(data_config)
            return self.data_feed is not None
        except Exception:
            _logger.exception("Error creating data feed for %s:", self.name)
            return False

    def _setup_backtrader(self, strategy_class) -> bool:
        """Setup Backtrader engine."""
        try:
            if not BACKTRADER_AVAILABLE:
                _logger.error(
                    "backtrader is not installed; cannot create Cerebro for instance %s",
                    self.name,
                )
                return False
            import backtrader as bt

            self.cerebro = bt.Cerebro()
            self.cerebro.adddata(self.data_feed)

            strategy_params = self.config["strategy"].get("parameters", {})

            def _bt_signal_execute(signal: dict) -> None:
                """Execute a signal synchronously in the Backtrader thread.

                When Backtrader is active, signals must be acted upon immediately
                inside the same thread that drives cerebro — queuing them for a
                separate BaseTradingBot loop causes a time-mismatch (bars processed
                in milliseconds vs. 1 s sleep between queue drains) and would
                leave signals unprocessed after the cerebro run completes.
                """
                if not self.trading_bot:
                    return
                side = (signal.get("type") or signal.get("side") or "").strip().lower()
                if not side:
                    return
                try:
                    price = float(signal.get("price") or 0.0)
                    s = signal.get("size")
                    q = signal.get("quantity")
                    size_val = s if s is not None else (q if q is not None else 0.0)
                    size = float(size_val)
                except (TypeError, ValueError):
                    _logger.warning("Ignoring Backtrader signal with non-numeric price/size: %s", signal)
                    return
                self.trading_bot.execute_trade(side, price, size)

            self.cerebro.addstrategy(
                strategy_class,
                strategy_config=strategy_params,
                on_signal_callback=_bt_signal_execute,
                on_order_executed_callback=self._on_bt_order_filled,
            )

            # Setup broker
            broker_config = self.config["broker"]
            if broker_config.get("type", "").lower() == "backtrader":
                self.cerebro.broker.setcash(broker_config.get("cash", 10000.0))
                self.cerebro.broker.setcommission(commission=broker_config.get("commission", 0.001))
            elif self.broker:
                from src.trading.broker.backtrader_broker_bridge import wrap_broker_for_cerebro

                self.cerebro.setbroker(wrap_broker_for_cerebro(self.broker))

            return True
        except Exception:
            _logger.exception("Error setting up Backtrader for %s:", self.name)
            return False

    async def _run_backtrader_async(self):
        """Run Backtrader engine in background."""
        try:
            if self.cerebro is not None:
                await asyncio.get_event_loop().run_in_executor(None, self.cerebro.run)
        except Exception:
            _logger.exception("Error in Backtrader engine for %s: %s", self.name)
            self.status = "error"

    def _monitor_data_feed(self):
        """Monitor data feed health and reconnect if needed."""
        while self.is_running and not self.should_stop:
            try:
                if self.data_feed and not self.data_feed.get_status().get("is_connected", False):
                    _logger.warning("Data feed disconnected for %s, reconnecting...", self.name)
                    self.data_feed.stop()
                    self._create_data_feed()
                time.sleep(30)
            except Exception:
                time.sleep(60)

    def _heartbeat_loop(self):
        """
        Main heartbeat loop for the instance.

        This is the *sole* DB writer for ``last_heartbeat`` and
        ``current_balance`` when BaseTradingBot is managed by a
        StrategyInstance.  Running a single writer here prevents
        last-write-wins contention with the BaseTradingBot loop.
        """
        while self.is_running and not self.should_stop:
            try:
                self.last_heartbeat = datetime.now(UTC)

                # Persist heartbeat timestamp to DB so health monitors see
                # a live signal.  Also sync current_balance if the trading
                # bot has updated it since the last heartbeat tick.
                try:
                    trading_service.heartbeat(self.instance_id)
                    if self.trading_bot is not None:
                        trading_service.update_bot_performance(
                            self.instance_id,
                            current_balance=self.trading_bot.current_balance,
                        )
                except Exception:
                    _logger.debug("Failed to persist heartbeat for instance %s", self.name)

                time.sleep(self.heartbeat_interval)
            except Exception:
                time.sleep(5)

    async def _send_trade_notification(self, order_type: str, price: float, size: float, pnl: float | None = None):
        """
        Send notification for trade execution (async fire-and-forget).
        """
        try:
            if is_file_based_simulation_config(self.config):
                return
            if not self.notification_client:
                return

            # Get user notification details
            user_details = self._get_user_notification_details()
            if not user_details:
                return

            # Check if this notification type is enabled
            notif_config = self.config.get("notifications", {})
            if order_type.lower() == "buy" and not notif_config.get("position_opened", False):
                return
            if order_type.lower() == "sell" and not notif_config.get("position_closed", False):
                return

            # Get symbol and trading mode
            symbol = self.config.get("symbol", "UNKNOWN")
            trading_mode = self.config["broker"].get("trading_mode", "paper")

            # Format message
            if order_type.lower() == "buy":
                title = f"Position Opened: {symbol}"
                message = f"BUY {size:.4f} {symbol} @ ${price:,.2f}"
                if trading_mode == "paper":
                    message += " (Paper Trading)"
                notification_type = MessageType.TRADE_ENTRY
            else:
                title = f"Position Closed: {symbol}"
                message = f"SELL {size:.4f} {symbol} @ ${price:,.2f}"
                if pnl is not None:
                    pnl_pct = (pnl / (price * size)) * 100 if (price * size) > 0 else 0
                    message += f" (P&L: ${pnl:,.2f} / {pnl_pct:+.2f}%)"
                if trading_mode == "paper":
                    message += " (Paper Trading)"
                notification_type = MessageType.TRADE_EXIT

            # Add bot name to message
            message = f"[{self.name}] {message}"

            # Send notification (fire-and-forget)
            await self.notification_client.send_notification(
                notification_type=notification_type,
                title=title,
                message=message,
                priority=MessagePriority.HIGH,
                channels=user_details["channels"],
                recipient_id=user_details["recipient_id"],
                email_receiver=user_details["email"],
                telegram_chat_id=user_details.get("telegram_user_id"),
                data={
                    "bot_id": self.instance_id,
                    "bot_name": self.name,
                    "symbol": symbol,
                    "order_type": order_type.upper(),
                    "price": price,
                    "size": size,
                    "pnl": pnl,
                    "trading_mode": trading_mode,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
                source="trading_bot",
            )

            _logger.info("Sent %s notification for %s", order_type.upper(), self.name)

        except Exception:
            _logger.exception("Error sending trade notification for %s:", self.name)

    async def _send_error_notification(self, error_message: str, error_type: str = "ERROR"):
        """
        Admins always receive bot errors. The owner receives them only when
        ``notifications.error_notifications`` is true.
        """
        try:
            if is_file_based_simulation_config(self.config):
                return
            if not self.notification_client:
                return

            notif_config = self.config.get("notifications", {})
            notify_owner = notif_config.get("error_notifications", False)

            symbol = self.config.get("symbol", "UNKNOWN")
            trading_mode = self.config["broker"].get("trading_mode", "paper")

            title = f"Bot Error: {self.name}"
            message = f"[{self.name}] {error_type}: {error_message}"
            if trading_mode == "paper":
                message += " (Paper Trading)"

            payload_data = {
                "bot_id": self.instance_id,
                "bot_name": self.name,
                "symbol": symbol,
                "error_type": error_type,
                "error_message": error_message,
                "trading_mode": trading_mode,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            if notify_owner:
                user_details = self._get_user_notification_details()
                if user_details:
                    tg_raw = user_details.get("telegram_user_id")
                    tg_id = None
                    if tg_raw is not None:
                        s = str(tg_raw).strip()
                        tg_id = int(s) if s.isdigit() else None

                    await self.notification_client.send_notification(
                        notification_type=MessageType.ERROR,
                        title=title,
                        message=message,
                        priority=MessagePriority.CRITICAL,
                        channels=user_details["channels"],
                        recipient_id=user_details["recipient_id"],
                        email_receiver=user_details["email"],
                        telegram_chat_id=tg_id,
                        data=payload_data,
                        source="trading_bot",
                    )

            await self.notification_client.send_to_admins(
                title=f"{title} (admin)",
                message=message,
                notification_type=MessageType.ERROR,
                priority=MessagePriority.CRITICAL,
                data={**payload_data, "admin_notification": True},
                channels=["telegram", "email"],
            )

            _logger.info("Sent error notification for %s", self.name)

        except Exception:
            _logger.exception("Error sending error notification for %s:", self.name)

    def _get_user_notification_details(self) -> Dict[str, Any] | None:
        """Resolve the sole bot owner's Telegram/email targets."""
        try:
            return get_trading_bot_notification_recipient(
                self.config,
                self.instance_id,
                purpose="any",
                log_name=self.name,
            )
        except Exception:
            _logger.exception("Error fetching user notification details for bot %s:", self.name)
            return None

    def _schedule_user_trade_notification(
        self,
        side: str,
        price: float,
        size: float,
        pnl: float | None = None,
    ) -> None:
        """Notify configured user channels after BaseTradingBot executes a trade."""
        _run_async(self._send_trade_notification(side, price, size, pnl))

    def _on_bt_order_filled(
        self,
        order_type: str,
        price: float,
        size: float,
        dt=None,
    ) -> None:
        """Backtrader order.Completed callback (may run on the BT thread)."""
        try:
            side = "buy" if order_type.upper() == "BUY" else "sell"
            _run_async(self._send_trade_notification(side, price, size))
        except Exception:
            _logger.exception("Error scheduling trade notification for %s:", self.name)

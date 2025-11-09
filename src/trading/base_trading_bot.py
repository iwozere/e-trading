"""
Base Trading Bot Module
----------------------

This module defines the BaseTradingBot class, which provides a framework for implementing trading bots with signal processing, trade execution, position management, and notification capabilities. It is designed to be subclassed by specific strategy bots and supports integration with notification systems (Telegram, email).

Main Features:
- Signal processing and trade execution logic
- Position and balance management
- Trade history tracking with database integration
- Notification via Telegram and email
- Designed for extension by concrete strategy bots

Classes:
- BaseTradingBot: Abstract base class for trading bots
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


from src.trading.services.trading_bot_service import trading_bot_service
from src.trading.risk.controller import RiskController
from src.notification.service.client import NotificationServiceClient, NotificationServiceError
from src.notification.model import NotificationType, NotificationPriority
from src.trading.broker.base_broker import PositionNotificationManager

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


class BaseTradingBot:
    """
    Base class for trading bots. Handles config, strategy class, parameters, broker, notifications, and state.
    Subclasses should only override methods if custom logic is needed.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        strategy_class: Any,
        parameters: Dict[str, Any],
        broker: Any = None,
        paper_trading: bool = True,
        bot_id: str = None,
    ) -> None:
        """
        Initialize the trading bot with config, strategy, broker, and mode.
        Args:
            config: Configuration dictionary
            strategy_class: Strategy class
            parameters: Strategy parameters
            broker: Broker instance (optional)
            paper_trading: Whether to use paper trading mode
            bot_id: Bot identifier (config filename or optimization result filename)
        """
        self.config = config
        self.trading_pair = config.get("trading_pair", "BTCUSDT")
        self.initial_balance = config.get("initial_balance", 1000.0)
        self.strategy_class = strategy_class
        self.parameters = parameters
        self.is_running = False
        self.active_positions = {}
        self.trade_history = []
        self.current_balance = self.initial_balance
        self.total_pnl = 0.0
        self.broker = broker
        self.paper_trading = paper_trading

        # Database integration
        self.bot_id = bot_id or f"bot_{uuid.uuid4().hex[:8]}"
        self.trade_type = "paper" if paper_trading else "live"
        self.trade_repository = trading_bot_service

        # Enhanced notification setup using new notification service
        self.notification_client = None
        self.position_notification_manager = None

        try:
            # Initialize notification service client (using Main API Service)
            notification_service_url = config.get('notification_service_url', 'http://localhost:8000')
            self.notification_client = NotificationServiceClient(
                service_url=notification_service_url,
                timeout=30,
                max_retries=3
            )

            # Initialize enhanced position notification manager
            notification_config = config.get('notifications', {
                'position_opened': True,
                'position_closed': True,
                'email_enabled': True,
                'telegram_enabled': True,
                'error_notifications': True
            })
            self.position_notification_manager = PositionNotificationManager({
                'notifications': notification_config
            })

        except Exception as e:
            _logger.exception("Notification client not initialized: %s", e)

        self.max_drawdown_pct = config.get("max_drawdown_pct", 20.0)
        self.max_exposure = config.get("max_exposure", 1.0)  # 1.0 = 100% of balance
        self.position_sizing_pct = config.get("position_sizing_pct", 0.1)  # 10% of balance per trade

        # Initialize bot instance in database
        self._initialize_bot_instance()

        # Load state (including open positions from database)
        self.load_state()

        self.risk_controller = RiskController(config.get("risk", {}))


    def _initialize_bot_instance(self):
        """Initialize bot instance in database."""
        try:
            bot_data = {
                'id': self.bot_id,
                'type': self.trade_type,
                'config_file': getattr(self.config, 'get', lambda x, y=None: y)('config_file', None),
                'status': 'stopped',
                'current_balance': self.current_balance,
                'total_pnl': self.total_pnl,
                'extra_metadata': {
                    'trading_pair': self.trading_pair,
                    'initial_balance': self.initial_balance,
                    'strategy_class': self.strategy_class.__name__ if hasattr(self.strategy_class, '__name__') else str(self.strategy_class),
                    'parameters': self.parameters
                }
            }

            # Check if bot instance already exists
            existing_bot = self.trade_repository.get_bot_instance(self.bot_id)
            if existing_bot:
                # Update existing bot instance
                self.trade_repository.update_bot_instance(self.bot_id, {
                    'status': 'stopped',
                    'current_balance': self.current_balance,
                    'total_pnl': self.total_pnl,
                    'last_heartbeat': datetime.now(timezone.utc)
                })
            else:
                # Create new bot instance
                self.trade_repository.create_bot_instance(bot_data)

            _logger.info("Initialized bot instance: %s", self.bot_id)

        except Exception:
            _logger.exception("Error initializing bot instance: %s")

    def run(self) -> None:
        """
        Main bot loop. Handles signals, order management, error handling, and state persistence.
        """
        self.is_running = True
        _logger.info("Starting bot for %s", self.trading_pair)

        # Update bot status to running
        try:
            self.trade_repository.update_bot_instance(self.bot_id, {
                'status': 'running',
                'started_at': datetime.now(timezone.utc),
                'last_heartbeat': datetime.now(timezone.utc)
            })
        except Exception:
            _logger.exception("Error updating bot status: %s")

        while self.is_running:
            try:
                signals = self.get_signals()
                self.process_signals(signals)
                self.update_positions()
                self.save_state()

                # Update heartbeat
                try:
                    self.trade_repository.update_bot_instance(self.bot_id, {
                        'last_heartbeat': datetime.now(timezone.utc),
                        'current_balance': self.current_balance,
                        'total_pnl': self.total_pnl
                    })
                except Exception:
                    _logger.exception("Error updating heartbeat: %s")

                time.sleep(1)
            except Exception as e:
                _logger.exception("Error in bot loop: %s")
                self.notify_error(str(e))
                time.sleep(5)

    def get_signals(self) -> List[Dict[str, Any]]:
        """
        Get trading signals from the strategy for the current trading pair.
        Returns:
            List of signal dictionaries
        """
        return self.strategy_class.get_signals(self.trading_pair)

    def process_signals(self, signals: List[Dict[str, Any]]) -> None:
        """
        Process a list of trading signals and execute trades as needed.
        Args:
            signals: List of signal dictionaries
        """
        for signal in signals:
            if (
                signal["type"] == "buy"
                and self.trading_pair not in self.active_positions
            ):
                self.execute_trade("buy", signal["price"], signal["size"])
            elif (
                signal["type"] == "sell" and self.trading_pair in self.active_positions
            ):
                self.execute_trade("sell", signal["price"], signal["size"])

    def execute_trade(self, trade_type: str, price: float, size: float) -> None:
        """
        Generic trade execution logic for buy/sell with database integration.
        """
        timestamp = datetime.now(timezone.utc)
        order = None

        try:
            if not self.paper_trading and self.broker:
                order = self.broker.place_order(
                    self.trading_pair, trade_type.upper(), size, price=price
                )
                self.log_order(order)

            if trade_type == "buy":
                # Create new trade record in database
                trade_data = {
                    'bot_id': self.bot_id,
                    'trade_type': self.trade_type,
                    'strategy_name': self.strategy_class.__name__ if hasattr(self.strategy_class, '__name__') else 'Unknown',
                    'entry_logic_name': self.parameters.get('strategy_config', {}).get('entry_logic', {}).get('name', 'Unknown'),
                    'exit_logic_name': self.parameters.get('strategy_config', {}).get('exit_logic', {}).get('name', 'Unknown'),
                    'symbol': self.trading_pair,
                    'interval': self.config.get('data', {}).get('interval', '1h'),
                    'entry_time': timestamp,
                    'buy_order_created': timestamp,
                    'entry_price': price,
                    'entry_value': price * size,
                    'size': size,
                    'direction': 'long',
                    'status': 'open',
                    'extra_metadata': {
                        'order_id': str(order) if order else None,
                        'paper_trading': self.paper_trading
                    }
                }

                # Create trade in database
                trade = self.trade_repository.create_trade(trade_data)

                # Update local state
                self.active_positions[self.trading_pair] = {
                    "entry_price": price,
                    "size": size,
                    "entry_time": timestamp,
                    "trade_id": str(trade.id) if trade else None
                }

                # Send both legacy and enhanced notifications
                self.notify_trade_event("BUY", price, size, timestamp)

                # Enhanced position notification
                if self.position_notification_manager:
                    position_data = {
                        'symbol': self.trading_pair,
                        'side': 'BUY',
                        'price': price,
                        'size': size,
                        'timestamp': timestamp,
                        'bot_id': self.bot_id,
                        'trading_mode': 'paper' if self.paper_trading else 'live',
                        'order_id': str(order) if order else trade_id,
                        'strategy': self.strategy_class.__name__ if hasattr(self.strategy_class, '__name__') else 'Unknown'
                    }
                    asyncio.run(self.position_notification_manager.notify_position_opened(position_data))

            else:  # sell
                if self.trading_pair in self.active_positions:
                    position = self.active_positions[self.trading_pair]
                    trade_id = position.get("trade_id")

                    # Calculate PnL
                    pnl = ((price - position["entry_price"]) / position["entry_price"]) * 100
                    gross_pnl = (price - position["entry_price"]) * position["size"]
                    commission = gross_pnl * 0.001  # 0.1% commission
                    net_pnl = gross_pnl - commission

                    # Update trade in database
                    if trade_id:
                        update_data = {
                            'exit_time': timestamp,
                            'sell_order_created': timestamp,
                            'sell_order_closed': timestamp,
                            'exit_price': price,
                            'exit_value': price * position["size"],
                            'commission': commission,
                            'gross_pnl': gross_pnl,
                            'net_pnl': net_pnl,
                            'pnl_percentage': pnl,
                            'exit_reason': 'signal',
                            'status': 'closed',
                            'extra_metadata': {
                                'order_id': str(order) if order else None,
                                'paper_trading': self.paper_trading
                            }
                        }
                        self.trade_repository.update_trade(trade_id, update_data)

                    # Update local state
                    trade_record = {
                        "bot_id": self.bot_id,
                        "pair": self.trading_pair,
                        "type": "long",
                        "entry_price": position["entry_price"],
                        "exit_price": price,
                        "size": position["size"],
                        "pl": pnl,
                        "time": timestamp.isoformat(),
                    }
                    self.trade_history.append(trade_record)

                    # Legacy JSON logging (for backward compatibility)
                    self.log_trade(trade_record)

                    # Update balance
                    self.current_balance *= 1 + pnl / 100
                    self.total_pnl += pnl

                    # Remove from active positions
                    del self.active_positions[self.trading_pair]

                    # Send both legacy and enhanced notifications
                    self.notify_trade_event(
                        "SELL",
                        price,
                        size,
                        timestamp,
                        entry_price=position["entry_price"],
                        pnl=pnl,
                    )

                    # Enhanced position notification
                    if self.position_notification_manager:
                        # Calculate hold duration
                        hold_duration = "Unknown"
                        if position.get("entry_time"):
                            duration = timestamp - position["entry_time"]
                            days = duration.days
                            hours, remainder = divmod(duration.seconds, 3600)
                            minutes, seconds = divmod(remainder, 60)

                            if days > 0:
                                hold_duration = f"{days}d {hours}h {minutes}m"
                            elif hours > 0:
                                hold_duration = f"{hours}h {minutes}m {seconds}s"
                            else:
                                hold_duration = f"{minutes}m {seconds}s"

                        position_data = {
                            'symbol': self.trading_pair,
                            'side': 'SELL',
                            'entry_price': position["entry_price"],
                            'exit_price': price,
                            'size': size,
                            'pnl': gross_pnl,
                            'pnl_percentage': pnl,
                            'timestamp': timestamp,
                            'bot_id': self.bot_id,
                            'trading_mode': 'paper' if self.paper_trading else 'live',
                            'order_id': str(order) if order else trade_id,
                            'strategy': self.strategy_class.__name__ if hasattr(self.strategy_class, '__name__') else 'Unknown',
                            'hold_duration': hold_duration
                        }
                        asyncio.run(self.position_notification_manager.notify_position_closed(position_data))

        except Exception as e:
            _logger.exception("Error executing trade: %s")
            self.notify_error(str(e))

    def log_order(self, order: Any) -> None:
        """
        Persist order details to logs/json/orders.json.
        Args:
            order: Order object or dictionary
        """
        folder = os.path.join("logs", "json")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "orders.json")
        try:
            if os.path.exists(path):
                with open(path, "r+", encoding="utf-8") as f:
                    try:
                        all_orders = json.load(f)
                    except Exception:
                        all_orders = []
                    all_orders.append(order)
                    f.seek(0)
                    json.dump(all_orders, f, default=str, indent=2)
                    f.truncate()
            else:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump([order], f, default=str, indent=2)
        except Exception:
            _logger.exception("Failed to log order: %s")

    def log_trade(self, trade: Dict[str, Any]) -> None:
        """
        Persist trade details to logs/json/trades.json.
        Args:
            trade: Trade dictionary
        """
        folder = os.path.join("logs", "json")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "trades.json")
        try:
            if os.path.exists(path):
                with open(path, "r+", encoding="utf-8") as f:
                    try:
                        all_trades = json.load(f)
                    except Exception:
                        all_trades = []
                    all_trades.append(trade)
                    f.seek(0)
                    json.dump(all_trades, f, default=str, indent=2)
                    f.truncate()
            else:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump([trade], f, default=str, indent=2)
        except Exception:
            _logger.exception("Failed to log trade: %s")

    def save_state(self) -> None:
        """
        Save open positions and bot state to disk for recovery.
        """
        folder = os.path.join("logs", "json")
        os.makedirs(folder, exist_ok=True)
        try:
            state = {
                "active_positions": self.active_positions,
                "trade_history": self.trade_history,
                "current_balance": self.current_balance,
                "total_pnl": self.total_pnl,
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, default=str, indent=2)
        except Exception:
            _logger.exception("Failed to save bot state: %s")

    def load_state(self) -> None:
        """
        Load open positions and bot state from database and legacy files.
        """
        # First try to load from database
        self._load_open_positions_from_db()

        # Then load from legacy state file (for backward compatibility)
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    # Only load trade history and balance from legacy file
                    # Active positions should come from database
                    self.trade_history = state.get("trade_history", [])
                    self.current_balance = state.get(
                        "current_balance", self.initial_balance
                    )
                    self.total_pnl = state.get("total_pnl", 0.0)
            except Exception:
                _logger.exception("Failed to load legacy bot state: %s")

    def _load_open_positions_from_db(self) -> None:
        """
        Load open positions from database for this bot.
        """
        try:
            open_trades = self.trade_repository.get_open_trades(
                bot_id=self.bot_id,
                symbol=self.trading_pair
            )

            for trade in open_trades:
                self.active_positions[trade.symbol] = {
                    "entry_price": float(trade.entry_price) if trade.entry_price else 0.0,
                    "size": float(trade.size) if trade.size else 0.0,
                    "entry_time": trade.entry_time,
                    "trade_id": str(trade.id)
                }

            _logger.info("Loaded %d open positions from database for %s", len(open_trades), self.bot_id)

        except Exception:
            _logger.exception("Error loading open positions from database: %s")
            # Fall back to empty active positions
            self.active_positions = {}

    def notify_error(self, error_msg: str) -> None:
        """
        Send error notification to admin users via notification service.
        Args:
            error_msg: Error message string
        """
        try:
            if not self.notification_client:
                _logger.warning("Notification client not available for error notification")
                return

            # Create error message
            error_message = (
                f"⚠️ Trading Bot Error\n\n"
                f"Bot ID: {self.bot_id}\n"
                f"Trading Pair: {self.trading_pair}\n"
                f"Error: {error_msg}\n\n"
                f"Please check the bot configuration and system logs."
            )

            # Send via notification service
            asyncio.run(self.notification_client.send_error_notification_async(
                error_message=error_message,
                source="trading_bot",
                channels=["telegram", "email"],
                metadata={
                    "bot_id": self.bot_id,
                    "trading_pair": self.trading_pair,
                    "error_type": "trading_bot_error"
                }
            ))

        except NotificationServiceError:
            _logger.exception("Failed to send error notification via service:")
            # Fallback to legacy method
            try:
                from src.telegram.screener.http_api_client import send_notification_to_admins
                asyncio.run(send_notification_to_admins(
                    message=error_message,
                    title="Trading Bot Error"
                ))
            except Exception as fallback_e:
                _logger.exception("Fallback error notification also failed: %s", fallback_e)
        except Exception as e:
            _logger.exception("Failed to send error notification: %s", e)

    def notify_trade_event(
        self,
        side: str,
        price: float,
        size: float,
        timestamp: datetime,
        entry_price: Optional[float] = None,
        pnl: Optional[float] = None,
    ) -> None:
        """
        Send trade event notification to admin users via notification service.
        Args:
            side: 'BUY' or 'SELL'
            price: Trade price
            size: Trade size
            timestamp: Trade timestamp
            entry_price: Entry price (optional)
            pnl: Profit/loss (optional)
        """
        try:
            if not self.notification_client:
                _logger.warning("Notification client not available for trade notification")
                return

            # Create trade message
            emoji = "🟢" if side == "BUY" else "🔴"
            pnl_text = f"PnL: {pnl:.2f}%" if pnl is not None else ""

            trade_message = (
                f"{emoji} Trade {side}\n\n"
                f"Bot ID: {self.bot_id}\n"
                f"Trading Pair: {self.trading_pair}\n"
                f"Price: ${price:.2f}\n"
                f"Size: {size}\n"
                f"{pnl_text}\n"
                f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            # Send via notification service
            asyncio.run(self.notification_client.send_trade_notification_async(
                symbol=self.trading_pair,
                side=side,
                price=price,
                quantity=size,
                entry_price=entry_price,
                pnl=pnl,
                channels=["telegram", "email"],
                metadata={
                    "bot_id": self.bot_id,
                    "timestamp": timestamp.isoformat()
                }
            ))

        except NotificationServiceError:
            _logger.exception("Failed to send trade notification via service:")
            # Fallback to legacy method
            try:
                from src.telegram.screener.http_api_client import send_notification_to_admins
                asyncio.run(send_notification_to_admins(
                    message=trade_message,
                    title=f"Trading Bot - {side}"
                ))
            except Exception as fallback_e:
                _logger.exception("Fallback trade notification also failed: %s", fallback_e)
        except Exception as e:
            _logger.exception("Failed to send trade notification: %s", e)

    def update_positions(self):
        """
        Generic position update logic for stop loss/take profit. Subclasses can override for custom behavior.
        """
        for pair, position in list(self.active_positions.items()):
            # Assumes the strategy instance is available and has get_current_price, sl_atr_mult, tp_atr_mult
            # If running in Backtrader, you may need to adapt this for live/real-time bots
            current_price = None
            if (
                hasattr(self, "strategy")
                and self.strategy
                and hasattr(self.strategy, "get_current_price")
            ):
                current_price = self.strategy.get_current_price(pair)
            if current_price is None:
                continue
            entry_price = position["entry_price"]
            pnl = ((current_price - entry_price) / entry_price) * 100
            # Check stop loss
            if (
                hasattr(self.strategy, "sl_atr_mult")
                and pnl <= -self.strategy.sl_atr_mult * 100
            ):
                self.execute_trade("sell", current_price, position["size"])
            # Check take profit
            elif (
                hasattr(self.strategy, "tp_atr_mult")
                and pnl >= self.strategy.tp_atr_mult * 100
            ):
                self.execute_trade("sell", current_price, position["size"])

    def stop(self):
        """
        Generic stop logic: set is_running to False and close all open positions. Subclasses can override for custom behavior.
        """
        self.is_running = False
        _logger.info("Stopping bot for %s", self.trading_pair)

        # Update bot status in database
        try:
            self.trade_repository.update_bot_instance(self.bot_id, {
                'status': 'stopped',
                'last_heartbeat': datetime.now(timezone.utc),
                'current_balance': self.current_balance,
                'total_pnl': self.total_pnl
            })
        except Exception:
            _logger.exception("Error updating bot status on stop: %s")

        # Close all open positions
        for pair in list(self.active_positions.keys()):
            current_price = None
            if (
                hasattr(self, "strategy")
                and self.strategy
                and hasattr(self.strategy, "get_current_price")
            ):
                current_price = self.strategy.get_current_price(pair)
            if current_price is not None:
                self.execute_trade(
                    "sell", current_price, self.active_positions[pair]["size"]
                )

    def log_bot_event(self, event: str):
        _logger.info("%s: %s", event, self.__class__.__name__)
        _logger.info("Trading pair: %s", self.trading_pair)
        _logger.info("Initial balance: %s", self.initial_balance)
        _logger.info("Strategy parameters: %s", getattr(self, 'parameters', {}))
        _logger.info("Broker: %s", self.broker.__class__.__name__ if self.broker else 'None')
        timestamp = datetime.now(timezone.utc)
        _logger.info("Timestamp: %s", timestamp)

    def notify_bot_event(self, event: str, emoji: str):
        msg = (
            f"{emoji} Bot {event.lower()}\n\n"
            f"Class: {self.__class__.__name__}\n"
            f"Pair: {self.trading_pair}\n"
            f"Initial balance: {self.initial_balance}\n"
            f"Strategy: {getattr(self, 'strategy_class', type(self.strategy_class).__name__)}\n"
            f"Parameters: {getattr(self, 'parameters', {})}\n"
            f"Broker: {self.broker.__class__.__name__ if self.broker else 'None'}"
        )
        try:
            if not self.notification_client:
                _logger.warning("Notification client not available for bot event notification")
                return

            # Determine notification type based on event
            if event.lower() in ["error", "failed", "stopped"]:
                notification_type = NotificationType.ERROR
                priority = NotificationPriority.HIGH
            elif event.lower() in ["started", "resumed"]:
                notification_type = NotificationType.INFO
                priority = NotificationPriority.NORMAL
            else:
                notification_type = NotificationType.SYSTEM
                priority = NotificationPriority.NORMAL

            # Send via notification service
            asyncio.run(self.notification_client.send_notification_async(
                notification_type=notification_type,
                title=f"Trading Bot - {event.title()}",
                message=msg,
                priority=priority,
                channels=["telegram", "email"],
                metadata={
                    "bot_id": self.bot_id,
                    "trading_pair": self.trading_pair,
                    "event": event,
                    "bot_class": self.__class__.__name__
                }
            ))

        except NotificationServiceError:
            _logger.exception("Failed to send bot event notification via service:")
            # Fallback to legacy method
            try:
                from src.telegram.screener.http_api_client import send_notification_to_admins
                asyncio.run(send_notification_to_admins(
                    message=msg,
                    title=f"Trading Bot - {event.title()}"
                ))
            except Exception as fallback_e:
                _logger.exception("Fallback bot event notification also failed: %s", fallback_e)
        except Exception as e:
            _logger.exception("Failed to send bot event notification: %s", e)

    def pre_run(self, data_feed):
        """
        Hook for custom logic before running the Backtrader engine.
        Subclasses can override this to add analyzers, observers, or other setup.
        Args:
            data_feed: The data feed object that will be added to Cerebro.
        """

    def post_run(self, data_feed):
        """
        Hook for custom logic after running the Backtrader engine.
        Subclasses can override this to process results, save reports, etc.
        Args:
            data_feed: The data feed object that was used in Cerebro.
        """

    def run_backtrader_engine(self, data_feed):
        self.log_bot_event("started")
        self.notify_bot_event("started", "🤖")
        import backtrader as bt

        cerebro = bt.Cerebro()
        cerebro.adddata(data_feed)
        _logger.info("Added data feed for %s", self.trading_pair)
        cerebro.broker.setcash(self.initial_balance)
        if self.broker:
            cerebro.setbroker(self.broker)
        cerebro.addstrategy(
            getattr(self, "strategy_class"), params=getattr(self, "parameters", {})
        )
        self.pre_run(data_feed)
        _logger.info("Starting Backtrader engine for %s", self.trading_pair)
        cerebro.run()
        self.post_run(data_feed)
        self.log_bot_event("stopped")
        self.notify_bot_event("stopped", "🛑")

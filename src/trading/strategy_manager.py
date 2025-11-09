#!/usr/bin/env python3
"""
Enhanced Strategy Manager
------------------------

This module manages multiple strategy instances within a single service.
Each strategy instance can run with different broker configurations (paper/live).

Features:
- Multiple strategy instances in one service
- Per-strategy broker configuration (paper/live)
- Integration with existing BaseTradingBot and CustomStrategy
- Health monitoring and auto-recovery
- Unified logging and notifications
"""

import asyncio
import json
import uuid
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path

import backtrader as bt

from src.trading.broker.broker_factory import get_broker
from src.trading.broker.broker_manager import BrokerManager
from src.trading.strategy_handler import strategy_handler
from src.data.feed.data_feed_factory import DataFeedFactory
from src.notification.logger import setup_logger
from src.data.db.services.trading_service import trading_service
from src.data.db.services.users_service import UsersService
from src.notification.service.client import NotificationServiceClient, MessageType, MessagePriority

_logger = setup_logger(__name__)
_users_service = UsersService()


class StrategyInstance:
    """Represents a single strategy instance with its own configuration and broker."""

    def __init__(self, instance_id: str, config: Dict[str, Any], notification_client: Optional[NotificationServiceClient] = None):
        """Initialize strategy instance."""
        self.instance_id = instance_id
        self.config = config
        self.name = config.get('name', f'Strategy_{instance_id}')
        self.notification_client = notification_client

        # Core components
        self.broker = None
        self.trading_bot = None
        self.data_feed = None
        self.cerebro = None

        # Status tracking
        self.status = 'stopped'
        self.start_time = None
        self.error_count = 0
        self.last_error = None
        self.is_running = False
        self.should_stop = False

        # Heartbeat tracking
        self.last_heartbeat = None
        self.heartbeat_interval = 60  # seconds

        # Threading for monitoring
        self.monitor_thread = None
        self.heartbeat_thread = None

    async def start(self) -> bool:
        """
        Start the strategy instance with full Backtrader integration.

        Refactored from LiveTradingBot.start() to include:
        - Data feed creation
        - Backtrader setup
        - Trading loop execution
        """
        try:
            _logger.info("Starting strategy instance: %s", self.name)

            # Create broker (skip for backtrader type - use Cerebro's built-in broker)
            broker_config = self.config['broker']
            broker_type = broker_config.get('type', '').lower()

            if broker_type == 'backtrader':
                _logger.info("Using Backtrader's built-in broker (no custom broker needed)")
                self.broker = None
            else:
                self.broker = get_broker(broker_config)
                if not self.broker:
                    raise Exception("Failed to create broker")

                _logger.info("Created broker for %s: %s (mode: %s)",
                            self.name,
                            broker_config.get('type'),
                            broker_config.get('trading_mode'))

            # Get strategy class using StrategyHandler
            strategy_config = self.config['strategy']
            strategy_class = self._get_strategy_class(strategy_config['type'])
            _logger.info("Loaded strategy class: %s", strategy_class.__name__)

            # Create data feed (from LiveTradingBot._create_data_feed)
            if not self._create_data_feed():
                raise RuntimeError("Failed to create data feed")

            # Setup Backtrader (from LiveTradingBot._setup_backtrader)
            if not self._setup_backtrader(strategy_class):
                raise RuntimeError("Failed to setup Backtrader")

            # Start monitoring thread for data feed health
            self.monitor_thread = threading.Thread(
                target=self._monitor_data_feed,
                daemon=True,
                name=f"Monitor-{self.name}"
            )
            self.monitor_thread.start()
            _logger.info("Started data feed monitor thread for %s", self.name)

            # Start heartbeat thread
            self.heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
                name=f"Heartbeat-{self.name}"
            )
            self.heartbeat_thread.start()
            _logger.info("Started heartbeat thread for %s", self.name)

            # Set status
            self.status = 'running'
            self.start_time = datetime.now(timezone.utc)
            self.is_running = True

            # Reset error count on successful start
            self.error_count = 0
            self.last_error = None

            # Update database status
            try:
                trading_service.update_bot_status(
                    int(self.instance_id),
                    "running",
                    started_at=self.start_time
                )
            except Exception as e:
                _logger.warning("Failed to update bot status in DB: %s", e)

            # Start Backtrader in background task
            asyncio.create_task(self._run_backtrader_async())

            _logger.info("✅ Strategy instance %s started successfully", self.name)
            return True

        except Exception as e:
            self.status = 'error'
            self.error_count += 1
            self.last_error = str(e)
            _logger.exception("❌ Failed to start strategy instance %s:", self.name)

            # Update database with error status
            try:
                trading_service.update_bot_status(
                    int(self.instance_id),
                    "error",
                    error_message=str(e)
                )
            except Exception:
                pass

            # Send error notification
            try:
                await self._send_error_notification(
                    f"Failed to start bot: {str(e)}",
                    error_type="START_ERROR"
                )
            except Exception:
                _logger.debug("Failed to send error notification")

            return False

    async def stop(self) -> bool:
        """
        Stop the strategy instance gracefully.

        Refactored from LiveTradingBot.stop() to include:
        - Data feed shutdown
        - Backtrader cleanup
        - State persistence
        """
        try:
            _logger.info("Stopping strategy instance: %s", self.name)

            self.should_stop = True
            self.is_running = False

            # Stop data feed
            if self.data_feed:
                try:
                    self.data_feed.stop()
                    _logger.info("Stopped data feed for %s", self.name)
                except Exception as e:
                    _logger.warning("Error stopping data feed: %s", e)

            # Stop Cerebro if running
            if self.cerebro:
                try:
                    # Backtrader doesn't have explicit stop, it completes when data ends
                    _logger.info("Backtrader will complete naturally for %s", self.name)
                except Exception as e:
                    _logger.warning("Error with Backtrader cleanup: %s", e)

            # Stop broker connection
            if self.broker:
                try:
                    await self.broker.disconnect()
                    _logger.info("Disconnected broker for %s", self.name)
                except Exception as e:
                    _logger.warning("Error disconnecting broker: %s", e)

            # Update status
            self.status = 'stopped'

            # Update database status
            try:
                trading_service.update_bot_status(
                    int(self.instance_id),
                    "stopped"
                )
            except Exception as e:
                _logger.warning("Failed to update bot status in DB: %s", e)

            _logger.info("✅ Strategy instance %s stopped successfully", self.name)
            return True

        except Exception as e:
            self.status = 'error'
            self.error_count += 1
            self.last_error = str(e)
            _logger.exception("❌ Failed to stop strategy instance %s:", self.name)
            return False

    async def restart(self) -> bool:
        """Restart the strategy instance."""
        _logger.info("Restarting strategy instance: %s", self.name)
        await self.stop()
        await asyncio.sleep(2)  # Brief pause
        return await self.start()

    def get_status(self) -> Dict[str, Any]:
        """Get strategy instance status."""
        uptime = 0
        if self.start_time:
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        # Calculate time since last heartbeat
        heartbeat_age = None
        if self.last_heartbeat:
            heartbeat_age = (datetime.now(timezone.utc) - self.last_heartbeat).total_seconds()

        return {
            'instance_id': self.instance_id,
            'name': self.name,
            'status': self.status,
            'uptime_seconds': uptime,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'broker_type': self.config['broker'].get('type'),
            'trading_mode': self.config['broker'].get('trading_mode'),
            'symbol': self.config.get('symbol'),
            'strategy_type': self.config['strategy'].get('type'),
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'heartbeat_age_seconds': heartbeat_age,
            'is_healthy': heartbeat_age < (self.heartbeat_interval * 2) if heartbeat_age else False
        }

    def _get_strategy_class(self, strategy_type: str):
        """
        Get strategy class based on type using StrategyHandler.

        Args:
            strategy_type: Strategy type from configuration

        Returns:
            Strategy class to instantiate
        """
        try:
            # Use StrategyHandler for dynamic strategy loading
            return strategy_handler.get_strategy_class(strategy_type)
        except Exception:
            _logger.exception("Error getting strategy class for %s:", strategy_type)
            # StrategyHandler already handles fallback to CustomStrategy
            raise

    def _build_bot_config(self) -> Dict[str, Any]:
        """Build configuration for BaseTradingBot."""
        return {
            'trading_pair': self.config.get('symbol', 'BTCUSDT'),
            'initial_balance': self.config['broker'].get('cash', 10000.0),
            'notifications': self.config.get('notifications', {}),
            'risk_management': self.config.get('risk_management', {}),
            'logging': self.config.get('logging', {}),
            'data': self.config.get('data', {}),
            'trading': self.config.get('trading', {})
        }

    async def _start_trading_bot(self):
        """Start the trading bot (placeholder for actual implementation)."""
        # This would integrate with your existing BaseTradingBot.start() method
        # For now, we'll simulate starting
        _logger.info("Trading bot for %s would start here", self.name)
        # self.trading_bot.start()  # Uncomment when ready

    async def _stop_trading_bot(self):
        """Stop the trading bot (placeholder for actual implementation)."""
        # This would integrate with your existing BaseTradingBot.stop() method
        _logger.info("Trading bot for %s would stop here", self.name)
        # self.trading_bot.stop()  # Uncomment when ready

    def _create_data_feed(self) -> bool:
        """
        Create and initialize the data feed.

        Refactored from LiveTradingBot._create_data_feed()
        """
        try:
            data_config = self.config.get('data', {})

            # Ensure required fields
            if 'data_source' not in data_config:
                data_config['data_source'] = self.config['broker'].get('type', 'binance')
            if 'symbol' not in data_config:
                data_config['symbol'] = self.config.get('symbol', 'BTCUSDT')
            if 'interval' not in data_config:
                data_config['interval'] = '1h'
            if 'lookback_bars' not in data_config:
                data_config['lookback_bars'] = 500

            # Add callback for new data notifications (optional)
            def on_new_bar(symbol, timestamp, data):
                self._notify_new_bar(symbol, timestamp, data)
            data_config["on_new_bar"] = on_new_bar

            self.data_feed = DataFeedFactory.create_data_feed(data_config)

            if self.data_feed is None:
                raise ValueError("Failed to create data feed")

            _logger.info("Created data feed for %s: %s (%s)",
                        self.name,
                        data_config.get('symbol'),
                        data_config.get('interval'))
            return True

        except Exception:
            _logger.exception("Error creating data feed for %s:", self.name)
            return False

    def _setup_backtrader(self, strategy_class) -> bool:
        """
        Setup Backtrader engine.

        Refactored from LiveTradingBot._setup_backtrader()
        """
        try:
            self.cerebro = bt.Cerebro()

            # Add data feed
            self.cerebro.adddata(self.data_feed)
            _logger.info("Added data feed to Cerebro for %s", self.name)

            # Add strategy with parameters
            # Wrap parameters in strategy_config for BaseStrategy compatibility
            strategy_params = self.config['strategy'].get('parameters', {})
            self.cerebro.addstrategy(strategy_class, strategy_config=strategy_params)
            _logger.info("Added strategy %s to Cerebro with config", strategy_class.__name__)

            # Setup broker
            broker_type = self.config['broker'].get('type', '').lower()

            if broker_type == 'backtrader':
                # Use Cerebro's built-in broker (default) for backtesting
                _logger.info("Using Backtrader's built-in broker")

                # Setup initial cash
                initial_balance = self.config['broker'].get('cash', 10000.0)
                self.cerebro.broker.setcash(initial_balance)
                _logger.info("Set initial cash: $%.2f", initial_balance)

                # Setup commission
                commission = self.config['broker'].get('commission', 0.001)  # 0.1% default
                self.cerebro.broker.setcommission(commission=commission)
                _logger.info("Set commission: %.4f", commission)
            else:
                # Use custom broker (Binance, IBKR, etc.)
                if self.broker:
                    self.cerebro.broker = self.broker
                    _logger.info("Assigned custom %s broker to Cerebro", broker_type)
                else:
                    _logger.warning("No broker created for type: %s", broker_type)

            _logger.info("✅ Backtrader setup complete for %s", self.name)
            return True

        except Exception:
            _logger.exception("Error setting up Backtrader for %s:", self.name)
            return False

    async def _run_backtrader_async(self):
        """
        Run Backtrader engine in async context.

        Refactored from LiveTradingBot._run_backtrader()
        """
        try:
            _logger.info("Starting Backtrader engine for %s...", self.name)

            # Run Backtrader (blocking call, but in separate task)
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._run_backtrader_sync
            )

            _logger.info("Backtrader engine completed for %s", self.name)

        except Exception as e:
            _logger.exception("Error in Backtrader engine for %s:", self.name)
            self.status = 'error'
            self.error_count += 1
            self.last_error = str(e)

            # Update database with error
            try:
                trading_service.update_bot_status(
                    int(self.instance_id),
                    "error",
                    error_message=f"Backtrader error: {str(e)}"
                )
            except Exception:
                pass

    def _run_backtrader_sync(self):
        """Synchronous Backtrader execution."""
        results = self.cerebro.run()
        return results

    def _notify_new_bar(self, symbol: str, timestamp, data: Dict[str, Any]):
        """
        Notify about new data bar.

        Refactored from LiveTradingBot._notify_new_bar()
        """
        try:
            _logger.debug(
                "New %s bar: O=%.4f H=%.4f L=%.4f C=%.4f",
                symbol,
                data.get('open', 0),
                data.get('high', 0),
                data.get('low', 0),
                data.get('close', 0)
            )
        except Exception as e:
            _logger.debug("Error notifying new bar: %s", e)

    def _monitor_data_feed(self):
        """
        Monitor data feed health and reconnect if needed.

        Refactored from LiveTradingBot._monitor_data_feed()
        """
        _logger.info("Data feed monitor started for %s", self.name)

        while self.is_running and not self.should_stop:
            try:
                if self.data_feed:
                    status = self.data_feed.get_status()
                    if not status.get("is_connected", False):
                        _logger.warning(
                            "Data feed disconnected for %s, attempting reconnect...",
                            self.name
                        )
                        self._reconnect_data_feed()

                time.sleep(30)  # Check every 30 seconds

            except Exception:
                _logger.exception("Error in data feed monitor for %s:", self.name)
                time.sleep(60)

        _logger.info("Data feed monitor stopped for %s", self.name)

    def _reconnect_data_feed(self):
        """
        Reconnect data feed.

        Refactored from LiveTradingBot._reconnect_data_feed()
        """
        try:
            if self.data_feed:
                self.data_feed.stop()
                time.sleep(5)

            if self._create_data_feed():
                _logger.info("Data feed reconnected successfully for %s", self.name)
                # Note: Backtrader reconnection would require strategy restart
                # For now, just log the reconnection
            else:
                _logger.error("Failed to reconnect data feed for %s", self.name)
                self.error_count += 1

                # Send error notification if failures exceed threshold
                if self.error_count >= 3 and self.notification_client:
                    try:
                        asyncio.create_task(
                            self._send_error_notification(
                                f"Data feed connection failed {self.error_count} times",
                                error_type="DATA_FEED_ERROR"
                            )
                        )
                    except RuntimeError:
                        pass

        except Exception:
            _logger.exception("Error reconnecting data feed for %s:", self.name)

    def _heartbeat_loop(self):
        """
        Send periodic heartbeat updates to the database.

        This runs in a separate thread and updates the last_heartbeat field
        in the database to indicate the bot is still alive.

        Also updates performance metrics every 5 heartbeats (5 minutes by default).
        """
        _logger.info("Heartbeat loop started for %s", self.name)

        heartbeat_count = 0
        performance_update_interval = 5  # Update performance every 5 heartbeats

        while self.is_running and not self.should_stop:
            try:
                # Update heartbeat timestamp
                self.last_heartbeat = datetime.now(timezone.utc)

                # Update database heartbeat
                try:
                    trading_service.heartbeat(self.instance_id)
                    _logger.debug("Heartbeat sent for bot %s", self.instance_id)
                except Exception as e:
                    _logger.warning("Failed to send heartbeat for bot %s: %s", self.instance_id, e)

                # Periodically update performance metrics
                heartbeat_count += 1
                if heartbeat_count % performance_update_interval == 0:
                    self.update_performance_metrics()

                # Sleep for heartbeat interval
                time.sleep(self.heartbeat_interval)

            except Exception:
                _logger.exception("Error in heartbeat loop for %s:", self.name)
                time.sleep(self.heartbeat_interval)

        _logger.info("Heartbeat loop stopped for %s", self.name)

    def update_performance_metrics(self):
        """
        Update performance metrics in the database.

        This should be called periodically and after significant trading events.
        """
        try:
            if not self.broker:
                return

            # Get current balance from broker
            current_balance = self.broker.get_value() if hasattr(self.broker, 'get_value') else None

            # Calculate P&L if we have initial balance
            total_pnl = None
            initial_balance = self.config['broker'].get('cash')
            if current_balance and initial_balance:
                total_pnl = current_balance - initial_balance

            # Update database
            if current_balance or total_pnl:
                try:
                    trading_service.update_bot_performance(
                        int(self.instance_id),
                        current_balance=current_balance,
                        total_pnl=total_pnl
                    )
                    _logger.debug(
                        "Updated performance for bot %s: balance=%.2f, pnl=%.2f",
                        self.instance_id,
                        current_balance or 0,
                        total_pnl or 0
                    )
                except Exception as e:
                    _logger.warning("Failed to update performance metrics: %s", e)

        except Exception:
            _logger.exception("Error updating performance metrics for %s:", self.name)

    def record_trade(self, trade_data: Dict[str, Any]):
        """
        Record a trade execution in the database.

        This should be called when a trade is executed (buy or sell).

        Args:
            trade_data: Dictionary containing trade information
                - trade_type: 'paper' or 'live'
                - symbol: Trading symbol
                - entry_price: Entry price (for buy orders)
                - exit_price: Exit price (for sell orders)
                - entry_value: Position size value
                - exit_value: Exit value (for sell orders)
                - pnl: Profit/loss (for sell orders)
                - entry_logic_name: Name of entry logic used
                - exit_logic_name: Name of exit logic used
                - entry_time: Timestamp of entry
                - exit_time: Timestamp of exit (for sell orders)
        """
        try:
            # Enrich trade data with bot information
            full_trade_data = {
                'bot_id': int(self.instance_id),
                'trade_type': self.config['broker'].get('trading_mode', 'paper'),
                'symbol': self.config.get('symbol', 'UNKNOWN'),
                'interval': self.config.get('data', {}).get('interval', '1h'),
                'strategy_name': self.config['strategy'].get('type', 'CustomStrategy'),
                **trade_data
            }

            # Ensure entry/exit logic names are present
            if 'entry_logic_name' not in full_trade_data:
                strategy_params = self.config['strategy'].get('parameters', {})
                entry_logic = strategy_params.get('entry_logic', {})
                full_trade_data['entry_logic_name'] = entry_logic.get('name', 'Unknown')

            if 'exit_logic_name' not in full_trade_data:
                strategy_params = self.config['strategy'].get('parameters', {})
                exit_logic = strategy_params.get('exit_logic', {})
                full_trade_data['exit_logic_name'] = exit_logic.get('name', 'Unknown')

            # Record trade in database
            try:
                result = trading_service.add_trade(full_trade_data)
                _logger.info(
                    "Recorded trade for bot %s: %s %s @ %.4f",
                    self.instance_id,
                    'BUY' if trade_data.get('entry_price') else 'SELL',
                    full_trade_data['symbol'],
                    trade_data.get('entry_price') or trade_data.get('exit_price', 0)
                )

                # Update performance metrics after trade
                self.update_performance_metrics()

                return result
            except Exception as e:
                _logger.error("Failed to record trade in database: %s", e)

        except Exception:
            _logger.exception("Error recording trade for %s:", self.name)

    def on_order_executed(self, order_type: str, price: float, size: float, timestamp: Optional[datetime] = None):
        """
        Callback for when an order is executed.

        This can be called from the trading strategy or broker to record trades.

        Args:
            order_type: 'buy' or 'sell'
            price: Execution price
            size: Order size
            timestamp: Execution timestamp (defaults to now)
        """
        try:
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)

            trade_data = {}
            pnl = None

            if order_type.lower() == 'buy':
                trade_data.update({
                    'entry_time': timestamp,
                    'entry_price': price,
                    'entry_value': price * size,
                    'buy_order_created': timestamp,
                    'buy_order_closed': timestamp,
                })
            elif order_type.lower() == 'sell':
                # Try to calculate PnL from last trade
                try:
                    result = trading_service.get_pnl_summary(self.instance_id)
                    if result:
                        pnl = result.get('net_pnl', 0)
                except Exception:
                    pass

                trade_data.update({
                    'exit_time': timestamp,
                    'exit_price': price,
                    'exit_value': price * size,
                    'sell_order_created': timestamp,
                    'sell_order_closed': timestamp,
                    'pnl': pnl
                })

            self.record_trade(trade_data)

            _logger.info(
                "Order executed for %s: %s %.4f @ %.4f",
                self.name, order_type.upper(), size, price
            )

            # Send notification (fire-and-forget)
            if self.notification_client:
                try:
                    asyncio.create_task(
                        self._send_trade_notification(order_type, price, size, pnl)
                    )
                except RuntimeError:
                    # No event loop running, schedule it differently
                    _logger.debug("No event loop for notification, skipping")

        except Exception:
            _logger.exception("Error in order execution callback for %s:", self.name)

    async def _send_trade_notification(self, order_type: str, price: float, size: float, pnl: Optional[float] = None):
        """
        Send notification for trade execution (async fire-and-forget).

        Args:
            order_type: 'buy' or 'sell'
            price: Execution price
            size: Order size
            pnl: Profit/loss for sell orders (optional)
        """
        try:
            if not self.notification_client:
                return

            # Get user notification details
            user_details = self._get_user_notification_details()
            if not user_details:
                return

            # Check if this notification type is enabled
            notif_config = self.config.get('notifications', {})
            if order_type.lower() == 'buy' and not notif_config.get('position_opened', False):
                return
            if order_type.lower() == 'sell' and not notif_config.get('position_closed', False):
                return

            # Get symbol and trading mode
            symbol = self.config.get('symbol', 'UNKNOWN')
            trading_mode = self.config['broker'].get('trading_mode', 'paper')

            # Format message
            if order_type.lower() == 'buy':
                title = f"Position Opened: {symbol}"
                message = f"BUY {size:.4f} {symbol} @ ${price:,.2f}"
                if trading_mode == 'paper':
                    message += " (Paper Trading)"
                notification_type = MessageType.TRADE_ENTRY
            else:
                title = f"Position Closed: {symbol}"
                message = f"SELL {size:.4f} {symbol} @ ${price:,.2f}"
                if pnl is not None:
                    pnl_pct = (pnl / (price * size)) * 100 if (price * size) > 0 else 0
                    message += f" (P&L: ${pnl:,.2f} / {pnl_pct:+.2f}%)"
                if trading_mode == 'paper':
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
                channels=user_details['channels'],
                recipient_id=user_details['recipient_id'],
                email_receiver=user_details['email'],
                telegram_chat_id=user_details.get('telegram_user_id'),
                data={
                    'bot_id': self.instance_id,
                    'bot_name': self.name,
                    'symbol': symbol,
                    'order_type': order_type.upper(),
                    'price': price,
                    'size': size,
                    'pnl': pnl,
                    'trading_mode': trading_mode,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                },
                source="trading_bot"
            )

            _logger.info("Sent %s notification for %s", order_type.upper(), self.name)

        except Exception:
            _logger.exception("Error sending trade notification for %s:", self.name)

    async def _send_error_notification(self, error_message: str, error_type: str = "ERROR"):
        """
        Send notification for bot errors (async fire-and-forget).

        Args:
            error_message: Error message to send
            error_type: Type of error (ERROR, WARNING, etc.)
        """
        try:
            if not self.notification_client:
                return

            # Get user notification details
            user_details = self._get_user_notification_details()
            if not user_details:
                return

            # Check if error notifications are enabled
            notif_config = self.config.get('notifications', {})
            if not notif_config.get('error_notifications', False):
                return

            # Format message
            symbol = self.config.get('symbol', 'UNKNOWN')
            trading_mode = self.config['broker'].get('trading_mode', 'paper')

            title = f"Bot Error: {self.name}"
            message = f"[{self.name}] {error_type}: {error_message}"
            if trading_mode == 'paper':
                message += " (Paper Trading)"

            # Send notification (fire-and-forget)
            await self.notification_client.send_notification(
                notification_type=MessageType.ERROR,
                title=title,
                message=message,
                priority=MessagePriority.CRITICAL,
                channels=user_details['channels'],
                recipient_id=user_details['recipient_id'],
                email_receiver=user_details['email'],
                telegram_chat_id=user_details.get('telegram_user_id'),
                data={
                    'bot_id': self.instance_id,
                    'bot_name': self.name,
                    'symbol': symbol,
                    'error_type': error_type,
                    'error_message': error_message,
                    'trading_mode': trading_mode,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                },
                source="trading_bot"
            )

            _logger.info("Sent error notification for %s", self.name)

        except Exception:
            _logger.exception("Error sending error notification for %s:", self.name)

    def _get_user_notification_details(self) -> Optional[Dict[str, Any]]:
        """
        Fetch user notification details (email, telegram_user_id) from database.

        Returns:
            Dict with 'user_id', 'email', 'telegram_user_id', 'recipient_id', 'channels'
            None if user not found or notifications disabled
        """
        try:
            # Get user_id from bot config
            user_id = self.config.get('user_id')
            if not user_id:
                _logger.debug("No user_id in bot config for %s, skipping notification", self.name)
                return None

            # Check if notifications are enabled for this bot
            notif_config = self.config.get('notifications', {})
            if not (notif_config.get('position_opened') or notif_config.get('position_closed') or notif_config.get('error_notifications')):
                _logger.debug("Notifications disabled for bot %s", self.name)
                return None

            # Get user from database by user_id
            # We need to query usr_users table and usr_auth_identities table
            from src.data.db.services.database_service import get_database_service
            from src.data.db.models.model_users import User, AuthIdentity

            db_service = get_database_service()
            with db_service.uow() as uow:
                # Get user by ID
                user = uow.s.get(User, user_id)
                if not user:
                    _logger.warning("User %s not found for bot %s", user_id, self.name)
                    return None

                # Get telegram identity
                from sqlalchemy import select
                telegram_identity = uow.s.execute(
                    select(AuthIdentity)
                    .where(AuthIdentity.user_id == user_id)
                    .where(AuthIdentity.provider == 'telegram')
                ).scalar_one_or_none()

                # Determine channels to use
                channels = []
                if notif_config.get('email_enabled') and user.email:
                    channels.append('email')
                if notif_config.get('telegram_enabled') and telegram_identity:
                    channels.append('telegram')

                if not channels:
                    _logger.debug("No notification channels enabled for bot %s", self.name)
                    return None

                return {
                    'user_id': user_id,
                    'email': user.email,
                    'telegram_user_id': telegram_identity.external_id if telegram_identity else None,
                    'recipient_id': str(user_id),
                    'channels': channels
                }

        except Exception:
            _logger.exception("Error fetching user notification details for bot %s:", self.name)
            return None


class StrategyManager:
    """
    Bot manager and SOLE CONFIGURATION LOADER.

    This is the ONLY component that loads configurations from the database.
    All other components receive configs through this manager.

    Responsibilities:
    - Load ALL bot configurations from database (ONLY place this happens)
    - Create and manage StrategyInstance objects
    - Handle bot lifecycle (start/stop/restart)
    - Monitor bot health and performance
    - Update database with bot status and metrics
    - DB polling for hot-reload
    """

    def __init__(self):
        """Initialize the strategy manager."""
        self.strategy_instances: Dict[str, StrategyInstance] = {}
        self.broker_manager = BrokerManager()

        # Global notification client (database-only mode for reliability)
        self.notification_client = NotificationServiceClient(service_url="database://")
        _logger.info("Initialized global notification client in database-only mode")

        self.is_running = False
        self.monitoring_task = None
        self.db_poll_task = None
        self._db_poll_running = False
        self._db_poll_user_id: Optional[int] = None
        self._db_poll_interval: int = 60

        # Crash recovery marker
        self._marker_path = Path(".trading_service_running")

    async def load_strategies_from_config(self, config_file: str) -> bool:
        """Load strategy configurations from JSON file."""
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                _logger.error("Configuration file not found: %s", config_path)
                return False

            with open(config_path, 'r') as f:
                config = json.load(f)

            strategies = config.get('strategies', [])
            if not strategies:
                _logger.error("No strategies found in configuration")
                return False

            _logger.info("Loading %d strategy configurations", len(strategies))

            for strategy_config in strategies:
                instance_id = strategy_config.get('id') or str(uuid.uuid4())

                # Validate required fields
                required_fields = ['name', 'symbol', 'broker', 'strategy']
                missing_fields = [field for field in required_fields if field not in strategy_config]

                if missing_fields:
                    _logger.error("Strategy %s: Missing required fields: %s", instance_id, missing_fields)
                    continue

                # Create strategy instance
                instance = StrategyInstance(instance_id, strategy_config, self.notification_client)
                self.strategy_instances[instance_id] = instance

                _logger.info("Loaded strategy: %s (%s)", instance.name, instance_id)

            _logger.info("Successfully loaded %d strategies", len(self.strategy_instances))
            return True

        except Exception:
            _logger.exception("Failed to load strategies from config:")
            return False

    async def start_all_strategies(self) -> int:
        """Start all configured strategy instances."""
        _logger.info("Starting all strategy instances...")

        started_count = 0

        for instance_id, instance in self.strategy_instances.items():
            # Check if strategy is enabled
            if not instance.config.get('enabled', True):
                _logger.info("Strategy %s is disabled, skipping", instance.name)
                continue

            success = await instance.start()
            if success:
                started_count += 1

            # Small delay between starts
            await asyncio.sleep(1)

        _logger.info("Started %d/%d strategy instances", started_count, len(self.strategy_instances))
        return started_count

    async def stop_all_strategies(self):
        """Stop all running strategy instances."""
        _logger.info("Stopping all strategy instances...")

        stop_tasks = []
        for instance in self.strategy_instances.values():
            if instance.status == 'running':
                task = asyncio.create_task(instance.stop())
                stop_tasks.append(task)

        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)

        _logger.info("All strategy instances stopped")

    async def start_strategy(self, instance_id: str) -> bool:
        """Start a specific strategy instance."""
        if instance_id not in self.strategy_instances:
            _logger.error("Strategy instance %s not found", instance_id)
            return False

        return await self.strategy_instances[instance_id].start()

    async def stop_strategy(self, instance_id: str) -> bool:
        """Stop a specific strategy instance."""
        if instance_id not in self.strategy_instances:
            _logger.error("Strategy instance %s not found", instance_id)
            return False

        return await self.strategy_instances[instance_id].stop()

    async def restart_strategy(self, instance_id: str) -> bool:
        """Restart a specific strategy instance."""
        if instance_id not in self.strategy_instances:
            _logger.error("Strategy instance %s not found", instance_id)
            return False

        return await self.strategy_instances[instance_id].restart()

    def get_all_status(self) -> List[Dict[str, Any]]:
        """Get status of all strategy instances."""
        return [instance.get_status() for instance in self.strategy_instances.values()]

    def get_strategy_status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific strategy instance."""
        if instance_id not in self.strategy_instances:
            return None

        return self.strategy_instances[instance_id].get_status()

    async def start_monitoring(self):
        """Start monitoring all strategy instances."""
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitor_strategies())

    async def stop_monitoring(self):
        """Stop monitoring."""
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        # Stop DB polling if running
        self._db_poll_running = False
        if self.db_poll_task:
            self.db_poll_task.cancel()
            try:
                await self.db_poll_task
            except asyncio.CancelledError:
                pass

    async def _monitor_strategies(self):
        """Monitor strategy instances and handle auto-recovery."""
        while self.is_running:
            try:
                current_time = datetime.now(timezone.utc)
                running_count = 0
                unhealthy_count = 0

                # Check health of all running strategies
                for instance_id, instance in self.strategy_instances.items():
                    if instance.status == 'running':
                        running_count += 1

                        # Check heartbeat health
                        if instance.last_heartbeat:
                            heartbeat_age = (current_time - instance.last_heartbeat).total_seconds()
                            max_heartbeat_age = instance.heartbeat_interval * 3  # 3x normal interval

                            if heartbeat_age > max_heartbeat_age:
                                unhealthy_count += 1
                                _logger.warning(
                                    "Bot %s heartbeat stale (%.1fs old, max %.1fs). Attempting recovery...",
                                    instance.name, heartbeat_age, max_heartbeat_age
                                )
                                # Try to restart unhealthy bot
                                if instance.error_count < 3:
                                    await instance.restart()
                                else:
                                    _logger.error(
                                        "Bot %s exceeded max restart attempts (%d), marking as error",
                                        instance.name, instance.error_count
                                    )
                                    instance.status = 'error'
                                    try:
                                        trading_service.update_bot_status(
                                            int(instance_id),
                                            "error",
                                            error_message=f"Exceeded max restart attempts ({instance.error_count})"
                                        )
                                    except Exception:
                                        pass

                    elif instance.status == 'error' and instance.error_count < 3:
                        # Auto-recovery for failed strategies (max 3 attempts)
                        _logger.warning("Attempting auto-recovery for %s (attempt %d/3)",
                                      instance.name, instance.error_count + 1)
                        await instance.restart()

                # Log periodic status with detailed metrics
                _logger.info(
                    "📊 Strategy Monitor: %d/%d running, %d unhealthy, %d total",
                    running_count, len(self.strategy_instances), unhealthy_count, len(self.strategy_instances)
                )

                await asyncio.sleep(60)  # Monitor every minute

            except Exception:
                _logger.exception("Error in strategy monitoring:")
                await asyncio.sleep(10)

    # -------------------- Crash Detection and Recovery --------------------
    def _detect_crash_recovery(self) -> bool:
        """
        Detect if this is a crash recovery (unclean shutdown).

        Uses a marker file that's created on startup and deleted on clean shutdown.
        If marker exists on startup, previous shutdown was unclean.

        Returns:
            True if previous shutdown was unclean (crash detected), False otherwise
        """
        if self._marker_path.exists():
            _logger.warning("⚠️ Found running marker from previous session - UNCLEAN SHUTDOWN detected")
            _logger.warning("This indicates the service crashed or was forcefully terminated")
            self._marker_path.unlink()  # Remove stale marker
            return True

        _logger.info("Clean startup detected - no previous running marker found")
        return False

    def _mark_service_running(self):
        """Mark that service is now running by creating marker file."""
        try:
            self._marker_path.touch()
            _logger.info("Created session marker file: %s", self._marker_path)
        except Exception as e:
            _logger.warning("Failed to create session marker: %s", e)

    def _mark_clean_shutdown(self):
        """Mark that service is shutting down cleanly by removing marker file."""
        try:
            if self._marker_path.exists():
                self._marker_path.unlink()
                _logger.info("Removed session marker - CLEAN SHUTDOWN")
            else:
                _logger.debug("Session marker already removed")
        except Exception as e:
            _logger.warning("Failed to remove session marker: %s", e)

    def _recover_bot_state(self, bot_id: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recover bot state from database on restart.

        Queries open positions and pending trades to reconstruct state context
        that can be used by the strategy on restart.

        Args:
            bot_id: Bot ID to recover state for
            config: Bot configuration dictionary

        Returns:
            Enhanced config with recovered state in '_recovered_positions' and '_recovered_trades'
        """
        try:
            # Get open positions
            open_positions = trading_service.get_open_positions(bot_id=str(bot_id))

            if open_positions:
                _logger.info(
                    "🔄 Recovered %d open position(s) for bot %d",
                    len(open_positions), bot_id
                )
                # Store positions in config for strategy to access
                config['_recovered_positions'] = open_positions

                # Log position details
                for pos in open_positions:
                    _logger.info(
                        "  Position: %s %s qty=%.8f avg_price=%.8f status=%s",
                        pos.get('symbol'),
                        pos.get('direction'),
                        pos.get('qty_open', 0),
                        pos.get('avg_price', 0),
                        pos.get('status')
                    )

            # Get open trades
            open_trades = trading_service.get_open_trades()
            bot_trades = [t for t in open_trades if t['bot_id'] == bot_id]

            if bot_trades:
                _logger.info(
                    "🔄 Recovered %d open trade(s) for bot %d",
                    len(bot_trades), bot_id
                )
                config['_recovered_trades'] = bot_trades

                # Log trade details
                for trade in bot_trades:
                    _logger.info(
                        "  Trade: %s entry=%.8f @ %s status=%s",
                        trade.get('symbol'),
                        trade.get('entry_price', 0),
                        trade.get('entry_time'),
                        trade.get('status')
                    )

            return config

        except Exception:
            _logger.exception("Error recovering state for bot %d:", bot_id)
            return config

    # -------------------- DB-backed loading and polling --------------------
    async def load_strategies_from_db(self, user_id: Optional[int] = None, resume_mode: bool = True) -> bool:
        """
        Load strategy configurations from the database - ONLY PLACE THIS HAPPENS.

        This is the SOLE configuration loader. No other component should load from DB.

        Behavior:
        - In resume_mode: Detects crash recovery and resumes only previously running bots
        - In normal mode: Loads all enabled bots
        - Validates each bot config using trading_service AND StrategyHandler
        - Recovers bot state (positions, trades) for crash recovery
        - Builds or updates StrategyInstance entries keyed by bot id
        - Invalid configs are skipped and marked as 'error' in DB

        Args:
            user_id: Optional user ID to filter bots
            resume_mode: If True, use smart resume logic (default: True)

        Returns:
            True if successfully loaded at least one bot, False otherwise
        """
        try:
            _logger.info("=" * 80)
            _logger.info("LOADING BOT CONFIGURATIONS FROM DATABASE (SOLE CONFIG LOADER)")
            _logger.info("=" * 80)

            # Check for crash recovery
            was_crashed = False
            if resume_mode:
                was_crashed = self._detect_crash_recovery()

                if was_crashed:
                    _logger.warning("🔄 CRASH RECOVERY MODE: Resuming previously running bots")
                    # Load only bots that were running before crash
                    bots = trading_service.get_bots_by_status("running", user_id)
                    _logger.info("Found %d bot(s) that were running before crash", len(bots))
                else:
                    _logger.info("🚀 NORMAL STARTUP: Loading all enabled bots")
                    bots = trading_service.get_enabled_bots(user_id)
                    _logger.info("Found %d enabled bot(s) in database", len(bots))
            else:
                _logger.info("🚀 NORMAL STARTUP (resume_mode=False): Loading all enabled bots")
                bots = trading_service.get_enabled_bots(user_id)
                _logger.info("Found %d enabled bot(s) in database", len(bots))

            # Create marker file to track this session
            self._mark_service_running()

            if not bots:
                _logger.warning("No bots to load%s",
                                f" for user_id={user_id}" if user_id else "")
                return False

            loaded = 0
            for bot in bots:
                instance_id = str(bot["id"])
                bot_name = bot.get("description") or f"Bot {bot['id']}"

                _logger.info("Processing bot: %s (ID: %s)", bot_name, instance_id)

                # Validate database record/config
                is_valid, errors, warnings = trading_service.validate_bot_configuration(bot["id"])
                if not is_valid:
                    _logger.error("Bot %s config invalid, skipping. Errors: %s", bot["id"], errors)
                    try:
                        trading_service.update_bot_status(bot["id"], "error", error_message="; ".join(errors))
                    except Exception:
                        pass
                    continue

                # Log warnings
                for warning in warnings:
                    _logger.warning("Bot %s: %s", bot["id"], warning)

                # Map db record to StrategyInstance config
                si_config = self._db_bot_to_strategy_config(bot)

                # Recover state if this is a crash recovery
                if was_crashed:
                    _logger.info("Recovering state for bot %s...", bot["id"])
                    si_config = self._recover_bot_state(bot["id"], si_config)

                # Additional validation using StrategyHandler
                strategy_type = si_config.get("strategy", {}).get("type", "CustomStrategy")
                strategy_config = si_config.get("strategy", {})

                is_strategy_valid, strategy_errors, strategy_warnings = strategy_handler.validate_strategy_config(
                    strategy_type,
                    strategy_config
                )

                if not is_strategy_valid:
                    _logger.error("Bot %s strategy config invalid: %s", bot["id"], strategy_errors)
                    try:
                        trading_service.update_bot_status(
                            bot["id"],
                            "error",
                            error_message="Strategy validation failed: " + "; ".join(strategy_errors)
                        )
                    except Exception:
                        pass
                    continue

                # Log strategy warnings
                for warning in strategy_warnings:
                    _logger.warning("Bot %s strategy: %s", bot["id"], warning)

                # Create or update instance
                if instance_id in self.strategy_instances:
                    # Update existing config
                    _logger.info("Updating existing bot instance: %s", bot_name)
                    self.strategy_instances[instance_id].config = si_config
                else:
                    # Create new instance
                    _logger.info("Creating new bot instance: %s", bot_name)
                    self.strategy_instances[instance_id] = StrategyInstance(instance_id, si_config, self.notification_client)

                loaded += 1
                _logger.info("✅ Successfully loaded bot: %s", bot_name)

            _logger.info("=" * 80)
            _logger.info("CONFIGURATION LOADING COMPLETE: %d/%d bots loaded", loaded, len(bots))
            _logger.info("=" * 80)

            return loaded > 0

        except Exception:
            _logger.exception("Failed to load strategies from DB:")
            return False

    def _db_bot_to_strategy_config(self, bot: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a trading_bots row dict into StrategyInstance config shape."""
        cfg = bot.get("config") or {}
        # Prefer fields from config; fallback to DB row
        name = cfg.get("name") or bot.get("description") or f"Bot {bot.get('id')}"
        symbol = cfg.get("symbol") or cfg.get("trading_pair") or cfg.get("pair") or cfg.get("ticker")
        broker_cfg = cfg.get("broker") or {}
        strategy_cfg = cfg.get("strategy") or {}

        # Enabled if DB status not 'disabled' and config doesn't explicitly disable
        enabled = bot.get("status") != "disabled" and cfg.get("enabled", True)

        return {
            "id": str(bot.get("id")),
            "name": name,
            "enabled": enabled,
            "symbol": symbol or "BTCUSDT",
            "broker": broker_cfg,
            "strategy": strategy_cfg,
            "data": cfg.get("data", {}),
            "trading": cfg.get("trading", {}),
            "risk_management": cfg.get("risk_management", {}),
            "notifications": cfg.get("notifications", {}),
        }

    async def start_db_polling(self, user_id: Optional[int] = None, interval_seconds: int = 60):
        """Continuously poll DB and sync strategy processes.

        Actions on each poll:
        - Start new StrategyInstances for bots present in DB with status 'enabled' that aren't running yet.
        - Stop and remove instances for bots that became 'disabled' or were deleted.
        - Refresh configs for bots that changed.
        - Update bot statuses in DB to 'running' when started and 'stopped' when stopped.
        """
        self._db_poll_user_id = user_id
        self._db_poll_interval = max(5, interval_seconds)
        self._db_poll_running = True

        # Initial load
        await self.load_strategies_from_db(user_id)

        async def _poll_loop():
            while self._db_poll_running:
                try:
                    # Fetch enabled bots (status != disabled)
                    enabled_bots = {str(b["id"]): b for b in trading_service.get_enabled_bots(user_id)}

                    # Start or update instances for enabled bots
                    for bot_id, bot in enabled_bots.items():
                        is_valid, errors, _ = trading_service.validate_bot_configuration(int(bot_id))
                        if not is_valid:
                            _logger.error("Bot %s invalid during polling, skipping start. Errors: %s", bot_id, errors)
                            try:
                                trading_service.update_bot_status(int(bot_id), "error", error_message="; ".join(errors))
                            except Exception:
                                pass
                            continue

                        desired_status = bot.get("status", "enabled")
                        exists = bot_id in self.strategy_instances
                        if not exists:
                            # Create instance
                            si_config = self._db_bot_to_strategy_config(bot)
                            self.strategy_instances[bot_id] = StrategyInstance(bot_id, si_config, self.notification_client)

                        instance = self.strategy_instances[bot_id]
                        # If bot should be running and isn't, start it
                        if desired_status in ("enabled", "starting") and instance.status != "running":
                            try:
                                trading_service.update_bot_status(int(bot_id), "starting")
                            except Exception:
                                pass
                            ok = await instance.start()
                            try:
                                trading_service.update_bot_status(int(bot_id), "running" if ok else "error",
                                                                 error_message=None if ok else instance.last_error)
                            except Exception:
                                pass

                        # If bot is running but config changed, we could implement hot-reload: simple restart
                        else:
                            # Refresh config in case changed
                            instance.config = self._db_bot_to_strategy_config(bot)

                    # Stop instances for bots no longer enabled
                    to_stop = [iid for iid in list(self.strategy_instances.keys()) if iid not in enabled_bots]
                    for iid in to_stop:
                        instance = self.strategy_instances.get(iid)
                        if instance and instance.status == "running":
                            await instance.stop()
                            try:
                                trading_service.update_bot_status(int(iid), "stopped")
                            except Exception:
                                pass
                        # Remove from manager
                        self.strategy_instances.pop(iid, None)

                    await asyncio.sleep(self._db_poll_interval)

                except asyncio.CancelledError:
                    break
                except Exception:
                    _logger.exception("Error during DB polling loop:")
                    await asyncio.sleep(self._db_poll_interval)

        # Launch poll loop
        self.db_poll_task = asyncio.create_task(_poll_loop())

    async def shutdown(self):
        """
        Gracefully shutdown the strategy manager with state persistence.

        This method ensures:
        - All running bots are stopped cleanly
        - Bot statuses are persisted to database
        - Resources are properly released
        - Clean shutdown is marked (no crash marker on restart)
        """
        _logger.info("🛑 Shutting down Enhanced Strategy Manager...")

        try:
            # Stop monitoring first
            await self.stop_monitoring()

            # Stop all strategy instances and persist their status
            _logger.info("Stopping all strategy instances and persisting statuses...")
            for instance_id, instance in self.strategy_instances.items():
                if instance.status == 'running':
                    _logger.info("Stopping bot %s (%s)", instance.name, instance_id)

                    # Stop the bot
                    await instance.stop()

                    # Persist stopped status to database
                    try:
                        trading_service.update_bot_status(
                            int(instance_id),
                            "stopped",
                            error_message=None  # Clear any error on clean shutdown
                        )
                        _logger.debug("Persisted stopped status for bot %s", instance_id)
                    except Exception as e:
                        _logger.warning("Failed to persist status for bot %s: %s", instance_id, e)

            # Close broker manager
            _logger.info("Closing broker manager...")
            await self.broker_manager.shutdown()

            # Close notification client
            if self.notification_client:
                _logger.info("Closing notification client...")
                await self.notification_client.close()

            # Mark clean shutdown (remove crash marker)
            self._mark_clean_shutdown()

            _logger.info("✅ Enhanced Strategy Manager shutdown complete")

        except Exception:
            _logger.exception("Error during shutdown:")
            # Don't mark clean shutdown if errors occurred
            _logger.warning("⚠️ Shutdown completed with errors - crash marker not removed")
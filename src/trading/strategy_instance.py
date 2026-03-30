"""
Strategy Instance
-----------------
Represents a single running instance of a trading strategy with its own 
broker connection, data feed, and Backtrader engine.
"""

import asyncio
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.trading.broker.backtrader_availability import BACKTRADER_AVAILABLE
from src.trading.broker.broker_factory import get_broker
from src.trading.strategy_handler import strategy_handler
from src.data.feed.data_feed_factory import DataFeedFactory
from src.trading.base_trading_bot import BaseTradingBot
from src.notification.logger import setup_logger
from src.data.db.services.trading_service import trading_service
from src.notification.service.client import NotificationServiceClient, MessageType, MessagePriority

_logger = setup_logger(__name__)

class StrategyInstance:
    """Represents a single strategy instance with its own configuration and broker."""

    def __init__(self, instance_id: str, config: Dict[str, Any], 
                 notification_client: Optional[NotificationServiceClient] = None, 
                 trade_repository: Any = None):
        """Initialize strategy instance."""
        self.instance_id = instance_id
        self.config = config
        self.name = config.get('name', f'Strategy_{instance_id}')
        self.notification_client = notification_client
        self.trade_repository = trade_repository

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
        """Start the strategy instance with full Backtrader integration."""
        try:
            _logger.info("Starting strategy instance: %s", self.name)

            # Create broker
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

            # Get strategy class
            strategy_config = self.config['strategy']
            strategy_class = self._get_strategy_class(strategy_config['type'])
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
            self.status = 'running'
            self.start_time = datetime.now(timezone.utc)
            self.is_running = True
            self.error_count = 0
            self.last_error = None

            # Update DB
            try:
                trading_service.update_bot_status(self.instance_id, "running", started_at=self.start_time)
            except Exception as e:
                _logger.warning("Failed to update bot status in DB: %s", e)

            # Start processing loops
            asyncio.create_task(self._run_backtrader_async())
            asyncio.create_task(self._start_trading_bot_loop())

            _logger.info("✅ Strategy instance %s started successfully", self.name)
            return True

        except Exception as e:
            self.status = 'error'
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
        self.monitor_thread = threading.Thread(
            target=self._monitor_data_feed,
            daemon=True,
            name=f"Monitor-{self.name}"
        )
        self.monitor_thread.start()
        
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"Heartbeat-{self.name}"
        )
        self.heartbeat_thread.start()

    async def stop(self) -> bool:
        """Stop the strategy instance gracefully."""
        try:
            _logger.info("Stopping strategy instance: %s", self.name)
            self.should_stop = True
            self.is_running = False

            if self.data_feed:
                try: self.data_feed.stop()
                except Exception: pass

            await self._stop_trading_bot()

            if self.broker:
                try: await self.broker.disconnect()
                except Exception: pass

            self.status = 'stopped'
            try:
                trading_service.update_bot_status(self.instance_id, "stopped")
            except Exception: pass

            _logger.info("✅ Strategy instance %s stopped successfully", self.name)
            return True
        except Exception as e:
            self.status = 'error'
            _logger.exception("❌ Failed to stop strategy instance %s:", self.name)
            return False

    async def restart(self) -> bool:
        """Restart the strategy instance."""
        await self.stop()
        await asyncio.sleep(2)
        return await self.start()

    def get_status(self) -> Dict[str, Any]:
        """Get strategy instance status summary."""
        uptime = 0
        if self.start_time:
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()

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
        """Get strategy class based on type using StrategyHandler."""
        return strategy_handler.get_strategy_class(strategy_type)

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

    async def _create_trading_bot(self) -> bool:
        """Instantiate the BaseTradingBot for this instance."""
        try:
            bot_config = self._build_bot_config()
            strategy_class = self._get_strategy_class(self.config['strategy']['type'])
            parameters = self.config['strategy'].get('parameters', {})
            paper = (self.config['broker'].get('trading_mode', 'paper') == 'paper')

            self.trading_bot = BaseTradingBot(
                config=bot_config,
                strategy_class=strategy_class,
                parameters=parameters,
                broker=self.broker,
                paper_trading=paper,
                bot_id=str(self.instance_id),
                trade_repository=self.trade_repository
            )
            return True
        except Exception:
            _logger.exception("Error creating BaseTradingBot for %s:", self.name)
            return False

    async def _start_trading_bot_loop(self):
        """Run the BaseTradingBot heartbeat/execution loop."""
        if not self.trading_bot: return
        try:
            await asyncio.get_event_loop().run_in_executor(None, self.trading_bot.run)
        except Exception:
            _logger.exception("Error in BaseTradingBot loop for %s:", self.name)
            self.status = 'error'

    async def _stop_trading_bot(self):
        """Stop the trading bot gracefully."""
        if self.trading_bot:
            try: self.trading_bot.stop()
            except Exception: pass

    def _create_data_feed(self) -> bool:
        """Create and initialize the data feed."""
        try:
            data_config = self.config.get('data', {
                'data_source': self.config['broker'].get('type', 'binance'),
                'symbol': self.config.get('symbol', 'BTCUSDT'),
                'interval': '1h',
                'lookback_bars': 500
            })

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
            
            strategy_params = self.config['strategy'].get('parameters', {})
            self.cerebro.addstrategy(
                strategy_class, 
                strategy_config=strategy_params,
                on_signal_callback=self.trading_bot.add_signal if self.trading_bot else None
            )

            # Setup broker
            broker_config = self.config['broker']
            if broker_config.get('type', '').lower() == 'backtrader':
                self.cerebro.broker.setcash(broker_config.get('cash', 10000.0))
                self.cerebro.broker.setcommission(commission=broker_config.get('commission', 0.001))
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
            await asyncio.get_event_loop().run_in_executor(None, self.cerebro.run)
        except Exception as e:
            _logger.exception("Error in Backtrader engine for %s:", self.name)
            self.status = 'error'

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
        """Main heartbeat loop for the instance."""
        while self.is_running and not self.should_stop:
            try:
                self.last_heartbeat = datetime.now(timezone.utc)
                # Heartbeat logic (e.g. reporting to DB)
                time.sleep(self.heartbeat_interval)
            except Exception:
                time.sleep(5)

    async def _send_trade_notification(self, order_type: str, price: float, size: float, pnl: float = None):
        """
        Send notification for trade execution (async fire-and-forget).
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
        Fetch user notification details from database.
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

            # Import here to avoid circular dependencies
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

    def on_order_executed(self, order):
        """Callback for order execution from strategy."""
        _logger.info("[%s] Order executed: %s", self.name, order)
        # Wire up notifications here as well
        side = "buy" if order.isbuy() else "sell"
        # Since BT order might not have easy PnL access here without refinement,
        # we'll use a placeholder or enhance later.
        asyncio.create_task(self._send_trade_notification(side, order.executed.price, order.executed.size))

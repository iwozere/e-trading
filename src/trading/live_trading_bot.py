"""
Live Trading Bot Module
----------------------

This module implements a comprehensive live trading bot that:
1. Reads configuration from JSON files
2. Constructs data feeds, strategies, and brokers
3. Manages live trading with real-time data
4. Handles error recovery and notifications
5. Integrates with existing components

Main Features:
- Configuration-driven setup
- Live data feed integration
- Strategy execution with Backtrader
- Position management and persistence
- Error handling and recovery
- Notification system integration
- Web interface integration

Classes:
- LiveTradingBot: Main live trading bot implementation
"""

import json
import os
import signal
import sys
import time
import threading
from datetime import datetime
from typing import Any, Dict, Optional

import backtrader as bt

from src.broker.broker_factory import get_broker
from src.data.data_feed_factory import DataFeedFactory
from src.notification.logger import setup_logger
from src.strategy.custom_strategy import CustomStrategy
from src.strategy.advanced_strategy_framework import AdvancedStrategyFramework
from src.trading.base_trading_bot import BaseTradingBot

_logger = setup_logger(__name__)

STRATEGY_REGISTRY = {
    "CustomStrategy": CustomStrategy,
    "AdvancedStrategyFramework": AdvancedStrategyFramework,
    # Add more strategies here
}

class LiveTradingBot(BaseTradingBot):
    """
    Comprehensive live trading bot that orchestrates all components.
    
    This bot inherits from BaseTradingBot and extends it with:
    - Live data feed management
    - Backtrader integration
    - Real-time strategy execution
    - Enhanced error handling and recovery
    
    It leverages BaseTradingBot's:
    - Position management
    - Trade history tracking
    - Notification system
    - State persistence
    - Balance management
    """
    
    def __init__(self, config_file: str):
        """
        Initialize the live trading bot.
        
        Args:
            config_file: Path to configuration file (e.g., '0001.json')
        """
        # Load configuration first
        self.config_file = config_file
        self.config = self._load_configuration()
        self._validate_configuration()
        
        # Extract components for BaseTradingBot
        broker = self._create_broker()
        strategy_name = self.config["strategy"].get("name", "CustomStrategy")
        strategy_class = STRATEGY_REGISTRY.get(strategy_name)
        if strategy_class is None:
            raise ValueError(f"Unknown strategy class: {strategy_name}")
        parameters = self._create_strategy_parameters()
        
        # Initialize BaseTradingBot
        super().__init__(
            config=self.config,
            strategy_class=strategy_class,
            parameters=parameters,
            broker=broker,
            paper_trading=self.config["broker"].get("type") == "binance_paper",
            bot_id=self.config_file  # Use config filename as bot_id
        )
        
        # LiveTradingBot specific attributes
        self.data_feed = None
        self.cerebro = None
        self.should_stop = False
        self.error_count = 0
        self.max_errors = 5
        self.error_retry_interval = 60  # seconds
        
        # Threading
        self.main_thread = None
        self.monitor_thread = None
        
        # Override trading pair from config
        self.trading_pair = self.config["trading"]["symbol"]
        
        # Load open positions
        self._load_open_positions()
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            config_path = f"config/trading/{self.config_file}"
            with open(config_path, 'r') as f:
                config = json.load(f)
            _logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            _logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _validate_configuration(self):
        """Validate configuration parameters."""
        required_sections = ["broker", "trading", "data", "strategy"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate data feed configuration
        data_config = self.config["data"]
        if "data_source" not in data_config:
            raise ValueError("Missing data_source in data configuration")
        
        # Validate strategy configuration
        strategy_config = self.config["strategy"]
        if strategy_config.get("type") != "custom":
            raise ValueError("Only 'custom' strategy type is supported")
        
        if "entry_logic" not in strategy_config:
            raise ValueError("Missing entry_logic in strategy configuration")
        
        if "exit_logic" not in strategy_config:
            raise ValueError("Missing exit_logic in strategy configuration")
        
        _logger.info("Configuration validation passed")
    
    def _create_broker(self):
        """Create and initialize the broker."""
        try:
            broker_config = self.config["broker"]
            broker = get_broker(broker_config)
            _logger.info(f"Created broker: {broker_config.get('type', 'unknown')}")
            return broker
        except Exception as e:
            _logger.error(f"Error creating broker: {e}")
            raise
    
    def _create_strategy_parameters(self) -> Dict[str, Any]:
        """Create the strategy parameters for BaseTradingBot."""
        try:
            strategy_config = self.config["strategy"]
            trading_config = self.config["trading"]
            
            # Build strategy parameters
            parameters = {
                "strategy_config": {
                    "entry_logic": strategy_config["entry_logic"],
                    "exit_logic": strategy_config["exit_logic"],
                    "position_size": trading_config.get("position_size", 0.1),
                    "use_talib": strategy_config.get("use_talib", False)
                }
            }
            
            _logger.info(f"Created strategy parameters for: {strategy_config.get('type', 'custom')}")
            return parameters
            
        except Exception as e:
            _logger.error(f"Error creating strategy parameters: {e}")
            raise
    
    def _create_data_feed(self):
        """Create and initialize the data feed."""
        try:
            data_config = self.config["data"]
            
            # Add callback for new data notifications
            def on_new_bar(symbol, timestamp, data):
                self._notify_new_bar(symbol, timestamp, data)
            data_config["on_new_bar"] = on_new_bar
            
            self.data_feed = DataFeedFactory.create_data_feed(data_config)
            
            if self.data_feed is None:
                raise ValueError("Failed to create data feed")
            
            _logger.info(f"Created data feed for {data_config.get('symbol', 'unknown')}")
            return True
            
        except Exception as e:
            _logger.error(f"Error creating data feed: {e}")
            return False
    
    def _setup_backtrader(self):
        """Setup Backtrader engine."""
        try:
            self.cerebro = bt.Cerebro()
            
            # Add data feed
            self.cerebro.adddata(self.data_feed)
            
            # Add strategy
            self.cerebro.addstrategy(self.strategy_class, **self.parameters)
            
            # Setup broker
            if self.broker:
                self.cerebro.broker = self.broker
            
            # Setup initial cash
            initial_balance = self.config["broker"].get("initial_balance", 1000.0)
            self.cerebro.broker.setcash(initial_balance)
            
            # Setup commission
            commission = self.config["broker"].get("commission", 0.001)
            self.cerebro.broker.setcommission(commission=commission)
            
            _logger.info(f"Setup Backtrader with initial balance: {initial_balance}")
            return True
            
        except Exception as e:
            _logger.error(f"Error setting up Backtrader: {e}")
            return False
    
    def _load_open_positions(self):
        """Load open positions from database or state file."""
        try:
            # Use BaseTradingBot's load_state method
            self.load_state()
            _logger.info(f"Loaded {len(self.active_positions)} open positions")
            return True
            
        except Exception as e:
            _logger.error(f"Error loading open positions: {e}")
            return False
    
    def _notify_new_bar(self, symbol: str, timestamp, data: Dict[str, Any]):
        """Notify about new data bar."""
        try:
            if self.telegram_notifier:
                message = f"📊 New {symbol} bar: O={data['open']:.4f} H={data['high']:.4f} L={data['low']:.4f} C={data['close']:.4f}"
                # Use BaseTradingBot's notification method
                self.log_message(message)
        except Exception as e:
            _logger.error(f"Error notifying new bar: {e}")
    
    def _monitor_data_feed(self):
        """Monitor data feed health and reconnect if needed."""
        while self.is_running and not self.should_stop:
            try:
                if self.data_feed:
                    status = self.data_feed.get_status()
                    if not status.get("is_connected", False):
                        _logger.warning("Data feed disconnected, attempting to reconnect...")
                        self._reconnect_data_feed()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                _logger.error(f"Error in data feed monitor: {e}")
                time.sleep(60)
    
    def _reconnect_data_feed(self):
        """Reconnect data feed."""
        try:
            if self.data_feed:
                self.data_feed.stop()
                time.sleep(5)
            
            if self._create_data_feed():
                self._setup_backtrader()
                _logger.info("Data feed reconnected successfully")
                # Use BaseTradingBot's notification method
                self.notify_bot_event("data_feed_reconnected", "🔌")
            else:
                _logger.error("Failed to reconnect data feed")
                self.notify_error("Failed to reconnect data feed")
                
        except Exception as e:
            _logger.error(f"Error reconnecting data feed: {e}")
    
    def _run_backtrader(self):
        """Run Backtrader engine."""
        try:
            _logger.info("Starting Backtrader engine...")
            self.notify_bot_event("backtrader_started", "🚀")
            
            # Run Backtrader
            results = self.cerebro.run()
            
            _logger.info("Backtrader engine completed")
            return True
            
        except Exception as e:
            _logger.error(f"Error in Backtrader engine: {e}")
            self.notify_error(f"Backtrader error: {str(e)}")
            return False
    
    def start(self):
        """Start the live trading bot."""
        try:
            _logger.info(f"Starting live trading bot: {self.config_file}")
            self.notify_bot_event("started", "🤖")
            
            # Initialize components
            if not self._create_data_feed():
                raise RuntimeError("Failed to create data feed")
            
            if not self._setup_backtrader():
                raise RuntimeError("Failed to setup Backtrader")
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitor_data_feed, daemon=True)
            self.monitor_thread.start()
            
            # Set running flag
            self.is_running = True
            
            # Start main trading loop
            self._run_backtrader()
            
        except Exception as e:
            _logger.error(f"Error starting bot: {e}")
            self.notify_error(f"Failed to start bot: {str(e)}")
            raise
    
    def stop(self):
        """Stop the live trading bot."""
        try:
            _logger.info("Stopping live trading bot...")
            self.notify_bot_event("stopping", "🛑")
            
            self.should_stop = True
            self.is_running = False
            
            # Stop data feed
            if self.data_feed:
                self.data_feed.stop()
            
            # Use BaseTradingBot's stop method
            super().stop()
            
            # Save state using BaseTradingBot's method
            self.save_state()
            
            _logger.info("Live trading bot stopped")
            self.notify_bot_event("stopped", "🛑")
            
        except Exception as e:
            _logger.error(f"Error stopping bot: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status."""
        try:
            # Get base status from BaseTradingBot
            status = {
                "config_file": self.config_file,
                "is_running": self.is_running,
                "should_stop": self.should_stop,
                "error_count": self.error_count,
                "trading_pair": self.trading_pair,
                "current_balance": self.current_balance,
                "total_pnl": self.total_pnl,
                "active_positions": len(self.active_positions),
                "trade_history_count": len(self.trade_history),
                "data_feed_status": None,
                "broker_status": None,
                "strategy_status": None
            }
            
            if self.data_feed:
                status["data_feed_status"] = self.data_feed.get_status()
            
            if self.broker:
                status["broker_status"] = {
                    "type": self.config["broker"].get("type"),
                    "cash": getattr(self.cerebro.broker, 'cash', 0) if self.cerebro else 0
                }
            
            if hasattr(self, 'parameters') and self.parameters:
                strategy_config = self.parameters.get("strategy_config", {})
                status["strategy_status"] = {
                    "type": "custom",
                    "entry_logic": strategy_config.get("entry_logic", {}).get("name", "unknown"),
                    "exit_logic": strategy_config.get("exit_logic", {}).get("name", "unknown")
                }
            
            return status
            
        except Exception as e:
            _logger.error(f"Error getting status: {e}")
            return {"error": str(e)}
    
    def restart(self):
        """Restart the live trading bot."""
        try:
            _logger.info("Restarting live trading bot...")
            self.notify_bot_event("restarting", "🔄")
            
            self.stop()
            time.sleep(5)  # Wait for cleanup
            
            # Reset state
            self.should_stop = False
            self.error_count = 0
            
            # Start again
            self.start()
            
        except Exception as e:
            _logger.error(f"Error restarting bot: {e}")
            self.notify_error(f"Failed to restart bot: {str(e)}")
    
    def execute_trade(self, trade_type: str, price: float, size: float) -> None:
        """
        Override BaseTradingBot's execute_trade to integrate with Backtrader.
        """
        try:
            # Use BaseTradingBot's trade execution logic
            super().execute_trade(trade_type, price, size)
            
            # Additional live trading specific logic
            if self.data_feed:
                # Update data feed status if needed
                pass
                
        except Exception as e:
            _logger.error(f"Error executing trade: {e}")
            self.notify_error(f"Trade execution error: {str(e)}")


def main():
    """Main function to run the live trading bot."""
    if len(sys.argv) != 2:
        print("Usage: python live_trading_bot.py <config_file>")
        print("Example: python live_trading_bot.py 0001.json")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    # Create and start bot
    bot = LiveTradingBot(config_file)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        _logger.info(f"Received signal {signum}, shutting down...")
        bot.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        bot.start()
    except KeyboardInterrupt:
        _logger.info("Received keyboard interrupt, shutting down...")
        bot.stop()
    except Exception as e:
        _logger.error(f"Unexpected error: {e}")
        bot.stop()
        sys.exit(1)


if __name__ == "__main__":
    main() 
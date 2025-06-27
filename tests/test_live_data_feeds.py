#!/usr/bin/env python3
"""
Test Live Data Feeds
-------------------

This script demonstrates how to use the live data feeds with different data sources.
It shows how to create data feeds for Binance, Yahoo Finance, and IBKR.

Usage:
    python test_live_data_feeds.py [data_source] [symbol] [interval]

Examples:
    python test_live_data_feeds.py binance BTCUSDT 1m
    python test_live_data_feeds.py yahoo AAPL 5m
    python test_live_data_feeds.py ibkr SPY 1m
"""

import sys
import time
import json
import signal
from datetime import datetime
from typing import Dict, Any

import backtrader as bt

from src.data.data_feed_factory import DataFeedFactory, DataFeedFactory
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def on_new_bar(symbol: str, timestamp, data: Dict[str, Any]):
    """Callback function called when new data arrives."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] New {symbol} bar: "
          f"O={data['open']:.4f} H={data['high']:.4f} L={data['low']:.4f} "
          f"C={data['close']:.4f} V={data['volume']:.2f}")


def create_test_config(data_source: str, symbol: str, interval: str) -> Dict[str, Any]:
    """Create a test configuration for the specified data source."""
    
    base_config = {
        "data_source": data_source,
        "symbol": symbol,
        "interval": interval,
        "lookback_bars": 100,
        "retry_interval": 60,
        "on_new_bar": on_new_bar
    }
    
    if data_source == "binance":
        base_config.update({
            "api_key": None,  # Use public data
            "api_secret": None,
            "testnet": False
        })
    elif data_source == "yahoo":
        base_config.update({
            "polling_interval": 60
        })
    elif data_source == "ibkr":
        base_config.update({
            "host": "127.0.0.1",
            "port": 7497,
            "client_id": 1
        })
    
    return base_config


def test_data_feed(data_source: str, symbol: str, interval: str, duration: int = 60):
    """
    Test a live data feed for the specified duration.
    
    Args:
        data_source: Data source (binance, yahoo, ibkr)
        symbol: Trading symbol
        interval: Data interval
        duration: Test duration in seconds
    """
    print(f"\n{'='*60}")
    print(f"Testing {data_source.upper()} Live Data Feed")
    print(f"Symbol: {symbol}")
    print(f"Interval: {interval}")
    print(f"Duration: {duration} seconds")
    print(f"{'='*60}")
    
    try:
        # Create configuration
        config = create_test_config(data_source, symbol, interval)
        
        # Create data feed
        print(f"\nCreating {data_source} data feed...")
        data_feed = DataFeedFactory.create_data_feed(config)
        
        if data_feed is None:
            print(f"Failed to create {data_source} data feed")
            return
        
        # Get initial status
        status = data_feed.get_status()
        print(f"\nInitial Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Create simple Backtrader strategy to consume data
        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.bar_count = 0
            
            def next(self):
                self.bar_count += 1
                if self.bar_count % 10 == 0:  # Print every 10 bars
                    print(f"Processed {self.bar_count} bars")
        
        # Create Backtrader engine
        cerebro = bt.Cerebro()
        cerebro.adddata(data_feed)
        cerebro.addstrategy(TestStrategy)
        
        # Run for specified duration
        print(f"\nStarting data feed test for {duration} seconds...")
        print("Press Ctrl+C to stop early")
        
        start_time = time.time()
        
        # Run Backtrader in a separate thread
        import threading
        
        def run_cerebro():
            cerebro.run()
        
        cerebro_thread = threading.Thread(target=run_cerebro, daemon=True)
        cerebro_thread.start()
        
        # Monitor for specified duration
        while time.time() - start_time < duration:
            time.sleep(5)
            
            # Print status every 30 seconds
            if int(time.time() - start_time) % 30 == 0:
                status = data_feed.get_status()
                print(f"\nStatus at {int(time.time() - start_time)}s:")
                print(f"  Connected: {status['is_connected']}")
                print(f"  Data points: {status['data_points']}")
                print(f"  Last update: {status['last_update']}")
        
        # Stop data feed
        print(f"\nStopping data feed...")
        data_feed.stop()
        
        # Final status
        final_status = data_feed.get_status()
        print(f"\nFinal Status:")
        for key, value in final_status.items():
            print(f"  {key}: {value}")
        
        print(f"\nTest completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\nTest interrupted by user")
        if 'data_feed' in locals():
            data_feed.stop()
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        _logger.error(f"Test error: {str(e)}")


def show_available_sources():
    """Show information about available data sources."""
    print("\nAvailable Data Sources:")
    print("=" * 50)
    
    source_info = DataFeedFactory.get_source_info()
    
    for source, info in source_info.items():
        print(f"\n{info['name']} ({source}):")
        print(f"  Description: {info['description']}")
        print(f"  Symbols: {info['symbols']}")
        print(f"  Intervals: {', '.join(info['intervals'])}")
        print(f"  Real-time: {info['real_time']}")
        print(f"  Requires Auth: {info['requires_auth']}")
        print(f"  Cost: {info['cost']}")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python test_live_data_feeds.py [data_source] [symbol] [interval]")
        print("\nExamples:")
        print("  python test_live_data_feeds.py binance BTCUSDT 1m")
        print("  python test_live_data_feeds.py yahoo AAPL 5m")
        print("  python test_live_data_feeds.py ibkr SPY 1m")
        print("\nOr run without arguments to see available sources:")
        print("  python test_live_data_feeds.py")
        
        show_available_sources()
        return
    
    data_source = sys.argv[1].lower()
    
    if data_source == "help":
        show_available_sources()
        return
    
    if data_source not in DataFeedFactory.get_supported_sources():
        print(f"Unknown data source: {data_source}")
        show_available_sources()
        return
    
    # Default values
    symbol = sys.argv[2] if len(sys.argv) > 2 else "BTCUSDT"
    interval = sys.argv[3] if len(sys.argv) > 3 else "1m"
    
    # Adjust default symbol based on data source
    if data_source == "yahoo" and symbol == "BTCUSDT":
        symbol = "AAPL"
    elif data_source == "ibkr" and symbol == "BTCUSDT":
        symbol = "SPY"
    
    # Test the data feed
    test_data_feed(data_source, symbol, interval, duration=120)  # 2 minutes


if __name__ == "__main__":
    main() 
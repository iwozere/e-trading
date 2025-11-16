#!/usr/bin/env python3
"""
CoinGecko Live Feed Example
---------------------------

This example demonstrates how to use the CoinGecko live feed to get real-time
cryptocurrency data. Since CoinGecko doesn't provide WebSocket API, this
implementation uses polling to simulate real-time updates.

Features demonstrated:
- Historical data loading
- Real-time polling updates
- Rate limiting (50 calls/minute)
- Error handling
- Callback functions
"""

import time
import signal
from datetime import datetime
from src.data.feed.coingecko_live_feed import CoinGeckoLiveDataFeed
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Global flag to control the main loop
running = True

def signal_handler(signum, frame):
    """Handle Ctrl+C to gracefully stop the feed."""
    global running
    print("\nStopping CoinGecko live feed...")
    running = False

def on_new_bar(symbol, timestamp, data):
    """
    Callback function called when new data arrives.

    Args:
        symbol: The cryptocurrency symbol
        timestamp: The timestamp of the new data
        data: Dictionary containing OHLCV data
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"New {symbol} data at {timestamp}:")
    print(f"  Open: ${data['open']:.2f}")
    print(f"  High: ${data['high']:.2f}")
    print(f"  Low: ${data['low']:.2f}")
    print(f"  Close: ${data['close']:.2f}")
    print(f"  Volume: {data['volume']:.2f}")
    print("-" * 50)

def main():
    """Main function to demonstrate CoinGecko live feed."""
    print("CoinGecko Live Feed Example")
    print("=" * 50)
    print("This example will fetch real-time data for Bitcoin.")
    print("Press Ctrl+C to stop.")
    print()

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Create CoinGecko live feed
        # Note: Using 60-second polling interval to respect rate limits
        feed = CoinGeckoLiveDataFeed(
            symbol="bitcoin",
            interval="1h",
            polling_interval=60,  # Poll every 60 seconds
            lookback_bars=100,    # Load last 100 bars
            on_new_bar=on_new_bar
        )

        print("Created CoinGecko live feed for bitcoin (1h interval)")
        print("Polling interval: 60 seconds")
        print("Rate limit: 50 calls/minute")
        print()

        # Start the feed
        print("Starting feed...")
        feed.start()

        # Main loop
        while running:
            # Get current status
            status = feed.get_status()

            if status['is_connected']:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Connected | Last update: {status['last_update']} | "
                      f"API calls: {status['api_calls_last_minute']}/{status['max_api_calls_per_minute']}")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Disconnected | Attempting to reconnect...")

            time.sleep(10)  # Print status every 10 seconds

    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    except Exception:
        _logger.exception("Error in main loop")
    finally:
        # Clean up
        if 'feed' in locals():
            print("Stopping feed...")
            feed.stop()
            print("Feed stopped.")

if __name__ == "__main__":
    main()
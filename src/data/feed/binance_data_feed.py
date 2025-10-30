import json
import threading
import queue
import requests
import time
import pandas as pd
from datetime import datetime, timezone
import asyncio
import websockets
import backtrader as bt
from src.notification.logger import setup_logger
from src.data.utils.retry import retry_on_exception
from src.data.utils.rate_limiting import get_provider_limiter
from src.data.feed.base_live_data_feed import BaseLiveDataFeed

"""
Data feed implementation for Binance, providing real-time and historical market data for trading strategies.

Binance Data Feed Module
-----------------------

This module provides a Backtrader-compatible data feed for Binance, supporting both historical and real-time data via REST API and WebSocket.

Classes:
- BinanceEnhancedFeed: Backtrader data feed for Binance with live updates
- SMACrossover: Example strategy using a simple moving average crossover
"""

_logger = setup_logger(__name__)

class BinanceEnhancedFeed(BaseLiveDataFeed):
    """
    Backtrader data feed for Binance supporting historical and real-time data via REST API and WebSocket.
    """
    lines = ('open', 'high', 'low', 'close', 'volume', 'openinterest')
    params = (
        ('symbol', 'BTCUSDT'),
        ('interval', '1m'),
        ('lookback', 1000),
    )

    # Interval to milliseconds mapping
    _INTERVAL_MS = {
        '1m': 60_000,
        '3m': 180_000,
        '5m': 300_000,
        '15m': 900_000,
        '30m': 1_800_000,
        '1h': 3_600_000,
        '2h': 7_200_000,
        '4h': 14_400_000,
        '6h': 21_600_000,
        '8h': 28_800_000,
        '12h': 43_200_000,
        '1d': 86_400_000,
        '3d': 259_200_000,
        '1w': 604_800_000,
        '1M': 2_592_000_000
    }

    def __init__(self):
        super().__init__()
        self.data_queue = queue.Queue(maxsize=10_000)  # avoid unbounded growth
        self._backfill_historical_data()
        self._start_websocket()
        self._state = self._ST_LIVE

    def _load(self):
        """Load data from queue into Backtrader lines."""
        if self._state == self._ST_LIVE and not self.data_queue.empty():
            try:
                nd = self.data_queue.get_nowait()
                self.lines.datetime[0] = bt.date2num(nd['datetime'])
                self.lines.open[0] = nd['open']
                self.lines.high[0] = nd['high']
                self.lines.low[0] = nd['low']
                self.lines.close[0] = nd['close']
                self.lines.volume[0] = nd['volume']
                self.lines.openinterest[0] = nd.get('openinterest', 0.0)
                return True
            except Exception as e:
                _logger.exception("Error loading data: %s", e)
        return False

    @retry_on_exception(max_attempts=3, base_delay=1.0, max_delay=10.0)
    def _backfill_historical_data(self):
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        step = self._INTERVAL_MS.get(self.p.interval, 60_000)
        start_time = end_time - (self.p.lookback * step)

        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': self.p.symbol.upper(),
            'interval': self.p.interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': min(self.p.lookback, 1000)
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code != 200:
                _logger.error("HTTP %s from Binance: %s", response.status_code, response.text)
                return

            klines = response.json()
            if not klines:
                return

            for k in klines:
                data_point = {
                    'datetime': timezone.utcfromtimestamp(k[0]/1000),
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                    'openinterest': 0.0
                }

                # Validate data point before adding to queue
                from src.data.utils.validation import validate_ohlcv_data
                df_point = pd.DataFrame([data_point])
                is_valid, errors = validate_ohlcv_data(df_point)
                if is_valid:
                    self.data_queue.put(data_point)
                else:
                    _logger.warning("Invalid data point: %s", errors)
        except Exception as e:
            _logger.exception("Error fetching historical data: %s", e)

    def _start_websocket(self):
        self.loop = asyncio.new_event_loop()
        self.ws_thread = threading.Thread(target=self._run_websocket_loop, daemon=True)
        self.ws_thread.start()

    def _run_websocket_loop(self):
        """Run the WebSocket event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run_websocket())

    async def _run_websocket(self):
        ws_url = f"wss://stream.binance.com:9443/ws/{self.p.symbol.lower()}@kline_{self.p.interval}"
        backoff = 1

        while True:
            try:
                async with websockets.connect(
                    ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:
                    self.ws = websocket
                    _logger.info("WebSocket connected for %s", self.p.symbol)

                    async for message in websocket:
                        await self._on_message(message)

            except websockets.exceptions.ConnectionClosed:
                _logger.info("WebSocket disconnected for %s", self.p.symbol)
            except Exception as e:
                _logger.exception("WebSocket error for %s: %s", self.p.symbol, e)

            _logger.info("WS disconnected; retrying in %ss", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

    async def _on_message(self, message):
        msg = json.loads(message)
        if msg['e'] == 'kline' and msg['k']['x']:
            k = msg['k']
            data_point = {
                'datetime': timezone.utcfromtimestamp(k['t']/1000),
                'open': float(k['o']),
                'high': float(k['h']),
                'low': float(k['l']),
                'close': float(k['c']),
                'volume': float(k['v']),
                'openinterest': 0.0
            }

            # Validate data point before adding to queue
            from src.data.utils.validation import validate_ohlcv_data
            df_point = pd.DataFrame([data_point])
            is_valid, errors = validate_ohlcv_data(df_point)
            if is_valid:
                self.data_queue.put(data_point)
            else:
                _logger.warning("Invalid WebSocket data point: %s", errors)


    def _load(self):
        if self._state == self._ST_LIVE and not self.data_queue.empty():
            new_data = self.data_queue.get()
            self.lines.datetime[0] = bt.date2num(new_data['datetime'])
            self.lines.open[0] = new_data['open']
            self.lines.high[0] = new_data['high']
            self.lines.low[0] = new_data['low']
            self.lines.close[0] = new_data['close']
            self.lines.volume[0] = new_data['volume']
            return True
        return False

class SMACrossover(bt.Strategy):
    """
    Example Backtrader strategy implementing a simple moving average crossover.
    """
    params = (
        ('sma_period', 50),
    )

    def __init__(self):
        self.sma = bt.indicators.SMA(self.data.close, period=self.params.sma_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.data.close[0] > self.sma[0]:
                self.order = self.buy()
        else:
            if self.data.close[0] < self.sma[0]:
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY EXECUTED at {order.executed.price}")
            elif order.issell():
                print(f"SELL EXECUTED at {order.executed.price}")
            self.order = None

if __name__ == '__main__':
    cerebro = bt.Cerebro()

    data = BinanceEnhancedFeed(
        symbol='BTCUSDT',
        interval='1m',
        lookback=200
    )

    cerebro.adddata(data)
    cerebro.addstrategy(SMACrossover)
    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.001)

    _logger.info("Starting Portfolio Value: %.2f", cerebro.broker.getvalue())
    cerebro.run()
    _logger.info("Final Portfolio Value:%.2f", cerebro.broker.getvalue())
    cerebro.plot()

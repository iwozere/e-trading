import json
import threading
import queue
import requests
from datetime import datetime
import websocket
import backtrader as bt
from src.notification.logger import setup_logger

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

class BinanceEnhancedFeed(bt.feed.DataBase):
    """
    Backtrader data feed for Binance supporting historical and real-time data via REST API and WebSocket.
    """
    params = (
        ('symbol', 'BTCUSDT'),
        ('interval', '1m'),
        ('lookback', 1000),
    )

    def __init__(self):
        super().__init__()
        self.data_queue = queue.Queue()
        self._backfill_historical_data()
        self._start_websocket()
        self._state = self._ST_LIVE

    def _backfill_historical_data(self):
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = end_time - (self.p.lookback * 60 * 1000)

        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': self.p.symbol.upper(),
            'interval': self.p.interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': self.p.lookback
        }

        response = requests.get(url, params=params)
        klines = response.json()

        for k in klines:
            self.data_queue.put({
                'datetime': datetime.utcfromtimestamp(k[0]/1000),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5])
            })

    def _start_websocket(self):
        self.ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
        self.ws_thread.start()

    def _run_websocket(self):
        ws_url = f"wss://stream.binance.com:9443/ws/{self.p.symbol.lower()}@kline_{self.p.interval}"
        self.ws = websocket.WebSocketApp(ws_url,
                                         on_open=self._on_open,
                                         on_message=self._on_message,
                                         on_error=self._on_error)
        self.ws.run_forever()

    def _on_open(self, ws):
        _logger.info(f"WebSocket connected for {self.p.symbol}")

    def _on_message(self, ws, message):
        msg = json.loads(message)
        if msg['e'] == 'kline' and msg['k']['x']:
            k = msg['k']
            self.data_queue.put({
                'datetime': datetime.utcfromtimestamp(k['t']/1000),
                'open': float(k['o']),
                'high': float(k['h']),
                'low': float(k['l']),
                'close': float(k['c']),
                'volume': float(k['v'])
            })

    def _on_error(self, ws, error):
        _logger.error(f"WebSocket error: {error}")
        # Auto-reconnect logic
        _logger.info("Reconnecting in 5 seconds...")
        threading.Timer(5.0, self._run_websocket).start()

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

    _logger.info(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    _logger.info(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.plot()

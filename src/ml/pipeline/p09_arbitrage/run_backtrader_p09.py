import backtrader as bt
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Optional, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

class PairsTradingStrategy(bt.Strategy):
    """
    Pairs Trading Strategy using a pre-calculated signal.
    Data0: Asset A
    Data1: Asset B
    Signal is assumed to be part of Data0 or passed as a separate feed.
    In this implementation, we assume Data0 has a 'signal' line.
    """
    params = (
        ('beta', 1.0),
    )

    def __init__(self):
        self.data_a = self.datas[0]
        self.data_b = self.datas[1]
        self.signal = self.data_a.signal
        self.beta = self.p.beta
        
        self.order_a = None
        self.order_b = None

    def notify_order(self, order):
        if order.status in [order.Completed]:
            # _logger.debug(f"ORDER COMPLETED: {order.data._name}, {order.isbuy()}, Size: {order.executed.size}")
            pass
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            _logger.warning(f"ORDER FAILED: {order.data._name}, Status: {order.getstatusname()}")

    def next(self):
        target_signal = self.signal[0]
        pos_a = self.getposition(self.data_a).size
        
        # Use Fixed Sizing for verification robustness
        size_a = 1.0
        size_b = self.beta
        
        if target_signal == 1: # Long Spread (Buy A, Sell B)
            if pos_a <= 0:
                if pos_a < 0:
                    self.close(self.data_a)
                    self.close(self.data_b)
                self.buy(self.data_a, size=size_a)
                self.sell(self.data_b, size=size_b)
                
        elif target_signal == -1: # Short Spread (Sell A, Buy B)
            if pos_a >= 0:
                if pos_a > 0:
                    self.close(self.data_a)
                    self.close(self.data_b)
                self.sell(self.data_a, size=size_a)
                self.buy(self.data_b, size=size_b)
                
        elif target_signal == 0: # Neutral
            if pos_a != 0:
                self.close(self.data_a)
                self.close(self.data_b)

class SignalData(bt.feeds.PandasData):
    """
    Custom BT data feed to include 'signal' line.
    """
    lines = ('signal',)
    params = (
        ('signal', -1), # default index for signal column
    )

def run_p09_backtest(signal_file: Path, symbol_a: str, symbol_b: str, beta: float):
    """
    Runs backtest for a specific pair using its arbitrage_signals.csv.
    """
    df = pd.read_csv(signal_file, index_col=0, parse_dates=True)
    df = df.dropna(subset=['price_a', 'price_b', 'signal']) # Clear any NaNs in critical columns
    if df.empty:
        return None

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(10000000.0) # 10M
    cerebro.broker.set_checksubmit(False)
    cerebro.broker.setcommission(commission=0.001) # 0.1%

    data_a = SignalData(
        dataname=df,
        datetime=None,
        high=-1, low=-1, open=-1, close=df.columns.get_loc('price_a'),
        volume=-1, openinterest=-1,
        signal=df.columns.get_loc('signal')
    )
    cerebro.adddata(data_a, name=symbol_a)

    data_b = bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        high=-1, low=-1, open=-1, close=df.columns.get_loc('price_b'),
        volume=-1, openinterest=-1
    )
    cerebro.adddata(data_b, name=symbol_b)

    cerebro.addstrategy(PairsTradingStrategy, beta=beta)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    _logger.info(f"Running Backtrader for {symbol_a}-{symbol_b}...")
    results = cerebro.run()
    strat = results[0]
    
    final_val = cerebro.broker.getvalue()
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
    sharpe = sharpe if sharpe is not None else 0.0
    
    metrics = {
        "final_value": final_val,
        "total_return_pct": ((final_val / 10000000.0) - 1) * 100.0,
        "sharpe": sharpe,
        "max_drawdown_pct": strat.analyzers.drawdown.get_analysis().max.drawdown
    }
    
    return metrics

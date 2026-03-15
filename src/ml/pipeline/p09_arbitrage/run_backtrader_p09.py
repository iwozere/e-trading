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
        
        # Trade tracking
        self.trades = []
        self.current_trade = None
        
        # Manual equity tracking for perfect alignment
        self.equity_values = []
        self.equity_times = []

    def next(self):
        target_signal = self.signal[0]
        pos_a = self.getposition(self.data_a).size
        dt = self.data_a.datetime.datetime(0)
        
        # Use Fixed Sizing for verification robustness
        size_a = 1.0
        size_b = self.beta
        
        price_a = float(self.data_a.close[0])
        price_b = float(self.data_b.close[0])
        
        # Manual Valuation check
        cash = self.broker.get_cash()
        val_a = self.getposition(self.data_a).size * price_a
        val_b = self.getposition(self.data_b).size * price_b
        manual_value = cash + val_a + val_b
        
        broker_value = float(self.broker.getvalue())
        
        # Use manual value if broker fails, or just use it as primary for now to ensure stability
        current_value = manual_value if not np.isnan(manual_value) else initial_cash # fallback
        
        if np.isnan(broker_value) and not np.isnan(manual_value):
             # Only log this occasionally to avoid spam
             if len(self.equity_values) % 1000 == 0:
                _logger.warning(f"Broker value is NaN but Manual value is valid @ {dt}. Using Manual.")
        
        if np.isnan(current_value):
            _logger.error(f"CRITICAL VALUATION FAILURE @ {dt}: cash={cash}, val_a={val_a}, val_b={val_b}")
            current_value = self.equity_values[-1] if self.equity_values else initial_cash

        # Track equity for every bar
        self.equity_values.append(current_value)
        self.equity_times.append(dt)
        
        # Determine if we need to close
        if self.current_trade:
            in_long = self.current_trade['side'] == 'Long Spread'
            in_short = self.current_trade['side'] == 'Short Spread'
            
            should_exit = (in_long and target_signal <= 0) or (in_short and target_signal >= 0)
                
            if should_exit:
                self.close(self.data_a)
                self.close(self.data_b)
                
                # Capture terminal data for the trade
                pnl = current_value - self.current_trade['start_value']
                self.current_trade.update({
                    'exit_time': dt,
                    'exit_price_a': price_a,
                    'exit_price_b': price_b,
                    'pnl': pnl
                })
                self.trades.append(self.current_trade.copy())
                self.current_trade = None

        # Determine if we need to open
        if pos_a == 0 and self.current_trade is None:
            if target_signal == 1: # Open Long
                self.buy(self.data_a, size=size_a)
                self.sell(self.data_b, size=size_b)
                self.current_trade = {
                    'entry_time': dt,
                    'side': 'Long Spread',
                    'entry_price_a': price_a,
                    'entry_price_b': price_b,
                    'start_value': current_value
                }
            elif target_signal == -1: # Open Short
                self.sell(self.data_a, size=size_a)
                self.buy(self.data_b, size=size_b)
                self.current_trade = {
                    'entry_time': dt,
                    'side': 'Short Spread',
                    'entry_price_a': price_a,
                    'entry_price_b': price_b,
                    'start_value': current_value
                }

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
    initial_cash = 10000000.0
    cerebro.broker.setcash(initial_cash) # 10M
    cerebro.broker.set_checksubmit(False)
    cerebro.broker.setcommission(commission=0.001) # 0.1%

    price_a_idx = df.columns.get_loc('price_a')
    price_b_idx = df.columns.get_loc('price_b')
    signal_idx = df.columns.get_loc('signal')

    data_a = SignalData(
        dataname=df,
        datetime=None,
        high=price_a_idx, low=price_a_idx, open=price_a_idx, close=price_a_idx,
        volume=-1, openinterest=-1,
        signal=signal_idx
    )
    cerebro.adddata(data_a, name=symbol_a)

    data_b = bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        high=price_b_idx, low=price_b_idx, open=price_b_idx, close=price_b_idx,
        volume=-1, openinterest=-1
    )
    cerebro.adddata(data_b, name=symbol_b)

    cerebro.addstrategy(PairsTradingStrategy, beta=beta)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    # Removed TimeReturn in favor of manual strategy tracking

    _logger.info(f"Running Backtrader for {symbol_a}-{symbol_b}...")
    results = cerebro.run()
    strat = results[0]
    
    final_val = cerebro.broker.getvalue()
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
    sharpe = sharpe if sharpe is not None else 0.0
    
    # Export trades list
    if strat.trades:
        _logger.info(f"Backtest complete. Captured {len(strat.trades)} trades.")
        _logger.debug(f"First trade sample: {strat.trades[0]}")
        trades_df = pd.DataFrame(strat.trades)
        trades_path = signal_file.parent / "trades.csv"
        trades_df.to_csv(trades_path, index=False)
        _logger.info(f"Trades exported to: {trades_path}")
    else:
        _logger.warning(f"No trades captured for {symbol_a}-{symbol_b}")

    # Extract equity curve from manual tracking
    equity_curve = pd.Series(strat.equity_values, index=strat.equity_times)
    _logger.debug(f"Equity curve sample (first 5):\n{equity_curve.head()}")
    _logger.debug(f"Equity curve has {equity_curve.isna().sum()} NaNs out of {len(equity_curve)} points.")
    
    metrics = {
        "final_value": final_val,
        "total_return_pct": ((final_val / initial_cash) - 1) * 100.0,
        "sharpe": sharpe,
        "max_drawdown_pct": strat.analyzers.drawdown.get_analysis().max.drawdown,
        "trades_count": len(strat.trades),
        "equity_curve": equity_curve
    }
    
    return metrics

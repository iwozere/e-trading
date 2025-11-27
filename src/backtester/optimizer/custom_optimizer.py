"""
Custom Optimizer Module

This module implements a custom optimization framework for trading strategies using Backtrader
and Optuna. It provides functionality to:
1. Run backtests with fixed parameters
2. Optimize strategy parameters using Optuna
3. Collect and analyze various performance metrics
4. Support multiple entry and exit strategies

The optimizer supports:
- Multiple entry and exit strategy combinations
- Parameter optimization with different types (int, float, categorical)
- Comprehensive performance analysis with multiple metrics
- Custom analyzers for detailed strategy evaluation

Parameters:
    config (dict): Configuration dictionary containing:
        - data: pandas DataFrame with OHLCV data
        - entry_logic: Entry logic configuration
        - exit_logic: Exit logic configuration
        - optimizer_settings: Optimization settings
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import backtrader as bt
from src.backtester.analyzer.bt_analyzers import (CAGR, CalmarRatio,
                                       ConsecutiveWinsLosses,
                                       PortfolioVolatility, ProfitFactor,
                                       SortinoRatio, WinRate)
from src.strategy.custom_strategy import CustomStrategy
from src.notification.logger import setup_logger

# Use multiprocessing-safe logging for optimizer (runs in worker processes)
_logger = setup_logger(__name__, use_multiprocessing=True)

class CustomOptimizer:
    def __init__(self, config: dict):
        """
        Initialize optimizer with configuration

        Args:
            config (dict): Configuration dictionary containing:
                - data: pandas DataFrame with OHLCV data
                - entry_logic: Entry logic configuration
                - exit_logic: Exit logic configuration
                - optimizer_settings: Optimization settings
        """
        self.config = config
        self.data = config.get("data")
        self.entry_logic = config.get("entry_logic")
        self.exit_logic = config.get("exit_logic")
        self.optimizer_settings = config.get("optimizer_settings", {})
        self.visualization_settings = config.get("visualization_settings", {})
        self.symbol = config.get("symbol", "")
        self.timeframe = config.get("timeframe", "")

        # Initialize settings
        self.initial_capital = self.optimizer_settings.get("initial_capital", 1000.0)
        self.commission = self.optimizer_settings.get("commission", 0.001)
        self.risk_free_rate = self.optimizer_settings.get("risk_free_rate", 0.01)
        self.use_talib = self.optimizer_settings.get("use_talib", False)
        self.output_dir = self.optimizer_settings.get("output_dir", "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def to_dict(self, obj):
        if isinstance(obj, dict):
            return {k: self.to_dict(v) for k, v in obj.items()}
        elif hasattr(obj, "items"):
            return {k: self.to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.to_dict(v) for v in obj]
        else:
            return obj

    def run_optimization(self, trial=None, include_analyzers=True):
        """
        Run optimization for a single trial or backtest with fixed parameters

        Args:
            trial: Optuna trial object (optional)
            include_analyzers: Whether to include analyzers (default: True)

        Returns:
            dict: Dictionary containing metrics and trades
        """
        # Create cerebro instance
        cerebro = None
        if include_analyzers:
            cerebro = bt.Cerebro()
        else:
            cerebro = bt.Cerebro(optdatas=True, optreturn=True)

                # Add data
        cerebro.adddata(self.data)

        # Add entry logic parameters
        entry_logic_params = {}
        for param_name, param_config in self.entry_logic["params"].items():
            if trial:
                if param_config["type"] == "int":
                    entry_logic_params[param_name] = trial.suggest_int(
                        param_name, param_config["low"], param_config["high"]
                    )
                elif param_config["type"] == "float":
                    entry_logic_params[param_name] = trial.suggest_float(
                        param_name, param_config["low"], param_config["high"]
                    )
                elif param_config["type"] == "categorical":
                    entry_logic_params[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )
            else:
                entry_logic_params[param_name] = param_config["default"]

        # Add exit logic parameters
        exit_logic_params = {}
        for param_name, param_config in self.exit_logic["params"].items():
            if trial:
                if param_config["type"] == "int":
                    exit_logic_params[param_name] = trial.suggest_int(
                        param_name, param_config["low"], param_config["high"]
                    )
                elif param_config["type"] == "float":
                    exit_logic_params[param_name] = trial.suggest_float(
                        param_name, param_config["low"], param_config["high"]
                    )
                elif param_config["type"] == "categorical":
                    exit_logic_params[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )
            else:
                exit_logic_params[param_name] = param_config["default"]

        # Prepare strategy parameters
        strategy_params = {
            "entry_logic": {
                "name": self.entry_logic["name"],
                "params": entry_logic_params,
            },
            "exit_logic": {
                "name": self.exit_logic["name"],
                "params": exit_logic_params,
            },
            "use_talib": self.use_talib,
            "position_size": self.optimizer_settings.get("position_size", 0.10),
            "enable_database_logging": self.optimizer_settings.get("enable_database_logging", False),
            "bot_type": "optimization",  # Mark as optimization run
        }

        # Add strategy with parameters
        cerebro.addstrategy(
            CustomStrategy,
            strategy_config=strategy_params,
            symbol=self.symbol,
            timeframe=self.timeframe
        )

        # Set broker parameters
        cerebro.broker.setcash(self.initial_capital)
        cerebro.broker.setcommission(commission=self.commission)

        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

        # Only add analyzers if requested (for final run or detailed analysis)
        if include_analyzers:
            # Add basic analyzers for optimization trials
            cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
            cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
            cerebro.addanalyzer(bt.analyzers.TimeDrawDown, _name="time_drawdown")
            cerebro.addanalyzer(bt.analyzers.VWR, _name="vwr")
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=self.risk_free_rate)

            # Add custom analyzers
            cerebro.addanalyzer(ProfitFactor, _name="profit_factor")
            cerebro.addanalyzer(WinRate, _name="winrate")
            cerebro.addanalyzer(CalmarRatio, _name="calmar", riskfreerate=self.risk_free_rate)
            cerebro.addanalyzer(CAGR, _name="cagr", timeframe=bt.TimeFrame.Years)
            cerebro.addanalyzer(SortinoRatio, _name="sortino", riskfreerate=self.risk_free_rate)
            cerebro.addanalyzer(ConsecutiveWinsLosses, _name="consecutivewinslosses")
            cerebro.addanalyzer(PortfolioVolatility, _name="portfoliovolatility")

        # Run backtest
        _logger.debug("Running backtest")
        results = cerebro.run(runonce=True, preload=True)
        strategy = results[0]
        _logger.debug("Backtest completed")

        # Process analyzers only if they were added
        analyzers = {}
        if include_analyzers:
            for name in strategy.analyzers._names:
                analyzer = getattr(strategy.analyzers, name)
                analysis = analyzer.get_analysis()
                analyzers[name] = self.to_dict(analysis)

        # Get trade analysis (always available from strategy.trades)
        trades_analysis = analyzers.get("trades", {}) if include_analyzers else {}

        # Calculate metrics
        # For optimization trials, we can calculate basic metrics from strategy.trades
        gross_profit = trades_analysis.get("pnl", {}).get("gross", {}).get("total", 0.0)
        net_profit = trades_analysis.get("pnl", {}).get("net", {}).get("total", 0.0)
        total_commission = gross_profit - net_profit

        # If gross profit is not available, calculate it from net profit + commission
        if gross_profit == 0.0 and net_profit != 0.0:
            gross_profit = net_profit + total_commission

        output = {
            "best_params": strategy_params,
            "total_profit": float(gross_profit),  # Gross profit (before commission)
            "total_profit_with_commission": float(net_profit),  # Net profit (after commission)
            "total_commission": float(total_commission),  # Total commission paid
            "analyzers": analyzers,
            "trades": strategy.trades,
        }

        return strategy, cerebro, output

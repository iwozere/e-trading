"""
Test Script for HMM-LSTM Pipeline Strategy

This script demonstrates how to use the HMM-LSTM strategy with the backtesting framework.
It loads the strategy configuration and runs a backtest using historical data.

Usage:
    python test_hmm_lstm_strategy.py [--config CONFIG_PATH] [--data DATA_PATH]
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import backtrader as bt

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.strategy.hmm_lstm_strategy import HMMLSTMPipelineStrategy
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

def load_strategy_config(config_path: str) -> dict:
    """Load strategy configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def prepare_data(data_path: str) -> pd.DataFrame:
    """Load and prepare data for backtesting."""
    # Load data
    df = pd.read_csv(data_path)

    # Ensure required columns exist
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Sort by timestamp
    df.sort_index(inplace=True)

    # Add log_return if not present
    if 'log_return' not in df.columns:
        df['log_return'] = (df['close'] / df['close'].shift(1)).apply(lambda x: 0 if x <= 0 else pd.np.log(x))
        df['log_return'].fillna(0, inplace=True)

    _logger.info("Loaded data: %s rows, %s to %s")
    return df

def create_backtrader_feed(df: pd.DataFrame, symbol: str = "BTCUSDT") -> bt.feeds.PandasData:
    """Create Backtrader data feed from DataFrame."""

    # Create custom data feed that includes additional columns
    class ExtendedPandasData(bt.feeds.PandasData):
        lines = ('log_return',)  # Add log_return as a line
        params = (
            ('log_return', -1),  # -1 means auto-detect column index
        )

    # Create the data feed
    data_feed = ExtendedPandasData(
        dataname=df,
        fromdate=df.index[0],
        todate=df.index[-1],
        timeframe=bt.TimeFrame.Minutes,
        compression=60,  # 1 hour
        name=symbol
    )

    return data_feed

def run_backtest(strategy_config: dict, data_path: str, variant: str = "default") -> dict:
    """Run backtest with HMM-LSTM strategy."""

    # Load configuration
    if variant == "default":
        config = strategy_config.get("default_config", {})
    else:
        base_config = strategy_config.get("default_config", {})
        variant_config = strategy_config.get("variants", {}).get(variant, {})
        config = {**base_config, **variant_config}

    symbol = config.get("symbol", "BTCUSDT")

    # Prepare data
    df = prepare_data(data_path)
    data_feed = create_backtrader_feed(df, symbol)

    # Initialize Cerebro
    cerebro = bt.Cerebro()

    # Add data
    cerebro.adddata(data_feed)

    # Add strategy
    cerebro.addstrategy(
        HMMLSTMPipelineStrategy,
        strategy_config=config,
        symbol=symbol,
        timeframe=config.get("timeframe", "1h")
    )

    # Set broker parameters
    initial_cash = strategy_config.get("backtesting_config", {}).get("initial_cash", 100000)
    commission = strategy_config.get("backtesting_config", {}).get("commission", 0.001)

    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    # Print starting conditions
    _logger.info("Starting Portfolio Value: %s")
    _logger.info("Strategy variant: %s")
    _logger.info("Configuration: %s")

    # Run backtest
    try:
        results = cerebro.run()

        # Get final portfolio value
        final_value = cerebro.broker.getvalue()

        # Extract results
        strat = results[0]

        # Get analyzer results
        sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        max_drawdown = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
        total_return = strat.analyzers.returns.get_analysis().get('rtot', 0)
        trades_analysis = strat.analyzers.trades.get_analysis()

        # Compile results
        results_summary = {
            'initial_value': initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': (final_value - initial_cash) / initial_cash * 100,
            'sharpe_ratio': sharpe_ratio if sharpe_ratio else 0,
            'max_drawdown': max_drawdown,
            'total_trades': trades_analysis.get('total', {}).get('total', 0),
            'winning_trades': trades_analysis.get('won', {}).get('total', 0),
            'losing_trades': trades_analysis.get('lost', {}).get('total', 0),
            'win_rate': 0,
            'avg_win': trades_analysis.get('won', {}).get('pnl', {}).get('average', 0),
            'avg_loss': trades_analysis.get('lost', {}).get('pnl', {}).get('average', 0),
        }

        # Calculate win rate
        total_trades = results_summary['total_trades']
        if total_trades > 0:
            results_summary['win_rate'] = results_summary['winning_trades'] / total_trades * 100

        # Log results
        _logger.info("\n%s")
        _logger.info("BACKTEST RESULTS (%s)")
        _logger.info("%s")
        _logger.info("Initial Value: $%s")
        _logger.info("Final Value: $%s")
        _logger.info("Total Return: %s%")
        _logger.info("Sharpe Ratio: %s")
        _logger.info("Max Drawdown: %s%")
        _logger.info("Total Trades: %s")
        _logger.info("Win Rate: %s%")
        _logger.info("Avg Win: $%s")
        _logger.info("Avg Loss: $%s")
        _logger.info("%s")

        return results_summary

    except Exception:
        _logger.exception("Error running backtest")
        raise

def compare_variants(strategy_config: dict, data_path: str) -> dict:
    """Compare different strategy variants."""

    variants = ["conservative", "balanced", "aggressive"]
    results = {}

    _logger.info("Comparing strategy variants...")

    for variant in variants:
        if variant in strategy_config.get("variants", {}):
            try:
                _logger.info("\nRunning %s variant...")
                variant_results = run_backtest(strategy_config, data_path, variant)
                results[variant] = variant_results
            except Exception:
                _logger.exception("Error running %s variant", variant)
                results[variant] = None

    # Print comparison
    _logger.info("\n%s")
    _logger.info("VARIANT COMPARISON")
    _logger.info("%s")
    _logger.info("%s %s %s %s %s %s")
    _logger.info("%s")

    for variant, result in results.items():
        if result:
                        _logger.info("%-12s %-10.2f %-8.3f %-12.2f %-8d %-10.1f",
                         variant, result['total_return_pct'], result['sharpe_ratio'],
                         result['max_drawdown'], result['total_trades'], result['win_rate'])
        else:
            _logger.info("%s %s")

    _logger.info("%s")

    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test HMM-LSTM Pipeline Strategy")
    parser.add_argument(
        "--config",
        type=str,
        default="config/strategy/hmm_lstm_simple.json",
        help="Path to strategy configuration file"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to CSV data file (must contain OHLCV data)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="balanced",
        choices=["conservative", "balanced", "aggressive", "compare"],
        help="Strategy variant to test or 'compare' to test all variants"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.data:
        _logger.error("Data file path is required. Use --data to specify CSV file path.")
        return

    if not Path(args.config).exists():
        _logger.error("Configuration file not found: %s", args.config)
        return

    if not Path(args.data).exists():
        _logger.error("Data file not found: %s", args.data)
        return

    try:
        # Load strategy configuration
        strategy_config = load_strategy_config(args.config)

        if args.variant == "compare":
            # Compare all variants
            compare_variants(strategy_config, args.data)
        else:
            # Run single variant
            run_backtest(strategy_config, args.data, args.variant)

    except Exception:
        _logger.exception("Error")
        raise

if __name__ == "__main__":
    main()

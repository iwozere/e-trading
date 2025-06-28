"""
Advanced Strategy Framework Example

This example demonstrates how to use the Advanced Strategy Framework with:
- Composite strategies (combining multiple strategies)
- Multi-timeframe analysis
- Dynamic strategy switching
- Portfolio optimization
"""

import sys
import os
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime, timedelta
import logging

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.strategy.advanced_backtrader_strategy import AdvancedBacktraderStrategy
from src.strategy.composite_strategy_manager import AdvancedStrategyFramework, CompositeSignal

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample market data for demonstration."""
    # Generate 1000 bars of sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='1H')
    
    # Create realistic price data with trends and volatility
    base_price = 100.0
    prices = [base_price]
    
    for i in range(1, 1000):
        # Add trend component
        trend = 0.0001 * i  # Slight upward trend
        
        # Add volatility component
        volatility = 0.02
        random_component = np.random.normal(0, volatility)
        
        # Add some mean reversion
        mean_reversion = -0.1 * (prices[-1] - base_price) / base_price
        
        # Calculate new price
        new_price = prices[-1] * (1 + trend + random_component + mean_reversion)
        prices.append(max(new_price, 1.0))  # Ensure price doesn't go negative
    
    # Create OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Create realistic OHLC from close price
        volatility = 0.01
        high = price * (1 + abs(np.random.normal(0, volatility)))
        low = price * (1 - abs(np.random.normal(0, volatility)))
        open_price = prices[i-1] if i > 0 else price
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'datetime': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def run_composite_strategy_example():
    """Example of running a composite strategy."""
    logger.info("=== Composite Strategy Example ===")
    
    # Create sample data
    data = create_sample_data()
    
    # Initialize the advanced framework
    framework = AdvancedStrategyFramework()
    framework.initialize_composite_strategies()
    
    # Create data feeds for different timeframes
    data_feeds = {
        '1h': data,
        '4h': data.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(),
        '1d': data.resample('1D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    }
    
    # Test different composite strategies
    strategies_to_test = [
        'momentum_trend_composite',
        'mean_reversion_momentum',
        'volatility_breakout'
    ]
    
    for strategy_name in strategies_to_test:
        logger.info(f"\nTesting strategy: {strategy_name}")
        
        try:
            # Generate composite signal
            signal = framework.get_composite_signal(strategy_name, data_feeds)
            
            logger.info(f"Signal type: {signal.signal_type}")
            logger.info(f"Confidence: {signal.confidence:.3f}")
            logger.info(f"Contributing strategies: {signal.contributing_strategies}")
            logger.info(f"Metadata: {signal.metadata}")
            
        except Exception as e:
            logger.error(f"Error testing strategy {strategy_name}: {e}")


def run_multi_timeframe_example():
    """Example of running multi-timeframe strategies."""
    logger.info("\n=== Multi-Timeframe Strategy Example ===")
    
    # Create sample data
    data = create_sample_data()
    
    # Initialize the advanced framework
    framework = AdvancedStrategyFramework()
    framework.initialize_multi_timeframe_strategies()
    
    # Create data feeds for different timeframes
    data_feeds = {
        '15m': data.resample('15T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(),
        '1h': data,
        '4h': data.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    }
    
    # Test multi-timeframe strategies
    mtf_strategies = [
        'trend_following_mtf',
        'breakout_mtf',
        'mean_reversion_mtf'
    ]
    
    for strategy_name in mtf_strategies:
        logger.info(f"\nTesting MTF strategy: {strategy_name}")
        
        try:
            # Execute strategy
            signal = framework.execute_strategy(strategy_name, data_feeds)
            
            logger.info(f"Signal type: {signal.signal_type}")
            logger.info(f"Confidence: {signal.confidence:.3f}")
            logger.info(f"Contributing strategies: {signal.contributing_strategies}")
            
        except Exception as e:
            logger.error(f"Error testing MTF strategy {strategy_name}: {e}")


def run_dynamic_switching_example():
    """Example of dynamic strategy switching."""
    logger.info("\n=== Dynamic Strategy Switching Example ===")
    
    # Create sample data with different market regimes
    data = create_sample_data()
    
    # Initialize the advanced framework
    framework = AdvancedStrategyFramework()
    framework.initialize_dynamic_switching()
    
    # Create data feeds
    data_feeds = {
        '1h': data,
        '4h': data.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    }
    
    # Test dynamic switching
    logger.info("Testing dynamic strategy switching...")
    
    try:
        # Get dynamic strategy recommendation
        recommended_strategy = framework.get_dynamic_strategy(data_feeds)
        logger.info(f"Recommended strategy: {recommended_strategy}")
        
        # Execute the recommended strategy
        signal = framework.execute_strategy(recommended_strategy, data_feeds)
        
        logger.info(f"Signal type: {signal.signal_type}")
        logger.info(f"Confidence: {signal.confidence:.3f}")
        
    except Exception as e:
        logger.error(f"Error in dynamic switching: {e}")


def run_backtrader_example():
    """Example of running the advanced strategy with Backtrader."""
    logger.info("\n=== Backtrader Integration Example ===")
    
    # Create sample data
    data = create_sample_data()
    
    # Create Backtrader data feed
    data_feed = bt.feeds.PandasData(
        dataname=data,
        datetime='datetime',
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=None
    )
    
    # Create Cerebro engine
    cerebro = bt.Cerebro()
    
    # Add data feed
    cerebro.adddata(data_feed)
    
    # Add strategy
    cerebro.addstrategy(
        AdvancedBacktraderStrategy,
        strategy_name='momentum_trend_composite',
        use_dynamic_switching=True,
        max_position_size=0.1,
        stop_loss_pct=0.02,
        take_profit_pct=0.04
    )
    
    # Set initial capital
    cerebro.broker.setcash(10000.0)
    
    # Set commission
    cerebro.broker.setcommission(commission=0.001)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # Run backtest
    logger.info("Running Backtrader backtest...")
    results = cerebro.run()
    
    # Get results
    strategy = results[0]
    
    # Print results
    logger.info(f"Final Portfolio Value: ${cerebro.broker.getvalue():.2f}")
    logger.info(f"Total Return: {strategy.analyzers.returns.get_analysis()['rtot']:.2%}")
    logger.info(f"Sharpe Ratio: {strategy.analyzers.sharpe.get_analysis()['sharperatio']:.3f}")
    logger.info(f"Max Drawdown: {strategy.analyzers.drawdown.get_analysis()['max']['drawdown']:.2%}")
    
    # Get strategy summary
    summary = strategy.get_strategy_summary()
    logger.info(f"Strategy used: {summary['strategy_name']}")
    logger.info(f"Total trades: {summary['performance_metrics']['total_trades']}")
    logger.info(f"Win rate: {summary['performance_metrics']['winning_trades']/max(summary['performance_metrics']['total_trades'], 1):.2%}")
    
    # Plot results
    try:
        cerebro.plot(style='candlestick', barup='green', bardown='red')
    except Exception as e:
        logger.warning(f"Could not plot results: {e}")


def run_portfolio_optimization_example():
    """Example of portfolio optimization strategies."""
    logger.info("\n=== Portfolio Optimization Example ===")
    
    # Create sample data for multiple assets
    assets = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']
    asset_data = {}
    
    for asset in assets:
        asset_data[asset] = create_sample_data()
        # Add some correlation between assets
        if asset != 'BTC':
            asset_data[asset]['close'] = asset_data['BTC']['close'] * np.random.uniform(0.1, 0.5)
    
    # Initialize the advanced framework
    framework = AdvancedStrategyFramework()
    
    # Get portfolio optimization configurations
    portfolio_configs = framework.configs.get("portfolio_optimization", {})
    
    for strategy_name, config in portfolio_configs.get("portfolio_optimization_strategies", {}).items():
        logger.info(f"\nPortfolio strategy: {config['name']}")
        logger.info(f"Description: {config['description']}")
        
        if 'optimization_method' in config:
            logger.info(f"Optimization method: {config['optimization_method']}")
        
        if 'constraints' in config:
            logger.info(f"Constraints: {config['constraints']}")
        
        if 'rebalancing' in config:
            logger.info(f"Rebalancing: {config['rebalancing']}")


def main():
    """Run all examples."""
    logger.info("Advanced Strategy Framework Examples")
    logger.info("=" * 50)
    
    try:
        # Run composite strategy example
        run_composite_strategy_example()
        
        # Run multi-timeframe example
        run_multi_timeframe_example()
        
        # Run dynamic switching example
        run_dynamic_switching_example()
        
        # Run Backtrader integration example
        run_backtrader_example()
        
        # Run portfolio optimization example
        run_portfolio_optimization_example()
        
        logger.info("\n" + "=" * 50)
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
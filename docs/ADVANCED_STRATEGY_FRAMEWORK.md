# Advanced Strategy Framework (Refactored)

## Overview

The Advanced Strategy Framework is now modular, split into three core components:
- `strategy_core.py`: Base abstractions, signals, risk, aggregation, regime detection
- `multi_timeframe_engine.py`: Data aggregation, multi-timeframe logic
- `composite_strategy_manager.py`: Strategy orchestration, composite signal management

## Usage

### 1. Core Abstractions
```python
from src.strategy.strategy_core import BaseStrategy, StrategySignal, CompositeSignal, SignalAggregator, AggregationMethod, MarketRegimeDetector
```

### 2. Multi-Timeframe Engine
```python
from src.strategy.multi_timeframe_engine import TimeframeSyncer, MultiTimeframeStrategy
```

### 3. Composite Strategy Manager
```python
from src.strategy.composite_strategy_manager import StrategyComposer, AdvancedStrategyFramework
```

## Example: Creating and Running a Composite Strategy
```python
from src.strategy.composite_strategy_manager import AdvancedStrategyFramework

framework = AdvancedStrategyFramework(config_path="config/strategy/")
framework.initialize_composite_strategies()
framework.initialize_multi_timeframe_strategies()
framework.initialize_dynamic_switching()

# Example: Get a composite signal
signal = framework.get_composite_signal("momentum_trend_composite", data_feeds)
print(signal)
```

## Configuration

Configuration files remain the same as before. See the original documentation below for details on composite, multi-timeframe, and dynamic switching configurations.

---

# [Legacy Documentation]

## Key Features

### 1. Composite Strategies
- **Strategy Combination**: Combines multiple individual strategies into composite strategies
- **Weighted Voting**: Uses weighted voting to aggregate signals from multiple strategies
- **Consensus Voting**: Requires agreement above a threshold for signal generation
- **Signal Aggregation**: Multiple methods for combining strategy signals

### 2. Multi-Timeframe Analysis
- **Higher Timeframe Trend Analysis**: Uses higher timeframes for trend direction
- **Lower Timeframe Entry/Exit**: Uses lower timeframes for precise entry/exit timing
- **Timeframe Synchronization**: Ensures signals align across different timeframes
- **Cross-Timeframe Validation**: Validates signals across multiple timeframes

### 3. Dynamic Strategy Switching
- **Market Regime Detection**: Automatically detects market conditions
- **Performance-Based Switching**: Switches strategies based on recent performance
- **Time-Based Switching**: Switches strategies based on time of day/market sessions
- **Smooth Transitions**: Gradual strategy transitions to avoid sudden changes

### 4. Portfolio Optimization
- **Modern Portfolio Theory**: Implements MPT for optimal asset allocation
- **Risk Parity**: Equalizes risk contribution across assets
- **Dynamic Asset Allocation**: Adapts allocation based on market conditions
- **Correlation-Based Position Sizing**: Adjusts position sizes based on correlations

## Architecture

### Core Components

#### 1. AdvancedStrategyFramework
The main framework class that orchestrates all advanced strategy features.

```python
from src.strategy.advanced_strategy_framework import AdvancedStrategyFramework

# Initialize the framework
framework = AdvancedStrategyFramework()
framework.initialize_composite_strategies()
framework.initialize_multi_timeframe_strategies()
framework.initialize_dynamic_switching()
```

#### 2. AdvancedBacktraderStrategy
A Backtrader strategy class that integrates with the advanced framework.

```python
from src.strategy.advanced_backtrader_strategy import AdvancedBacktraderStrategy

# Use in Backtrader
cerebro.addstrategy(AdvancedBacktraderStrategy, strategy_name='momentum_trend_composite')
```

#### 3. Signal Aggregation
Multiple methods for combining signals from different strategies:

- **Weighted Voting**: Combines signals based on strategy weights and confidence
- **Consensus Voting**: Requires agreement above a threshold
- **Majority Voting**: Simple majority rule
- **Weighted Average**: Average of signal confidences

#### 4. Market Regime Detection
Automatically detects market conditions:

- **Trending Volatile**: Strong trend with high volatility
- **Trending Stable**: Strong trend with low volatility
- **Ranging Volatile**: Weak trend with high volatility
- **Ranging Stable**: Weak trend with low volatility

## Configuration

### Composite Strategies Configuration

Located in `config/strategy/composite_strategies.json`:

```json
{
    "composite_strategies": {
        "momentum_trend_composite": {
            "name": "Momentum Trend Composite",
            "description": "Combines momentum and trend-following strategies",
            "strategies": [
                {
                    "name": "rsi_momentum",
                    "weight": 0.4,
                    "timeframe": "1h",
                    "params": {
                        "rsi_period": 14,
                        "rsi_oversold": 30,
                        "rsi_overbought": 70,
                        "volume_threshold": 1.5
                    }
                },
                {
                    "name": "supertrend_trend",
                    "weight": 0.6,
                    "timeframe": "4h",
                    "params": {
                        "supertrend_period": 10,
                        "supertrend_multiplier": 3.0,
                        "atr_period": 14
                    }
                }
            ],
            "aggregation_method": "weighted_voting",
            "consensus_threshold": 0.6,
            "risk_management": {
                "max_position_size": 0.1,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04
            }
        }
    }
}
```

### Multi-Timeframe Strategies Configuration

Located in `config/strategy/multi_timeframe.json`:

```json
{
    "multi_timeframe_strategies": {
        "trend_following_mtf": {
            "name": "Multi-Timeframe Trend Following",
            "description": "Uses higher timeframe for trend direction and lower timeframe for entries",
            "timeframes": {
                "trend_timeframe": "4h",
                "entry_timeframe": "1h",
                "exit_timeframe": "15m"
            },
            "strategy_config": {
                "trend_analysis": {
                    "method": "supertrend",
                    "params": {
                        "period": 10,
                        "multiplier": 3.0,
                        "atr_period": 14
                    }
                },
                "entry_signals": {
                    "method": "rsi_volume",
                    "params": {
                        "rsi_period": 14,
                        "rsi_oversold": 30,
                        "rsi_overbought": 70,
                        "volume_threshold": 1.5
                    }
                }
            },
            "rules": {
                "trend_confirmation_required": true,
                "entry_only_in_trend_direction": true,
                "use_higher_tf_stops": true
            }
        }
    }
}
```

### Dynamic Switching Configuration

Located in `config/strategy/dynamic_switching.json`:

```json
{
    "dynamic_switching_strategies": {
        "market_regime_adaptive": {
            "name": "Market Regime Adaptive Strategy",
            "description": "Dynamically switches strategies based on market volatility and trend conditions",
            "regime_detection": {
                "volatility_threshold": 0.02,
                "trend_strength_threshold": 0.6,
                "lookback_period": 20
            },
            "strategies": {
                "trending_volatile": {
                    "name": "momentum_trend_composite",
                    "conditions": {
                        "volatility": "high",
                        "trend_strength": "strong",
                        "market_regime": "trending"
                    },
                    "weight": 1.0
                }
            },
            "switching_rules": {
                "minimum_regime_duration": 24,
                "smooth_transition": true,
                "position_adjustment": "gradual"
            }
        }
    }
}
```

### Portfolio Optimization Configuration

Located in `config/strategy/portfolio_optimization.json`:

```json
{
    "portfolio_optimization_strategies": {
        "modern_portfolio_theory": {
            "name": "Modern Portfolio Theory Strategy",
            "description": "Uses MPT for optimal asset allocation and risk management",
            "optimization_method": "sharpe_ratio_maximization",
            "constraints": {
                "max_position_size": 0.25,
                "min_position_size": 0.05,
                "max_portfolio_volatility": 0.15,
                "target_return": 0.12
            },
            "rebalancing": {
                "frequency": "weekly",
                "threshold": 0.05,
                "transaction_cost_consideration": true
            },
            "risk_management": {
                "var_limit": 0.02,
                "max_drawdown_limit": 0.10,
                "correlation_threshold": 0.7
            }
        }
    }
}
```

## Usage Examples

### Basic Usage

```python
from src.strategy.advanced_strategy_framework import AdvancedStrategyFramework

# Initialize framework
framework = AdvancedStrategyFramework()

# Generate composite signal
signal = framework.get_composite_signal('momentum_trend_composite', data_feeds)

print(f"Signal: {signal.signal_type}")
print(f"Confidence: {signal.confidence}")
print(f"Strategies: {signal.contributing_strategies}")
```

### Multi-Timeframe Usage

```python
# Initialize multi-timeframe strategies
framework.initialize_multi_timeframe_strategies()

# Execute multi-timeframe strategy
signal = framework.execute_strategy('trend_following_mtf', data_feeds)
```

### Dynamic Switching Usage

```python
# Initialize dynamic switching
framework.initialize_dynamic_switching()

# Get recommended strategy based on market conditions
recommended_strategy = framework.get_dynamic_strategy(data_feeds)
signal = framework.execute_strategy(recommended_strategy, data_feeds)
```

### Backtrader Integration

```python
import backtrader as bt
from src.strategy.advanced_backtrader_strategy import AdvancedBacktraderStrategy

# Create Cerebro engine
cerebro = bt.Cerebro()

# Add data
cerebro.adddata(data_feed)

# Add advanced strategy
cerebro.addstrategy(
    AdvancedBacktraderStrategy,
    strategy_name='momentum_trend_composite',
    use_dynamic_switching=True,
    max_position_size=0.1,
    stop_loss_pct=0.02,
    take_profit_pct=0.04
)

# Run backtest
results = cerebro.run()
```

## Available Strategies

### Composite Strategies

1. **momentum_trend_composite**
   - Combines RSI momentum and SuperTrend trend following
   - Uses weighted voting for signal aggregation
   - Suitable for trending markets

2. **mean_reversion_momentum**
   - Combines Bollinger Bands mean reversion and MACD momentum
   - Uses consensus voting for signal aggregation
   - Suitable for ranging markets

3. **volatility_breakout**
   - Combines ATR breakout, Bollinger Bands breakout, and SuperTrend confirmation
   - Uses weighted voting for signal aggregation
   - Suitable for volatile markets

### Multi-Timeframe Strategies

1. **trend_following_mtf**
   - Uses 4H for trend direction, 1H for entries, 15M for exits
   - Requires trend confirmation for entries
   - Uses higher timeframe stops

2. **breakout_mtf**
   - Uses 1H for breakout detection, 4H for confirmation, 15M for execution
   - Requires trend alignment and volume confirmation
   - Uses breakout levels as support/resistance

3. **mean_reversion_mtf**
   - Uses 4H for mean calculation, 1H for entries, 30M for exits
   - Only trades at extremes
   - Uses dynamic targets

### Dynamic Switching Strategies

1. **market_regime_adaptive**
   - Switches based on volatility and trend strength
   - Uses different strategies for different market regimes
   - Smooth transitions between strategies

2. **performance_based_switching**
   - Switches based on recent performance metrics
   - Uses Sharpe ratio, drawdown, and win rate
   - Fallback strategies for poor performance

3. **time_based_switching**
   - Switches based on market sessions
   - Different strategies for Asian, London, and New York sessions
   - Handles session overlaps

### Portfolio Optimization Strategies

1. **modern_portfolio_theory**
   - Uses MPT for optimal asset allocation
   - Maximizes Sharpe ratio
   - Weekly rebalancing

2. **risk_parity**
   - Equalizes risk contribution across assets
   - Uses volatility as risk measure
   - Daily rebalancing with volatility adjustment

3. **momentum_portfolio**
   - Allocates based on relative momentum
   - Top N assets with equal weights
   - Monthly rebalancing

4. **dynamic_asset_allocation**
   - Adapts allocation based on market conditions
   - Different allocations for trending, ranging, and crisis markets
   - Weekly rebalancing with regime change detection

## Performance Metrics

The framework tracks comprehensive performance metrics:

- **Total Return**: Overall strategy performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Total Trades**: Number of completed trades
- **Strategy Switches**: Number of strategy changes
- **Regime Detection**: Market regime classification accuracy

## Risk Management

### Position Sizing
- Dynamic position sizing based on signal confidence
- Maximum position size limits
- Risk-adjusted position sizing

### Stop Loss and Take Profit
- Percentage-based stop losses
- Trailing stops for open positions
- Dynamic take profit levels

### Portfolio Risk
- Maximum portfolio volatility limits
- Correlation-based position sizing
- VaR and CVaR limits

## Best Practices

### 1. Strategy Selection
- Choose strategies based on market conditions
- Use dynamic switching for adaptive behavior
- Monitor strategy performance regularly

### 2. Risk Management
- Set appropriate position size limits
- Use trailing stops for open positions
- Monitor drawdown and adjust accordingly

### 3. Configuration
- Start with default configurations
- Adjust parameters based on backtesting results
- Use walk-forward analysis for parameter validation

### 4. Monitoring
- Track strategy performance metrics
- Monitor strategy switching frequency
- Watch for regime change detection accuracy

## Troubleshooting

### Common Issues

1. **No Signals Generated**
   - Check data feed availability
   - Verify strategy configuration
   - Ensure sufficient data for indicators

2. **Frequent Strategy Switching**
   - Adjust regime detection thresholds
   - Increase minimum regime duration
   - Check market data quality

3. **Poor Performance**
   - Review strategy selection logic
   - Check risk management parameters
   - Validate market regime detection

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check strategy summary:

```python
summary = strategy.get_strategy_summary()
print(summary)
```

## Future Enhancements

### Planned Features

1. **Machine Learning Integration**
   - ML-based regime detection
   - Automated strategy selection
   - Predictive signal generation

2. **Advanced Portfolio Optimization**
   - Black-Litterman model
   - Risk parity with multiple risk measures
   - Dynamic correlation modeling

3. **Real-time Optimization**
   - Online parameter optimization
   - Adaptive strategy weights
   - Real-time performance monitoring

4. **Enhanced Risk Management**
   - Options-based hedging
   - Dynamic volatility targeting
   - Stress testing framework

## Conclusion

The Advanced Strategy Framework provides a comprehensive solution for sophisticated trading strategies. It combines multiple approaches to create robust, adaptive trading systems that can perform well across different market conditions.

For more information, see the example scripts in the `examples/` directory and the configuration files in `config/strategy/`. 
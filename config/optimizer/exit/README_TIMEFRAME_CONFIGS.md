# Timeframe-Specific Exit Mixin Configurations

## Overview

This directory contains timeframe-specific optimization configurations for exit mixins, based on the comprehensive trading guide in `docs/_TODO.md`. Each configuration is tailored to specific timeframes and trading styles to optimize exit performance.

## Configuration Files

### RSIBBExitMixin (RSI + Bollinger Bands Exit Strategy)

| Timeframe | Trading Style | RSI Period | RSI Overbought | BB Period | BB Deviation |
|-----------|---------------|------------|----------------|-----------|--------------|
| **5m**    | Day Trading   | 7-9        | 70-80          | 10-14     | 1.5-2.0      |
| **15m**   | Day Trading   | 9-12       | 70-80          | 12-18     | 1.5-2.0      |
| **1h**    | Swing Trading | 12-20      | 65-80          | 14-22     | 1.8-2.2      |
| **4h**    | Swing Trading | 14-24      | 65-80          | 16-24     | 2.0-2.5      |
| **1d**    | Position      | 14-30      | 60-80          | 18-26     | 2.0-2.5      |

### TrailingStopExitMixin (ATR-Based Trailing Stop)

| Timeframe | Trading Style | Trail % | ATR Period | ATR Multiplier | Activation % |
|-----------|---------------|---------|------------|----------------|--------------|
| **5m**    | Day Trading   | 0.5-2.0 | 7-10       | 1.0-2.0        | 0.0-0.5      |
| **15m**   | Day Trading   | 1.0-3.0 | 8-12       | 1.0-2.0        | 0.0-0.5      |
| **1h**    | Swing Trading | 1.0-4.0 | 10-14      | 1.5-2.5        | 0.0-0.5      |
| **4h**    | Swing Trading | 2.0-5.0 | 12-18      | 1.8-2.5        | 0.0-0.5      |
| **1d**    | Position      | 3.0-8.0 | 14-20      | 2.0-3.0        | 0.0-0.5      |

## Key Principles

### 1. **Timeframe Scaling for Exit Strategies**
- **Shorter timeframes (5m, 15m)**: Tighter stops, faster RSI, more responsive exits
- **Longer timeframes (4h, 1d)**: Wider stops, slower RSI, more patient exits
- **Exit vs Entry**: Exit strategies typically use wider parameters to avoid premature exits

### 2. **RSI Overbought Levels**
- **Day Trading (5m, 15m)**: 70-80 (tighter exits)
- **Swing Trading (1h, 4h)**: 65-80 (balanced exits)
- **Position Trading (1d)**: 60-80 (wider exits to catch major moves)

### 3. **ATR-Based Trailing Stops**
- **Day Trading**: 1.0-2.0x ATR multiplier (tight stops)
- **Swing Trading**: 1.5-2.5x ATR multiplier (balanced stops)
- **Position Trading**: 2.0-3.0x ATR multiplier (wide stops)

### 4. **Bollinger Bands for Exits**
- **Day Trading**: Period 10-18, Deviation 1.5-2.0 (responsive)
- **Swing Trading**: Period 14-24, Deviation 1.8-2.5 (balanced)
- **Position Trading**: Period 18-26, Deviation 2.0-2.5 (patient)

## Usage

### For Optimization
```bash
# Use specific timeframe configuration
python run_optimizer.py --config config/optimizer/exit/RSIBBExitMixin_1h.json
```

### For Backtesting
```python
# Load timeframe-specific parameters
with open('config/optimizer/exit/TrailingStopExitMixin_4h.json') as f:
    config = json.load(f)
    
# Apply to strategy
strategy_params = config['params']
```

## Expected Improvements

### 1. **Better Exit Timing**
- **5m/15m**: Faster exits to capture quick profits
- **1h/4h**: Balanced exits for swing trades
- **1d**: Patient exits for major trends

### 2. **Reduced False Exits**
- **Timeframe-appropriate parameters**: Less noise, better signal
- **Volatility-adjusted stops**: Adapt to market conditions
- **Trading style alignment**: Match exit strategy to timeframe

### 3. **Improved Risk Management**
- **Tighter stops for day trading**: Protect capital in fast markets
- **Wider stops for position trading**: Allow for normal market fluctuations
- **ATR-based stops**: Adapt to current volatility

## Migration from Generic Configs

The original generic configurations used suboptimal parameter ranges:

| Parameter | Generic Range | 1h Optimized | Improvement |
|-----------|---------------|--------------|-------------|
| RSI Period | 5-30 | 12-20 | Better responsiveness |
| RSI Overbought | 60-90 | 65-80 | More appropriate levels |
| BB Period | 35-52 | 14-22 | Better market fit |
| ATR Period | 5-30 | 10-14 | Optimal volatility capture |
| ATR Multiplier | 1.0-5.0 | 1.5-2.5 | More reasonable stops |

## Exit Strategy Logic

### RSIBBExitMixin
- **Exits when**: RSI is overbought AND price touches upper Bollinger Band
- **Logic**: Take profits when momentum is high and price is at resistance
- **Timeframe adaptation**: Faster RSI for shorter timeframes, wider bands for longer timeframes

### TrailingStopExitMixin
- **Exits when**: Price moves against position by ATR-based distance
- **Logic**: Protect profits while allowing for normal market fluctuations
- **Timeframe adaptation**: Tighter stops for shorter timeframes, wider stops for longer timeframes

## Next Steps

1. **Test each timeframe configuration** with historical data
2. **Implement volatility-based adjustments** in the optimizer
3. **Add regime detection** for automatic parameter adjustment
4. **Create similar configurations** for other exit mixins
5. **Implement walk-forward optimization** for parameter validation

## References

- Trading guide: `docs/_TODO.md`
- Original configurations: `RSIBBExitMixin.json`, `TrailingStopExitMixin.json`
- Mixin implementations: `src/strategy/exit/rsi_bb_exit_mixin.py`, `src/strategy/exit/trailing_stop_exit_mixin.py`

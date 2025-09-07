# Timeframe-Specific Optimization Configurations

## Overview

This directory contains timeframe-specific optimization configurations for entry mixins, based on the comprehensive trading guide in `docs/_TODO.md`. Each configuration is tailored to specific timeframes and trading styles to optimize performance.

## Configuration Files

### RSIBBEntryMixin (AND Logic - Both RSI and BB conditions required)

| Timeframe | Trading Style | RSI Period | RSI Oversold | BB Period | BB Deviation |
|-----------|---------------|------------|--------------|-----------|--------------|
| **5m**    | Day Trading   | 7-9        | 20-25        | 10-14     | 1.5-2.0      |
| **15m**   | Day Trading   | 9-12       | 20-30        | 12-18     | 1.5-2.0      |
| **1h**    | Swing Trading | 12-20      | 25-35        | 14-22     | 1.8-2.2      |
| **4h**    | Swing Trading | 14-24      | 25-35        | 16-24     | 2.0-2.5      |
| **1d**    | Position      | 14-30      | 20-35        | 18-26     | 2.0-2.5      |

### RSIOrBBEntryMixin (OR Logic - Either RSI or BB condition sufficient)

| Timeframe | Trading Style | RSI Period | RSI Oversold | BB Period | BB Deviation |
|-----------|---------------|------------|--------------|-----------|--------------|
| **5m**    | Day Trading   | 7-9        | 20-25        | 10-14     | 1.5-2.0      |
| **15m**   | Day Trading   | 9-12       | 20-30        | 12-18     | 1.5-2.0      |
| **1h**    | Swing Trading | 12-20      | 25-35        | 14-22     | 1.8-2.2      |
| **4h**    | Swing Trading | 14-24      | 25-35        | 16-24     | 2.0-2.5      |
| **1d**    | Position      | 14-30      | 20-35        | 18-26     | 2.0-2.5      |

## Key Principles

### 1. **Timeframe Scaling**
- **Shorter timeframes (5m, 15m)**: Faster indicators, tighter parameters
- **Longer timeframes (4h, 1d)**: Slower indicators, wider parameters
- **Crypto markets**: Reduce periods by 20% due to higher volatility

### 2. **Trading Style Alignment**
- **Day Trading**: Focus on speed and responsiveness
- **Swing Trading**: Balance between responsiveness and noise reduction
- **Position Trading**: Focus on trend following and noise reduction

### 3. **Volatility Adjustments**
- **High Volatility**: Decrease periods by 15-25%
- **Low Volatility**: Increase periods by 20-30%
- **Medium Volatility**: Use default parameters

## Usage

### For Optimization
```bash
# Use specific timeframe configuration
python run_optimizer.py --config config/optimizer/entry/RSIBBEntryMixin_1h.json
```

### For Backtesting
```python
# Load timeframe-specific parameters
with open('config/optimizer/entry/RSIBBEntryMixin_4h.json') as f:
    config = json.load(f)
    
# Apply to strategy
strategy_params = config['params']
```

## Expected Improvements

### 1. **More Frequent Entries**
- **5m/15m**: Faster RSI periods (7-12 vs 10-20) = more signals
- **1h/4h**: Better balanced parameters = optimal signal frequency
- **1d**: Wider ranges = catch major moves

### 2. **Better Signal Quality**
- **Day Trading**: Tighter RSI oversold (20-25) = higher quality signals
- **Swing Trading**: Balanced parameters = good risk/reward
- **Position Trading**: Wider ranges = major trend entries

### 3. **Reduced False Signals**
- **Timeframe-appropriate periods**: Less noise, more signal
- **Volatility-adjusted parameters**: Adapt to market conditions
- **Trading style alignment**: Match strategy to timeframe

## Migration from Generic Configs

The original generic configurations (`RSIBBEntryMixin.json`, `RSIOrBBEntryMixin.json`) used suboptimal parameter ranges:

| Parameter | Generic Range | 1h Optimized | Improvement |
|-----------|---------------|--------------|-------------|
| RSI Period | 10-20 | 12-20 | Better responsiveness |
| RSI Oversold | 35-50 | 25-35 | More opportunities |
| BB Period | 15-25 | 14-22 | Better market fit |
| BB Deviation | 1.5-2.5 | 1.8-2.2 | Optimal volatility capture |

## Next Steps

1. **Test each timeframe configuration** with historical data
2. **Implement volatility-based adjustments** in the optimizer
3. **Add regime detection** for automatic parameter adjustment
4. **Create similar configurations** for other entry mixins (MACD, Ichimoku, etc.)
5. **Implement walk-forward optimization** for parameter validation

## References

- Trading guide: `docs/_TODO.md`
- Original configurations: `RSIBBEntryMixin.json`, `RSIOrBBEntryMixin.json`
- Mixin implementations: `src/strategy/entry/rsi_bb_entry_mixin.py`, `src/strategy/entry/rsi_or_bb_entry_mixin.py`

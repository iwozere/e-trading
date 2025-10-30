# Advanced ATR Exit Strategy

## Overview

The `AdvancedATRExitMixin` is a sophisticated, volatility-adaptive trailing stop exit strategy that implements a comprehensive state machine with multiple phases, structural ratcheting, and time-based tightening. This strategy is designed for both backtesting and live trading with consistent behavior.

## Key Features

### 🎯 **Multi-Timeframe ATR Analysis**
- **Fast ATR**: Short-term volatility (default: 7 periods)
- **Slow ATR**: Long-term volatility (default: 21 periods)  
- **Higher Timeframe ATR**: Regime-level volatility (optional)
- **Effective ATR**: Weighted aggregation of all ATR components

### 🔄 **State Machine**
- **INIT**: Initial stop placement
- **ARMED**: Break-even ready state
- **PHASE1**: Running trail with standard multiplier
- **PHASE2**: Tighter trail after profit target
- **LOCKED**: Parabolic-style tightening
- **EXIT**: Position closed

### 🏗️ **Structural Ratcheting**
- **Swing Detection**: Identifies significant price levels
- **Buffer Management**: Places stops beyond swing levels
- **Monotonic Ratcheting**: Never loosens stops

### ⏰ **Time-Based Tightening**
- **Stagnation Detection**: Identifies markets without new highs/lows
- **Adaptive Multipliers**: Tightens stops in stagnant conditions
- **Cooldown Periods**: Prevents over-tightening

### 💰 **Partial Take-Profit**
- **Multiple Levels**: Configurable profit-taking points
- **Size Management**: Fractional position reduction
- **Retuning**: Adjusts strategy after each PT

## Configuration

### Core Parameters

```json
{
  "anchor": "high",           // Price reference: high, close, mid
  "k_init": 2.5,             // Initial ATR multiplier (1.0-6.0)
  "k_run": 2.0,              // Running ATR multiplier (1.0-5.0)
  "k_phase2": 1.5,           // Phase 2 ATR multiplier (0.8-3.0)
  "p_fast": 7,               // Fast ATR period (5-21)
  "p_slow": 21,              // Slow ATR period (14-100)
  "use_htf_atr": true,       // Enable higher timeframe ATR
  "htf": "4h"                // Higher timeframe
}
```

### ATR Aggregation

```json
{
  "alpha_fast": 1.0,         // Fast ATR weight (0.25-2.0)
  "alpha_slow": 1.0,         // Slow ATR weight (0.25-2.0)
  "alpha_htf": 1.0,          // HTF ATR weight (0.25-2.0)
  "atr_floor": 0.0           // Minimum ATR value
}
```

### Break-Even & Phases

```json
{
  "arm_at_R": 1.0,           // Break-even trigger (0.5-2.0)
  "breakeven_offset_atr": 0.0, // BE offset in ATR units (-0.25-0.25)
  "phase2_at_R": 2.0         // Phase 2 trigger (1.0-4.0)
}
```

### Structural Ratchet

```json
{
  "use_swing_ratchet": true, // Enable swing-based ratcheting
  "swing_lookback": 10,      // Swing detection lookback (5-30)
  "struct_buffer_atr": 0.25  // Structural buffer (0.0-0.6)
}
```

### Time-Based Tightening

```json
{
  "tighten_if_stagnant_bars": 20,  // Stagnation threshold (10-60)
  "tighten_k_factor": 0.8,         // Tightening factor (0.6-0.95)
  "min_bars_between_tighten": 5    // Cooldown period (1-10)
}
```

### Noise & Step Filters

```json
{
  "min_stop_step": 0.0,      // Minimum stop movement (0.0-0.001)
  "noise_filter_atr": 0.0,   // Noise filter threshold (0.0-0.4)
  "max_trail_freq": 1,       // Maximum trail frequency (1-5)
  "tick_size": 0.01          // Price rounding (0.0001-0.1)
}
```

### Partial Take-Profit

```json
{
  "pt_levels_R": [1.0, 2.0], // Profit levels in R multiples
  "pt_sizes": [0.33, 0.33],  // Position sizes to exit
  "retune_after_pt": true    // Adjust strategy after PT
}
```

## Usage Examples

### Basic Usage

```python
from src.strategy.exit.exit_mixin_factory import get_exit_mixin

# Create with default parameters
exit_mixin = get_exit_mixin("AdvancedATRExitMixin")

# Create with custom parameters
params = {
    "k_init": 3.0,
    "k_run": 2.5,
    "use_swing_ratchet": True,
    "pt_levels_R": [1.5, 3.0],
    "pt_sizes": [0.5, 0.3]
}
exit_mixin = get_exit_mixin("AdvancedATRExitMixin", params)
```

### Configuration-Based Usage

```python
from src.strategy.exit.exit_mixin_factory import get_exit_mixin_from_config

config = {
    "name": "AdvancedATRExitMixin",
    "params": {
        "anchor": "high",
        "k_init": 2.5,
        "k_run": 2.0,
        "p_fast": 7,
        "p_slow": 21,
        "use_htf_atr": True,
        "arm_at_R": 1.0,
        "phase2_at_R": 2.0,
        "use_swing_ratchet": True,
        "swing_lookback": 10,
        "pt_levels_R": [1.0, 2.0],
        "pt_sizes": [0.33, 0.33]
    }
}

exit_mixin = get_exit_mixin_from_config(config)
```

## Optimization Strategy

The strategy supports multi-stage optimization:

### Stage A: Core Parameters
- `k_init`, `k_run`, `p_fast`, `p_slow`, `use_htf_atr`

### Stage B: Risk Management
- `arm_at_R`, `breakeven_offset_atr`, `phase2_at_R`, `k_phase2`
- `use_swing_ratchet`, `struct_buffer_atr`

### Stage C: Fine-Tuning
- `tighten_if_stagnant_bars`, `tighten_k_factor`, `noise_filter_atr`
- `alpha_fast`, `alpha_slow`, `alpha_htf`

## Performance Metrics

The strategy tracks comprehensive metrics:

### Trade-Level Metrics
- R multiple achieved
- MAE (Maximum Adverse Excursion)
- MFE (Maximum Favorable Excursion)
- Exit reason (stop hit, partial TP, etc.)

### Strategy-Level Metrics
- State transitions
- ATR effectiveness
- Structural ratchet usage
- Time-based tightening events

## Best Practices

### 1. **Parameter Selection**
- Start with default parameters
- Optimize in stages (A → B → C)
- Use walk-forward analysis for robustness

### 2. **Market Conditions**
- **Trending Markets**: Use higher `k_init` and `k_run`
- **Ranging Markets**: Enable time-based tightening
- **High Volatility**: Increase `atr_floor` and `noise_filter_atr`

### 3. **Timeframe Considerations**
- **Intraday**: Use shorter ATR periods (5-14)
- **Daily**: Use longer ATR periods (14-50)
- **Crypto**: Enable HTF ATR to avoid micro-noise

### 4. **Risk Management**
- Set appropriate `arm_at_R` for break-even
- Use partial take-profits for large positions
- Monitor maximum drawdown constraints

## Integration

The mixin integrates seamlessly with:

- **Base Strategy**: Automatic state management
- **Optimizer**: Multi-stage parameter optimization
- **Live Trading**: Consistent backtest/live behavior
- **Logging**: Comprehensive event tracking

## Troubleshooting

### Common Issues

1. **Stops Too Tight**: Increase `k_init` and `k_run`
2. **Stops Too Loose**: Decrease multipliers or enable time-based tightening
3. **Frequent Whipsaws**: Increase `noise_filter_atr` or `min_stop_step`
4. **Missed Exits**: Check `max_trail_freq` and update frequency

### Debug Mode

Enable detailed logging to track:
- State transitions
- ATR calculations
- Stop updates
- Partial take-profit events

## Advanced Features

### Custom Anchors
- **High**: Most aggressive trailing
- **Close**: Balanced approach
- **Mid**: Conservative trailing

### Multi-Timeframe Analysis
- Combines multiple ATR timeframes
- Avoids over-tightening in micro-noise
- Respects regime-level volatility

### Structural Intelligence
- Identifies significant price levels
- Places stops beyond swing points
- Prevents premature exits on pullbacks

This advanced exit strategy provides institutional-grade risk management with sophisticated volatility adaptation and comprehensive state management.

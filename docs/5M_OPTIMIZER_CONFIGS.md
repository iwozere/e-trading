# 5-Minute Timeframe Optimizer Configurations

**Created:** 2025-11-12  
**Purpose:** Optimized parameter ranges for 5-minute scalping strategies

## Overview

All entry and exit strategies now have 5m-specific configuration files with parameters tuned for scalping on the 5-minute timeframe.

## Key Differences from Longer Timeframes

### Indicator Periods
- **Shorter periods**: RSI 10-18 (vs 14-24 for 4h), BB 15-25 (vs 16-24 for 4h)
- **Faster response**: Indicators need to react quickly to price changes in scalping

### Volume Parameters
- **Higher volume requirements**: min_volume_ratio 1.2-2.0 (vs 1.0-1.5 for 4h)
- **More sensitive**: Volume spikes are more significant on 5m charts

### Exit Parameters
- **Tighter stops**: Take profit 0.5-3% (vs 1-10% for 4h)
- **Shorter duration**: Max bars 6-48 (30min-4h) vs 5-100 bars for 4h

### Cooldown Periods
- **Longer in bars**: 3-12 bars (15-60 minutes) to avoid overtrading
- **More restrictive**: Prevents chasing every small move

## Entry Strategies - 5m Configs

### 1. BBVolumeSuperTrendEntryMixin_5m.json

**Parameters:**
- `e_bb_period`: 10-30 (default 20)
- `e_bb_dev`: 1.5-2.5 (default 2.0)
- `e_vol_ma_period`: 10-30 (default 20)
- `e_min_volume_ratio`: 1.1-2.0 (default 1.3)
- `e_st_period`: 7-20 (default 10)
- `e_st_multiplier`: 2.0-4.0 (default 3.0)

**Rationale:**
- Shorter BB periods to catch quick moves
- Higher volume ratio requirement for signal quality
- SuperTrend with moderate period for trend confirmation

### 2. RSIBBEntryMixin_5m.json

**Parameters:**
- `e_rsi_period`: 10-18 (default 14)
- `e_rsi_oversold`: 20-35 (default 30)
- `e_bb_period`: 15-25 (default 20)
- `e_bb_dev`: 1.8-2.5 (default 2.0)
- `e_cooldown_bars`: 3-12 (default 5)

**Rationale:**
- Faster RSI for scalping
- Slightly relaxed oversold level (20-35 vs 18-32 for longer TF)
- Cooldown prevents overtrading

### 3. RSIBBVolumeEntryMixin_5m.json

**Parameters:**
- All RSI/BB params as above
- `e_vol_ma_period`: 10-30 (default 20)
- `e_min_volume_ratio`: 1.2-2.0 (default 1.5)
- `e_rsi_cross`: true/false (default true)
- `e_bb_reentry`: true/false (default true)

**Rationale:**
- Combines RSI, BB, and volume for higher probability setups
- Volume filter critical for 5m to avoid noise

### 4. RSIIchimokuEntryMixin_5m.json

**Parameters:**
- `e_rsi_period`: 10-18 (default 14)
- `e_rsi_oversold`: 20-35 (default 30)
- `e_tenkan`: 7-12 (default 9)
- `e_kijun`: 20-32 (default 26)
- `e_senkou`: 40-60 (default 52)
- `e_cooldown_bars`: 3-12 (default 5)

**Rationale:**
- Standard Ichimoku periods work well even on 5m
- Provides multi-timeframe perspective in single indicator

### 5. RSIOrBBEntryMixin_5m.json

**Parameters:**
- `e_rsi_period`: 10-18 (default 14)
- `e_rsi_oversold`: 20-35 (default 30)
- `e_bb_period`: 15-25 (default 20)
- `e_bb_dev`: 1.8-2.5 (default 2.0)
- `e_rsi_cross`: true/false (default true)
- `e_bb_reentry`: true/false (default true)
- `e_cooldown_bars`: 3-12 (default 5)

**Rationale:**
- OR logic provides more signals than AND logic
- Good for active scalping

### 6. RSIVolumeSupertrendEntryMixin_5m.json

**Parameters:**
- `e_rsi_period`: 10-18 (default 14)
- `e_rsi_oversold`: 20-35 (default 30)
- `e_vol_ma_period`: 10-30 (default 20)
- `e_min_volume_ratio`: 1.2-2.0 (default 1.5)
- `e_st_period`: 7-20 (default 10)
- `e_st_multiplier`: 2.0-4.0 (default 3.0)
- `e_cooldown_bars`: 3-12 (default 5)

**Rationale:**
- Triple confirmation: RSI + Volume + SuperTrend
- High quality signals but fewer entries

## Exit Strategies - 5m Configs

### 1. AdvancedATRExitMixin_5m.json

Already exists (created earlier)

### 2. ATRExitMixin_5m.json

Already exists (created earlier)

### 3. FixedRatioExitMixin_5m.json

**Parameters:**
- `x_take_profit`: 0.005-0.03 (0.5-3%, default 1.5%)
- `x_stop_loss`: 0.003-0.02 (0.3-2%, default 1%)

**Rationale:**
- Tight profit targets for scalping
- Risk/reward ratio ~1.5:1
- Quick in/out for 5m timeframe

### 4. MACrossoverExitMixin_5m.json

**Parameters:**
- `x_ma_period`: 10-30 (default 20)
- `x_ma_type`: SMA/EMA (default EMA)

**Rationale:**
- Shorter MA period for faster exits
- EMA default for quicker response to price changes

### 5. RSIBBExitMixin_5m.json

Already exists (created earlier)

### 6. RSIOrBBExitMixin_5m.json

Already exists (created earlier)

### 7. SimpleATRExitMixin_5m.json

Already exists (created earlier)

### 8. TimeBasedExitMixin_5m.json

**Parameters:**
- `x_max_bars`: 6-48 (30min-4h, default 12 bars = 1h)
- `x_max_minutes`: 30-240 (default 60 minutes)
- `x_use_time`: true/false (default true)

**Rationale:**
- Force exit after 1-4 hours maximum
- Prevents holding losing positions overnight
- Use time-based exit by default for scalping discipline

### 9. TrailingStopExitMixin_5m.json

Already exists (created earlier)

## Usage

The walk-forward optimizer will automatically load the 5m configuration when optimizing strategies on 5m data:

```bash
# Example walk_forward_config.json
{
  "timeframes": ["5m"],
  "symbols": ["BTCUSDT"],
  "entry_strategies": ["RSIOrBBEntryMixin"],
  "exit_strategies": ["FixedRatioExitMixin"]
}
```

The optimizer looks for:
1. `config/optimizer/entry/RSIOrBBEntryMixin_5m.json` (timeframe-specific) ✅
2. Falls back to `config/optimizer/entry/RSIOrBBEntryMixin.json` (generic) if not found

## Performance Expectations

### Typical 5m Strategy Characteristics
- **Trade frequency**: 5-20 trades per day
- **Hold time**: 30 minutes - 2 hours
- **Win rate**: 50-60% (higher frequency = more noise)
- **Profit per trade**: 0.5-2%
- **Max drawdown**: 5-15%

### Optimization Considerations
- More trials needed (100-200) due to higher variance
- Commission costs more significant (0.1% = 20% of 0.5% target)
- Slippage more impactful on tight profit targets
- Data quality critical (every tick matters)

## Risk Management for 5m

### Position Sizing
- Smaller positions (2-5% per trade vs 10% for longer TF)
- Max 3-5 concurrent positions
- Account for commission drag

### Stop Losses
- Tighter stops (0.3-1% vs 1-3% for 4h)
- Use ATR-based stops for volatility adjustment
- Quick exits on adverse moves

### Profit Taking
- Scale out at multiple levels (50% at 1%, 50% at 2%)
- Trail stops aggressively
- Don't get greedy - 1% is good on 5m

## Comparison: 5m vs 4h Parameters

| Parameter | 5m Range | 4h Range | Reason |
|-----------|----------|----------|--------|
| RSI Period | 10-18 | 14-24 | Faster response needed |
| RSI Oversold | 20-35 | 18-32 | Slightly more relaxed |
| BB Period | 15-25 | 16-24 | Similar range |
| BB Deviation | 1.8-2.5 | 2.0-3.0 | Tighter bands |
| Volume Ratio | 1.2-2.0 | 1.0-1.5 | Higher threshold |
| Take Profit | 0.5-3% | 1-10% | Much tighter |
| Stop Loss | 0.3-2% | 0.5-5% | Tighter stops |
| Cooldown Bars | 3-12 | 2-8 | More bars (same time) |

## Files Created

**Entry Strategies:**
1. `config/optimizer/entry/BBVolumeSuperTrendEntryMixin_5m.json`
2. `config/optimizer/entry/RSIBBVolumeEntryMixin_5m.json`
3. `config/optimizer/entry/RSIIchimokuEntryMixin_5m.json`
4. `config/optimizer/entry/RSIVolumeSupertrendEntryMixin_5m.json`

**Exit Strategies:**
1. `config/optimizer/exit/FixedRatioExitMixin_5m.json`
2. `config/optimizer/exit/MACrossoverExitMixin_5m.json`
3. `config/optimizer/exit/TimeBasedExitMixin_5m.json`

**Already Existed:**
- `RSIBBEntryMixin_5m.json`
- `RSIOrBBEntryMixin_5m.json`
- `AdvancedATRExitMixin_5m.json`
- `ATRExitMixin_5m.json`
- `RSIBBExitMixin_5m.json`
- `RSIOrBBExitMixin_5m.json`
- `SimpleATRExitMixin_5m.json`
- `TrailingStopExitMixin_5m.json`

## Total: 6 Entry + 9 Exit = 54 Strategy Combinations for 5m

---

*All configurations tested and ready for walk-forward optimization on 5-minute timeframe data.*

# HMM Regime Analysis Debug Guide

## Issue: Missing Bearish Regimes in 4h Timeframe

### Problem Description
When running HMM training on 4h timeframes, you might observe that only "Sideways" and "Bullish" regimes are detected, with no "Bearish" regime appearing. This is a common issue with longer timeframes.

### Root Cause Analysis

#### 1. **Timeframe Characteristics**
- **4h timeframes have less volatile data** - longer timeframes smooth out short-term noise
- **All detected regimes might have positive average returns** - even the "worst" regime could be slightly positive
- **HMM focuses on relative differences** - it identifies distinct states, not necessarily bearish/bullish

#### 2. **Original Labeling Logic (Problematic)**
```python
# OLD LOGIC - Simple sorting by average return
regime_stats.sort(key=lambda x: x['avg_return'])
if i == 0:
    labels[stat['regime_id']] = 'Bearish'  # Always lowest return
elif i == 1:
    labels[stat['regime_id']] = 'Sideways' # Always middle return
else:
    labels[stat['regime_id']] = 'Bullish'  # Always highest return
```

**Problem**: If all three regimes have positive returns (e.g., 0.001, 0.002, 0.003), you get "Sideways", "Sideways", "Bullish" instead of "Bearish", "Sideways", "Bullish".

### Solutions Implemented

#### 1. **Improved Labeling Logic**
```python
# NEW LOGIC - Dynamic thresholds based on data characteristics
if self.n_components == 3:
    # Calculate return range for better thresholds
    returns = [stat['avg_return'] for stat in regime_stats]
    min_return = min(returns)
    max_return = max(returns)
    return_range = max_return - min_return
    
    # Dynamic thresholds based on data characteristics
    if return_range > 0.001:  # Good separation between regimes
        # Use relative positioning with dynamic thresholds
        for i, stat in enumerate(regime_stats):
            avg_return = stat['avg_return']
            relative_position = (avg_return - min_return) / return_range
            
            if i == 0:  # Lowest return regime
                if relative_position < 0.3:  # Bottom 30% of return range
                    labels[stat['regime_id']] = 'Bearish'
                else:
                    labels[stat['regime_id']] = 'Sideways'
            elif i == 1:  # Middle return regime
                if relative_position > 0.7:  # Top 30% of return range
                    labels[stat['regime_id']] = 'Bullish'
                else:
                    labels[stat['regime_id']] = 'Sideways'
            else:  # Highest return regime
                labels[stat['regime_id']] = 'Bullish'
    else:
        # Poor separation - use absolute thresholds and ensure all three labels
        for i, stat in enumerate(regime_stats):
            avg_return = stat['avg_return']
            
            if i == 0:  # Always label lowest as Bearish or Sideways
                if avg_return < 0:
                    labels[stat['regime_id']] = 'Bearish'
                else:
                    labels[stat['regime_id']] = 'Sideways'
            elif i == 1:  # Middle gets Sideways or Bullish
                if avg_return > 0.0005:  # Higher threshold for bullish
                    labels[stat['regime_id']] = 'Bullish'
                else:
                    labels[stat['regime_id']] = 'Sideways'
            else:  # Highest always gets Bullish
                labels[stat['regime_id']] = 'Bullish'
```

#### 2. **Enhanced Debugging**
Added `debug_regime_analysis()` method that provides:
- Overall data statistics (mean, std, negative returns percentage)
- Detailed regime analysis for each detected state
- Bearish candidate identification
- Comprehensive logging for troubleshooting

#### 3. **Color Mapping Improvements**
- Tracks used colors to ensure all three main colors are present
- Fallback mechanism to force assign missing colors
- Better visual variety in regime visualizations

#### 4. **Better Logging**
- Timeframe-specific information
- Sample counts per regime
- Return thresholds and volatility metrics
- Regime distribution statistics
- Return range analysis and labeling method used

### Expected Behavior by Timeframe

#### **5m Timeframes**
- ✅ Usually shows all three regimes clearly
- High volatility allows distinct bearish/bullish/sideways states
- Short-term noise creates clear regime separation

#### **15m Timeframes**
- ✅ Generally shows all three regimes
- Good balance of volatility and trend detection
- Clear regime differentiation

#### **1h Timeframes**
- ✅ Usually shows all three regimes
- Moderate volatility with trend persistence
- Good regime separation

#### **4h Timeframes**
- ⚠️ May show only 2 regimes (Sideways + Bullish)
- Lower volatility, longer trends
- All regimes might have positive returns
- **Solution**: Improved labeling logic handles this case

#### **1d Timeframes**
- ⚠️ Often shows only 2 regimes
- Very low volatility, long-term trends
- May need different approach (2-regime model)

### Debugging Steps

#### 1. **Check Logs**
Look for debug output like:
```
=== Regime Analysis Debug for 4h ===
Overall data: mean_return=0.000234, std_return=0.023456, negative_pct=45.2%
Regime counts: {0: 1234, 1: 2345, 2: 3456}
Regime 0: samples=1234, avg_return=0.000123, negative_pct=48.5%, bearish_candidate=False
Regime 1: samples=2345, avg_return=0.000234, negative_pct=45.2%, bearish_candidate=False
Regime 2: samples=3456, avg_return=0.000345, negative_pct=42.1%, bearish_candidate=False
```

#### 2. **Interpret Results**
- **All positive returns**: Normal for 4h, use improved labeling
- **Low negative percentage**: Expected for longer timeframes
- **Bearish candidates**: Check if any regime qualifies

#### 3. **Consider Alternatives**
If 4h consistently shows only 2 regimes:
- **Option 1**: Use 2-regime model (Bearish/Bullish or Sideways/Bullish)
- **Option 2**: Adjust HMM parameters (covariance_type, n_iter)
- **Option 3**: Use different features (volatility-based instead of return-based)

### Configuration Recommendations

#### For 4h Timeframes
```yaml
hmm:
  n_components: 3  # Keep 3, let improved labeling handle it
  covariance_type: 'full'  # Better for complex regimes
  n_iter: 200  # More iterations for convergence
```

#### Alternative: 2-Regime Model
```yaml
hmm:
  n_components: 2  # Simpler model for 4h
  covariance_type: 'diag'  # Faster, sufficient for 2 regimes
```

### Best Practices

1. **Always check debug logs** for timeframe-specific behavior
2. **Don't assume 3 regimes** - let the data determine the number
3. **Use improved labeling** - handles edge cases automatically
4. **Consider timeframe characteristics** when interpreting results
5. **Validate regime quality** - ensure meaningful separation

### Future Improvements

1. **Adaptive regime count** based on data characteristics
2. **Volatility-based regime detection** for longer timeframes
3. **Multi-timeframe regime alignment** for consistency
4. **Regime quality metrics** to validate detection quality
5. **Custom labeling rules** per timeframe

---

*This guide helps understand and resolve regime detection issues across different timeframes.*

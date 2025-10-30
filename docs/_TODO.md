Here’s the cleaned and corrected `_TODO.md` in full **Markdown** — streamlined with only relevant timeframe/style combinations and corrected indicator parameters:

````markdown
# Trading Style Alignment

This guide provides recommended indicator parameters across trading styles and timeframes.  
The focus is on practicality: only realistic timeframe–style combinations are included.  

---

## 📊 Indicator Parameter Guide

| Indicator       | Timeframe | Day Trading              | Swing Trading            | Position Trading         |
|-----------------|-----------|--------------------------|--------------------------|--------------------------|
| **RSI**         | 5m        | Period: 7–9 Levels: 20/80 | – | – |
|                 | 15m       | Period: 9–10 Levels: 20/80 | – | – |
|                 | 1h        | Period: 12 Levels: 25/75 | Period: 14 Levels: 30/70 | – |
|                 | 4h        | – | Period: 18 Levels: 30/70 | – |
|                 | 1d        | – | Period: 20 Levels: 30/70 | Period: 21–30 Levels: 30/70 |
| **Bollinger Bands** | 5m    | Period: 10, SD: 2.0 | – | – |
|                 | 15m       | Period: 14, SD: 2.0 | – | – |
|                 | 1h        | Period: 20, SD: 2.0 | Period: 20, SD: 2.0 | – |
|                 | 4h        | – | Period: 20, SD: 2.0 | – |
|                 | 1d        | – | Period: 20, SD: 2.0 | Period: 20, SD: 2.0 |
| **Ichimoku**    | 5m        | Tenkan: 9, Kijun: 26, Span B: 52 | – | – |
|                 | 15m       | Tenkan: 9, Kijun: 26, Span B: 52 | – | – |
|                 | 1h        | Tenkan: 9, Kijun: 26, Span B: 52 | Tenkan: 9, Kijun: 26, Span B: 52 | – |
|                 | 4h        | – | Tenkan: 9, Kijun: 26, Span B: 52 | – |
|                 | 1d        | – | Tenkan: 9, Kijun: 26, Span B: 52 | Tenkan: 9, Kijun: 26, Span B: 52 |
| **ATR**         | 5m        | Period: 7, Multiplier: 1.5 | – | – |
|                 | 15m       | Period: 10, Multiplier: 1.5 | – | – |
|                 | 1h        | Period: 14, Multiplier: 2.0 | Period: 14, Multiplier: 2.0 | – |
|                 | 4h        | – | Period: 14–18, Multiplier: 2.0–2.2 | – |
|                 | 1d        | – | Period: 14–20, Multiplier: 2.0–2.5 | Period: 20–30, Multiplier: 2.5–3.0 |
| **MACD**        | 5m        | Fast: 8, Slow: 21, Signal: 5 | – | – |
|                 | 15m       | Fast: 8, Slow: 24, Signal: 5 | – | – |
|                 | 1h        | Fast: 12, Slow: 26, Signal: 9 | Fast: 12, Slow: 26, Signal: 9 | – |
|                 | 4h        | – | Fast: 12, Slow: 26, Signal: 9 | – |
|                 | 1d        | – | Fast: 12, Slow: 26, Signal: 9 | Fast: 12–18, Slow: 26–40, Signal: 9–12 |
| **Stochastic**  | 5m        | %K: 7, %D: 3, Slowing: 3 | – | – |
|                 | 15m       | %K: 10, %D: 3, Slowing: 3 | – | – |
|                 | 1h        | %K: 14, %D: 3, Slowing: 3 | %K: 14, %D: 3, Slowing: 3 | – |
|                 | 4h        | – | %K: 14, %D: 3, Slowing: 3 | – |
|                 | 1d        | – | %K: 14, %D: 3, Slowing: 3 | %K: 20, %D: 5, Slowing: 5 |
| **EMA**         | 5m        | 9, 21 | – | – |
|                 | 15m       | 12, 26 | – | – |
|                 | 1h        | 20, 50 | 20, 50 | – |
|                 | 4h        | – | 50, 100 | – |
|                 | 1d        | – | 50, 200 | 200, 500 |
| **ADX**         | All       | Period: 10, Level: 25 | Period: 14, Level: 20 | Period: 20, Level: 20 |
| **Volume Profile** | 15m–4h | Value Area: 70% | Value Area: 70% | – |
|                 | 1d        | – | Value Area: 70% | Value Area: 80% |
| **Pivot Points**| All       | Daily | Daily + Weekly | Weekly + Monthly |

---

## 📌 Key Implementation Notes

1. **Volatility Adjustments**  
   - Increase periods by 20–30% in low-volatility markets  
   - Decrease periods by 15–25% in high-volatility markets  
   ```python
   # Example: volatility-based ATR adjustment
   atr_period = 14 if current_volatility == 'medium' else (10 if high_vol else 18)
````

2. **Multi-Timeframe Confirmation**

   * Day traders: confirm 5m signals with 15m/1h
   * Swing traders: confirm 4h signals with 1d
   * Position traders: confirm 1d signals with weekly/monthly charts

3. **Indicator Combinations**

   | Trading Style | Primary Indicators     | Confirmation Indicators |
   | ------------- | ---------------------- | ----------------------- |
   | Day Trading   | EMA + Bollinger Bands  | Stochastic + Volume     |
   | Swing Trading | MACD + Ichimoku        | RSI + Pivot Points      |
   | Position      | EMA 200 + ADX + Pivots | Volume Profile + ATR    |

4. **Backtesting Protocol**

   ```python
   def optimize_params(timeframe, trading_style):
       # Walk-forward optimization with regime filtering
       params = {
           'rsi_period': optuna.suggest_int('rsi', 5, 30),
           'macd_fast': optuna.suggest_int('macd_f', 3, 15),
           # ... other parameters ...
       }
       sharpe = backtest(params, timeframe, trading_style)
       return sharpe
   ```

---

## 💡 Pro Tips

1. **Day Trading Focus**

   * On 5m/15m, prioritize speed with:

     * Shorter indicator periods
     * Tighter stop losses (1.5× ATR)
     * Volume confirmation for breakouts

2. **Swing Trading Essentials**

   * Use 4h Ichimoku cloud for trend direction
   * Combine MACD histogram with daily pivot points
   * Position size based on 4h ATR

3. **Position Trading Strategies**

   * Use daily ATR for volatility-adjusted targets
   * Confirm with weekly Ichimoku and EMA 200 trend
   * Always add fundamental analysis for validation

---

**Final Recommendation**: Always validate parameters through walk-forward optimization.
For crypto markets, reduce indicator periods by \~20% due to higher volatility.
Implement a regime detection system to auto-adjust parameters with market conditions.

```

---

✅ This is a **drop-in replacement** for your `_TODO.md`.  
Would you like me to also prepare a **shorter “cheat-sheet” table** (just RSI, BB, ATR, MACD, EMA) for quick reference during backtesting?
```

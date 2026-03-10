# P07 Combined Pipeline: Parameter Specification

This document details all tunable parameters used in the P07 Combined Pipeline for strategy optimization via Optuna.

## Optuna Configuration
- **Trials**: Default set to **500**.
- **Optimization Goal**: Maximize Adjusted Sharpe Ratio (Sharpe Ratio * log10(Total Trades)).

## Parameter List

### Strategy & Entry Parameters
| Parameter | Range | Type | Description |
| :--- | :--- | :--- | :--- |
| `rsi_period` | 7 – 21 | Integer | Period for relative strength index momentum. |
| `bb_period` | 10 – 30 | Integer | Lookback period for Bollinger Bands. |
| `bb_std` | 1.5 – 3.0 | Float | Standard deviation multiplier for Bollinger Bands. |
| `buy_prob_min` | 0.35 – 0.65| Float | Confidence threshold required to enter a Long position. |
| `sell_prob_min`| 0.35 – 0.65| Float | Confidence threshold required to enter a Short position. |

### Labeling & Risk Management
| Parameter | Range | Type | Description |
| :--- | :--- | :--- | :--- |
| `atr_period` | 10 – 20 | Integer | Period for volatility (ATR) used in barrier calculations. |
| `pt_mult` | 0.5 – 4.0 | Float | Profit Take multiplier (multiplied by ATR). |
| `sl_mult` | 0.25 – 3.0 | Float | Stop Loss multiplier (multiplied by ATR). |
| `tpl_hours` | 1.0 – 96.0 | Float | Vertical Barrier: maximum holding time in hours. |

### Feature Engineering
| Parameter | Range | Type | Description |
| :--- | :--- | :--- | :--- |
| `vol_lookback`| 10 – 40 | Integer | Lookback for volume Z-score calculation. |

### XGBoost Model Hyperparameters
| Parameter | Range | Type | Description |
| :--- | :--- | :--- | :--- |
| `max_depth` | 3 – 5 | Integer | Maximum depth of the decision trees. Limited to 5 to prevent overfitting. |
| `learning_rate`| 0.01 – 0.3 | Float | Step size shrinkage used to prevent overfitting (Log scale).|
| `n_estimators` | 50 – 200 | Integer | Number of boosting rounds (trees). |

---

## Statistical Guardrails
- **Minimum Trades**: Strategies must take at least **30 trades** during the optimization test window to be considered valid (fails with -1.0 score otherwise). This ensures a frequency of approx 1-2 trades/month.
- **Complexity Limit**: `max_depth` is hard-capped at **5** to ensure the model captures broad trends rather than memorizing noise.

# Strategy Framework

**Document Version:** 1.0  
**Status:** Active

## Related Documentation
- **[Unified Indicator Architecture](UNIFIED_INDICATOR_ARCHITECTURE.md)** — Config-driven indicator system (IndicatorFactory, registry, mixin access pattern)
- **[Trading Engine](modules/trading-engine.md)** — Execution, backtesting, live trading
- **[Data Management](modules/data-management.md)** — Market data providers and feeds
- **[Documentation Index](INDEX.md)** — Full documentation guide

---

## Overview

The Strategy Framework provides the architectural layer between raw market data and trading decisions. It is built around three interlocking concepts:

1. **Modular mixin system** — strategy behaviour (entry logic, exit logic, risk rules) is composed from small, reusable mixins rather than written as monolithic classes.
2. **Config-driven indicators** — all technical indicators are declared in JSON and instantiated by `IndicatorFactory`; mixin code never creates indicators directly.
3. **Composite and adaptive orchestration** — multiple single-timeframe strategies can be combined into composite strategies, run across multiple timeframes, or switched dynamically based on market regime.

The indicator layer is described in full in [UNIFIED_INDICATOR_ARCHITECTURE.md](UNIFIED_INDICATOR_ARCHITECTURE.md). This document focuses on the strategy layer that *uses* that infrastructure.

---

## Module Layout

```
src/strategy/
├── strategy_core.py              # BaseStrategy, StrategySignal, SignalAggregator, MarketRegimeDetector
├── custom_strategy.py            # CustomStrategy — concrete bt.Strategy subclass
├── multi_timeframe_engine.py     # TimeframeSyncer, MultiTimeframeStrategy
├── composite_strategy_manager.py # StrategyComposer, AdvancedStrategyFramework
├── advanced_backtrader_strategy.py  # AdvancedBacktraderStrategy (Backtrader integration)
├── mixins/
│   ├── base_entry_mixin.py       # BaseEntryMixin
│   └── base_exit_mixin.py        # BaseExitMixin
└── tests/
```

Config files live in `config/strategy/`:

```
config/strategy/
├── composite_strategies.json
├── multi_timeframe.json
├── dynamic_switching.json
└── portfolio_optimization.json
```

---

## BaseStrategy and the Mixin System

### Class Hierarchy

```
bt.Strategy  (Backtrader)
    └── BaseStrategy           # indicator registry, param helpers, shared state
            └── CustomStrategy # concrete class; assembles mixins from JSON config
                    ├── uses BaseEntryMixin  (should_enter)
                    └── uses BaseExitMixin   (should_exit)
```

`BaseStrategy` owns:
- `self.indicators` — the central registry of alias → indicator line (populated by `IndicatorFactory`).
- `get_indicator(alias)` — safe accessor used by all mixins.
- `get_param(name)` — parameter accessor; reads from the bot's JSON config.

### CustomStrategy

`CustomStrategy` is the primary concrete class used in backtesting and live trading. On `__init__` it:

1. Reads the JSON configuration (`entry_logic`, `exit_logic`, `risk_management` blocks).
2. Calls `IndicatorFactory.create_indicators(all_configs)` to populate `self.indicators`.
3. Attaches the appropriate entry and exit mixin instances.

The strategy loop (`next()`) delegates entirely to the mixins:

```python
def next(self):
    if not self.position:
        if self.entry_mixin.should_enter():
            self.buy(size=self.compute_position_size())
    else:
        if self.exit_mixin.should_exit():
            self.close()
```

### Logic Mixins

Mixins contain pure decision logic. They **never** create indicators; they only read them:

```python
class MyEntryMixin(BaseEntryMixin):
    def should_enter(self) -> bool:
        rsi = self.get_indicator('entry_rsi')
        ma  = self.get_indicator('trend_ma')
        return rsi[0] < self.get_param('rsi_oversold') and self.data.close[0] > ma[0]
```

Multiple mixins can share the same indicator alias without recalculation — the registry returns the same line object.

---

## Indicator Factory and Service Integration

> The full indicator architecture is documented in [UNIFIED_INDICATOR_ARCHITECTURE.md](UNIFIED_INDICATOR_ARCHITECTURE.md). This section summarises the integration points relevant to strategy authors.

### JSON Declaration

Indicators are declared in the `entry_logic` or `exit_logic` sections of a strategy's bot config:

```json
{
  "entry_logic": {
    "indicators": [
      {
        "type": "RSI",
        "params": { "timeperiod": 14 },
        "fields_mapping": { "rsi": "entry_rsi" }
      },
      {
        "type": "SMA",
        "params": { "timeperiod": 50 },
        "fields_mapping": { "sma": "trend_ma" }
      }
    ]
  }
}
```

### IndicatorFactory

`IndicatorFactory` maps `type` strings to `bt.talib` (TA-Lib) or custom implementations, validates parameters, selects the correct input series (Close, HLC, OHLCV), and returns `{alias: indicator_line}` mappings that are merged into `self.indicators`.

Supported backends:
- **TA-Lib** (`bt.talib` wrappers) — default; C-performance, vectorised in `runonce` mode.
- **pandas-ta** — via `PandasTAAdapter` for indicators not covered by TA-Lib.
- **Custom** — project-specific indicator classes registered in `src/indicators/registry.py`.

### UnifiedIndicatorService (Batch / Async)

For non-Backtrader contexts (screeners, alerts, dashboards) the higher-level `UnifiedIndicatorService` provides an async batch API:

```python
from src.indicators.service import UnifiedIndicatorService
from src.indicators.models import IndicatorRequest

service = UnifiedIndicatorService()
result  = await service.calculate(IndicatorRequest(
    ticker="AAPL",
    indicators=["rsi", "macd", "bbands"]
))
```

Strategies running inside Backtrader do **not** use this service directly — they use `IndicatorFactory` and the registry.

---

## Composite Strategy Framework

When a single strategy is insufficient, the composite layer combines multiple strategies into a unified signal.

### AdvancedStrategyFramework

The top-level orchestrator. Initialise once and drive all composite/multi-timeframe/dynamic-switching operations through it:

```python
from src.strategy.composite_strategy_manager import AdvancedStrategyFramework

framework = AdvancedStrategyFramework(config_path="config/strategy/")
framework.initialize_composite_strategies()
framework.initialize_multi_timeframe_strategies()
framework.initialize_dynamic_switching()

signal = framework.get_composite_signal("momentum_trend_composite", data_feeds)
```

### Signal Aggregation Methods

`SignalAggregator` (in `strategy_core.py`) merges per-strategy `StrategySignal` objects:

| Method | Behaviour |
|---|---|
| `weighted_voting` | Weighted sum of signal confidences |
| `consensus_voting` | All contributing strategies must agree above a threshold |
| `majority_voting` | Simple majority of BUY/SELL signals |
| `weighted_average` | Average confidence across strategies |

Aggregation method and `consensus_threshold` are set per composite in `config/strategy/composite_strategies.json`.

### Composite Strategy Configuration

```json
{
  "composite_strategies": {
    "momentum_trend_composite": {
      "strategies": [
        { "name": "rsi_momentum",    "weight": 0.4, "timeframe": "1h" },
        { "name": "supertrend_trend","weight": 0.6, "timeframe": "4h" }
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

---

## Multi-Timeframe Strategies

`MultiTimeframeStrategy` (in `multi_timeframe_engine.py`) separates trend analysis, entry timing, and exit timing across different resolutions.

```
4H  ──► Trend direction (SuperTrend / EMA)
1H  ──► Entry signal    (RSI, volume)
15M ──► Exit signal     (fast RSI, ATR trailing stop)
```

`TimeframeSyncer` ensures the higher-timeframe bar is complete before the lower-timeframe signal is evaluated.

### Configuration (`config/strategy/multi_timeframe.json`)

```json
{
  "multi_timeframe_strategies": {
    "trend_following_mtf": {
      "timeframes": {
        "trend_timeframe": "4h",
        "entry_timeframe": "1h",
        "exit_timeframe": "15m"
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

---

## Dynamic Strategy Switching

`MarketRegimeDetector` (in `strategy_core.py`) classifies the current market state and `StrategyComposer` activates the appropriate strategy:

| Regime | Characteristics | Typical Strategy |
|---|---|---|
| Trending Volatile | Strong trend, high ATR | `momentum_trend_composite` |
| Trending Stable | Strong trend, low ATR | `trend_following_mtf` |
| Ranging Volatile | Weak trend, high ATR | `volatility_breakout` |
| Ranging Stable | Weak trend, low ATR | `mean_reversion_momentum` |

Switching configuration (`config/strategy/dynamic_switching.json`) controls `volatility_threshold`, `trend_strength_threshold`, `lookback_period`, and `minimum_regime_duration` to prevent thrashing.

Performance-based switching (`performance_based_switching`) monitors Sharpe ratio, drawdown, and win rate to trigger substitutions regardless of regime.

---

## Strategy Registration and Lifecycle

### Registration

Strategies are registered by name in the framework's internal registry at initialisation time. The `AdvancedStrategyFramework` loads all entries from the JSON config directories on startup; no code changes are needed to add a new strategy variant.

### Lifecycle

```
load JSON config
      │
      ▼
IndicatorFactory.create_indicators()   ← populates self.indicators
      │
      ▼
Mixin.__init__()                       ← mixins receive strategy reference
      │
      ▼
Strategy loop: next()
      │
      ├── entry_mixin.should_enter()   ← reads self.indicators via get_indicator()
      └── exit_mixin.should_exit()     ← reads self.indicators via get_indicator()
```

For composite strategies, `StrategyComposer` drives multiple `CustomStrategy` instances and aggregates their `StrategySignal` outputs before any order is placed.

### Backtrader Integration

```python
import backtrader as bt
from src.strategy.advanced_backtrader_strategy import AdvancedBacktraderStrategy

cerebro = bt.Cerebro()
cerebro.adddata(data_feed)
cerebro.addstrategy(
    AdvancedBacktraderStrategy,
    strategy_name='momentum_trend_composite',
    use_dynamic_switching=True,
    max_position_size=0.1,
    stop_loss_pct=0.02,
    take_profit_pct=0.04
)
results = cerebro.run()
```

---

## Risk Management

Risk parameters are attached at the strategy level (per-strategy JSON config) and at the composite level (`composite_strategies.json`):

| Parameter | Scope | Description |
|---|---|---|
| `max_position_size` | composite / strategy | Fraction of capital per trade |
| `stop_loss_pct` | composite / strategy | Fixed stop as % of entry price |
| `take_profit_pct` | composite / strategy | Fixed target as % of entry price |
| `var_limit` | portfolio | Max Value-at-Risk per day |
| `max_drawdown_limit` | portfolio | Circuit-breaker drawdown threshold |
| `correlation_threshold` | portfolio | Position sizing adjustment for correlated assets |

Position size is scaled by signal confidence when `use_dynamic_sizing` is enabled.

---

## Available Pre-Built Strategies

### Composite
| Name | Indicators | Suitable For |
|---|---|---|
| `momentum_trend_composite` | RSI + SuperTrend | Trending markets |
| `mean_reversion_momentum` | Bollinger Bands + MACD | Ranging markets |
| `volatility_breakout` | ATR + BB + SuperTrend | High-volatility breakouts |

### Multi-Timeframe
| Name | Timeframes | Notes |
|---|---|---|
| `trend_following_mtf` | 4H / 1H / 15M | Trend confirmation required |
| `breakout_mtf` | 4H / 1H / 15M | Volume confirmation required |
| `mean_reversion_mtf` | 4H / 1H / 30M | Only trades at statistical extremes |

### Dynamic Switching
| Name | Switching Trigger |
|---|---|
| `market_regime_adaptive` | Volatility + trend strength |
| `performance_based_switching` | Sharpe ratio, drawdown, win rate |
| `time_based_switching` | Market session (Asian / London / NY) |

---

## Extending the Framework

### Adding a New Strategy Variant

1. Create entry and exit mixin classes in `src/strategy/mixins/`.
2. Declare indicator requirements in the bot's JSON config (`entry_logic.indicators`).
3. Add the composite/MTF/switching config entry to the relevant JSON file in `config/strategy/`.
4. No Python changes needed in `AdvancedStrategyFramework` — it loads configs dynamically.

### Adding a New Indicator

See [UNIFIED_INDICATOR_ARCHITECTURE.md](UNIFIED_INDICATOR_ARCHITECTURE.md) for the full procedure. In summary:
1. Register in `src/indicators/registry.py`.
2. Implement in the appropriate adapter.
3. Add to the bot JSON config with a `fields_mapping` alias.

---

## Troubleshooting

| Symptom | Likely Cause | Resolution |
|---|---|---|
| No signals generated | Insufficient historical data for indicator warmup | Ensure data length > max indicator period |
| Frequent strategy switches | Regime detection thresholds too sensitive | Increase `minimum_regime_duration` and `lookback_period` |
| Poor backtest performance | Wrong regime-strategy pairing | Review `dynamic_switching.json` strategy assignments |
| `KeyError` in `get_indicator` | Missing alias in JSON config | Check `fields_mapping` keys match mixin alias strings |
| TA-Lib not found | Native library not installed | Follow TA-Lib installation in [deployment.md](deployment.md) |

Enable debug logging for detailed signal traces:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Inspect strategy state:

```python
summary = strategy.get_strategy_summary()
print(summary)
```

---

**Last Updated:** 2026-04-30

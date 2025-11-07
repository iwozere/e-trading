# **Specification: Simple TALib-Based Indicator Architecture**

## **1. Overview**

### **1.1 Goals**
- ✅ Simple, explicit indicator configuration in strategy config
- ✅ Fast execution using TALib's C-based implementation
- ✅ Same code path for optimization, backtesting, and live trading
- ✅ No UnifiedIndicatorService complexity in Backtrader flows
- ✅ Clean separation between indicator params and logic params

### **1.2 Non-Goals**
- ❌ Pre-computed batch indicator calculation
- ❌ Auto-detection of required indicators
- ❌ DataFrame-based indicator computation
- ❌ UnifiedIndicatorService integration in strategies

### **1.3 Architecture Principle**
**"Strategy config is the single source of truth. Strategy creates and owns all indicators. Mixins access indicators via strategy."**

---

## **2. Configuration Schema**

### **2.1 Strategy Configuration Format**

```json
{
  "strategy": {
    "type": "CustomStrategy",
    "parameters": {
      "entry_logic": {
        "name": "RSIBBEntryMixin",
        "indicators": [
          {
            "type": "RSI",
            "params": {
              "timeperiod": 22
            },
            "fields_mapping": {
              "rsi": "entry_rsi"
            }
          },
          {
            "type": "BBANDS",
            "params": {
              "timeperiod": 20,
              "nbdevup": 2.08,
              "nbdevdn": 2.08,
              "matype": 0
            },
            "fields_mapping": {
              "upperband": "entry_bb_upper",
              "middleband": "entry_bb_middle",
              "lowerband": "entry_bb_lower"
            }
          }
        ],
        "logic_params": {
          "oversold": 34,
          "use_bb_touch": true,
          "rsi_cross": false,
          "bb_reentry": false,
          "cooldown_bars": 0
        }
      },
      "exit_logic": {
        "name": "ATRExitMixin",
        "indicators": [
          {
            "type": "ATR",
            "params": {
              "timeperiod": 17
            },
            "fields_mapping": {
              "atr": "exit_atr"
            }
          }
        ],
        "logic_params": {
          "tp_multiplier": 2.9,
          "sl_multiplier": 2.9
        }
      },
      "position_size": 0.1
    }
  }
}
```

### **2.2 Indicator Configuration Structure**

Each indicator config has three parts:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `type` | string | TALib indicator name (uppercase) | `"RSI"`, `"BBANDS"`, `"ATR"` |
| `params` | object | TALib-specific parameters | `{"timeperiod": 14}` |
| `fields_mapping` | object | Map TALib output names to strategy-specific aliases | `{"rsi": "entry_rsi"}` |

### **2.3 Supported Indicators**

Initial implementation supports:

| Indicator | TALib Name | Output Fields | Common Params |
|-----------|------------|---------------|---------------|
| RSI | `RSI` | `rsi` | `timeperiod` |
| Bollinger Bands | `BBANDS` | `upperband`, `middleband`, `lowerband` | `timeperiod`, `nbdevup`, `nbdevdn`, `matype` |
| ATR | `ATR` | `atr` | `timeperiod` |
| MACD | `MACD` | `macd`, `macdsignal`, `macdhist` | `fastperiod`, `slowperiod`, `signalperiod` |
| SMA | `SMA` | `sma` | `timeperiod` |
| EMA | `EMA` | `ema` | `timeperiod` |
| Stochastic | `STOCH` | `slowk`, `slowd` | `fastk_period`, `slowk_period`, `slowd_period` |

**Extension mechanism**: Add new indicators by updating indicator factory mapping.

### **2.4 Field Mapping Examples**

**Single output indicator (RSI)**:
```json
{
  "type": "RSI",
  "params": {"timeperiod": 14},
  "fields_mapping": {
    "rsi": "my_rsi_14"
  }
}
```

**Multi-output indicator (BBANDS)**:
```json
{
  "type": "BBANDS",
  "params": {"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0},
  "fields_mapping": {
    "upperband": "bb20_upper",
    "middleband": "bb20_middle",
    "lowerband": "bb20_lower"
  }
}
```

**Using same indicator twice with different params**:
```json
{
  "entry_logic": {
    "indicators": [
      {
        "type": "RSI",
        "params": {"timeperiod": 14},
        "fields_mapping": {"rsi": "rsi_14"}
      },
      {
        "type": "RSI",
        "params": {"timeperiod": 22},
        "fields_mapping": {"rsi": "rsi_22"}
      }
    ]
  }
}
```

---

## **3. Component Architecture**

### **3.1 Component Overview**

```
┌─────────────────────────────────────────────────────────┐
│                   Strategy Config (JSON)                 │
│  - entry_logic.indicators                                │
│  - exit_logic.indicators                                 │
│  - logic_params                                          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  CustomStrategy                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  IndicatorFactory                                   │ │
│  │  - Creates bt.talib indicators                      │ │
│  │  - Maps outputs to aliases                          │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  self.indicators = {                                     │
│    'entry_rsi': <bt.talib.RSI>,                         │
│    'entry_bb_upper': <bt.talib.BBANDS.upperband>,       │
│    'exit_atr': <bt.talib.ATR>                           │
│  }                                                       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    Entry/Exit Mixins                     │
│                                                          │
│  def should_enter(self):                                │
│    rsi = self.get_indicator('entry_rsi')                │
│    bb_lower = self.get_indicator('entry_bb_lower')      │
│    return rsi <= self.params['oversold'] ...            │
└─────────────────────────────────────────────────────────┘
```

### **3.2 File Structure**

```
src/
├── strategy/
│   ├── indicator_factory.py          # NEW: Creates TALib indicators
│   ├── base_strategy.py               # MODIFY: Add indicator management
│   ├── custom_strategy.py             # MODIFY: Use IndicatorFactory
│   ├── entry/
│   │   ├── base_entry_mixin.py        # MODIFY: Add get_indicator()
│   │   ├── rsi_bb_entry_mixin.py      # MODIFY: Use new config format
│   │   └── ...
│   └── exit/
│       ├── base_exit_mixin.py         # MODIFY: Add get_indicator()
│       ├── atr_exit_mixin.py          # MODIFY: Use new config format
│       └── ...
├── backtester/
│   └── tests/
│       └── backtester_test_framework.py  # MODIFY: Pass config to strategy
└── indicators/
    ├── service.py                     # KEEP: For non-Backtrader uses
    └── adapters/
        ├── backtrader_adapter.py      # DELETE or DEPRECATE
        └── backtrader_wrappers.py     # DELETE or DEPRECATE
```

---

## **4. Detailed Component Specifications**

### **4.1 IndicatorFactory** (NEW)

**File**: `src/strategy/indicator_factory.py`

**Purpose**: Create TALib indicators from config and manage field mappings

**Interface**:
```python
class IndicatorFactory:
    """Factory for creating TALib indicators from configuration"""

    @staticmethod
    def create_indicators(
        data: bt.feeds.DataBase,
        indicator_configs: List[Dict[str, Any]]
    ) -> Dict[str, bt.Indicator]:
        """
        Create TALib indicators from configuration.

        Args:
            data: Backtrader data feed
            indicator_configs: List of indicator configurations

        Returns:
            Dictionary mapping field aliases to indicator line objects

        Example:
            configs = [
                {
                    "type": "RSI",
                    "params": {"timeperiod": 14},
                    "fields_mapping": {"rsi": "entry_rsi"}
                }
            ]

            indicators = IndicatorFactory.create_indicators(data, configs)
            # Returns: {'entry_rsi': <bt.talib.RSI indicator>}
        """

    @staticmethod
    def _create_single_indicator(
        data: bt.feeds.DataBase,
        ind_config: Dict[str, Any]
    ) -> Dict[str, bt.Indicator]:
        """Create a single indicator and return mapped outputs"""

    @staticmethod
    def validate_config(ind_config: Dict[str, Any]) -> None:
        """Validate indicator configuration"""
```

**Validation Rules**:
1. `type` must be a supported TALib indicator name
2. `params` must contain all required TALib parameters
3. `fields_mapping` keys must match TALib output field names
4. `fields_mapping` values (aliases) must be unique across all indicators

**Error Handling**:
- Raise `ValueError` for invalid/unsupported indicator types
- Raise `ValueError` for missing required parameters
- Raise `ValueError` for field mapping mismatches
- Raise `ValueError` for duplicate field aliases

---

### **4.2 BaseStrategy Modifications** (MODIFY)

**File**: `src/strategy/base_strategy.py`

**New Attributes**:
```python
class BaseBacktraderStrategy(bt.Strategy):
    def __init__(self):
        super().__init__()
        self.indicators = {}  # Dict[str, bt.Indicator]
        self.indicator_config = None
```

**New Methods**:
```python
def _create_indicators_from_config(self, strategy_config: Dict[str, Any]):
    """
    Create all indicators from strategy configuration.

    Combines entry_logic.indicators + exit_logic.indicators,
    creates them via IndicatorFactory, and stores in self.indicators.
    """

def get_indicator(self, alias: str) -> bt.Indicator:
    """
    Get indicator by alias.

    Args:
        alias: Field alias from fields_mapping

    Returns:
        Backtrader indicator line object

    Raises:
        KeyError: If indicator not found
    """

def _validate_indicators_ready(self) -> bool:
    """Check if all indicators have valid values (not NaN)"""
```

---

### **4.3 CustomStrategy Modifications** (MODIFY)

**File**: `src/strategy/custom_strategy.py`

**Changes**:
```python
class CustomStrategy(BaseBacktraderStrategy):
    params = (
        ('strategy_config', {}),  # Full strategy configuration
    )

    def __init__(self):
        super().__init__()

        # Create indicators from config FIRST
        self._create_indicators_from_config(self.params.strategy_config)

        # Then initialize mixins (they can now access indicators)
        self._initialize_strategy()

    def _initialize_strategy(self):
        """Initialize mixins with new config format"""
        strategy_params = self.params.strategy_config.get('parameters', {})

        # Create entry mixin
        entry_config = strategy_params.get('entry_logic', {})
        self.entry_mixin = self._create_entry_mixin(entry_config)

        # Create exit mixin
        exit_config = strategy_params.get('exit_logic', {})
        self.exit_mixin = self._create_exit_mixin(exit_config)

    def _create_entry_mixin(self, entry_config: dict):
        """Create entry mixin from config"""
        from src.strategy.entry.entry_mixin_factory import ENTRY_MIXIN_REGISTRY

        mixin_name = entry_config['name']
        logic_params = entry_config.get('logic_params', {})

        mixin_class = ENTRY_MIXIN_REGISTRY[mixin_name]
        return mixin_class(params=logic_params)
```

---

### **4.4 BaseMixin Modifications** (MODIFY)

**File**: `src/strategy/entry/base_entry_mixin.py` and `src/strategy/exit/base_exit_mixin.py`

**New Methods**:
```python
class BaseEntryMixin:
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        self.strategy = None  # Set later

    def get_indicator(self, alias: str) -> Any:
        """
        Get indicator value by alias.

        Args:
            alias: Field alias from fields_mapping

        Returns:
            Current indicator value (for current bar)

        Example:
            rsi_value = self.get_indicator('entry_rsi')
            # Returns: self.strategy.indicators['entry_rsi'][0]
        """
        if self.strategy is None:
            raise RuntimeError("Mixin not attached to strategy")

        if alias not in self.strategy.indicators:
            raise KeyError(f"Indicator '{alias}' not found in strategy")

        indicator = self.strategy.indicators[alias]
        return indicator[0]  # Current bar value

    def get_indicator_prev(self, alias: str, offset: int = 1) -> Any:
        """Get previous indicator value"""
        indicator = self.strategy.indicators[alias]
        return indicator[-offset]

    def _init_indicators(self):
        """
        Verify required indicators exist.

        Subclasses should override to check for their specific indicators.
        No longer creates indicators - just validates they exist.
        """
        pass
```

---

### **4.5 RSIBBEntryMixin Updates** (MODIFY)

**File**: `src/strategy/entry/rsi_bb_entry_mixin.py`

**New Structure**:
```python
class RSIBBEntryMixin(BaseEntryMixin):
    """
    Entry mixin using RSI and Bollinger Bands.

    Required indicators (configured in strategy config):
    - RSI with alias containing 'rsi'
    - BBANDS with aliases containing 'bb_upper', 'bb_middle', 'bb_lower'

    Logic params:
    - oversold: RSI oversold threshold
    - use_bb_touch: Whether price must touch BB lower band
    - rsi_cross: Require RSI to cross back above oversold
    - bb_reentry: Require price to bounce back above BB lower
    - cooldown_bars: Minimum bars between entries
    """

    REQUIRED_INDICATOR_PATTERNS = ['rsi', 'bb_lower']

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)
        self.last_entry_bar = None

        # Aliases will be auto-detected from strategy.indicators
        self.rsi_alias = None
        self.bb_upper_alias = None
        self.bb_middle_alias = None
        self.bb_lower_alias = None

    def _init_indicators(self):
        """Detect and validate indicator aliases"""
        # Auto-detect aliases by pattern matching
        for alias in self.strategy.indicators.keys():
            if 'rsi' in alias.lower():
                self.rsi_alias = alias
            elif 'bb_upper' in alias.lower():
                self.bb_upper_alias = alias
            elif 'bb_middle' in alias.lower():
                self.bb_middle_alias = alias
            elif 'bb_lower' in alias.lower():
                self.bb_lower_alias = alias

        # Validate required indicators exist
        if not self.rsi_alias:
            raise ValueError("RSIBBEntryMixin requires RSI indicator")
        if not self.bb_lower_alias:
            raise ValueError("RSIBBEntryMixin requires BBANDS indicator")

        logger.debug(f"Using indicators: {self.rsi_alias}, {self.bb_lower_alias}")

    def should_enter(self) -> bool:
        """Check entry conditions"""
        # Cooldown check
        if self.params.get('cooldown_bars', 0) > 0:
            current_bar = len(self.strategy.data)
            if (self.last_entry_bar and
                current_bar - self.last_entry_bar < self.params['cooldown_bars']):
                return False

        # Get indicator values
        rsi_value = self.get_indicator(self.rsi_alias)
        bb_lower = self.get_indicator(self.bb_lower_alias)
        current_price = self.strategy.data.close[0]

        # RSI condition
        if self.params.get('rsi_cross', False):
            rsi_prev = self.get_indicator_prev(self.rsi_alias)
            rsi_condition = (rsi_prev <= self.params['oversold'] and
                           rsi_value > self.params['oversold'])
        else:
            rsi_condition = rsi_value <= self.params['oversold']

        # BB condition
        if self.params.get('use_bb_touch', True):
            bb_condition = current_price <= bb_lower
        else:
            bb_condition = current_price < bb_lower

        should_enter = rsi_condition and bb_condition

        if should_enter:
            self.last_entry_bar = len(self.strategy.data)

        return should_enter
```

---

## **5. Execution Flow**

### **5.1 Initialization Sequence**

```
1. Backtrader creates strategy instance
   ↓
2. CustomStrategy.__init__()
   ↓
3. _create_indicators_from_config()
   ├── Extract entry_logic.indicators
   ├── Extract exit_logic.indicators
   ├── Validate configs
   ├── Call IndicatorFactory.create_indicators()
   │   ├── For each indicator config:
   │   │   ├── Create bt.talib indicator
   │   │   ├── Map output fields to aliases
   │   │   └── Store in indicators dict
   │   └── Return complete indicators dict
   └── Store in self.indicators
   ↓
4. _initialize_strategy()
   ├── Create entry mixin with logic_params
   ├── Attach strategy reference to mixin
   ├── Call mixin._init_indicators() (validates)
   ├── Create exit mixin with logic_params
   ├── Attach strategy reference to mixin
   └── Call mixin._init_indicators() (validates)
   ↓
5. Ready for backtesting/trading
```

### **5.2 Runtime Execution (Per Bar)**

```
1. Backtrader calls strategy.next()
   ↓
2. TALib indicators auto-update (Backtrader handles this)
   ↓
3. Strategy checks entry conditions
   ├── Calls entry_mixin.should_enter()
   ├── Mixin calls self.get_indicator('entry_rsi')
   ├── Returns indicator[0] (current value)
   └── Evaluates logic, returns True/False
   ↓
4. If in position, check exit conditions
   ├── Calls exit_mixin.should_exit()
   ├── Mixin calls self.get_indicator('exit_atr')
   └── Evaluates logic, returns (bool, reason)
   ↓
5. Execute trades based on signals
```

---

## **6. Migration Path**

### **6.1 Backward Compatibility**

**Strategy**: Support both old and new config formats temporarily

```python
def _detect_config_version(config):
    """Detect if config uses old or new format"""
    entry_logic = config.get('parameters', {}).get('entry_logic', {})

    if 'indicators' in entry_logic:
        return 'v2'  # New format
    else:
        return 'v1'  # Old format

def _create_indicators_from_config(self, strategy_config):
    version = self._detect_config_version(strategy_config)

    if version == 'v2':
        # New TALib-based approach
        self._create_talib_indicators(strategy_config)
    else:
        # Old UnifiedIndicatorService approach (deprecated)
        logger.warning("Using deprecated indicator format. Please migrate to v2.")
        self._create_unified_indicators(strategy_config)
```

### **6.2 Migration Steps**

1. **Phase 1**: Implement new architecture alongside old (both work)
2. **Phase 2**: Update all test configs to new format
3. **Phase 3**: Update all mixins to use new format
4. **Phase 4**: Deprecate old format (log warnings)
5. **Phase 5**: Remove old UnifiedIndicatorService integration

### **6.3 Config Migration Tool**

Provide utility to convert old configs to new:

```python
def migrate_config_v1_to_v2(old_config):
    """Convert old config format to new indicator-explicit format"""
    # Parse old params, generate indicator configs
    # Return new format
```

---

## **7. Testing Strategy**

### **7.1 Unit Tests**

- `test_indicator_factory.py`: Test indicator creation
- `test_base_mixin.py`: Test get_indicator() helper
- `test_custom_strategy.py`: Test indicator initialization
- `test_config_validation.py`: Test config validation

### **7.2 Integration Tests**

- `test_rsi_bb_entry.py`: Full entry mixin with TALib indicators
- `test_atr_exit.py`: Full exit mixin with TALib indicators
- `test_optimization.py`: Multi-parameter optimization run

### **7.3 Performance Benchmarks**

```python
def benchmark_talib_incremental():
    """Benchmark TALib incremental vs old approach"""
    # Run 30K bars
    # Measure time
    # Compare to old UnifiedIndicatorService approach
```

**Success Criteria**: Complete 30K bar backtest in < 60 seconds

---

## **8. Documentation**

### **8.1 User Documentation**

- Configuration guide with examples
- Supported indicators reference
- Migration guide from v1 to v2

### **8.2 Developer Documentation**

- Architecture overview
- Adding new indicators
- Creating custom mixins
- Debugging indicator issues

---

## **9. Future Enhancements**

### **9.1 Phase 2 Features** (Post-MVP)

1. **Custom Indicators**: Support user-defined indicator functions
2. **Indicator Caching**: Cache indicator calculations for optimization
3. **Pre-computation Mode**: Optional batch calculation for backtesting
4. **Indicator Validation**: Runtime checks for indicator sanity
5. **Performance Monitoring**: Track indicator calculation overhead

### **9.2 Potential Optimizations** (If Needed)

1. **Lazy Indicator Creation**: Only create indicators when first accessed
2. **Shared Indicators**: Detect when entry/exit use same indicator (with same params)
3. **Batch Mode**: Pre-compute for backtesting if TALib incremental proves too slow

---

## **10. Success Criteria**

✅ **Functional**:
- All existing strategies work with new config format
- Backtests complete successfully
- Optimization runs complete successfully
- Live trading works (when implemented)

✅ **Performance**:
- 30K bar backtest completes in < 60 seconds
- Minimal memory overhead
- No performance regression vs old approach

✅ **Code Quality**:
- Clean, understandable code
- Comprehensive test coverage (>80%)
- Clear documentation

✅ **Maintainability**:
- Easy to add new indicators
- Easy to debug issues
- Clear error messages

---

## **11. Open Questions**

1. **Q**: Should we support indicators that use multiple data inputs (e.g., custom formulas)?
   **A**: Not in MVP, Phase 2 feature

2. **Q**: How to handle indicators that require forward-looking data?
   **A**: Not supported (would break backtest integrity)

3. **Q**: Should mixins be able to create their own indicators dynamically?
   **A**: No, all indicators must be in config (explicit > implicit)

4. **Q**: Error handling for TALib calculation failures (e.g., insufficient data)?
   **A**: Return NaN, strategy checks via `_validate_indicators_ready()`

---

**Status**: Specification complete and ready for implementation.

**Next Steps**: Implement Phase 1 (IndicatorFactory) and validate with simple backtest.

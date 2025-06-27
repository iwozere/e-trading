# Risk Management Module

This module provides a comprehensive risk management framework for trading systems, covering pre-trade, real-time, and post-trade risk controls.

## Directory Structure

```
risk/
├── pre_trade/
│   ├── position_sizing.py       # Kelly Criterion, Fixed Fractional
│   ├── exposure_limits.py       # Position and portfolio limits
│   └── correlation_check.py     # Correlation-based limits
├── real_time/
│   ├── stop_loss_manager.py     # Dynamic stop-loss adjustment
│   ├── drawdown_control.py      # Circuit breakers
│   └── volatility_scaling.py    # Vol-adjusted position sizing
└── post_trade/
    ├── pnl_attribution.py      # P&L analysis
    ├── trade_analysis.py        # Trade quality metrics
    └── risk_reporting.py        # Risk dashboard
```

## Components

### Pre-Trade Risk
- **position_sizing.py**: Implements position sizing algorithms, including Kelly Criterion and Fixed Fractional methods.
- **exposure_limits.py**: Enforces position and portfolio exposure limits to prevent over-concentration.
- **correlation_check.py**: Limits positions in highly correlated assets to avoid systemic risk.

### Real-Time Risk
- **stop_loss_manager.py**: Dynamically adjusts stop-loss levels (e.g., trailing stops) to protect profits and limit losses.
- **drawdown_control.py**: Implements circuit breaker logic to halt trading or reduce risk when drawdowns exceed thresholds.
- **volatility_scaling.py**: Adjusts position sizes based on realized or forecast volatility to maintain target risk levels.

### Post-Trade Risk
- **pnl_attribution.py**: Attributes profit and loss to different sources (e.g., by symbol, strategy) for analysis.
- **trade_analysis.py**: Calculates trade quality metrics such as win rate, average win/loss, and expectancy.
- **risk_reporting.py**: Generates risk reports and dashboards for monitoring and compliance.

## Usage
Each module provides clear function interfaces and can be integrated into trading workflows for robust risk management at every stage of the trade lifecycle.

## Integration Guidance

To integrate the risk management module into your trading system:

1. **Import the relevant modules** in your trading scripts:

```python
from src.risk.pre_trade import position_sizing, exposure_limits, correlation_check
from src.risk.real_time import stop_loss_manager, drawdown_control, volatility_scaling
from src.risk.post_trade import pnl_attribution, trade_analysis, risk_reporting
```

2. **Pre-Trade Checks:**
   - Use `position_sizing.kelly_criterion` or `fixed_fractional` to determine position size before placing a trade.
   - Check position and portfolio limits with `exposure_limits`.
   - Use `correlation_check` to avoid overexposure to correlated assets.

3. **Real-Time Risk Controls:**
   - Adjust stop-losses dynamically with `stop_loss_manager.dynamic_stop_loss`.
   - Monitor drawdowns using `drawdown_control.check_drawdown` and trigger circuit breakers if needed.
   - Scale positions based on volatility using `volatility_scaling.volatility_scaled_position`.

4. **Post-Trade Analysis:**
   - Attribute P&L with `pnl_attribution.pnl_attribution`.
   - Analyze trade quality using `trade_analysis.trade_quality_metrics`.
   - Generate risk reports with `risk_reporting.generate_risk_report`.

See `examples/risk_management_example.py` for a complete usage demonstration. 
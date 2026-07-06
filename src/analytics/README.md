# Analytics Module

## Overview
The `analytics` module is responsible for advanced performance tracking and statistical analysis of trading strategies. It consumes trade history, parses PnL data, and computes risk adjusted metrics like Sharpe, Sortino, and Calmar ratios, as well as Value-at-Risk (VaR), Conditional Value-at-Risk (CVaR), and composite performance ranking scores.

## Features
- Comprehensive performance metric calculations (Win Rate, Profit Factor, Expectancy, Payoff Ratio)
- Risk profiling (Sharpe, Sortino, Calmar ratios, Max Drawdown, Recovery Factor)
- Tail risk estimation (VaR 95%, CVaR 95%)
- Automated reporting in PDF, Excel (XLSX), and JSON formats
- Multi-strategy comparison and composite scoring/ranking

## Quick Start
Example code showing how to use the advanced analytics module:

```python
from src.analytics.advanced_analytics import AdvancedAnalytics

# Initialize the analytics engine
analytics = AdvancedAnalytics(risk_free_rate=0.02)

# Load trade historical data
trades_data = [
    {
        "entry_time": "2026-07-01T10:00:00Z",
        "exit_time": "2026-07-01T11:30:00Z",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "entry_price": 50000.0,
        "exit_price": 51000.0,
        "quantity": 1.0,
        "pnl": 1000.0,
        "commission": 10.0,
        "net_pnl": 990.0,
        "exit_reason": "take_profit",
    }
]

analytics.add_trades(trades_data)

# Compute performance metrics
metrics = analytics.calculate_metrics()
print(f"Total return: ${metrics.total_return}")
print(f"Win rate: {metrics.win_rate}%")

# Generate structured reports
report_path = analytics.generate_performance_report(output_dir="reports")
print(f"Report saved to: {report_path}")
```

## Integration
This module integrates with:
- `src.strategy` - To calculate historical simulation performance
- `src.backtester` - To analyze test results and optimize strategy parameters
- `src.model` - Shared data classes (`Trade`, `PerformanceMetrics`)

## Configuration
The analytics class accepts a `risk_free_rate` float setting. External output format generation depends on the presence of optional third-party packages: `reportlab` for PDFs, and `openpyxl` for Excel generation.

## Related Documentation
- [Requirements](docs/Requirements.md) - Technical requirements
- [Design](docs/Design.md) - Architecture and design
- [Tasks](docs/Tasks.md) - Implementation roadmap

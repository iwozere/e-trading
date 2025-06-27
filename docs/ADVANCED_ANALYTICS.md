# Advanced Analytics & Reporting System

## Overview

The Advanced Analytics & Reporting System provides comprehensive analysis of trading strategies with advanced metrics, risk analysis, Monte Carlo simulations, and automated reporting capabilities.

## Key Features

### 📊 Advanced Performance Metrics
- **Risk-adjusted returns**: Sharpe, Sortino, Calmar ratios
- **Risk metrics**: Value at Risk (VaR), Conditional VaR (CVaR)
- **Trade analysis**: Win rate, profit factor, average win/loss
- **Drawdown analysis**: Maximum drawdown, recovery factor
- **Kelly Criterion**: Optimal position sizing calculation

### 🎲 Monte Carlo Simulations
- **Future performance estimation**: 10,000+ simulations
- **Risk assessment**: Probability of profit/loss scenarios
- **Percentile analysis**: 10th, 25th, 50th, 75th, 90th percentiles
- **Confidence intervals**: Statistical significance testing

### 🏆 Strategy Comparison & Ranking
- **Multi-strategy analysis**: Compare multiple strategies
- **Performance ranking**: Composite scoring system
- **Benchmark comparison**: Against market indices
- **Strategy optimization**: Parameter sensitivity analysis

### 📋 Automated Reporting
- **Multiple formats**: PDF, Excel, JSON reports
- **Visual charts**: Performance graphs and risk diagrams
- **Executive summary**: Key metrics and recommendations
- **Detailed analysis**: Comprehensive breakdown of results

## Usage

### Basic Analytics

```python
from src.analytics.advanced_analytics import AdvancedAnalytics

# Create analytics instance
analytics = AdvancedAnalytics(risk_free_rate=0.02)

# Add trade data
trades_data = [
    {
        'entry_time': '2024-01-01T10:00:00',
        'exit_time': '2024-01-01T12:00:00',
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'entry_price': 45000.0,
        'exit_price': 45500.0,
        'quantity': 0.1,
        'pnl': 50.0,
        'commission': 0.5,
        'net_pnl': 49.5,
        'exit_reason': 'TP'
    }
    # ... more trades
]

analytics.add_trades(trades_data)

# Calculate comprehensive metrics
metrics = analytics.calculate_metrics()

print(f"Win Rate: {metrics.win_rate:.2f}%")
print(f"Profit Factor: {metrics.profit_factor:.2f}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
```

### Monte Carlo Simulation

```python
# Run Monte Carlo simulation
simulation_results = analytics.run_monte_carlo_simulation(
    n_simulations=10000,
    n_trades=100
)

print(f"Probability of Profit: {simulation_results['prob_profit']:.1f}%")
print(f"Expected Return: ${simulation_results['mean_return']:.2f}")
print(f"VaR (95%): ${simulation_results['var_95']:.2f}")
```

### Strategy Comparison

```python
from src.analytics.advanced_analytics import StrategyComparator

# Create comparator
comparator = StrategyComparator()

# Add multiple strategies
comparator.add_strategy("Strategy A", analytics_a)
comparator.add_strategy("Strategy B", analytics_b)
comparator.add_strategy("Strategy C", analytics_c)

# Compare strategies
comparison_df = comparator.compare_strategies()
print(comparison_df)

# Rank strategies
rankings = comparator.rank_strategies()
for strategy, rank in rankings.items():
    print(f"{rank}. {strategy}")
```

### Automated Reporting

```python
# Generate comprehensive report
report_path = analytics.generate_performance_report("reports")

# Report includes:
# - Performance summary
# - Risk analysis
# - Trade breakdown
# - Recommendations
# - Visual charts
```

## Performance Metrics

### Basic Metrics

| Metric | Description | Target |
|--------|-------------|---------|
| **Win Rate** | Percentage of profitable trades | >50% |
| **Profit Factor** | Gross profit / Gross loss | >1.5 |
| **Total Return** | Net profit/loss | Positive |
| **Total Trades** | Number of completed trades | >30 |

### Risk-Adjusted Metrics

| Metric | Description | Target |
|--------|-------------|---------|
| **Sharpe Ratio** | Risk-adjusted return | >1.0 |
| **Sortino Ratio** | Downside risk-adjusted return | >1.0 |
| **Calmar Ratio** | Return / Max drawdown | >1.0 |
| **Recovery Factor** | Total return / Max drawdown | >1.0 |

### Risk Metrics

| Metric | Description | Target |
|--------|-------------|---------|
| **Max Drawdown** | Largest peak-to-trough decline | <20% |
| **VaR (95%)** | Value at Risk (95% confidence) | <$100 |
| **CVaR (95%)** | Conditional Value at Risk | <$150 |
| **Kelly Criterion** | Optimal position size | 0.1-0.3 |

### Advanced Metrics

| Metric | Description | Target |
|--------|-------------|---------|
| **Expectancy** | Expected profit per trade | Positive |
| **Payoff Ratio** | Avg win / Avg loss | >1.0 |
| **Consecutive Wins** | Max consecutive winning trades | >5 |
| **Consecutive Losses** | Max consecutive losing trades | <5 |

## Risk Analysis

### Value at Risk (VaR)
- **Definition**: Maximum expected loss over a given time period
- **Calculation**: 95th percentile of trade returns
- **Interpretation**: 95% confidence that losses won't exceed VaR

### Conditional Value at Risk (CVaR)
- **Definition**: Expected loss given that VaR is exceeded
- **Calculation**: Average of returns below VaR threshold
- **Interpretation**: Expected loss in worst 5% of scenarios

### Maximum Drawdown
- **Definition**: Largest peak-to-trough decline in portfolio value
- **Calculation**: Maximum decline from any peak
- **Interpretation**: Worst historical decline experienced

### Recovery Factor
- **Definition**: Total return divided by maximum drawdown
- **Calculation**: Total return / |Max drawdown|
- **Interpretation**: How quickly strategy recovers from losses

## Monte Carlo Simulation

### Purpose
Monte Carlo simulations estimate future performance by running thousands of scenarios based on historical trade characteristics.

### Process
1. **Extract historical parameters**: Mean return, standard deviation
2. **Generate random scenarios**: 10,000+ simulations
3. **Calculate outcomes**: Cumulative returns for each scenario
4. **Analyze results**: Percentiles, probabilities, confidence intervals

### Output
- **Probability of profit**: % of scenarios with positive returns
- **Expected return**: Mean return across all scenarios
- **Risk metrics**: VaR, CVaR at different confidence levels
- **Percentiles**: 10th, 25th, 50th, 75th, 90th percentiles

### Example Results
```python
{
    "mean_return": 1250.50,
    "std_return": 450.25,
    "var_95": -850.00,
    "cvar_95": -1200.00,
    "prob_profit": 78.5,
    "percentiles": {
        "10": -500.00,
        "25": 200.00,
        "50": 1200.00,
        "75": 2200.00,
        "90": 3000.00
    }
}
```

## Strategy Comparison

### Composite Scoring
Strategies are ranked using a weighted composite score:

```python
score = (
    win_rate * 0.2 +
    min(profit_factor, 5.0) * 20 * 0.2 +
    sharpe_ratio * 10 * 0.2 +
    (100 - max_drawdown_pct) * 0.2 +
    calmar_ratio * 10 * 0.2
)
```

### Comparison Table
| Strategy | Trades | Win Rate | Profit Factor | Sharpe | Max DD | Rank |
|----------|--------|----------|---------------|--------|--------|------|
| Strategy A | 150 | 65% | 2.1 | 1.8 | 12% | 1 |
| Strategy B | 120 | 55% | 1.8 | 1.5 | 15% | 2 |
| Strategy C | 200 | 45% | 1.6 | 1.2 | 18% | 3 |

## Automated Reporting

### Report Formats

#### PDF Report
- **Professional layout**: Clean, business-ready format
- **Executive summary**: Key metrics and highlights
- **Detailed analysis**: Comprehensive breakdown
- **Visual elements**: Charts and graphs

#### Excel Report
- **Interactive data**: Sortable tables and filters
- **Color coding**: Green/red status indicators
- **Charts**: Performance and risk visualizations
- **Multiple sheets**: Summary, details, charts

#### JSON Report
- **Machine readable**: API-friendly format
- **Complete data**: All metrics and analysis
- **Structured format**: Easy to parse and process
- **Metadata**: Report generation info

### Report Contents

#### Performance Summary
- Total trades and win rate
- Profit factor and total return
- Risk-adjusted metrics
- Key performance indicators

#### Risk Analysis
- Maximum drawdown analysis
- VaR and CVaR calculations
- Risk assessment and recommendations
- Historical risk patterns

#### Trade Analysis
- Win/loss distribution
- Average trade characteristics
- Consecutive trade patterns
- Trade duration analysis

#### Recommendations
- Performance improvement suggestions
- Risk management recommendations
- Strategy optimization tips
- Action items for traders

## Integration with Existing System

### Optimization Results
```python
# Load optimization results
with open('results/optimization_result.json', 'r') as f:
    result = json.load(f)

# Extract trades
trades_data = result['trades']

# Analyze with advanced analytics
analytics = AdvancedAnalytics()
analytics.add_trades(trades_data)
metrics = analytics.calculate_metrics()
```

### Database Integration
```python
# Load trades from database
from src.data.trade_repository import TradeRepository

repo = TradeRepository()
trades = repo.get_trades_by_bot_id("bot_001")

# Convert to analytics format
trades_data = [trade.to_dict() for trade in trades]
analytics.add_trades(trades_data)
```

### Web Interface Integration
```python
# API endpoint for analytics
@app.route('/api/analytics/<bot_id>')
def get_analytics(bot_id):
    analytics = AdvancedAnalytics()
    trades = get_trades_from_db(bot_id)
    analytics.add_trades(trades)
    metrics = analytics.calculate_metrics()
    return jsonify(metrics.__dict__)
```

## Best Practices

### Data Quality
- **Sufficient sample size**: Minimum 30 trades for reliable analysis
- **Clean data**: Remove outliers and invalid trades
- **Consistent format**: Standardize trade data structure
- **Time accuracy**: Precise entry/exit timestamps

### Interpretation
- **Multiple metrics**: Don't rely on single metric
- **Context matters**: Consider market conditions
- **Statistical significance**: Large sample sizes preferred
- **Risk vs return**: Balance performance with risk

### Reporting
- **Regular updates**: Generate reports weekly/monthly
- **Trend analysis**: Track performance over time
- **Benchmark comparison**: Compare against market indices
- **Actionable insights**: Provide specific recommendations

## Troubleshooting

### Common Issues

#### Insufficient Data
```python
# Error: "Insufficient trade data for simulation"
# Solution: Need at least 10 trades for Monte Carlo
if len(trades) < 10:
    print("Need more trade data for reliable analysis")
```

#### Missing Dependencies
```python
# Error: "REPORTLAB_AVAILABLE = False"
# Solution: Install reportlab for PDF reports
pip install reportlab

# Error: "OPENPYXL_AVAILABLE = False"
# Solution: Install openpyxl for Excel reports
pip install openpyxl
```

#### Performance Issues
```python
# Large datasets: Use sampling for Monte Carlo
simulation_results = analytics.run_monte_carlo_simulation(
    n_simulations=5000,  # Reduce for faster processing
    n_trades=50
)
```

### Performance Optimization
- **Batch processing**: Process multiple strategies together
- **Caching**: Cache calculation results
- **Sampling**: Use representative samples for large datasets
- **Parallel processing**: Run simulations in parallel

## Future Enhancements

### Planned Features
- **Machine learning integration**: ML-based performance prediction
- **Real-time analytics**: Live performance monitoring
- **Portfolio optimization**: Multi-asset portfolio analysis
- **Advanced visualizations**: Interactive charts and dashboards
- **Custom metrics**: User-defined performance indicators

### Extensibility
The system is designed to be easily extensible:
- **Custom analyzers**: Add new performance metrics
- **New report formats**: Support additional output formats
- **Integration APIs**: Connect with external systems
- **Plugin architecture**: Modular component system

## Conclusion

The Advanced Analytics & Reporting System provides comprehensive analysis capabilities for trading strategies, enabling data-driven decision making and performance optimization. With advanced metrics, risk analysis, and automated reporting, traders can gain deep insights into their strategy performance and make informed improvements. 
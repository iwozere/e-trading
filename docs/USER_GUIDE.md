# Crypto Trading Platform User Guide

This guide will help you get started with backtesting, optimizing, and running live trading using this platform.

---

## 1. Environment Setup

### a. Clone the Repository
```bash
# Clone the repo (replace with your repo URL)
git clone <your-repo-url>
cd crypto-trading
```

### b. Create and Activate a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### c. Install Dependencies
```bash
pip install -r requirements.txt
```

If you plan to use live trading with specific brokers, also install:
```bash
pip install cbpro ib_insync python-binance
```

---

## 2. Data Preparation

Before running any strategies, you need to prepare your data:

### a. Data Format
Place your CSV data files in the `data/` directory. The expected format is:
- `timestamp`: Unix timestamp or datetime
- `open`, `high`, `low`, `close`: OHLC prices
- `volume`: Trading volume

### b. Data Naming Convention
Use the following naming convention for your data files:
```
{symbol}_{interval}_{start_date}_{end_date}.csv
```
Example: `BTCUSDT_1h_20240101_20241231.csv`

---

## 3. Running Optimizations

The platform uses a sophisticated optimization system with configurable entry and exit strategies.

### a. Configure Optimization Settings
Edit `config/optimizer/optimizer.json` to customize optimization parameters:
- `n_trials`: Number of optimization trials (default: 100)
- `initial_capital`: Starting capital for backtests
- `commission`: Trading commission rate
- `position_size`: Position sizing percentage

### b. Configure Entry/Exit Strategies
- Entry strategies: `config/optimizer/entry/` directory
- Exit strategies: `config/optimizer/exit/` directory
- Each strategy has its own JSON configuration file

### c. Run Optimization
```bash
python src/optimizer/run_optimizer.py
```

This will:
- Load data from the `data/` directory
- Test different parameter combinations for your strategies
- Save optimization results as JSON files in the `results/` directory
- Generate performance metrics and trade analysis

### d. Optimization Results
Results are saved as JSON files with the naming pattern:
```
{symbol}_{interval}_{start_date}_{end_date}_{entry_strategy}_{exit_strategy}_{timestamp}.json
```

---

## 4. Generating Plots

After optimization, generate visual plots from the results:

### a. Run Plotter
```bash
python src/plotter/run_plotter.py
```

This will:
- Read all JSON result files from the `results/` directory
- Load corresponding price data from the `data/` directory
- Generate PNG plots showing:
  - Price charts with indicators
  - Trade entry/exit points
  - Equity curves
  - Performance metrics

### b. Plot Configuration
The plotter automatically detects which indicators to show based on the strategy mixins used in optimization. Configuration is handled in `config/plotter/mixin_indicators.json`.

### c. Output
Plots are saved as PNG files in the `results/` directory with the same naming convention as the JSON files.

---

## 5. Data Processing and Analysis

For easier analysis and Excel import:

### a. Generate Summary CSV
```bash
python src/optimizer/run_json2csv.py
```

This will:
- Process all optimization JSON files in the `results/` directory
- Extract key metrics and parameters
- Generate a summary CSV file: `results/optimization_results_summary.csv`

### b. CSV Contents
The summary CSV includes:
- Symbol, interval, and date range
- Strategy type and parameters
- Performance metrics (win rate, profit factor, Sharpe ratio, etc.)
- Best parameters found during optimization

### c. Excel Analysis
Import the CSV into Excel for:
- Comparative analysis across different strategies
- Parameter sensitivity analysis
- Performance ranking and filtering

---

## 6. Live Trading

Live trading is managed by trading bots using configuration files.

### a. Configure Trading Bot
Create a configuration file in `config/trading/` (see `rsi_bb_volume1.json` for example):

```json
{
    "description": "Strategy description",
    "type": "binance_paper",
    "bot_type": "strategy_name",
    "strategy_type": "strategy_type",
    "trading_pair": "BTCUSDT",
    "initial_balance": 1000.0,
    "strategy_params": {
        "param1": 14,
        "param2": 20
    }
}
```

### b. Run Trading Bot
```bash
python src/trading/run_bot.py <config_file.json>
```

Example:
```bash
python src/trading/run_bot.py rsi_bb_volume1.json
```

### c. Available Brokers
- **Binance Paper**: `binance_paper` (for testing)
- **Binance Live**: `binance` (requires API keys)
- **IBKR**: `ibkr` (requires IBKR connection)

### d. API Key Configuration
For live trading, place your API keys in `config/donotshare/`:
- Never commit API keys to version control
- Use paper trading first to test strategies

---

## 7. Web Interface

The platform includes a web-based management interface:

### a. Start Web Server
```bash
python src/management/webgui/app.py
```

### b. Access Interface
Open your browser to `http://localhost:5000`

### c. Features
- Strategy configuration
- Data download management
- System status monitoring
- User management (with 2FA support)

---

## 8. Telegram Integration

For notifications and remote management:

### a. Configure Telegram Bot
Set up your Telegram bot token in the configuration files.

### b. Features
- Trade notifications
- System status updates
- Remote bot control
- Screener alerts

---

## 9. Workflow Summary

Typical workflow for strategy development:

1. **Data Preparation**: Place CSV files in `data/` directory
2. **Optimization**: Run `src/optimizer/run_optimizer.py`
3. **Visualization**: Run `src/plotter/run_plotter.py`
4. **Analysis**: Run `src/data/process_results.py` for CSV export
5. **Live Trading**: Configure and run `src/trading/run_bot.py`

---

## 10. Troubleshooting

### Common Issues
- **Missing data files**: Ensure CSV files are in the `data/` directory with correct naming
- **Configuration errors**: Check JSON syntax in config files
- **API errors**: Verify API keys and network connection
- **Memory issues**: Reduce `n_trials` in optimizer settings

### Logs
- Check `logs/` directory for detailed error information
- Use the web interface to monitor system status

---

## 11. Extending the Platform

### Adding New Strategies
- Create new entry/exit mixins in `src/entry/` and `src/exit/`
- Add configuration files in `config/optimizer/entry/` and `config/optimizer/exit/`
- Update `src/entry/entry_mixin_factory.py` and `src/exit/exit_mixin_factory.py`

### Adding New Indicators
- Create indicator classes in `src/indicator/`
- Add plotting support in `src/plotter/indicators/`
- Update `config/plotter/mixin_indicators.json`

### Adding New Brokers
- Subclass `AbstractBroker` in `src/broker/`
- Implement required methods for order management
- Add to `src/broker/broker_factory.py`

---

For more details, see the code comments and docstrings throughout the project. 
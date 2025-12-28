# Trading Bot Implementation Tasks

## Phase 1: Environment Setup

### Task 1.1: Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python -c "import backtrader, pandas, numpy, requests; print('Dependencies OK')"
```

### Task 1.2: Setup Binance Testnet
1. **Create Testnet Account**
   - Visit: https://testnet.binance.vision/
   - Register with email
   - Verify email address

2. **Generate API Keys**
   - Go to API Management
   - Create new API key
   - Enable "Enable Trading" permission
   - Copy API key and secret

3. **Get Testnet Funds**
   - Go to "Faucet" section
   - Request testnet USDT (free)
   - Verify balance in wallet

### Task 1.3: Environment Variables
Create `.env` file in project root:
```bash
# Binance Testnet API Keys
BINANCE_KEY=your_testnet_api_key_here
BINANCE_API_SECRET=your_testnet_secret_here

# Trading Settings
DEFAULT_SYMBOL=LTCUSDT
DEFAULT_TIMEFRAME=4h
DEFAULT_POSITION_SIZE=0.1
LOG_LEVEL=INFO
```

### Task 1.4: Directory Structure
```bash
# Create required directories
mkdir -p logs
mkdir -p db
mkdir -p data
mkdir -p results
```

## Phase 2: Configuration Setup

### Task 2.1: Create Trading Configuration
Create `config/trading/paper_trading_rsi_atr.json`:

```json
{
  "bot_id": "paper_trading_rsi_atr",
  "environment": "development",
  "version": "1.0.0",
  "description": "Paper trading bot with RSIOrBBEntryMixin + ATRExitMixin strategy",
  "symbol": "LTCUSDT",
  "timeframe": "4h",
  "risk_per_trade": 0.02,
  "max_open_trades": 3,
  "position_size": 0.1,
  "broker_type": "binance_paper",
  "initial_balance": 10000.0,
  "commission": 0.001,
  "data_source": "binance",
  "lookback_bars": 1000,
  "retry_interval": 60,
  "strategy_type": "custom",
  "strategy_params": {
    "entry_logic": {
      "name": "RSIOrBBEntryMixin",
      "params": {
        "e_rsi_period": 20,
        "e_rsi_oversold": 25,
        "e_bb_period": 16,
        "e_bb_dev": 2.08,
        "e_rsi_cross": false,
        "e_bb_reentry": false,
        "e_cooldown_bars": 4,
        "e_use_bb_touch": true
      }
    },
    "exit_logic": {
      "name": "ATRExitMixin",
      "params": {
        "x_atr_period": 14,
        "x_sl_multiplier": 2.0
      }
    },
    "use_talib": true
  },
  "stop_loss_pct": 5.0,
  "take_profit_pct": 10.0,
  "max_daily_trades": 10,
  "max_daily_loss": 100.0,
  "max_drawdown_pct": 20.0,
  "max_exposure": 1.0,
  "log_level": "INFO",
  "save_trades": true,
  "save_equity_curve": true,
  "log_file": "logs/paper_trading_rsi_atr.log",
  "notifications_enabled": true,
  "telegram_enabled": false,
  "email_enabled": false,
  "paper_trading": true,
  "trading_settings": {
    "enable_database_logging": true,
    "bot_type": "paper",
    "description": "Paper trading with RSI/BB entry and ATR exit strategy"
  }
}
```

### Task 2.2: Validate Configuration
```bash
# Validate the configuration file
python src/trading/config_validator.py config/trading/paper_trading_rsi_atr.json
```

Expected output:
```
✅ Configuration is valid
✅ All required fields present
✅ Parameter ranges valid
```

### Task 2.3: Test API Connection
```bash
# Test Binance API connection
python -c "
import os
from src.trading.broker.binance_paper_broker import BinancePaperBroker
broker = BinancePaperBroker(
    os.getenv('BINANCE_KEY'),
    os.getenv('BINANCE_API_SECRET')
)
print('✅ API connection successful!')
print(f'Account balance: {broker.get_balance()}')
"
```

## Phase 3: Strategy Testing

### Task 3.1: Test Entry Mixin
```bash
# Test RSIOrBBEntryMixin
python -c "
from src.strategy.entry.rsi_or_bb_entry_mixin import RSIOrBBEntryMixin
mixin = RSIOrBBEntryMixin()
print('✅ RSIOrBBEntryMixin loaded successfully')
print(f'Default params: {mixin.get_default_params()}')
"
```

### Task 3.2: Test Exit Mixin
```bash
# Test ATRExitMixin
python -c "
from src.strategy.exit.atr_exit_mixin import ATRExitMixin
mixin = ATRExitMixin()
print('✅ ATRExitMixin loaded successfully')
print(f'Default params: {mixin.get_default_params()}')
"
```

### Task 3.3: Test CustomStrategy
```bash
# Test CustomStrategy with both mixins
python -c "
from src.strategy.custom_strategy import CustomStrategy
config = {
    'entry_logic': {'name': 'RSIOrBBEntryMixin', 'params': {}},
    'exit_logic': {'name': 'ATRExitMixin', 'params': {}}
}
strategy = CustomStrategy(config)
print('✅ CustomStrategy with mixins loaded successfully')
"
```

## Phase 4: Paper Trading Execution

### Task 4.1: Start Paper Trading Bot
```bash
# Run the paper trading bot
python src/trading/trading_bot.py paper_trading_rsi_atr.json
```

Expected output:
```
2025-09-08 10:00:00,000 - INFO - Starting live trading bot runner.
2025-09-08 10:00:00,001 - INFO - Validating configuration: config/trading/paper_trading_rsi_atr.json
✅ Configuration is valid
2025-09-08 10:00:00,002 - INFO - Creating live trading bot with config: paper_trading_rsi_atr.json
2025-09-08 10:00:00,003 - INFO - Starting live trading bot...
2025-09-08 10:00:00,004 - INFO - Bot initialized successfully
2025-09-08 10:00:00,005 - INFO - Starting data feed...
2025-09-08 10:00:00,006 - INFO - Bot is running...
```

### Task 4.2: Monitor Trading Activity
Watch the logs for:
- Data feed updates
- Entry signals
- Position entries
- Exit signals
- Trade closures

Example log output:
```
2025-09-08 10:15:00,000 - DEBUG - RSIOrBBEntryMixin.should_enter - ENTRY: Price: 45.23, RSI: 28.5, BB Lower: 44.89, RSI Cross: True, BB Reentry: False
2025-09-08 10:15:00,001 - INFO - base_strategy.py - _enter_position - LONG entry - Size: 0.100, Price: 45.2300, Reason: Entry mixin signal
2025-09-08 10:15:00,002 - INFO - base_strategy.py - notify_trade - Trade opened - Price: 45.2300, Size: 0.100, Position ID: abc123
```

### Task 4.3: Verify Database Storage
```bash
# Check if trades are being stored in database
python -c "
from src.data.database import TradeRepository
repo = TradeRepository()
trades = repo.get_trades_by_bot_id('paper_trading_rsi_atr')
print(f'✅ Found {len(trades)} trades in database')
for trade in trades[-3:]:  # Show last 3 trades
    print(f'Trade: {trade.entry_price} -> {trade.exit_price}, PnL: {trade.net_pnl}')
"
```

## Phase 5: Performance Monitoring

### Task 5.1: Check Trade Performance
```bash
# Analyze trade performance
python -c "
from src.data.database import TradeRepository
repo = TradeRepository()
trades = repo.get_trades_by_bot_id('paper_trading_rsi_atr')

if trades:
    total_pnl = sum(t.net_pnl for t in trades)
    winning_trades = [t for t in trades if t.net_pnl > 0]
    losing_trades = [t for t in trades if t.net_pnl < 0]
    
    print(f'Total Trades: {len(trades)}')
    print(f'Total PnL: {total_pnl:.2f}')
    print(f'Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)')
    print(f'Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)')
    print(f'Average Win: {sum(t.net_pnl for t in winning_trades)/len(winning_trades):.2f}' if winning_trades else 'No wins')
    print(f'Average Loss: {sum(t.net_pnl for t in losing_trades)/len(losing_trades):.2f}' if losing_trades else 'No losses')
else:
    print('No trades found in database')
"
```

### Task 5.2: Monitor Risk Metrics
```bash
# Check risk metrics
python -c "
from src.data.database import TradeRepository
repo = TradeRepository()
trades = repo.get_trades_by_bot_id('paper_trading_rsi_atr')

if trades:
    pnls = [t.net_pnl for t in trades]
    max_drawdown = min(pnls)
    current_balance = 10000 + sum(pnls)  # Starting balance + total PnL
    
    print(f'Current Balance: {current_balance:.2f}')
    print(f'Max Drawdown: {max_drawdown:.2f}')
    print(f'Drawdown %: {abs(max_drawdown)/10000*100:.2f}%')
    
    # Check if within risk limits
    if abs(max_drawdown)/10000*100 > 20:
        print('⚠️  WARNING: Drawdown exceeds 20% limit!')
    else:
        print('✅ Drawdown within acceptable limits')
"
```

## Phase 6: Optimization and Tuning

### Task 6.1: Parameter Optimization
```bash
# Run optimization to find better parameters
python src/backtester/optimizer/run_optimizer.py \
    --data-file data/LTCUSDT_4h_20220101_20250707.csv \
    --entry-mixin RSIOrBBEntryMixin \
    --exit-mixin ATRExitMixin \
    --timeframe 4h \
    --trials 100
```

### Task 6.2: Backtest Validation
```bash
# Run backtest with optimized parameters
python src/backtester/run_backtester.py \
    --config config/trading/paper_trading_rsi_atr.json \
    --data-file data/LTCUSDT_4h_20220101_20250707.csv \
    --output results/backtest_rsi_atr.json
```

### Task 6.3: Update Configuration
Update `paper_trading_rsi_atr.json` with optimized parameters from backtest results.

## Phase 7: Production Deployment

### Task 7.1: Pre-Production Checklist
- [ ] Paper trading successful for 1+ weeks
- [ ] All trades logged to database
- [ ] Risk metrics within limits
- [ ] No critical errors in logs
- [ ] Performance meets expectations

### Task 7.2: Live Trading Setup
1. **Create Live Configuration**
   ```bash
   cp config/trading/paper_trading_rsi_atr.json config/trading/live_trading_rsi_atr.json
   ```

2. **Update for Live Trading**
   - Change `"paper_trading": false`
   - Change `"bot_type": "live"`
   - Update API keys to mainnet
   - Reduce position size for safety

3. **Start Live Trading**
   ```bash
   python src/trading/trading_bot.py live_trading_rsi_atr.json
   ```

## Troubleshooting Tasks

### Task T.1: Common Issues
```bash
# Check logs for errors
tail -f logs/paper_trading_rsi_atr.log

# Validate configuration
python src/trading/config_validator.py config/trading/paper_trading_rsi_atr.json

# Test API connection
python -c "from src.trading.broker.binance_paper_broker import BinancePaperBroker; print('API OK')"
```

### Task T.2: Performance Issues
```bash
# Check system resources
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Disk: {psutil.disk_usage(\".\").percent}%')
"

# Check database performance
python -c "
from src.data.database import TradeRepository
repo = TradeRepository()
import time
start = time.time()
trades = repo.get_trades_by_bot_id('paper_trading_rsi_atr')
print(f'Database query time: {time.time() - start:.3f}s')
"
```

### Task T.3: Strategy Issues
```bash
# Test individual mixins
python -c "
from src.strategy.entry.rsi_or_bb_entry_mixin import RSIOrBBEntryMixin
from src.strategy.exit.atr_exit_mixin import ATRExitMixin
print('Entry mixin:', RSIOrBBEntryMixin().get_default_params())
print('Exit mixin:', ATRExitMixin().get_default_params())
"
```

## Success Criteria

### Phase 1-3: Setup Complete
- [ ] All dependencies installed
- [ ] Binance testnet configured
- [ ] Configuration validated
- [ ] API connection working

### Phase 4: Paper Trading Working
- [ ] Bot starts without errors
- [ ] Trades are being executed
- [ ] Database logging working
- [ ] No critical errors

### Phase 5: Performance Acceptable
- [ ] Positive PnL over time
- [ ] Risk metrics within limits
- [ ] Stable execution
- [ ] Good win rate

### Phase 6-7: Production Ready
- [ ] Optimized parameters
- [ ] Thoroughly tested
- [ ] Risk management working
- [ ] Ready for live trading

## Next Steps

After completing all tasks:
1. **Monitor Performance**: Track metrics for 1-2 weeks
2. **Optimize Parameters**: Use backtest results to improve
3. **Scale Up**: Increase position size gradually
4. **Add Features**: Implement additional risk controls
5. **Deploy Live**: Move to live trading when ready

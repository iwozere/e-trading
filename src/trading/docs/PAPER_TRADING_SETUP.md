# Paper Trading Setup Guide

## 🎯 Overview

This guide will help you set up paper trading on Binance testnet with your optimized strategies:
- **RSIOrBBEntryMixin + RSIOrBBExitMixin** (82 trades, +122 profit)
- **RSIOrBBEntryMixin + SimpleATRExitMixin** (new simplified exit)

## 🔧 Prerequisites

1. **Python Environment**: Ensure you have Python 3.8+ with all dependencies installed
2. **Binance Account**: You need a Binance account (real or testnet)
3. **Project Setup**: Ensure the e-trading project is properly set up

## 🚀 Quick Start

### Step 1: Set Up Binance Testnet

1. **Visit Binance Testnet**: https://testnet.binance.vision/
2. **Login**: Use your existing Binance credentials
3. **Create API Key**:
   - Go to API Management
   - Create new API key for testnet
   - Copy the API Key and Secret Key
   - **Important**: Use testnet keys only!

### Step 2: Set Environment Variables

Create a `.env` file in your project root or set environment variables:

```bash
# Option 1: Create .env file
echo "BINANCE_KEY=your_testnet_api_key_here" > .env
echo "BINANCE_API_SECRET=your_testnet_secret_key_here" >> .env

# Option 2: Set environment variables
export BINANCE_KEY="your_testnet_api_key_here"
export BINANCE_API_SECRET="your_testnet_secret_key_here"
```

### Step 3: Run Setup Validation

```bash
python setup_paper_trading.py
```

This will:
- ✅ Check environment variables
- ✅ Validate configuration files
- ✅ Test Binance testnet connection
- ✅ Provide next steps

### Step 4: Start Paper Trading

**Option A: RSI/BB Strategy (Best Performance)**
```bash
python src/trading/trading_bot.py paper_trading_rsi_or_bb.json
```

**Option B: Simple ATR Strategy (Simplified)**
```bash
python src/trading/trading_bot.py paper_trading_simple_atr.json
```

## 📊 Strategy Configurations

### RSI/BB Strategy (Optimized)
- **Entry**: RSIOrBBEntryMixin
  - RSI period: 20, oversold: 25
  - BB period: 16, deviation: 2.08
  - Cooldown: 4 bars
- **Exit**: RSIOrBBExitMixin
  - RSI period: 18, overbought: 76.22
  - BB period: 16, deviation: 2.37
- **Performance**: 82 trades, +122 profit

### Simple ATR Strategy (New)
- **Entry**: RSIOrBBEntryMixin (same as above)
- **Exit**: SimpleATRExitMixin
  - ATR period: 14, multiplier: 2.0
  - Breakeven: 1.0 ATR
- **Advantage**: Only 4 parameters, easier to optimize

## 📈 Monitoring

### Logs
- **Location**: `logs/` directory
- **Files**: 
  - `paper_trading_rsi_or_bb.log`
  - `paper_trading_simple_atr.log`

### Database (if enabled)
- **Location**: `trading_bot.db`
- **Tables**: trades, positions, orders
- **Query**: Use SQLite browser or command line

### Real-time Monitoring
- Check console output for trade signals
- Monitor position updates
- Watch for error messages

## ⚙️ Configuration Options

### Risk Management
```json
{
  "risk_per_trade": 0.02,        // 2% risk per trade
  "max_open_trades": 3,          // Maximum concurrent positions
  "max_daily_loss": 100.0,       // Daily loss limit
  "max_drawdown_pct": 20.0       // Maximum drawdown
}
```

### Trading Settings
```json
{
  "symbol": "LTCUSDT",           // Trading pair
  "timeframe": "4h",             // Chart timeframe
  "initial_balance": 10000.0,    // Starting capital
  "commission": 0.001            // 0.1% commission
}
```

## 🔍 Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Check API keys are correct
   - Ensure using testnet keys (not mainnet)
   - Verify network connectivity

2. **Configuration Validation Failed**
   - Run `python setup_paper_trading.py`
   - Check JSON syntax in config files
   - Verify all required fields are present

3. **No Trades Generated**
   - Check if market conditions meet strategy criteria
   - Verify data feed is working
   - Check logs for entry/exit signals

4. **Database Errors**
   - Ensure database file permissions
   - Check if database is locked by another process
   - Verify database schema is up to date

### Debug Mode

Enable debug logging:
```json
{
  "log_level": "DEBUG"
}
```

## 📞 Support

If you encounter issues:
1. Check the logs in `logs/` directory
2. Run the setup validation script
3. Verify your Binance testnet credentials
4. Check the configuration files for syntax errors

## 🎯 Next Steps

After successful paper trading:
1. **Analyze Performance**: Compare with backtest results
2. **Optimize Parameters**: Fine-tune based on live performance
3. **Risk Management**: Adjust position sizes and risk limits
4. **Scale Up**: Consider live trading with real funds (carefully!)

---

**⚠️ Important**: This is paper trading on testnet. No real money is involved, but the strategies and logic are the same as live trading.

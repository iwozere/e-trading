# Trading Bot Documentation

This documentation provides comprehensive guidance for running the live trading bot with CustomStrategy using RSIOrBBEntryMixin and ATRExitMixin with Binance paper broker.

## Overview

The trading bot system consists of:
- **CustomStrategy**: Modular strategy framework using entry and exit mixins
- **RSIOrBBEntryMixin**: Entry logic based on RSI and Bollinger Bands
- **ATRExitMixin**: Exit logic using Average True Range trailing stop loss
- **BinancePaperBroker**: Paper trading broker using Binance testnet

## Quick Start

1. **Setup Environment Variables** (see Requirements.md)
2. **Create Configuration** (see Design.md)
3. **Run the Bot**:
   ```bash
   python src/trading/trading_bot.py paper_trading_rsi_atr.json
   ```

## Documentation Structure

- **[Requirements.md](Requirements.md)**: System requirements and environment setup
- **[Design.md](Design.md)**: Architecture and configuration design
- **[Tasks.md](Tasks.md)**: Step-by-step implementation tasks

## Key Features

### Strategy Components
- **Entry Logic**: RSI oversold + Bollinger Band lower touch
- **Exit Logic**: ATR-based trailing stop loss
- **Risk Management**: Position sizing, stop loss, take profit
- **Paper Trading**: Safe testing on Binance testnet

### Monitoring & Logging
- Real-time trade execution logs
- Database persistence of all trades
- Performance metrics and analytics
- Error handling and recovery

### Configuration
- JSON-based configuration files
- Environment-specific settings
- Parameter validation and testing

## Getting Started

1. Read [Requirements.md](Requirements.md) for setup prerequisites
2. Follow [Tasks.md](Tasks.md) for step-by-step implementation
3. Review [Design.md](Design.md) for architecture understanding

## Support

For issues or questions:
1. Check the logs in `logs/` directory
2. Review configuration validation errors
3. Consult the troubleshooting section in Tasks.md

## Safety Notice

⚠️ **Always test with paper trading first!**
- This system supports both paper and live trading
- Paper trading uses Binance testnet (no real money)
- Live trading executes real orders with real money
- Always validate strategies thoroughly before live trading

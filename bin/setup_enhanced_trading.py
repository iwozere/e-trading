#!/usr/bin/env python3
"""
Enhanced Trading System Setup
----------------------------

This script sets up the enhanced multi-strategy trading system.
"""

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def create_directories():
    """Create necessary directories."""
    directories = [
        "logs/enhanced_trading",
        "config/enhanced_trading",
        "data/enhanced_trading",
        "db"
    ]

    for directory in directories:
        path = PROJECT_ROOT / directory
        path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {path}")


def create_env_file():
    """Create .env file if it doesn't exist."""
    env_file = PROJECT_ROOT / ".env"

    if env_file.exists():
        print(f"✅ .env file already exists: {env_file}")
        return

    env_content = """# Enhanced Trading System Configuration
# Copy this to .env and fill in your actual values

# Binance API Configuration
# For paper trading, you can use testnet keys from: https://testnet.binance.vision/
BINANCE_KEY=your_BINANCE_KEY_here
BINANCE_API_SECRET=your_binance_api_secret_here

# IBKR Configuration (optional)
IBKR_HOST=127.0.0.1
IBKR_PAPER_PORT=7497
IBKR_LIVE_PORT=7496

# Telegram Configuration (optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Email Configuration (optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Database Configuration
DATABASE_URL=sqlite:///db/trading.db

# System Configuration
TRADING_ENV=development
DEBUG_MODE=true
LOG_LEVEL=INFO
MAX_CONCURRENT_STRATEGIES=10
"""

    with open(env_file, 'w') as f:
        f.write(env_content)

    print(f"✅ Created .env template: {env_file}")
    print("⚠️  Please edit .env file with your actual API keys!")


def create_multi_strategy_configs():
    """Create sample multi-strategy configurations."""

    # Configuration for multiple Binance strategies
    binance_multi_config = {
        "system": {
            "name": "Enhanced Multi-Strategy Trading System",
            "version": "2.0.0",
            "description": "Multiple strategies running simultaneously with enhanced broker system",
            "max_concurrent_strategies": 5,
            "health_check_interval": 30,
            "auto_recovery": True
        },
        "strategies": [
            {
                "id": "rsi_btc_strategy",
                "name": "RSI BTC Strategy",
                "symbol": "BTCUSDT",
                "enabled": True,
                "broker_config": {
                    "type": "binance",
                    "trading_mode": "paper",
                    "name": "rsi_btc_bot",
                    "cash": 5000.0,
                    "paper_trading_config": {
                        "mode": "realistic",
                        "initial_balance": 5000.0,
                        "commission_rate": 0.001,
                        "slippage_model": "linear",
                        "base_slippage": 0.0005,
                        "latency_simulation": True,
                        "min_latency_ms": 20,
                        "max_latency_ms": 100
                    }
                },
                "strategy_config": {
                    "type": "RSI",
                    "timeframe": "1h",
                    "rsi_period": 14,
                    "rsi_oversold": 30,
                    "rsi_overbought": 70,
                    "position_size": 0.1
                },
                "risk_management": {
                    "max_position_size": 1000.0,
                    "stop_loss_pct": 3.0,
                    "take_profit_pct": 6.0,
                    "max_daily_loss": 200.0
                }
            },
            {
                "id": "bb_eth_strategy",
                "name": "Bollinger Bands ETH Strategy",
                "symbol": "ETHUSDT",
                "enabled": True,
                "broker_config": {
                    "type": "binance",
                    "trading_mode": "paper",
                    "name": "bb_eth_bot",
                    "cash": 3000.0,
                    "paper_trading_config": {
                        "mode": "realistic",
                        "initial_balance": 3000.0,
                        "commission_rate": 0.001,
                        "slippage_model": "linear",
                        "base_slippage": 0.0005,
                        "latency_simulation": True,
                        "min_latency_ms": 25,
                        "max_latency_ms": 120
                    }
                },
                "strategy_config": {
                    "type": "BollingerBands",
                    "timeframe": "1h",
                    "bb_period": 20,
                    "bb_std": 2.0,
                    "position_size": 0.15
                },
                "risk_management": {
                    "max_position_size": 800.0,
                    "stop_loss_pct": 2.5,
                    "take_profit_pct": 5.0,
                    "max_daily_loss": 150.0
                }
            },
            {
                "id": "macd_ada_strategy",
                "name": "MACD ADA Strategy",
                "symbol": "ADAUSDT",
                "enabled": True,
                "broker_config": {
                    "type": "binance",
                    "trading_mode": "paper",
                    "name": "macd_ada_bot",
                    "cash": 2000.0,
                    "paper_trading_config": {
                        "mode": "realistic",
                        "initial_balance": 2000.0,
                        "commission_rate": 0.001,
                        "slippage_model": "linear",
                        "base_slippage": 0.0008,
                        "latency_simulation": True,
                        "min_latency_ms": 30,
                        "max_latency_ms": 150
                    }
                },
                "strategy_config": {
                    "type": "MACD",
                    "timeframe": "4h",
                    "macd_fast": 12,
                    "macd_slow": 26,
                    "macd_signal": 9,
                    "position_size": 0.2
                },
                "risk_management": {
                    "max_position_size": 500.0,
                    "stop_loss_pct": 4.0,
                    "take_profit_pct": 8.0,
                    "max_daily_loss": 100.0
                }
            }
        ],
        "notifications": {
            "system_notifications": True,
            "strategy_notifications": True,
            "email_enabled": True,
            "telegram_enabled": True,
            "notification_levels": ["ERROR", "WARNING", "TRADE"]
        },
        "monitoring": {
            "performance_tracking": True,
            "health_monitoring": True,
            "resource_monitoring": True,
            "alert_thresholds": {
                "cpu_usage": 80,
                "memory_usage": 85,
                "error_rate": 5
            }
        }
    }

    config_file = PROJECT_ROOT / "config/enhanced_trading/multi_strategy_binance.json"
    with open(config_file, 'w') as f:
        json.dump(binance_multi_config, f, indent=2)

    print(f"✅ Created multi-strategy config: {config_file}")

    # Create a simpler 2-strategy config for testing
    simple_config = {
        "system": {
            "name": "Simple Multi-Strategy Test",
            "version": "2.0.0",
            "description": "Two strategies for testing",
            "max_concurrent_strategies": 2
        },
        "strategies": [
            binance_multi_config["strategies"][0],  # RSI BTC
            binance_multi_config["strategies"][1]   # BB ETH
        ],
        "notifications": binance_multi_config["notifications"],
        "monitoring": binance_multi_config["monitoring"]
    }

    simple_config_file = PROJECT_ROOT / "config/enhanced_trading/simple_multi_strategy.json"
    with open(simple_config_file, 'w') as f:
        json.dump(simple_config, f, indent=2)

    print(f"✅ Created simple multi-strategy config: {simple_config_file}")


def main():
    """Main setup function."""
    print("🚀 Setting up Enhanced Multi-Strategy Trading System")
    print("=" * 60)

    create_directories()
    print()

    create_env_file()
    print()

    create_multi_strategy_configs()
    print()

    print("✅ Setup complete!")
    print()
    print("Next steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: python enhanced_multi_strategy_runner.py")
    print("3. Or run: python examples/multi_strategy_paper_trading.py")


if __name__ == "__main__":
    main()
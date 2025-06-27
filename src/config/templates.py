"""
Configuration Templates
======================

Pre-defined configuration templates for different use cases.
Makes it easy to create new configurations with sensible defaults.
"""

from typing import Dict, Any, List, Optional
from .schemas import Environment, BrokerType, DataSourceType, StrategyType, NotificationType


class ConfigTemplates:
    """
    Configuration templates for different use cases.
    
    Provides pre-defined templates for:
    - Trading bots (paper/live)
    - Optimizers
    - Data feeds
    - Development/Staging/Production environments
    """
    
    def __init__(self):
        """Initialize configuration templates"""
        self._templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load all configuration templates"""
        return {
            # Trading Bot Templates
            "trading_paper": self._get_paper_trading_template(),
            "trading_live": self._get_live_trading_template(),
            "trading_dev": self._get_dev_trading_template(),
            
            # Optimizer Templates
            "optimizer_basic": self._get_basic_optimizer_template(),
            "optimizer_advanced": self._get_advanced_optimizer_template(),
            
            # Data Feed Templates
            "data_binance": self._get_binance_data_template(),
            "data_yahoo": self._get_yahoo_data_template(),
            "data_ibkr": self._get_ibkr_data_template(),
            
            # Environment Templates
            "env_development": self._get_development_env_template(),
            "env_staging": self._get_staging_env_template(),
            "env_production": self._get_production_env_template(),
        }
    
    def get_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get a configuration template by name"""
        return self._templates.get(template_name, {}).copy()
    
    def list_templates(self) -> List[str]:
        """List all available template names"""
        return list(self._templates.keys())
    
    def get_template_description(self, template_name: str) -> str:
        """Get description for a template"""
        descriptions = {
            "trading_paper": "Paper trading bot with safe defaults",
            "trading_live": "Live trading bot with full risk management",
            "trading_dev": "Development trading bot with debugging enabled",
            "optimizer_basic": "Basic optimization with common parameters",
            "optimizer_advanced": "Advanced optimization with extended parameters",
            "data_binance": "Binance data feed configuration",
            "data_yahoo": "Yahoo Finance data feed configuration",
            "data_ibkr": "IBKR data feed configuration",
            "env_development": "Development environment settings",
            "env_staging": "Staging environment settings",
            "env_production": "Production environment settings",
        }
        return descriptions.get(template_name, "No description available")
    
    def _get_paper_trading_template(self) -> Dict[str, Any]:
        """Paper trading bot template"""
        return {
            "environment": Environment.DEVELOPMENT,
            "version": "1.0.0",
            "description": "Paper trading bot for strategy testing",
            
            "bot_id": "paper_bot_001",
            "broker": {
                "type": BrokerType.BINANCE_PAPER,
                "initial_balance": 10000.0,
                "commission": 0.001
            },
            "trading": {
                "symbol": "BTCUSDT",
                "position_size": 0.1,
                "max_positions": 1,
                "max_drawdown_pct": 20.0,
                "max_exposure": 1.0
            },
            "data": {
                "data_source": DataSourceType.BINANCE,
                "symbol": "BTCUSDT",
                "interval": "1h",
                "lookback_bars": 1000,
                "retry_interval": 60,
                "testnet": True
            },
            "strategy": {
                "type": StrategyType.CUSTOM,
                "entry_logic": {
                    "name": "RSIBBVolumeEntryMixin",
                    "params": {
                        "rsi_period": 14,
                        "rsi_oversold": 30,
                        "bb_period": 20,
                        "bb_dev": 2.0,
                        "vol_ma_period": 20
                    }
                },
                "exit_logic": {
                    "name": "RSIBBExitMixin",
                    "params": {
                        "rsi_period": 14,
                        "rsi_overbought": 70
                    }
                },
                "position_size": 0.1
            },
            "risk_management": {
                "stop_loss_pct": 5.0,
                "take_profit_pct": 10.0,
                "max_daily_trades": 10,
                "max_daily_loss": 50.0,
                "max_drawdown_pct": 20.0,
                "max_exposure": 1.0,
                "trailing_stop": {
                    "enabled": False,
                    "activation_pct": 3.0,
                    "trailing_pct": 2.0
                }
            },
            "logging": {
                "level": "INFO",
                "save_trades": True,
                "save_equity_curve": True,
                "log_file": "logs/paper/trading_bot_paper.log"
            },
            "scheduling": {
                "enabled": False,
                "start_time": "09:00",
                "end_time": "17:00",
                "timezone": "UTC",
                "trading_days": ["monday", "tuesday", "wednesday", "thursday", "friday"]
            },
            "performance": {
                "target_sharpe_ratio": 1.0,
                "target_win_rate": 60.0,
                "target_profit_factor": 1.5,
                "max_consecutive_losses": 5,
                "performance_check_interval": 24
            },
            "notifications": {
                "enabled": True,
                "telegram": {
                    "enabled": False,
                    "notify_on": [NotificationType.TRADE_ENTRY, NotificationType.TRADE_EXIT, NotificationType.ERROR]
                },
                "email": {
                    "enabled": False,
                    "notify_on": [NotificationType.TRADE_ENTRY, NotificationType.TRADE_EXIT, NotificationType.ERROR]
                }
            }
        }
    
    def _get_live_trading_template(self) -> Dict[str, Any]:
        """Live trading bot template"""
        template = self._get_paper_trading_template()
        template.update({
            "environment": Environment.PRODUCTION,
            "description": "Live trading bot with full risk management",
            "bot_id": "live_bot_001",
            "broker": {
                "type": BrokerType.BINANCE_LIVE,
                "initial_balance": 1000.0,
                "commission": 0.001,
                "api_key": "YOUR_API_KEY",
                "api_secret": "YOUR_API_SECRET"
            },
            "risk_management": {
                "stop_loss_pct": 3.0,  # Tighter stop loss for live trading
                "take_profit_pct": 6.0,
                "max_daily_trades": 5,  # Fewer trades for live
                "max_daily_loss": 25.0,  # Lower daily loss limit
                "max_drawdown_pct": 15.0,
                "max_exposure": 0.5,  # Lower exposure for live
                "trailing_stop": {
                    "enabled": True,
                    "activation_pct": 2.0,
                    "trailing_pct": 1.5
                }
            },
            "logging": {
                "level": "WARNING",
                "save_trades": True,
                "save_equity_curve": True,
                "log_file": "logs/live/trading_bot_live.log"
            },
            "scheduling": {
                "enabled": True,
                "start_time": "09:00",
                "end_time": "17:00",
                "timezone": "UTC",
                "trading_days": ["monday", "tuesday", "wednesday", "thursday", "friday"]
            },
            "notifications": {
                "enabled": True,
                "telegram": {
                    "enabled": True,
                    "notify_on": [NotificationType.TRADE_ENTRY, NotificationType.TRADE_EXIT, NotificationType.ERROR, NotificationType.STATUS]
                },
                "email": {
                    "enabled": True,
                    "notify_on": [NotificationType.TRADE_ENTRY, NotificationType.TRADE_EXIT, NotificationType.ERROR]
                }
            }
        })
        return template
    
    def _get_dev_trading_template(self) -> Dict[str, Any]:
        """Development trading bot template"""
        template = self._get_paper_trading_template()
        template.update({
            "environment": Environment.DEVELOPMENT,
            "description": "Development trading bot with debugging enabled",
            "bot_id": "dev_bot_001",
            "logging": {
                "level": "DEBUG",
                "save_trades": True,
                "save_equity_curve": True,
                "log_file": "logs/dev/trading_bot_dev.log"
            },
            "scheduling": {
                "enabled": False
            },
            "notifications": {
                "enabled": False
            }
        })
        return template
    
    def _get_basic_optimizer_template(self) -> Dict[str, Any]:
        """Basic optimizer template"""
        return {
            "environment": Environment.DEVELOPMENT,
            "version": "1.0.0",
            "description": "Basic optimization configuration",
            
            "optimizer_type": "optuna",
            "initial_capital": 1000.0,
            "commission": 0.001,
            "risk_free_rate": 0.01,
            "n_trials": 100,
            "n_jobs": 1,
            "position_size": 0.1,
            
            "entry_strategies": [
                {
                    "name": "RSIBBVolumeEntryMixin",
                    "params": {
                        "rsi_period": {"type": "int", "low": 5, "high": 30, "default": 14},
                        "bb_period": {"type": "int", "low": 10, "high": 50, "default": 20},
                        "bb_dev": {"type": "float", "low": 1.0, "high": 3.0, "default": 2.0}
                    }
                }
            ],
            "exit_strategies": [
                {
                    "name": "RSIBBExitMixin",
                    "params": {
                        "rsi_period": {"type": "int", "low": 5, "high": 30, "default": 14},
                        "rsi_overbought": {"type": "int", "low": 60, "high": 90, "default": 70}
                    }
                }
            ],
            
            "plot": True,
            "save_trades": True,
            "output_dir": "results"
        }
    
    def _get_advanced_optimizer_template(self) -> Dict[str, Any]:
        """Advanced optimizer template"""
        template = self._get_basic_optimizer_template()
        template.update({
            "description": "Advanced optimization with extended parameters",
            "n_trials": 500,
            "n_jobs": -1,  # Use all cores
            
            "entry_strategies": [
                {
                    "name": "RSIBBVolumeEntryMixin",
                    "params": {
                        "rsi_period": {"type": "int", "low": 5, "high": 30, "default": 14},
                        "bb_period": {"type": "int", "low": 10, "high": 50, "default": 20},
                        "bb_dev": {"type": "float", "low": 1.0, "high": 3.0, "default": 2.0},
                        "vol_ma_period": {"type": "int", "low": 5, "high": 50, "default": 20},
                        "rsi_oversold": {"type": "int", "low": 20, "high": 40, "default": 30}
                    }
                },
                {
                    "name": "RSIVolumeSupertrendEntryMixin",
                    "params": {
                        "rsi_period": {"type": "int", "low": 5, "high": 30, "default": 14},
                        "st_period": {"type": "int", "low": 5, "high": 30, "default": 10},
                        "st_multiplier": {"type": "float", "low": 1.0, "high": 5.0, "default": 3.0}
                    }
                }
            ],
            "exit_strategies": [
                {
                    "name": "ATRExitMixin",
                    "params": {
                        "atr_period": {"type": "int", "low": 5, "high": 30, "default": 14},
                        "tp_multiplier": {"type": "float", "low": 1.0, "high": 5.0, "default": 2.0},
                        "sl_multiplier": {"type": "float", "low": 0.5, "high": 3.0, "default": 1.0}
                    }
                },
                {
                    "name": "TrailingStopExitMixin",
                    "params": {
                        "trail_pct": {"type": "float", "low": 0.01, "high": 0.1, "default": 0.02}
                    }
                }
            ]
        })
        return template
    
    def _get_binance_data_template(self) -> Dict[str, Any]:
        """Binance data feed template"""
        return {
            "environment": Environment.DEVELOPMENT,
            "version": "1.0.0",
            "description": "Binance data feed configuration",
            
            "data_source": DataSourceType.BINANCE,
            "symbol": "BTCUSDT",
            "interval": "1h",
            "lookback_bars": 1000,
            "retry_interval": 60,
            "testnet": True
        }
    
    def _get_yahoo_data_template(self) -> Dict[str, Any]:
        """Yahoo Finance data feed template"""
        return {
            "environment": Environment.DEVELOPMENT,
            "version": "1.0.0",
            "description": "Yahoo Finance data feed configuration",
            
            "data_source": DataSourceType.YAHOO,
            "symbol": "AAPL",
            "interval": "5m",
            "lookback_bars": 500,
            "retry_interval": 60,
            "polling_interval": 60
        }
    
    def _get_ibkr_data_template(self) -> Dict[str, Any]:
        """IBKR data feed template"""
        return {
            "environment": Environment.DEVELOPMENT,
            "version": "1.0.0",
            "description": "IBKR data feed configuration",
            
            "data_source": DataSourceType.IBKR,
            "symbol": "SPY",
            "interval": "1m",
            "lookback_bars": 1000,
            "retry_interval": 60,
            "host": "127.0.0.1",
            "port": 7497,
            "client_id": 1
        }
    
    def _get_development_env_template(self) -> Dict[str, Any]:
        """Development environment template"""
        return {
            "environment": Environment.DEVELOPMENT,
            "debug": True,
            "log_level": "DEBUG",
            "hot_reload": True,
            "database": {
                "url": "sqlite:///dev_trading.db",
                "echo": True
            },
            "api": {
                "host": "localhost",
                "port": 5000,
                "debug": True
            }
        }
    
    def _get_staging_env_template(self) -> Dict[str, Any]:
        """Staging environment template"""
        return {
            "environment": Environment.STAGING,
            "debug": False,
            "log_level": "INFO",
            "hot_reload": False,
            "database": {
                "url": "sqlite:///staging_trading.db",
                "echo": False
            },
            "api": {
                "host": "0.0.0.0",
                "port": 5000,
                "debug": False
            }
        }
    
    def _get_production_env_template(self) -> Dict[str, Any]:
        """Production environment template"""
        return {
            "environment": Environment.PRODUCTION,
            "debug": False,
            "log_level": "WARNING",
            "hot_reload": False,
            "database": {
                "url": "sqlite:///prod_trading.db",
                "echo": False
            },
            "api": {
                "host": "0.0.0.0",
                "port": 5000,
                "debug": False
            },
            "security": {
                "require_ssl": True,
                "rate_limit": True,
                "max_requests_per_minute": 100
            }
        } 
# Model Module (`src/model`)

This directory contains all core data models, configuration schemas, and enums used throughout the trading and analytics platform. Models are implemented using Python dataclasses, enums, and Pydantic for validation and serialization.

## File Descriptions

### telegram_bot.py
Data models for Telegram bot integration:
- Fundamentals and technicals data structures
- Ticker analysis result encapsulation
- Command specification and parsing models

### strategy.py
Models for trading strategy logic:
- Aggregation methods and market regime enums
- Trading signal and composite signal dataclasses

### analytics.py
Models for analytics and performance tracking:
- Trade data structure
- Performance metrics (returns, drawdown, Sharpe, win rate, etc.)

### machine_learning.py
Models for machine learning and feature engineering:
- Feature types, training triggers, and model types enums
- Feature, model metadata, training config, and performance metrics dataclasses

### notification.py
Notification and alerting models:
- Notification, alert, and escalation rule dataclasses
- Notification types, priorities, alert severity, and channels enums

### error_handling.py
Error handling and resilience models:
- Circuit breaker, error event, alert, recovery, and retry config dataclasses
- Circuit state, error severity, recovery and retry strategy enums

### config_models.py
Pydantic-based configuration models for the trading platform:
- TradingBotConfig, OptimizerConfig, DataConfig (with validation)
- Environment, BrokerType, DataSourceType, StrategyType enums

### schemas.py
Pydantic-based configuration schemas and validation logic:
- ConfigSchema, RiskManagementConfig, LoggingConfig, SchedulingConfig, and more
- Environment, BrokerType, DataSourceType, StrategyType, NotificationType enums

---

For more details, see the docstrings and code in each file.

# Consolidated Alert System

This directory contains the centralized alert evaluation system that replaces the previous distributed alert logic.

## Architecture

```
src/common/alerts/
├── alert_evaluator.py     # Core alert evaluation logic
├── alert_runner.py        # Standalone alert evaluation runner
├── schema_validator.py    # Alert configuration validation
├── cron_parser.py         # Cron expression parsing
└── schemas/               # Alert configuration schemas
```

## Key Components

### AlertEvaluator
The core alert evaluation engine that:
- Validates alert configurations
- Fetches market data and calculates indicators
- Evaluates rule trees and rearm logic
- Manages alert state and notifications

### AlertsService
Service layer that provides a clean API for:
- Creating and managing alerts
- Batch evaluation of alerts
- Integration with the jobs system

### AlertRunner
Standalone script for:
- Running alert evaluations manually or via cron
- Sending notifications for triggered alerts
- Batch processing of all alerts

## Usage Examples

### Creating an Alert via Telegram
```
/alerts add BTCUSDT 65000 above -email
```

### Running Alert Evaluation
```bash
# Evaluate all alerts
python src/common/alerts/alert_runner.py

# Evaluate alerts for specific user
python src/common/alerts/alert_runner.py --user-id 123

# Evaluate with limit
python src/common/alerts/alert_runner.py --limit 50
```

### Using AlertsService in Code
```python
from src.data.db.services.alerts_service import AlertsService
from src.data.db.services.jobs_service import JobsService
from src.data.data_manager import DataManager
from src.indicators.service import IndicatorService

# Initialize services
jobs_service = JobsService(session)
data_manager = DataManager()
indicator_service = IndicatorService()
alerts_service = AlertsService(jobs_service, data_manager, indicator_service)

# Create an alert
alert_config = {
    "ticker": "BTCUSDT",
    "timeframe": "15m",
    "rule": {
        "type": "price",
        "condition": "above",
        "value": 65000
    }
}
result = await alerts_service.create_alert(user_id=123, alert_config=alert_config)

# Evaluate user alerts
results = await alerts_service.evaluate_user_alerts(user_id=123)
```

## Migration from Old System

The old `src/telegram/services/alerts_eval_service.py` has been removed and replaced with this centralized system.

### Benefits of Consolidation
- ✅ Single source of truth for alert evaluation
- ✅ Better testability and maintainability
- ✅ Consistent behavior across all interfaces
- ✅ Proper separation of concerns
- ✅ Service layer abstraction
- ✅ Centralized schema validation
- ✅ Unified error handling and logging

### Integration Points
- **Telegram Bot**: Uses AlertsService for alert management commands
- **Web UI**: Can use AlertsService for alert management interface
- **Scheduler**: Uses AlertRunner for periodic alert evaluation
- **API**: Can expose AlertsService methods via REST endpoints

## Configuration

Alert configurations support:
- Price-based alerts (above/below thresholds)
- Technical indicator alerts (RSI, MACD, etc.)
- Complex rule trees with AND/OR logic
- Rearm conditions with hysteresis
- Multiple notification channels (Telegram, email)

See `schemas/` directory for detailed configuration schemas.
# Service Layer Integration Guide

## Overview

The Telegram bot module has been refactored to follow clean architecture principles by using proper service layers for all database operations and indicator calculations. This guide explains the integration patterns and best practices for working with the service layer.

## Architecture

### Service Layer Components

#### 1. Telegram Service (`src/data/db/services/telegram_service.py`)
Handles all telegram-related database operations:
- User management (registration, verification, approval)
- Alert management (create, update, delete, list)
- Schedule management (create, update, delete, list)
- Settings management (get/set configuration values)
- Audit logging (command tracking, performance metrics)

#### 2. Indicator Service (`src/indicators/service.py`)
Handles all technical and fundamental indicator calculations:
- Technical indicators (RSI, MACD, Bollinger Bands, SMA, etc.)
- Fundamental indicators (PE, PB, ROE, etc.)
- Batch processing for multiple tickers
- Error handling and fallback mechanisms

### Dependency Injection Pattern

The business logic layer uses dependency injection to receive service instances:

```python
class TelegramBusinessLogic:
    def __init__(self, telegram_service, indicator_service: Optional[IndicatorService] = None):
        """
        Initialize business logic with service dependencies.
        
        Args:
            telegram_service: Service for telegram-related database operations
            indicator_service: Service for technical and fundamental indicator calculations
        """
        self.telegram_service = telegram_service
        self.indicator_service = indicator_service or IndicatorService()
```

## Service Usage Patterns

### 1. Database Operations

#### User Management
```python
# Get user status
user_status = self.telegram_service.get_user_status(telegram_user_id)

# Update user settings
self.telegram_service.set_user_limit(telegram_user_id, "max_alerts", 10)

# Log command audit
self.telegram_service.log_command_audit(
    telegram_user_id, 
    command, 
    response_time=response_time,
    status="success"
)
```

#### Alert Management
```python
# Create alert
alert_id = self.telegram_service.add_alert(
    telegram_user_id=telegram_user_id,
    ticker=ticker,
    price=price,
    condition=condition
)

# List user alerts
alerts = self.telegram_service.list_alerts(telegram_user_id)

# Update alert
self.telegram_service.update_alert(alert_id, active=False)
```

#### Schedule Management
```python
# Create schedule
schedule_id = self.telegram_service.add_schedule(
    telegram_user_id=telegram_user_id,
    ticker=ticker,
    scheduled_time=scheduled_time,
    config=config
)

# List user schedules
schedules = self.telegram_service.list_schedules(telegram_user_id)
```

### 2. Indicator Calculations

#### Single Ticker Analysis
```python
# Create indicator request
request = TickerIndicatorsRequest(
    ticker="AAPL",
    indicators=["RSI", "MACD", "BollingerBands"],
    timeframe="1d",
    period="1y"
)

# Compute indicators
try:
    result = await self.indicator_service.compute_for_ticker(request)
    
    # Access technical indicators
    rsi_value = result.technical.get("RSI")
    macd_data = result.technical.get("MACD")
    
    # Access fundamental indicators
    pe_ratio = result.fundamental.get("PE")
    
except Exception as e:
    _logger.exception("Indicator calculation failed:")
    # Handle error gracefully
```

#### Batch Processing
```python
# For multiple tickers or complex analysis
indicators_config = {
    "RSI": {"period": 14},
    "MACD": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
}

results = self.indicator_service.compute(df, indicators_config, fund_params)
```

### 3. Error Handling Patterns

#### Service Layer Error Handling
```python
def safe_telegram_service_call(self, method, *args, **kwargs):
    """
    Safely call telegram service methods with error handling.
    """
    try:
        return method(*args, **kwargs)
    except Exception as e:
        _logger.exception("Telegram service error:")
        return None

# Usage
user_status = self.safe_telegram_service_call(
    self.telegram_service.get_user_status,
    telegram_user_id
)
```

#### Indicator Service Error Handling
```python
async def safe_indicator_calculation(self, request: TickerIndicatorsRequest):
    """
    Safely calculate indicators with fallback behavior.
    """
    try:
        return await self.indicator_service.compute_for_ticker(request)
    except Exception as e:
        _logger.warning("Indicator calculation failed for %s: %s", request.ticker, e)
        # Return empty result or use fallback data
        return IndicatorResultSet(
            ticker=request.ticker,
            technical={},
            fundamental={}
        )
```

## Integration Examples

### 1. Command Handler Integration

```python
def handle_report(self, parsed_command: ParsedCommand) -> Dict[str, Any]:
    """
    Handle report command using service layer integration.
    """
    try:
        # Get user status through service layer
        user_status = self.telegram_service.get_user_status(parsed_command.telegram_user_id)
        if not user_status or not user_status.get("approved"):
            return {"status": "error", "message": "Access denied"}
        
        # Create indicator request
        request = TickerIndicatorsRequest(
            ticker=parsed_command.args.get("tickers"),
            indicators=parsed_command.args.get("indicators", []),
            timeframe=parsed_command.args.get("interval", "1d"),
            period=parsed_command.args.get("period", "1y")
        )
        
        # Calculate indicators through service layer
        result = await self.indicator_service.compute_for_ticker(request)
        
        # Log command audit through service layer
        self.telegram_service.log_command_audit(
            parsed_command.telegram_user_id,
            "report",
            ticker=request.ticker,
            status="success"
        )
        
        return {"status": "success", "data": result}
        
    except Exception as e:
        _logger.exception("Report generation failed:")
        return {"status": "error", "message": "Report generation failed"}
```

### 2. Background Service Integration

```python
class AlertMonitor:
    def __init__(self, api_client: BotHttpApiClient = None, telegram_service=None):
        self.api_client = api_client
        self.telegram_service = telegram_service or get_telegram_service()
        self.indicator_service = IndicatorService()
    
    async def check_alerts(self):
        """
        Check alerts using service layer integration.
        """
        # Get active alerts through service layer
        alerts = self.telegram_service.get_active_alerts()
        
        for alert in alerts:
            try:
                # Calculate current indicators
                request = TickerIndicatorsRequest(
                    ticker=alert["ticker"],
                    indicators=["RSI"],  # or based on alert config
                    timeframe="15m",
                    period="1d"
                )
                
                result = await self.indicator_service.compute_for_ticker(request)
                
                # Evaluate alert condition
                if self.evaluate_alert_condition(alert, result):
                    # Update alert status through service layer
                    self.telegram_service.update_alert(
                        alert["id"], 
                        status="TRIGGERED",
                        last_trigger_time=datetime.now(timezone.utc)
                    )
                    
                    # Send notification
                    await self.send_alert_notification(alert, result)
                    
            except Exception as e:
                _logger.exception("Alert evaluation failed for %s:", alert["id"])
```

## Best Practices

### 1. Service Initialization
- Always use dependency injection for service instances
- Provide fallback initialization for backward compatibility
- Handle service initialization errors gracefully

### 2. Error Handling
- Wrap service calls in try-catch blocks
- Log errors with appropriate context
- Provide user-friendly error messages
- Implement fallback behavior when possible

### 3. Performance Considerations
- Use batch processing for multiple operations
- Cache service instances when appropriate
- Implement async patterns for I/O operations
- Monitor service call performance

### 4. Testing
- Mock service dependencies in unit tests
- Test error handling scenarios
- Verify service integration contracts
- Use dependency injection for test isolation

## Migration from Direct Access

### Before (Direct Database Access)
```python
# OLD PATTERN - DO NOT USE
import sqlite3
from src.data.db.models import User

def get_user_info(telegram_user_id):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE telegram_user_id = ?", (telegram_user_id,))
    result = cursor.fetchone()
    conn.close()
    return result
```

### After (Service Layer)
```python
# NEW PATTERN - USE THIS
def get_user_info(self, telegram_user_id):
    return self.telegram_service.get_user_status(telegram_user_id)
```

### Before (Direct Indicator Calculation)
```python
# OLD PATTERN - DO NOT USE
import talib

def calculate_rsi(data, period=14):
    return talib.RSI(data['close'].values, timeperiod=period)
```

### After (Service Layer)
```python
# NEW PATTERN - USE THIS
async def calculate_indicators(self, ticker, indicators):
    request = TickerIndicatorsRequest(
        ticker=ticker,
        indicators=indicators,
        timeframe="1d",
        period="1y"
    )
    return await self.indicator_service.compute_for_ticker(request)
```

## Service Layer Benefits

1. **Separation of Concerns**: Business logic is separated from data access and calculations
2. **Testability**: Easy to mock service dependencies for unit testing
3. **Maintainability**: Changes to database schema or calculation methods don't affect business logic
4. **Consistency**: All modules use the same service interfaces
5. **Error Handling**: Centralized error handling and logging
6. **Performance**: Service layer can implement caching and optimization
7. **Scalability**: Service layer can be scaled independently

## Troubleshooting

### Common Issues

1. **Service Not Initialized**: Ensure service instances are properly injected
2. **Import Errors**: Use absolute imports from project root
3. **Async/Await**: Remember to use async/await for indicator service calls
4. **Error Handling**: Always wrap service calls in try-catch blocks
5. **Type Hints**: Use proper type hints for service interfaces

### Debugging Tips

1. Enable debug logging to see service layer calls
2. Check service initialization in bot startup
3. Verify service method signatures and parameters
4. Test service layer integration with unit tests
5. Monitor service performance and error rates

## Future Enhancements

1. **Service Discovery**: Automatic service registration and discovery
2. **Circuit Breaker**: Implement circuit breaker pattern for service failures
3. **Metrics**: Add service layer metrics and monitoring
4. **Caching**: Implement distributed caching for service responses
5. **Load Balancing**: Support for multiple service instances
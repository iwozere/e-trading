# Design Document

## Purpose

This design document outlines the architectural refactoring of the telegram bot module to eliminate direct database access and indicator calculations, ensuring all operations go through proper service layers. The refactoring will improve maintainability, testability, and adherence to clean architecture principles.

## Architecture

### Current Architecture Issues

The current telegram module has several architectural violations:

1. **Direct Database Access**: Files like `business_logic.py` contain raw SQL queries and direct database connections
2. **Embedded Indicator Calculations**: `indicator_calculator.py` duplicates functionality available in `src/indicators`
3. **Tight Coupling**: Business logic is tightly coupled to database models and calculation implementations
4. **Testing Difficulties**: Direct dependencies make unit testing complex and require database setup

### Target Architecture

The refactored architecture will follow a clean layered approach:

```
┌─────────────────────────────────────────┐
│           Telegram Bot Layer            │
│  (bot.py, command_parser.py, etc.)     │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Business Logic Layer           │
│     (business_logic.py, etc.)          │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│           Service Layer                 │
│  ┌─────────────────┐ ┌─────────────────┐│
│  │ telegram_service│ │ IndicatorService││
│  │                 │ │                 ││
│  └─────────────────┘ └─────────────────┘│
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│          Data Access Layer              │
│     (database_service, models)          │
└─────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Service Layer Integration

#### Database Service Integration
- **telegram_service.py**: Already exists and provides the interface we need
- **Key Methods**:
  - `get_user_status(telegram_user_id)` - User information retrieval
  - `set_user_limit(telegram_user_id, key, value)` - User limit management
  - `log_command_audit(telegram_user_id, command, **kwargs)` - Command logging
  - `add_feedback(telegram_user_id, type_, message)` - Feedback management

#### Indicator Service Integration
- **IndicatorService**: Located in `src/indicators/service.py`
- **Key Methods**:
  - `compute_for_ticker(req: TickerIndicatorsRequest)` - Compute indicators for a ticker
  - `compute(df, config, fund_params)` - Batch indicator computation
- **Models**:
  - `TickerIndicatorsRequest` - Request specification
  - `IndicatorResultSet` - Response with technical and fundamental indicators

### 2. Refactored Components

#### Business Logic Layer (`business_logic.py`)
```python
class TelegramBusinessLogic:
    def __init__(self, telegram_service, indicator_service):
        self.telegram_service = telegram_service
        self.indicator_service = indicator_service
    
    async def handle_report(self, parsed_command):
        # Use indicator_service for calculations
        # Use telegram_service for user data and logging
    
    def handle_user_settings(self, telegram_user_id, settings):
        # Use telegram_service for all database operations
```

#### Indicator Calculator Replacement
- **Remove**: `src/telegram/screener/indicator_calculator.py`
- **Replace with**: Service calls to `src/indicators/service.py`
- **Migration Strategy**: Map existing indicator calls to service requests

#### Database Access Elimination
- **Remove**: Direct `sqlite3` imports and raw SQL queries
- **Replace with**: Service layer method calls
- **Files to modify**:
  - `business_logic.py` - Remove raw SQL, use telegram_service
  - `rearm_alert_system.py` - Use service layer for updates
  - Test files - Use service mocks instead of direct DB

### 3. Service Interface Contracts

#### Telegram Service Interface
```python
# User Management
def get_user_status(telegram_user_id: str) -> Optional[Dict[str, Any]]
def set_user_limit(telegram_user_id: str, key: str, value: int) -> None

# Settings Management  
def get_setting(key: str) -> Optional[str]
def set_setting(key: str, value: Optional[str]) -> None

# Feedback and Audit
def add_feedback(telegram_user_id: str, type_: str, message: str) -> int
def log_command_audit(telegram_user_id: str, command: str, **kwargs) -> int
```

#### Indicator Service Interface
```python
async def compute_for_ticker(req: TickerIndicatorsRequest) -> IndicatorResultSet

# Request Model
@dataclass
class TickerIndicatorsRequest:
    ticker: str
    indicators: List[str]  # ["RSI", "MACD", "SMA", "BollingerBands"]
    timeframe: str = "1d"
    period: str = "1y"
    provider: Optional[str] = None

# Response Model
@dataclass  
class IndicatorResultSet:
    ticker: str
    technical: Dict[str, IndicatorValue]
    fundamental: Dict[str, IndicatorValue]
```

## Data Models

### Service Request/Response Models

#### Indicator Request Models
```python
@dataclass
class TickerIndicatorsRequest:
    ticker: str
    indicators: List[str]
    timeframe: str = "1d"
    period: str = "1y" 
    provider: Optional[str] = None

@dataclass
class IndicatorValue:
    name: str
    value: Optional[float]

@dataclass
class IndicatorResultSet:
    ticker: str
    technical: Dict[str, IndicatorValue]
    fundamental: Dict[str, IndicatorValue]
```

#### User Management Models
```python
@dataclass
class UserStatus:
    approved: bool
    verified: bool
    email: Optional[str]
    language: str
    is_admin: bool
    max_alerts: int
    max_schedules: int
```

### Data Flow Patterns

#### Report Generation Flow
```
1. User Request → Command Parser
2. Business Logic → IndicatorService.compute_for_ticker()
3. IndicatorService → Data Providers (Yahoo, Binance, etc.)
4. Results → Business Logic → Response Formatter
5. Response → Notification Manager → User
```

#### User Settings Flow
```
1. User Settings Change → Business Logic
2. Business Logic → telegram_service.set_user_limit()
3. telegram_service → database_service
4. Confirmation → User
```

## Error Handling

### Service Layer Error Handling
- **Database Errors**: Handled by `database_service` and propagated through `telegram_service`
- **Indicator Errors**: Handled by `IndicatorService` with graceful degradation
- **Network Errors**: Handled by data providers with retry logic
- **Validation Errors**: Handled at business logic layer with user-friendly messages

### Error Propagation Strategy
```python
try:
    result = await self.indicator_service.compute_for_ticker(request)
except IndicatorCalculationError as e:
    _logger.warning("Indicator calculation failed: %s", e)
    return {"status": "partial", "message": "Some indicators unavailable"}
except DataProviderError as e:
    _logger.exception("Data provider error:")
    return {"status": "error", "message": "Unable to fetch market data"}
```

## Testing Strategy

### Unit Testing Approach
- **Service Mocking**: Mock `telegram_service` and `indicator_service` dependencies
- **Dependency Injection**: Inject service instances for easy testing
- **Isolated Testing**: Test business logic without database or external API dependencies

### Test Structure
```python
class TestTelegramBusinessLogic:
    def setup_method(self):
        self.mock_telegram_service = Mock()
        self.mock_indicator_service = Mock()
        self.business_logic = TelegramBusinessLogic(
            self.mock_telegram_service,
            self.mock_indicator_service
        )
    
    async def test_handle_report_success(self):
        # Setup mocks
        self.mock_indicator_service.compute_for_ticker.return_value = mock_result
        
        # Test business logic
        result = await self.business_logic.handle_report(mock_command)
        
        # Verify service calls
        self.mock_indicator_service.compute_for_ticker.assert_called_once()
```

### Integration Testing
- **Service Integration**: Test actual service layer integration
- **Database Integration**: Test with real database using service layer
- **End-to-End**: Test complete command flow through service layers

## Migration Strategy

### Phase 1: Service Layer Integration
1. Modify business logic to accept service dependencies
2. Update initialization to inject service instances
3. Replace direct database calls with service calls

### Phase 2: Indicator Service Integration  
1. Replace `indicator_calculator.py` usage with `IndicatorService`
2. Update screener modules to use service layer
3. Remove duplicate indicator calculation code

### Phase 3: Testing and Validation
1. Update unit tests to use service mocks
2. Add integration tests for service layer usage
3. Validate functionality with existing test cases

### Phase 4: Cleanup
1. Remove unused direct database imports
2. Remove duplicate indicator calculation files
3. Update documentation and examples

## Performance Considerations

### Service Layer Overhead
- **Minimal Impact**: Service layer adds minimal overhead compared to direct access
- **Caching**: Service layer can implement caching for frequently accessed data
- **Connection Pooling**: Database service handles connection management efficiently

### Indicator Calculation Optimization
- **Batch Processing**: Use `IndicatorService.compute()` for multiple indicators
- **Async Processing**: Leverage async capabilities of `IndicatorService`
- **Result Caching**: Service layer can cache indicator results

## Security Considerations

### Data Access Control
- **Service Layer Validation**: All data access goes through validated service methods
- **User Authorization**: Service layer enforces user permissions and limits
- **Input Sanitization**: Service layer handles input validation and sanitization

### Error Information Disclosure
- **Controlled Error Messages**: Service layer provides user-appropriate error messages
- **Logging**: Detailed errors logged securely without exposing to users
- **Audit Trail**: All operations logged through service layer audit functions
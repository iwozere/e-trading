# Strategy Framework Design Document

## 1. System Architecture

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Strategy Framework                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Custom    │  │  HMM-LSTM   │  │   Other     │        │
│  │  Strategy   │  │  Strategy   │  │ Strategies  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                │                │               │
│         └────────────────┼────────────────┘               │
│                          │                                │
│  ┌─────────────────────────────────────────────────────────┤
│  │              BaseStrategy (bt.Strategy)                 │
│  │  ┌─────────────────┐  ┌─────────────────┐              │
│  │  │   Trade         │  │   Position      │              │
│  │  │   Tracking      │  │   Management    │              │
│  │  └─────────────────┘  └─────────────────┘              │
│  │  ┌─────────────────┐  ┌─────────────────┐              │
│  │  │   Performance   │  │   Database      │              │
│  │  │   Monitoring    │  │   Integration   │              │
│  │  └─────────────────┘  └─────────────────┘              │
│  └─────────────────────────────────────────────────────────┤
│                          │                                │
│  ┌─────────────────────────────────────────────────────────┤
│  │              Mixin System                               │
│  │  ┌─────────────┐              ┌─────────────┐          │
│  │  │   Entry     │              │    Exit     │          │
│  │  │   Mixins    │              │   Mixins    │          │
│  │  │             │              │             │          │
│  │  │ • RSI/BB    │              │ • ATR       │          │
│  │  │ • Volume    │              │ • Trailing  │          │
│  │  │ • ML        │              │ • Time      │          │
│  │  │ • Custom    │              │ • Custom    │          │
│  │  └─────────────┘              └─────────────┘          │
│  └─────────────────────────────────────────────────────────┤
│                          │                                │
│  ┌─────────────────────────────────────────────────────────┤
│  │              Data Layer                                 │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  │   Trade     │  │    Bot      │  │ Performance │    │
│  │  │ Repository  │  │ Instance    │  │  Metrics    │    │
│  │  │             │  │ Repository  │  │ Repository  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │
│  └─────────────────────────────────────────────────────────┤
│                          │                                │
│  ┌─────────────────────────────────────────────────────────┤
│  │              Database Layer                             │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  │    Trade    │  │   Bot       │  │ Performance │    │
│  │  │   Table     │  │ Instance    │  │  Metrics    │    │
│  │  │             │  │   Table     │  │   Table     │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │
│  └─────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Component Relationships

```
BaseStrategy
├── Trade Tracking
│   ├── Entry/Exit Recording
│   ├── PnL Calculation
│   ├── Performance Metrics
│   └── Partial Exit Handling
├── Position Management
│   ├── Size Validation
│   ├── Asset Type Detection
│   ├── Risk Management
│   └── State Tracking
├── Database Integration
│   ├── Trade Persistence
│   ├── Bot Instance Management
│   ├── Performance Analytics
│   └── Partial Exit Tracking
└── Mixin System
    ├── Entry Mixins
    │   ├── Signal Generation
    │   ├── Confidence Scoring
    │   └── Parameter Management
    └── Exit Mixins
        ├── Exit Signal Generation
        ├── State Management
        └── Partial Exit Support
```

## 2. Core Components Design

### 2.1 BaseStrategy Class

#### 2.1.1 Enhanced Trade Tracking

The BaseStrategy includes comprehensive trade tracking with partial exit support:

- **Dynamic Position Size Tracking**: Uses `current_position_size` to track remaining position after partial exits
- **Asset Type Validation**: Validates position sizes based on asset type (stocks vs crypto)
- **Entry Price Management**: Proper tracking of entry prices throughout trade lifecycle
- **Partial Exit Support**: Handles partial exits with proper size and relationship tracking

#### 2.1.2 Class Hierarchy
```python
class BaseStrategy(bt.Strategy):
    """
    Base class for all trading strategies.
    
    Provides:
    - Trade tracking and management
    - Position sizing and validation
    - Performance monitoring
    - Database integration
    - Partial exit support
    """
    
    def __init__(self):
        # Configuration management
        self.config = {}
        self.symbol = ""
        self.timeframe = ""
        self.asset_type = ""
        
        # Trade tracking
        self.current_trade = None
        self.entry_price = None
        self.current_position_size = None
        self.current_position_id = None
        
        # Database integration
        self.trade_repository = None
        self.bot_instance_id = None
        self.enable_database_logging = False
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
```

#### 2.1.2 Key Methods
```python
def _enter_position(self, direction: str, confidence: float = 1.0, 
                   risk_multiplier: float = 1.0, reason: str = ""):
    """Enter a new position with validation and tracking."""
    
def _exit_position(self, reason: str = ""):
    """Exit current position completely."""
    
def _exit_partial_position(self, exit_size: float, reason: str = ""):
    """Exit a partial position with proper tracking."""
    
def _validate_position_size(self, shares: float) -> bool:
    """Validate position size based on asset type."""
    
def notify_trade(self, trade):
    """Handle trade notifications and update metrics."""
    
def _store_trade_in_database(self, trade_record: Dict[str, Any], 
                           is_partial_exit: bool = False):
    """Store trade in database with proper partial exit handling."""
```

### 2.2 Mixin System Design

#### 2.2.1 Entry Mixin Interface
```python
class BaseEntryMixin(ABC):
    """Base class for all entry mixins."""
    
    @abstractmethod
    def should_enter(self) -> bool:
        """Determine if entry signal is present."""
        
    @abstractmethod
    def get_confidence(self) -> float:
        """Get confidence score for entry signal."""
        
    @abstractmethod
    def get_required_params(self) -> List[str]:
        """Get list of required parameters."""
        
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameter values."""
```

#### 2.2.2 Exit Mixin Interface
```python
class BaseExitMixin(ABC):
    """Base class for all exit mixins."""
    
    @abstractmethod
    def should_exit(self) -> bool:
        """Check if position should be exited."""
        
    @abstractmethod
    def get_exit_reason(self) -> str:
        """Get the reason for exit."""
        
    @abstractmethod
    def get_required_params(self) -> List[str]:
        """Get list of required parameters."""
        
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameter values."""
```

### 2.3 Database Design

#### 2.3.1 TradeRepository Integration

The TradeRepository provides a clean abstraction layer for database operations:

```python
class TradeRepository:
    """Repository for trade-related database operations."""
    
    def create_trade(self, trade_data: Dict[str, Any]) -> Trade:
        """Create a new trade record."""
        
    def create_partial_exit_trade(self, trade_data: Dict[str, Any], 
                                parent_trade_id: str) -> Trade:
        """Create a partial exit trade linked to parent trade."""
        
    def get_position_summary(self, position_id: str) -> Dict[str, Any]:
        """Get complete position summary including all partial exits."""
        
    def get_trades_by_position(self, position_id: str) -> List[Trade]:
        """Get all trades for a specific position."""
```

#### 2.3.2 Partial Exit Support

The database schema supports partial exits with proper relationships:

- **Position ID**: Groups related trades (original + partial exits)
- **Parent Trade ID**: Links partial exits to original position
- **Sequence Number**: Tracks order of partial exits
- **Remaining Size**: Tracks remaining position after each partial exit

#### 2.3.3 Trade Table Schema
```sql
CREATE TABLE trades (
    -- Primary identification
    id VARCHAR(36) PRIMARY KEY,
    
    -- Bot/Config identification
    bot_id VARCHAR(255) NOT NULL,
    trade_type VARCHAR(10) NOT NULL,  -- 'paper', 'live', 'optimization'
    
    -- Strategy identification
    strategy_name VARCHAR(100),
    entry_logic_name VARCHAR(100) NOT NULL,
    exit_logic_name VARCHAR(100) NOT NULL,
    
    -- Trade identification
    symbol VARCHAR(20) NOT NULL,
    interval VARCHAR(10) NOT NULL,
    
    -- Trade timing
    entry_time DATETIME,
    exit_time DATETIME,
    
    -- Trade details
    entry_price DECIMAL(20, 8),
    exit_price DECIMAL(20, 8),
    size DECIMAL(20, 8),
    direction VARCHAR(10) NOT NULL,  -- 'long', 'short'
    
    -- Partial exit tracking
    original_position_size DECIMAL(20, 8),
    partial_exit_sequence INTEGER,
    parent_trade_id VARCHAR(36),
    remaining_position_size DECIMAL(20, 8),
    is_partial_exit BOOLEAN DEFAULT FALSE,
    
    -- Position tracking
    position_id VARCHAR(36),
    total_position_pnl DECIMAL(20, 8),
    
    -- Financial calculations
    commission DECIMAL(20, 8),
    gross_pnl DECIMAL(20, 8),
    net_pnl DECIMAL(20, 8),
    pnl_percentage DECIMAL(10, 4),
    
    -- Trade metadata
    exit_reason VARCHAR(100),
    status VARCHAR(20) NOT NULL,  -- 'open', 'closed', 'cancelled'
    extra_metadata JSON,
    
    -- System fields
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### 2.3.2 Bot Instance Table Schema
```sql
CREATE TABLE bot_instances (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(20) NOT NULL,  -- 'paper', 'live', 'optimization'
    status VARCHAR(20) NOT NULL,  -- 'running', 'stopped', 'error'
    strategy_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    config JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### 2.3.3 Performance Metrics Table Schema
```sql
CREATE TABLE performance_metrics (
    id VARCHAR(36) PRIMARY KEY,
    bot_id VARCHAR(36) NOT NULL,
    calculated_at DATETIME NOT NULL,
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    win_rate DECIMAL(5, 2),
    total_pnl DECIMAL(20, 8),
    max_drawdown DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    metrics_data JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## 3. Design Patterns

### 3.1 Repository Pattern

The Repository pattern is used for database operations to provide a clean abstraction layer:

```python
class TradeRepository:
    """Repository for trade-related database operations."""
    
    def create_trade(self, trade_data: Dict[str, Any]) -> Trade:
        """Create a new trade record."""
        
    def create_partial_exit_trade(self, trade_data: Dict[str, Any], 
                                parent_trade_id: str) -> Trade:
        """Create a partial exit trade linked to parent trade."""
        
    def get_position_summary(self, position_id: str) -> Dict[str, Any]:
        """Get complete position summary including all partial exits."""
        
    def get_trades_by_position(self, position_id: str) -> List[Trade]:
        """Get all trades for a specific position."""
```

### 3.2 Factory Pattern

The Factory pattern is used for creating mixin instances:

```python
class EntryMixinFactory:
    """Factory for creating entry mixin instances."""
    
    @staticmethod
    def create_mixin(mixin_name: str, params: Dict[str, Any]) -> BaseEntryMixin:
        """Create an entry mixin instance."""
        
class ExitMixinFactory:
    """Factory for creating exit mixin instances."""
    
    @staticmethod
    def create_mixin(mixin_name: str, params: Dict[str, Any]) -> BaseExitMixin:
        """Create an exit mixin instance."""
```

### 3.3 Strategy Pattern

The Strategy pattern is used for different trading strategies:

```python
class CustomStrategy(BaseStrategy):
    """Configurable strategy using entry/exit mixins."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.entry_mixins = []
        self.exit_mixins = []
        self._initialize_mixins()
    
    def _initialize_mixins(self):
        """Initialize entry and exit mixins based on configuration."""
        
    def next(self):
        """Main strategy logic executed on each bar."""
```

### 3.4 Observer Pattern

The Observer pattern is used for trade notifications:

```python
class BaseStrategy(bt.Strategy):
    def notify_trade(self, trade):
        """Handle trade notifications from Backtrader."""
        # Update internal state
        # Store in database
        # Update performance metrics
        # Log trade information
```

## 4. Data Flow Design

### 4.1 Trade Execution Flow

```
1. Strategy.next() called
   ↓
2. Entry mixins check for signals
   ↓
3. If entry signal found:
   - Validate position size
   - Create position
   - Set entry price
   - Generate position ID
   - Store in database
   ↓
4. Exit mixins check for exit signals
   ↓
5. If exit signal found:
   - Determine if partial or full exit
   - Execute exit order
   - Calculate PnL
   - Update position tracking
   - Store trade in database
   - Update performance metrics
```

### 4.2 Partial Exit Flow

```
1. Partial exit signal detected
   ↓
2. Validate exit size
   ↓
3. Execute partial exit order
   ↓
4. Update current position size
   ↓
5. Create partial exit trade record
   ↓
6. Link to parent trade via position_id
   ↓
7. Update parent trade remaining size
   ↓
8. Store in database with sequence number
   ↓
9. Continue monitoring for next exit
```

### 4.3 Database Integration Flow

```
1. Trade event occurs
   ↓
2. Create trade record
   ↓
3. Determine if partial exit
   ↓
4. If partial exit:
   - Get parent trade
   - Calculate sequence number
   - Update parent trade
   - Store partial exit record
   ↓
5. If full exit:
   - Store complete trade record
   - Reset position tracking
   ↓
6. Update performance metrics
   ↓
7. Commit transaction
```

## 5. Error Handling Design

### 5.1 Exception Hierarchy

```python
class StrategyFrameworkError(Exception):
    """Base exception for strategy framework."""
    pass

class TradeValidationError(StrategyFrameworkError):
    """Raised when trade validation fails."""
    pass

class PositionSizeError(StrategyFrameworkError):
    """Raised when position size validation fails."""
    pass

class DatabaseError(StrategyFrameworkError):
    """Raised when database operations fail."""
    pass

class MixinError(StrategyFrameworkError):
    """Raised when mixin operations fail."""
    pass
```

### 5.2 Error Handling Strategy

```python
def _enter_position(self, direction: str, confidence: float = 1.0, 
                   risk_multiplier: float = 1.0, reason: str = ""):
    """Enter a new position with comprehensive error handling."""
    try:
        # Validate inputs
        if not direction or direction.lower() not in ['long', 'short']:
            raise TradeValidationError(f"Invalid direction: {direction}")
        
        # Calculate position size
        position_size = self._calculate_position_size(confidence, risk_multiplier)
        shares = self._calculate_shares(position_size)
        
        # Validate position size
        if not self._validate_position_size(shares):
            raise PositionSizeError(f"Invalid position size: {shares}")
        
        # Execute trade
        if direction.lower() == 'long':
            order = self.buy(size=shares)
        else:
            order = self.sell(size=shares)
        
        # Update tracking
        if order:
            self.entry_price = self.data.close[0]
            self.current_position_size = abs(shares)
            self.highest_profit = 0.0
            
    except TradeValidationError as e:
        _logger.error("Trade validation error: %s", e)
        raise
    except PositionSizeError as e:
        _logger.error("Position size error: %s", e)
        raise
    except Exception as e:
        _logger.exception("Unexpected error in _enter_position: %s", e)
        raise StrategyFrameworkError(f"Failed to enter position: {e}")
```

## 6. Performance Design

### 6.1 Database Optimization

#### 6.1.1 Indexing Strategy
```sql
-- Primary indexes for frequent queries
CREATE INDEX idx_trades_symbol_status ON trades(symbol, status);
CREATE INDEX idx_trades_bot_id ON trades(bot_id);
CREATE INDEX idx_trades_position_id ON trades(position_id);
CREATE INDEX idx_trades_entry_time ON trades(entry_time);
CREATE INDEX idx_trades_is_partial_exit ON trades(is_partial_exit);

-- Composite indexes for complex queries
CREATE INDEX idx_trades_bot_symbol_time ON trades(bot_id, symbol, entry_time);
CREATE INDEX idx_trades_position_sequence ON trades(position_id, partial_exit_sequence);
```

#### 6.1.2 Query Optimization
```python
def get_trades_by_position(self, position_id: str) -> List[Trade]:
    """Optimized query for position trades."""
    return (self.session.query(Trade)
            .filter(Trade.position_id == position_id)
            .order_by(Trade.partial_exit_sequence.asc().nullsfirst())
            .all())

def get_position_summary(self, position_id: str) -> Dict[str, Any]:
    """Optimized query for position summary."""
    # Use single query with joins instead of multiple queries
    result = (self.session.query(Trade)
              .filter(Trade.position_id == position_id)
              .all())
    
    # Process results in memory for better performance
    return self._process_position_summary(result)
```

### 6.2 Memory Management

#### 6.2.1 Trade Record Management
```python
class BaseStrategy(bt.Strategy):
    def __init__(self):
        # Limit in-memory trade storage
        self.max_trades_in_memory = 1000
        self.trades = []
        
    def _cleanup_old_trades(self):
        """Clean up old trades from memory."""
        if len(self.trades) > self.max_trades_in_memory:
            # Keep only recent trades in memory
            self.trades = self.trades[-self.max_trades_in_memory:]
```

#### 6.2.2 Database Connection Management
```python
class TradeRepository:
    def __init__(self, session: Session = None):
        self.session = session or get_session()
        self._owns_session = session is None
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.close()
        
    def close(self):
        """Close the database session."""
        if self._owns_session and self.session:
            close_session(self.session)
            self.session = None
```

## 7. Security Design

### 7.1 Input Validation

```python
def _validate_position_size(self, shares: float) -> bool:
    """Comprehensive position size validation."""
    # Type validation
    if not isinstance(shares, (int, float)):
        raise TypeError(f"Position size must be numeric, got {type(shares)}")
    
    # Range validation
    if shares <= 0:
        raise ValueError(f"Position size must be positive, got {shares}")
    
    # Asset-specific validation
    if self.asset_type.lower() == "stock":
        if not shares.is_integer():
            raise ValueError(f"Stock position size must be whole number, got {shares}")
        if shares < 1:
            raise ValueError(f"Stock position size must be >= 1, got {shares}")
    
    return True
```

### 7.2 Database Security

```python
def create_trade(self, trade_data: Dict[str, Any]) -> Trade:
    """Create trade with input sanitization."""
    try:
        # Sanitize input data
        sanitized_data = self._sanitize_trade_data(trade_data)
        
        # Validate required fields
        self._validate_trade_data(sanitized_data)
        
        # Create trade object
        trade = Trade(**sanitized_data)
        self.session.add(trade)
        self.commit()
        
        return trade
        
    except Exception as e:
        self.rollback()
        _logger.exception("Error creating trade: %s", e)
        raise
```

## 8. Database Logging Configuration

### 8.1 Process-Level Configuration

The database logging system is configured at the process level, not per strategy:

- **Backtester Processes**: `enable_database_logging = false` (default)
- **Paper Trading Processes**: `enable_database_logging = true`
- **Live Trading Processes**: `enable_database_logging = true`

### 8.2 Bot Type Classification

Trades are automatically classified by bot type in the database:

- **`optimization`**: Backtesting and optimization runs
- **`paper`**: Paper trading simulations
- **`live`**: Live trading with real money

### 8.3 Configuration Examples

#### Backtester Configuration
```json
{
    "backtester_settings": {
        "enable_database_logging": false,
        "bot_type": "optimization"
    }
}
```

#### Paper Trading Configuration
```json
{
    "trading_settings": {
        "enable_database_logging": true,
        "bot_type": "paper"
    }
}
```

#### Live Trading Configuration
```json
{
    "trading_settings": {
        "enable_database_logging": true,
        "bot_type": "live"
    }
}
```

### 8.4 Performance Impact

- **Backtesting**: ~2-3x faster execution with database logging disabled
- **Live Trading**: Full audit trail with all trades stored in database
- **Paper Trading**: Complete trade tracking for analysis and testing

## 9. Testing Design

### 9.1 Unit Testing Strategy

```python
class TestBaseStrategy(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.strategy = BaseStrategy()
        self.strategy.config = {
            'enable_database_logging': False,
            'position_size': 0.1
        }
        
    def test_position_size_validation(self):
        """Test position size validation."""
        # Test crypto validation
        self.strategy.asset_type = 'crypto'
        self.assertTrue(self.strategy._validate_position_size(0.1))
        self.assertFalse(self.strategy._validate_position_size(0))
        
        # Test stock validation
        self.strategy.asset_type = 'stock'
        self.assertTrue(self.strategy._validate_position_size(1))
        self.assertFalse(self.strategy._validate_position_size(0.5))
        
    def test_partial_exit_tracking(self):
        """Test partial exit tracking."""
        # Setup initial position
        self.strategy.current_position_size = 0.1
        self.strategy.current_position_id = "test-position-id"
        
        # Test partial exit
        self.strategy._exit_partial_position(0.05, "test_reason")
        
        # Verify tracking
        self.assertEqual(self.strategy.current_position_size, 0.05)
```

### 8.2 Integration Testing Strategy

```python
class TestTradeRepositoryIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test database."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.repository = TradeRepository()
        
    def test_partial_exit_workflow(self):
        """Test complete partial exit workflow."""
        # Create original trade
        original_trade = self.repository.create_trade({
            'bot_id': 'test-bot',
            'symbol': 'BTCUSDT',
            'size': 0.1,
            'direction': 'long',
            'status': 'closed'
        })
        
        # Create partial exit
        partial_exit = self.repository.create_partial_exit_trade({
            'bot_id': 'test-bot',
            'symbol': 'BTCUSDT',
            'size': 0.05,
            'direction': 'long',
            'status': 'closed'
        }, original_trade.id)
        
        # Verify relationship
        self.assertEqual(partial_exit.parent_trade_id, original_trade.id)
        self.assertEqual(partial_exit.partial_exit_sequence, 1)
        
    def tearDown(self):
        """Clean up test database."""
        os.unlink(self.temp_db.name)
```

## 9. Deployment Design

### 9.1 Configuration Management

```python
# Default configuration
DEFAULT_CONFIG = {
    'enable_database_logging': False,
    'bot_type': 'paper',
    'position_size': 0.1,
    'max_position_size': 0.2,
    'min_position_size': 0.05,
    'entry_mixins': [],
    'exit_mixins': [],
    'bot_instance_name': None
}

# Environment-specific configurations
ENVIRONMENT_CONFIGS = {
    'development': {
        'enable_database_logging': True,
        'bot_type': 'paper'
    },
    'production': {
        'enable_database_logging': True,
        'bot_type': 'live'
    },
    'testing': {
        'enable_database_logging': False,
        'bot_type': 'paper'
    }
}
```

### 9.2 Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Run application
CMD ["python", "src/strategy/main.py"]
```

---

*This design document provides the architectural foundation for the strategy framework. It should be updated as the system evolves and new requirements are identified.*

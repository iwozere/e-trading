# Design

## Purpose

The Telegram Screener Bot module provides a comprehensive Telegram-based interface for financial market data analysis and reporting. It serves as the user-facing frontend for the e-trading platform, enabling users to request market reports, set price alerts, schedule recurring reports, and manage their trading data preferences through a conversational interface.

**Core Objectives:**
- Democratize access to financial market data through familiar messaging interface
- Provide real-time and scheduled market analysis capabilities
- Enable automated alerting for price movements and market conditions
- Support multi-user environment with secure authentication and data isolation
- Integrate seamlessly with the platform's data and notification infrastructure

## Architecture

### High-Level System Architecture

The screener module follows a layered, event-driven architecture with clear separation between user interface, business logic, and data access:

```
┌─────────────────────────────────────────────────────────────┐
│                     Telegram Client                         │
│               (User Interface Layer)                        │
└─────────────────────┬───────────────────────────────────────┘
                      │ Telegram Bot API
┌─────────────────────▼───────────────────────────────────────┐
│                 Bot Framework Layer                         │
│     ┌─────────────────┐    ┌─────────────────────────────┐  │
│     │   Command       │    │      Message                │  │
│     │   Handlers      │    │      Routing                │  │
│     │   (bot.py)      │    │   (aiogram dispatcher)      │  │
│     └─────────────────┘    └─────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │ Parsed Commands
┌─────────────────────▼────────────────────────────────────────┐
│              Command Processing Layer                        │
│  ┌─────────────────┐  ┌─────────────────┐ ┌────────────────┐ │
│  │ Command Parser  │  │ Business Logic  │ │ Notification   │ │
│  │ (command_parser)│  │ (business_logic)│ │ (notifications)│ │
│  └─────────────────┘  └─────────────────┘ └────────────────┘ │
└─────────────┬─────────────────┬─────────────────┬────────────┘
              │                 │                 │
┌─────────────▼─────────────────▼─────────────────▼───────────┐
│                   Service Layer                             │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│ │  Database   │ │ Data Module │ │   Notification Manager  │ │
│ │   (db.py)   │ │ (src.data)  │ │  (async notifications)  │ │
│ └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Component Architecture

#### 1. Bot Framework Layer (`bot.py`)

**Responsibilities:**
- Telegram Bot API integration using aiogram framework
- Message routing and command dispatching
- User session management and context handling
- Error handling and graceful degradation

**Key Components:**
```python
# Message Handlers
@dp.message(Command("report"))  # Report generation
@dp.message(Command("alerts"))  # Alert management
@dp.message(Command("schedules"))  # Schedule management
@dp.message(Command("admin"))  # Administrative functions

# Middleware & Filters
- User authentication middleware
- Rate limiting middleware
- Admin privilege checking
```

#### 2. Command Processing Layer

**Command Parser (`command_parser.py`):**
- Advanced command parsing with flag support
- Parameter validation and type conversion
- Extensible command specification system

```python
EnterpriseCommandParser:
├── parse() -> ParsedCommand
├── Support for -flag and --flag=value syntax
├── Positional argument handling
└── Parameter type conversion

Command Specifications:
├── report: {tickers, email, indicators, period, interval, provider}
├── alerts: {action, ticker, price, condition}
└── schedules: {action, ticker, time, flags}
```

**Business Logic (`business_logic.py`):**
- Command execution orchestration
- Data retrieval and processing coordination
- User permission and limit enforcement

```python
Command Handlers:
├── handle_report() -> Ticker analysis and report generation
├── handle_help() -> Dynamic help text based on user role
├── handle_info() -> User account information
└── handle_admin() -> Administrative functions (future)
```

**Notification Processing (`notifications.py`):**
- Telegram message formatting and delivery
- Email report generation and sending
- Multi-channel notification coordination

```python
Notification Processors:
├── process_report_command() -> Report delivery
├── process_alert_command() -> Alert management
├── process_schedule_command() -> Schedule management
└── process_admin_command() -> Admin notifications
```

#### 3. Service Layer Integration

**Database Layer (`db.py`):**
- SQLite-based data persistence
- User management and authentication
- Alert and schedule storage
- Migration and schema management

**Data Integration:**
- Seamless integration with `src.data` module
- Multi-provider data aggregation
- Caching and performance optimization

**Notification Integration:**
- Async notification manager integration
- Multi-channel delivery (Telegram + Email)
- Template-based message formatting

### Data Flow Architecture

#### User Command Flow

```
1. User Message → Telegram API → aiogram Dispatcher
2. Command Router → Specific Handler (cmd_report, cmd_alerts, etc.)
3. Command Parser → ParsedCommand with structured arguments
4. Business Logic → Data retrieval, processing, validation
5. Notification Processor → Format and deliver response
6. Multiple Channels → Telegram message + Optional email
```

#### Report Generation Flow

```
1. /report AAPL BTCUSDT -email -period=1y
2. Parse → {tickers: [AAPL, BTCUSDT], email: true, period: 1y}
3. Business Logic:
   ├── For each ticker:
   │   ├── Get OHLCV data (src.data)
   │   ├── Get fundamentals (src.common)
   │   ├── Calculate technicals (src.common)
   │   └── Format report (src.common.ticker_analyzer)
   └── Return reports array
4. Notification Processing:
   ├── Generate Telegram messages with charts
   ├── Generate HTML email with embedded charts
   └── Send to both channels
```

#### Alert Management Flow

```
1. /alerts add BTCUSDT 65000 above
2. Parse → {action: add, ticker: BTCUSDT, price: 65000, condition: above}
3. Business Logic:
   ├── Validate user limits
   ├── Validate ticker and price
   ├── Store alert in database
   └── Return confirmation
4. Background Alert Processing:
   ├── Periodic price checking
   ├── Condition evaluation
   └── Trigger notifications when met
```

### Database Design

#### Schema Architecture

```sql
-- Core user management
users (
    telegram_user_id PRIMARY KEY,
    email,
    verification_code,
    code_sent_time,
    verified,
    language,
    is_admin,
    max_alerts,
    max_schedules
)

-- Price alerting system
alerts (
    id PRIMARY KEY,
    ticker,
    user_id FOREIGN KEY,
    price,
    condition,  -- 'above' or 'below'
    active,
    created,
    updated_at
)

-- Scheduled reporting system
schedules (
    id PRIMARY KEY,
    ticker,
    user_id FOREIGN KEY,
    scheduled_time,  -- '09:00' format
    period,  -- 'daily', 'weekly', 'monthly'
    active,
    email,  -- send to email flag
    indicators,  -- JSON array
    interval,
    provider,
    created,
    updated_at
)

-- Verification code tracking
codes (
    telegram_user_id,
    code,
    sent_time
)

-- System configuration
settings (
    key PRIMARY KEY,
    value
)
```

#### Data Relationships

```
users (1) ───┬─── (N) alerts
             └─── (N) schedules
             
codes (N) ─── (1) users  [verification tracking]
```

### Security Architecture

#### Authentication Flow

```
1. User Registration:
   ├── /register user@email.com
   ├── Generate 6-digit verification code
   ├── Send code via email
   ├── Store temporary code in database
   └── Set 1-hour expiration

2. Email Verification:
   ├── /verify 123456
   ├── Validate code and expiration
   ├── Mark user as verified
   ├── Clean up verification codes
   └── Enable full bot functionality

3. Session Management:
   ├── Telegram user ID as primary identifier
   ├── No additional session tokens needed
   ├── Per-request authentication check
   └── Admin role verification
```

#### Authorization Model

```python
Permission Levels:
├── Unverified Users:
│   ├── Can use /help, /start, /register, /verify
│   └── Cannot access reports, alerts, schedules
├── Verified Users:
│   ├── Full access to personal features
│   ├── Respect per-user limits (alerts, schedules)
│   └── Cannot access admin functions
└── Admin Users:
    ├── All user functionality
    ├── User management capabilities
    ├── System configuration access
    └── Broadcast messaging permissions
```

#### Data Protection

- **Encryption**: Sensitive data encrypted at rest
- **API Keys**: Environment variable storage
- **Rate Limiting**: Per-user and global rate limits
- **Input Validation**: All user inputs sanitized and validated

## Design Decisions

### 1. Aiogram Framework Selection

**Decision:** Use aiogram 3.x as the Telegram Bot framework

**Rationale:**
- **Async Support**: Native async/await for high concurrency
- **Modern Architecture**: Clean, decorator-based message handling
- **Type Safety**: Full type hints and static analysis support
- **Performance**: Efficient for high-volume bot operations

**Implementation:**
```python
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command

@dp.message(Command("report"))
async def cmd_report(message: Message):
    # Handler implementation
```

### 2. Command Parsing Architecture

**Decision:** Custom enterprise-grade command parser with flag support

**Rationale:**
- **Flexibility**: Support for complex command syntax with flags
- **Extensibility**: Easy addition of new commands and parameters
- **User Experience**: Familiar CLI-like interface for power users
- **Validation**: Built-in parameter validation and type conversion

**Implementation:**
```python
# Support for: /report AAPL BTCUSDT -email -period=1y -provider=yf
parsed = parse_command(message.text)
# Result: {command: "report", args: {tickers: [AAPL, BTCUSDT], email: true, period: "1y"}}
```

### 3. Multi-Channel Notification System

**Decision:** Unified notification system supporting Telegram and Email

**Rationale:**
- **User Choice**: Users can choose delivery channels
- **Reliability**: Fallback to alternative channels if primary fails
- **Rich Content**: HTML emails with embedded charts
- **Consistency**: Same content formatted for different channels

**Implementation:**
```python
await notification_manager.send_notification(
    notification_type="INFO",
    title="Report for AAPL",
    message=report_content,
    channels=["telegram", "email"],
    attachments=chart_images
)
```

### 4. Database Schema Design

**Decision:** SQLite for development, PostgreSQL-ready schema

**Rationale:**
- **Simplicity**: SQLite for local development and small deployments
- **Scalability**: Schema designed for PostgreSQL migration
- **Performance**: Optimized indexes for common query patterns
- **Data Integrity**: Foreign keys and constraints

**Migration Strategy:**
- Start with SQLite for development
- Provide migration scripts for PostgreSQL
- Abstract database operations through ORM-like interface

### 5. Verification and Security Model

**Decision:** Email-based verification with time-limited codes

**Rationale:**
- **Security**: Email ownership verification prevents spam
- **User Experience**: Simple 6-digit code process
- **Privacy**: No phone number collection required
- **Automation**: Automated code generation and validation

**Security Features:**
- Time-limited verification codes (1 hour)
- Rate limiting on verification attempts
- Automatic cleanup of expired codes
- Secure code generation with sufficient entropy

### 6. Modular Business Logic Design

**Decision:** Separate business logic from Telegram API handling

**Rationale:**
- **Testability**: Business logic can be unit tested independently
- **Reusability**: Same logic can be used for web interface
- **Maintainability**: Clear separation of concerns
- **Integration**: Easy integration with other modules

**Architecture Pattern:**
```python
# bot.py (UI Layer) -> notifications.py (Coordination) -> business_logic.py (Core)
async def cmd_report(message):
    await process_report_command(message, user_id, args, notification_manager)

async def process_report_command(...):
    parsed = parse_command(message.text)
    result = handle_command(parsed)  # Business logic
    await send_notifications(result)  # Delivery
```

### 7. Error Handling Strategy

**Decision:** Multi-layered error handling with user-friendly messages

**Rationale:**
- **User Experience**: Clear, actionable error messages
- **Debugging**: Comprehensive logging for troubleshooting
- **Reliability**: Graceful degradation when services fail
- **Security**: No sensitive information leaked in error messages

**Error Handling Layers:**
1. **Input Validation**: Command syntax and parameter validation
2. **Business Logic**: Data validation and business rule enforcement
3. **Service Integration**: API failures and network issues
4. **User Notification**: Friendly error messages with suggested actions

## Performance Considerations

### Response Time Optimization

1. **Async Operations**: All I/O operations use async/await patterns
2. **Data Caching**: Frequently accessed data cached in memory
3. **Lazy Loading**: Load data only when needed
4. **Connection Pooling**: Reuse database and HTTP connections

### Memory Management

1. **Resource Cleanup**: Explicit cleanup of large objects
2. **Streaming Processing**: Process large datasets in chunks
3. **Garbage Collection**: Proper object lifecycle management
4. **Memory Monitoring**: Track memory usage patterns

### Scalability Design

1. **Stateless Architecture**: No server-side session state
2. **Database Optimization**: Efficient queries and indexing
3. **Load Distribution**: Support for multiple bot instances
4. **Rate Limiting**: Protect against abuse and overload

### Caching Strategy

```python
# Multi-level caching
├── In-Memory Cache: Frequently accessed user data
├── Database Cache: Recent API responses
└── Application Cache: Rendered reports and charts
```

## Integration Architecture

### Data Module Integration

**Seamless Integration with src.data:**
```python
# Direct integration with data downloaders
from src.common import get_ohlcv
from src.common.fundamentals import get_fundamentals
from src.common.technicals import calculate_technicals_from_df

# Unified data processing pipeline
def analyze_ticker_business(ticker, provider, period, interval):
    df = get_ohlcv(ticker, interval, period, provider)
    fundamentals = get_fundamentals(ticker, provider)
    df_with_technicals, technicals = calculate_technicals_from_df(df)
    return TickerAnalysis(...)
```

### Notification System Integration

**Async Notification Manager:**
```python
# Multi-channel notification delivery
notification_manager = await initialize_notification_manager(
    telegram_token=TELEGRAM_BOT_TOKEN,
    telegram_chat_id=TELEGRAM_CHAT_ID,
    email_api_key=SMTP_PASSWORD,
    email_sender=SMTP_USER
)

# Unified notification interface
await notification_manager.send_notification(
    notification_type="INFO",
    channels=["telegram", "email"],
    # ... other parameters
)
```

### Configuration Integration

**Centralized Configuration:**
```python
# Environment-based configuration
from config.donotshare.donotshare import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    SMTP_USER,
    SMTP_PASSWORD
)

# Dynamic configuration from database
settings = db.get_global_settings()
max_alerts = settings.get("max_alerts", 5)
```

## Extensibility Design

### Command Extension Framework

**Easy Addition of New Commands:**
```python
# 1. Add command specification
COMMAND_SPECS["newcommand"] = CommandSpec(
    parameters={"param1": str, "param2": int},
    defaults={"param1": None, "param2": 0},
    positional=["args"]
)

# 2. Add business logic handler
def handle_newcommand(parsed: ParsedCommand) -> Dict[str, Any]:
    # Implementation
    pass

# 3. Add notification processor
async def process_newcommand(message, user_id, args, notification_manager):
    # Implementation
    pass

# 4. Register command handler
@dp.message(Command("newcommand"))
async def cmd_newcommand(message: Message):
    # Handler registration
    pass
```

### Plugin Architecture

**Future Plugin Support:**
- Command plugins for specialized functionality
- Data provider plugins for new markets
- Notification channel plugins for new delivery methods
- Authentication plugins for enterprise integration

### Internationalization Framework

**Multi-Language Support:**
```python
# Message key-based localization
def get_localized_message(user_id: str, message_key: str, **kwargs) -> str:
    user_language = db.get_user_language(user_id)
    template = get_message_template(message_key, user_language)
    return template.format(**kwargs)

# Usage in business logic
return {
    "message": get_localized_message(
        user_id, "report.success", 
        ticker=ticker, count=len(reports)
    )
}
```

## Monitoring and Observability

### Logging Architecture

**Structured Logging:**
```python
# Hierarchical logger setup
logger = setup_logger("telegram_screener_bot")

# Contextual logging with user information
logger.info("Report generated", extra={
    "user_id": telegram_user_id,
    "ticker": ticker,
    "provider": provider,
    "execution_time": execution_time
})
```

### Metrics Collection

**Key Performance Indicators:**
- Command response times
- User engagement metrics
- Error rates by command type
- Data provider success rates
- Email delivery rates

### Health Monitoring

**System Health Checks:**
- Database connection status
- External API availability
- Email service connectivity
- Bot API responsiveness

## Future Architecture Considerations

### Microservice Migration

**Service Decomposition Strategy:**
```
Current Monolith → Future Microservices:
├── User Management Service
├── Report Generation Service
├── Alert Processing Service
├── Notification Delivery Service
└── Data Aggregation Service
```

### Event-Driven Architecture

**Event Sourcing for Audit Trail:**
- User action events
- System state changes
- Alert triggering events
- Report generation events

### Advanced Caching

**Distributed Caching:**
- Redis for session management
- Memcached for data caching
- CDN for static assets

### Auto-Scaling Infrastructure

**Kubernetes Deployment:**
- Horizontal pod autoscaling
- Database connection pooling
- Load balancing across instances
- Rolling deployments

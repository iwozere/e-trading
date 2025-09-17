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
@dp.message(Command("schedules"))  # Schedule management (includes screener)
@dp.message(Command("admin"))  # Administrative functions

# Middleware & Filters
- User authentication middleware
- Rate limiting middleware
- Admin privilege checking
```

#### 2. Command Processing Architecture

**Command Parser (`command_parser.py`):**
- Intelligent case-insensitive command parsing
- Smart argument type handling and conversion
- Support for flags, positional arguments, and named parameters

```python
CommandParser:
├── parse_command(command_text: str) -> ParsedCommand
├── handle_case_conversion(command: str, args: List[str]) -> Tuple[str, List[str]]
├── process_flags(tokens: List[str]) -> Dict[str, Any]
└── validate_command_structure(parsed: ParsedCommand) -> bool

ParsedCommand:
├── command: str (lowercase)
├── positionals: List[str] (tickers -> uppercase, actions -> lowercase)
├── args: Dict[str, Any] (flags and named parameters)
└── raw_args: List[str] (original arguments)
```

**Business Logic (`business_logic.py`):**
- Centralized command handling and routing
- User access control and approval checking
- Integration with all bot features (reports, alerts, schedules, screener)

```python
BusinessLogic:
├── handle_command(telegram_user_id: str, command: str, positionals: List[str], args: Dict) -> Dict
├── check_approved_access(telegram_user_id: str) -> Dict[str, Any]
├── handle_report(parsed: ParsedCommand) -> Dict[str, Any]
├── handle_alerts(parsed: ParsedCommand) -> Dict[str, Any]
├── handle_schedules(parsed: ParsedCommand) -> Dict[str, Any]
└── handle_admin(parsed: ParsedCommand) -> Dict[str, Any]
```

#### 3. Fundamental Screener Architecture

**Screener Component (`fundamental_screener.py`):**
- Core screening logic and data processing
- Ticker list management and data collection
- Fundamental analysis and scoring algorithms
- Report generation and formatting

```python
FundamentalScreener:
├── load_ticker_list(list_type: str) -> List[str]
├── collect_fundamentals(tickers: List[str]) -> Dict[str, Fundamentals]
├── apply_screening_criteria(fundamentals: Dict) -> List[ScreenerResult]
├── calculate_dcf_valuation(fundamentals: Fundamentals) -> DCFResult
├── generate_composite_score(fundamentals: Fundamentals) -> float
├── generate_report(results: List[ScreenerResult]) -> ScreenerReport
└── format_telegram_message(report: ScreenerReport) -> str

ScreenerResult:
├── ticker: str
├── fundamentals: Fundamentals
├── dcf_valuation: DCFResult
├── composite_score: float
├── screening_status: Dict[str, bool]
├── recommendation: str
└── reasoning: str

DCFResult:
├── fair_value: float
├── growth_rate: float
├── discount_rate: float
├── terminal_value: float
├── assumptions: Dict[str, float]
└── confidence_level: str
```

**Ticker List Manager:**
- Integration with existing ticker list functions
- Support for custom list creation and management
- List validation and error handling

```python
TickerListManager:
├── get_us_small_cap_tickers() -> List[str]
├── get_us_medium_cap_tickers() -> List[str]
├── get_us_large_cap_tickers() -> List[str]
├── get_swiss_shares_tickers() -> List[str]
├── get_custom_list_tickers(list_name: str) -> List[str]
└── validate_ticker_list(tickers: List[str]) -> bool
```

**Fundamental Data Collector:**
- Sequential data collection with rate limiting
- Error handling and data validation
- Progress tracking and user feedback

```python
FundamentalDataCollector:
├── collect_fundamentals_sequential(tickers: List[str]) -> Dict[str, Fundamentals]
├── validate_fundamental_data(fundamentals: Fundamentals) -> bool
├── handle_api_errors(ticker: str, error: Exception) -> None
├── track_progress(current: int, total: int) -> None
└── respect_rate_limits() -> None
```

**Screening Engine:**
- Multi-criteria screening with configurable thresholds
- Composite scoring algorithm
- Buy/Sell/Hold recommendation logic

```python
ScreeningEngine:
├── apply_valuation_criteria(fundamentals: Fundamentals) -> Dict[str, bool]
├── apply_financial_health_criteria(fundamentals: Fundamentals) -> Dict[str, bool]
├── apply_profitability_criteria(fundamentals: Fundamentals) -> Dict[str, bool]
├── apply_growth_criteria(fundamentals: Fundamentals) -> Dict[str, bool]
├── apply_cash_flow_criteria(fundamentals: Fundamentals) -> Dict[str, bool]
├── calculate_composite_score(criteria_results: Dict) -> float
├── generate_recommendation(score: float, fundamentals: Fundamentals) -> str
└── generate_reasoning(criteria_results: Dict, score: float) -> str
```

**DCF Valuation Engine:**
- Discounted Cash Flow calculation
- Growth rate estimation
- Risk-adjusted discount rate calculation

```python
DCFEngine:
├── calculate_free_cash_flow(fundamentals: Fundamentals) -> float
├── estimate_growth_rate(fundamentals: Fundamentals) -> float
├── calculate_discount_rate(fundamentals: Fundamentals) -> float
├── calculate_terminal_value(fcf: float, growth: float, discount: float) -> float
├── calculate_fair_value(fcf_series: List[float], terminal_value: float, discount: float) -> float
└── assess_confidence_level(assumptions: Dict) -> str
```

#### 4. Admin Panel Architecture

**Web-Based Admin Interface (`admin_panel.py`):**
- Flask-based web application for administrative tasks
- User management and approval workflow
- System monitoring and statistics dashboard

```python
AdminPanel:
├── Authentication & Security
│   ├── login_required decorator for route protection
│   ├── session management with Flask sessions
│   └── environment variable-based credentials
├── User Management
│   ├── list_users() -> Display all users with status
│   ├── approve_user(user_id) -> Grant access to restricted features
│   ├── reject_user(user_id) -> Revoke access
│   ├── verify_user(user_id) -> Manual verification
│   └── reset_user_email(user_id) -> Email reset
├── Dashboard
│   ├── system_statistics() -> User counts, alerts, schedules
│   ├── pending_approvals() -> Users waiting for approval
│   └── recent_activity() -> System activity overview
└── System Management
    ├── alerts_management() -> View/edit all user alerts
    ├── schedules_management() -> View/edit all schedules
    ├── broadcast_messaging() -> Send messages to users
    └── feedback_management() -> Handle user feedback
```

**Admin Panel Routes:**
```python
Routes:
├── /login - Admin authentication
├── /logout - Session termination
├── / - Dashboard with statistics and navigation links
├── /users - User management interface with filtering
├── /users/<user_id>/approve - User approval
├── /users/<user_id>/reject - User rejection
├── /users/<user_id>/verify - Manual verification
├── /users/<user_id>/reset - Email reset
├── /alerts - Alert management with filtering
├── /schedules - Schedule management with filtering
├── /feedback - Feedback management with filtering
├── /broadcast - Broadcast messaging
├── /audit - Command audit dashboard with comprehensive filtering
├── /audit/user/<user_id> - User-specific command history
└── /audit/non-registered - Non-registered users audit view
```

**Command Audit System Architecture:**
```python
AuditSystem:
├── Database Layer
│   ├── command_audit table with comprehensive tracking
│   ├── Indexed queries for performance optimization
│   └── Data retention and cleanup policies
├── Bot Integration
│   ├── audit_command_wrapper() - Automatic command logging
│   ├── Performance tracking with response times
│   ├── Error handling and detailed error logging
│   └── User classification (registered vs non-registered)
├── Admin Panel Integration
│   ├── /audit - Main audit dashboard with statistics
│   ├── /audit/user/<user_id> - Individual user history
│   ├── /audit/non-registered - Non-registered users view
│   └── Enhanced filtering and search capabilities
└── Statistics and Analytics
    ├── System health monitoring
    ├── User activity analysis
    ├── Performance metrics
    └── Error rate tracking
```

**Enhanced Navigation System:**
```python
NavigationFeatures:
├── Dashboard Navigation
│   ├── Stat card navigation links
│   ├── Direct access to filtered views
│   └── Visual feedback and indicators
├── Filter Support
│   ├── URL-based filtering for all pages
│   ├── Quick filter buttons
│   ├── Advanced filter combinations
│   └── Filter state persistence
├── User Management Navigation
│   ├── Verified users filter
│   ├── Approved users filter
│   ├── Pending approvals filter
│   └── User history integration
└── Audit Navigation
    ├── Time-based filtering (24h, custom range)
    ├── User type filtering (registered/non-registered)
    ├── Success/failure filtering
    └── Command-specific filtering
```

**JSON Generator Tool Architecture:**
```python
JSONGenerator:
├── Web Interface
│   ├── Tabbed interface (Alerts, Schedules, Reports, Screeners)
│   ├── Responsive design with modern UI/UX
│   ├── Real-time JSON generation and preview
│   └── Copy-to-clipboard and validation features
├── Multiple Indicators Support
│   ├── Single indicator mode (traditional)
│   ├── Multiple indicators with AND logic
│   ├── Multiple indicators with OR logic
│   └── Dynamic parameter configuration
├── Indicator Management
│   ├── Dynamic form generation for indicator parameters
│   ├── Parameter persistence across indicator additions
│   ├── Visual feedback for configured indicators
│   └── Add/remove functionality for indicator lists
├── Template System
│   ├── Pre-configured templates for common scenarios
│   ├── Quick template loading and application
│   ├── Template customization and extension
│   └── Template validation and error checking
├── Integration
│   ├── Admin panel integration as new tab
│   ├── Standalone shareable version
│   ├── Command generation from JSON configurations
│   └── Backend compatibility with existing bot commands
└── Supported Indicators
    ├── Technical Indicators: RSI, MACD, Bollinger Bands, SMA, EMA, ADX, ATR, Stochastic, Williams %R, CCI, ROC, MFI
    ├── Parameter Configuration: Periods, deviations, thresholds, signal lines
    ├── Operator Support: Comparison operators, crossovers, band conditions
    └── Logic Combinations: AND/OR logic for multiple conditions
```

**JSON Generator Routes:**
```python
Routes:
├── /json-generator - Main JSON generator interface
└── /templates/standalone_json_generator.html - Standalone shareable version
```

**JSON Generator Features:**
```python
Features:
├── Alerts Tab
│   ├── Price alerts with above/below conditions
│   ├── Single indicator alerts with custom parameters
│   ├── Multiple indicator alerts with AND/OR logic
│   ├── Template system for common alert scenarios
│   └── Real-time command generation
├── Schedules Tab
│   ├── Report schedules with multiple indicators
│   ├── Screener schedules with fundamental criteria
│   ├── Enhanced screener with technical criteria
│   ├── Time configuration and period selection
│   └── Email integration and delivery options
├── Reports Tab
│   ├── Multi-ticker report generation
│   ├── Technical analysis with multiple indicators
│   ├── Fundamental analysis integration
│   ├── Data provider selection
│   └── Comprehensive analysis options
└── Screeners Tab
    ├── Fundamental, technical, and hybrid screening
    ├── Multiple list types and market cap categories
    ├── Custom fundamental and technical criteria
    ├── Result limits and scoring thresholds
    └── Advanced screening configurations
```

#### 5. Command Processing Layer

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

#### 6. Re-Arm Alert System Architecture

**Enhanced Alert Processing:**
- Professional-grade crossing detection eliminates notification spam
- Automatic re-arming when price moves back across hysteresis level
- Configurable parameters for different trading styles

```python
ReArmAlertSystem:
├── ReArmAlertEvaluator: Core crossing detection and re-arming logic
├── EnhancedAlertConfig: JSON-based configuration management
├── ReArmConfig: Re-arm specific settings (hysteresis, cooldown, persistence)
├── NotificationConfig: Multi-channel notification preferences
└── Migration System: Safe conversion of existing alerts

Alert State Machine:
[ARMED] → Price crosses threshold → [TRIGGERED] → Send notification
    ↑                                      ↓
    └── Price crosses hysteresis level ←──┘

Database Schema Extensions:
├── is_armed: BOOLEAN - Current armed state
├── last_price: DECIMAL - Last known price for crossing detection  
├── last_triggered_at: TIMESTAMP - When alert last triggered
└── re_arm_config: TEXT - JSON configuration for re-arm behavior
```

**Re-Arm Configuration Options:**
```json
{
  "alert_type": "price",
  "ticker": "AAPL", 
  "threshold": 150.00,
  "direction": "above",
  "re_arm_config": {
    "enabled": true,
    "hysteresis": 0.25,
    "hysteresis_type": "percentage",
    "cooldown_minutes": 15,
    "persistence_bars": 1,
    "close_only": false
  },
  "notification_config": {
    "channels": ["telegram", "email"],
    "template": "{ticker} crossed {direction} {threshold} at {current_price}. Will re-arm below {rearm_level}."
  }
}
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

#### JSON Configuration Flow

```
1. /report -config='{"report_type":"analysis","tickers":["AAPL","MSFT"],"period":"1y","indicators":["RSI","MACD"],"email":true}'
2. Parse → {config: JSON_STRING}
3. Business Logic:
   ├── Validate JSON configuration using ReportConfigParser
   ├── Extract parameters from validated config
   ├── Process report with extracted parameters
   └── Return reports array
4. Notification Processing:
   ├── Generate Telegram messages with charts
   ├── Generate HTML email with embedded charts
   └── Send to both channels
```

**JSON Configuration Architecture:**
- **ReportConfigParser**: Validates and parses JSON configuration strings
- **ReportConfig**: Dataclass defining the structure for report configurations
- **Validation**: Comprehensive validation of all JSON fields and values
- **Fallback**: Traditional flag-based parsing if no JSON config provided

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
    # telegram_chat_id=TELEGRAM_CHAT_ID,  # No longer needed - admin notifications use HTTP API
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
    # TELEGRAM_CHAT_ID,  # No longer needed - admin notifications use HTTP API
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

## Access Control and Security Architecture

### Multi-Tier Access Control System

The bot implements a comprehensive access control system with three distinct levels:

#### 1. Public Commands (No restrictions)
- `/start`, `/help` - Show welcome and help messages
- `/info` - Show user status and approval information
- `/register`, `/verify` - Email registration and verification
- `/request_approval` - Request admin approval
- `/feedback`, `/feature` - Send feedback and feature requests

#### 2. Restricted Commands (Require `approved=1`)
- `/report` - Generate ticker reports
- `/alerts` - Manage price alerts
- `/schedules` - Manage scheduled reports
- `/language` - Change language preference

#### 3. Admin Commands (Require `is_admin=1`)
- `/admin users` - List all users
- `/admin approve USER_ID` - Approve user for restricted features
- `/admin reject USER_ID` - Reject user's approval request
- `/admin pending` - List users waiting for approval
- `/admin broadcast MESSAGE` - Send broadcast message to all users

### User Registration and Approval Workflow

**Complete User Journey:**
```
1. User sends `/register email@example.com`
2. User receives 6-digit verification code via email
3. User sends `/verify CODE` to verify email
4. User sends `/request_approval` to request access
5. Admin receives notification of approval request
6. Admin sends `/admin approve USER_ID` to approve user
7. User receives notification of approval
8. User can now use restricted commands
```

### Database Schema for Access Control

**Enhanced Users Table:**
```sql
CREATE TABLE users (
    telegram_user_id TEXT PRIMARY KEY,
    email TEXT,
    verification_code TEXT,
    code_sent_time INTEGER,
    verified INTEGER DEFAULT 0,
    approved INTEGER DEFAULT 0,  -- New: Admin approval status
    language TEXT DEFAULT 'en',
    is_admin INTEGER DEFAULT 0,
    max_alerts INTEGER DEFAULT 5,
    max_schedules INTEGER DEFAULT 5
);
```

### Access Control Implementation

**Business Logic Functions:**
```python
def is_approved_user(telegram_user_id: str) -> bool:
    """Check if user is approved for restricted features."""
    
def check_admin_access(telegram_user_id: str) -> Dict[str, Any]:
    """Check if user has admin access. Returns error dict if not."""
    
def check_approved_access(telegram_user_id: str) -> Dict[str, Any]:
    """Check if user has approved access for restricted features."""
```

**Command Handler Integration:**
```python
def handle_report(parsed: ParsedCommand) -> Dict[str, Any]:
    # Check if user has approved access
    telegram_user_id = args.get("telegram_user_id")
    access_check = check_approved_access(telegram_user_id)
    if access_check["status"] != "ok":
        return access_check
    # ... rest of handler logic
```

### Admin Management System

**Admin Setup Script:**
- `src/util/create_admin.py` - Automated admin user creation
- Registers, verifies, and approves admin users automatically
- Usage: `python src/util/create_admin.py <telegram_user_id> <email>`

**Admin Commands:**
- User approval/rejection workflow
- User management (list, verify, reset email)
- System settings and limits configuration
- Broadcast messaging to all users

### Security Features

**Authentication and Authorization:**
- Email verification required for full functionality
- Admin role verification for admin commands
- User approval workflow for restricted features
- Rate limiting on verification code requests
- Input validation and sanitization
- User data isolation and permission checking

## Reply Architecture and Message Routing

### Problem Analysis

**Original Architecture Issues:**
1. **Fixed Chat ID**: The `TelegramChannel` was hardcoded to send all responses to a fixed admin chat ID (no longer needed - admin notifications use HTTP API)
2. **Reply Mismatch**: Bot tried to reply to messages in the admin chat, but the original messages were in user chats
3. **User Experience**: Users sending commands didn't receive responses in their own chat

**Root Cause:**
```
User Chat: [User sends: "/help"] → Bot receives message ID 123
Admin Chat: [Bot tries to reply to message ID 123] → ❌ FAILS (message 123 doesn't exist in admin chat)
Admin Chat: [Bot sends without reply] → ✅ SUCCESS (but user doesn't see it)
```

### Solution: Dynamic Chat ID Routing

**Implemented Solution:**
```python
# Use dynamic chat_id if provided, otherwise fall back to default
target_chat_id = notification.data.get('telegram_chat_id') if notification.data else None
if target_chat_id is None:
    target_chat_id = self.chat_id  # Fallback to admin chat
    _logger.debug("Using default chat_id: %s", target_chat_id)
else:
    _logger.debug("Using dynamic chat_id: %s", target_chat_id)
```

**New Flow:**
```
User Chat: [User sends: "/help"] → Bot receives message ID 123
User Chat: [Bot replies to message ID 123] → ✅ SUCCESS (same chat, message exists)
```

### Data Flow Architecture

**Complete Message Routing:**
1. **Command Handler**: Extracts `message.chat.id` from incoming message
2. **Notification Processing**: Passes `telegram_chat_id=message.chat.id` to notification
3. **TelegramChannel**: Uses dynamic `target_chat_id` from notification data
4. **Reply Success**: Bot can now reply to the original message in the same chat

**Notification Data Structure:**
```python
data={
    "channels": ["telegram"],
    "telegram_chat_id": message.chat.id,  # ✅ Dynamic chat ID
    "reply_to_message_id": message.message_id  # ✅ Original message ID
}
```

### Robust Reply Handling

**Fallback Mechanism:**
```python
# Try to send with reply first, fall back to regular message if reply fails
try:
    await self.bot.send_message(
        chat_id=target_chat_id,
        text=notification.message,
        parse_mode=None,
        reply_to_message_id=reply_to_message_id
    )
except Exception as reply_error:
    _logger.warning("Failed to send message with reply, sending without reply: %s", reply_error)
    await self.bot.send_message(
        chat_id=target_chat_id,
        text=notification.message,
        parse_mode=None
    )
```

### Benefits of New Architecture

**1. Improved User Experience:**
- Users receive responses in their own chat
- Replies work correctly (no more "message not found" errors)
- Natural conversation flow

**2. Maintained Security:**
- Admin chat still receives all notifications (if needed)
- No changes to authentication or authorization
- Backward compatibility preserved

**3. Better Debugging:**
- Clear logging shows which chat ID is being used
- Easier to troubleshoot notification issues

### Testing and Validation

**Test Scenarios:**
1. **User sends command** → Response appears in user's chat ✅
2. **Reply functionality** → Bot can reply to original message ✅
3. **Admin notifications** → Still work (fallback to admin chat) ✅
4. **Multiple users** → Each gets responses in their own chat ✅

**Expected Behavior:**
- ✅ No more "message to be replied not found" errors
- ✅ Users see bot responses in their chat
- ✅ Replies work correctly
- ✅ Admin monitoring still possible

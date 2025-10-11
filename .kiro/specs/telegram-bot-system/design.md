# Design Document

## Purpose

This document outlines the design for a comprehensive Telegram bot system that provides financial market analysis, real-time alerting, scheduled reporting, and administrative management capabilities. The system serves as the primary user interface for the e-trading platform, enabling users to access sophisticated market analysis tools through a conversational Telegram interface.

The design emphasizes modularity, scalability, and maintainability while providing professional-grade financial analysis capabilities including fundamental screening, technical analysis, advanced alert systems, and comprehensive administrative controls.

## Architecture

### High-Level System Architecture

The Telegram bot system follows a layered, event-driven architecture with clear separation between user interface, business logic, and data access:

```
┌─────────────────────────────────────────────────────────────┐
│                    Telegram Client                          │
│                 (User Interface Layer)                      │
└─────────────────────┬───────────────────────────────────────┘
                      │ Telegram Bot API
┌─────────────────────▼───────────────────────────────────────┐
│                 Bot Framework Layer                         │
│  ┌─────────────────┐  ┌─────────────────┐ ┌──────────────┐ │
│  │   Command       │  │   Message       │ │   HTTP API   │ │
│  │   Handlers      │  │   Routing       │ │   Server     │ │
│  │   (bot.py)      │  │ (aiogram dp)    │ │ (aiohttp)    │ │
│  └─────────────────┘  └─────────────────┘ └──────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │ Parsed Commands & API Requests
┌─────────────────────▼───────────────────────────────────────┐
│              Command Processing Layer                       │
│ ┌─────────────────┐ ┌─────────────────┐ ┌────────────────┐ │
│ │ Command Parser  │ │ Business Logic  │ │ Notification   │ │
│ │ (command_parser)│ │ (business_logic)│ │ (notifications)│ │
│ └─────────────────┘ └─────────────────┘ └────────────────┘ │
└─────────────┬─────────────────┬─────────────────┬──────────┘
              │                 │                 │
┌─────────────▼─────────────────▼─────────────────▼──────────┐
│                   Service Layer                            │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│ │  Database   │ │ Data Module │ │   Notification Manager  ││
│ │ (telegram_  │ │ (src.data)  │ │  (async notifications)  ││
│ │  service)   │ │             │ │                         ││
│ └─────────────┘ └─────────────┘ └─────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
              │                 │                 │
┌─────────────▼─────────────────▼─────────────────▼──────────┐
│                Infrastructure Layer                        │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│ │   SQLite/   │ │   Market    │ │      Email/SMTP         ││
│ │ PostgreSQL  │ │ Data APIs   │ │      Services           ││
│ │  Database   │ │ (Yahoo, FMP)│ │                         ││
│ └─────────────┘ └─────────────┘ └─────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Component Design

#### 1. Bot Framework Layer

**Core Bot Application (`bot.py`):**
- **Technology Stack**: aiogram 3.x with async/await support
- **Message Handling**: Decorator-based command routing with case-insensitive processing
- **HTTP API Server**: aiohttp-based REST API for external integrations
- **Error Handling**: Comprehensive error catching with user-friendly messages

**Key Components:**
```python
# Command Handlers with Audit Wrapper
@dp.message(Command("report"))
async def cmd_report(message: Message):
    await audit_command_wrapper(message, process_report_command, ...)

@dp.message(Command("alerts"))  
async def cmd_alerts(message: Message):
    await audit_command_wrapper(message, process_alerts_command, ...)

# HTTP API Endpoints
POST /api/send_message     # Send message to specific user
POST /api/broadcast        # Broadcast to all users  
GET  /api/status          # Health check and statistics
GET  /api/test            # API connectivity test
```

**Audit Command Wrapper:**
- Automatic logging of all user interactions
- Performance tracking with response time measurement
- Error handling and detailed error logging
- User classification (registered vs non-registered)
- Success/failure status tracking

#### 2. Command Processing Architecture

**Enterprise Command Parser (`command_parser.py`):**
- **Case-Insensitive Processing**: All commands work regardless of capitalization
- **Smart Type Conversion**: Tickers → uppercase, actions → lowercase, parameters → appropriate types
- **Flag Support**: `-flag value`, `--flag=value`, and `--flag value` syntax
- **Extensible Specifications**: JSON-based command definitions with validation

```python
EnterpriseCommandParser:
├── parse(message_text: str) -> ParsedCommand
├── Case-insensitive command processing
├── Smart argument type handling and conversion  
├── Support for flags, positional arguments, and named parameters
└── Extensible command specification system

Command Specifications:
├── report: {tickers, email, indicators, period, interval, provider, config}
├── alerts: {action, ticker, price, condition, email, timeframe, config}
├── schedules: {action, ticker, time, email, indicators, period, config}
├── screener: {screener_name_or_config, email, immediate}
└── admin: {action, params} (flexible admin command structure)
```

**Business Logic Engine (`business_logic.py`):**
- **Command Orchestration**: Centralized command handling and routing
- **Access Control**: User verification and approval checking
- **Data Integration**: Coordination with data providers and analysis modules
- **Result Processing**: Formatting and preparation for notification delivery

```python
BusinessLogic:
├── handle_command(user_id, command, positionals, args) -> Dict
├── check_approved_access(user_id) -> Dict[str, Any]
├── handle_report(parsed: ParsedCommand) -> Dict[str, Any]
├── handle_alerts(parsed: ParsedCommand) -> Dict[str, Any]  
├── handle_schedules(parsed: ParsedCommand) -> Dict[str, Any]
├── handle_screener(parsed: ParsedCommand) -> Dict[str, Any]
└── handle_admin(parsed: ParsedCommand) -> Dict[str, Any]
```

#### 3. Advanced Alert System Architecture

**Re-Arm Alert System:**
- **Professional Crossing Detection**: Eliminates notification spam through proper threshold crossing
- **Automatic Re-Arming**: Alerts re-arm when price moves back across hysteresis levels
- **Configurable Parameters**: Percentage, fixed, or ATR-based hysteresis with cooldown periods
- **State Persistence**: Maintains alert state across system restarts

```python
ReArmAlertSystem:
├── AlertsEvalService: Core crossing detection and re-arming logic
├── EnhancedAlertConfig: JSON-based configuration management
├── ReArmConfig: Hysteresis, cooldown, and persistence settings
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
├── re_arm_config: TEXT - JSON configuration for re-arm behavior
└── state_json: TEXT - Alert state persistence across restarts
```

**Alert Configuration Options:**
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

#### 4. Fundamental Screener Architecture

**Core Screening Engine (`fundamental_screener.py`):**
- **Multi-Criteria Analysis**: Valuation, financial health, profitability, and growth metrics
- **DCF Valuation**: Discounted Cash Flow calculations with confidence levels
- **Composite Scoring**: Weighted scoring system (0-10 scale) with buy/sell/hold recommendations
- **Market Cap Support**: Small, medium, and large cap categories plus custom lists

```python
FundamentalScreener:
├── load_ticker_list(list_type: str) -> List[str]
├── collect_fundamentals(tickers: List[str]) -> Dict[str, Fundamentals]
├── apply_screening_criteria(fundamentals: Dict) -> List[ScreenerResult]
├── calculate_dcf_valuation(fundamentals: Fundamentals) -> DCFResult
├── generate_composite_score(fundamentals: Fundamentals) -> float
├── generate_report(results: List[ScreenerResult]) -> ScreenerReport
└── format_telegram_message(report: ScreenerReport) -> str

Screening Criteria:
├── Valuation: P/E < 15, P/B < 1.5, P/S < 1, PEG < 1.5
├── Financial Health: Debt/Equity < 0.5, Current Ratio > 1.5, Quick Ratio > 1
├── Profitability: ROE > 15%, ROA > 5%, Operating Margin > 10%
├── Growth: Revenue Growth > 5%, Net Income Growth > 5%
├── Cash Flow: Positive Free Cash Flow
└── Dividends: Dividend Yield > 2%, Payout Ratio < 60%
```

**Enhanced Screener with FMP Integration:**
- **90% Performance Improvement**: Single FMP API call vs processing all tickers individually
- **Professional Pre-filtering**: Uses FMP's sophisticated screening algorithms
- **Hybrid Analysis**: Combines fundamental and technical analysis with weighted scoring
- **Fallback Support**: Automatic fallback to traditional methods when FMP unavailable

```python
EnhancedScreener:
├── FMP Integration Layer
│   ├── Pre-filtering with custom criteria or predefined strategies
│   ├── Professional screening algorithms
│   ├── Significant performance improvements
│   └── Automatic fallback to traditional methods
├── Hybrid Analysis Engine
│   ├── Fundamental criteria with weighted scoring
│   ├── Technical criteria with indicator conditions
│   ├── Composite scoring (0-10 scale) with minimum thresholds
│   └── Buy/Sell/Hold recommendations with reasoning
├── Configuration Management
│   ├── JSON-based configuration with validation
│   ├── Predefined strategies (conservative_value, growth, etc.)
│   ├── Custom criteria with flexible operators
│   └── Template system for common configurations
└── Result Processing
    ├── Ranked results with detailed analysis
    ├── Individual company analysis and recommendations
    ├── Risk assessment and confidence levels
    └── Multi-channel delivery (Telegram + Email)
```

#### 5. Administrative Management System

**Web-Based Admin Panel (`admin_panel.py`):**
- **Flask-Based Interface**: Modern web interface for administrative tasks
- **User Management**: Approval workflow, verification, and role management
- **System Monitoring**: Real-time statistics, audit logs, and performance metrics
- **JSON Generator Tool**: Interactive tool for creating complex configurations

```python
AdminPanel:
├── Authentication & Security
│   ├── Environment variable-based credentials
│   ├── Session management with Flask sessions
│   └── Route protection with login_required decorator
├── User Management Interface
│   ├── User listing with filtering (verified, approved, pending)
│   ├── Approval workflow (approve/reject with notifications)
│   ├── Manual verification and email reset capabilities
│   └── User activity tracking and history
├── System Dashboard
│   ├── Real-time statistics (users, alerts, schedules, commands)
│   ├── System health monitoring and performance metrics
│   ├── Recent activity overview and trend analysis
│   └── Navigation links to filtered views
├── Audit and Monitoring
│   ├── Comprehensive command audit with filtering
│   ├── User-specific command history
│   ├── Performance tracking and error analysis
│   └── Non-registered user activity monitoring
└── JSON Generator Tool
    ├── Interactive web interface for configuration creation
    ├── Multi-tab interface (Alerts, Schedules, Reports, Screeners)
    ├── Real-time JSON generation and validation
    ├── Template system with pre-built configurations
    ├── Copy-to-clipboard and command generation
    └── Support for complex multi-indicator configurations
```

**Admin Panel Routes:**
```python
Routes:
├── /login, /logout - Authentication management
├── / - Main dashboard with statistics and navigation
├── /users - User management with filtering and actions
├── /users/<user_id>/{approve,reject,verify,reset} - User actions
├── /alerts - Alert management and monitoring
├── /schedules - Schedule management and monitoring  
├── /feedback - User feedback management
├── /broadcast - Broadcast messaging to users
├── /audit - Command audit dashboard with filtering
├── /audit/user/<user_id> - User-specific audit history
├── /audit/non-registered - Non-registered user activity
└── /json-generator - Interactive JSON configuration tool
```

#### 6. JSON Configuration System

**Configuration Parser and Validator:**
- **Comprehensive Validation**: Field validation with detailed error messages
- **Multiple Format Support**: JSON strings, flag-based parsing, and hybrid approaches
- **Template System**: Pre-built configurations for common use cases
- **Error Handling**: Graceful fallback to traditional parsing methods

```python
JSONConfigurationSystem:
├── ReportConfigParser: Validates and parses report configurations
├── AlertConfigParser: Handles complex alert configurations
├── ScheduleConfigParser: Manages schedule configurations
├── ScreenerConfigParser: Processes screener configurations
├── ValidationEngine: Comprehensive field and value validation
├── TemplateManager: Pre-built configuration templates
└── ErrorHandler: Graceful error handling and fallback mechanisms

Supported Configuration Types:
├── Report Configurations
│   ├── Single and multi-ticker reports
│   ├── Technical and fundamental analysis options
│   ├── Data provider and period selection
│   └── Output format and delivery preferences
├── Alert Configurations
│   ├── Price alerts with re-arm settings
│   ├── Technical indicator alerts with complex conditions
│   ├── Multi-indicator alerts with AND/OR logic
│   └── Notification preferences and channels
├── Schedule Configurations
│   ├── Recurring report schedules
│   ├── Screener schedules with criteria
│   ├── Time zone and frequency settings
│   └── Delivery channel preferences
└── Screener Configurations
    ├── Fundamental, technical, and hybrid screening
    ├── Custom criteria with weighted scoring
    ├── Market cap categories and custom lists
    └── Result limits and scoring thresholds
```

#### 7. Multi-Channel Notification System

**Notification Architecture:**
- **Unified Interface**: Single API for multiple delivery channels
- **Rich Content Support**: HTML emails with embedded charts and Telegram messages with formatting
- **Delivery Tracking**: Success/failure tracking with retry mechanisms
- **Template System**: Consistent formatting across channels

```python
NotificationSystem:
├── Async Notification Manager
│   ├── Multi-channel delivery (Telegram + Email)
│   ├── Queue management and batch processing
│   ├── Retry mechanisms and error handling
│   └── Delivery status tracking and reporting
├── Content Formatting
│   ├── Telegram message formatting with Markdown
│   ├── HTML email generation with embedded charts
│   ├── Template-based message generation
│   └── Consistent styling across channels
├── Channel Management
│   ├── Dynamic chat ID routing for Telegram
│   ├── SMTP configuration and email delivery
│   ├── Channel preference management per user
│   └── Fallback mechanisms for failed deliveries
└── Integration Points
    ├── Chart generation and embedding
    ├── Report formatting and attachment handling
    ├── Alert notification templates
    └── Broadcast message distribution
```

### Database Design

#### Core Schema Architecture

```sql
-- Enhanced user management with approval workflow
CREATE TABLE users (
    telegram_user_id TEXT PRIMARY KEY,
    email TEXT,
    verification_code TEXT,
    code_sent_time INTEGER,
    verified INTEGER DEFAULT 0,
    approved INTEGER DEFAULT 0,  -- Admin approval status
    language TEXT DEFAULT 'en',
    is_admin INTEGER DEFAULT 0,
    max_alerts INTEGER DEFAULT 5,
    max_schedules INTEGER DEFAULT 5,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Enhanced alerts system with re-arm functionality
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    user_id TEXT NOT NULL,
    price REAL,
    condition TEXT,  -- 'above', 'below', or NULL for indicator alerts
    active INTEGER DEFAULT 1,
    created TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    -- Enhanced alert fields
    alert_type TEXT DEFAULT 'price',  -- 'price' or 'indicator'
    timeframe TEXT DEFAULT '15m',
    alert_action TEXT DEFAULT 'notify',
    config_json TEXT,  -- JSON configuration for complex alerts
    -- Re-arm system fields
    re_arm_config TEXT,  -- JSON configuration for re-arm behavior
    is_armed INTEGER DEFAULT 1,  -- Whether alert is currently armed
    last_price REAL,  -- Last known price for crossing detection
    last_triggered_at TEXT,  -- ISO timestamp of last trigger
    trigger_count INTEGER DEFAULT 0,  -- Number of times triggered
    state_json TEXT,  -- Alert state persistence
    FOREIGN KEY (user_id) REFERENCES users(telegram_user_id)
);

-- Enhanced schedules system with JSON configuration
CREATE TABLE schedules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,  -- Can be NULL for screener schedules
    user_id TEXT NOT NULL,
    scheduled_time TEXT NOT NULL, -- '09:00' format
    schedule_type TEXT DEFAULT 'report',  -- 'report' or 'screener'
    period TEXT DEFAULT 'daily', -- 'daily', 'weekly', 'monthly'
    active INTEGER DEFAULT 1,
    email INTEGER DEFAULT 0, -- Send to email flag
    indicators TEXT, -- JSON array of indicators
    interval TEXT DEFAULT '1d',
    provider TEXT,
    config_json TEXT,  -- JSON configuration for complex schedules
    created TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(telegram_user_id)
);

-- Command audit system for comprehensive tracking
CREATE TABLE command_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_user_id TEXT NOT NULL,
    command TEXT NOT NULL,
    full_message TEXT,
    is_registered_user INTEGER DEFAULT 0,
    user_email TEXT,
    success INTEGER DEFAULT 1,
    error_message TEXT,
    response_time_ms INTEGER,
    created TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Broadcast tracking for admin messaging
CREATE TABLE broadcast_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message TEXT NOT NULL,
    sent_by TEXT NOT NULL,  -- Admin user ID
    total_recipients INTEGER DEFAULT 0,
    successful_deliveries INTEGER DEFAULT 0,
    failed_deliveries INTEGER DEFAULT 0,
    created TEXT DEFAULT CURRENT_TIMESTAMP
);

-- System settings and configuration
CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT,
    description TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Performance indexes for query optimization
CREATE INDEX idx_alerts_user_active ON alerts(user_id, active);
CREATE INDEX idx_schedules_user_active ON schedules(user_id, active);
CREATE INDEX idx_command_audit_user_id ON command_audit(telegram_user_id);
CREATE INDEX idx_command_audit_created ON command_audit(created);
CREATE INDEX idx_command_audit_command ON command_audit(command);
CREATE INDEX idx_users_verified_approved ON users(verified, approved);
```

#### Data Relationships

```
users (1) ───┬─── (N) alerts
             ├─── (N) schedules  
             ├─── (N) command_audit
             └─── (N) broadcast_history

Settings and configuration data stored separately for system-wide parameters
```

### Security Architecture

#### Authentication and Authorization

**Multi-Tier Access Control:**
```python
Access Levels:
├── Public Commands (No restrictions):
│   ├── /start, /help - Welcome and help information
│   ├── /info - User status and account information
│   ├── /register, /verify - Email registration and verification
│   ├── /request_approval - Request admin approval
│   └── /feedback, /feature - User feedback and suggestions
├── Restricted Commands (Require approved=1):
│   ├── /report - Generate ticker reports and analysis
│   ├── /alerts - Manage price and indicator alerts
│   ├── /schedules - Manage scheduled reports and screeners
│   ├── /screener - Run fundamental and enhanced screening
│   └── /language - Change language preferences
└── Admin Commands (Require is_admin=1):
    ├── /admin users - List and manage all users
    ├── /admin approve/reject - User approval workflow
    ├── /admin broadcast - Send messages to all users
    ├── /admin pending - List users awaiting approval
    └── /admin setlimit - Manage user limits and quotas
```

**Security Features:**
- **Email Verification**: 6-digit codes with 1-hour expiration
- **Admin Approval Workflow**: Two-tier verification (email + admin approval)
- **Rate Limiting**: Per-user and global rate limits to prevent abuse
- **Input Validation**: Comprehensive sanitization of all user inputs
- **Audit Logging**: Complete tracking of all user interactions
- **Secure Storage**: Environment variable-based API key management

#### Data Protection

**Privacy and Compliance:**
- **Data Minimization**: Collect only necessary user information
- **Encryption**: Sensitive data encrypted at rest and in transit
- **Retention Policies**: 30-day audit log retention with automatic cleanup
- **GDPR Compliance**: User data handling and deletion capabilities
- **Access Controls**: Role-based access to sensitive operations

### Performance Considerations

#### Scalability Design

**Async Architecture:**
- **Non-blocking Operations**: All I/O operations use async/await patterns
- **Concurrent Processing**: Multiple user requests handled simultaneously
- **Queue Management**: Background processing for long-running operations
- **Connection Pooling**: Efficient database and HTTP connection reuse

**Memory and Resource Management:**
- **Streaming Processing**: Large datasets processed in chunks
- **Resource Cleanup**: Explicit cleanup of large objects and connections
- **Caching Strategy**: Multi-level caching for frequently accessed data
- **Garbage Collection**: Proper object lifecycle management

**Database Optimization:**
- **Efficient Queries**: Optimized queries with proper indexing
- **Connection Pooling**: Database connection reuse and management
- **Query Caching**: Cache results for repeated operations
- **Batch Operations**: Bulk operations for improved performance

#### Monitoring and Observability

**Performance Metrics:**
- **Response Time Tracking**: All commands tracked with millisecond precision
- **Error Rate Monitoring**: Success/failure rates for all operations
- **User Activity Analytics**: Command usage patterns and trends
- **System Resource Monitoring**: Memory, CPU, and database performance

**Logging and Debugging:**
- **Structured Logging**: JSON-formatted logs for analysis
- **Log Levels**: Appropriate logging levels for different environments
- **Error Tracking**: Comprehensive error logging with stack traces
- **Audit Trails**: Complete user interaction history

### Integration Architecture

#### External Service Integration

**Market Data Providers:**
```python
DataProviderIntegration:
├── Yahoo Finance (yfinance) - Primary data source
├── Financial Modeling Prep (FMP) - Professional screening
├── Binance API - Cryptocurrency data
├── Alpha Vantage - Alternative data source
└── Fallback Mechanisms - Multi-provider reliability
```

**Notification Services:**
```python
NotificationIntegration:
├── Telegram Bot API - Primary messaging channel
├── SMTP Email Services - HTML email delivery
├── Chart Generation - Embedded chart creation
└── Template System - Consistent message formatting
```

#### Internal Module Integration

**Clean Architecture Boundaries:**
```python
ModuleIntegration:
├── src.data - Market data retrieval and processing
├── src.common - Shared utilities and analysis functions
├── src.notification - Async notification management
├── src.data.db.services - Database operations and services
└── src.model - Data models and type definitions
```

### Deployment Architecture

#### Development Environment
```
┌─────────────────────────────────────────────────────────────┐
│                Development Environment                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │   Telegram Bot  │    │      Admin Panel                │ │
│  │   (bot.py)      │    │   (Flask Web App)               │ │
│  │   Port: 8080    │    │   Port: 5000                    │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              SQLite Database                            │ │
│  │           (Local Development)                           │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

#### Production Environment
```
┌─────────────────────────────────────────────────────────────┐
│                Production Environment                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │   Telegram Bot  │    │      Admin Panel                │ │
│  │   (Systemd)     │    │   (Nginx + Gunicorn)           │ │
│  │   HTTP API      │    │   Web Interface                 │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            PostgreSQL Database                          │ │
│  │         (Production Scale)                              │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Monitoring & Logging                       │ │
│  │    (Structured Logs + Performance Metrics)             │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Status

### Current Implementation Status

- **Core Bot Framework** ✅ **COMPLETED**
  - aiogram 3.x integration with async/await support
  - Case-insensitive command processing
  - Comprehensive error handling and user feedback
  - HTTP API server for external integrations

- **Command Processing System** ✅ **COMPLETED**
  - Enterprise command parser with flag support
  - Business logic engine with access control
  - JSON configuration system with validation
  - Audit logging with performance tracking

- **Alert Management System** ✅ **COMPLETED**
  - Basic price alerts with CRUD operations
  - Re-arm alert system with crossing detection
  - Indicator-based alerts with JSON configuration
  - Multi-channel notification delivery

- **Reporting System** ✅ **COMPLETED**
  - Technical and fundamental analysis reports
  - Multi-ticker report generation
  - JSON configuration support
  - Email delivery with embedded charts

- **Fundamental Screener** ✅ **COMPLETED**
  - Multi-criteria screening with DCF valuation
  - Market cap category support
  - Composite scoring and recommendations
  - Enhanced screener with FMP integration

- **Administrative System** ✅ **COMPLETED**
  - Web-based admin panel with Flask
  - User management and approval workflow
  - Command audit system with filtering
  - JSON generator tool with templates

- **Scheduling System** ✅ **COMPLETED**
  - Recurring report schedules
  - Screener schedules with criteria
  - JSON configuration support
  - Background task processing

### Testing Implementation Status

- **Unit Testing Coverage** ⚠️ **PARTIALLY IMPLEMENTED**
  - Some existing tests in `src/telegram/tests/`
  - Coverage needs significant expansion
  - Missing comprehensive test suite

- **Integration Testing** ❌ **NOT IMPLEMENTED**
  - No comprehensive integration tests
  - Missing end-to-end workflow testing
  - No API integration testing

- **Performance Testing** ❌ **NOT IMPLEMENTED**
  - No load testing framework
  - Missing performance benchmarks
  - No scalability testing

## Design Decisions

### 1. aiogram Framework Selection

**Decision:** Use aiogram 3.x as the primary Telegram Bot framework

**Rationale:**
- **Modern Async Support**: Native async/await for high concurrency
- **Type Safety**: Full type hints and static analysis support
- **Performance**: Efficient handling of multiple concurrent users
- **Maintainability**: Clean, decorator-based message handling

### 2. Re-Arm Alert System Design

**Decision:** Implement professional-grade crossing detection with automatic re-arming

**Rationale:**
- **User Experience**: Eliminates notification spam while maintaining utility
- **Professional Standards**: Matches behavior of professional trading platforms
- **Configurability**: Flexible parameters for different trading styles
- **Backward Compatibility**: Existing alerts continue working unchanged

### 3. Multi-Channel Notification Architecture

**Decision:** Unified notification system supporting Telegram and Email

**Rationale:**
- **User Choice**: Users can select preferred delivery channels
- **Reliability**: Fallback options if primary channel fails
- **Rich Content**: HTML emails with embedded charts and formatted Telegram messages
- **Consistency**: Same content appropriately formatted for each channel

### 4. JSON Configuration System

**Decision:** Support both traditional flags and JSON configuration

**Rationale:**
- **Power User Support**: Complex configurations for advanced users
- **Ease of Use**: Simple flag-based commands for basic operations
- **Validation**: Comprehensive validation with helpful error messages
- **Extensibility**: Easy addition of new configuration options

### 5. Web-Based Admin Panel

**Decision:** Flask-based web interface for administrative functions

**Rationale:**
- **User Experience**: Visual interface more user-friendly than Telegram commands
- **Functionality**: Rich interface supports complex operations
- **Monitoring**: Real-time dashboards and comprehensive audit views
- **Scalability**: Web interface scales better than Telegram for admin tasks

### 6. FMP Integration for Enhanced Screening

**Decision:** Integrate Financial Modeling Prep API for professional screening

**Rationale:**
- **Performance**: 90% improvement through pre-filtering
- **Professional Quality**: Access to sophisticated screening algorithms
- **Reliability**: Fallback to traditional methods ensures continuous operation
- **Value**: Significant performance and quality improvements justify integration

## Future Enhancements

### Planned Improvements

1. **Advanced Analytics Dashboard**
   - Real-time performance metrics visualization
   - User behavior analytics and insights
   - System health monitoring with alerts
   - Custom reporting and data export

2. **Enhanced Screening Capabilities**
   - Sector-relative analysis and comparisons
   - Custom screening criteria builder
   - Portfolio integration and tracking
   - Risk-adjusted return analysis

3. **Mobile Application Integration**
   - Native mobile app with bot integration
   - Push notifications for alerts
   - Offline capability for reports
   - Enhanced chart visualization

4. **Advanced Alert Features**
   - Multi-condition alerts with complex logic
   - Portfolio-level alerts and monitoring
   - Sentiment analysis integration
   - News-based alert triggers

5. **API Expansion**
   - RESTful API for third-party integrations
   - Webhook support for external systems
   - Real-time data streaming endpoints
   - Developer documentation and SDKs

The Telegram bot system represents a comprehensive, production-ready solution for financial market analysis and alerting, with a strong foundation for future enhancements and scaling.
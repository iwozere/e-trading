# Requirements

## Python Dependencies

### Telegram Bot Framework
- `aiogram >= 3.0.0` - Modern async Telegram Bot API framework
- `aiohttp >= 3.8.5` - Async HTTP client for API requests (aiogram dependency)

### Database and Storage
- `sqlite3` - Built-in SQLite database support (Python standard library)
- `aiosqlite >= 0.19.0` - Async SQLite operations (optional for async DB access)

### Data Processing and Analysis
- `pandas >= 2.0.0` - DataFrame operations for financial data processing
- `numpy >= 1.24.0` - Numerical computations for technical indicators

### Email and Communications
- `aiosmtplib >= 2.0.0` - Async SMTP client for email delivery
- `email-validator >= 2.0.0` - Email address validation

### Command Parsing and Text Processing
- `shlex` - Shell-like command parsing (Python standard library)
- `re` - Regular expressions for text processing (Python standard library)

### Date and Time Handling
- `datetime` - Date and time operations (Python standard library)
- `pytz >= 2023.3` - Timezone handling for global users

### Logging and Monitoring
- `logging` - Built-in logging framework (Python standard library)
- `colorlog >= 6.7.0` - Colored logging output for development

### Testing Framework
- `pytest >= 7.4.0` - Testing framework
- `pytest-asyncio >= 0.21.1` - Async testing support
- `pytest-mock >= 3.11.1` - Mocking utilities for testing

### Data Validation and Serialization
- `dataclasses` - Data structure definitions (Python 3.7+ standard library)
- `typing` - Type hints and annotations (Python standard library)
- `json` - JSON parsing and validation (Python standard library)

## External Dependencies

### Required from Other Modules
- `src.data` - Data downloaders and live feeds for market data
- `src.common` - Shared utilities for OHLCV and fundamental data processing
- `src.model.telegram_bot` - Data models (Fundamentals, TickerAnalysis, etc.)
- `src.notification` - Notification and logging infrastructure
- `src.config` - Configuration management system

### Internal Module Dependencies
```python
# Business Logic Dependencies
from src.telegram.command_parser import ParsedCommand, parse_command
from src.data.db.services import telegram_service as db  # Database operations
from src.common import get_ohlcv  # Market data retrieval
from src.common.fundamentals import get_fundamentals  # Fundamental data
from src.common.technicals import calculate_technicals_from_df  # Technical indicators
from src.common.ticker_analyzer import format_ticker_report  # Report formatting
from src.notification.logger import setup_logger  # Logging infrastructure

# JSON Configuration Dependencies
from src.telegram.screener.report_config_parser import ReportConfigParser, ReportConfig
```

## JSON Configuration Requirements

### Report Command JSON Configuration
- **JSON Schema Validation**: Comprehensive validation of all JSON configuration fields
- **Supported Report Types**: "analysis", "screener", "custom"
- **Data Periods**: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
- **Data Intervals**: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
- **Data Providers**: "yf", "alpha_vantage", "polygon"
- **Technical Indicators**: RSI, MACD, BollingerBands, SMA, EMA, ADX, ATR, Stochastic, WilliamsR
- **Fundamental Indicators**: PE, PB, ROE, ROA, DebtEquity, CurrentRatio, EPS, Revenue, ProfitMargin
- **Fallback Support**: Traditional flag-based parsing when JSON config not provided

### JSON Configuration Features
- **Validation**: Comprehensive field validation with detailed error messages
- **Parsing**: Robust JSON parsing with error handling
- **Summary Generation**: Human-readable configuration summaries
- **Default Configuration**: Helper functions for creating default configurations
- **Example Configurations**: Pre-built examples for common use cases

## External Services
- **Telegram Bot Token** - For Telegram Bot API access
  - Environment variable: `TELEGRAM_BOT_TOKEN`
  - Registration: [BotFather](https://t.me/botfather)
  - Scope: Bot creation, message sending, command handling

### Admin Panel Configuration
- **Admin Login Credentials** - For web-based admin panel access
  - Environment variables: `WEBGUI_LOGIN`, `WEBGUI_PASSWORD`
  - Purpose: Secure access to admin panel for user management and approvals
  - Default: Set strong credentials in production environment

### JSON Generator Tool Configuration
- **Web Framework** - Flask-based web application for JSON generator
  - Dependencies: `Flask >= 2.3.0`, `render_template` for HTML rendering
  - Purpose: Interactive web interface for creating complex JSON configurations
  - Features: Multiple indicators support, real-time generation, template system
- **Frontend Technologies** - Modern web interface for JSON generator
  - HTML5, CSS3, JavaScript (ES6+) for responsive design
  - Real-time JSON generation and validation
  - Copy-to-clipboard functionality
  - Template system for common configurations
- **File Organization** - Template files for JSON generator
  - `templates/json_generator.html` - Main admin panel integration
  - `templates/standalone_json_generator.html` - Standalone shareable version
  - Purpose: Consistent file organization and easy maintenance

### Financial Data APIs
- **Yahoo Finance (yfinance)** - For fundamental and market data
  - No API key required (free tier)
  - Rate limits: Sequential processing to respect limits
  - Data: OHLCV, fundamentals, company info

### Email Service Configuration
- **SMTP Server Credentials** - For email verification and report delivery
  - Environment variables: `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_HOST`, `SMTP_PORT`
  - Supported providers: Gmail, Outlook, SendGrid, Amazon SES
  - Features required: HTML email support, attachment handling

### Telegram Configuration
- **Chat ID** - For admin notifications and broadcasts
  - Environment variable: `TELEGRAM_CHAT_ID` (no longer needed - admin notifications use HTTP API)
  - Purpose: Admin notifications, error reporting, system alerts

### Optional Services
- **Redis** - For advanced caching and session management
  - Environment variables: `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`
  - Use case: High-frequency bot deployments, distributed systems

## Database Requirements

### SQLite Schema (Development/Small Scale)
```sql
-- User management and authentication
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

-- Enhanced price alerts system with re-arm functionality
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    user_id TEXT NOT NULL,
    price REAL NOT NULL,
    condition TEXT NOT NULL, -- 'above' or 'below'
    active INTEGER DEFAULT 1,
    created TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    -- Re-arm system fields
    re_arm_config TEXT,  -- JSON configuration for re-arm behavior
    is_armed INTEGER DEFAULT 1,  -- Whether alert is currently armed
    last_price REAL,  -- Last known price for crossing detection
    last_triggered_at TEXT,  -- ISO timestamp of last trigger
    FOREIGN KEY (user_id) REFERENCES users(telegram_user_id)
);

-- Scheduled reports system
CREATE TABLE schedules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    user_id TEXT NOT NULL,
    scheduled_time TEXT NOT NULL, -- '09:00' format
    period TEXT DEFAULT 'daily', -- 'daily', 'weekly', 'monthly'
    active INTEGER DEFAULT 1,
    email INTEGER DEFAULT 0, -- Send to email flag
    indicators TEXT, -- JSON array of indicators
    interval TEXT DEFAULT '1d',
    provider TEXT,
    created TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(telegram_user_id)
);

-- Verification code tracking
CREATE TABLE codes (
    telegram_user_id TEXT,
    code TEXT,
    sent_time INTEGER
);

-- Global settings
CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Command audit system
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

-- Indexes for audit system performance
CREATE INDEX idx_command_audit_user_id ON command_audit(telegram_user_id);
CREATE INDEX idx_command_audit_created ON command_audit(created);
CREATE INDEX idx_command_audit_command ON command_audit(command);
```

### PostgreSQL Support (Production Scale)
- Migration scripts for SQLite to PostgreSQL
- Connection pooling support
- Async query processing
- Advanced indexing strategies

## System Requirements

### Performance Requirements
- **Response Time**: Bot commands should respond within 3 seconds
- **Throughput**: Support 100+ concurrent users
- **Availability**: 99.9% uptime for production deployments

### Memory Requirements
- **Minimum**: 512MB RAM for development
- **Recommended**: 2GB+ RAM for production
- **Database**: 100MB+ storage for user data and logs

### Network Requirements
- **Bandwidth**: Moderate (image generation and email delivery)
- **Latency**: Low latency preferred for real-time price alerts
- **Connectivity**: Stable internet for Telegram API and email services

## Security Requirements

### Data Protection
- **API Keys**: Secure storage using environment variables
- **Database**: Encryption at rest for sensitive user data
- **Communications**: HTTPS/TLS for all external API calls

### Authentication and Authorization
- **User Verification**: Email-based verification with 6-digit codes
- **Admin Access**: Role-based access control for admin functions
- **User Approval**: Admin approval workflow for restricted features
- **Access Control**: 
  - Admin commands require `is_admin=1` role
  - Restricted commands require `approved=1` status
  - Public commands available to all users
- **Rate Limiting**: Protection against spam and abuse

### Audit and Compliance
- **Command Audit**: Complete tracking of all user commands (registered and non-registered)
- **Performance Monitoring**: Response time tracking for all commands
- **Error Tracking**: Detailed error logging and analysis
- **User Classification**: Distinguish between registered and non-registered users
- **Data Retention**: Audit logs maintained for compliance and debugging
- **Admin Access**: Audit data accessible only through admin panel
- **Privacy Protection**: Email addresses only stored for registered users

### Privacy Compliance
- **Data Retention**: 30-day log retention policy
- **User Data**: GDPR-compliant user data handling
- **Email Privacy**: Secure email storage and verification

## Development Dependencies

### Code Quality Tools
- `black >= 23.7.0` - Code formatting
- `isort >= 5.12.0` - Import sorting
- `flake8 >= 6.0.0` - Code linting
- `mypy >= 1.5.0` - Static type checking

### Development Tools
- `python-dotenv >= 1.0.0` - Environment variable management
- `ipython >= 8.14.0` - Interactive development shell
- `jupyter >= 1.0.0` - Notebook support for data analysis

## Installation Instructions

### Environment Setup
1. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

3. **Environment configuration:**
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your credentials
# TELEGRAM_BOT_TOKEN=your_bot_token
# SMTP_USER=your_email@gmail.com
# SMTP_PASSWORD=your_app_password
```

### Database Initialization
```bash
# Initialize SQLite database
python -c "from src.data.db.telegram_service import init_db; init_db()"

# Or run the bot once to auto-initialize
python src/frontend/telegram/bot.py
```

### Telegram Bot Setup
1. **Create bot with BotFather:**
   - Message [@BotFather](https://t.me/botfather)
   - Use `/newbot` command
   - Set bot name and username
   - Save the bot token

2. **Configure bot settings:**
   - Set bot description and about text
   - Configure command menu
   - Set privacy mode if needed

3. **Set up admin user:**
```bash
# Create admin user with full privileges
python src/util/create_admin.py <telegram_user_id> <email>

# Example:
python src/util/create_admin.py 123456789 admin@example.com
```

4. **Test bot deployment:**
```bash
# Run bot in development mode
python src/frontend/telegram/bot.py

# Test basic commands
# Send /start to your bot in Telegram
```

## Configuration Management

### Environment Variables
```bash
# Telegram Configuration
TELEGRAM_BOT_TOKEN=your_bot_token_here
# TELEGRAM_CHAT_ID is no longer needed - admin notifications use HTTP API

# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Database Configuration (Optional)
DATABASE_URL=sqlite:///db/telegram_screener.sqlite3
# or for PostgreSQL:
# DATABASE_URL=postgresql://user:pass@host:port/dbname

# Optional Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# Development Settings
DEBUG=True
LOG_LEVEL=DEBUG
```

### Default Configuration Values
```python
# Default limits (can be overridden via environment)
DEFAULT_MAX_ALERTS = 5
DEFAULT_MAX_SCHEDULES = 5
DEFAULT_VERIFICATION_CODE_EXPIRY = 3600  # 1 hour
DEFAULT_LOG_RETENTION_DAYS = 30
```

## API Integration Requirements

### Telegram Bot API
- **Version**: Bot API 6.0+
- **Features**: Message handling, inline keyboards, file uploads, dynamic chat routing
- **Rate Limits**: 30 messages per second, 20 requests per minute per chat
- **Reply Handling**: Robust reply mechanism with fallback for invalid message IDs

### Data Provider APIs
- **Integration**: Via `src.data` module
- **Providers**: Yahoo Finance, Alpha Vantage, Binance, etc.
- **Fallback**: Multi-provider failover for reliability

### Email Service Integration
- **SMTP Support**: Standard SMTP with STARTTLS
- **HTML Email**: Rich formatting with embedded images
- **Attachments**: Chart images and report files

## Testing Requirements

### Unit Testing
- **Coverage**: 90%+ code coverage for business logic
- **Framework**: pytest with async support
- **Mocking**: Mock external APIs and services
- **Access Control**: Test admin and user permission scenarios
- **Approval Workflow**: Test user approval and rejection flows

### Integration Testing
- **Database**: Test all CRUD operations
- **API Integration**: Test Telegram API interactions
- **Email**: Test email delivery with mock SMTP

### End-to-End Testing
- **User Flows**: Complete command workflows including registration, verification, and approval
- **Error Handling**: API failures and edge cases
- **Performance**: Load testing with concurrent users
- **Access Control**: Test restricted command access and admin workflows

## Deployment Requirements

### Development Deployment
- **Platform**: Local development machine
- **Database**: SQLite file-based storage
- **Monitoring**: Console logging and debug output

### Production Deployment
- **Platform**: Linux server (Ubuntu 20.04+) or Docker container
- **Database**: PostgreSQL with connection pooling
- **Monitoring**: Structured logging with log aggregation
- **Process Management**: systemd service or Docker Compose
- **Reverse Proxy**: Nginx for webhook endpoints (if used)

### Scalability Considerations
- **Horizontal Scaling**: Stateless bot design for multiple instances
- **Database**: Read replicas for high-load scenarios
- **Caching**: Redis for session and frequently accessed data
- **Load Balancing**: Multiple bot instances behind load balancer

## Monitoring and Observability

### Logging Requirements
- **Structured Logging**: JSON format for log aggregation
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Log Rotation**: Daily rotation with compression
- **Retention**: 30-day retention policy

### Metrics and Monitoring
- **User Metrics**: Active users, command usage, error rates, approval status
- **System Metrics**: Memory usage, database connections, response times
- **Business Metrics**: Reports generated, alerts triggered, email deliveries
- **Admin Metrics**: Approval requests, user management actions, admin command usage

### Error Tracking
- **Exception Handling**: Comprehensive error catching and logging
- **User Notifications**: Friendly error messages for users with clear next steps
- **Admin Alerts**: Critical error notifications to administrators
- **Access Denied**: Clear messaging for unauthorized command attempts

## Backup and Recovery

### Data Backup
- **Database**: Daily automated backups
- **Configuration**: Version-controlled configuration files

## Fundamental Screener Requirements

### Core Screening Functionality
1. **Ticker List Support**
   - US Small Cap stocks (from CSV files)
   - US Medium Cap stocks (from CSV files)
   - US Large Cap stocks (from CSV files)
   - Swiss Shares (SIX exchange via API)
   - Custom ticker lists (user-defined)

2. **Fundamental Data Collection**
   - Sequential processing to respect API rate limits
   - Yahoo Finance (yfinance) as primary data source
   - Comprehensive fundamental metrics collection
   - Error handling for missing or invalid data
   - Skip tickers with insufficient fundamental data

3. **Screening Criteria**
   - **Valuation Ratios**: P/E < 15, P/B < 1.5, P/S < 1, PEG < 1.5
   - **Financial Health**: Debt/Equity < 0.5, Current Ratio > 1.5, Quick Ratio > 1
   - **Profitability**: ROE > 15%, ROA > 5%, Operating Margin > 10%, Profit Margin > 5%
   - **Growth**: Revenue Growth > 5%, Net Income Growth > 5%
   - **Cash Flow**: Positive Free Cash Flow
   - **Dividends**: Dividend Yield > 2% (optional), Payout Ratio < 60%

4. **DCF Valuation**
   - Discounted Cash Flow calculation for fair value estimation
   - Growth rate assumptions based on historical data
   - Discount rate calculation using risk-free rate + beta
   - Terminal value calculation
   - Fair value comparison with current price

5. **Composite Scoring**
   - Multi-factor scoring system (0-10 scale)
   - Weighted combination of valuation, financial health, profitability, and growth metrics
   - Sector-relative scoring (future enhancement)
   - Risk-adjusted scoring

### Report Generation
1. **Summary Report**
   - Top 10 undervalued tickers with key metrics
   - Composite scores and rankings
   - Sector distribution analysis
   - Overall market sentiment summary

2. **Detailed Analysis**
   - Individual company analysis for each top 10 ticker
   - Company overview and sector information
   - Key financial ratios with interpretation
   - Buy/Sell/Hold recommendation with detailed reasoning
   - DCF valuation and fair value estimate
   - Risk assessment and considerations

3. **Report Format**
   - Telegram message formatting with proper markdown
   - Email delivery with HTML formatting
   - Chart generation for key metrics visualization
   - Export options (future enhancement)

### Command Interface
1. **Screener Command Structure**
   - `/schedules screener LIST_TYPE [TIME] [flags]`
   - Support for all existing schedule flags (-email, -indicators, etc.)
   - Custom indicator selection for focused screening
   - Custom threshold specification (future enhancement)

2. **List Type Support**
   - `us_small_cap`: US Small Cap stocks
   - `us_medium_cap`: US Medium Cap stocks
   - `us_large_cap`: US Large Cap stocks
   - `swiss_shares`: Swiss SIX exchange stocks
   - `custom_list`: Custom ticker list (prompted during creation)

### Performance and Scalability
1. **Processing Strategy**
   - Sequential processing to respect API rate limits
   - Progress tracking and user feedback
   - Timeout handling for long-running operations
   - Partial result reporting for large lists

2. **Caching Strategy** (Future Enhancement)
   - Cache fundamental data to reduce API calls
   - Cache screening results for repeated requests
   - Cache invalidation based on data freshness
   - Redis-based caching for distributed deployments

3. **Error Handling**
   - Graceful handling of API failures
   - Skip tickers with missing data
   - Log errors for debugging and monitoring
   - User-friendly error messages

### Future Enhancements
1. **Advanced Screening**
   - Sector-average comparisons
   - Percentile rankings within peer groups
   - Custom user-defined screening criteria
   - Multi-factor model integration

2. **Portfolio Integration**
   - Track screened stocks in user portfolios
   - Performance monitoring of screened stocks
   - Rebalancing recommendations
   - Risk-adjusted return analysis

3. **Export and Integration**
   - CSV/Excel export of screening results
   - API endpoints for external integrations
   - Webhook notifications for new opportunities
   - Third-party portfolio management integration
- **Logs**: Archive to long-term storage

### Disaster Recovery
- **Recovery Time**: Target 4-hour recovery time
- **Data Loss**: Maximum 24-hour data loss acceptable
- **Failover**: Automated failover to backup instances

## Access Control and User Management

### User Registration Workflow
1. **Registration**: User sends `/register email@example.com`
2. **Verification**: User receives 6-digit code via email and verifies with `/verify CODE`
3. **Approval Request**: User sends `/request_approval` to request access to restricted features
4. **Admin Review**: Admin receives notification and reviews the request
5. **Approval/Rejection**: Admin uses `/admin approve USER_ID` or `/admin reject USER_ID`
6. **Access Granted**: User receives notification and can use restricted commands

### Command Access Levels
- **Public Commands** (No restrictions):
  - `/start`, `/help` - Show welcome and help messages
  - `/info` - Show user status and approval information
  - `/register`, `/verify` - Email registration and verification
  - `/request_approval` - Request admin approval
  - `/feedback`, `/feature` - Send feedback and feature requests

- **Restricted Commands** (Require `approved=1`):
  - `/report` - Generate ticker reports
  - `/alerts` - Manage price alerts
  - `/schedules` - Manage scheduled reports
  - `/language` - Change language preference

- **Admin Commands** (Require `is_admin=1`):
  - `/admin users` - List all users
  - `/admin approve USER_ID` - Approve user for restricted features
  - `/admin reject USER_ID` - Reject user's approval request
  - `/admin pending` - List users waiting for approval
  - `/admin broadcast MESSAGE` - Send broadcast message to all users

### Admin Management
- **Admin Setup**: Use `src/util/create_admin.py` script to create admin users
- **User Management**: Admins can approve, reject, and manage user access
- **System Monitoring**: Admins receive notifications for approval requests and system events

## Re-Arm Alert System Requirements

### Core Re-Arm Functionality
- **Crossing Detection**: Alerts trigger only when price crosses threshold (not just exceeds)
- **Automatic Re-Arming**: Alert re-arms when price moves back across hysteresis level
- **Hysteresis Configuration**: Configurable buffer to prevent noise (percentage, fixed, or ATR-based)
- **Cooldown Management**: Minimum time between triggers to prevent spam
- **State Persistence**: Maintain alert state across system restarts

### Re-Arm Configuration Options
- **Hysteresis Types**: 
  - `percentage`: Percentage of threshold (e.g., 0.25%)
  - `fixed`: Fixed dollar amount (e.g., $0.50)
  - `atr`: ATR-based dynamic adjustment (future enhancement)
- **Cooldown Periods**: Configurable minutes between triggers (default: 15)
- **Persistence Requirements**: Number of bars condition must persist (default: 1)
- **Price Evaluation**: Close-only or all price data options

### Default Re-Arm Settings
```json
{
  "enabled": true,
  "hysteresis": 0.25,
  "hysteresis_type": "percentage",
  "cooldown_minutes": 15,
  "persistence_bars": 1,
  "close_only": false
}
```

### Migration Requirements
- **Backward Compatibility**: Existing alerts continue working unchanged
- **Safe Migration**: Conversion script with rollback capability
- **Data Integrity**: Preserve all existing alert data during migration
- **Verification**: Post-migration validation of all alerts

### JSON Generator Requirements
- **Interactive UI**: Web-based interface for creating complex configurations
- **Real-time Generation**: Live JSON preview as user configures options
- **Template System**: Pre-built templates for common alert scenarios
- **Validation**: Client-side and server-side JSON validation
- **Integration**: Seamless integration with admin panel

## Recent Architectural Improvements

### Re-Arm Alert System Implementation
- **Problem Solved**: Alert spam when price stays above/below threshold
- **Solution**: Professional-grade crossing detection with automatic re-arming
- **Benefits**: 
  - Eliminates notification spam while maintaining utility
  - Matches professional trading platform behavior
  - Configurable sensitivity for different trading styles
  - Backward compatible with existing alerts

### Case-Insensitive Command Processing
- **Problem Solved**: Commands were case-sensitive, causing user confusion and errors
- **Solution**: Implemented intelligent case-insensitive command parsing
- **Benefits**: 
  - All commands work regardless of case (e.g., `/REPORT`, `/Report`, `/report`)
  - Tickers automatically converted to uppercase for consistency
  - Actions converted to lowercase for internal processing
  - Improved user experience and reduced support requests

### Web-Based Admin Panel
- **Problem Solved**: Admin functions only available via Telegram commands
- **Solution**: Implemented comprehensive web-based admin interface
- **Benefits**:
  - User-friendly approval workflow with visual interface
  - Real-time dashboard with system statistics
  - Bulk user management capabilities
  - Enhanced monitoring and reporting features

### JSON Generator Tool
- **Problem Solved**: Complex JSON configurations difficult to create manually
- **Solution**: Implemented interactive web-based JSON generator with multiple indicators support
- **Benefits**:
  - User-friendly interface for creating complex configurations
  - Support for multiple indicators with AND/OR logic across all tabs
  - Real-time JSON generation and command preview
  - Template system for common configurations
  - Standalone shareable version for external users
  - Dynamic parameter configuration for all technical indicators
  - Copy-to-clipboard and validation features
  - Integration with admin panel as new tab

### Dynamic Chat Routing
- **Problem Solved**: Bot responses were sent to fixed admin chat instead of user's chat
- **Solution**: Implemented dynamic chat ID routing in notification system
- **Benefits**: Users receive responses in their own chat, improved user experience

### Robust Reply Handling
- **Problem Solved**: Telegram API errors when replying to invalid message IDs
- **Solution**: Implemented try-catch fallback mechanism for reply operations
- **Benefits**: Graceful error handling, no more bot crashes from reply failures

### Price Consistency Fix
- **Problem Solved**: Price discrepancy between fundamental and technical analysis sections
- **Solution**: Updated technical analysis to use actual current price instead of moving average
- **Benefits**: Consistent pricing information across all report sections

### Access Control System
- **Problem Solved**: No user approval workflow for restricted features
- **Solution**: Implemented multi-tier access control with admin approval workflow
- **Benefits**: Secure access to restricted features, admin oversight
- **Technical Details**: See `Design.md` for complete architecture documentation

### Access Control System
- **Problem Solved**: No user approval workflow for restricted features
- **Solution**: Implemented multi-tier access control with admin approval workflow
- **Benefits**: Secure access to restricted features, admin oversight
- **Technical Details**: See `Design.md` for complete architecture documentation

## Compliance and Legal

### Data Privacy
- **GDPR Compliance**: User data handling and deletion
- **Data Minimization**: Collect only necessary user data
- **Consent**: Clear user consent for data processing

### Financial Data
- **Disclaimers**: Investment advice disclaimers
- **Data Accuracy**: Best-effort data accuracy notifications
- **Rate Limiting**: Respect third-party API terms of service

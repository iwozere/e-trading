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
from src.frontend.telegram.command_parser import ParsedCommand, parse_command
from src.frontend.telegram import db  # Database operations
from src.common import get_ohlcv  # Market data retrieval
from src.common.fundamentals import get_fundamentals  # Fundamental data
from src.common.technicals import calculate_technicals_from_df  # Technical indicators
from src.common.ticker_analyzer import format_ticker_report  # Report formatting
from src.notification.logger import setup_logger  # Logging infrastructure
```

## External Services

### Required API Keys (Production)
- **Telegram Bot Token** - For Telegram Bot API access
  - Environment variable: `TELEGRAM_BOT_TOKEN`
  - Registration: [BotFather](https://t.me/botfather)
  - Scope: Bot creation, message sending, command handling

### Email Service Configuration
- **SMTP Server Credentials** - For email verification and report delivery
  - Environment variables: `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_HOST`, `SMTP_PORT`
  - Supported providers: Gmail, Outlook, SendGrid, Amazon SES
  - Features required: HTML email support, attachment handling

### Telegram Configuration
- **Chat ID** - For admin notifications and broadcasts
  - Environment variable: `TELEGRAM_CHAT_ID`
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
    language TEXT DEFAULT 'en',
    is_admin INTEGER DEFAULT 0,
    max_alerts INTEGER DEFAULT 5,
    max_schedules INTEGER DEFAULT 5
);

-- Price alerts system
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    user_id TEXT NOT NULL,
    price REAL NOT NULL,
    condition TEXT NOT NULL, -- 'above' or 'below'
    active INTEGER DEFAULT 1,
    created TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
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
- **Rate Limiting**: Protection against spam and abuse

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
python -c "from src.frontend.telegram.db import init_db; init_db()"

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

3. **Test bot deployment:**
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
TELEGRAM_CHAT_ID=your_admin_chat_id

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
- **Features**: Message handling, inline keyboards, file uploads
- **Rate Limits**: 30 messages per second, 20 requests per minute per chat

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

### Integration Testing
- **Database**: Test all CRUD operations
- **API Integration**: Test Telegram API interactions
- **Email**: Test email delivery with mock SMTP

### End-to-End Testing
- **User Flows**: Complete command workflows
- **Error Handling**: API failures and edge cases
- **Performance**: Load testing with concurrent users

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
- **User Metrics**: Active users, command usage, error rates
- **System Metrics**: Memory usage, database connections, response times
- **Business Metrics**: Reports generated, alerts triggered, email deliveries

### Error Tracking
- **Exception Handling**: Comprehensive error catching and logging
- **User Notifications**: Friendly error messages for users
- **Admin Alerts**: Critical error notifications to administrators

## Backup and Recovery

### Data Backup
- **Database**: Daily automated backups
- **Configuration**: Version-controlled configuration files
- **Logs**: Archive to long-term storage

### Disaster Recovery
- **Recovery Time**: Target 4-hour recovery time
- **Data Loss**: Maximum 24-hour data loss acceptable
- **Failover**: Automated failover to backup instances

## Compliance and Legal

### Data Privacy
- **GDPR Compliance**: User data handling and deletion
- **Data Minimization**: Collect only necessary user data
- **Consent**: Clear user consent for data processing

### Financial Data
- **Disclaimers**: Investment advice disclaimers
- **Data Accuracy**: Best-effort data accuracy notifications
- **Rate Limiting**: Respect third-party API terms of service

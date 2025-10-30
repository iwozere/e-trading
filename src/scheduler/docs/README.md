# Scheduler Service

The Scheduler Service is the main APScheduler-based job scheduling and execution service for the Advanced Trading Framework. It provides centralized scheduling with database persistence, alert evaluation, and notification integration.

## Features

- **APScheduler Integration**: Uses APScheduler with PostgreSQL job store for reliable job scheduling
- **Alert Evaluation**: Centralized alert evaluation with rule-based logic and rearm functionality
- **Notification Integration**: Seamless integration with the notification service for alert delivery
- **Configuration Management**: Environment-based configuration with validation
- **Dependency Injection**: Clean service architecture with proper dependency injection
- **Error Handling**: Comprehensive error handling with retry logic and graceful degradation
- **Health Monitoring**: Built-in health checks and status reporting
- **Deployment Ready**: Docker and systemd configurations included

## Quick Start

### Using the CLI

```bash
# Start the scheduler service
python -m src.scheduler.cli start

# Check service status
python -m src.scheduler.cli status

# Reload schedules from database
python -m src.scheduler.cli reload

# Validate configuration
python -m src.scheduler.cli validate
```

### Using the Main Module

```python
import asyncio
from src.scheduler.main import SchedulerApplication
from src.scheduler.config import SchedulerServiceConfig

async def main():
    config = SchedulerServiceConfig()
    app = SchedulerApplication(config)
    
    await app.start()
    await app.wait_for_shutdown()

asyncio.run(main())
```

## Configuration

The scheduler service supports configuration through environment variables:

### Database Configuration
- `DATABASE_URL`: Database connection URL
- `DB_POOL_SIZE`: Database connection pool size (default: 10)
- `DB_MAX_OVERFLOW`: Maximum connection overflow (default: 20)
- `SQL_ECHO`: Enable SQL query logging (default: false)

### Scheduler Configuration
- `SCHEDULER_MAX_WORKERS`: Maximum worker threads (default: 10)
- `SCHEDULER_JOB_TIMEOUT`: Job timeout in seconds (default: 300)
- `SCHEDULER_TIMEZONE`: Scheduler timezone (default: UTC)

### Notification Configuration
- `NOTIFICATION_SERVICE_URL`: Notification service URL (default: http://localhost:8000)
- `NOTIFICATION_TIMEOUT`: Notification timeout in seconds (default: 30)
- `NOTIFICATION_RETRIES`: Maximum notification retries (default: 3)
- `NOTIFICATION_ENABLED`: Enable notifications (default: true)

### Data Configuration
- `DATA_CACHE_ENABLED`: Enable data caching (default: true)
- `DATA_CACHE_TTL`: Cache TTL in seconds (default: 300)
- `DATA_DEFAULT_LOOKBACK`: Default lookback period (default: 200)
- `DATA_MAX_RETRIES`: Maximum data fetch retries (default: 3)
- `DATA_TIMEOUT`: Data fetch timeout in seconds (default: 60)

### Alert Configuration
- `ALERT_SCHEMA_DIR`: Alert schema directory (default: src/common/alerts/schemas)
- `ALERT_DEFAULT_LOOKBACK`: Default alert lookback (default: 200)
- `ALERT_MAX_EVAL_TIME`: Maximum evaluation time in seconds (default: 120)
- `ALERT_ONCE_PER_BAR`: Enable once-per-bar evaluation (default: true)

### Service Configuration
- `TRADING_ENV`: Environment (development/staging/production)
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `HEALTH_CHECK_INTERVAL`: Health check interval in seconds (default: 60)

## Architecture

The scheduler service follows a clean architecture pattern with dependency injection:

```
SchedulerApplication
├── SchedulerService (APScheduler orchestration)
├── AlertEvaluator (Alert processing)
├── DataManager (Market data)
├── IndicatorService (Technical indicators)
├── JobsService (Database operations)
├── NotificationServiceClient (Notifications)
└── AlertSchemaValidator (Configuration validation)
```

## Integration

### Database Integration
- Uses existing JobsService for database operations
- Maintains transactional consistency for state updates
- Supports both SQLite and PostgreSQL

### Market Data Integration
- Uses DataManager for provider-agnostic data access
- Implements provider failover for reliability
- Caches frequently accessed data

### Notification Integration
- Calls TelegramBot FastAPI `/api/notify` endpoint
- Includes comprehensive alert context in notifications
- Handles delivery failures gracefully

## Deployment

### Docker Deployment

```bash
# Build and run with docker-compose
cd src/scheduler/deployment
docker-compose up -d
```

### Systemd Service

```bash
# Copy service file
sudo cp src/scheduler/deployment/scheduler.service /etc/systemd/system/

# Enable and start service
sudo systemctl enable scheduler.service
sudo systemctl start scheduler.service

# Check status
sudo systemctl status scheduler.service
```

### Manual Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export TRADING_ENV=production
export DATABASE_URL=postgresql://user:pass@localhost/trading_db

# Start service
python -m src.scheduler.main
```

## Monitoring

The scheduler service provides several monitoring endpoints and capabilities:

- **Health Checks**: Built-in health check functionality
- **Status Reporting**: Comprehensive status information
- **Logging**: Structured logging with configurable levels
- **Metrics**: Job execution metrics and timing

## Error Handling

The service implements comprehensive error handling:

- **Job Execution**: 3 retries with exponential backoff
- **Data Retrieval**: Provider failover with 2 retries per provider
- **Notifications**: Handled by AsyncNotificationManager (5 retries)
- **Database**: Connection pooling with automatic retry

## Development

### Running Tests

```bash
# Run unit tests
python -m pytest src/scheduler/tests/

# Run integration tests
python -m pytest src/scheduler/tests/integration/

# Run with coverage
python -m pytest --cov=src.scheduler src/scheduler/tests/
```

### Development Environment

```bash
# Set development environment
export TRADING_ENV=development
export LOG_LEVEL=DEBUG

# Start with hot reload (if using watchdog)
python -m src.scheduler.cli start --env development
```

## Related Documentation

- [Requirements](docs/Requirements.md) - Technical requirements
- [Design](docs/Design.md) - Architecture and design decisions
- [Tasks](docs/Tasks.md) - Implementation roadmap
- [API Documentation](docs/API.md) - API reference
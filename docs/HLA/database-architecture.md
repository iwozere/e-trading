# Database Architecture Documentation

## Overview

The Advanced Trading Framework uses PostgreSQL as its primary database with a well-structured schema supporting user management, trading operations, job scheduling, communication systems, and web UI functionality. The architecture follows the Repository pattern with a Service Layer for clean separation of concerns.

## Database Schema Overview

### Entity Relationship Diagram

```mermaid
erDiagram
    USR_USERS {
        int id PK
        varchar email
        varchar role
        bool is_active
        timestamptz created_at
        timestamptz updated_at
        timestamptz last_login
    }

    USR_AUTH_IDENTITIES {
        int id PK
        int user_id FK
        varchar provider
        varchar external_id
        jsonb metadata
        timestamptz created_at
    }

    USR_VERIFICATION_CODES {
        int id PK
        int user_id FK
        varchar code
        int sent_time
        varchar provider
        timestamptz created_at
    }

    JOB_SCHEDULES {
        int id PK
        int user_id
        varchar name
        varchar job_type
        varchar target
        jsonb task_params
        varchar cron
        bool enabled
        timestamptz next_run_at
        timestamptz created_at
        timestamptz updated_at
    }

    JOB_SCHEDULE_RUNS {
        int id PK
        text job_type
        int job_id FK
        bigint user_id
        text status
        timestamptz scheduled_for
        timestamptz enqueued_at
        timestamptz started_at
        timestamptz finished_at
        jsonb job_snapshot
        jsonb result
        text error
        varchar worker_id
    }

    TRADING_BOTS {
        int id PK
        int user_id FK
        varchar type
        varchar status
        timestamptz started_at
        timestamptz last_heartbeat
        int error_count
        numeric current_balance
        numeric total_pnl
        jsonb extra_metadata
        jsonb config
        text description
        timestamptz created_at
        timestamptz updated_at
    }

    TRADING_POSITIONS {
        int id PK
        int bot_id FK
        varchar trade_type
        varchar symbol
        varchar direction
        timestamptz opened_at
        timestamptz closed_at
        numeric qty_open
        numeric avg_price
        numeric realized_pnl
        varchar status
        jsonb extra_metadata
    }

    TRADING_TRADES {
        int id PK
        int bot_id FK
        int position_id FK
        varchar trade_type
        varchar strategy_name
        varchar entry_logic_name
        varchar exit_logic_name
        varchar symbol
        varchar interval
        timestamptz entry_time
        timestamptz exit_time
        numeric entry_price
        numeric exit_price
        numeric size
        varchar direction
        numeric commission
        numeric gross_pnl
        numeric net_pnl
        numeric pnl_percentage
        varchar exit_reason
        varchar status
        jsonb extra_metadata
        timestamptz created_at
        timestamptz updated_at
    }

    TRADING_PERFORMANCE_METRICS {
        int id PK
        int bot_id FK
        varchar trade_type
        varchar symbol
        varchar interval
        varchar entry_logic_name
        varchar exit_logic_name
        jsonb metrics
        timestamptz calculated_at
        timestamptz created_at
    }

    TELEGRAM_COMMAND_AUDITS {
        int id PK
        varchar telegram_user_id
        varchar command
        text full_message
        bool is_registered_user
        varchar user_email
        bool success
        text error_message
        int response_time_ms
        timestamptz created_at
    }

    TELEGRAM_BROADCAST_LOGS {
        int id PK
        text message
        varchar sent_by
        int success_count
        int total_count
        timestamptz created_at
    }

    TELEGRAM_FEEDBACKS {
        int id PK
        int user_id FK
        varchar type
        text message
        varchar status
        timestamptz created_at
    }

    TELEGRAM_SETTINGS {
        varchar key PK
        text value
    }

    WEBUI_AUDIT_LOGS {
        int id PK
        int user_id FK
        varchar action
        varchar resource_type
        varchar resource_id
        jsonb details
        varchar ip_address
        varchar user_agent
        timestamptz created_at
    }

    WEBUI_PERFORMANCE_SNAPSHOTS {
        int id PK
        varchar strategy_id
        timestamptz timestamp
        jsonb pnl
        jsonb positions
        int trades_count
        jsonb win_rate
        jsonb drawdown
        jsonb metrics
    }

    WEBUI_STRATEGY_TEMPLATES {
        int id PK
        varchar name
        text description
        jsonb template_data
        bool is_public
        int created_by FK
        timestamptz created_at
        timestamptz updated_at
    }

    WEBUI_SYSTEM_CONFIG {
        int id PK
        varchar key
        jsonb value
        text description
        timestamptz created_at
        timestamptz updated_at
    }

    %% Relationships
    USR_USERS ||--o{ USR_AUTH_IDENTITIES : "has"
    USR_USERS ||--o{ USR_VERIFICATION_CODES : "has"
    USR_USERS ||--o{ TRADING_BOTS : "owns"
    USR_USERS ||--o{ TELEGRAM_FEEDBACKS : "sends"
    USR_USERS ||--o{ WEBUI_AUDIT_LOGS : "performs"
    USR_USERS ||--o{ WEBUI_STRATEGY_TEMPLATES : "creates"

    JOB_SCHEDULES ||--o{ JOB_SCHEDULE_RUNS : "executes"

    TRADING_BOTS ||--o{ TRADING_POSITIONS : "holds"
    TRADING_BOTS ||--o{ TRADING_TRADES : "executes"
    TRADING_BOTS ||--o{ TRADING_PERFORMANCE_METRICS : "generates"

    TRADING_POSITIONS ||--o{ TRADING_TRADES : "contains"
```

## Core Database Modules

### 1. User Management Module

**Tables:**
- `usr_users`: Core user accounts with role-based access control
- `usr_auth_identities`: Multi-provider authentication (Telegram, email, etc.)
- `usr_verification_codes`: Temporary verification codes for user authentication

**Key Features:**
- Multi-provider authentication support
- Role-based access control (admin, trader, viewer)
- Telegram integration with verification system
- Flexible metadata storage in JSONB fields

**Constraints:**
- Email uniqueness across users
- Role validation (admin, trader, viewer)
- Foreign key cascading for data integrity

### 2. Job Scheduling Module

**Tables:**
- `job_schedules`: Persistent job definitions with cron scheduling
- `job_schedule_runs`: Execution history and status tracking

**Key Features:**
- Cron-based job scheduling
- Job type categorization (report, screener, alert, notification, data_processing, backup)
- Execution history with snapshots
- User-specific job isolation
- Failure tracking and retry mechanisms

**Job Types:**
- `report`: Generate trading reports
- `screener`: Run stock/crypto screening
- `alert`: Send trading alerts
- `notification`: User notifications
- `data_processing`: Background data tasks
- `backup`: System backup operations

### 3. Trading Engine Module

**Tables:**
- `trading_bots`: Trading bot configurations and status
- `trading_positions`: Open and closed trading positions
- `trading_trades`: Individual trade records with full lifecycle
- `trading_performance_metrics`: Aggregated performance analytics

**Key Features:**
- Multi-bot support per user
- Position and trade lifecycle management
- Performance metrics calculation
- Strategy pattern tracking (entry/exit logic)
- Real-time status monitoring with heartbeat system

**Trading Lifecycle:**
1. Bot creation with configuration
2. Position opening with entry logic
3. Trade execution and monitoring
4. Position closing with exit logic
5. Performance metrics calculation

### 4. Communication Module

**Tables:**
- `telegram_command_audits`: Command execution tracking
- `telegram_broadcast_logs`: Message broadcast history
- `telegram_feedbacks`: User feedback collection
- `telegram_settings`: System-wide Telegram configuration
- `msg_messages`: Notification message queue
- `msg_delivery_status`: Message delivery tracking
- `msg_system_health`: System and component health monitoring
- `msg_rate_limits`: Per-user rate limiting
- `msg_channel_configs`: Channel configuration management

**Key Features:**
- Command audit trail with performance metrics
- Broadcast message tracking
- User feedback system
- Configurable Telegram settings
- Multi-channel notification system with queuing
- Comprehensive system health monitoring
- Per-user rate limiting and delivery tracking

**System Health Monitoring:**
The `msg_system_health` table provides comprehensive health monitoring for all platform components:

```sql
CREATE TABLE msg_system_health (
    id BIGSERIAL PRIMARY KEY,
    system VARCHAR(50) NOT NULL,              -- System name (notification, telegram_bot, etc.)
    component VARCHAR(100),                   -- Component name (email, slack, etc.) - optional
    status VARCHAR(20) NOT NULL,              -- HEALTHY, DEGRADED, DOWN, UNKNOWN
    last_success TIMESTAMPTZ,                 -- Last successful health check
    last_failure TIMESTAMPTZ,                 -- Last failed health check
    failure_count INTEGER NOT NULL DEFAULT 0, -- Consecutive failure count
    avg_response_time_ms INTEGER,             -- Average response time
    error_message TEXT,                       -- Error details
    metadata TEXT,                            -- JSON metadata
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW() -- Last check timestamp
);
```

**Monitored Systems:**
- `notification`: Main notification service and individual channels (email, telegram, sms, slack)
- `telegram_bot`: Telegram bot health and connectivity
- `api_service`: REST API service health
- `web_ui`: Web UI frontend and backend health
- `trading_bot`: Trading bot execution health
- `database`: Database connectivity and performance

### 5. Web UI Module

**Tables:**
- `webui_audit_logs`: User action tracking
- `webui_performance_snapshots`: Strategy performance caching
- `webui_strategy_templates`: Reusable strategy configurations
- `webui_system_config`: Application configuration

**Key Features:**
- User action auditing with IP and user agent tracking
- Performance snapshot caching for fast UI rendering
- Strategy template sharing system
- Dynamic system configuration

## Data Access Patterns

### Repository Pattern Implementation

The system implements the Repository pattern for clean data access abstraction:

```python
# Base repository structure
class BaseRepository:
    def __init__(self, session: Session):
        self.session = session
    
    def get_by_id(self, id: int) -> Optional[Model]:
        """Generic get by ID implementation"""
        return self.session.query(self.model).filter(self.model.id == id).first()
    
    def create(self, **kwargs) -> Model:
        """Generic create implementation"""
        instance = self.model(**kwargs)
        self.session.add(instance)
        self.session.commit()
        return instance
    
    def update(self, id: int, **kwargs) -> Optional[Model]:
        """Generic update implementation"""
        instance = self.get_by_id(id)
        if instance:
            for key, value in kwargs.items():
                setattr(instance, key, value)
            self.session.commit()
        return instance
    
    def delete(self, id: int) -> bool:
        """Generic delete implementation"""
        instance = self.get_by_id(id)
        if instance:
            self.session.delete(instance)
            self.session.commit()
            return True
        return False
```

**Repository Classes:**
- `UsersRepo`: User and authentication management
- `TradingRepo`: Trading operations and bot management
- `JobsRepo`: Job scheduling and execution
- `TelegramRepo`: Telegram-specific operations
- `WebUIRepo`: Web interface data operations

### Service Layer Architecture

The Service Layer provides business logic abstraction over repositories:

```python
# Service layer structure
class UserService:
    def __init__(self, uow: UnitOfWork):
        self.uow = uow
    
    def authenticate_user(self, provider: str, external_id: str) -> User:
        """Business logic for user authentication"""
        with self.uow:
            user = self.uow.users.get_by_provider(provider, external_id)
            if not user:
                user = self.uow.users.create(
                    provider=provider,
                    external_id=external_id,
                    is_active=True
                )
            return user
    
    def create_telegram_user(self, telegram_id: str, **profile) -> User:
        """Telegram-specific user creation logic"""
        with self.uow:
            return self.uow.users.create(
                provider="telegram",
                external_id=telegram_id,
                is_active=True,
                **profile
            )
```

**Service Classes:**
- `UsersService`: User management and authentication
- `TradingService`: Trading bot and position management
- `JobsService`: Job scheduling and execution
- `TelegramService`: Telegram bot integration
- `WebUIService`: Web interface business logic

### Unit of Work Pattern

The system uses Unit of Work pattern for transaction management:

```python
class UnitOfWork:
    def __init__(self, session: Session):
        self.session = session
        self.users = UsersRepo(session)
        self.trading = TradingRepo(session)
        self.jobs = JobsRepo(session)
        self.telegram = TelegramRepo(session)
        self.webui = WebUIRepo(session)
    
    def commit(self):
        self.session.commit()
    
    def rollback(self):
        self.session.rollback()
```

## Database Configuration

### Connection Management

```python
# Database URL configuration
DB_URL = os.getenv("DB_URL", "postgresql://user:pass@localhost/trading_db")

# Engine configuration
engine = create_engine(
    DB_URL,
    future=True,
    pool_pre_ping=True,
    echo=SQL_ECHO,
    pool_size=20,
    max_overflow=30
)

# Session factory
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True
)
```

### Connection Pooling

- **Pool Size**: 20 connections
- **Max Overflow**: 30 additional connections
- **Pool Pre-ping**: Enabled for connection health checks
- **Connection Timeout**: 30 seconds

### Transaction Management

```python
@contextmanager
def session_scope() -> Session:
    """Transaction-scoped session pattern."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

## Indexing Strategy

### Primary Indexes

All tables use auto-incrementing integer primary keys with B-tree indexes for optimal performance.

### Secondary Indexes

**User Management:**
- `ix_users_email`: Email lookup optimization
- `ix_auth_identities_provider_external`: Multi-column index for provider authentication
- `ix_verification_codes_user_id`: User verification lookup

**Job Scheduling:**
- `idx_schedules_enabled`: Active job filtering
- `idx_schedules_next_run_at`: Next execution time queries
- `ux_runs_job_scheduled_for`: Unique constraint preventing duplicate runs

**Trading Operations:**
- `ix_trading_positions_bot_id`: Bot position queries
- `ix_trading_trades_symbol`: Symbol-based trade filtering
- `ix_trading_trades_strategy_name`: Strategy performance analysis
- `ix_trading_performance_metrics_calculated_at`: Time-based metrics queries

**Communication:**
- `ix_telegram_command_audits_telegram_user_id`: User command history
- `ix_telegram_command_audits_success`: Success rate analysis

**Web UI:**
- `ix_webui_audit_logs_user_id`: User action tracking
- `ix_webui_performance_snapshots_strategy_id`: Strategy performance lookup

## Data Integrity Constraints

### Foreign Key Constraints

- **Cascading Deletes**: User deletion cascades to related authentication and verification records
- **Set Null**: Trade position references set to null when position is deleted
- **Restrict**: Bot deletion restricted if active trades exist

### Check Constraints

- **Role Validation**: User roles limited to 'admin', 'trader', 'viewer'
- **Job Type Validation**: Job types restricted to predefined values
- **Status Validation**: Trading and job statuses validated against allowed values

### Unique Constraints

- **User Email**: Unique across all users
- **Provider Authentication**: Unique provider/external_id combinations
- **Job Scheduling**: Unique job/scheduled_for combinations prevent duplicates
- **System Configuration**: Unique configuration keys

## Performance Optimization

### Query Optimization

1. **Prepared Statements**: All queries use parameterized statements
2. **Index Usage**: Strategic indexing for common query patterns
3. **Query Planning**: Regular EXPLAIN ANALYZE for performance monitoring
4. **Connection Pooling**: Efficient connection reuse

### Data Partitioning

**Time-based Partitioning** (planned):
- `trading_trades`: Partition by month for historical data
- `telegram_command_audits`: Partition by month for audit logs
- `webui_audit_logs`: Partition by month for user actions

### Caching Strategy

1. **Application-level Caching**: Redis for frequently accessed data
2. **Query Result Caching**: Strategic caching of expensive queries
3. **Session Caching**: User session data caching
4. **Configuration Caching**: System configuration in memory

## Backup and Recovery

### Backup Strategy

1. **Full Backups**: Daily full database backups
2. **Incremental Backups**: Hourly transaction log backups
3. **Point-in-time Recovery**: WAL archiving for precise recovery
4. **Cross-region Replication**: Disaster recovery replicas

### Recovery Procedures

1. **Automated Recovery**: Scripts for common recovery scenarios
2. **Data Validation**: Post-recovery data integrity checks
3. **Rollback Procedures**: Safe rollback for failed deployments
4. **Testing**: Regular recovery testing procedures

## Security Considerations

### Access Control

1. **Role-based Access**: Database roles matching application roles
2. **Connection Security**: SSL/TLS encryption for all connections
3. **Credential Management**: Secure credential storage and rotation
4. **Audit Logging**: Comprehensive database activity logging

### Data Protection

1. **Sensitive Data**: Encryption for sensitive fields
2. **PII Handling**: Proper handling of personally identifiable information
3. **Data Retention**: Automated cleanup of old audit logs
4. **Compliance**: GDPR and financial regulation compliance

## Monitoring and Maintenance

### Performance Monitoring

1. **Query Performance**: Slow query logging and analysis
2. **Connection Monitoring**: Pool utilization tracking
3. **Index Usage**: Regular index effectiveness analysis
4. **Resource Usage**: CPU, memory, and disk monitoring

### Maintenance Tasks

1. **Index Maintenance**: Regular REINDEX operations
2. **Statistics Updates**: Automated ANALYZE operations
3. **Vacuum Operations**: Regular VACUUM and VACUUM FULL
4. **Log Rotation**: Automated log file management

## Migration Strategy

### Schema Migrations

1. **Version Control**: All schema changes in version control
2. **Migration Scripts**: Automated migration execution
3. **Rollback Support**: Safe rollback for failed migrations
4. **Testing**: Migration testing in staging environments

### Data Migrations

1. **Batch Processing**: Large data migrations in batches
2. **Progress Tracking**: Migration progress monitoring
3. **Validation**: Post-migration data validation
4. **Downtime Minimization**: Zero-downtime migration strategies

## Future Enhancements

### Planned Improvements

1. **Read Replicas**: Read-only replicas for reporting queries
2. **Sharding**: Horizontal partitioning for large datasets
3. **Time-series Data**: Specialized storage for trading data
4. **Real-time Analytics**: Stream processing integration

### Scalability Considerations

1. **Horizontal Scaling**: Multi-master replication setup
2. **Caching Layer**: Distributed caching implementation
3. **Query Optimization**: Advanced query optimization techniques
4. **Resource Scaling**: Dynamic resource allocation

---

*Last updated: 2025-01-18*
*Version: 1.0*
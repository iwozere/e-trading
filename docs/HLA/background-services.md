# Background Services and Job Management

## Overview

The Advanced Trading Framework implements a dual-layer background service architecture:

1. **System Services**: Independent services that run as separate processes (TradingBot, TelegramBot)
2. **Scheduled Job Services**: APScheduler-based services for executing scheduled tasks (alerts, screeners, reports)

This architecture separates long-running system services from scheduled job execution, providing better scalability and fault isolation.

## Architecture Components

### 1. System Services Layer

#### 1.1 Trading Bot Service (`src/trading/trading_runner.py`, `src/trading/strategy_manager.py`)
**Purpose**: Manages multi-bot trading operations and strategy execution with crash recovery

**Service Characteristics**:
- Runs as independent system process
- Database-driven configuration (no JSON files)
- Supports multiple concurrent trading bots
- Manages broker connections and data feeds
- **Production-grade crash recovery and automatic resume**
- Heartbeat-based health monitoring

**Architecture**:
```
trading_runner.py (Service Orchestrator)
    ↓
strategy_manager.py (SOLE Config Loader)
    ↓
StrategyInstance objects (Individual Bots)
    ↓
BaseTradingBot + Backtrader Integration
```

**Responsibilities**:
- Execute trading strategies with real-time data
- Manage positions and orders across multiple bots
- Handle broker communication for each bot instance
- Process market events and signals
- Maintain trading state persistence in database
- **Crash detection and automatic bot recovery**
- **State recovery (positions, trades) on service restart**
- Performance metrics and trade recording

**Service Pattern**:
```python
# Trading Runner (Service Orchestrator)
class TradingServiceRunner:
    def __init__(self, user_id=None, resume_mode=True):
        self.strategy_manager = StrategyManager()
        self.resume_mode = resume_mode  # Enables crash recovery

    async def start_service(self):
        # Delegates to StrategyManager with crash recovery
        await self.strategy_manager.load_strategies_from_db(
            user_id=self.user_id,
            resume_mode=self.resume_mode
        )
        await self.strategy_manager.start_all_strategies()
        await self.strategy_manager.start_monitoring()
        await self.strategy_manager.start_db_polling()

    async def stop_service(self):
        # Graceful shutdown with status persistence
        await self.strategy_manager.shutdown()
```

**Key Features**:
- **Smart Resume**: Automatically detects crashes and resumes only previously running bots
- **State Recovery**: Recovers open positions and trades from database after crash
- **Graceful Shutdown**: Persists all bot statuses to database on clean shutdown
- **Hot Reload**: Database polling for configuration changes without restart
- **Multi-Bot Management**: Run multiple strategies simultaneously with isolation
- **Heartbeat Monitoring**: Automatic health checks and recovery for unhealthy bots

#### 1.2 Telegram Bot Service (`src/telegram/bot.py`)
**Purpose**: Handles Telegram bot operations and user interactions

**Service Characteristics**:
- Runs as independent system process
- Not APScheduler-based
- Handles incoming Telegram commands
- Manages user communications

**Responsibilities**:
- Process incoming Telegram commands
- Handle user registration and verification
- Manage user alerts and schedules (via API)
- Send notifications and reports
- Provide real-time user interaction

**Service Pattern**:
```python
class TelegramBot:
    def __init__(self, token: str):
        self.bot = Bot(token=token)
        self.dispatcher = Dispatcher()
        
    async def start_polling(self):
        # Start bot polling for messages
        
    async def process_command(self, message: Message):
        # Handle incoming commands
```

### 2. Scheduled Job Services Layer

#### 2.1 System Scheduler Implementation

The System Scheduler module provides APScheduler-based job scheduling and alert evaluation capabilities. It consists of several key components:

**Core Components**:
- **SchedulerService** (`src/scheduler/scheduler_service.py`): Main service orchestrating job scheduling
- **AlertEvaluator** (`src/common/alerts/alert_evaluator.py`): Centralized alert evaluation with rearm logic
- **CronParser** (`src/common/alerts/cron_parser.py`): Cron expression parsing and validation
- **AlertSchemaValidator** (`src/common/alerts/schema_validator.py`): JSON schema validation for configurations

**Service Architecture**:
```mermaid
graph TB
    subgraph "System Scheduler Service"
        SS[SchedulerService]
        APS[APScheduler]
        AE[AlertEvaluator]
    end
    
    subgraph "Alert Services"
        CP[CronParser]
        ASV[AlertSchemaValidator]
    end
    
    subgraph "Data Layer"
        JS[JobsService]
        DM[DataManager]
        IS[IndicatorService]
    end
    
    subgraph "Notification Layer"
        NSC[NotificationServiceClient]
        TB[TelegramBot]
    end
    
    SS --> APS
    APS --> AE
    AE --> CP
    AE --> ASV
    AE --> JS
    AE --> DM
    AE --> IS
    SS --> NSC
    NSC --> TB
    
    classDef scheduler fill:#e1f5fe
    classDef alert fill:#f3e5f5
    classDef data fill:#e8f5e8
    classDef notification fill:#fff3e0
    
    class SS,APS scheduler
    class AE,CP,ASV alert
    class JS,DM,IS data
    class NSC,TB notification
```

**Key Features**:
- **Centralized Alert Evaluation**: All alert logic consolidated in `AlertEvaluator`
- **Rearm Logic**: Sophisticated rearm mechanisms with hysteresis, cooldown, and persistence
- **Schema Validation**: JSON schema validation for alert and schedule configurations
- **State Persistence**: Alert state maintained in database across service restarts
- **Timezone Support**: Full timezone awareness for cron scheduling
- **Error Handling**: Comprehensive error handling with retry logic and graceful degradation

#### 2.2 Job Scheduling System

#### APScheduler Integration
- **Framework**: Advanced Python Scheduler (APScheduler) - *✅ Implemented*
- **Storage**: PostgreSQL-based job store for persistence
- **Executors**: AsyncIOExecutor for concurrent I/O-bound alert evaluations
- **Triggers**: Cron-based scheduling with 5/6 field support and timezone awareness
- **Background Service**: `src/scheduler/scheduler_service.py` - *✅ Implemented*
- **Alert Services**: Centralized in `src/common/alerts/` with schema validation - *✅ Implemented*

#### Job Types
The system supports multiple job types:

```python
class JobType(str, Enum):
    REPORT = "report"           # Generate trading reports
    SCREENER = "screener"       # Run stock screeners
    ALERT = "alert"            # Process price alerts
    NOTIFICATION = "notification" # Send notifications
    DATA_PROCESSING = "data_processing" # Process market data
    BACKUP = "backup"          # System backups
```

#### 2.2 Job Persistence Layer

#### Schedule Model (`job_schedules` table)
Persistent schedule definitions stored in PostgreSQL with cron-style scheduling:

```sql
CREATE TABLE job_schedules (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    name VARCHAR(255) NOT NULL,
    job_type VARCHAR(50) NOT NULL,
    target VARCHAR(255) NOT NULL,
    task_params JSONB NOT NULL DEFAULT '{}',
    cron VARCHAR(100) NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT true,
    next_run_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
```

#### Run Tracking Model (`job_schedule_runs` table)
Execution history and status tracking for scheduled jobs:

```sql
CREATE TABLE job_schedule_runs (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL DEFAULT gen_random_uuid(),
    job_type TEXT NOT NULL,
    job_id BIGINT,
    user_id BIGINT,
    status TEXT,
    scheduled_for TIMESTAMP WITH TIME ZONE,
    enqueued_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    finished_at TIMESTAMP WITH TIME ZONE,
    job_snapshot JSONB,
    result JSONB,
    error TEXT,
    worker_id VARCHAR(255)
);
```

### 3. Trading Service Crash Recovery Architecture

#### 3.1 Overview

The Trading Service implements a production-grade crash recovery system that ensures **24/7 reliable bot operation** even after system crashes, power failures, or forced terminations. The system automatically detects unclean shutdowns and intelligently resumes trading bots with full state recovery.

**Key Capabilities**:
- **Crash Detection**: Automatic detection of unclean shutdowns using marker files
- **Smart Resume Logic**: Selective bot restart (crash recovery vs normal startup)
- **State Recovery**: Complete restoration of open positions and trades
- **Graceful Shutdown**: Proper status persistence on clean shutdowns
- **Zero Data Loss**: All critical state persisted in database

#### 3.2 Crash Detection System

**Marker File Approach**:
The system uses a `.trading_service_running` marker file to track service lifecycle:

```python
class StrategyManager:
    def __init__(self):
        self._marker_path = Path(".trading_service_running")

    def _detect_crash_recovery(self) -> bool:
        """
        Detect if this is a crash recovery (unclean shutdown).

        - If marker exists → Previous shutdown was unclean (CRASH)
        - If marker missing → Clean startup
        """
        if self._marker_path.exists():
            _logger.warning("⚠️ UNCLEAN SHUTDOWN detected - entering crash recovery mode")
            self._marker_path.unlink()  # Remove stale marker
            return True

        _logger.info("Clean startup detected")
        return False

    def _mark_service_running(self):
        """Create marker when service starts."""
        self._marker_path.touch()

    def _mark_clean_shutdown(self):
        """Remove marker on graceful shutdown."""
        if self._marker_path.exists():
            self._marker_path.unlink()
```

**Crash Detection Flow**:
```mermaid
graph TD
    A[Service Starts] --> B{Marker File Exists?}
    B -->|YES| C[🚨 CRASH DETECTED]
    B -->|NO| D[✅ Clean Startup]
    C --> E[Enter Crash Recovery Mode]
    D --> F[Normal Startup Mode]
    E --> G[Resume Only Running Bots]
    F --> H[Load All Enabled Bots]
    G --> I[Create New Marker]
    H --> I
```

#### 3.3 Smart Resume Logic

The system implements intelligent bot selection based on startup mode:

```python
async def load_strategies_from_db(self, user_id: Optional[int] = None,
                                 resume_mode: bool = True) -> bool:
    """
    Load bot configurations with smart resume logic.

    Behavior:
    - Crash Recovery: Load only bots with status='running'
    - Normal Startup: Load all enabled bots
    """
    was_crashed = False

    if resume_mode:
        was_crashed = self._detect_crash_recovery()

        if was_crashed:
            _logger.warning("🔄 CRASH RECOVERY MODE: Resuming previously running bots")
            # Load ONLY bots that were running before crash
            bots = trading_service.get_bots_by_status("running", user_id)
            _logger.info("Found %d bot(s) that were running before crash", len(bots))
        else:
            _logger.info("🚀 NORMAL STARTUP: Loading all enabled bots")
            bots = trading_service.get_enabled_bots(user_id)

    # Create marker file to track this session
    self._mark_service_running()

    # ... continue loading
```

**Startup Mode Comparison**:

| Mode | Trigger | Bots Loaded | Use Case |
|------|---------|-------------|----------|
| **Crash Recovery** | Marker file exists | Only `status='running'` | Automatic after crash |
| **Normal Startup** | No marker file | All `status != 'disabled'` | Clean start or manual restart |
| **Force Normal** | `--no-resume` flag | All `status != 'disabled'` | Admin override |

#### 3.4 State Recovery System

After a crash, the system recovers full bot state from the database:

```python
def _recover_bot_state(self, bot_id: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recover complete bot state from database.

    Recovers:
    - Open positions (qty, avg_price, direction)
    - Pending trades (entry_price, entry_time, status)
    - Account state (balance, P&L)
    """
    try:
        # Recover open positions
        open_positions = trading_service.get_open_positions(bot_id=str(bot_id))

        if open_positions:
            _logger.info("🔄 Recovered %d open position(s) for bot %d",
                        len(open_positions), bot_id)
            config['_recovered_positions'] = open_positions

            # Log position details
            for pos in open_positions:
                _logger.info("  Position: %s %s qty=%.8f avg_price=%.8f status=%s",
                           pos.get('symbol'), pos.get('direction'),
                           pos.get('qty_open', 0), pos.get('avg_price', 0),
                           pos.get('status'))

        # Recover open trades
        open_trades = trading_service.get_open_trades()
        bot_trades = [t for t in open_trades if t['bot_id'] == bot_id]

        if bot_trades:
            _logger.info("🔄 Recovered %d open trade(s) for bot %d",
                        len(bot_trades), bot_id)
            config['_recovered_trades'] = bot_trades

        return config
    except Exception as e:
        _logger.exception("Error recovering state for bot %d:", bot_id)
        return config
```

**Recovered State Structure**:
```json
{
  "config": {
    "symbol": "BTCUSDT",
    "broker": {...},
    "strategy": {...},
    "_recovered_positions": [
      {
        "id": 123,
        "symbol": "BTCUSDT",
        "direction": "LONG",
        "qty_open": 0.01,
        "avg_price": 45000.0,
        "status": "open",
        "realized_pnl": 0.0
      }
    ],
    "_recovered_trades": [
      {
        "id": 456,
        "symbol": "BTCUSDT",
        "entry_price": 45000.0,
        "entry_time": "2025-10-22T10:30:00Z",
        "status": "open",
        "size": 0.01
      }
    ]
  }
}
```

#### 3.5 Graceful Shutdown Protocol

On clean shutdown, the system persists all bot statuses to ensure accurate crash detection:

```python
async def shutdown(self):
    """
    Gracefully shutdown with complete state persistence.

    Ensures:
    - All bots are stopped cleanly
    - Bot statuses persisted to database
    - Resources properly released
    - Clean shutdown marked (no crash on next startup)
    """
    _logger.info("🛑 Shutting down Enhanced Strategy Manager...")

    try:
        # Stop monitoring first
        await self.stop_monitoring()

        # Stop all strategy instances and persist their status
        _logger.info("Stopping all strategy instances and persisting statuses...")
        for instance_id, instance in self.strategy_instances.items():
            if instance.status == 'running':
                _logger.info("Stopping bot %s (%s)", instance.name, instance_id)

                # Stop the bot
                await instance.stop()

                # Persist stopped status to database
                try:
                    trading_service.update_bot_status(
                        int(instance_id),
                        "stopped",
                        error_message=None  # Clear any error
                    )
                    _logger.debug("Persisted stopped status for bot %s", instance_id)
                except Exception as e:
                    _logger.warning("Failed to persist status for bot %s: %s",
                                  instance_id, e)

        # Close resources
        await self.broker_manager.shutdown()
        await self.notification_client.close()

        # Mark clean shutdown (remove crash marker)
        self._mark_clean_shutdown()

        _logger.info("✅ Enhanced Strategy Manager shutdown complete")

    except Exception as e:
        _logger.exception("Error during shutdown:")
        # Don't mark clean shutdown if errors occurred
        _logger.warning("⚠️ Shutdown completed with errors - crash marker not removed")
```

**Shutdown Flow**:
```mermaid
sequenceDiagram
    participant User as User/Signal
    participant Runner as TradingRunner
    participant Manager as StrategyManager
    participant DB as Database
    participant FS as File System

    User->>Runner: Ctrl+C / SIGTERM
    Runner->>Manager: shutdown()

    loop For each running bot
        Manager->>Manager: Stop bot instance
        Manager->>DB: Update status to 'stopped'
    end

    Manager->>Manager: Close broker manager
    Manager->>Manager: Close notification client
    Manager->>FS: Remove marker file
    Manager-->>Runner: Shutdown complete
    Runner-->>User: Service stopped
```

#### 3.6 Operational Scenarios

**Scenario 1: Service Crash**
```
1. Bot A, B, C running normally
2. Service crashes (power loss, kill -9, system failure)
3. Marker file remains on disk
4. Bot statuses remain 'running' in database

Next Startup:
1. ✅ Marker detected → Crash recovery mode
2. ✅ Query bots with status='running'
3. ✅ Find bots A, B, C
4. ✅ Recover positions and trades for each
5. ✅ Resume A, B, C with full state
6. ✅ Bots D, E (disabled/stopped) are NOT started
```

**Scenario 2: Clean Shutdown**
```
1. Bot A, B, C running normally
2. User presses Ctrl+C
3. Service stops all bots gracefully
4. Updates A, B, C status to 'stopped'
5. Removes marker file

Next Startup:
1. ✅ No marker → Normal startup mode
2. ✅ Query all enabled bots
3. ✅ Find bots A, B, C, D, E (all enabled)
4. ✅ Start all 5 bots normally
5. ✅ No state recovery (clean start)
```

**Scenario 3: Manual Restart**
```
1. Admin wants to restart service
2. systemctl stop trading-service (clean)
3. systemctl start trading-service

Behavior:
1. ✅ No marker → Normal startup
2. ✅ All enabled bots start
```

**Scenario 4: Force Normal Startup**
```
# Admin wants to override crash recovery
python src/trading/trading_runner.py --no-resume

Behavior:
1. ✅ resume_mode=False
2. ✅ Crash detection skipped
3. ✅ All enabled bots loaded
4. ✅ No state recovery
```

#### 3.7 CLI Usage

```bash
# Normal startup with crash recovery (default)
python src/trading/trading_runner.py

# Disable crash recovery (force normal startup)
python src/trading/trading_runner.py --no-resume

# With user filter and crash recovery
python src/trading/trading_runner.py --user-id 1

# Full example with all options
python src/trading/trading_runner.py \
    --user-id 1 \
    --poll-interval 30 \
    --no-resume
```

#### 3.8 Database Schema Support

**Bot Status Tracking** (`trading_bots` table):
```sql
CREATE TABLE trading_bots (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    status VARCHAR(20) NOT NULL,  -- 'running', 'stopped', 'error', 'disabled'
    started_at TIMESTAMP WITH TIME ZONE,
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    error_count INTEGER DEFAULT 0,
    current_balance NUMERIC(20, 8),
    total_pnl NUMERIC(20, 8),
    config JSONB NOT NULL,
    extra_metadata JSONB,
    ...
);

-- Crash recovery queries
SELECT * FROM trading_bots WHERE status = 'running';  -- Crashed bots
SELECT * FROM trading_bots WHERE status != 'disabled';  -- Normal startup
```

**Position Recovery** (`trading_positions` table):
```sql
CREATE TABLE trading_positions (
    id SERIAL PRIMARY KEY,
    bot_id INTEGER NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(10) NOT NULL,  -- 'LONG', 'SHORT'
    qty_open NUMERIC(20, 8) NOT NULL DEFAULT 0,
    avg_price NUMERIC(20, 8),
    realized_pnl NUMERIC(20, 8) DEFAULT 0,
    status VARCHAR(12) NOT NULL,  -- 'open', 'closed'
    ...
);

-- State recovery queries
SELECT * FROM trading_positions WHERE bot_id = ? AND status = 'open';
SELECT * FROM trading_trades WHERE bot_id = ? AND status = 'open';
```

#### 3.9 Reliability Features

**Error Count Reset**:
```python
# On successful bot start, reset error count
self.error_count = 0
self.last_error = None
```

**Heartbeat-Based Health Monitoring**:
- Every 60 seconds, bot sends heartbeat to database
- Monitor loop detects stale heartbeats (>3x interval)
- Automatic restart for unhealthy bots (max 3 attempts)

**Auto-Recovery**:
```python
# Monitor loop in StrategyManager
if heartbeat_age > max_heartbeat_age:
    _logger.warning("Bot %s heartbeat stale, attempting recovery...", bot.name)
    if bot.error_count < 3:
        await bot.restart()
    else:
        _logger.error("Bot %s exceeded max restart attempts", bot.name)
        bot.status = 'error'
```

#### 3.10 Integration with Health Monitoring

The crash recovery system integrates with the platform-wide health monitoring:

```python
# Each bot reports its health via HeartbeatManager
from src.common.heartbeat_manager import HeartbeatManager

heartbeat_manager = HeartbeatManager(
    system='trading_bot',
    interval_seconds=60
)

def trading_bot_health_check():
    return {
        'status': 'HEALTHY' if is_running else 'DOWN',
        'metadata': {
            'bot_count': len(strategy_instances),
            'running_bots': running_count,
            'unhealthy_bots': unhealthy_count,
            'crash_recovery_enabled': resume_mode
        }
    }

heartbeat_manager.set_health_check_function(trading_bot_health_check)
heartbeat_manager.start_heartbeat()
```

For comprehensive health monitoring documentation, see [System Health Monitoring](system-health-monitoring.md).

---

### 4. Service Communication Architecture

#### 4.1 System Service Independence
- **TradingBot** and **TelegramBot** run as separate **systemd services** (Linux deployment)
- Each service manages its own lifecycle and error handling
- Services communicate through PostgreSQL database and FastAPI endpoints (src/api/)
- No direct inter-process communication required

#### 3.2 Alert/Schedule Creation Flow
```mermaid
sequenceDiagram
    participant User
    participant TelegramBot
    participant Database
    participant SchedulerService
    
    User->>TelegramBot: /alert AAPL > 150
    TelegramBot->>Database: Insert job_schedules record
    Note over Database: Alert stored with cron schedule
    SchedulerService->>Database: Poll for new schedules
    SchedulerService->>Database: Calculate next_run_at
    SchedulerService->>Database: Create job_schedule_runs
```

#### 3.3 Job Execution and Notification Flow
```mermaid
sequenceDiagram
    participant SchedulerService
    participant APScheduler
    participant AlertEvaluator
    participant DataManager
    participant IndicatorService
    participant NotificationClient
    participant TelegramBot
    participant Database
    
    SchedulerService->>APScheduler: Register scheduled jobs
    APScheduler->>AlertEvaluator: Execute alert job
    AlertEvaluator->>Database: Create job run record
    AlertEvaluator->>DataManager: Fetch market data
    AlertEvaluator->>IndicatorService: Calculate indicators
    AlertEvaluator->>AlertEvaluator: Evaluate alert conditions
    AlertEvaluator->>AlertEvaluator: Apply rearm logic
    AlertEvaluator->>Database: Update alert state
    alt Alert triggered
        AlertEvaluator->>NotificationClient: Send notification
        NotificationClient->>TelegramBot: Deliver message
        TelegramBot->>User: Send Telegram message
    end
    AlertEvaluator->>Database: Update job run result
```

#### 3.4 Service Communication Patterns

**Alert/Schedule Storage**:
- TelegramBot stores user alerts/schedules directly in PostgreSQL
- Background scheduler service polls database for new/updated schedules
- No real-time communication needed between TelegramBot and SchedulerService

**Notification Delivery**:
- SchedulerService executes alerts/schedules via AlertEvaluator
- AlertEvaluator uses NotificationServiceClient for message delivery
- NotificationServiceClient handles message formatting and delivery
- TelegramBot receives and processes notification messages

**Service Dependencies**:
- **TelegramBot**: Independent systemd service, exposes FastAPI for notifications
- **TradingBot**: Independent systemd service, no direct scheduler interaction
- **SchedulerService**: New systemd service (to be implemented), spawns workers
- **NotificationService**: Background service within TelegramBot process

### 4. Job Execution States

#### Run Status Lifecycle
```mermaid
stateDiagram-v2
    [*] --> PENDING : Job Created
    PENDING --> RUNNING : Worker Claims Job
    PENDING --> CANCELLED : User Cancellation
    RUNNING --> COMPLETED : Successful Execution
    RUNNING --> FAILED : Execution Error
    COMPLETED --> [*]
    FAILED --> [*]
    CANCELLED --> [*]
```

#### Status Definitions
- **PENDING**: Job queued for execution
- **RUNNING**: Currently being executed by a worker
- **COMPLETED**: Successfully finished
- **FAILED**: Execution failed with error
- **CANCELLED**: Cancelled before execution

### 5. Alert Evaluation Architecture

#### 5.1 AlertEvaluator Service

The `AlertEvaluator` is the core component responsible for processing alert jobs with sophisticated evaluation logic:

**Key Capabilities**:
- **Rule Tree Evaluation**: Supports complex logical operators (AND, OR, NOT)
- **Market Data Integration**: Seamless integration with DataManager for OHLCV data
- **Indicator Integration**: Uses IndicatorService for technical indicator calculations
- **Rearm Logic**: Advanced rearm mechanisms including:
  - Hysteresis-based rearm conditions
  - Cooldown periods between triggers
  - Persistence bar requirements
  - Crossing detection with side tracking
- **State Management**: Persistent alert state in database `state_json` field
- **Error Handling**: Graceful handling of data unavailability and calculation errors

**Alert Configuration Structure**:
```json
{
  "ticker": "AAPL",
  "timeframe": "1h",
  "rule": {
    "operator": "and",
    "conditions": [
      {
        "indicator": "rsi",
        "operator": ">",
        "value": 70
      },
      {
        "indicator": "price",
        "operator": ">",
        "value": 150
      }
    ]
  },
  "rearm": {
    "type": "hysteresis",
    "threshold": 0.02,
    "cooldown": 300
  },
  "notify": {
    "channels": ["telegram"],
    "message_template": "Alert triggered for {ticker}"
  }
}
```

**Rearm Logic Types**:
- **Simple**: Alert rearms immediately after trigger
- **Hysteresis**: Requires value to move away from trigger by threshold percentage
- **Cooldown**: Prevents retriggering for specified time period
- **Persistence**: Requires condition to persist for specified number of bars
- **Crossing**: Tracks value crossing above/below threshold with side detection

#### 5.2 CronParser Service

Handles cron expression parsing with enhanced capabilities:

**Features**:
- **Dual Format Support**: Both 5-field (minute precision) and 6-field (second precision)
- **Timezone Awareness**: Full timezone support for global trading
- **Validation**: Comprehensive cron expression validation
- **Next Run Calculation**: Accurate next execution time calculation

**Usage Examples**:
```python
# 5-field cron (every minute)
cron_5 = "* * * * *"

# 6-field cron (every 30 seconds)
cron_6 = "*/30 * * * * *"

# Timezone-aware scheduling
next_run = CronParser.calculate_next_run(
    expression="0 9 * * 1-5",  # 9 AM weekdays
    timezone="America/New_York"
)
```

#### 5.3 AlertSchemaValidator Service

Provides JSON schema validation for alert configurations:

**Schema Types**:
- **Alert Schema**: Validates alert job configurations
- **Schedule Schema**: Validates schedule job configurations
- **Custom Schemas**: Extensible for new job types

**Validation Features**:
- **Schema Caching**: Efficient schema loading and caching
- **Detailed Error Messages**: Clear validation error reporting
- **Version Support**: Schema versioning for backward compatibility

### 6. Scheduled Job Types

#### 5.1 Report Generation Jobs
**Purpose**: Generate and deliver scheduled trading reports

**Execution Context**: APScheduler worker process
**Trigger**: Cron-based scheduling from `job_schedules.cron`
**Storage**: Results stored in `job_schedule_runs.result`

**Job Parameters** (stored in `task_params`):
```json
{
    "report_type": "daily_summary|portfolio_analysis|performance_report",
    "parameters": {
        "symbols": ["AAPL", "MSFT"],
        "timeframe": "1d",
        "include_charts": true
    }
}
```

**Integration with System Services**:
- Results delivered via TelegramBot notifications
- Can trigger trading actions via TradingBot API

#### 5.2 Screener Jobs
**Purpose**: Execute stock screening with custom criteria

**Execution Context**: APScheduler worker process
**Trigger**: Cron-based scheduling from `job_schedules.cron`
**Target**: Screener set name or ticker list (stored in `target` field)

**Job Parameters** (stored in `task_params`):
```json
{
    "screener_set": "sp500|nasdaq|custom",
    "filter_criteria": {
        "min_market_cap": 1000000000,
        "max_pe_ratio": 25,
        "min_volume": 1000000
    },
    "top_n": 10
}
```

**Integration with System Services**:
- Results delivered via TelegramBot
- Can create alerts for matching stocks

#### 5.3 Alert Jobs
**Purpose**: Monitor market conditions and trigger notifications

**Execution Context**: APScheduler worker process
**Trigger**: Cron-based scheduling (typically frequent, e.g., every minute)
**Target**: Symbol or condition to monitor

**Job Parameters** (stored in `task_params`):
```json
{
    "symbol": "AAPL",
    "condition": "price_above|price_below|volume_spike|rsi_oversold",
    "threshold": 150.00,
    "notification_channels": ["telegram", "email"]
}
```

**Integration with System Services**:
- Notifications sent via TelegramBot
- Can trigger trading actions via TradingBot

#### 5.4 Data Processing Jobs
**Purpose**: Scheduled data collection and processing tasks

**Execution Context**: APScheduler worker process
**Trigger**: Cron-based scheduling (e.g., every 5 minutes, end-of-day)
**Target**: Data source or processing task identifier

**Job Parameters** (stored in `task_params`):
```json
{
    "data_source": "yahoo|binance|ibkr",
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "operations": ["fetch", "normalize", "cache", "indicators"]
}
```

**Integration with System Services**:
- Updated data available to TradingBot for strategy execution
- Cache updates improve TelegramBot response times

### 7. Service Deployment and Management

#### 7.1 System Service Deployment (Current Implementation)

**TradingBot Service** (`systemd` service):
- Deployed as independent systemd service on Linux
- Configuration via JSON files with Pydantic validation
- Independent lifecycle management and automatic restart on failure
- No direct interaction with scheduler service

**TelegramBot Service** (`systemd` service):
- Deployed as independent systemd service
- Exposes FastAPI endpoints for notification delivery (integrated with src/api/)
- Handles webhook or polling mode for Telegram API
- Contains background notification service for message queuing
- Stores user alerts/schedules in PostgreSQL database

#### 7.2 Scheduled Service Deployment (✅ Implemented)

**Background Scheduler Service** (`systemd` service - *✅ Implemented*):
- Independent systemd service for alert/schedule processing using APScheduler
- Centralized alert evaluation with rearm logic in `src/common/alerts/`
- Reads `job_schedules` table on startup and registers with APScheduler
- Uses AsyncIOExecutor for concurrent I/O-bound alert evaluations
- Integrates with existing DataManager and IndicatorService
- Communicates back to TelegramBot via NotificationServiceClient

**Service Architecture**:
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   TradingBot    │    │  TelegramBot    │    │ SchedulerService│
│  (systemd)      │    │   (systemd)     │    │   (systemd)     │
│                 │    │                 │    │                 │
│ - Live Trading  │    │ - User Commands │    │ - Alert Exec    │
│ - Strategies    │    │ - FastAPI       │    │ - Screener Exec │
│ - Broker Conn   │    │ - Notifications │    │ - Worker Spawn  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   PostgreSQL    │
                    │   Database      │
                    │                 │
                    │ - job_schedules │
                    │ - schedule_runs │
                    │ - trading_data  │
                    └─────────────────┘
```

#### 6.3 Service Communication Patterns

**Database-Mediated Communication**:
- Primary data exchange through PostgreSQL
- TelegramBot writes alerts/schedules to `job_schedules` table
- SchedulerService reads schedules and writes execution results
- No direct inter-service messaging required

**API-Based Notification Flow**:
- Background workers execute scheduled jobs
- Workers POST results to API service: `/api/notifications` (src/api/notification_routes.py)
- TelegramBot queues notifications in background service
- Asynchronous message delivery to users

**Service Independence**:
- Each service can restart independently without affecting others
- Database provides persistent state across service restarts
- Loose coupling enables independent scaling and deployment

### 7. Job Management Service Layer

#### JobsService Class
Central service for job management operations:

```python
class JobsService:
    def __init__(self, session: Session):
        self.session = session
        self.repository = JobsRepository(session)
    
    # Schedule Management
    def create_schedule(self, user_id: int, schedule_data: ScheduleCreate) -> Schedule
    def update_schedule(self, schedule_id: int, update_data: ScheduleUpdate) -> Schedule
    def delete_schedule(self, schedule_id: int) -> bool
    def trigger_schedule(self, schedule_id: int) -> Run
    
    # Run Management
    def create_run(self, user_id: int, run_data: RunCreate) -> Run
    def claim_run(self, run_id: UUID, worker_id: str) -> Run
    def update_run(self, run_id: UUID, update_data: RunUpdate) -> Run
    def cancel_run(self, run_id: UUID) -> bool
    
    # Statistics and Monitoring
    def get_run_statistics(self, user_id: int, job_type: JobType, days: int) -> dict
    def cleanup_old_runs(self, days_to_keep: int) -> int
```

### 8. Worker Architecture

#### Worker Pool Management
- **Thread Pool**: For I/O-bound tasks (API calls, database operations)
- **Process Pool**: For CPU-intensive tasks (data processing, ML computations)
- **Async Workers**: For concurrent operations

#### Worker Claiming Pattern
```python
async def worker_loop():
    while running:
        # Claim available jobs
        runs = jobs_service.get_pending_runs(job_type=worker_job_type, limit=1)
        
        for run in runs:
            claimed_run = jobs_service.claim_run(run.run_id, worker_id)
            if claimed_run:
                await execute_job(claimed_run)
        
        await asyncio.sleep(polling_interval)
```

### 9. Job Execution Patterns

#### 7.1 Report Generation Jobs
```python
async def execute_report_job(run: Run) -> dict:
    try:
        # Update status to running
        jobs_service.update_run(run.run_id, RunUpdate(
            status=RunStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
            worker_id=worker_id
        ))
        
        # Execute report generation
        snapshot = run.job_snapshot
        report_data = await generate_report(
            report_type=snapshot['report_type'],
            parameters=snapshot['parameters']
        )
        
        # Update with results
        jobs_service.update_run(run.run_id, RunUpdate(
            status=RunStatus.COMPLETED,
            finished_at=datetime.now(timezone.utc),
            result=report_data
        ))
        
        return report_data
        
    except Exception as e:
        # Update with error
        jobs_service.update_run(run.run_id, RunUpdate(
            status=RunStatus.FAILED,
            finished_at=datetime.now(timezone.utc),
            error=str(e)
        ))
        raise
```

#### 7.2 Screener Jobs
```python
async def execute_screener_job(run: Run) -> dict:
    snapshot = run.job_snapshot
    
    # Expand tickers from screener set
    tickers = jobs_service.expand_screener_target(snapshot['screener_set'])
    
    # Apply filter criteria
    filtered_results = await apply_screener_filters(
        tickers=tickers,
        criteria=snapshot['filter_criteria']
    )
    
    # Return top N results
    top_results = filtered_results[:snapshot.get('top_n', 10)]
    
    return {
        'screener_set': snapshot['screener_set'],
        'total_tickers': len(tickers),
        'filtered_count': len(filtered_results),
        'results': top_results
    }
```

### 10. Failure Recovery Mechanisms

#### 8.1 Job Retry Logic
- **Exponential Backoff**: Increasing delays between retries
- **Max Retry Attempts**: Configurable retry limits
- **Dead Letter Queue**: Failed jobs moved to separate queue

#### 8.2 Worker Health Monitoring
- **Heartbeat System**: Workers send periodic health checks
- **Stale Job Detection**: Identify jobs stuck in RUNNING state
- **Automatic Recovery**: Restart failed workers

#### 8.3 Database Consistency
- **Transactional Updates**: Atomic job status changes
- **Optimistic Locking**: Prevent concurrent job claiming
- **Cleanup Procedures**: Remove orphaned jobs and old runs

### 11. Monitoring and Observability

#### 11.1 Job Metrics
- **Execution Times**: Track job duration and performance
- **Success Rates**: Monitor job completion rates
- **Queue Depths**: Track pending job counts
- **Worker Utilization**: Monitor worker pool usage

#### 11.2 System Health Monitoring

**Integrated Health System**: All background services participate in the platform-wide health monitoring system:

**Monitored Services**:
- **Notification Service**: Message processor health, channel connectivity
- **Telegram Bot**: Bot API connectivity, command processing capability
- **API Service**: REST API health, service initialization status
- **Web UI**: Frontend/backend process health, static file serving
- **Trading Bot**: Strategy execution health, broker connectivity
- **Database**: Connection pool health, query performance

**Health Reporting**:
```python
# Each service implements automatic heartbeat reporting
from src.common.heartbeat_manager import HeartbeatManager

# Service-specific health check
def service_health_check():
    return {
        'status': 'HEALTHY' | 'DEGRADED' | 'DOWN' | 'UNKNOWN',
        'response_time_ms': Optional[int],
        'error_message': Optional[str],
        'metadata': Optional[Dict[str, Any]]
    }

# Automatic heartbeat every 30-60 seconds
heartbeat_manager = HeartbeatManager(
    system='service_name',
    interval_seconds=30
)
heartbeat_manager.set_health_check_function(service_health_check)
heartbeat_manager.start_heartbeat()
```

**Health Monitoring APIs**:
- `GET /api/v1/health` - Overall platform health
- `GET /api/v1/health/systems` - All systems health details
- `GET /api/v1/health/system/{system_name}` - Specific system health
- `GET /api/v1/health/unhealthy` - All unhealthy systems

**Health Data Storage**: All health data is stored in `msg_system_health` table with automatic cleanup and historical tracking.

For comprehensive health monitoring documentation, see [System Health Monitoring](system-health-monitoring.md).

#### 11.3 Traditional Job Alerting
- **Failed Jobs**: Alert on job failures
- **Queue Backlog**: Alert on growing queues
- **Worker Failures**: Alert on worker crashes
- **Performance Degradation**: Alert on slow jobs

### 12. Configuration Management

#### 12.1 System Scheduler Configuration

The System Scheduler uses a comprehensive configuration system with environment variable support:

**Main Configuration** (`src/scheduler/config.py`):
```python
class SchedulerConfig:
    # Database Configuration
    database_url: str = "postgresql+asyncpg://user:pass@localhost/db"
    
    # APScheduler Configuration
    max_workers: int = 10
    job_defaults: Dict[str, Any] = {
        'coalesce': False,
        'max_instances': 3,
        'misfire_grace_time': 30
    }
    
    # Alert Evaluation Configuration
    alert_timeout: int = 300  # 5 minutes
    data_cache_ttl: int = 60  # 1 minute
    
    # Notification Configuration
    notification_timeout: int = 30
    notification_retries: int = 3
    
    # Service Configuration
    startup_delay: int = 10
    shutdown_timeout: int = 30
    health_check_interval: int = 60
```

**Environment Variables**:
```bash
# Database
SCHEDULER_DATABASE_URL=postgresql+asyncpg://user:pass@localhost/trading_db

# APScheduler
SCHEDULER_MAX_WORKERS=10
SCHEDULER_MISFIRE_GRACE_TIME=30

# Alert Processing
SCHEDULER_ALERT_TIMEOUT=300
SCHEDULER_DATA_CACHE_TTL=60

# Notifications
SCHEDULER_NOTIFICATION_TIMEOUT=30
SCHEDULER_NOTIFICATION_RETRIES=3

# Service
SCHEDULER_STARTUP_DELAY=10
SCHEDULER_SHUTDOWN_TIMEOUT=30
```

#### 12.2 APScheduler Configuration
```python
SCHEDULER_CONFIG = {
    'jobstores': {
        'default': {
            'type': 'sqlalchemy',
            'url': DATABASE_URL,
            'tablename': 'apscheduler_jobs'
        }
    },
    'executors': {
        'default': {'type': 'threadpool', 'max_workers': 20},
        'processpool': {'type': 'processpool', 'max_workers': 5}
    },
    'job_defaults': {
        'coalesce': False,
        'max_instances': 3,
        'misfire_grace_time': 30
    }
}
```

#### 10.2 Job-Specific Configuration
```python
JOB_CONFIGS = {
    'report_generation': {
        'max_runtime': 300,  # 5 minutes
        'retry_attempts': 3,
        'retry_delay': 60
    },
    'data_processing': {
        'max_runtime': 600,  # 10 minutes
        'retry_attempts': 2,
        'retry_delay': 120
    },
    'notifications': {
        'max_runtime': 30,   # 30 seconds
        'retry_attempts': 5,
        'retry_delay': 10
    }
}
```

### 13. API Integration

#### 11.1 Job Management Endpoints
- `POST /api/reports/run` - Execute report immediately
- `POST /api/screeners/run` - Execute screener immediately
- `GET /api/runs/{id}` - Get run status
- `GET /api/runs` - List runs with filtering
- `DELETE /api/runs/{id}` - Cancel pending run

#### 11.2 Schedule Management Endpoints
- `POST /api/schedules` - Create new schedule
- `GET /api/schedules` - List schedules
- `PUT /api/schedules/{id}` - Update schedule
- `DELETE /api/schedules/{id}` - Delete schedule
- `POST /api/schedules/{id}/trigger` - Manually trigger schedule

### 14. Performance Considerations

#### 12.1 Scalability
- **Horizontal Scaling**: Multiple worker instances
- **Load Balancing**: Distribute jobs across workers
- **Resource Isolation**: Separate pools for different job types

#### 12.2 Optimization
- **Job Batching**: Group similar jobs for efficiency
- **Connection Pooling**: Reuse database connections
- **Caching**: Cache frequently accessed data
- **Lazy Loading**: Load job data only when needed

### 15. Notification Service Architecture

#### 15.1 Background Notification Service Design

The notification service runs as a background component within the TelegramBot process, handling asynchronous message delivery and queuing.

**Service Characteristics**:
- Embedded within TelegramBot systemd service
- Handles message queuing and delivery
- Provides FastAPI endpoint for other services
- Manages rate limiting and retry logic

**Integration Pattern**:
```python
class TelegramBot:
    def __init__(self):
        self.bot = Bot(token=token)
        self.notification_service = BackgroundNotificationService()
        self.fastapi_app = FastAPI()
        
    async def start(self):
        # Start notification service
        await self.notification_service.start()
        
        # Start FastAPI server for external notifications
        await self.start_fastapi_server()
        
        # Start Telegram bot polling
        await self.start_polling()

class BackgroundNotificationService:
    def __init__(self):
        self.message_queue = asyncio.Queue()
        self.running = False
        
    async def queue_notification(self, user_id: int, message: str):
        await self.message_queue.put({
            'user_id': user_id,
            'message': message,
            'timestamp': datetime.now(timezone.utc)
        })
        
    async def process_notifications(self):
        while self.running:
            try:
                notification = await asyncio.wait_for(
                    self.message_queue.get(), timeout=1.0
                )
                await self.send_telegram_message(notification)
            except asyncio.TimeoutError:
                continue
```

#### 15.2 FastAPI Notification Endpoint

**Endpoint Design**:
```python
@app.post("/api/notify")
async def notify_user(notification: NotificationRequest):
    """
    Receive notifications from background services (SchedulerService workers)
    """
    await notification_service.queue_notification(
        user_id=notification.user_id,
        message=notification.message
    )
    return {"status": "queued", "message_id": str(uuid.uuid4())}

class NotificationRequest(BaseModel):
    user_id: int
    message: str
    priority: str = "normal"  # normal, high, urgent
    notification_type: str = "alert"  # alert, report, screener
```

#### 15.3 Service Communication Discussion Points

**Current Architecture Questions**:
1. **Notification Service Location**: Should it be embedded in TelegramBot or separate service?
2. **Communication Method**: FastAPI calls vs message queue vs database polling?
3. **Service Discovery**: How do other services find TelegramBot FastAPI endpoint?
4. **Failure Handling**: What happens if TelegramBot is down when worker tries to notify?

**Proposed Solutions**:

**Option A: Embedded Notification Service (Current Plan)**
- ✅ Simpler deployment (one less systemd service)
- ✅ Direct access to Telegram bot instance
- ❌ TelegramBot becomes single point of failure for notifications
- ❌ Harder to scale notification processing independently

**Option B: Separate Notification Service**
- ✅ Independent scaling and failure isolation
- ✅ Can serve multiple communication channels (Telegram, email, SMS)
- ❌ More complex deployment and service coordination
- ❌ Additional service to monitor and maintain

**Recommended Approach**: Start with embedded service (Option A) for simplicity, with architecture that allows future extraction to separate service.

#### 15.4 Cross-Service Communication Patterns

**Service Registry Pattern** (Future Enhancement):
```python
# Services register their endpoints in database
service_registry = {
    "telegram_bot": {
        "host": "localhost",
        "port": 8001,
        "endpoints": {
            "notify": "/api/notify",
            "health": "/health"
        }
    }
}
```

**Configuration-Based Discovery** (Current Approach):
```python
# Each service configured with other service endpoints
TELEGRAM_BOT_CONFIG = {
    "notification_endpoint": "http://localhost:8001/api/notify"
}
```

### 16. Security Considerations

#### 16.1 Job Isolation
- **User Permissions**: Jobs run with user-specific permissions
- **Resource Limits**: Prevent resource exhaustion
- **Sandboxing**: Isolate job execution environments

#### 16.2 Data Protection
- **Encrypted Storage**: Sensitive job data encrypted at rest
- **Audit Logging**: Track all job operations
- **Access Control**: Restrict job management operations

#### 16.3 Inter-Service Security
- **API Authentication**: FastAPI endpoints require authentication tokens
- **Network Security**: Services communicate over localhost or secured network
- **Input Validation**: All API inputs validated and sanitized

### 16. Service Architecture Summary

#### System Services (Independent systemd Services)
- **TradingBot** (`src/trading/live_trading_bot.py`): Real-time trading operations
- **TelegramBot** (`src/telegram/bot.py`): User interaction, command processing, and FastAPI notification endpoint
- **SchedulerService** (`src/scheduler/background_bot.py`): *Planned - Alert/schedule execution with worker spawning*
- **Characteristics**: Long-running systemd services, independent lifecycle management

#### Scheduled Job Services (APScheduler-based Workers)
- **Background Scheduler** spawns workers for job execution
- **Job Types**: Reports, Screeners, Alerts, Data Processing, Notifications, Backups
- **Worker Communication**: Results sent via FastAPI calls to TelegramBot
- **Characteristics**: Cron-scheduled, worker-based execution, database persistence

#### Service Communication Flow

**Alert/Schedule Creation**:
1. User creates alert via TelegramBot
2. TelegramBot stores in PostgreSQL `job_schedules` table
3. SchedulerService polls database for new schedules

**Alert/Schedule Execution**:
1. SchedulerService spawns worker for scheduled job
2. Worker executes alert/screener logic
3. Worker POSTs result to TelegramBot FastAPI endpoint
4. TelegramBot queues notification in background service
5. Background notification service delivers message to user

#### Key Architecture Principles
1. **Service Independence**: Each systemd service can restart independently
2. **Database-Mediated State**: PostgreSQL provides persistent communication layer
3. **API-Based Notifications**: FastAPI enables loose coupling for result delivery
4. **Worker Isolation**: Scheduled jobs run in separate processes for fault tolerance
5. **Scalable Design**: Services can be scaled independently based on load

#### Implementation Status
- ✅ **TradingBot**: Implemented as systemd service
- ✅ **TelegramBot**: Implemented as systemd service with FastAPI
- ✅ **SchedulerService**: Fully implemented and operational
  - APScheduler-based job scheduling and execution
  - Centralized alert evaluation with rearm logic
  - Integration with existing data and notification services
  - Comprehensive error handling and state management
  - JSON schema validation for configurations
  - Timezone-aware cron parsing with 5/6 field support
- ✅ **Alert Services**: Complete implementation in `src/common/alerts/`
  - AlertEvaluator with sophisticated rearm logic
  - CronParser with dual format support
  - AlertSchemaValidator with caching
- ✅ **Notification Integration**: NotificationServiceClient integration

### 17. Operational Procedures

#### 17.1 Service Management

**Starting the System Scheduler**:
```bash
# Using systemd (production)
sudo systemctl start trading-scheduler
sudo systemctl enable trading-scheduler

# Using CLI (development)
cd /path/to/trading-framework
python -m src.scheduler.cli start

# Using main module (development)
python -m src.scheduler.main
```

**Service Status and Monitoring**:
```bash
# Check service status
sudo systemctl status trading-scheduler

# View logs
sudo journalctl -u trading-scheduler -f

# Check scheduler health
curl http://localhost:8002/health

# View job statistics
curl http://localhost:8002/stats
```

#### 17.2 Configuration Management

**Configuration Files**:
- **Main Config**: `config/scheduler/scheduler.json`
- **Alert Schemas**: `src/common/alerts/schemas/`
- **Systemd Service**: `src/scheduler/deployment/trading-scheduler.service`
- **Docker Compose**: `src/scheduler/deployment/docker-compose.yml`

**Configuration Validation**:
```bash
# Validate configuration
python -m src.scheduler.cli validate-config

# Test database connection
python -m src.scheduler.cli test-db

# Validate alert schemas
python -m src.scheduler.cli validate-schemas
```

#### 17.3 Database Operations

**Schedule Management**:
```sql
-- View active schedules
SELECT id, name, job_type, cron, enabled, next_run_at 
FROM job_schedules 
WHERE enabled = true;

-- View recent job runs
SELECT run_id, job_type, status, started_at, finished_at, error
FROM job_schedule_runs 
ORDER BY started_at DESC 
LIMIT 20;

-- Check failed jobs
SELECT * FROM job_schedule_runs 
WHERE status = 'FAILED' 
AND started_at > NOW() - INTERVAL '24 hours';
```

**Maintenance Operations**:
```sql
-- Clean up old job runs (keep last 30 days)
DELETE FROM job_schedule_runs 
WHERE started_at < NOW() - INTERVAL '30 days';

-- Reset stuck jobs
UPDATE job_schedule_runs 
SET status = 'FAILED', error = 'Service restart cleanup'
WHERE status = 'RUNNING' 
AND started_at < NOW() - INTERVAL '1 hour';
```

### 18. Monitoring and Alerting

#### 18.1 Health Checks

The scheduler service provides comprehensive health monitoring:

**Health Endpoint** (`/health`):
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "scheduler": "running",
    "database": "connected",
    "job_store": "operational",
    "alert_evaluator": "ready"
  },
  "metrics": {
    "active_jobs": 15,
    "pending_runs": 3,
    "failed_jobs_24h": 1,
    "avg_execution_time": 2.5
  }
}
```

**Statistics Endpoint** (`/stats`):
```json
{
  "job_counts": {
    "total_schedules": 25,
    "enabled_schedules": 20,
    "active_jobs": 15
  },
  "execution_stats": {
    "runs_24h": 1440,
    "success_rate": 0.995,
    "avg_duration": 2.3,
    "max_duration": 45.2
  },
  "error_stats": {
    "failed_runs_24h": 7,
    "common_errors": [
      "Data unavailable: 3",
      "Timeout: 2",
      "Network error: 2"
    ]
  }
}
```

#### 18.2 Logging and Observability

**Log Levels and Categories**:
- **INFO**: Service lifecycle, job execution, successful operations
- **WARNING**: Recoverable errors, data unavailability, timeout warnings
- **ERROR**: Job failures, configuration errors, database issues
- **DEBUG**: Detailed execution flow, data processing, state changes

**Key Log Messages**:
```
[INFO] SchedulerService started with 15 active schedules
[INFO] Job alert_123 executed successfully in 2.3s
[WARNING] Market data unavailable for AAPL, using cached data
[ERROR] Job screener_456 failed: Database connection timeout
[DEBUG] Alert state updated: {"status": "ARMED", "last_value": 150.25}
```

**Structured Logging**:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "service": "scheduler",
  "component": "alert_evaluator",
  "job_id": 123,
  "user_id": 456,
  "message": "Alert triggered",
  "context": {
    "ticker": "AAPL",
    "condition": "price > 150",
    "current_value": 152.50,
    "execution_time": 2.3
  }
}
```

This architecture provides robust separation of concerns while enabling efficient communication between services through well-defined interfaces and persistent storage. The comprehensive monitoring and operational procedures ensure reliable service operation and easy troubleshooting.
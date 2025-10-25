# Short Squeeze Detection Pipeline Design

## Overview

The Short Squeeze Detection Pipeline is designed as a modular system that integrates seamlessly with the existing trading platform infrastructure. The system follows a hybrid scheduling approach with weekly structural analysis and daily focused monitoring, leveraging existing data providers and notification systems for efficient operation.

The architecture emphasizes modularity, configurability, and integration with existing platform components while maintaining the flexibility to operate as standalone scripts initially and integrate with the scheduler system in the future.

## Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph "External Data Sources"
        FMP[FMP API<br/>Short Interest, Fundamentals]
        FINNHUB[Finnhub API<br/>Sentiment, Options, Borrow Rates]
    end
    
    subgraph "Short Squeeze Pipeline"
        CONFIG[Configuration<br/>YAML/JSON]
        
        subgraph "Core Modules"
            UNIVERSE[Universe Loader]
            SCREENER[Weekly Screener]
            DEEPSCAN[Daily Deep Scan]
            SCORING[Scoring Engine]
            ALERTS[Alert Engine]
            REPORTS[Reporting Engine]
        end
        
        subgraph "Data Layer"
            STORE[Candidate Store]
            ADHOC[Ad-hoc Manager]
        end
    end
    
    subgraph "Existing Platform Infrastructure"
        DB[(PostgreSQL Database)]
        NOTIFY[Notification System<br/>Telegram/Email]
        DOWNLOADERS[Data Downloaders<br/>FMP/Finnhub]
        LOGGER[Logging System]
    end
    
    subgraph "Future Integration"
        SCHEDULER[Scheduler System<br/>Future Integration]
    end
    
    FMP --> DOWNLOADERS
    FINNHUB --> DOWNLOADERS
    DOWNLOADERS --> UNIVERSE
    DOWNLOADERS --> SCREENER
    DOWNLOADERS --> DEEPSCAN
    
    CONFIG --> SCREENER
    CONFIG --> DEEPSCAN
    CONFIG --> SCORING
    CONFIG --> ALERTS
    
    UNIVERSE --> SCREENER
    SCREENER --> STORE
    STORE --> DEEPSCAN
    ADHOC --> DEEPSCAN
    DEEPSCAN --> SCORING
    SCORING --> ALERTS
    SCORING --> REPORTS
    
    STORE --> DB
    ALERTS --> DB
    ALERTS --> NOTIFY
    REPORTS --> DB
    
    SCREENER --> LOGGER
    DEEPSCAN --> LOGGER
    ALERTS --> LOGGER
    
    SCHEDULER -.-> SCREENER
    SCHEDULER -.-> DEEPSCAN
```

### Component Architecture

The system is organized into distinct layers:

1. **Configuration Layer**: YAML-based configuration management
2. **Core Processing Layer**: Main pipeline modules for screening, scanning, and scoring
3. **Data Management Layer**: Candidate storage and ad-hoc management
4. **Integration Layer**: Interfaces with existing platform infrastructure
5. **Future Integration Layer**: Prepared interfaces for scheduler integration

## Components and Interfaces

### 1. Configuration Manager

**Purpose**: Centralized configuration management for all pipeline parameters.

**Interface**:
```python
class PipelineConfig:
    def load_config(self, config_path: str) -> Dict[str, Any]
    def get_screener_config(self) -> ScreenerConfig
    def get_deep_scan_config(self) -> DeepScanConfig
    def get_alert_config(self) -> AlertConfig
    def get_scheduling_config(self) -> SchedulingConfig
```

**Key Responsibilities**:
- Load and validate YAML configuration
- Provide typed configuration objects to modules
- Support configuration hot-reloading for future scheduler integration

### 2. Universe Loader

**Purpose**: Fetch and filter the initial universe of stocks for analysis.

**Interface**:
```python
class UniverseLoader:
    def __init__(self, fmp_downloader: FMPDataDownloader)
    def load_universe(self) -> List[str]
    def filter_by_market_cap(self, min_cap: float, max_cap: float) -> List[str]
    def filter_by_volume(self, min_volume: int) -> List[str]
```

**Key Responsibilities**:
- Fetch stock universe from FMP
- Apply basic filters (market cap, volume, exchange)
- Return filtered ticker list for screener processing

### 3. Weekly Screener Module

**Purpose**: Perform weekly structural analysis to identify short squeeze candidates.

**Interface**:
```python
class WeeklyScreener:
    def __init__(self, fmp_downloader: FMPDataDownloader, config: ScreenerConfig)
    def run_screener(self, universe: List[str]) -> ScreenerResults
    def calculate_screener_score(self, metrics: StructuralMetrics) -> float
    def filter_candidates(self, results: ScreenerResults) -> List[Candidate]
```

**Key Responsibilities**:
- Fetch short interest, float, and volume data
- Calculate structural metrics (days to cover, SI%, float ratio)
- Score and rank candidates
- Store results in database with data quality metrics

### 4. Daily Deep Scan Module

**Purpose**: Perform daily analysis on identified candidates with real-time metrics.

**Interface**:
```python
class DailyDeepScan:
    def __init__(self, fmp_downloader: FMPDataDownloader, 
                 finnhub_downloader: FinnhubDataDownloader, 
                 config: DeepScanConfig)
    def run_deep_scan(self, candidates: List[Candidate]) -> DeepScanResults
    def calculate_transient_metrics(self, ticker: str) -> TransientMetrics
    def update_candidate_scores(self, results: DeepScanResults) -> None
```

**Key Responsibilities**:
- Load active candidates from database
- Fetch real-time volume, sentiment, and options data
- Calculate transient metrics (volume spike, sentiment, call/put ratio)
- Update daily metrics in database

### 5. Scoring Engine

**Purpose**: Compute comprehensive squeeze probability scores.

**Interface**:
```python
class ScoringEngine:
    def __init__(self, config: ScoringConfig)
    def calculate_squeeze_score(self, structural: StructuralMetrics, 
                               transient: TransientMetrics) -> float
    def normalize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]
    def apply_weights(self, normalized_metrics: Dict[str, float]) -> float
```

**Key Responsibilities**:
- Combine structural and transient metrics
- Apply configurable weights and normalization
- Generate final squeeze probability score (0-1 scale)

### 6. Alert Engine

**Purpose**: Manage alert generation and cooldown logic.

**Interface**:
```python
class AlertEngine:
    def __init__(self, notification_system: NotificationSystem, config: AlertConfig)
    def evaluate_alerts(self, scored_candidates: List[ScoredCandidate]) -> List[Alert]
    def check_cooldown(self, ticker: str, alert_level: AlertLevel) -> bool
    def send_alert(self, alert: Alert) -> bool
    def update_cooldown(self, ticker: str, alert_level: AlertLevel) -> None
```

**Key Responsibilities**:
- Evaluate alert conditions based on scores and thresholds
- Enforce cooldown periods to prevent spam
- Interface with existing notification system
- Log all alert events

### 7. Database Service Integration

**Purpose**: Use centralized database service for all short squeeze operations.

**Interface**:
```python
# Pipeline modules use the centralized service directly:
from src.data.db.core.database import session_scope
from src.data.db.services.short_squeeze_service import ShortSqueezeService

with session_scope() as session:
    service = ShortSqueezeService(session)
    service.save_screener_results(results, run_date)
    service.get_candidates_for_deep_scan()
    service.add_adhoc_candidate(ticker, reason, ttl_days)
```

**Key Responsibilities**:
- Use centralized `ShortSqueezeService` for all database operations
- Handle session management through `session_scope()` context manager
- Follow established patterns from other platform modules
- No additional abstraction layers needed

### 8. Ad-hoc Candidate Manager

**Purpose**: Handle manually added candidates for monitoring.

**Interface**:
```python
class AdHocManager:
    def __init__(self, candidate_store: CandidateStore, config: AdHocConfig)
    def add_candidate(self, ticker: str, reason: str) -> bool
    def remove_candidate(self, ticker: str) -> bool
    def get_active_candidates(self) -> List[AdHocCandidate]
    def expire_candidates(self) -> List[str]
```

**Key Responsibilities**:
- Manage ad-hoc candidate additions and removals
- Handle automatic expiration based on TTL
- Integrate with deep scan processing

### 9. Reporting Engine

**Purpose**: Generate summary reports and performance metrics.

**Interface**:
```python
class ReportingEngine:
    def __init__(self, candidate_store: CandidateStore, config: ReportConfig)
    def generate_weekly_summary(self) -> WeeklyReport
    def generate_daily_report(self) -> DailyReport
    def export_to_html(self, report: Report) -> str
    def export_to_csv(self, report: Report) -> str
```

**Key Responsibilities**:
- Generate weekly and daily summary reports
- Support multiple export formats (HTML, CSV)
- Include performance metrics and trend analysis

## Data Models

### Database Schema

The pipeline uses the centralized database infrastructure in `src/data/db/` and introduces four new tables to the existing PostgreSQL database:

**Database Models Location**: `src/data/db/models/model_short_squeeze.py`
**Repository Layer**: `src/data/db/repos/repo_short_squeeze.py`  
**Service Layer**: `src/data/db/services/short_squeeze_service.py`
**Migration**: `src/data/db/migrations/add_short_squeeze_tables.py`

The four tables are:

1. **ss_snapshot**: Weekly screener snapshots (append-only)
2. **ss_deep_metrics**: Daily deep scan metrics with unique constraint on (ticker, date)
3. **ss_alerts**: Alert history and cooldown tracking
4. **ss_ad_hoc_candidates**: Ad-hoc candidate management with unique ticker constraint

All tables include proper constraints, indexes, and follow the existing database naming conventions. The pipeline uses the centralized database connection management, session handling, and follows the established repository/service pattern.

### Core Data Structures

```python
@dataclass
class StructuralMetrics:
    short_interest_pct: float
    days_to_cover: float
    float_shares: int
    avg_volume_14d: int
    market_cap: int

@dataclass
class TransientMetrics:
    volume_spike: float
    call_put_ratio: Optional[float]
    sentiment_24h: float
    borrow_fee_pct: Optional[float]

@dataclass
class Candidate:
    ticker: str
    screener_score: float
    structural_metrics: StructuralMetrics
    last_updated: datetime
    source: str  # 'screener' or 'adhoc'

@dataclass
class ScoredCandidate:
    candidate: Candidate
    transient_metrics: TransientMetrics
    squeeze_score: float
    alert_level: Optional[str]

@dataclass
class Alert:
    ticker: str
    alert_level: str
    reason: str
    squeeze_score: float
    timestamp: datetime
    cooldown_expires: datetime
```

## Error Handling

### Error Categories and Strategies

1. **API Rate Limiting**:
   - Implement exponential backoff with jitter
   - Respect provider-specific rate limits (FMP: 300/min, Finnhub: 60/min)
   - Queue requests and batch where possible

2. **Data Quality Issues**:
   - Validate JSON responses and key field presence
   - Log data quality metrics per run
   - Continue processing with partial data, flag quality issues

3. **Database Connectivity**:
   - Implement connection pooling and retry logic
   - Use transactions for data consistency
   - Graceful degradation for non-critical operations

4. **External Service Failures**:
   - Implement circuit breaker pattern for repeated failures
   - Cache recent data for fallback scenarios
   - Alert administrators on prolonged outages

### Error Recovery Patterns

```python
class PipelineErrorHandler:
    def __init__(self, config: ErrorConfig):
        self.max_retries = config.max_retries
        self.backoff_factor = config.backoff_factor
        
    @retry_with_backoff
    def safe_api_call(self, api_func: Callable, *args, **kwargs):
        """Wrapper for API calls with retry logic"""
        
    def handle_partial_failure(self, results: List[Result], errors: List[Exception]):
        """Continue processing with partial results"""
        
    def should_abort_run(self, error_rate: float) -> bool:
        """Determine if error rate requires run abortion"""
```

## Testing Strategy

### Unit Testing

- **Configuration Management**: Test YAML parsing and validation
- **Data Processing**: Test metric calculations and scoring algorithms
- **Database Operations**: Test CRUD operations with test database
- **Alert Logic**: Test threshold evaluation and cooldown management

### Integration Testing

- **Data Provider Integration**: Test with mock API responses
- **Database Integration**: Test with containerized PostgreSQL
- **Notification Integration**: Test with mock notification system
- **End-to-End Pipeline**: Test complete workflow with sample data

### Performance Testing

- **API Rate Limiting**: Verify compliance with provider limits
- **Database Performance**: Test with realistic data volumes
- **Memory Usage**: Monitor memory consumption during large runs
- **Runtime Performance**: Ensure weekly runs complete within 3 hours

### Test Data Management

```python
class TestDataManager:
    def create_mock_screener_data(self, num_tickers: int) -> List[Dict]
    def create_mock_deep_scan_data(self, tickers: List[str]) -> List[Dict]
    def setup_test_database(self) -> DatabaseConnection
    def cleanup_test_data(self) -> None
```

## Integration Points

### Existing System Integration

1. **Data Downloaders**:
   - Use existing `FMPDataDownloader` and `FinnhubDataDownloader`
   - Extend with short squeeze specific methods if needed
   - Maintain compatibility with existing API key management

2. **Database System**:
   - **INTEGRATED**: Uses centralized database infrastructure in `src/data/db/`
   - **Models**: SQLAlchemy models in `src/data/db/models/model_short_squeeze.py`
   - **Repositories**: Data access layer in `src/data/db/repos/repo_short_squeeze.py`
   - **Services**: Business logic in `src/data/db/services/short_squeeze_service.py`
   - **Migrations**: Centralized migration system with `add_short_squeeze_tables.py`
   - **Connection Management**: Uses existing `session_scope()` and connection pooling

3. **Notification System**:
   - Use existing `NotificationSystem` for alerts
   - Support existing Telegram and email channels
   - Follow existing message formatting patterns

4. **Configuration System**:
   - Store pipeline-specific config in `src/ml/pipeline/p04_short_squeeze/config/`
   - Use existing environment variable patterns for sensitive data
   - Support existing configuration validation patterns

5. **Logging System**:
   - Use existing logger setup from `src.notification.logger`
   - Follow existing structured logging patterns
   - Integrate with existing log aggregation

### Future Scheduler Integration

The pipeline is designed with scheduler integration in mind:

```python
class SchedulerInterface:
    def register_weekly_job(self, job_func: Callable, schedule: str) -> str
    def register_daily_job(self, job_func: Callable, schedule: str) -> str
    def get_job_status(self, job_id: str) -> JobStatus
    def cancel_job(self, job_id: str) -> bool

class PipelineSchedulerAdapter:
    def __init__(self, scheduler: SchedulerInterface, config: SchedulingConfig)
    def schedule_screener(self) -> str
    def schedule_deep_scan(self) -> str
    def handle_job_failure(self, job_id: str, error: Exception) -> None
```

## Performance Considerations

### Scalability Design

1. **Batch Processing**: Process candidates in configurable batches to manage memory
2. **Parallel Processing**: Use thread pools for independent API calls
3. **Database Optimization**: Use bulk inserts and proper indexing
4. **Caching Strategy**: Cache frequently accessed configuration and reference data

### Resource Management

```python
class ResourceManager:
    def __init__(self, config: ResourceConfig):
        self.api_pool = ThreadPoolExecutor(max_workers=config.api_workers)
        self.db_pool = ConnectionPool(max_connections=config.db_connections)
        
    def manage_api_calls(self, calls: List[Callable]) -> List[Result]:
        """Manage concurrent API calls with rate limiting"""
        
    def batch_database_operations(self, operations: List[DatabaseOp]) -> None:
        """Batch database operations for efficiency"""
```

### Monitoring and Metrics

- **Runtime Metrics**: Track execution time for each module
- **API Usage**: Monitor API call counts and rate limit compliance
- **Data Quality**: Track percentage of successful data retrievals
- **Alert Performance**: Monitor alert accuracy and false positive rates

## Security Considerations

### API Key Management

- Use existing environment variable patterns for API keys
- Never log API keys or sensitive data
- Implement key rotation support for future use

### Data Privacy

- Store only necessary data for analysis
- Implement data retention policies
- Ensure compliance with financial data regulations

### Access Control

- Integrate with existing database access controls
- Use existing authentication patterns for future web interfaces
- Implement audit logging for sensitive operations

## Configuration Management

### Configuration Structure

```yaml
# config/short_squeeze_config.yml
scheduling:
  screener:
    frequency: weekly
    day: monday
    time: '08:00'
    timezone: Europe/Zurich
  deep_scan:
    frequency: daily
    time: '10:00'
    timezone: Europe/Zurich

screener:
  universe:
    min_market_cap: 100_000_000  # $100M
    max_market_cap: 10_000_000_000  # $10B
    min_avg_volume: 200_000
    exchanges: ['NYSE', 'NASDAQ']
  
  filters:
    si_percent_min: 0.15
    days_to_cover_min: 5.0
    float_max: 100_000_000
    top_k_candidates: 50
  
  scoring:
    weights:
      short_interest_pct: 0.4
      days_to_cover: 0.3
      float_ratio: 0.2
      volume_consistency: 0.1

deep_scan:
  batch_size: 10
  api_delay_seconds: 0.2
  
  metrics:
    volume_lookback_days: 14
    sentiment_lookback_hours: 24
    options_min_volume: 100
  
  scoring:
    weights:
      volume_spike: 0.35
      sentiment_24h: 0.25
      call_put_ratio: 0.20
      borrow_fee: 0.20

alerting:
  thresholds:
    high:
      squeeze_score: 0.8
      min_si_percent: 0.25
      min_volume_spike: 4.0
      min_sentiment: 0.6
    medium:
      squeeze_score: 0.6
      min_si_percent: 0.20
      min_volume_spike: 3.0
      min_sentiment: 0.5
    low:
      squeeze_score: 0.4
      min_si_percent: 0.15
      min_volume_spike: 2.0
      min_sentiment: 0.4
  
  cooldown:
    high_alert_days: 7
    medium_alert_days: 5
    low_alert_days: 3
  
  channels:
    telegram:
      enabled: true
      chat_ids: ['@trading_alerts']
    email:
      enabled: true
      recipients: ['trader@example.com']

adhoc:
  default_ttl_days: 7
  max_active_candidates: 20
  auto_promote_threshold: 0.7

reporting:
  weekly_summary:
    top_candidates: 20
    include_charts: true
    formats: ['html', 'csv']
  
  daily_report:
    top_scores: 10
    include_trends: true
    formats: ['html']

performance:
  api_rate_limits:
    fmp_calls_per_minute: 250  # Leave buffer from 300 limit
    finnhub_calls_per_minute: 50  # Leave buffer from 60 limit
  
  database:
    batch_size: 100
    connection_timeout: 30
    query_timeout: 60
  
  error_handling:
    max_retries: 3
    backoff_factor: 2.0
    circuit_breaker_threshold: 0.5
```

This design provides a comprehensive, modular, and extensible foundation for the Short Squeeze Detection Pipeline while maintaining seamless integration with the existing trading platform infrastructure.
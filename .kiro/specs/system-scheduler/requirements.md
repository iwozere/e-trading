# Requirements Document

## Introduction

The System Scheduler module provides APScheduler-based job scheduling and alert evaluation capabilities for the Advanced Trading Framework. This module processes alerts and schedules stored in the job_schedules table, evaluates complex rule-based alerts with rearm logic, and integrates with existing market data and notification systems.

## Glossary

- **System_Scheduler**: The main APScheduler-based service that manages job scheduling and execution
- **Alert_Evaluator**: Service component responsible for evaluating alert conditions and rearm logic
- **Cron_Parser**: Service component that parses and validates cron expressions (5 and 6 field formats)
- **Job_Service**: Database service layer for accessing job_schedules and job_schedule_runs tables
- **Market_Data_Manager**: Existing service for retrieving OHLCV and fundamental data
- **Indicator_Service**: Existing service for calculating technical indicators
- **Notification_Manager**: Component responsible for sending notifications to users
- **Worker_Thread**: APScheduler thread pool executor thread that processes individual jobs
- **Alert_State**: JSON-based state storage for maintaining alert evaluation context between runs

## Requirements

### Requirement 1

**User Story:** As a system administrator, I want the scheduler service to automatically load and schedule all enabled jobs from the database on startup, so that alerts and schedules continue running after service restarts.

#### Acceptance Criteria

1. WHEN the System_Scheduler starts, THE System_Scheduler SHALL load all enabled jobs from the job_schedules table
2. WHEN loading jobs, THE System_Scheduler SHALL calculate next_run_at timestamps using the cron field for each job
3. WHEN a job has an invalid cron expression, THE System_Scheduler SHALL log an error and skip that job
4. WHEN all jobs are loaded, THE System_Scheduler SHALL register them with APScheduler for execution
5. WHEN the database is unavailable during startup, THE System_Scheduler SHALL retry connection with exponential backoff

### Requirement 2

**User Story:** As a trading system user, I want my complex rule-based alerts to be evaluated accurately with proper rearm logic, so that I receive timely notifications without spam.

#### Acceptance Criteria

1. WHEN an alert job executes, THE Alert_Evaluator SHALL retrieve market data using the Market_Data_Manager
2. WHEN evaluating alert conditions, THE Alert_Evaluator SHALL compute required indicators using the Indicator_Service
3. WHEN an alert rule contains logical operators (and/or/not), THE Alert_Evaluator SHALL evaluate them correctly
4. WHEN an alert has rearm configuration, THE Alert_Evaluator SHALL apply rearm logic after trigger evaluation
5. WHEN alert state needs persistence, THE Alert_Evaluator SHALL update the state_json field in the database

### Requirement 3

**User Story:** As a system developer, I want alert and schedule configurations to be validated against JSON schemas, so that invalid configurations are rejected early and system stability is maintained.

#### Acceptance Criteria

1. WHEN processing an alert job, THE System_Scheduler SHALL validate task_params against the alert JSON schema
2. WHEN processing a schedule job, THE System_Scheduler SHALL validate task_params against the schedule JSON schema
3. WHEN validation fails, THE System_Scheduler SHALL log the error and mark the job as failed
4. WHEN a job type is not recognized, THE System_Scheduler SHALL log an error and skip the job
5. WHERE schema validation is enabled, THE System_Scheduler SHALL reject jobs with invalid task_params

### Requirement 4

**User Story:** As a trading system user, I want to receive notifications when my alerts trigger, so that I can take appropriate trading actions.

#### Acceptance Criteria

1. WHEN an alert successfully triggers, THE System_Scheduler SHALL call the TelegramBot FastAPI notification endpoint
2. WHEN the notification call fails, THE System_Scheduler SHALL log the error but continue processing
3. WHEN sending notifications, THE System_Scheduler SHALL include alert details and current market conditions
4. WHEN multiple alerts trigger simultaneously, THE System_Scheduler SHALL send individual notifications for each
5. WHEN the TelegramBot service is unavailable, THE System_Scheduler SHALL log the failure and continue operation

### Requirement 5

**User Story:** As a system administrator, I want comprehensive job execution tracking and error handling, so that I can monitor system health and troubleshoot issues.

#### Acceptance Criteria

1. WHEN a job starts execution, THE System_Scheduler SHALL create a job_schedule_runs record with RUNNING status
2. WHEN a job completes successfully, THE System_Scheduler SHALL update the run record with COMPLETED status and results
3. WHEN a job fails, THE System_Scheduler SHALL update the run record with FAILED status and error details
4. WHEN a job exceeds maximum runtime, THE System_Scheduler SHALL cancel it and mark as FAILED
5. WHEN database operations fail, THE System_Scheduler SHALL log errors and continue with other jobs

### Requirement 6

**User Story:** As a system developer, I want the scheduler to use existing data services and maintain clean separation of concerns, so that the system remains maintainable and follows established patterns.

#### Acceptance Criteria

1. WHEN accessing job data, THE System_Scheduler SHALL use only the Job_Service from src/data/db/services
2. WHEN retrieving market data, THE System_Scheduler SHALL use the Market_Data_Manager from src/data/data_manager.py
3. WHEN calculating indicators, THE System_Scheduler SHALL use the Indicator_Service from src/indicators/service.py
4. WHEN parsing cron expressions, THE System_Scheduler SHALL use the Cron_Parser service from src/common/alerts
5. WHEN evaluating alert logic, THE System_Scheduler SHALL use the Alert_Evaluator service from src/common/alerts

### Requirement 7

**User Story:** As a system administrator, I want the scheduler to handle both 5-field and 6-field cron expressions, so that I can schedule jobs with second-level precision when needed.

#### Acceptance Criteria

1. WHEN parsing a cron expression, THE Cron_Parser SHALL detect whether it has 5 or 6 fields
2. WHEN a 5-field cron expression is provided, THE Cron_Parser SHALL treat it as minute-level precision
3. WHEN a 6-field cron expression is provided, THE Cron_Parser SHALL treat it as second-level precision
4. WHEN a cron expression is invalid, THE Cron_Parser SHALL raise a validation error with details
5. WHEN calculating next run times, THE Cron_Parser SHALL return timezone-aware datetime objects

### Requirement 8

**User Story:** As a trading system user, I want my alert states to be preserved across service restarts, so that alert evaluation continues correctly without losing context.

#### Acceptance Criteria

1. WHEN an alert evaluation updates state, THE Alert_Evaluator SHALL persist the state to the state_json field
2. WHEN loading alert state, THE Alert_Evaluator SHALL parse the state_json field and handle missing or invalid data
3. WHEN alert state includes crossing detection data, THE Alert_Evaluator SHALL maintain side tracking across evaluations
4. WHEN state persistence fails, THE Alert_Evaluator SHALL log the error but continue with evaluation
5. WHERE state is missing or corrupted, THE Alert_Evaluator SHALL initialize with default state values
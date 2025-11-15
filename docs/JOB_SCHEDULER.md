# Job Scheduler System

A comprehensive job scheduling and execution system for reports and screeners using Postgres, Redis, Dramatiq, and APScheduler.

## Overview

The job scheduler system provides:
- **Ad-hoc execution**: Run reports and screeners immediately
- **Scheduled execution**: Set up recurring jobs with cron expressions
- **Snapshot-based execution**: Full reproducibility with complete job parameters
- **Scalable workers**: Dramatiq workers for async job processing
- **REST API**: FastAPI endpoints for job management

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   APScheduler   │    │   Dramatiq      │
│   (Web API)     │    │   (Scheduler)   │    │   (Workers)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   PostgreSQL    │
                    │   (Database)    │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │     Redis       │
                    │   (Message      │
                    │    Queue)       │
                    └─────────────────┘
```

## Components

### 1. Database Schema

**Tables:**
- `schedules`: Persistent schedule definitions
- `runs`: Job execution history with snapshots

**Key Features:**
- Unique constraints prevent duplicate runs
- JSONB snapshots for full reproducibility
- Atomic run claiming prevents duplicate execution

### 2. Screener Sets Configuration

**File:** `config/schemas/screener_sets.yml`

Predefined sets of tickers for screening:
- `us_large_caps`: S&P 500 constituents
- `tech_majors`: Major technology companies
- `crypto_majors`: Cryptocurrency-related stocks
- `dividend_aristocrats`: Companies with consistent dividend growth
- And many more...

### 3. FastAPI Endpoints

**Ad-hoc Execution:**
- `POST /api/reports/run` - Run report immediately
- `POST /api/screeners/run` - Run screener immediately
- `GET /api/runs/{run_id}` - Get run status
- `DELETE /api/runs/{run_id}` - Cancel pending run

**Scheduling:**
- `POST /api/schedules` - Create schedule
- `GET /api/schedules` - List schedules
- `PUT /api/schedules/{id}` - Update schedule
- `DELETE /api/schedules/{id}` - Delete schedule
- `POST /api/schedules/{id}/trigger` - Manual trigger

**Screener Sets:**
- `GET /api/screener-sets` - List available sets
- `GET /api/screener-sets/{name}` - Get set details

### 4. Dramatiq Workers

**Report Worker** (`src/backend/workers/report_worker.py`):
- Executes report generation jobs
- Supports multiple report types (system status, strategy performance, etc.)
- Saves artifacts to `artifacts/{run_id}/`

**Screener Worker** (`src/backend/workers/screener_worker.py`):
- Executes screener jobs
- Applies filter criteria to market data
- Returns top N results with scoring

### 5. Scheduler Process

**File:** `src/backend/scheduler/scheduler_process.py`

- Runs as separate process
- Checks for pending schedules every minute
- Expands screener sets to ticker lists
- Creates runs and enqueues jobs
- Updates next run times

## Quick Start

### 1. Database Setup

Run the migration to create the required tables:

```sql
-- Execute the SQL in db/migrations/add_job_tables.sql
```

### 2. Start Services

**Start Redis** (if not already running):
```bash
redis-server
```

**Start Dramatiq Workers:**
```bash
# Linux/Mac
./bin/start_dramatiq_workers.sh

# Windows
bin\start_dramatiq_workers.bat
```

**Start Scheduler Process:**
```bash
# Linux/Mac
./bin/start_job_scheduler.sh

# Windows
bin\start_job_scheduler.bat
```

**Start FastAPI Server:**
```bash
cd src/web_ui/backend
python main.py
```

### 3. Test the System

Run the test script to verify everything is working:

```bash
python examples/job_scheduler_test.py
```

## Usage Examples

### Create a Screener Schedule

```python
import requests

# Create a daily screener schedule
schedule_data = {
    "name": "Daily Tech Screener",
    "job_type": "screener",
    "target": "tech_majors",
    "task_params": {
        "filter_criteria": {
            "market_cap_min": 1000000000,
            "pe_ratio_max": 25,
            "return_on_equity_min": 0.15
        },
        "top_n": 20
    },
    "cron": "0 9 * * *",  # Daily at 9 AM
    "enabled": True
}

response = requests.post("http://localhost:8000/api/schedules", json=schedule_data)
```

### Run a Report Immediately

```python
# Run a system status report
report_data = {
    "report_type": "system_status",
    "parameters": {
        "include_details": True
    }
}

response = requests.post("http://localhost:8000/api/reports/run", json=report_data)
run_id = response.json()["run_id"]

# Check status
status_response = requests.get(f"http://localhost:8000/api/runs/{run_id}")
print(status_response.json()["status"])
```

### List Available Screener Sets

```python
response = requests.get("http://localhost:8000/api/screener-sets")
sets = response.json()

for screener_set in sets:
    print(f"{screener_set['name']}: {screener_set['description']}")
    print(f"  Tickers: {screener_set['ticker_count']}")
```

## Configuration

### Environment Variables

- `REDIS_HOST`: Redis host (default: localhost)
- `REDIS_PORT`: Redis port (default: 6379)
- `REDIS_DB`: Redis database (default: 0)
- `REDIS_PASSWORD`: Redis password (optional)

### Screener Sets

Edit `config/schemas/screener_sets.yml` to:
- Add new screener sets
- Modify existing ticker lists
- Update filter criteria

The configuration is loaded at startup and can be reloaded without restart.

## Monitoring

### Run Statistics

Get statistics about job execution:

```python
response = requests.get("http://localhost:8000/api/runs/statistics?days=30")
stats = response.json()

print(f"Total runs: {stats['total_runs']}")
print(f"Status counts: {stats['status_counts']}")
```

### Logs

- **Scheduler logs**: Check console output of scheduler process
- **Worker logs**: `logs/dramatiq_workers.log`
- **API logs**: FastAPI application logs

## Troubleshooting

### Common Issues

1. **Redis Connection Error**
   - Ensure Redis is running
   - Check Redis host/port configuration

2. **Database Connection Error**
   - Ensure PostgreSQL is running
   - Check database credentials

3. **Worker Not Processing Jobs**
   - Check worker logs for errors
   - Verify Redis connection
   - Ensure workers are subscribed to correct queues

4. **Scheduler Not Triggering Jobs**
   - Check scheduler process logs
   - Verify cron expressions are valid
   - Check database connection

### Debug Mode

Enable debug logging by setting the log level:

```python
import logging
logging.getLogger("src.backend").setLevel(logging.DEBUG)
```

## Development

### Adding New Report Types

1. Add report type to `_execute_report()` in `report_worker.py`
2. Implement the report generation function
3. Update API documentation

### Adding New Filter Criteria

1. Add filter logic to `_apply_filters()` in `screener_worker.py`
2. Update API documentation
3. Test with different criteria combinations

### Custom Screener Sets

1. Add new set to `config/schemas/screener_sets.yml`
2. Define tickers and metadata
3. Test with the API endpoints

## Security

- All endpoints require authentication
- Users can only access their own schedules/runs
- Admin users have full access
- Input validation on all API endpoints

## Performance

- Workers can be scaled horizontally
- Database indexes optimize query performance
- Redis provides fast message queuing
- Artifacts are stored on disk for large results

## Future Enhancements

- Web UI for job management
- Email notifications for job completion
- Job result caching
- Advanced scheduling (timezone support, holidays)
- Integration with existing trading system
- Real-time job progress updates



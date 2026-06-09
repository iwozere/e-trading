# P18 — Institutional Flow Tracker

## Overview

Tracks where large financial institutions ($1B+ AUM) are reducing equity positions,
combining quarterly 13F filings with daily Form 4 insider transactions and volume
anomaly detection.  Runs as a daily `DATA_PROCESSING` scheduler job.

## Signal Architecture

```
Layer 1  SEC EDGAR 13F-HR (quarterly, updated daily during 45-day filing window)
         → position deltas → exit screener → consensus detector

Layer 2  SEC Form 4 + Schedule 13D/G (near-real-time, filed within 2 business days)
         → significant insider sells → ownership stake drops

Layer 3  DataManager OHLCV (daily)
         → volume anomaly detection on current watchlist
```

## Quick Start

```python
from src.ml.pipeline.p18_institutional_flow_tracker.pipeline import InstitutionalFlowPipeline
from src.ml.pipeline.p18_institutional_flow_tracker.config import P18Config

pipeline = InstitutionalFlowPipeline(P18Config.create_default())
result = pipeline.run()
print(result)
```

Run manually via CLI:

```bash
python src/ml/pipeline/p18_institutional_flow_tracker/run_p18_scan.py --as-of-date 2024-02-14
```

## Scheduler Registration

```sql
-- Daily at 07:00 UTC
INSERT INTO job_schedules (user_id, name, job_type, cron, task_params, enabled)
VALUES (
  1, 'P18 Institutional Flow Daily', 'data_processing',
  '0 7 * * *',
  '{
    "script_path": "src/ml/pipeline/p18_institutional_flow_tracker/run_p18_scan.py",
    "script_args": [],
    "notification_rules": {
      "conditions": [{
        "check_field": "high_score_count",
        "operator": ">=",
        "threshold": 1,
        "channels": ["telegram"]
      }]
    }
  }',
  true
);
```

## Cache Layout

```
DATA_CACHE_DIR/
    edgar/13f/
        index/<YYYY>_Q<N>.csv.gz          # 13F filer index per quarter
        holdings/<YYYY>_Q<N>/<cik>.csv.gz  # per-institution holdings
        consensus/<YYYY>_Q<N>.csv.gz       # computed consensus exits
        form4/<YYYY-MM-DD>.csv.gz          # Form 4 filings per day
        13dg/<YYYY-MM-DD>.csv.gz           # 13D/G filings per day
    openfigi/cusip_map.csv.gz              # accumulated CUSIP→ticker cache
```

## Output Signals

| Signal | Trigger | Use case |
|--------|---------|----------|
| Avoidance | Score ≥60, consensus exit | Remove from buy watchlist |
| Contrarian entry | High-quality stock + forced selling | Dislocation opportunity |
| Sector rotation | Multiple exits in same sector | Rebalance sector weights |

## Related Documentation

- [Plan](docs/plan.md) — Implementation plan
- [Input ideas](docs/input.md) — Original signal research notes
- [Requirements](docs/Requirements.md)
- [Design](docs/Design.md)
- [Tasks](docs/Tasks.md)

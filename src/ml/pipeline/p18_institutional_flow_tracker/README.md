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

## Quarterly Consensus Backfill (important)

The daily scan **only loads** the quarterly 13F consensus cache
(`consensus/<YYYY>_Q<N>.csv.gz`); on a day with no new filings it does not build
it. If that cache is missing the daily scan produces **zero signals** (empty
results folder) — see the `run_summary.json` `consensus_rows` field to confirm.

Building the consensus is a **heavy** job (thousands of filers × EDGAR rate-limit
+ FIGI CUSIP mapping = hours; ~8 h cold on the Pi).

### Scheduled: "P18 Quarterly Consensus Backfill"

Registered in `src/data/pipeline/specs/p18_specs.py` and applied with
`python -m src.data.pipeline.register_jobs`. It runs
`backfill_consensus.py --auto-quarter` at **08:00 UTC on the 15th of Feb, May,
Aug and Nov** (13F-HR are due 45 days after quarter-end) with a 24 h
`timeout_seconds`, and sends a Telegram message if it fails.

`--auto-quarter` targets the most recently completed quarter. Infotables already
on disk are reused, so triggering the job again later in the month (Run now)
only downloads filers that were not cached yet and rebuilds the consensus. Use
this when the deadline moved past the 15th (weekend/holiday), to pick up late
filers, or when the run was missed (the scheduler drops firings that are more
than 5 minutes late, e.g. during a restart) or killed by a scheduler restart.

Logs go to the scheduler's `app.log`; the run's outcome is in `job_schedule_runs`.

### Manual one-off (recovery / first seed)

```bash
# Detached so a dropped SSH session / restart can't kill it:
nohup python src/ml/pipeline/p18_institutional_flow_tracker/backfill_consensus.py \
      --year 2026 --quarter 1 > /tmp/p18_backfill.log 2>&1 &
```

Watch for `Backfill complete: N tickers in consensus for <Y> Q<N>` (success) or
`Backfill produced an empty consensus` (failure — check EDGAR connectivity and
that the filing window has closed).

### Daily-scan timeout during filing season

During filing windows the *daily* scan can also approach its 3600s timeout
because it downloads each new filer's infotable. See
`src/data/pipeline/specs/p18_specs.py` for how to seasonally raise
`timeout_seconds` for the daily scan (this is the incremental job and is fine to
run through the scheduler — only the full rebuild must not be). The old
`bin/scheduler/insert_p18_schedules.sql` this used to point to is archived
under `bin/scheduler/archive/` — no longer the live source.

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

# P20 Kestrel — Semi-Automated Trading Intelligence

## Overview

P20 Kestrel is a three-sleeve trading intelligence pipeline that screens, scores, and monitors equity candidates. All orders are placed manually by the human operator — the pipeline generates intelligence and alerts only.

## Sleeves

- **Sleeve A (Turnaround / Fallen Angels)**: Stocks down 40–75% from 2-year highs with catalyst for recovery. Scored via insider buying, balance sheet, technicals, sentiment.
- **Sleeve B (Event Catalysts)**: FDA run-ups (B1), spin-offs (B2), activists/index events (B3).
- **Sleeve C (Momentum)**: RS-ranked momentum names in SPY-above-200DMA regime, crowding-filtered.

## Daily Schedule (UTC)

| Time  | Cadence       | Job                           |
|-------|---------------|-------------------------------|
| 03:00 | Mon–Fri       | Google Trends poll            |
| 05:00 | Monday        | Weekly maintenance            |
| 06:00 | Mon–Fri       | Data health check             |
| 06:15 | Mon–Fri       | GDELT processing              |
| 06:30 | Mon–Fri       | Social sentiment poll         |
| 06:30 | Mon–Fri       | Daily digest send             |
| 06:45 | Mon–Fri       | AV news sentiment             |
| 07:00 | Mon–Fri       | Sentiment aggregation         |
| 20:00 | Mon–Fri       | EOD ingest                    |
| 20:30 | Mon–Fri       | Filings ingest                |
| 20:45 | Mon–Fri       | Catalyst sync                 |
| 21:00 | Mon–Fri       | Sleeve A screen               |
| 21:15 | Mon–Fri       | Sleeve B screen (B1/B2/B3)   |
| 21:30 | Mon–Fri       | Sleeve C RS rank              |
| 22:00 | Mon–Fri       | LLM 8-K classification        |
| 22:30 | Mon–Fri       | LLM dossier generation        |
| */30m | Mon–Fri 09–17 | Risk check                    |
| 17:00 | Sunday        | Weekly report                 |
| 18:00 | Sunday        | LLM risk diff (10-K/Q)        |

## Quick Start

```python
# Run Sleeve A screen manually
from src.ml.pipeline.p20_kestrel.screening.sleeve_a import run
result = run()

# Parse a Telegram /pos command
from src.ml.pipeline.p20_kestrel.pos.pos_commands import handle_command
reply, pending = handle_command("/pos add AAPL A 150 2.0")

# Register all jobs (run once at deploy)
from src.ml.pipeline.p20_kestrel.jobs.register_jobs import run as register
register()
```

## Deploy Runbook

One-time setup (run in order):

```bash
# 1. Apply DB migration (creates all k20_* tables)
python src/data/db/migrations/002_kestrel_schema.py

# 2. Insert scheduler jobs (idempotent — safe to re-run)
psql -d $DB_NAME < bin/scheduler/insert_p20_schedules.sql

# 3. Seed universe (first run — downloads Nasdaq screener via P15 weekly first)
python src/ml/pipeline/p20_kestrel/jobs/run_weekly_maintenance.py

# 4. Seed alias table (alias builder is included in weekly_maintenance)
# Already done by step 3

# 5. Enable jobs in the scheduler
# Either restart the scheduler service or wait for LISTEN/NOTIFY reload
```

GDELT backfill (catch up on historical sentiment — run before first Monday):

```bash
python src/ml/pipeline/p20_kestrel/jobs/run_gdelt_backfill.py \
    --start 2024-01-01 --end 2024-12-31
```

## Telegram Bot Hook — /pos Handler

Wire the `/pos` command into your Telegram bot handler:

```python
from src.ml.pipeline.p20_kestrel.pos.pos_commands import (
    PosCommandError,
    confirm_add,
    handle_command,
)

# In-memory store keyed by user_id for pending confirmations
_pending: dict[int, dict] = {}

async def on_message(bot, message):
    text = message.text or ""
    user_id = message.from_user.id

    if not text.startswith("/pos"):
        return

    # Check if this is a confirmation tap (inline button callback)
    # — handled separately in on_callback_query below

    try:
        reply, pending = handle_command(text)
        if pending:
            _pending[user_id] = pending
            # Show inline confirm button alongside the echo card
            await bot.send_message(user_id, reply, reply_markup=confirm_keyboard())
        else:
            await bot.send_message(user_id, reply)
    except PosCommandError as exc:
        await bot.send_message(user_id, f"Error: {exc}")


async def on_callback_query(bot, query):
    if query.data != "pos_confirm":
        return
    user_id = query.from_user.id
    pending = _pending.pop(user_id, None)
    if not pending:
        await bot.answer_callback_query(query.id, "No pending position.")
        return
    confirmed = confirm_add(pending)
    await bot.answer_callback_query(query.id, "Position opened.")
    await bot.send_message(
        user_id,
        f"✅ {confirmed['ticker']} (Sleeve {confirmed['sleeve']}) opened "
        f"@ ${confirmed['entry_px']:.4f} (id={confirmed['id']})",
    )
```

## Module Structure

```
p20_kestrel/
├── config.py              # All tuning constants
├── db/
│   ├── models.py          # SQLAlchemy ORM for k20_* tables
│   └── repos.py           # Repository layer (no Session leakage)
├── ingest/
│   ├── universe_loader.py # Weekly Nasdaq screener + fundamentals
│   ├── eod_ingest.py      # Daily OHLCV + technicals
│   ├── filings_ingest.py  # Form 4, 8-K, 13D/G
│   └── calendar_sync.py   # Catalyst T-10/T-3 alerts
├── sentiment/
│   ├── alias_builder.py   # Company alias table (GDELT matching)
│   ├── gdelt_processor.py # GKG z-score pipeline
│   ├── social_poll.py     # StockTwits / Reddit / ApeWisdom
│   ├── trends_poll.py     # Google Trends
│   ├── av_budgeted.py     # AlphaVantage news (budget-capped)
│   └── sentiment_aggregator.py  # §7.6 crowding score
├── screening/
│   ├── sleeve_a.py        # Turnaround hard filters + scoring
│   ├── sleeve_b.py        # FDA run-up + activist screen
│   └── sleeve_c.py        # RS rank + regime filter
├── llm/
│   ├── prompts.py         # All prompt constants
│   ├── client.py          # Budget-aware Anthropic client
│   ├── classifier_8k.py   # 8-K thesis impact classifier
│   ├── dossier.py         # Full candidate dossier generator
│   └── risk_diff.py       # 10-K/Q risk factor change detector
├── risk/
│   └── risk_checker.py    # Intraday stop/target/loss check
├── pos/
│   └── pos_commands.py    # /pos Telegram command parser
├── reporting/
│   ├── daily_digest.py    # 07:30 digest builder + sender
│   ├── data_health.py     # 07:00 freshness guard
│   └── weekly_report.py   # Sunday 18:00 performance report
└── jobs/
    ├── register_jobs.py   # One-time job schedule registration (19 jobs)
    └── run_*.py           # Scheduler entry points (19 scheduled + 1 manual backfill)
```

## Integration

- `src.data.db` — PostgreSQL via SQLAlchemy session_scope()
- `src.notification.service.client` — Push alerts
- `src.data.market_data.data_manager` — OHLCV feeds
- `src.data.fundamentals` — get_fundamentals_unified()
- `src.data.edgar` — EDGAR Form 4, 8-K, 10-K/Q, 13D/G
- P15 GDELT GKG files at `R:/data-cache/gdelt/gkg/`

## Related Documentation

- [Pipeline Specification](docs/pipeline-specification.md) — v1.2 full spec
- [Implementation Plan](docs/implementation-plan.md) — architecture decisions
- [Requirements](docs/Requirements.md) — dependencies
- [Design](docs/Design.md) — architecture overview
- [Tasks](docs/Tasks.md) — implementation status

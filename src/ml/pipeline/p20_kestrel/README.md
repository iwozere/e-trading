# P20 Kestrel — Semi-Automated Trading Intelligence

## Overview

P20 Kestrel is a three-sleeve trading intelligence pipeline that screens, scores, and monitors equity candidates. All orders are placed manually by the human operator — the pipeline generates intelligence and alerts only.

## Sleeves

- **Sleeve A (Turnaround / Fallen Angels)**: Stocks down 40–75% from 2-year highs with catalyst for recovery. Scored via insider buying, balance sheet, technicals, sentiment.
- **Sleeve B (Event Catalysts)**: FDA run-ups (B1), spin-offs (B2), activists/index events (B3).
- **Sleeve C (Momentum)**: RS-ranked momentum names in SPY-above-200DMA regime, crowding-filtered.

## Daily Schedule (UTC)

| Time  | Job                        |
|-------|----------------------------|
| 03:00 | Google Trends poll         |
| 04:00 | Universe refresh (Mon)     |
| 06:00 | Data health check          |
| 06:15 | GDELT processing           |
| 06:30 | Social sentiment poll      |
| 06:30 | Daily digest send          |
| 06:45 | AV news sentiment          |
| 07:00 | Sentiment aggregation      |
| 20:00 | EOD ingest                 |
| 20:30 | Filings ingest             |
| 20:45 | Catalyst sync              |
| 21:00 | Sleeve A screen            |
| 21:15 | Sleeve B screen            |
| 21:30 | Sleeve C RS rank           |
| 22:00 | LLM 8-K classification     |
| 22:30 | LLM dossier generation     |
| */30m | Risk check (09–17 UTC)     |
| 17:00 | Weekly report (Sunday)     |

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
    ├── register_jobs.py   # One-time job schedule registration
    └── run_*.py           # Scheduler entry points (18 scripts)
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

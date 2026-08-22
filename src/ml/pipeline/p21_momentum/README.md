# P21 Momentum

## Overview

A DIY 20-stock momentum strategy, paper-traded, with a monthly QDVA (MSCI USA Momentum ETF) comparison.
Three scheduled jobs compute a 12-1 month risk-adjusted momentum signal over the S&P 500, select and size a
20-name portfolio with a sector cap and a regime overlay, simulate fills the next trading day, and mark NAV
daily across five attribution tracks (A: DIY+overlay, B: DIY no overlay, C: MTUM no overlay, D: MTUM+overlay,
E: SPY anchor).

**This is a paper simulation, not a return forecast.** See `docs/pipeline-specification.md` §0 before reading
any output — a 20-position portfolio's tracking error against its benchmark makes month-to-month return
comparisons statistically meaningless for years. The pipeline's actual purpose is operational validation: does
the code run cleanly, does the operator keep the monthly discipline, and does the regime overlay behave as
designed during stress.

## Features

- 12-1 month risk-adjusted momentum signal (§4), ranked on `raw_return / vol`, never on raw return alone
- Six-filter survivor screen (history, liquidity, gap-dominance, quality, manual exclusions, earnings blackout)
- Rank + hysteresis (`ENTRY_RANK=20` / `HOLD_RANK=60`) + sector cap (max 4/sector) selection
- Inverse-volatility position sizing with a hard 1%-of-NAV per-position cap
- Bear/high-vol regime overlay with asymmetric hysteresis (downgrade immediate, upgrade needs 2 confirmed months)
- Deterministic fill simulation (slippage + IBKR-tiered commissions) and an append-only trade ledger
- Five-track NAV attribution answering "does the DIY approach beat QDVA + the same overlay" (§9)
- A §13 data-quality gate suite that aborts a run rather than writing bad state

## Quick Start

```bash
# Smoke-test the rebalance job locally (writes under results/p21_momentum/)
python -m src.ml.pipeline.p21_momentum.jobs.run_monthly_rebalance

# Force a re-run even if today's targets.json already exists
python -c "from src.ml.pipeline.p21_momentum.jobs import run_monthly_rebalance as j; print(j.run(force=True))"
```

Afterward, inspect:

```
results/p21_momentum/YYYY-MM-DD/    # this run's universe.json, signals.json, targets.json, ...
results/p21_momentum/_state/        # current_positions.json, ledger.jsonl, nav_daily.csv, regime_history.json
```

Every dated folder is a permanent, browsable snapshot — `ls results/p21_momentum/` shows the full run history.

## Integration

This module integrates with:
- `src.data.downloader` — all price/fundamentals fetching goes through `YahooDataDownloader`, never raw `yfinance`
- `src.util.tickers_list` — S&P 500 constituents + GICS sector (`get_sp500_constituents_with_sector`)
- `src.notification` — logger convention and ABORT-alert delivery (`NotificationServiceClient`)
- The project scheduler (`job_schedules` table) — see `jobs/register_jobs.py`

## Configuration

All parameters live in `config.py` as module constants (spec §16) — there is no YAML/JSON config for
strategy parameters, only for the two operator-facing files:
- `config/pipeline/p21_exclusions.json` — manual M&A/special-situation exclusion list (read-only to the pipeline)
- `config/frozen_params.json` — written once after Phase 1 ends; every job verifies against it thereafter

## Related Documentation
- [Requirements](docs/Requirements.md) — dependencies and external services
- [Design](docs/Design.md) — architecture and key design decisions
- [Tasks](docs/Tasks.md) — implementation roadmap and status
- [Specification](docs/pipeline-specification.md) — the executable spec this module implements
- [Implementation Plan](docs/implementation-plan.md) — build-order plan derived from the spec

# P05 AI Selector

## Overview

Daily AI-powered screener that combines quantitative signals with Claude LLM synthesis to produce a ranked shortlist of the top 5 equities and crypto instruments worth watching — each with a complete position management guide (entry conditions, hold conditions, thesis-breakers, profit targets).

Runs weekdays at 10:00 UTC after P18 (07:00 UTC) ensures all institutional flow signals are available.

## Features

- **4-stage funnel**: ~3,020 symbols → ~200 (liquidity filter) → top-25 (deterministic scoring) → top-5 (LLM synthesis)
- **Signal confluence**: technical (RSI, SMA, volume, ATR, momentum) + fundamental (P/E, margins, debt, growth) + P18 institutional flow boosts
- **Full exit strategies**: entry conditions, hold conditions, thesis-breakers, profit targets, time-horizon notes
- **Dual-trigger notifications**: P18 high-score count ≥ 1 OR LLM confidence ≥ 9

## Quick Start

```python
from src.ml.pipeline.p05_ai_selector.pipeline import P05Pipeline
from datetime import date

result = P05Pipeline().run(user_id="1", as_of_date=date(2026, 6, 14))
print(result["top_ticker"], result["top_confidence"])
```

Or via CLI (scheduler entry point):

```bash
python src/ml/pipeline/p05_ai_selector/run_p05_scan.py --user-id 1 --as-of-date 2026-06-14
```

## Universe

| Source | Count | Provider |
|--------|-------|----------|
| Russell 3000 equities | ~3,000 | `Russell3000Downloader` |
| Top-20 crypto | 20 | Fixed list (BTC-USD, ETH-USD, …) |

## Output Files

```
results/p05_ai_selector/{YYYY-MM-DD}/
    top_picks.csv        ← top 5 picks with first profit target
    full_ranking.csv     ← top 25 with all signal scores
    report.md            ← narrative report with full exit strategies
    metadata.json        ← run metadata: timing, counts, tokens used
```

## Cold Start Warning

The first ever run fetches 60-day OHLCV for all ~3,020 symbols via DataManager. This takes approximately 20–40 minutes. Subsequent runs are fast (gap fill only — typically 1–2 days).

## Integration

This module integrates with:
- `src/data/downloader/russell3000_downloader.py` — universe list
- `src/data/data_manager.py` — OHLCV and fundamentals
- `results/p18_institutional_flow/{date}/` — P18 signal boosts
- `src/notification/` — Telegram and email delivery
- `src/scheduler/scheduler_service.py` — daily 10:00 UTC scheduling

## Related Documentation

- [Requirements](docs/Requirements.md)
- [Design](docs/Design.md)
- [Tasks](docs/Tasks.md)
- [Pipeline Specification](docs/pipeline-specification.md)

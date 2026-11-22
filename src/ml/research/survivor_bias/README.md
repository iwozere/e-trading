# Survivor Bias Study

## Overview

Measures how much the results recorded on the FMP vintage were flattered by its missing delisted stocks. The same
frozen rules run on three subsets of the EODHD store: every common stock (`full`), the stocks the vintage also has
(`vintage_like`), and today's survivors (`survivors`). Prices are identical across the three, so the difference is the
survivor effect alone. Delisted stocks exit at their last close, with a −30% haircut on distressed ends as a second
version.

The rules were written down before any result was seen: [Design](docs/Design.md).

## Features

- Month-end panel of 22,000+ stocks from the EODHD store (cached in the data root, not in git)
- Delisting rule: live / ordinary / distressed end, `last_close` and `haircut` versions
- Monthly strategies: equal-weight universe, 12-1 momentum top decile, P21 momentum core
- P24 falling knife on the EODHD store with the frozen live parameters
- Period statistics and bias tables (`vintage_like` − `full`, `survivors` − `full`)

## Quick Start

```bash
.venv/Scripts/python.exe -X utf8 src/ml/research/survivor_bias/jobs/run_survivor_bias.py
.venv/Scripts/python.exe -X utf8 src/ml/research/survivor_bias/jobs/run_survivor_bias.py --skip-knife --run-id quick
```

Output in `results/survivor_bias/<run_id>/`: `monthly_strategies.csv`, `bias_cagr.csv`, `bias_excess.csv`, `p24.csv`,
`bias_p24_mean.csv`, `end_kinds.csv`.

## Integration

This module integrates with:
- `src.data.eodhd_pipeline` — `EodhdStore` (prices, splits, dividends, FMP mapping)
- `src.data.fmp_pipeline.adjust` — `adjust()` for total-return closes
- `src.ml.pipeline.p24_falling_knife` — detector, exits, cost tiers, portfolio replay, universe helpers
- `src.notification.logger` — logging

## Configuration

All rule parameters are in `config.py`. The EODHD data root defaults to `<DATA_CACHE_DIR>/eodhd-data`
(`--data-root` overrides).

## Related Documentation
- [Requirements](docs/Requirements.md) - Technical requirements
- [Design](docs/Design.md) - Frozen rules, architecture and design
- [Tasks](docs/Tasks.md) - Status and open questions

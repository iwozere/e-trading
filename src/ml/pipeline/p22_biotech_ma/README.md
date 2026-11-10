# P22 Biotech M&A

## Overview
Reproducible, point-in-time-correct screening pipeline that ranks US-listed biotech companies
by likelihood of acquisition within 12–24 months, and surfaces the reasoning behind each rank.
See `docs/pipeline-specification.md` for the full technical specification (v0.5) and
`docs/implementation-plan.md` for how it maps onto this repo's existing infrastructure.

**Non-goals:** no order execution, no price prediction, no investment advice, no clinical-outcome
prediction. Every exported artifact carries the disclaimer in spec §11.

## Features (M1 — current)
- Bitemporal Postgres schema (`p22_*` tables) covering the full data model in spec §3.2.
- Daily raw-zone ingest landing: SEC EDGAR (submissions + XBRL company facts), ClinicalTrials.gov,
  openFDA Drugs@FDA.
- Quarterly raw-zone ingest landing: FDA Orange Book and Purple Book.
- Rate-limited, idempotent, content-addressed raw-zone storage (`DATA_CACHE_DIR/p22/raw/...`).

## Quick Start
```python
from src.ml.pipeline.p22_biotech_ma.ingest.clinicaltrials_client import ClinicalTrialsClient

client = ClinicalTrialsClient()
studies = client.fetch_studies_for_sponsor("Example Therapeutics Inc")
```

Jobs are run as standalone scripts (see `jobs/`) and registered with the scheduler via
`src.data.pipeline` (`specs/p22_specs.py`), which supersedes this module's own
`jobs/register_jobs.py`.

Historical raw prices and corporate actions (delisted companies included) are loaded by hand from
the shared FMP vintage, not by a scheduled job:

```bash
python -m src.ml.pipeline.p22_biotech_ma.cli.fmp_vintage_load_cli --dry-run   # match report only
python -m src.ml.pipeline.p22_biotech_ma.cli.fmp_vintage_load_cli --limit 5
python -m src.ml.pipeline.p22_biotech_ma.cli.fmp_vintage_load_cli
```

Read-only size-floor checks (`ingest/size_floor.py`; need the `public_float` fact, written by
`jobs/run_financial_facts_normalization.py`):

```bash
python -m src.ml.pipeline.p22_biotech_ma.cli.size_floor_report_cli price-quality --out /tmp/p22_price_quality.csv
python -m src.ml.pipeline.p22_biotech_ma.cli.size_floor_report_cli history --since 2015q1
```

Deal labels (spec §2.5, M6) are hand-verified from the `deal_candidate` review queue, one at a time
or through a CSV round trip (`ingest/deal_entry.py` holds the logic, premiums and reverse-merger
flags). `scan` queues historical candidates; the daily job only looks back one week:

```bash
python -m src.ml.pipeline.p22_biotech_ma.cli.deal_entry_cli scan --start 2010-01-01
python -m src.ml.pipeline.p22_biotech_ma.cli.deal_entry_cli export deals.csv
python -m src.ml.pipeline.p22_biotech_ma.cli.deal_entry_cli import deals.csv --reviewer ak --dry-run
python -m src.ml.pipeline.p22_biotech_ma.cli.deal_entry_cli list-deals
python -m src.ml.pipeline.p22_biotech_ma.cli.deal_entry_cli amend 6468 --by-item --reviewer ak --upfront-per-share 65.60
```

`scan` searches SC 14D9, DEFM14A, DEFM14C and S-4 filings that list a universe company's CIK. Deals it
cannot see (only the acquirer filed, foreign schemes of arrangement) are queued by hand from a filing URL,
then labelled with `enter`/`link`/`reject` as usual:

```bash
python -m src.ml.pipeline.p22_biotech_ma.cli.deal_entry_cli add-candidate --cik 1586105 --form S-4 --filed 2020-04-08 \
    --url https://www.sec.gov/Archives/edgar/data/1808665/000104746920002165/a2241255zs-4.htm --note "Assertio S-4 for Zyla"
```

`amend` corrects an already-entered deal (by `deal_id`, or by the review item it came from with
`--by-item`); premiums and reverse-merger flags are re-derived, `-` clears a field.

Phase 1 scoring (spec §5.1–5.2, M4) runs daily as the `P22 Scoring` job and can be run by hand;
weights live in `config/pipeline/p22_weights.yaml`:

```bash
python src/ml/pipeline/p22_biotech_ma/jobs/run_scoring.py --dry-run   # watchlist + dossiers, DB rolled back
```

Domicile history for backtests (filing-time business addresses; rerunnable, inserts only older history):

```bash
python -m src.ml.pipeline.p22_biotech_ma.cli.domicile_history_cli --start 2010-01-01 --dry-run
python -m src.ml.pipeline.p22_biotech_ma.cli.collapse_facts_cli --dry-run   # merge repeated daily domicile rows
```

Label coverage, walk-forward backtest (spec §0.3, M6) and the lookahead audit (spec §8.3), all read-only:

```bash
python -m src.ml.pipeline.p22_biotech_ma.cli.deal_entry_cli coverage --since 2010-01-01 --out coverage.csv
python -m src.ml.pipeline.p22_biotech_ma.cli.backtest_cli --labels-through 2026-10-30   # results/.../backtest/
python -m src.ml.pipeline.p22_biotech_ma.cli.lookahead_audit_cli --samples 200          # exit 1 on a violation
```

Weekly automation (2026-11-09): `P22 Weekly Backtest` (Saturday 06:00 UTC) runs the backtest with the
label cut-off taken from the review queue, retrains the live learned model that `P22 Scoring` ranks the
watchlist with, and emails lift vs the baselines; `P22 Label Review Digest` (Monday 07:30 UTC) emails the
deal candidates waiting for review. Since 2026-11-10 the backtest also reports portfolio returns: each
ranking's top 25/50 bought at every snapshot and held 24 months or to the deal, against the whole universe
(`portfolio_learned_window.csv`). Both can be run by hand:

```bash
python src/ml/pipeline/p22_biotech_ma/jobs/run_backtest.py --no-email           # about an hour on the Pi
python src/ml/pipeline/p22_biotech_ma/jobs/run_label_review_digest.py --no-email
```

## Integration
This module integrates with:
- `src.data.downloader.edgar_downloader` — SEC EDGAR access (submissions, XBRL, EFTS, 13F, Form 4, 13D/G)
- `src.data.fmp_pipeline` — read-only FMP vintage (`FmpStore`): historical raw daily bars, splits, dividends
- `src.data.db` — Postgres storage (models, migrations, repos, UoW service)
- `src.data.utils.rate_limiting` — per-host token-bucket rate limiting
- `src.notification.logger` — project-wide logging

## Configuration
See `config.py` for cache paths, source URLs, and feature flags (e.g. the market-data vendor
adapter is currently a stub — see `docs/Tasks.md`).

## Related Documentation
- [Pipeline Specification](docs/pipeline-specification.md) — the normative spec
- [Implementation Plan](docs/implementation-plan.md) — reuse map, spec/repo deviations, M1 scope
- [Requirements](docs/Requirements.md)
- [Design](docs/Design.md)
- [Tasks](docs/Tasks.md)

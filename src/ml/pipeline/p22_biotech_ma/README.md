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
`jobs/register_jobs.py`, following the same pattern as `p20_kestrel`.

## Integration
This module integrates with:
- `src.data.downloader.edgar_downloader` — SEC EDGAR access (submissions, XBRL, EFTS, 13F, Form 4, 13D/G)
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

# Requirements

## Python Dependencies
- `httpx` — shared rate-limited HTTP client for CT.gov / openFDA / Orange Book / Purple Book fetches
- `sqlalchemy` >= 2.0 — ORM (already a repo dependency)
- `alembic` — migrations (already a repo dependency)
- `pandera` >= 0.24.0 — **correction, 2026-08-30:** this line previously claimed pandera was "already
  a repo dependency" — that was never actually verified and was wrong; it wasn't in `requirements.txt`
  or installed anywhere in the repo before this session. Added for real in M3 (`features/quality.py`,
  spec §8.2), verified via `pip install` + a real `DataFrameSchema.validate()` call this time, not
  just asserted.
- `pydantic` v2 — schema validation for ingest payloads (already a repo dependency)
- `rapidfuzz` >= 3.9.0 — **new dependency, added M2** — spec §3.3's token-set-ratio fuzzy alias
  matching (`ingest/alias_matching.py`); nothing else in this repo already does fuzzy string matching.
- `PyYAML` >= 6.0.3 — parses `config/pipeline/p22_acquirers.yaml` (`ingest/acquirer_config.py`, M3).
  Already installed and used by ~15 other repo modules (e.g.
  `p04_short_squeeze/config/config_manager.py`) but was not pinned in the root `requirements.txt` — a
  pre-existing, repo-wide gap discovered 2026-08-30; pinned for real the same day, at the version
  already installed.

## External Dependencies (in-repo)
- `src.data.downloader.edgar_downloader.EdgarDownloader` — SEC EDGAR submissions, XBRL company
  facts, full-text search, 13F, Form 4, Schedule 13D/G
- `src.data.db.core.base.Base`, `src.data.db.core.database.session_scope` — shared ORM/session layer
- `src.data.db.services.database_service.DatabaseService` — Unit-of-Work (`ReposBundle`)
- `src.data.utils.rate_limiting.RateLimiter` — per-host token-bucket rate limiting
- `src.notification.logger.setup_logger` — logging

## External Services
- **SEC EDGAR** (`www.sec.gov`, `data.sec.gov`, `efts.sec.gov`) — public, free. Requires a
  descriptive `User-Agent` (already set by `EdgarDownloader`'s default:
  `"e-trading-research akossyrev@gmail.com"`). Hard cap 10 requests/second.
- **ClinicalTrials.gov API v2** (`clinicaltrials.gov/api/v2`) — public, free, no key required.
- **openFDA** (`api.fda.gov`) — public, free. An API key raises the rate limit but is not required
  at M1 ingest volumes; add `OPENFDA_API_KEY` to `config/donotshare` if throughput becomes an issue.
- **FDA Orange Book / Purple Book** — quarterly ZIP / CSV downloads from fda.gov, no auth.
- **SEC DERA Financial Statement Data Sets** (spec §2.0, added v0.5) — quarterly ZIP archives at
  `https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets`; public, free.
  Sole basis for point-in-time, survivorship-free universe construction — derive archive URLs from
  the landing page rather than hardcoding, per the spec's own caution that the path has moved
  before.
- **IBKR** (already integrated, `src.data.downloader.ibkr_downloader`) — assigned by spec §2.0.5 as
  the source for daily adjusted prices, corporate actions, options IV, and short interest/borrow
  for *currently listed* names. Not a new integration; mind pacing limits at universe scale (~700
  tickers) — see `Tasks.md`. **Open question, not yet live-verified:** IBKR's documented behavior is
  that historical TRADES bars are split-adjusted server-side with no raw-print option, which would
  make it unsuitable as the `p22_price_daily` raw-storage source (spec §2.0.7, added v0.6) even though
  it's fine for options IV / short interest / borrow. See `Tasks.md` Known Issues.
- **Delisted-ticker historical price vendor** (spec §2.0.6) — **not yet selected**, and narrower
  than earlier drafts assumed: EDGAR (fundamentals) + IBKR (live-name prices) cover everything
  except historical prices for companies that no longer trade, which `E[return | deal]` labeling
  (M6/M7) needs. Spec recommends FMP Starter (~$15/mo) as the cheapest validated path. See
  `Tasks.md`.

## System Requirements
- Postgres 15+ (shared instance; new `p22_*` tables only, no new database).
- Local disk for raw-zone cache under `DATA_CACHE_DIR/p22/raw/` — grows with daily SEC/CT.gov/FDA
  snapshots; no cleanup job exists yet (tracked in `Tasks.md`).

## Security Requirements
- No API keys required for M1 sources (EDGAR, CT.gov, openFDA free tier, Orange/Purple Book).
- Vendor API keys (once selected, §2.4) go in `config/donotshare/donotshare.py`, never committed.
- Postgres credentials via existing `DB_URL` env/config — no new credential surface.

## Performance Requirements
- SEC EDGAR: hard 10 requests/second ceiling, enforced by `EdgarDownloader`'s existing limiter.
- CT.gov / openFDA: rate-limited via `src.data.utils.rate_limiting.RateLimiter`, one instance per
  host, conservative defaults (documented in `ingest/rate_limits.py`) until each API's actual
  published limits are confirmed against the free-tier docs.
- Daily ingest jobs (SEC, CT.gov, openFDA) must complete within the scheduler's job timeout;
  quarterly jobs (Orange Book, Purple Book) are large one-shot downloads and get an explicit
  `timeout_seconds` override in `jobs/register_jobs.py`, matching the pattern P20 uses for its
  full-universe jobs.

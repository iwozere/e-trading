"""
P22 Biotech M&A — Central configuration.

All pipeline-wide constants live here. Import this module from every P22 file
that needs a path, source URL, or feature flag. See docs/implementation-plan.md
for why each of these choices was made.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"

DATA_CACHE_PATH = Path(DATA_CACHE_DIR)

# ---------------------------------------------------------------------------
# Raw zone (spec §1, §7.3)
# ---------------------------------------------------------------------------
# Partitioned source/date/, content-addressed, immutable, gzipped JSON.
RAW_ZONE_ROOT = DATA_CACHE_PATH / "p22" / "raw"

# ---------------------------------------------------------------------------
# SEC EDGAR (spec §2.1)
# ---------------------------------------------------------------------------
# EdgarDownloader's own default already satisfies the "{AppName} {contact-email}"
# User-Agent requirement — reused as-is rather than duplicating the string here.
EDGAR_USER_AGENT = "e-trading-research akossyrev@gmail.com"
EDGAR_RATE_LIMIT_RPS: int = 10  # hard SEC cap; EdgarDownloader enforces this internally

# ---------------------------------------------------------------------------
# SEC DERA Financial Statement Data Sets (spec §2.0, universe construction)
# ---------------------------------------------------------------------------
# Landing page from which quarterly archive URLs are derived — do NOT hardcode
# archive URLs directly, the path has moved before (spec §2.0.1 caution).
SEC_DERA_LANDING_PAGE = "https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets"
BIOTECH_SIC_CODES: list[str] = [
    "2833",  # Medicinal Chemicals & Botanical Products
    "2834",  # Pharmaceutical Preparations
    "2835",  # In Vitro & In Vivo Diagnostic Substances
    "2836",  # Biological Products (No Diagnostic Substances)
    "8731",  # Services — Commercial Physical & Biological Research
]
# Hand-curated acquirer universe (~25 companies, spec §2.0.4) — not screened.
ACQUIRERS_YAML = PROJECT_ROOT / "config" / "pipeline" / "p22_acquirers.yaml"

# ---------------------------------------------------------------------------
# ClinicalTrials.gov API v2 (spec §2.2)
# ---------------------------------------------------------------------------
CLINICALTRIALS_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
# The spec (§2.2) lists these as bare field names, but the live API only
# accepts bare names for a handful of top-level fields (verified 2026-08-30:
# NCTId and hasResults work bare; everything else 400s with "invalid field
# name" and needs its full protocolSection.<module>.<field> path). No flat
# "locationCountries" field exists in v2 — the nearest equivalent is the full
# `locations` array (each entry carries its own `country`), used here.
CLINICALTRIALS_FIELDS: list[str] = [
    "NCTId",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.statusModule.overallStatus",
    "protocolSection.designModule.phases",
    "protocolSection.designModule.studyType",
    "protocolSection.conditionsModule.conditions",
    "protocolSection.armsInterventionsModule.interventions",
    "protocolSection.sponsorCollaboratorsModule.leadSponsor",
    "protocolSection.sponsorCollaboratorsModule.collaborators",
    "protocolSection.outcomesModule.primaryOutcomes",
    "protocolSection.designModule.enrollmentInfo",
    "protocolSection.statusModule.startDateStruct",
    "protocolSection.statusModule.primaryCompletionDateStruct",
    "protocolSection.statusModule.completionDateStruct",
    "protocolSection.contactsLocationsModule.locations",
    "protocolSection.designModule.designInfo",
    "hasResults",
    "protocolSection.statusModule.lastUpdatePostDateStruct",
]
# Conservative default — CT.gov's public API does not publish a hard numeric
# rate limit as of this writing; keep this cautious until confirmed otherwise
# (see docs/Tasks.md).
CLINICALTRIALS_RATE_LIMIT_RPS: int = 5
# Undocumented internal endpoint backing CT.gov's own history-viewer UI — the
# only source for spec §2.2's "Critical" version-history requirement; no
# endpoint under the documented /api/v2 surface serves this. See
# clinicaltrials_client.py's module docstring and docs/Tasks.md.
CLINICALTRIALS_HISTORY_BASE_URL = "https://clinicaltrials.gov/api/int/studies"
# Separate, more conservative limiter for the history endpoint above — live
# production data (2026-09-02 first full run) showed it throttles far harder
# than the public /api/v2/studies endpoint: at the shared 5 rps limit, 1600 of
# 6532+1600≈8132 history requests got 429'd (~20%), and the resulting
# exponential-backoff sleeps (2/4/8s per retry) ate ~5390s of the 7200s
# budget — the run only covered 215/1705 companies before its timeout fired.
# 2 rps is an unverified starting guess, not a confirmed safe threshold (same
# "no published limit, tune from observed 429 rate" situation as FMP/openFDA
# elsewhere in this file) — see docs/Tasks.md.
CLINICALTRIALS_HISTORY_RATE_LIMIT_RPS: int = 2

# ---------------------------------------------------------------------------
# openFDA (spec §2.3)
# ---------------------------------------------------------------------------
OPENFDA_DRUGSFDA_URL = "https://api.fda.gov/drug/drugsfda.json"
# openFDA's published unauthenticated limits (as of this writing): 240 req/min,
# 120,000 req/day. Conservative default well under that; raise once an
# OPENFDA_API_KEY is configured (see docs/Tasks.md).
OPENFDA_RATE_LIMIT_RPS: int = 3
# Not yet in config/donotshare/donotshare.py (no key needed at M1 unauthenticated
# volumes) — read directly from the environment so adding it there later, or
# setting it ad hoc, both work without a code change here.
OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY")

# ---------------------------------------------------------------------------
# FDA Orange Book / Purple Book (spec §2.3)
# ---------------------------------------------------------------------------
# Orange Book quarterly full-database ZIP (products.txt, patent.txt, exclusivity.txt).
# Verified reachable 2026-08-30; FDA media IDs have moved before (same caution
# as the SEC DERA landing page below) — re-verify if this starts 404ing.
ORANGE_BOOK_ZIP_URL = "https://www.fda.gov/media/76860/download"
# Purple Book: no stable "latest" URL exists — the site publishes one dated
# CSV per month (e.g. ".../2026/purplebook-search-August-data-download.csv"),
# each a full current snapshot with that month's New/Updated rows flagged,
# not a diff. The download URL is derived from this listing page rather than
# guessed/hardcoded, mirroring the DERA landing-page approach in §2.0.1.
PURPLE_BOOK_DOWNLOADS_PAGE = "https://purplebooksearch.fda.gov/index.cfm?event=downloads"

# ---------------------------------------------------------------------------
# Market-data vendor (spec §2.4) — FMP selected 2026-08-31, see docs/Tasks.md
# item 1. Reuses src.data.downloader.fmp_data_downloader.FMPDataDownloader
# for API-key resolution; the constants below are for the NEW direct calls
# P22 makes that downloader doesn't already implement (full raw historical
# price JSON, name search for delisted tickers with no ticker on file) —
# see ingest/fmp_client.py.
# ---------------------------------------------------------------------------
# Set True once a real MarketDataProvider implementation is wired in behind
# ingest/vendor_market_data.py's Protocol.
VENDOR_MARKET_DATA_AVAILABLE: bool = False
# Conservative fallback lag when a vendor cannot supply first-publication
# timestamps (spec §2.4): known_from = period_end + this many days.
VENDOR_FUNDAMENTALS_LAG_DAYS: int = 45
VENDOR_PRICE_LAG_DAYS: int = 0
FMP_STABLE_URL = "https://financialmodelingprep.com/stable"
# Conservative default, well under Premium's published 750 req/min — not yet
# tuned against a real key's actual observed behavior (see docs/Tasks.md).
FMP_RATE_LIMIT_RPS: int = 5

# ---------------------------------------------------------------------------
# yfinance — ongoing/daily current-price ingest (spec §2.0.7), 2026-09-01.
# Free, no API key. Deliberately NOT used for historical backfill — see
# ingest/yfinance_client.py's docstring for the live-verified retroactive
# split-adjustment trap that disqualifies it from that role. FMP (above)
# remains the historical-backfill source; yfinance covers the narrow,
# ongoing "today's bar" role IBKR was originally slated for (spec §2.0.5),
# sidestepping IBKR's own unverified raw-vs-adjusted question (docs/Tasks.md
# item 6) and the need for a live TWS/Gateway connection.
# ---------------------------------------------------------------------------
YFINANCE_LOOKBACK_DAYS: int = 7  # narrow trailing window only — see ingest/yfinance_client.py
YFINANCE_REQUEST_DELAY_SECONDS: float = 0.3  # no official yfinance rate limit; considerate pacing

# ---------------------------------------------------------------------------
# Logging / results directory
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results" / "p22_biotech_ma"

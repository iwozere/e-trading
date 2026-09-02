"""
P22 — shared per-host rate limiters (spec §7.2: "single shared token-bucket
limiter per host").

SEC EDGAR is not included here — `EdgarDownloader` already enforces its own
10 rps cap internally. This module covers the hosts P22 talks to directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.utils.rate_limiting import RateLimiter
from src.ml.pipeline.p22_biotech_ma.config import (
    CLINICALTRIALS_HISTORY_RATE_LIMIT_RPS,
    CLINICALTRIALS_RATE_LIMIT_RPS,
    FMP_RATE_LIMIT_RPS,
    OPENFDA_RATE_LIMIT_RPS,
)

# One shared instance per host, reused across every request that host receives
# in a process's lifetime — do not construct a fresh RateLimiter per call.
clinicaltrials_limiter = RateLimiter(requests_per_second=CLINICALTRIALS_RATE_LIMIT_RPS)
# Separate limiter for CT.gov's internal history endpoint — live-verified
# 2026-09-02 to throttle far harder than the public studies endpoint; sharing
# one limiter between the two let history-endpoint 429s burn the request
# budget the sponsor-search calls also needed. See config.py's
# CLINICALTRIALS_HISTORY_RATE_LIMIT_RPS docstring.
clinicaltrials_history_limiter = RateLimiter(requests_per_second=CLINICALTRIALS_HISTORY_RATE_LIMIT_RPS)
openfda_limiter = RateLimiter(requests_per_second=OPENFDA_RATE_LIMIT_RPS)
# Orange Book / Purple Book are infrequent (quarterly), single large downloads —
# no per-request limiter needed, but keep a conservative one for retry backoff.
fda_book_limiter = RateLimiter(requests_per_second=2)
fmp_limiter = RateLimiter(requests_per_second=FMP_RATE_LIMIT_RPS)

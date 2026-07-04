"""
api/middleware.py — aiohttp middleware for the Telegram bot's HTTP API.

Write-path endpoints and the full /api/status endpoint require an X-API-Key header
matching TELEGRAM_API_KEY.  The lightweight /api/health probe (used by load-balancers
and container health-checks) is intentionally unauthenticated — it returns only
{"status": "ok"} or {"status": "error"} with no internal information.
"""

import hmac
import os

from aiohttp import web

# Paths that do NOT require authentication.
# /api/health   — minimal liveness probe; exposes no internal data.
# /api/test     — legacy alias kept for backward compatibility.
# NOTE: /api/status was intentionally removed from this list (P1-TG-3).
#       It now requires X-API-Key because it discloses adapter names,
#       user counts, and service internals.
_OPEN_PATHS = {"/api/health", "/api/test"}

_API_KEY: str = os.environ.get("TELEGRAM_API_KEY", "")

if not _API_KEY:
    import logging as _logging

    _logging.getLogger(__name__).error(
        "TELEGRAM_API_KEY is not set — all non-open API endpoints will be denied. "
        "Set the env var to enable authenticated access."
    )


@web.middleware
async def api_key_middleware(request: web.Request, handler):
    """
    Require a valid X-API-Key header for every endpoint outside _OPEN_PATHS.

    Fails *closed*: if TELEGRAM_API_KEY is not configured, all protected paths
    are denied rather than left open.  The comparison uses hmac.compare_digest
    to avoid timing side-channels.
    """
    if request.path in _OPEN_PATHS:
        return await handler(request)

    if not _API_KEY:
        return web.json_response(
            {"success": False, "error": "API key not configured"},
            status=503,
        )

    provided = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(provided, _API_KEY):
        return web.json_response(
            {"success": False, "error": "Unauthorized"},
            status=401,
        )

    return await handler(request)

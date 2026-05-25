"""
api/middleware.py — aiohttp middleware for the Telegram bot's HTTP API.

Write-path endpoints and the full /api/status endpoint require an X-API-Key header
matching TELEGRAM_API_KEY.  The lightweight /api/health probe (used by load-balancers
and container health-checks) is intentionally unauthenticated — it returns only
{"status": "ok"} or {"status": "error"} with no internal information.
"""
import os
from aiohttp import web

# Paths that do NOT require authentication.
# /api/health   — minimal liveness probe; exposes no internal data.
# /api/test     — legacy alias kept for backward compatibility.
# NOTE: /api/status was intentionally removed from this list (P1-TG-3).
#       It now requires X-API-Key because it discloses adapter names,
#       user counts, and service internals.
_OPEN_PATHS = {"/api/health", "/api/test"}


@web.middleware
async def api_key_middleware(request: web.Request, handler):
    """Require X-API-Key for mutating endpoints when TELEGRAM_API_KEY is set."""
    api_key = os.environ.get("TELEGRAM_API_KEY", "")
    if request.path not in _OPEN_PATHS and api_key:
        provided = request.headers.get("X-API-Key", "")
        if provided != api_key:
            return web.json_response(
                {"success": False, "error": "Unauthorized"},
                status=401,
            )
    return await handler(request)

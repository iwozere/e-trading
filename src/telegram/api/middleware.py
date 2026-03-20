"""
api/middleware.py — aiohttp middleware for the Telegram bot's HTTP API.

Write-path endpoints require an X-API-Key header matching TELEGRAM_API_KEY.
Read-only probes (/api/status, /api/test) are left unauthenticated.
"""
import os
from aiohttp import web

# Paths that do not require authentication
_OPEN_PATHS = {"/api/status", "/api/test"}


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

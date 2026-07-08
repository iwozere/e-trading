"""
telegram_bot.py — Entry point for the Alkotrader Telegram bot.

This file is intentionally thin. All business logic lives in:
  src/telegram/handlers/   — command handlers grouped by domain
  src/telegram/api/        — internal HTTP REST API
  src/telegram/lifecycle.py — service init and health checks
  src/telegram/screener/   — report, screener, schedule processing

Architecture notes
──────────────────
• Bot creation is deferred to main() so importing this module does not open
  a live TCP connection to the Telegram API (P2.1).
• Dispatcher is module-level because @dp.message() decorators must run at
  import time when the handler modules call register(dp).
• The X-API-Key middleware on the HTTP API is wired inside api/routes.py
  via create_api_app().
"""

import asyncio
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from aiogram import Bot, Dispatcher
from aiohttp import web

from config.donotshare.donotshare import TELEGRAM_BOT_TOKEN
from src.notification.logger import set_logging_context, setup_logger

_logger = setup_logger("telegram_screener_bot")

# ─── Dispatcher (module-level — needed for decorator registration) ────────────
dp = Dispatcher()

# ─── Register all command handler groups ─────────────────────────────────────
from src.telegram.handlers import account, admin, alerts, content, misc  # noqa: E402

account.register(dp)
alerts.register(dp)
admin.register(dp)
content.register(dp)
misc.register(dp)  # misc must be last — contains the catch-all @dp.message()


# ─── Entry point ─────────────────────────────────────────────────────────────


async def main() -> None:
    _logger.info("Starting bot initialisation…")
    if not TELEGRAM_BOT_TOKEN:
        _logger.error("TELEGRAM_BOT_TOKEN is not set!")
        return
    _logger.info("Bot token present, starting initialisation")

    # P2.1: Bot created here, not at import time.
    bot = Bot(token=TELEGRAM_BOT_TOKEN)

    set_logging_context("telegram_screener_bot")

    # Service layer
    from src.telegram.lifecycle import initialize_services

    try:
        ok = await initialize_services()
        if ok:
            _logger.info("Service layer initialised successfully")
        else:
            _logger.warning("Service layer init failed — bot running with limited functionality")
    except Exception as exc:
        _logger.warning("Service layer init error: %s — limited functionality", exc)

    # Heartbeat
    try:
        from src.common.heartbeat_manager import HeartbeatManager

        def _health_check():
            bot_ok = bot is not None and bool(TELEGRAM_BOT_TOKEN)
            if bot_ok:
                return {
                    "status": "HEALTHY",
                    "metadata": {
                        "bot_token_present": True,
                        "notification_system": "lazy_initialization",
                        "last_check": time.time(),
                    },
                }
            return {"status": "DOWN", "error_message": "Bot not properly initialised"}

        hb = HeartbeatManager(system="telegram_bot", interval_seconds=30)
        hb.set_health_check_function(_health_check)
        hb.start_heartbeat()
        _logger.info("Heartbeat manager started")
    except Exception:
        _logger.exception("Failed to initialise heartbeat manager:")

    # Queue processor (polls DB every 5 s for queued messages from scheduler etc.)
    try:
        from src.telegram.services.telegram_queue_processor import TelegramQueueProcessor

        queue_processor = TelegramQueueProcessor(bot=bot, poll_interval=5)
        await queue_processor.start()
        _logger.info("TelegramQueueProcessor started")
    except Exception:
        _logger.exception("Failed to start TelegramQueueProcessor:")

    # HTTP API
    try:
        from src.telegram.api.routes import create_api_app

        api_app = create_api_app()
        runner = web.AppRunner(api_app)
        await runner.setup()

        api_host = os.environ.get("TELEGRAM_API_HOST", "localhost")
        api_port = int(os.environ.get("TELEGRAM_API_PORT", "5004"))
        site = web.TCPSite(runner, api_host, api_port)
        await site.start()
        _logger.info("HTTP API server started on http://%s:%d", api_host, api_port)
        _logger.info("  POST /api/send_message | POST /api/broadcast | POST /api/notify | GET /api/status")
    except Exception:
        _logger.exception("Failed to start HTTP API server:")

    # Bot polling
    _logger.info("Starting Telegram bot polling…")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

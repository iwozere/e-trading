"""
api/routes.py — aiohttp HTTP API endpoints for the Telegram bot.

Provides internal REST endpoints used by the scheduler, admin scripts, and
any other service that needs to trigger Telegram messages programmatically.
Authentication is handled by api/middleware.py (X-API-Key).
"""

from aiohttp import web

from src.notification.logger import setup_logger
from src.notification.service.client import MessagePriority, MessageType
from src.telegram.lifecycle import get_notification_client, get_service_instances, perform_service_health_checks

_logger = setup_logger("telegram_screener_bot")


# ─── Route handlers ──────────────────────────────────────────────────────────


async def api_send_message(request: web.Request) -> web.Response:
    """Send a message to a specific user."""
    try:
        data = await request.json()
        user_id = data.get("user_id")
        message = data.get("message")
        title = data.get("title", "Alkotrader Notification")

        if not user_id or not message:
            return web.json_response({"success": False, "error": "Missing user_id or message"}, status=400)

        client = await get_notification_client()
        if not client:
            return web.json_response({"success": False, "error": "Notification service unavailable"}, status=503)

        success = await client.send_notification(
            notification_type=MessageType.INFO,
            title=title,
            message=message,
            priority=MessagePriority.NORMAL,
            channels=["telegram"],
            recipient_id=str(user_id),
        )
        return web.json_response(
            {
                "success": success,
                "message": "Message queued for delivery" if success else "Failed to queue message",
            }
        )
    except Exception as exc:
        _logger.exception("Error in api_send_message:")
        return web.json_response({"success": False, "error": str(exc)}, status=500)


async def api_broadcast(request: web.Request) -> web.Response:
    """Broadcast a message to all registered users."""
    try:
        data = await request.json()
        message = data.get("message")
        title = data.get("title", "Alkotrader Announcement")

        if not message:
            return web.json_response({"success": False, "error": "Missing message"}, status=400)

        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return web.json_response({"success": False, "error": "Service not available"}, status=503)

        users = telegram_svc.list_users()
        if not users:
            return web.json_response({"success": False, "error": "No registered users found"}, status=404)

        client = await get_notification_client()
        if not client:
            return web.json_response({"success": False, "error": "Notification service unavailable"}, status=503)

        success_count = 0
        for user in users:
            user_id = user.get("telegram_user_id", "")
            if user_id and str(user_id).isdigit():
                ok = await client.send_notification(
                    notification_type=MessageType.INFO,
                    title=title,
                    message=message,
                    priority=MessagePriority.NORMAL,
                    channels=["telegram"],
                    recipient_id=str(user_id),
                )
                if ok:
                    success_count += 1

        return web.json_response(
            {
                "success": True,
                "message": f"Broadcast queued for {success_count}/{len(users)} users",
                "success_count": success_count,
                "total_count": len(users),
            }
        )
    except Exception as exc:
        _logger.exception("Error in api_broadcast:")
        return web.json_response({"success": False, "error": str(exc)}, status=500)


async def api_notify(request: web.Request) -> web.Response:
    """Send a notification from the scheduler or another internal service."""
    try:
        data = await request.json()
        notification_type = data.get("notification_type", "INFO")
        title = data.get("title", "Alert Notification")
        message = data.get("message")
        priority = data.get("priority", "NORMAL")
        telegram_chat_id = data.get("telegram_chat_id")

        if not message:
            return web.json_response({"success": False, "error": "Missing message field"}, status=400)
        if not telegram_chat_id:
            return web.json_response({"success": False, "error": "Missing telegram_chat_id field"}, status=400)

        client = await get_notification_client()
        if not client:
            return web.json_response({"success": False, "error": "Notification service unavailable"}, status=503)

        success = await client.send_notification(
            notification_type=notification_type,
            title=title,
            message=message,
            priority=priority,
            channels=["telegram"],
            telegram_chat_id=int(telegram_chat_id),
            recipient_id=str(telegram_chat_id),
            data=data.get("data", {}),
        )
        return web.json_response(
            {
                "success": success,
                "message": "Notification queued" if success else "Failed to queue notification",
            }
        )
    except Exception as exc:
        _logger.exception("Error in api_notify:")
        return web.json_response({"success": False, "error": str(exc)}, status=500)


async def api_health(request: web.Request) -> web.Response:
    """Minimal liveness probe — no authentication required, no internal data exposed."""
    return web.json_response({"status": "ok"})


async def api_status(request: web.Request) -> web.Response:
    """Detailed service status — requires X-API-Key (P1-TG-3)."""
    try:
        from src.telegram.lifecycle import get_notification_client as _get_client

        client = await _get_client()
        stats = client.get_stats() if client else {}

        telegram_svc, indicator_svc = get_service_instances()
        user_count = len(telegram_svc.list_users()) if telegram_svc else 0
        healthy = await perform_service_health_checks()

        return web.json_response(
            {
                "success": True,
                "status": "healthy" if healthy else "degraded",
                "services": {
                    "telegram_service": {"initialized": telegram_svc is not None, "healthy": healthy},
                    "indicator_service": {
                        "initialized": indicator_svc is not None,
                        "healthy": healthy,
                        "adapters": list(indicator_svc.adapters.keys()) if indicator_svc else [],
                    },
                },
                "notification_stats": stats,
                "user_count": user_count,
                "queue_size": 0,
            }
        )
    except Exception as exc:
        _logger.exception("Error in api_status:")
        return web.json_response({"success": False, "status": "error", "error": str(exc)}, status=500)


# ─── Application factory ─────────────────────────────────────────────────────


def create_api_app() -> web.Application:
    """Create and configure the aiohttp application with the auth middleware."""
    from src.telegram.api.middleware import api_key_middleware

    app = web.Application(middlewares=[api_key_middleware])
    app.router.add_post("/api/send_message", api_send_message)
    app.router.add_post("/api/broadcast", api_broadcast)
    app.router.add_post("/api/notify", api_notify)
    app.router.add_get("/api/status", api_status)  # authenticated — returns service internals
    app.router.add_get("/api/health", api_health)  # unauthenticated — minimal liveness probe
    async def _test_handler(r: web.Request) -> web.StreamResponse:
        return web.json_response({"status": "ok", "message": "Bot API is working!"})

    app.router.add_get("/api/test", _test_handler)
    return app

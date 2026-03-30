"""
Resolve the single owner recipient for trading-bot notifications (Telegram + email).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def _resolve_notification_channels(
    config: Dict[str, Any],
    instance_id: str,
    *,
    log_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build channel list and recipient for the bot owner; no feature-flag gating."""
    name = log_name or instance_id
    notif_config = config.get("notifications", {})

    direct_email = notif_config.get("notify_email") or os.environ.get("TRADING_NOTIFY_EMAIL")
    direct_tg = notif_config.get("notify_telegram_chat_id")
    if direct_tg is None and os.environ.get("TRADING_NOTIFY_TELEGRAM_CHAT_ID"):
        direct_tg = os.environ.get("TRADING_NOTIFY_TELEGRAM_CHAT_ID")

    channels: List[str] = []
    email: Optional[str] = None
    telegram_user_id: Optional[str] = None

    if notif_config.get("email_enabled", True) and direct_email:
        channels.append("email")
        email = direct_email.strip()
    if notif_config.get("telegram_enabled", True) and direct_tg not in (None, ""):
        channels.append("telegram")
        telegram_user_id = str(direct_tg).strip()

    if channels:
        rid = config.get("user_id") or f"manifest-{instance_id}"
        return {
            "user_id": config.get("user_id"),
            "email": email,
            "telegram_user_id": telegram_user_id,
            "recipient_id": str(rid),
            "channels": channels,
        }

    user_id = config.get("user_id")
    if not user_id:
        _logger.debug(
            "No user_id or notify_email/notify_telegram_chat_id for %s, skipping notification",
            name,
        )
        return None

    from src.data.db.services.database_service import get_database_service
    from src.data.db.models.model_users import User, AuthIdentity
    from sqlalchemy import select

    db_service = get_database_service()
    with db_service.uow() as uow:
        user = uow.s.get(User, user_id)
        if not user:
            _logger.warning("User %s not found for bot %s", user_id, name)
            return None

        telegram_identity = uow.s.execute(
            select(AuthIdentity)
            .where(AuthIdentity.user_id == user_id)
            .where(AuthIdentity.provider == "telegram")
        ).scalar_one_or_none()

        channels_db: List[str] = []
        if notif_config.get("email_enabled") and user.email:
            channels_db.append("email")
        if notif_config.get("telegram_enabled") and telegram_identity:
            channels_db.append("telegram")

        if not channels_db:
            _logger.debug("No notification channels enabled for bot %s", name)
            return None

        return {
            "user_id": user_id,
            "email": user.email,
            "telegram_user_id": telegram_identity.external_id if telegram_identity else None,
            "recipient_id": str(user_id),
            "channels": channels_db,
        }


def get_trading_bot_notification_recipient(
    config: Dict[str, Any],
    instance_id: str,
    *,
    purpose: str = "any",
    log_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Resolve the bot owner's notification targets.

    Args:
        config: Full or partial bot manifest (must include ``notifications`` and optional ``user_id``).
        instance_id: Stable bot instance id (for manifest-only recipients).
        purpose: ``any`` | ``trade_buy`` | ``trade_sell`` | ``error`` | ``bot_event``
        log_name: Optional friendly name for logs.
    """
    notif_config = config.get("notifications", {})

    if purpose == "any":
        if not (
            notif_config.get("position_opened")
            or notif_config.get("position_closed")
            or notif_config.get("error_notifications")
        ):
            _logger.debug("Notifications disabled for bot %s", log_name or instance_id)
            return None
    elif purpose == "trade_buy":
        if not notif_config.get("position_opened"):
            return None
    elif purpose == "trade_sell":
        if not notif_config.get("position_closed"):
            return None
    elif purpose == "error":
        if not notif_config.get("error_notifications"):
            return None
    elif purpose == "bot_event":
        if not notif_config.get("bot_events", True):
            return None
    else:
        _logger.warning("Unknown notification purpose %r for bot %s", purpose, log_name or instance_id)
        return None

    return _resolve_notification_channels(config, instance_id, log_name=log_name)

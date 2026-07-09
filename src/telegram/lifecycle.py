"""
lifecycle.py — Service initialisation, health checks, and shared accessor helpers.

All mutable state is stored in `src.telegram.screener.business_logic` via
set_service_instances / get_service_instances so that command handlers in
every sub-module can reach the same live objects without circular imports.
"""

import os
from typing import Any, Tuple

from src.indicators.service import IndicatorService
from src.notification.logger import setup_logger
from src.notification.service.client import NotificationServiceClient

_logger = setup_logger("telegram_screener_bot")

# ─── Notification client (lazy) ──────────────────────────────────────────────

_notification_client: NotificationServiceClient | None = None


async def get_notification_client() -> NotificationServiceClient | None:
    """Lazily initialise the notification service client."""
    global _notification_client

    if _notification_client is None:
        try:
            service_url = os.getenv("NOTIFICATION_SERVICE_URL", "database://localhost")
            _notification_client = NotificationServiceClient(
                service_url=service_url,
                timeout=30,
                max_retries=3,
            )
            _logger.info("Notification service client initialised for heavy commands")
        except Exception as exc:
            _logger.warning("Could not initialise notification service client: %s", exc)
            _notification_client = None

    return _notification_client


# ─── Service accessors ───────────────────────────────────────────────────────

# ─── Service instances (internal state) ───────────────────────────────────────

_telegram_service_instance = None
_indicator_service_instance = None


def set_service_instances(telegram_service, indicator_service):
    """Inject service instances for global access."""
    global _telegram_service_instance, _indicator_service_instance
    _telegram_service_instance = telegram_service
    _indicator_service_instance = indicator_service
    _logger.info("Service instances set in lifecycle layer.")


def get_service_instances() -> Tuple:
    """Return (telegram_service, indicator_service) for global access."""
    return _telegram_service_instance, _indicator_service_instance


# ─── Initialisation ──────────────────────────────────────────────────────────


async def initialize_services() -> bool:
    """
    Create and wire all service dependencies into the business logic layer.

    Returns True when at least the TelegramService is available; False on
    critical failure.
    """
    telegram_service_instance: Any = None
    indicator_service_instance: Any = None

    try:
        _logger.info("Initialising service layer…")

        # TelegramService
        try:
            from src.data.db.services.telegram_service import (
                telegram_service as telegram_service_instance,
            )

            required_methods = ["get_user_status", "set_user_limit"]
            for method in required_methods:
                if not hasattr(telegram_service_instance, method):
                    raise RuntimeError(f"TelegramService missing method: {method}")
            _logger.info("TelegramService initialised and validated")
        except Exception:
            _logger.exception("Failed to initialise TelegramService:")
            return False

        # IndicatorService (optional — some commands don't need it)
        try:
            indicator_service_instance = IndicatorService()
            if not hasattr(indicator_service_instance, "compute_for_ticker"):
                raise RuntimeError("IndicatorService missing method: compute_for_ticker")
            if hasattr(indicator_service_instance, "adapters") and not indicator_service_instance.adapters:
                _logger.warning("IndicatorService has no adapters — some commands may be limited")
            _logger.info("IndicatorService initialised and validated")
        except Exception:
            _logger.exception("Failed to initialise IndicatorService:")
            indicator_service_instance = None
            _logger.warning("Continuing without IndicatorService")

        # Inject into lifecycle layer
        try:
            set_service_instances(telegram_service_instance, indicator_service_instance)
            _logger.info("Service instances injected into lifecycle layer")
        except Exception:
            _logger.exception("Failed to inject services into lifecycle:")
            return False

        # Health checks
        try:
            healthy = await perform_service_health_checks()
            if healthy:
                _logger.info("All service health checks passed")
                return True
            _logger.error("Service health checks failed — limited functionality")
            if telegram_service_instance is not None:
                _logger.info("TelegramService available — continuing with limited functionality")
                return True
            return False
        except Exception as exc:
            _logger.error("Error during health checks: %s", exc)
            return telegram_service_instance is not None

    except Exception as exc:
        _logger.exception("Unexpected error during service initialisation: %s", exc)
        return False


# ─── Health checks ───────────────────────────────────────────────────────────


async def perform_service_health_checks() -> bool:
    """Run all service health checks. Returns True only if all pass."""
    try:
        _logger.info("Performing service health checks...")
        if not await check_telegram_service_health():
            _logger.error("TelegramService health check failed")
            return False
        if not await check_indicator_service_health():
            _logger.error("IndicatorService health check failed")
            return False
        _logger.info("All service health checks passed")
        return True
    except Exception:
        _logger.exception("Error during service health checks:")
        return False


async def check_telegram_service_health() -> bool:
    """Probe TelegramService with lightweight DB reads."""
    try:
        svc, _ = get_service_instances()
        if svc is None:
            return False
        svc.get_setting("health_check_test")
        svc.get_user_status("health_check_test_user")
        _logger.debug("TelegramService health check OK")
        return True
    except Exception:
        _logger.exception("TelegramService health check failed:")
        return False


async def check_indicator_service_health() -> bool:
    """Verify IndicatorService adapters and metadata are available."""
    try:
        _, indicator_svc = get_service_instances()
        if indicator_svc is None:
            return False
        if not hasattr(indicator_svc, "adapters"):
            _logger.error("IndicatorService missing adapters attribute")
            return False
        required_adapters = ["ta-lib", "pandas-ta", "fundamentals"]
        for name in required_adapters:
            if name not in indicator_svc.adapters:
                _logger.error("IndicatorService missing adapter: %s", name)
                return False
        from src.indicators.registry import INDICATOR_META

        if not INDICATOR_META:
            _logger.error("No indicator metadata available")
            return False
        _logger.debug("IndicatorService health check OK")
        return True
    except Exception:
        _logger.exception("IndicatorService health check failed:")
        return False

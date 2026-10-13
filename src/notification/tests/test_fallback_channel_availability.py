"""
Tests for FallbackManager channel availability decisions.

A channel whose first health check has not completed yet (UNKNOWN) must be usable;
otherwise every delivery claimed right after service startup fails outright.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.service.fallback_manager import FallbackManager  # noqa: E402
from src.notification.service.health_monitor import ChannelHealthStatus, HealthStatus  # noqa: E402


def _manager_with_status(status: HealthStatus, is_enabled: bool = True) -> FallbackManager:
    """
    Build a FallbackManager whose health monitor reports the given status for every channel.

    Args:
        status: Overall health status to report.
        is_enabled: Whether the channel is enabled.

    Returns:
        FallbackManager instance.
    """
    health_monitor = Mock()
    health_monitor.get_channel_status.return_value = ChannelHealthStatus(
        channel="email", overall_status=status, last_updated=datetime.now(timezone.utc), is_enabled=is_enabled
    )
    return FallbackManager(health_monitor)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (HealthStatus.HEALTHY, True),
        (HealthStatus.DEGRADED, True),
        (HealthStatus.UNKNOWN, True),
        (HealthStatus.CRITICAL, False),
    ],
)
def test_channel_availability_by_health_status(status: HealthStatus, expected: bool) -> None:
    """HEALTHY, DEGRADED and not-yet-checked (UNKNOWN) channels are available; CRITICAL is not."""
    manager = _manager_with_status(status)

    assert asyncio.run(manager._is_channel_available("email")) is expected


def test_disabled_channel_unavailable_even_if_unknown() -> None:
    """A disabled channel stays unavailable regardless of health status."""
    manager = _manager_with_status(HealthStatus.UNKNOWN, is_enabled=False)

    assert asyncio.run(manager._is_channel_available("email")) is False

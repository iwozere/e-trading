"""
Regression tests for NotificationRepository's optimized repository wiring.

NotificationRepository(use_optimized=True) swaps in the Optimized* repositories. When
those were standalone classes, methods added to the standard repositories (e.g.
DeliveryStatusRepository.claim_pending_deliveries) were missing from the optimized
ones, and every delivery poll raised AttributeError in production.
"""

from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.repos.repo_notification import (  # noqa: E402
    DeliveryStatusRepository,
    MessageRepository,
    NotificationRepository,
    RateLimitRepository,
)


def _public_methods(cls: type) -> set[str]:
    """
    Return the names of public callables defined on a class.

    Args:
        cls: Class to inspect.

    Returns:
        Set of public method names.
    """
    return {name for name in dir(cls) if not name.startswith("_") and callable(getattr(cls, name))}


@pytest.mark.parametrize(
    ("attr", "base_cls"),
    [
        ("messages", MessageRepository),
        ("delivery_status", DeliveryStatusRepository),
        ("rate_limits", RateLimitRepository),
    ],
)
def test_optimized_repositories_expose_full_standard_interface(attr: str, base_cls: type) -> None:
    """Each optimized repository must provide every public method of its standard counterpart."""
    repo = NotificationRepository(Mock(), use_optimized=True)
    component = getattr(repo, attr)

    assert isinstance(component, base_cls)
    missing = {name for name in _public_methods(base_cls) if not callable(getattr(component, name, None))}
    assert not missing, f"{type(component).__name__} is missing {sorted(missing)}"


def test_claim_pending_deliveries_delegates_with_optimized_repos(monkeypatch: pytest.MonkeyPatch) -> None:
    """claim_pending_deliveries must reach the standard implementation when optimized repos are used."""
    claimed = [{"delivery_status_id": 1, "channel": "telegram"}]
    calls: list[tuple[list[str], dict]] = []

    def _fake_claim(_self: DeliveryStatusRepository, channels: list[str], **kwargs: object) -> list[dict]:
        calls.append((channels, kwargs))
        return claimed

    monkeypatch.setattr(DeliveryStatusRepository, "claim_pending_deliveries", _fake_claim)

    repo = NotificationRepository(Mock(), use_optimized=True)

    assert repo.claim_pending_deliveries(channels=["telegram"], limit=10) == claimed
    assert calls == [(["telegram"], {"limit": 10})]

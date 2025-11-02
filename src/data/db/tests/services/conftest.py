"""
Service-layer test fixtures.

Uses the same database setup as repository tests but adds service-specific fixtures.
"""
from __future__ import annotations
import sys
import pathlib
import pytest
from unittest.mock import Mock, MagicMock
from typing import Generator
from sqlalchemy.orm import Session

# Ensure repository root is on sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import repository test fixtures to reuse database setup
pytest_plugins = ["src.data.db.tests.repos.conftest"]

from src.data.db.services.database_service import DatabaseService, ReposBundle
from src.data.db.repos.repo_jobs import JobsRepository
from src.data.db.repos.repo_users import UsersRepo, VerificationRepo
from src.data.db.repos.repo_notification import NotificationRepository
from src.data.db.repos.repo_short_squeeze import ShortSqueezeRepo
from src.data.db.repos.repo_system_health import SystemHealthRepository

# Import with same aliases as database_service.py
from src.data.db.repos.repo_telegram import (
    FeedbackRepo as TelegramFeedbackRepo,
    CommandAuditRepo as TelegramCommandAuditRepo,
    BroadcastRepo as TelegramBroadcastRepo,
    SettingsRepo as TelegramSettingsRepo,
)

from src.data.db.repos.repo_webui import (
    AuditRepo as WebUIAuditRepo,
    SnapshotRepo as WebUISnapshotRepo,
    StrategyTemplateRepo as WebUIStrategyTemplateRepo,
    SystemConfigRepo as WebUISystemConfigRepo,
)

from src.data.db.repos.repo_trading import (
    BotsRepo as TradingBotsRepo,
    TradesRepo as TradingTradesRepo,
    PositionsRepo as TradingPositionsRepo,
    MetricsRepo as TradingMetricsRepo,
)


@pytest.fixture
def mock_database_service(db_session: Session) -> Mock:
    """
    Create a mock DatabaseService that returns a repos bundle.

    This fixture provides a mock database service that can be injected into
    service classes during testing. It uses the real db_session for actual
    database operations.
    """
    mock_db = Mock(spec=DatabaseService)

    # Create repos bundle with real repositories using the test session
    # Match the exact structure from database_service.py ReposBundle
    repos = ReposBundle(
        s=db_session,
        users=UsersRepo(db_session),
        telegram_verification=VerificationRepo(db_session),
        jobs=JobsRepository(db_session),
        notifications=NotificationRepository(db_session),
        system_health=SystemHealthRepository(db_session),
        telegram_feedback=TelegramFeedbackRepo(db_session),
        telegram_audit=TelegramCommandAuditRepo(db_session),
        telegram_broadcast=TelegramBroadcastRepo(db_session),
        telegram_settings=TelegramSettingsRepo(db_session),
        webui_audit=WebUIAuditRepo(db_session),
        webui_snapshots=WebUISnapshotRepo(db_session),
        webui_templates=WebUIStrategyTemplateRepo(db_session),
        webui_config=WebUISystemConfigRepo(db_session),
        bots=TradingBotsRepo(db_session),
        trades=TradingTradesRepo(db_session),
        positions=TradingPositionsRepo(db_session),
        metrics=TradingMetricsRepo(db_session),
        short_squeeze=ShortSqueezeRepo(db_session),
    )

    # Mock the uow context manager to return the repos bundle
    mock_db.uow.return_value.__enter__ = Mock(return_value=repos)
    mock_db.uow.return_value.__exit__ = Mock(return_value=None)

    return mock_db


@pytest.fixture
def repos_bundle(db_session: Session) -> ReposBundle:
    """
    Create a ReposBundle with all repositories using the test session.

    This fixture provides direct access to repositories for service testing.
    """
    return ReposBundle(
        s=db_session,
        users=UsersRepo(db_session),
        telegram_verification=VerificationRepo(db_session),
        jobs=JobsRepository(db_session),
        notifications=NotificationRepository(db_session),
        system_health=SystemHealthRepository(db_session),
        telegram_feedback=TelegramFeedbackRepo(db_session),
        telegram_audit=TelegramCommandAuditRepo(db_session),
        telegram_broadcast=TelegramBroadcastRepo(db_session),
        telegram_settings=TelegramSettingsRepo(db_session),
        webui_audit=WebUIAuditRepo(db_session),
        webui_snapshots=WebUISnapshotRepo(db_session),
        webui_templates=WebUIStrategyTemplateRepo(db_session),
        webui_config=WebUISystemConfigRepo(db_session),
        bots=TradingBotsRepo(db_session),
        trades=TradingTradesRepo(db_session),
        positions=TradingPositionsRepo(db_session),
        metrics=TradingMetricsRepo(db_session),
        short_squeeze=ShortSqueezeRepo(db_session),
    )


@pytest.fixture
def jobs_repo(db_session: Session) -> JobsRepository:
    """Provide a JobsRepository for testing."""
    return JobsRepository(db_session)


@pytest.fixture
def users_repo(db_session: Session) -> UsersRepo:
    """Provide a UsersRepo for testing."""
    return UsersRepo(db_session)


@pytest.fixture
def notification_repo(db_session: Session) -> NotificationRepository:
    """Provide a NotificationRepository for testing."""
    return NotificationRepository(db_session)


@pytest.fixture
def short_squeeze_repo(db_session: Session) -> ShortSqueezeRepo:
    """Provide a ShortSqueezeRepo for testing."""
    return ShortSqueezeRepo(db_session)


@pytest.fixture
def bots_repo(db_session: Session) -> TradingBotsRepo:
    """Provide a TradingBotsRepo for testing."""
    return TradingBotsRepo(db_session)

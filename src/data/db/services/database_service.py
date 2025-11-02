from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from sqlalchemy.orm import Session

# Engine / session factory
from src.data.db.core import database as core_db
SessionLocal = core_db.SessionLocal
engine = core_db.engine

# Model Bases — used by init_databases()
from src.data.db.models.model_users import Base as UsersBase
from src.data.db.models.model_telegram import Base as TelegramBase
from src.data.db.models.model_trading import Base as TradingBase
from src.data.db.models.model_webui import Base as WebUIBase
from src.data.db.models.model_jobs import Base as JobsBase
from src.data.db.models.model_notification import Base as NotificationBase
from src.data.db.models.model_system_health import Base as SystemHealthBase
from src.data.db.models.model_short_squeeze import Base as ShortSqueezeBase

# Repositories
# NOTE: every repo must accept a sqlalchemy.orm.Session in __init__
from src.data.db.repos.repo_users import UsersRepo, VerificationRepo

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

from src.data.db.repos.repo_jobs import JobsRepository
from src.data.db.repos.repo_notification import NotificationRepository
from src.data.db.repos.repo_system_health import SystemHealthRepository
from src.data.db.repos.repo_short_squeeze import ShortSqueezeRepo


# ------------------------------- UoW bundle ----------------------------------

@dataclass
class ReposBundle:
    """A single-session bundle of all repos. Constructed per UoW."""
    s: Session

    # Users
    users: UsersRepo
    telegram_verification: VerificationRepo

    # Jobs (replaces telegram alerts/schedules)
    jobs: JobsRepository

    # Notifications
    notifications: NotificationRepository

    # System Health
    system_health: SystemHealthRepository

    # Telegram
    telegram_feedback: TelegramFeedbackRepo
    telegram_audit: TelegramCommandAuditRepo
    telegram_broadcast: TelegramBroadcastRepo
    telegram_settings: TelegramSettingsRepo

    # WebUI
    webui_audit: WebUIAuditRepo
    webui_snapshots: WebUISnapshotRepo
    webui_templates: WebUIStrategyTemplateRepo
    webui_config: WebUISystemConfigRepo

    # Trading
    bots: TradingBotsRepo
    trades: TradingTradesRepo
    positions: TradingPositionsRepo
    metrics: TradingMetricsRepo

    # Short Squeeze Pipeline
    short_squeeze: ShortSqueezeRepo


# ----------------------------- Database service ------------------------------

class DatabaseService:
    """Provides Unit-of-Work sessions and database initialization."""

    def __init__(self) -> None:
        # Don't capture SessionLocal/engine here; tests monkeypatch them.
        pass

    def init_databases(self) -> None:
        """Create all tables for every model base (idempotent)."""
        eng = getattr(self, "engine", None) or engine
        for base in (UsersBase, TelegramBase, TradingBase, WebUIBase, JobsBase, NotificationBase, SystemHealthBase, ShortSqueezeBase):
            base.metadata.create_all(bind=eng)

    # for tests that call ds.get_database_service()
    def get_database_service(self):
        return self

    @contextmanager
    def uow(self) -> Iterator[ReposBundle]:
        """
        Open a new Session and yield a bundle of repos bound to that session.
        Commits on success; rolls back on error; always closes the session.
        """
        sf = getattr(self, "SessionLocal", None) or SessionLocal
        s: Session = sf()
        try:
            repos = ReposBundle(
                s=s,

                # Users
                users=UsersRepo(s),
                telegram_verification=VerificationRepo(s),

                # Jobs (replaces telegram alerts/schedules)
                jobs=JobsRepository(s),

                # Notifications
                notifications=NotificationRepository(s),

                # System Health
                system_health=SystemHealthRepository(s),

                # Telegram
                telegram_feedback=TelegramFeedbackRepo(s),
                telegram_audit=TelegramCommandAuditRepo(s),
                telegram_broadcast=TelegramBroadcastRepo(s),
                telegram_settings=TelegramSettingsRepo(s),

                # WebUI
                webui_audit=WebUIAuditRepo(s),
                webui_snapshots=WebUISnapshotRepo(s),
                webui_templates=WebUIStrategyTemplateRepo(s),
                webui_config=WebUISystemConfigRepo(s),

                # Trading
                bots=TradingBotsRepo(s),
                trades=TradingTradesRepo(s),
                positions=TradingPositionsRepo(s),
                metrics=TradingMetricsRepo(s),

                # Short Squeeze Pipeline
                short_squeeze=ShortSqueezeRepo(s),
            )
            yield repos
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()


# ------------------------- Singleton-style accessors -------------------------
# Make a module-level singleton that tests import as `ds`
# ------------------------- Singleton-style accessors -------------------------
_db_service_singleton: DatabaseService | None = None

def get_database_service() -> DatabaseService:
    """Global accessor used by services (webui_service, trading_service, telegram_service)."""
    global _db_service_singleton
    if _db_service_singleton is None:
        _db_service_singleton = DatabaseService()
    return _db_service_singleton

# Make a module-level singleton that tests import as `ds`
database_service = get_database_service()


def init_databases() -> None:
    """Convenience to match existing service init_db() calls."""
    get_database_service().init_databases()


# ------------------------------ Deprecated APIs ------------------------------
# If you had helpers like get_telegram_repo() / get_webui_repo() / get_trading_repo(),
# remove them. They encouraged ad-hoc sessions and made multi-repo transactions non-atomic.
# Use:
#   with get_database_service().uow() as r:
#       r.telegram_alerts.create(...)
#       r.telegram_audit.log(...)
# which guarantees a single transaction and consistent session.



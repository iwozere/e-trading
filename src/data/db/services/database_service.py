from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from sqlalchemy.orm import Session

# Engine / session factory
from src.data.db.core import database as core_db
SessionLocal = core_db.SessionLocal

# Shared metadata — all model Bases inherit from the same DeclarativeBase
# so a single create_all() call covers every table.
from src.data.db.core.base import Base as _SharedBase

# Model Bases — kept for backward-compat imports by callers; all point to the
# same underlying metadata as _SharedBase.
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
from src.data.db.repos.repo_kestrel import KestrelRepo


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

    # P20 Kestrel Pipeline
    kestrel: KestrelRepo


# ----------------------------- Database service ------------------------------

class DatabaseService:
    """Provides Unit-of-Work sessions and database initialization."""

    def __init__(self) -> None:
        # Don't capture SessionLocal/engine here; tests monkeypatch them.
        pass

    def init_databases(self) -> None:
        """Create all tables (idempotent).

        All model Bases share a single ``MetaData`` instance (via the common
        ``DeclarativeBase`` in ``core.base``), so one ``create_all`` call is
        sufficient to create every table across every model module.
        """
        eng = getattr(self, "engine", None) or core_db.get_engine()
        _SharedBase.metadata.create_all(bind=eng)

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

                # P20 Kestrel Pipeline
                kestrel=KestrelRepo(s),
            )
            yield repos
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()


# ------------------------- Singleton-style accessors -------------------------

_db_service_singleton: DatabaseService | None = None


def get_database_service() -> DatabaseService:
    """Global accessor used by services (webui_service, trading_service, telegram_service)."""
    global _db_service_singleton
    if _db_service_singleton is None:
        _db_service_singleton = DatabaseService()
    return _db_service_singleton


def init_databases() -> None:
    """Convenience to match existing service init_db() calls."""
    get_database_service().init_databases()


# ---------------------------------------------------------------------------
# Lazy module-level ``database_service`` attribute
# ---------------------------------------------------------------------------
# ``database_service = get_database_service()`` at module scope would create
# the singleton on every *import*, even in code paths that never use the DB.
# Using module ``__getattr__`` defers construction until the name is first
# accessed, while remaining fully transparent to ``from ... import`` callers.
def __getattr__(name: str) -> DatabaseService:
    if name == "database_service":
        return get_database_service()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")



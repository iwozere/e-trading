"""
BaseDBService: Provides UoW pattern and error handling for DB services.
"""

from contextvars import ContextVar
from functools import wraps
from typing import Callable, TypeVar

from src.data.db.services.database_service import ReposBundle, get_database_service
from src.notification.logger import setup_logger

T = TypeVar("T")

# Per-context (thread / asyncio task) variable holding the current ReposBundle.
# Each concurrent caller gets its own value; no instance attribute is needed.
_current_repos: ContextVar[ReposBundle | None] = ContextVar("_current_repos", default=None)


def with_uow(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to wrap methods with a UoW context.

    Uses a ``ContextVar`` so that concurrent callers (threads, asyncio tasks)
    each maintain their own session without interfering with each other.
    Nested calls reuse the already-open session from the current context.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> T:
        # If already inside a UoW for this context, reuse it (no nesting).
        if _current_repos.get() is not None:
            return func(self, *args, **kwargs)

        # Open a new UoW and store it in the current context.
        with self._db.uow() as repos:
            token = _current_repos.set(repos)
            try:
                return func(self, *args, **kwargs)
            finally:
                _current_repos.reset(token)

    return wrapper


def handle_db_error(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to handle database errors consistently.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> T:
        try:
            return func(self, *args, **kwargs)
        except Exception:
            self._logger.exception("Database error in %s", func.__name__)
            raise

    return wrapper


class BaseDBService:
    """Base class for all database services with UoW pattern."""

    def __init__(self, db_service=None):
        self._db = db_service or get_database_service()
        self._logger = setup_logger(self.__class__.__name__)

    @property
    def repos(self) -> ReposBundle:
        """Access the current context's UoW repositories."""
        repos = _current_repos.get()
        if repos is None:
            raise RuntimeError(
                "Repositories not available. This property can only be accessed from methods decorated with @with_uow."
            )
        return repos

    @property
    def uow(self) -> ReposBundle:
        """Alias for self.repos (backward compatibility)."""
        return self.repos

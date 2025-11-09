"""
BaseDBService: Provides UoW pattern and error handling for DB services.
"""
from typing import Callable, TypeVar
from functools import wraps
from src.data.db.services.database_service import get_database_service, ReposBundle
from src.notification.logger import setup_logger
import threading

T = TypeVar('T')

# Thread-local storage for current UoW repos
_thread_local = threading.local()

def with_uow(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to wrap methods with UoW context.
    Automatically handles session management and transactions.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> T:
        with self._db.uow() as repos:
            # Store repos in thread-local storage so self.uow can access it
            old_repos = getattr(_thread_local, 'repos', None)
            _thread_local.repos = repos
            try:
                return func(self, *args, **kwargs)
            finally:
                # Restore previous repos (for nested calls)
                _thread_local.repos = old_repos
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
            self._logger.exception(f"Database error in {func.__name__}")
            raise
    return wrapper

class BaseDBService:
    """Base class for all database services with UoW pattern."""
    def __init__(self, db_service=None):
        self._db = db_service or get_database_service()
        self._logger = setup_logger(self.__class__.__name__)

    @property
    def uow(self) -> ReposBundle:
        """
        Get the current Unit of Work (repos bundle) from thread-local storage.
        This property is only available within methods decorated with @with_uow.
        """
        repos = getattr(_thread_local, 'repos', None)
        if repos is None:
            raise RuntimeError(
                "UoW not available. This property can only be accessed from methods "
                "decorated with @with_uow."
            )
        return repos
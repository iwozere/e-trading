"""
BaseDBService: Provides UoW pattern and error handling for DB services.
"""
from typing import Callable, TypeVar
from functools import wraps
from src.data.db.services.database_service import get_database_service, ReposBundle
from src.notification.logger import setup_logger

T = TypeVar('T')

def with_uow(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to wrap methods with UoW context.
    Automatically handles session management and transactions.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> T:
        # If already in a UoW context, just call the function
        if hasattr(self, '_repos') and self._repos is not None:
            return func(self, *args, **kwargs)

        # Create a new UoW context
        with self._db.uow() as repos:
            # Store repos in the instance
            old_repos = getattr(self, '_repos', None)
            self._repos = repos

            try:
                return func(self, *args, **kwargs)
            finally:
                # Restore previous repos (for nested calls)
                self._repos = old_repos
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
        self._repos = None  # Will be set by the UoW context

    @property
    def repos(self):
        """Access the current UoW's repositories."""
        if self._repos is None:
            raise RuntimeError(
                "Repositories not available. This property can only be accessed from methods "
                "decorated with @with_uow."
            )
        return self._repos

    @property
    def uow(self):
        """Alias for self.repos for backward compatibility."""
        return self.repos
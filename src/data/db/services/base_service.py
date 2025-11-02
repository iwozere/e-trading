"""
BaseDBService: Provides UoW pattern and error handling for DB services.
"""
from typing import Callable, TypeVar, Any
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
        with self._db.uow() as repos:
            return func(self, repos, *args, **kwargs)
    return wrapper

def handle_db_error(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to handle database errors consistently.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> T:
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            self._logger.exception(f"Database error in {func.__name__}")
            raise
    return wrapper

class BaseDBService:
    """Base class for all database services with UoW pattern."""
    def __init__(self, db_service=None):
        self._db = db_service or get_database_service()
        self._logger = setup_logger(self.__class__.__name__)

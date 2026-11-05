"""
JSON type handling for SQLAlchemy models.
"""

import json
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON, Text, TypeDecorator

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class JsonText(TypeDecorator):
    """
    JSON stored in a TEXT column (legacy columns such as job_schedules.state_json).

    Writes dicts/lists as JSON text (strings are stored as given, assumed to be JSON already)
    and always reads back the parsed value, so callers see the same shape as a JSONB column.
    """

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None or isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, default=str)

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None or not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except ValueError:
            _logger.warning("JsonText column holds invalid JSON (%d chars); returning {}", len(value))
            return {}


class JsonType(TypeDecorator):
    """
    Platform-independent JSON type with PostgreSQL-specific JSONB support.
    """

    impl = JSON
    cache_ok = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_jsonb = None

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            self._is_jsonb = True
            return dialect.type_descriptor(JSONB())
        self._is_jsonb = False
        return dialect.type_descriptor(JSON())

    def __repr__(self):
        return "JSONB" if self._is_jsonb else "JSON"

    def __eq__(self, other):
        if isinstance(other, (JSONB, JsonType)):
            return True
        return super().__eq__(other)

    def __hash__(self):
        """Make the type hashable for SQLAlchemy's type caching"""
        # Hash based on class since all instances are functionally identical
        return hash("JsonType")

    def __str__(self):
        return "JSONB" if self._is_jsonb else "JSON"

    def __instancecheck__(self, instance):
        """Support isinstance checks for JSONB"""
        if isinstance(instance, JSONB):
            return True
        return isinstance(instance, self.__class__)

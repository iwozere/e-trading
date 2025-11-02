"""
JSON type handling for SQLAlchemy models.
"""

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON, TypeDecorator

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
        if dialect.name == 'postgresql':
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
        return hash('JsonType')

    def __str__(self):
        return "JSONB" if self._is_jsonb else "JSON"

    def __instancecheck__(self, instance):
        """Support isinstance checks for JSONB"""
        if isinstance(instance, JSONB):
            return True
        return isinstance(instance, self.__class__)
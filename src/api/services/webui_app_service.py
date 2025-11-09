"""
Web UI Application Service
-------------------------

Application service for web UI operations that orchestrates domain services.
This service provides the interface that the web UI needs while using proper
domain services underneath.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Generator
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.services import users_service, webui_service
from src.data.db.services.database_service import get_database_service
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class WebUIAppService:
    """
    Application service for web UI operations.

    This service provides the interface that the web UI expects
    while using proper domain services underneath.
    """

    def __init__(self):
        """Initialize the web UI application service."""
        pass

    # ---------- Database Session Management ----------

    def get_db_session(self) -> Generator:
        """
        Get database session for dependency injection.

        This replaces the old get_db() function from database.py
        """
        db_service = get_database_service()
        with db_service.uow() as session:
            yield session.session

    # ---------- Database Initialization ----------

    def init_database(self) -> None:
        """
        Initialize database tables and create default users.

        This replaces the old init_database() function from database.py
        """
        try:
            # Create database directory if it doesn't exist
            db_dir = PROJECT_ROOT / "db"
            db_dir.mkdir(parents=True, exist_ok=True)

            # Use the database service to initialize tables
            db_service = get_database_service()
            db_service.init_databases()
            _logger.info("Database tables created successfully")

            # Create default users if they don't exist
            self.create_default_users()

        except Exception:
            _logger.exception("Failed to initialize database:")
            raise

    def create_default_users(self) -> None:
        """
        Create default admin and trader users if they don't exist.
        """
        try:
            # Check if any users exist
            existing_users = users_service.list_telegram_users_dto()
            if existing_users:
                _logger.info("Users already exist, skipping default user creation")
                return

            # Create default users using the users service
            default_users = [
                {
                    "email": "admin@trading-system.local",
                    "role": "admin",
                    "is_active": True
                },
                {
                    "email": "trader@trading-system.local",
                    "role": "trader",
                    "is_active": True
                },
                {
                    "email": "viewer@trading-system.local",
                    "role": "viewer",
                    "is_active": True
                }
            ]

            # Use users service to create users
            for user_data in default_users:
                users_service.create_user(
                    email=user_data["email"],
                    role=user_data["role"],
                    is_active=user_data["is_active"]
                )

            _logger.info("Created default users: admin, trader, viewer")

        except Exception:
            _logger.exception("Failed to create default users:")
            raise

    # ---------- User Management ----------

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email."""
        try:
            # Use users service to get user profile
            # Note: The current users service is Telegram-focused
            # This would need to be extended for web UI users
            return None
        except Exception:
            _logger.exception("Error getting user by email %s:", email)
            return None

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with username/email and password."""
        try:
            # Try to authenticate with email first
            if "@" in username:
                result = users_service.authenticate_user_by_email(username, password)
            else:
                # Try with username, first directly
                result = users_service.authenticate_user_by_username(username, password)

                # If not found, try with default domain
                if not result:
                    email = f"{username}@trading-system.local"
                    result = users_service.authenticate_user_by_email(email, password)

            return result

        except Exception:
            _logger.exception("Error authenticating user %s:", username)
            return None

        except Exception:
            _logger.exception("Error authenticating user %s:", username)
            return None

    # ---------- Audit Logging ----------

    def log_user_action(
        self,
        user_id: int,
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> int:
        """Log user action for audit purposes."""
        try:
            return webui_service.audit_log(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                details=details,
                ip_address=ip_address,
                user_agent=user_agent,
            )
        except Exception:
            _logger.exception("Error logging user action:")
            return 0

    # ---------- Configuration Management ----------

    def get_system_config(self, key: str) -> Optional[Dict[str, Any]]:
        """Get system configuration."""
        try:
            return webui_service.get_config(key)
        except Exception:
            _logger.exception("Error getting system config %s:", key)
            return None

    def set_system_config(self, key: str, value: Dict[str, Any], description: Optional[str] = None) -> Dict[str, Any]:
        """Set system configuration."""
        try:
            return webui_service.set_config(key, value, description)
        except Exception:
            _logger.exception("Error setting system config %s:", key)
            raise

    # ---------- Strategy Templates ----------

    def create_strategy_template(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create strategy template."""
        try:
            return webui_service.create_template(template_data)
        except Exception:
            _logger.exception("Error creating strategy template:")
            raise

    def get_strategy_templates_by_author(self, user_id: int) -> List[Dict[str, Any]]:
        """Get strategy templates by author."""
        try:
            return webui_service.get_templates_by_author(user_id)
        except Exception:
            _logger.exception("Error getting strategy templates for user %s:", user_id)
            return []

    # ---------- Performance Snapshots ----------

    def add_performance_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Add performance snapshot."""
        try:
            return webui_service.add_snapshot(snapshot)
        except Exception:
            _logger.exception("Error adding performance snapshot:")
            raise

    def get_latest_performance_snapshots(self, strategy_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get latest performance snapshots."""
        try:
            return webui_service.latest_snapshots(strategy_id, limit)
        except Exception:
            _logger.exception("Error getting performance snapshots for strategy %s:", strategy_id)
            return []


# Create a singleton instance for easy import
webui_app_service = WebUIAppService()
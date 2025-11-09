#!/usr/bin/env python3
"""
Test Configuration and Fixtures
------------------------------

Shared test configuration, fixtures, and utilities for the web UI backend tests.
Provides database setup, authentication mocking, and common test data.
"""

import pytest
from unittest.mock import Mock
from fastapi.testclient import TestClient
from pathlib import Path
import sys
from datetime import datetime, timezone

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.api.main import app
from src.data.db.models.model_users import User
from src.api.auth import create_access_token, get_current_user, require_admin, require_trader_or_admin


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_admin_user():
    """Create mock admin user for testing."""
    return User(
        id=1,
        email="admin@test.com",
        role="admin",
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        last_login=datetime.now(timezone.utc)
    )


@pytest.fixture
def mock_trader_user():
    """Create mock trader user for testing."""
    return User(
        id=2,
        email="trader@test.com",
        role="trader",
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        last_login=datetime.now(timezone.utc)
    )


@pytest.fixture
def mock_viewer_user():
    """Create mock viewer user for testing."""
    return User(
        id=3,
        email="viewer@test.com",
        role="viewer",
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        last_login=datetime.now(timezone.utc)
    )


@pytest.fixture
def mock_inactive_user():
    """Create mock inactive user for testing."""
    return User(
        id=4,
        email="inactive@test.com",
        role="trader",
        is_active=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        last_login=datetime.now(timezone.utc)
    )


@pytest.fixture
def admin_token(mock_admin_user):
    """Create JWT token for admin user."""
    return create_access_token(
        data={"sub": str(mock_admin_user.id), "username": "admin"}
    )


@pytest.fixture
def trader_token(mock_trader_user):
    """Create JWT token for trader user."""
    return create_access_token(
        data={"sub": str(mock_trader_user.id), "username": "trader"}
    )


@pytest.fixture
def viewer_token(mock_viewer_user):
    """Create JWT token for viewer user."""
    return create_access_token(
        data={"sub": str(mock_viewer_user.id), "username": "viewer"}
    )


@pytest.fixture
def auth_headers_admin(admin_token):
    """Create authorization headers for admin user."""
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture
def auth_headers_trader(trader_token):
    """Create authorization headers for trader user."""
    return {"Authorization": f"Bearer {trader_token}"}


@pytest.fixture
def auth_headers_viewer(viewer_token):
    """Create authorization headers for viewer user."""
    return {"Authorization": f"Bearer {viewer_token}"}


@pytest.fixture
def mock_database_service():
    """Mock database service for testing."""
    mock_service = Mock()
    mock_uow = Mock()
    mock_session = Mock()

    # Setup the context manager chain
    mock_service.uow.return_value.__enter__.return_value = mock_uow
    mock_service.uow.return_value.__exit__.return_value = None
    mock_uow.s = mock_session

    return mock_service


@pytest.fixture
def mock_strategy_manager():
    """Mock strategy manager for testing."""
    mock_manager = Mock()
    mock_manager.get_all_strategies_status.return_value = []
    mock_manager.create_strategy.return_value = {"message": "Strategy created", "strategy_id": "test-strategy"}
    mock_manager.get_strategy_status.return_value = None
    mock_manager.update_strategy.return_value = {"message": "Strategy updated", "strategy_id": "test-strategy"}
    mock_manager.delete_strategy.return_value = {"message": "Strategy deleted", "strategy_id": "test-strategy"}
    mock_manager.start_strategy.return_value = {"message": "Strategy started", "strategy_id": "test-strategy"}
    mock_manager.stop_strategy.return_value = {"message": "Strategy stopped", "strategy_id": "test-strategy"}
    mock_manager.restart_strategy.return_value = {"message": "Strategy restarted", "strategy_id": "test-strategy"}
    return mock_manager


@pytest.fixture
def mock_monitoring_service():
    """Mock monitoring service for testing."""
    mock_service = Mock()
    mock_service.get_comprehensive_metrics.return_value = {
        'cpu': {'usage_percent': 25.5},
        'memory': {'usage_percent': 60.2},
        'temperature': {'average_celsius': 45.0},
        'disk': {
            'partitions': {
                '/': {'usage_percent': 75.0}
            }
        }
    }
    mock_service.get_alerts.return_value = []
    mock_service.acknowledge_alert.return_value = True
    mock_service.get_performance_history.return_value = []
    return mock_service


@pytest.fixture
def mock_telegram_service():
    """Mock telegram service for testing."""
    mock_service = Mock()

    # Mock user data
    mock_service.list_users.return_value = [
        {
            'telegram_user_id': '123456789',
            'email': 'test@example.com',
            'verified': True,
            'approved': True,
            'language': 'en',
            'is_admin': False,
            'max_alerts': 5,
            'max_schedules': 5,
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc)
        }
    ]

    # Mock alert data
    mock_service.list_active_alerts.return_value = [
        {
            'id': 1,
            'user_id': '123456789',
            'ticker': 'BTCUSDT',
            'price': 50000.0,
            'condition': 'above',
            'active': True,
            'email': False,
            'created': datetime.now(timezone.utc)
        }
    ]

    # Mock audit data
    mock_service.get_all_command_audit.return_value = [
        {
            'id': 1,
            'telegram_user_id': '123456789',
            'command': '/start',
            'full_message': '/start',
            'is_registered_user': True,
            'user_email': 'test@example.com',
            'success': True,
            'error_message': None,
            'response_time_ms': 150,
            'created': datetime.now(timezone.utc)
        }
    ]

    return mock_service


@pytest.fixture
def sample_strategy_config():
    """Sample strategy configuration for testing."""
    return {
        "id": "test-strategy",
        "name": "Test Strategy",
        "enabled": True,
        "symbol": "BTCUSDT",
        "broker": {
            "type": "paper",
            "balance": 10000
        },
        "strategy": {
            "type": "sma_crossover",
            "fast_period": 10,
            "slow_period": 20
        },
        "data": {
            "provider": "binance",
            "timeframe": "1h"
        },
        "trading": {
            "position_size": 0.1,
            "max_positions": 1
        },
        "risk_management": {
            "stop_loss": 0.02,
            "take_profit": 0.04
        },
        "notifications": {
            "enabled": True,
            "channels": ["telegram"]
        }
    }


@pytest.fixture
def sample_strategy_status():
    """Sample strategy status for testing."""
    return {
        "instance_id": "test-strategy-1",
        "name": "Test Strategy",
        "status": "running",
        "uptime_seconds": 3600.0,
        "error_count": 0,
        "last_error": None,
        "broker_type": "paper",
        "trading_mode": "paper",
        "symbol": "BTCUSDT",
        "strategy_type": "sma_crossover"
    }


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    """Clear FastAPI dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def mock_webui_app_service():
    """Mock WebUI app service for testing."""
    mock_service = Mock()
    mock_service.init_database.return_value = None
    mock_service.log_user_action.return_value = 1
    mock_service.authenticate_user.return_value = None
    return mock_service


class MockUser:
    """Mock user class for testing authentication."""

    def __init__(self, id, email, role, is_active=True):
        self.id = id
        self.email = email
        self.role = role
        self.is_active = is_active
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.last_login = datetime.now(timezone.utc)

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None
        }

    def verify_password(self, password):
        """Mock password verification - accepts any password for testing."""
        return True


def override_get_current_user(user):
    """Helper to override get_current_user dependency."""
    def _get_current_user():
        return user
    return _get_current_user


def override_require_admin(user):
    """Helper to override require_admin dependency."""
    def _require_admin():
        if user.role != "admin":
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. Required roles: admin"
            )
        return user
    return _require_admin


def override_require_trader_or_admin(user):
    """Helper to override require_trader_or_admin dependency."""
    def _require_trader_or_admin():
        if user.role not in ["trader", "admin"]:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. Required roles: trader, admin"
            )
        return user
    return _require_trader_or_admin


@pytest.fixture
def authenticated_client_admin(client, mock_admin_user):
    """Client with admin authentication."""
    app.dependency_overrides[get_current_user] = override_get_current_user(mock_admin_user)
    app.dependency_overrides[require_admin] = override_require_admin(mock_admin_user)
    app.dependency_overrides[require_trader_or_admin] = override_require_trader_or_admin(mock_admin_user)
    return client


@pytest.fixture
def authenticated_client_trader(client, mock_trader_user):
    """Client with trader authentication."""
    app.dependency_overrides[get_current_user] = override_get_current_user(mock_trader_user)
    app.dependency_overrides[require_trader_or_admin] = override_require_trader_or_admin(mock_trader_user)
    return client


@pytest.fixture
def authenticated_client_viewer(client, mock_viewer_user):
    """Client with viewer authentication."""
    app.dependency_overrides[get_current_user] = override_get_current_user(mock_viewer_user)
    return client


# Test data constants
TEST_TELEGRAM_USER_ID = "123456789"
TEST_EMAIL = "test@example.com"
TEST_STRATEGY_ID = "test-strategy"
TEST_ALERT_ID = 1
TEST_SCHEDULE_ID = 1

# Mock responses
MOCK_TELEGRAM_USERS = [
    {
        'telegram_user_id': '123456789',
        'email': 'test1@example.com',
        'verified': True,
        'approved': True,
        'language': 'en',
        'is_admin': False,
        'max_alerts': 5,
        'max_schedules': 5
    },
    {
        'telegram_user_id': '987654321',
        'email': 'test2@example.com',
        'verified': True,
        'approved': False,
        'language': 'en',
        'is_admin': False,
        'max_alerts': 5,
        'max_schedules': 5
    },
    {
        'telegram_user_id': '555666777',
        'email': 'test3@example.com',
        'verified': False,
        'approved': False,
        'language': 'en',
        'is_admin': False,
        'max_alerts': 5,
        'max_schedules': 5
    }
]

MOCK_TELEGRAM_ALERTS = [
    {
        'id': 1,
        'user_id': '123456789',
        'ticker': 'BTCUSDT',
        'price': 50000.0,
        'condition': 'above',
        'active': True,
        'email': False,
        'alert_type': 'price',
        'timeframe': '15m',
        'created': datetime.now(timezone.utc)
    },
    {
        'id': 2,
        'user_id': '123456789',
        'ticker': 'ETHUSDT',
        'price': 3000.0,
        'condition': 'below',
        'active': False,
        'email': True,
        'alert_type': 'price',
        'timeframe': '1h',
        'created': datetime.now(timezone.utc)
    }
]

MOCK_AUDIT_LOGS = [
    {
        'id': 1,
        'telegram_user_id': '123456789',
        'command': '/start',
        'full_message': '/start',
        'is_registered_user': True,
        'user_email': 'test@example.com',
        'success': True,
        'error_message': None,
        'response_time_ms': 150,
        'created': datetime.now(timezone.utc)
    },
    {
        'id': 2,
        'telegram_user_id': '123456789',
        'command': '/report',
        'full_message': '/report BTCUSDT',
        'is_registered_user': True,
        'user_email': 'test@example.com',
        'success': True,
        'error_message': None,
        'response_time_ms': 2500,
        'created': datetime.now(timezone.utc)
    }
]

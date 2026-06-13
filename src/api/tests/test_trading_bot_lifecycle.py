"""
Integration tests for the trading-bot lifecycle endpoints (§3.5 roadmap).

Coverage:
- GET /api/trading/bots/{bot_id} — enriched response includes live_state and active_positions
- PUT /api/trading/bots/{bot_id}/status action=start — DB write + direct manager call
- PUT /api/trading/bots/{bot_id}/status action=stop — DB write + direct manager call
- PUT /api/trading/bots/{bot_id}/status action=restart — DB write + direct manager call
- Fallback: manager unavailable → response still reports queued-via-polling
- Guard: starting an already-running bot returns 400
- Guard: stopping an already-stopped bot returns 400
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.api.auth import get_current_user
from src.api.main import app
from src.data.db.models.model_users import User


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_user(user_id: int = 1, role: str = "admin") -> User:
    return User(id=user_id, email="trader@test.com", role=role, is_active=True)


def _make_bot(
    bot_id: str = "42",
    user_id: int = 1,
    bot_status: str = "stopped",
) -> Dict[str, Any]:
    return {
        "id": bot_id,
        "user_id": user_id,
        "status": bot_status,
        "config": {
            "name": "Paper BTC Bot",
            "broker": {"type": "backtrader", "trading_mode": "paper", "cash": 10000.0},
            "strategy": {"type": "CustomStrategy", "parameters": {}},
        },
        "description": "Paper BTC Bot",
        "started_at": None,
        "last_heartbeat": None,
        "error_count": 0,
        "current_balance": None,
        "total_pnl": None,
        "extra_metadata": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": None,
    }


def _make_live_state(bot_id: str = "42") -> Dict[str, Any]:
    return {
        "instance_id": bot_id,
        "name": "Paper BTC Bot",
        "status": "running",
        "uptime_seconds": 120.0,
        "error_count": 0,
        "last_error": None,
        "broker_type": "backtrader",
        "trading_mode": "paper",
        "symbol": "BTCUSDT",
        "strategy_type": "CustomStrategy",
        "last_heartbeat": datetime.now(timezone.utc).isoformat(),
        "heartbeat_age_seconds": 5.0,
        "is_healthy": True,
    }


def _make_manager_mock(
    start_ok: bool = True,
    stop_ok: bool = True,
    restart_ok: bool = True,
    live_state: Optional[Dict[str, Any]] = None,
) -> MagicMock:
    mgr = MagicMock()
    mgr.strategy_instances = {}
    mgr.instance_service = MagicMock()
    mgr._db_bot_to_strategy_config = Mock(return_value={"id": "42", "name": "Paper BTC Bot"})
    mgr.start_strategy = AsyncMock(return_value=start_ok)
    mgr.stop_strategy = AsyncMock(return_value=stop_ok)
    mgr.restart_strategy = AsyncMock(return_value=restart_ok)
    mgr.get_strategy_status = Mock(return_value=live_state)
    return mgr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def admin_user():
    return _make_user()


@pytest.fixture()
def authed_client(client, admin_user):
    """Client with auth dependency bypassed."""
    app.dependency_overrides[get_current_user] = lambda: admin_user
    return client


# ---------------------------------------------------------------------------
# GET /api/trading/bots/{bot_id} — enriched response
# ---------------------------------------------------------------------------

class TestGetBotEnrichedResponse:
    def test_includes_live_state_when_manager_available(self, authed_client):
        bot = _make_bot(bot_status="running")
        live = _make_live_state()
        mgr = _make_manager_mock(live_state=live)

        with (
            patch("src.api.trading_bot_routes.trading_service") as ts,
        ):
            ts.get_bot_by_id.return_value = bot
            ts.get_open_positions.return_value = []
            app.state.strategy_manager = mgr

            resp = authed_client.get("/api/trading/bots/42")

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["bot"]["live_state"]["status"] == "running"
        assert data["bot"]["live_state"]["is_healthy"] is True

    def test_live_state_none_when_manager_unavailable(self, authed_client):
        bot = _make_bot()

        with patch("src.api.trading_bot_routes.trading_service") as ts:
            ts.get_bot_by_id.return_value = bot
            ts.get_open_positions.return_value = []
            app.state.strategy_manager = None

            resp = authed_client.get("/api/trading/bots/42")

        assert resp.status_code == 200
        assert resp.json()["bot"]["live_state"] is None

    def test_includes_active_positions(self, authed_client):
        bot = _make_bot(bot_status="running")
        positions = [
            {
                "id": "pos-1",
                "bot_id": "42",
                "symbol": "BTCUSDT",
                "direction": "long",
                "qty_open": 0.1,
                "avg_price": 50000.0,
                "status": "open",
            }
        ]

        with patch("src.api.trading_bot_routes.trading_service") as ts:
            ts.get_bot_by_id.return_value = bot
            ts.get_open_positions.return_value = positions
            app.state.strategy_manager = None

            resp = authed_client.get("/api/trading/bots/42")

        assert resp.status_code == 200
        assert len(resp.json()["bot"]["active_positions"]) == 1
        assert resp.json()["bot"]["active_positions"][0]["symbol"] == "BTCUSDT"

    def test_returns_404_for_missing_bot(self, authed_client):
        with patch("src.api.trading_bot_routes.trading_service") as ts:
            ts.get_bot_by_id.return_value = None
            app.state.strategy_manager = None

            resp = authed_client.get("/api/trading/bots/99")

        assert resp.status_code == 404

    def test_returns_403_for_wrong_owner(self, authed_client):
        bot = _make_bot(user_id=999)  # different owner

        with patch("src.api.trading_bot_routes.trading_service") as ts:
            ts.get_bot_by_id.return_value = bot
            app.state.strategy_manager = None

            resp = authed_client.get("/api/trading/bots/42")

        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# PUT /api/trading/bots/{bot_id}/status — start
# ---------------------------------------------------------------------------

class TestStartBot:
    def test_start_calls_db_and_manager(self, authed_client):
        bot = _make_bot(bot_status="stopped")
        mgr = _make_manager_mock(start_ok=True)

        with patch("src.api.trading_bot_routes.trading_service") as ts:
            ts.get_bot_by_id.return_value = bot
            ts.update_bot_status.return_value = True
            app.state.strategy_manager = mgr

            resp = authed_client.put(
                "/api/trading/bots/42/status",
                json={"action": "start"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "confirmed live" in data["message"]
        ts.update_bot_status.assert_called_once_with("42", "starting")
        mgr.start_strategy.assert_awaited_once_with("42")

    def test_start_queued_when_manager_unavailable(self, authed_client):
        bot = _make_bot(bot_status="stopped")

        with patch("src.api.trading_bot_routes.trading_service") as ts:
            ts.get_bot_by_id.return_value = bot
            ts.update_bot_status.return_value = True
            app.state.strategy_manager = None

            resp = authed_client.put(
                "/api/trading/bots/42/status",
                json={"action": "start"},
            )

        assert resp.status_code == 200
        assert "queued via DB polling" in resp.json()["message"]

    def test_start_already_running_returns_400(self, authed_client):
        bot = _make_bot(bot_status="running")

        with patch("src.api.trading_bot_routes.trading_service") as ts:
            ts.get_bot_by_id.return_value = bot
            app.state.strategy_manager = None

            resp = authed_client.put(
                "/api/trading/bots/42/status",
                json={"action": "start"},
            )

        assert resp.status_code == 400
        assert "already running" in resp.json()["detail"]

    def test_start_creates_instance_if_not_in_manager(self, authed_client):
        bot = _make_bot(bot_status="stopped")
        mgr = _make_manager_mock(start_ok=True)
        # instance is NOT already registered
        mgr.strategy_instances = {}

        with patch("src.api.trading_bot_routes.trading_service") as ts:
            ts.get_bot_by_id.return_value = bot
            ts.update_bot_status.return_value = True
            app.state.strategy_manager = mgr

            authed_client.put("/api/trading/bots/42/status", json={"action": "start"})

        mgr.instance_service.create_instance.assert_called_once()


# ---------------------------------------------------------------------------
# PUT /api/trading/bots/{bot_id}/status — stop
# ---------------------------------------------------------------------------

class TestStopBot:
    def test_stop_calls_db_and_manager(self, authed_client):
        bot = _make_bot(bot_status="running")
        mgr = _make_manager_mock(stop_ok=True)

        with patch("src.api.trading_bot_routes.trading_service") as ts:
            ts.get_bot_by_id.return_value = bot
            ts.update_bot_status.return_value = True
            app.state.strategy_manager = mgr

            resp = authed_client.put(
                "/api/trading/bots/42/status",
                json={"action": "stop"},
            )

        assert resp.status_code == 200
        assert "confirmed live" in resp.json()["message"]
        ts.update_bot_status.assert_called_once_with("42", "stopping")
        mgr.stop_strategy.assert_awaited_once_with("42")

    def test_stop_already_stopped_returns_400(self, authed_client):
        bot = _make_bot(bot_status="stopped")

        with patch("src.api.trading_bot_routes.trading_service") as ts:
            ts.get_bot_by_id.return_value = bot
            app.state.strategy_manager = None

            resp = authed_client.put(
                "/api/trading/bots/42/status",
                json={"action": "stop"},
            )

        assert resp.status_code == 400
        assert "already stopped" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# PUT /api/trading/bots/{bot_id}/status — restart
# ---------------------------------------------------------------------------

class TestRestartBot:
    def test_restart_calls_db_and_manager(self, authed_client):
        bot = _make_bot(bot_status="running")
        mgr = _make_manager_mock(restart_ok=True)

        with patch("src.api.trading_bot_routes.trading_service") as ts:
            ts.get_bot_by_id.return_value = bot
            ts.update_bot_status.return_value = True
            app.state.strategy_manager = mgr

            resp = authed_client.put(
                "/api/trading/bots/42/status",
                json={"action": "restart"},
            )

        assert resp.status_code == 200
        assert "confirmed live" in resp.json()["message"]
        ts.update_bot_status.assert_called_once_with("42", "restarting")
        mgr.restart_strategy.assert_awaited_once_with("42")

    def test_restart_queued_when_manager_fails(self, authed_client):
        bot = _make_bot(bot_status="running")
        mgr = _make_manager_mock()
        mgr.restart_strategy = AsyncMock(side_effect=RuntimeError("cerebro busy"))

        with patch("src.api.trading_bot_routes.trading_service") as ts:
            ts.get_bot_by_id.return_value = bot
            ts.update_bot_status.return_value = True
            app.state.strategy_manager = mgr

            resp = authed_client.put(
                "/api/trading/bots/42/status",
                json={"action": "restart"},
            )

        # Should still succeed — falls back to DB polling
        assert resp.status_code == 200
        assert "queued via DB polling" in resp.json()["message"]

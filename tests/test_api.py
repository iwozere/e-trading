"""
Integration tests for the Trading Bot Management REST API (api.py).

Prerequisites:
- The API server must be running (e.g., python src/management/api/api.py)
- The test strategy (e.g., rsi_bb_volume) and its bot class must be available in src/trading
- API credentials must match those used by the server (default: admin/changeme)

Environment variables:
- API_URL: Base URL of the API (default: http://localhost:5000)
- API_LOGIN: Username for HTTP Basic Auth (default: admin)
- API_PASSWORD: Password for HTTP Basic Auth (default: changeme)

How to run:
    pytest tests/test_api.py

What is tested:
- Start and stop a bot, check status and trades
- Backtest endpoint
- Error handling for missing bot_id
- Authentication required for protected endpoints
"""

import os

import pytest
import requests
from requests.auth import HTTPBasicAuth

API_URL = os.environ.get("API_URL", "http://localhost:5000")
API_LOGIN = os.environ.get("API_LOGIN", "admin")
API_PASSWORD = os.environ.get("API_PASSWORD", "changeme")


@pytest.mark.integration
def test_start_and_stop_bot():
    # Start bot
    payload = {
        "strategy": "rsi_bb_volume",
        "id": "apitestbot",
        "config": {"trading_pair": "BTCUSDT", "initial_balance": 1000.0},
    }
    r = requests.post(
        f"{API_URL}/start_bot",
        json=payload,
        auth=HTTPBasicAuth(API_LOGIN, API_PASSWORD),
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "bot_id" in data
    bot_id = data["bot_id"]

    # Status
    r = requests.get(f"{API_URL}/status", auth=HTTPBasicAuth(API_LOGIN, API_PASSWORD))
    assert r.status_code == 200
    assert bot_id in r.json()

    # Trades (should be empty or a list)
    r = requests.get(
        f"{API_URL}/trades",
        params={"bot_id": bot_id},
        auth=HTTPBasicAuth(API_LOGIN, API_PASSWORD),
    )
    assert r.status_code == 200
    assert isinstance(r.json(), list)

    # Stop bot
    r = requests.post(
        f"{API_URL}/stop_bot",
        json={"bot_id": bot_id},
        auth=HTTPBasicAuth(API_LOGIN, API_PASSWORD),
    )
    assert r.status_code == 200
    assert "message" in r.json() or "status" in r.json()


@pytest.mark.integration
def test_backtest_stub():
    payload = {"strategy": "rsi_bb_volume", "ticker": "BTCUSDT", "tf": "1h"}
    r = requests.post(
        f"{API_URL}/backtest", json=payload, auth=HTTPBasicAuth(API_LOGIN, API_PASSWORD)
    )
    assert r.status_code == 200
    assert "message" in r.json()


@pytest.mark.integration
def test_missing_bot_id():
    r = requests.post(
        f"{API_URL}/stop_bot", json={}, auth=HTTPBasicAuth(API_LOGIN, API_PASSWORD)
    )
    assert r.status_code == 400
    assert "error" in r.json()


@pytest.mark.integration
def test_auth_required():
    r = requests.get(f"{API_URL}/status")
    assert r.status_code == 401

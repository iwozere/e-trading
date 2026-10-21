"""
Tests for monitoring routes (services status).
"""

from unittest.mock import MagicMock, patch

from src.api import monitoring_routes
from src.data.db.models.model_jobs import Schedule


def test_services_status_reports_ib_gateway_via_port(authenticated_client_viewer):
    """IB Gateway status comes from a port probe; there is no systemd wrapper unit."""
    client = authenticated_client_viewer

    def fake_port_open(host, port, timeout=1.0):
        del host, timeout  # signature must match _port_open; only port matters here
        # Paper (4002) is up; Live (4001) is not started.
        return port == 4002

    mock_monitor = MagicMock()
    mock_monitor.check_service_status.return_value = (True, "active")
    mock_monitor.check_service_logs.return_value = []

    with (
        patch("src.api.monitoring_routes._port_open", side_effect=fake_port_open),
        patch("src.api.monitoring_routes.ServiceMonitor", return_value=mock_monitor),
        patch.object(monitoring_routes._telegram_health, "get_health_status", return_value={"status": "healthy"}),
        patch.object(
            monitoring_routes._notification_health,
            "get_health_status",
            return_value={"status": "healthy", "channels": {"owned": ["email"]}},
        ),
    ):
        response = client.get("/api/monitoring/services")

    assert response.status_code == 200
    data = response.json()
    statuses = {s["display_name"]: s["status"] for s in data["services"]}

    assert statuses["IB Gateway (Paper)"] == "active"
    assert statuses["IB Gateway (Live)"] == "inactive"
    # No systemd wrapper unit exists for the Docker gateways; nothing should reference it.
    assert all(s["name"] != "ibgateway-docker.service" for s in data["services"])


def _schedules() -> list:
    return [
        Schedule(id=28, user_id=1, name="P15 Options Daily", job_type="data_processing", target="t", cron="0 6 * * 1-5"),
        Schedule(id=29, user_id=1, name="P17 Penny Stock Screener Daily", job_type="data_processing", target="t", cron="0 8 * * 1-5"),
    ]


def test_match_schedule_id_accepts_integer_job_id():
    # Prod's job_schedule_runs.job_id column is an integer, not the model's String.
    assert monitoring_routes._match_schedule_id({}, 29, _schedules()) == 29
    assert monitoring_routes._match_schedule_id({}, "28", _schedules()) == 28
    assert monitoring_routes._match_schedule_id({}, 999, _schedules()) is None
    assert monitoring_routes._match_schedule_id({}, None, _schedules()) is None


def test_match_schedule_id_prefers_snapshot_then_legacy_name():
    schedules = _schedules()
    assert monitoring_routes._match_schedule_id({"schedule_id": "29"}, 28, schedules) == 29
    assert monitoring_routes._match_schedule_id({"schedule_name": "P15 Options Daily"}, None, schedules) == 28
    assert monitoring_routes._match_schedule_id({}, "legacy-P15 Options Daily-run", schedules) == 28

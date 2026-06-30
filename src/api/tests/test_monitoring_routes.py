"""
Tests for monitoring routes (services status).
"""

from unittest.mock import MagicMock, patch

from src.api import monitoring_routes


def test_services_status_reports_ib_gateway_via_port(authenticated_client_viewer):
    """IB Gateway status comes from a port probe, not the oneshot systemd unit."""
    client = authenticated_client_viewer

    def fake_port_open(host, port, timeout=1.0):
        # Paper (4002) is up; Live (4001) is not started.
        return port == 4002

    mock_monitor = MagicMock()
    mock_monitor.check_service_status.return_value = (True, "active")
    mock_monitor.check_service_logs.return_value = []

    with patch('src.api.monitoring_routes._port_open', side_effect=fake_port_open), \
         patch('src.api.monitoring_routes.ServiceMonitor', return_value=mock_monitor), \
         patch.object(monitoring_routes._telegram_health, 'get_health_status',
                      return_value={"status": "healthy"}), \
         patch.object(monitoring_routes._notification_health, 'get_health_status',
                      return_value={"status": "healthy", "channels": {"owned": ["email"]}}):
        response = client.get("/api/monitoring/services")

    assert response.status_code == 200
    data = response.json()
    statuses = {s["display_name"]: s["status"] for s in data["services"]}

    assert statuses["IB Gateway (Paper)"] == "active"
    assert statuses["IB Gateway (Live)"] == "inactive"
    # The (always-inactive) Docker systemd unit must not be listed separately.
    assert all(s["name"] != "ibgateway-docker.service" for s in data["services"])

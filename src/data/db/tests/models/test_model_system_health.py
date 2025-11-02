from datetime import datetime, timezone

from src.data.db.models.model_system_health import SystemHealth, SystemHealthStatus


def test_system_health_record_and_helpers(db_session):
    sh = SystemHealth()
    sh.system = "test_service"
    sh.component = None
    sh.status = SystemHealthStatus.HEALTHY.value
    sh.checked_at = datetime.now(timezone.utc)

    db_session.add(sh)
    db_session.flush()

    fetched = SystemHealth.get_system_status(db_session, "test_service")
    assert fetched is not None
    assert fetched.system == "test_service"
    assert fetched.is_healthy is True

    # update to degraded
    fetched.update_health_status(SystemHealthStatus.DEGRADED, response_time_ms=250, error_message="timeout")
    assert fetched.is_healthy is False
    assert fetched.is_degraded is True or fetched.status == SystemHealthStatus.DEGRADED.value

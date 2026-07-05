"""
Monitoring API Routes
--------------------

Endpoints for service health and pipeline execution status.
"""

import socket
from datetime import UTC, datetime
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import get_current_user
from src.api.services.notification_health_service import NotificationHealthService
from src.api.services.telegram_health_service import TelegramHealthService
from src.data.db.models.model_jobs import Schedule, ScheduleRun
from src.data.db.models.model_users import User
from src.data.db.services.database_service import get_database_service
from src.notification.logger import setup_logger
from src.notification.service_monitor import SERVICES_TO_MONITOR, ServiceMonitor

_logger = setup_logger(__name__)

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

_telegram_health = TelegramHealthService()
_notification_health = NotificationHealthService()

_DISPLAY_NAMES = {
    "ibgateway-docker.service": "IB Gateway",
    "notification-bot.service": "Notification Bot",
    "scheduler.service": "Scheduler",
    "telegram-bot.service": "Telegram Bot",
    "trading.service": "Trading",
    "trading-webui.service": "Web UI",
    "trading-api.service": "API",
}

# The IB Gateway runs as Docker containers (ibgw-paper / ibgw-live) started by a
# oneshot systemd unit that exits after `docker compose up -d`. As a result
# `systemctl is-active ibgateway-docker.service` reports "inactive" even while
# the gateway is up. We therefore detect the gateways by probing their API
# ports directly, which reflects whether they are actually reachable.
_IB_GATEWAY_HOST = "127.0.0.1"
# (display name, API port). Paper auto-starts on boot; Live is manual-only and
# is expected to be inactive unless deliberately started.
_IB_GATEWAYS: List[Tuple[str, int]] = [
    ("IB Gateway (Paper)", 4002),
    ("IB Gateway (Live)", 4001),
]


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Return True if a TCP connection to host:port succeeds within timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@router.get("/services")
async def get_services_status(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Return live status for all monitored systemd services and notification channels.

    Service status degrades gracefully to 'unknown' when systemctl is unavailable
    (e.g., Windows development environments).
    """
    try:
        monitor = ServiceMonitor(admin_user_id=current_user.id)
        services: List[Dict[str, Any]] = []

        # IB Gateway runs in Docker; probe its API ports instead of the
        # (oneshot, always-inactive) systemd unit so the live state is accurate.
        for display_name, port in _IB_GATEWAYS:
            up = _port_open(_IB_GATEWAY_HOST, port)
            services.append(
                {
                    "name": f"ib-gateway-{port}",
                    "display_name": display_name,
                    "status": "active" if up else "inactive",
                    "has_errors": False,
                }
            )

        for svc in SERVICES_TO_MONITOR:
            # The Docker gateway unit is reported above via port probing.
            if svc == "ibgateway-docker.service":
                continue
            is_active, raw_status = monitor.check_service_status(svc)
            if raw_status == "error":
                status = "unknown"
            elif is_active:
                status = "active"
            else:
                status = "inactive"
            errors = monitor.check_service_logs(svc) if status == "active" else []
            services.append(
                {
                    "name": svc,
                    "display_name": _DISPLAY_NAMES.get(svc, svc),
                    "status": status,
                    "has_errors": bool(errors),
                }
            )

        try:
            tg = _telegram_health.get_health_status()
            tg_status = tg.get("status", "unknown")
        except Exception:
            tg_status = "unknown"

        try:
            notif = _notification_health.get_health_status()
            notif_status = notif.get("status", "unknown")
            owned = notif.get("channels", {}).get("owned", [])
        except Exception:
            notif_status = "unknown"
            owned = []

        channels: Dict[str, Any] = {"telegram": {"status": tg_status}}
        for ch in ["email", "sms"]:
            channels[ch] = {"status": notif_status if ch in owned else "unknown"}
        if not owned and notif_status != "unknown":
            channels["email"] = {"status": notif_status}

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "services": services,
            "channels": channels,
        }

    except Exception:
        _logger.exception("Error fetching services status:")
        raise HTTPException(status_code=500, detail="Failed to fetch services status")


@router.get("/pipelines")
async def get_pipelines_status(_current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Return all configured pipelines with their last run status and 10-run history.

    Matches ScheduleRun records to Schedule records via job_snapshot['schedule_id']
    or job_snapshot['schedule_name'], falling back to job_id substring matching.
    """
    try:
        db_service = get_database_service()
        with db_service.uow() as uow:
            schedules: List[Schedule] = uow.s.query(Schedule).order_by(Schedule.name).all()
            recent_runs: List[ScheduleRun] = (
                uow.s.query(ScheduleRun).order_by(ScheduleRun.enqueued_at.desc()).limit(500).all()
            )

        schedule_runs: Dict[int, List[ScheduleRun]] = {s.id: [] for s in schedules}
        unmatched: List[ScheduleRun] = []

        for run in recent_runs:
            snapshot = run.job_snapshot or {}
            matched = False

            sid = snapshot.get("schedule_id")
            if sid is not None:
                try:
                    sid = int(sid)
                    if sid in schedule_runs:
                        schedule_runs[sid].append(run)
                        matched = True
                except (ValueError, TypeError):
                    pass

            if not matched:
                sname = snapshot.get("schedule_name")
                for s in schedules:
                    if sname and sname == s.name:
                        schedule_runs[s.id].append(run)
                        matched = True
                        break
                    if not matched and run.job_id and s.name in run.job_id:
                        schedule_runs[s.id].append(run)
                        matched = True
                        break

            if not matched:
                unmatched.append(run)

        pipelines = []
        for schedule in schedules:
            runs = schedule_runs[schedule.id][:10]

            last_run = runs[0] if runs else None
            last_status = "never"
            last_run_at = None
            last_duration_s = None

            if last_run:
                last_status = last_run.status or "unknown"
                last_run_at = last_run.started_at or last_run.enqueued_at
                if last_run.started_at and last_run.finished_at:
                    last_duration_s = round((last_run.finished_at - last_run.started_at).total_seconds(), 1)

            completed = sum(1 for r in runs if r.status == "completed")
            success_rate = round(completed / len(runs), 2) if runs else None

            recent = []
            for r in runs:
                dur = None
                if r.started_at and r.finished_at:
                    dur = round((r.finished_at - r.started_at).total_seconds(), 1)
                recent.append(
                    {
                        "id": r.id,
                        "status": r.status,
                        "started_at": r.started_at.isoformat() if r.started_at else None,
                        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                        "duration_s": dur,
                        "error": r.error,
                    }
                )

            pipelines.append(
                {
                    "id": schedule.id,
                    "name": schedule.name,
                    "job_type": schedule.job_type,
                    "target": schedule.target,
                    "enabled": schedule.enabled,
                    "cron": schedule.cron,
                    "next_run_at": schedule.next_run_at.isoformat() if schedule.next_run_at else None,
                    "last_status": last_status,
                    "last_run_at": last_run_at.isoformat() if last_run_at else None,
                    "last_duration_s": last_duration_s,
                    "success_rate_10": success_rate,
                    "recent_runs": recent,
                }
            )

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "pipelines": pipelines,
        }

    except Exception:
        _logger.exception("Error fetching pipelines status:")
        raise HTTPException(status_code=500, detail="Failed to fetch pipelines status")

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from sqlalchemy import desc

from src.notification.logger import setup_logger
from src.data.database import get_session, close_session
from src.data.db.telegram_models import (
    TelegramUser, VerificationCode, Alert, Schedule, Setting,
    Feedback, CommandAudit, BroadcastLog
)


_logger = setup_logger(__name__)


class TelegramRepository:
    """Repository for Telegram-related DB operations using SQLAlchemy."""

    def __init__(self, session: Session = None):
        self.session = session or get_session()
        self._owns_session = session is None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._owns_session and self.session:
            close_session(self.session)
            self.session = None

    # Users
    def upsert_user(self, telegram_user_id: str, **fields) -> TelegramUser:
        user = self.session.get(TelegramUser, telegram_user_id)
        if not user:
            user = TelegramUser(telegram_user_id=telegram_user_id)
            self.session.add(user)
        for k, v in fields.items():
            if hasattr(user, k):
                setattr(user, k, v)
        self.session.commit()
        return user

    def get_user(self, telegram_user_id: str) -> Optional[TelegramUser]:
        return self.session.get(TelegramUser, telegram_user_id)

    def get_user_status(self, telegram_user_id: str) -> Optional[Dict[str, Any]]:
        user = self.get_user(telegram_user_id)
        if not user:
            return None
        return {
            'email': user.email,
            'telegram_verified': bool(user.verified),
            'telegram_approved': bool(user.approved),
            'telegram_code_sent_time': user.code_sent_time,
            'telegram_language': user.language,
            'telegram_is_admin': bool(user.is_admin)
        }

    def set_verification_code(self, telegram_user_id: str, code: str, sent_time: int) -> VerificationCode:
        self.upsert_user(telegram_user_id, verification_code=code, code_sent_time=sent_time, verified=False)
        vc = VerificationCode(telegram_user_id=telegram_user_id, code=code, sent_time=sent_time)
        self.session.add(vc)
        self.session.commit()
        return vc

    def count_codes_last_hour(self, telegram_user_id: str) -> int:
        """Count verification codes sent in last hour for rate limiting."""
        import time
        from sqlalchemy import func, and_
        one_hour_ago = int(time.time()) - 3600
        count = self.session.query(func.count(VerificationCode.id)).filter(
            and_(VerificationCode.telegram_user_id == telegram_user_id,
                 VerificationCode.sent_time > one_hour_ago)
        ).scalar() or 0
        return int(count)

    def approve_user(self, telegram_user_id: str, approved: bool = True) -> bool:
        self.upsert_user(telegram_user_id, approved=approved)
        return True

    def list_pending_approvals(self) -> List[Dict[str, Any]]:
        users = (self.session.query(TelegramUser)
                 .filter(TelegramUser.verified == True)
                 .filter(TelegramUser.approved == False)
                 .all())
        return [{'telegram_user_id': u.telegram_user_id, 'email': u.email, 'code_sent_time': u.code_sent_time} for u in users]

    def set_user_limit(self, telegram_user_id: str, limit_type: str, value: int):
        assert limit_type in ('max_alerts', 'max_schedules')
        self.upsert_user(telegram_user_id, **{limit_type: value})

    def list_users(self) -> List[TelegramUser]:
        """List all users."""
        return self.session.query(TelegramUser).all()

    def get_admin_user_ids(self) -> List[str]:
        """Get all admin user IDs."""
        admin_users = self.session.query(TelegramUser.telegram_user_id).filter(
            TelegramUser.is_admin == True
        ).all()
        return [user[0] for user in admin_users]

    # Alerts
    def add_alert(self, user_id: str, ticker: str, price: float, condition: str, email: bool = False) -> int:
        from src.telegram.screener.rearm_alert_system import EnhancedAlertConfig

        created = datetime.now(timezone.utc).isoformat()

        # Create enhanced alert config with re-arm enabled by default
        enhanced_config = EnhancedAlertConfig.from_simple_params(
            ticker=ticker,
            threshold=price,
            direction=condition,
            email=email,
            rearm_enabled=True  # Re-arm enabled by default
        )

        alert = Alert(
            ticker=ticker,
            user_id=user_id,
            price=price,
            condition=condition,
            active=True,
            email=email,
            created=created,
            re_arm_config=enhanced_config.to_json(),  # Store enhanced config
            is_armed=True,  # Initially armed
            last_price=None,  # Will be set on first evaluation
            last_triggered_at=None
        )
        self.session.add(alert)
        self.session.commit()
        return alert.id

    def get_alert(self, alert_id: int) -> Optional[Alert]:
        return self.session.get(Alert, alert_id)

    def list_alerts(self, user_id: str) -> List[Alert]:
        return self.session.query(Alert).filter(Alert.user_id == user_id).all()

    def update_alert(self, alert_id: int, **fields) -> bool:
        alert = self.get_alert(alert_id)
        if not alert:
            return False
        for k, v in fields.items():
            if hasattr(alert, k):
                setattr(alert, k, v)
        self.session.commit()
        return True

    def delete_alert(self, alert_id: int) -> bool:
        alert = self.get_alert(alert_id)
        if not alert:
            return False
        self.session.delete(alert)
        self.session.commit()
        return True

    def add_indicator_alert(self, user_id: str, ticker: str, config_json: str,
                            alert_action: str = 'notify', timeframe: str = '15m', email: bool = False) -> int:
        created = datetime.now(timezone.utc).isoformat()
        alert = Alert(ticker=ticker, user_id=user_id, alert_type='indicator', config_json=config_json,
                      alert_action=alert_action, timeframe=timeframe, active=True, email=email, created=created)
        self.session.add(alert)
        self.session.commit()
        return alert.id

    def get_active_alerts(self) -> List[Alert]:
        return self.session.query(Alert).filter(Alert.active == True).all()

    def get_alerts_by_type(self, alert_type: Optional[str] = None) -> List[Alert]:
        q = self.session.query(Alert)
        if alert_type:
            q = q.filter(Alert.alert_type == alert_type)
        return q.all()

    # Schedules
    def add_schedule(self, user_id: str, ticker: str, scheduled_time: str, period: str = None,
                     email: bool = False, indicators: str = None, interval: str = None, provider: str = None) -> int:
        created = datetime.now(timezone.utc).isoformat()
        schedule = Schedule(
            user_id=user_id, ticker=ticker, scheduled_time=scheduled_time, period=period,
            active=True, email=email, indicators=indicators, interval=interval, provider=provider,
            created=created, schedule_type='report'
        )
        self.session.add(schedule)
        self.session.commit()
        return schedule.id

    def create_schedule(self, schedule_data: Dict[str, Any]) -> int:
        created = datetime.now(timezone.utc).isoformat()
        schedule = Schedule(
            ticker=schedule_data.get('ticker'),
            user_id=schedule_data.get('telegram_user_id'),
            scheduled_time=schedule_data.get('scheduled_time'),
            period=schedule_data.get('period', 'daily'),
            email=bool(schedule_data.get('email', False)),
            indicators=schedule_data.get('indicators'),
            interval=schedule_data.get('interval', '1d'),
            provider=schedule_data.get('provider', 'yf'),
            schedule_type=schedule_data.get('schedule_type', 'report'),
            list_type=schedule_data.get('list_type'),
            created=created
        )
        self.session.add(schedule)
        self.session.commit()
        return schedule.id

    def get_schedule(self, schedule_id: int) -> Optional[Schedule]:
        return self.session.get(Schedule, schedule_id)

    def list_schedules(self, user_id: str) -> List[Schedule]:
        return self.session.query(Schedule).filter(Schedule.user_id == user_id).all()

    def update_schedule(self, schedule_id: int, **fields) -> bool:
        schedule = self.get_schedule(schedule_id)
        if not schedule:
            return False
        for k, v in fields.items():
            if hasattr(schedule, k):
                setattr(schedule, k, v)
        self.session.commit()
        return True

    def delete_schedule(self, schedule_id: int) -> bool:
        schedule = self.get_schedule(schedule_id)
        if not schedule:
            return False
        self.session.delete(schedule)
        self.session.commit()
        return True

    def add_json_schedule(self, user_id: str, config_json: str, schedule_config: str = 'advanced') -> int:
        created = datetime.now(timezone.utc).isoformat()
        schedule = Schedule(user_id=user_id, config_json=config_json, schedule_config=schedule_config,
                            active=True, created=created)
        self.session.add(schedule)
        self.session.commit()
        return schedule.id

    def get_active_schedules(self) -> List[Schedule]:
        return self.session.query(Schedule).filter(Schedule.active == True).all()

    def get_schedules_by_config(self, schedule_config: Optional[str] = None) -> List[Schedule]:
        q = self.session.query(Schedule)
        if schedule_config:
            q = q.filter(Schedule.schedule_config == schedule_config)
        return q.all()

    # Settings
    def set_setting(self, key: str, value: str):
        setting = self.session.get(Setting, key)
        if not setting:
            setting = Setting(key=key, value=value)
            self.session.add(setting)
        else:
            setting.value = value
        self.session.commit()

    def get_setting(self, key: str) -> Optional[str]:
        setting = self.session.get(Setting, key)
        return setting.value if setting else None

    # Feedback
    def add_feedback(self, user_id: str, feedback_type: str, message: str) -> int:
        created = datetime.now(timezone.utc).isoformat()
        fb = Feedback(user_id=user_id, type=feedback_type, message=message, created=created, status='open')
        self.session.add(fb)
        self.session.commit()
        return fb.id

    def list_feedback(self, feedback_type: Optional[str] = None) -> List[Feedback]:
        q = self.session.query(Feedback).order_by(desc(Feedback.created))
        if feedback_type:
            q = q.filter(Feedback.type == feedback_type)
        return q.all()

    def update_feedback_status(self, feedback_id: int, status: str) -> bool:
        fb = self.session.get(Feedback, feedback_id)
        if not fb:
            return False
        fb.status = status
        self.session.commit()
        return True

    # Command audit
    def log_command_audit(self, telegram_user_id: str, command: str, full_message: str = None,
                          is_registered_user: bool = False, user_email: str = None,
                          success: bool = True, error_message: str = None,
                          response_time_ms: int = None) -> int:
        created = datetime.now(timezone.utc).isoformat()
        audit = CommandAudit(
            telegram_user_id=telegram_user_id,
            command=command,
            full_message=full_message,
            is_registered_user=is_registered_user,
            user_email=user_email,
            success=success,
            error_message=error_message,
            response_time_ms=response_time_ms,
            created=created
        )
        self.session.add(audit)
        self.session.commit()
        return audit.id

    def get_user_command_history(self, telegram_user_id: str, limit: int = 50) -> List[CommandAudit]:
        return (self.session.query(CommandAudit)
                .filter(CommandAudit.telegram_user_id == telegram_user_id)
                .order_by(desc(CommandAudit.created))
                .limit(limit)
                .all())

    def get_all_command_audit(self, limit: int = 100, offset: int = 0,
                              user_id: str = None, command: str = None,
                              success_only: Optional[bool] = None,
                              start_date: str = None, end_date: str = None) -> List[CommandAudit]:
        q = self.session.query(CommandAudit)
        if user_id:
            q = q.filter(CommandAudit.telegram_user_id == user_id)
        if command:
            q = q.filter(CommandAudit.command.like(f"%{command}%"))
        if success_only is not None:
            q = q.filter(CommandAudit.success == bool(success_only))
        if start_date:
            q = q.filter(CommandAudit.created >= start_date)
        if end_date:
            q = q.filter(CommandAudit.created <= end_date)
        return q.order_by(desc(CommandAudit.created)).offset(offset).limit(limit).all()

    def get_command_audit_stats(self) -> Dict[str, Any]:
        total = self.session.query(CommandAudit).count()
        success = self.session.query(CommandAudit).filter(CommandAudit.success == True).count()
        failed = self.session.query(CommandAudit).filter(CommandAudit.success == False).count()
        # Top commands
        top = (self.session.query(CommandAudit.command)
               .all())
        # Simple aggregation client-side to keep implementation brief
        counts: Dict[str, int] = {}
        for (cmd,) in top:
            counts[cmd] = counts.get(cmd, 0) + 1
        top_commands = [{'command': k, 'count': v} for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]]
        # Recent activity not strictly needed; omit heavy time math for now
        return {
            'total_commands': total,
            'successful_commands': success,
            'failed_commands': failed,
            'top_commands': top_commands,
            'success_rate': (success / total * 100) if total > 0 else 0
        }

    def get_unique_users_command_history(self) -> List[Dict[str, Any]]:
        """Get unique users command history summary."""
        from sqlalchemy import func
        rows = self.session.query(
            CommandAudit.telegram_user_id,
            func.count(CommandAudit.id),
            func.count(func.nullif(CommandAudit.success == False, True)),
            func.count(func.nullif(CommandAudit.is_registered_user == False, True)),
            func.max(CommandAudit.created),
            func.min(CommandAudit.created)
        ).group_by(CommandAudit.telegram_user_id).order_by(func.count(CommandAudit.id).desc()).all()

        result = []
        for r in rows:
            total = r[1]
            success = r[2]
            result.append({
                'telegram_user_id': r[0],
                'total_commands': total,
                'successful_commands': success,
                'registered_commands': r[3],
                'last_command': r[4],
                'first_command': r[5],
                'success_rate': (success / total * 100) if total > 0 else 0
            })
        return result

    # Broadcast log
    def add_broadcast_log(self, message: str, sent_by: str, success_count: int = 0, total_count: int = 0) -> int:
        bl = BroadcastLog(message=message, sent_by=sent_by, success_count=success_count, total_count=total_count,
                          created=datetime.now(timezone.utc).isoformat())
        self.session.add(bl)
        self.session.commit()
        return bl.id



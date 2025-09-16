from datetime import datetime
from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, Numeric, ForeignKey, Index
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import relationship

# Reuse the existing Base from src.data.database to share metadata
from src.data.database import Base


class TelegramUser(Base):
    __tablename__ = 'telegram_users'

    telegram_user_id = Column(String(255), primary_key=True)
    email = Column(String(255), nullable=True)
    verification_code = Column(String(32), nullable=True)
    code_sent_time = Column(Integer, nullable=True)
    verified = Column(Boolean, default=False, index=True)
    approved = Column(Boolean, default=False, index=True)
    language = Column(String(10), nullable=True)
    is_admin = Column(Boolean, default=False, index=True)
    max_alerts = Column(Integer, default=5)
    max_schedules = Column(Integer, default=5)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_telegram_users_email', 'email'),
    )


class VerificationCode(Base):
    __tablename__ = 'telegram_verification_codes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    telegram_user_id = Column(String(255), ForeignKey('telegram_users.telegram_user_id'), nullable=False)
    code = Column(String(32), nullable=False)
    sent_time = Column(Integer, nullable=False)


class Alert(Base):
    __tablename__ = 'telegram_alerts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(50), nullable=False, index=True)
    user_id = Column(String(255), ForeignKey('telegram_users.telegram_user_id'), nullable=False, index=True)
    price = Column(Numeric(20, 8), nullable=True)
    condition = Column(String(50), nullable=True)
    active = Column(Boolean, default=True, index=True)
    email = Column(Boolean, default=False)
    created = Column(String(40), nullable=True)
    alert_type = Column(String(20), default='price', index=True)
    timeframe = Column(String(10), default='15m')
    config_json = Column(Text, nullable=True)
    alert_action = Column(String(50), default='notify')

    __table_args__ = (
        Index('idx_alerts_user_active', 'user_id', 'active'),
    )


class Schedule(Base):
    __tablename__ = 'telegram_schedules'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(50), nullable=True)
    user_id = Column(String(255), ForeignKey('telegram_users.telegram_user_id'), nullable=False, index=True)
    scheduled_time = Column(String(20), nullable=False)
    period = Column(String(20), nullable=True)
    active = Column(Boolean, default=True, index=True)
    email = Column(Boolean, default=False)
    indicators = Column(Text, nullable=True)
    interval = Column(String(10), nullable=True)
    provider = Column(String(20), nullable=True)
    created = Column(String(40), nullable=True)
    schedule_type = Column(String(20), default='report', index=True)
    list_type = Column(String(50), nullable=True)
    config_json = Column(Text, nullable=True)
    schedule_config = Column(String(20), default='simple', index=True)


class Setting(Base):
    __tablename__ = 'telegram_settings'

    key = Column(String(100), primary_key=True)
    value = Column(Text, nullable=True)


class Feedback(Base):
    __tablename__ = 'telegram_feedback'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), ForeignKey('telegram_users.telegram_user_id'), nullable=False)
    type = Column(String(50), nullable=False)  # 'feedback' or 'feature_request'
    message = Column(Text, nullable=False)
    created = Column(String(40), nullable=True)
    status = Column(String(20), default='open')  # 'open', 'in_progress', 'closed'


class CommandAudit(Base):
    __tablename__ = 'telegram_command_audit'

    id = Column(Integer, primary_key=True, autoincrement=True)
    telegram_user_id = Column(String(255), nullable=False, index=True)
    command = Column(String(255), nullable=False)
    full_message = Column(Text, nullable=True)
    is_registered_user = Column(Boolean, default=False)
    user_email = Column(String(255), nullable=True)
    success = Column(Boolean, default=True, index=True)
    error_message = Column(Text, nullable=True)
    response_time_ms = Column(Integer, nullable=True)
    created = Column(String(40), nullable=True)

    __table_args__ = (
        Index('idx_command_audit_created', 'created'),
        Index('idx_command_audit_command', 'command'),
    )


class BroadcastLog(Base):
    __tablename__ = 'telegram_broadcast_log'

    id = Column(Integer, primary_key=True, autoincrement=True)
    message = Column(Text, nullable=False)
    sent_by = Column(String(255), nullable=False)
    success_count = Column(Integer, default=0)
    total_count = Column(Integer, default=0)
    created = Column(String(40), nullable=True)



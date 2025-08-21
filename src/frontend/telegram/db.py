import sqlite3
import time
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

DB_PATH = "db/telegram_screener.sqlite3"

# --- DB Schema ---
# users: telegram_user_id TEXT PRIMARY KEY, email TEXT, verification_code TEXT, code_sent_time INTEGER, verified INTEGER, approved INTEGER DEFAULT 0, language TEXT, is_admin INTEGER DEFAULT 0
# codes: telegram_user_id TEXT, code TEXT, sent_time INTEGER
# alerts: id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, user_id TEXT, price REAL, condition TEXT, active INTEGER DEFAULT 1, created TEXT
# schedules: id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, user_id TEXT, scheduled_time TEXT, period TEXT, active INTEGER DEFAULT 1, email INTEGER, indicators TEXT, interval TEXT, provider TEXT, created TEXT

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        telegram_user_id TEXT PRIMARY KEY,
        email TEXT,
        verification_code TEXT,
        code_sent_time INTEGER,
        verified INTEGER DEFAULT 0,
        approved INTEGER DEFAULT 0,
        language TEXT,
        is_admin INTEGER DEFAULT 0,
        max_alerts INTEGER DEFAULT 5,
        max_schedules INTEGER DEFAULT 5
    )''')
    # Migration: add columns if missing
    c.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in c.fetchall()]
    if "language" not in columns:
        c.execute("ALTER TABLE users ADD COLUMN language TEXT")
    if "is_admin" not in columns:
        c.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
    if "approved" not in columns:
        c.execute("ALTER TABLE users ADD COLUMN approved INTEGER DEFAULT 0")
    if "max_alerts" not in columns:
        c.execute("ALTER TABLE users ADD COLUMN max_alerts INTEGER DEFAULT 5")
    if "max_schedules" not in columns:
        c.execute("ALTER TABLE users ADD COLUMN max_schedules INTEGER DEFAULT 5")
    # Set default values for existing users if NULL
    c.execute("UPDATE users SET max_alerts=5 WHERE max_alerts IS NULL")
    c.execute("UPDATE users SET max_schedules=5 WHERE max_schedules IS NULL")
    c.execute('''CREATE TABLE IF NOT EXISTS codes (
        telegram_user_id TEXT,
        code TEXT,
        sent_time INTEGER
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        user_id TEXT,
        price REAL,
        condition TEXT,
        active INTEGER DEFAULT 1,
        email INTEGER DEFAULT 0,
        created TEXT,
        alert_type TEXT DEFAULT 'price',
        timeframe TEXT DEFAULT '15m',
        config_json TEXT,
        alert_action TEXT DEFAULT 'notify'
    )''')
    # Migration: add new columns if missing
    c.execute("PRAGMA table_info(alerts)")
    columns = [row[1] for row in c.fetchall()]
    if "alert_type" not in columns:
        c.execute("ALTER TABLE alerts ADD COLUMN alert_type TEXT DEFAULT 'price'")
    if "timeframe" not in columns:
        c.execute("ALTER TABLE alerts ADD COLUMN timeframe TEXT DEFAULT '15m'")
    if "config_json" not in columns:
        c.execute("ALTER TABLE alerts ADD COLUMN config_json TEXT")
    if "alert_action" not in columns:
        c.execute("ALTER TABLE alerts ADD COLUMN alert_action TEXT DEFAULT 'notify'")
    c.execute('''CREATE TABLE IF NOT EXISTS schedules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        user_id TEXT,
        scheduled_time TEXT,
        period TEXT,
        active INTEGER DEFAULT 1,
        email INTEGER,
        indicators TEXT,
        interval TEXT,
        provider TEXT,
        created TEXT,
        schedule_type TEXT DEFAULT 'report',
        list_type TEXT,
        config_json TEXT,
        schedule_config TEXT DEFAULT 'simple'
    )''')
    # Migration: add new columns if missing
    c.execute("PRAGMA table_info(schedules)")
    columns = [row[1] for row in c.fetchall()]
    if "schedule_type" not in columns:
        c.execute("ALTER TABLE schedules ADD COLUMN schedule_type TEXT DEFAULT 'report'")
    if "list_type" not in columns:
        c.execute("ALTER TABLE schedules ADD COLUMN list_type TEXT")
    if "config_json" not in columns:
        c.execute("ALTER TABLE schedules ADD COLUMN config_json TEXT")
    if "schedule_config" not in columns:
        c.execute("ALTER TABLE schedules ADD COLUMN schedule_config TEXT DEFAULT 'simple'")
    c.execute('''CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        type TEXT,  -- 'feedback' or 'feature_request'
        message TEXT,
        created TEXT,
        status TEXT DEFAULT 'open'  -- 'open', 'in_progress', 'closed'
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS command_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        telegram_user_id TEXT NOT NULL,
        command TEXT NOT NULL,
        full_message TEXT,
        is_registered_user INTEGER DEFAULT 0,
        user_email TEXT,
        success INTEGER DEFAULT 1,
        error_message TEXT,
        response_time_ms INTEGER,
        created TEXT DEFAULT CURRENT_TIMESTAMP
    )''')
    # Create index for better query performance
    c.execute('''CREATE INDEX IF NOT EXISTS idx_command_audit_user_id ON command_audit(telegram_user_id)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_command_audit_created ON command_audit(created)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_command_audit_command ON command_audit(command)''')

    # Broadcast log table
    c.execute('''CREATE TABLE IF NOT EXISTS broadcast_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message TEXT NOT NULL,
        sent_by TEXT NOT NULL,
        success_count INTEGER DEFAULT 0,
        total_count INTEGER DEFAULT 0,
        created TEXT DEFAULT CURRENT_TIMESTAMP
    )''')

    conn.commit()
    conn.close()

def set_user_email(telegram_user_id: str, email: str, code: str, sent_time: int, language: Optional[str] = None, is_admin: Optional[bool] = None):
    """Set or update user's email, store verification code, language, and is_admin. Defaults max_alerts and max_schedules to 5 on insert."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    admin_val = 1 if is_admin else 0 if is_admin is not None else None
    # Always set max_alerts and max_schedules to 5 on insert if not present
    if language is not None and admin_val is not None:
        c.execute("REPLACE INTO users (telegram_user_id, email, verification_code, code_sent_time, verified, language, is_admin, max_alerts, max_schedules) VALUES (?, ?, ?, ?, 0, ?, ?, 5, 5)",
                  (telegram_user_id, email, code, sent_time, language, admin_val))
    elif language is not None:
        c.execute("REPLACE INTO users (telegram_user_id, email, verification_code, code_sent_time, verified, language, max_alerts, max_schedules) VALUES (?, ?, ?, ?, 0, ?, 5, 5)",
                  (telegram_user_id, email, code, sent_time, language))
    elif admin_val is not None:
        c.execute("REPLACE INTO users (telegram_user_id, email, verification_code, code_sent_time, verified, is_admin, max_alerts, max_schedules) VALUES (?, ?, ?, ?, 0, ?, 5, 5)",
                  (telegram_user_id, email, code, sent_time, admin_val))
    else:
        c.execute("REPLACE INTO users (telegram_user_id, email, verification_code, code_sent_time, verified, max_alerts, max_schedules) VALUES (?, ?, ?, ?, 0, 5, 5)",
                  (telegram_user_id, email, code, sent_time))
    c.execute("INSERT INTO codes (telegram_user_id, code, sent_time) VALUES (?, ?, ?)",
              (telegram_user_id, code, sent_time))
    conn.commit()
    conn.close()

def get_user_status(telegram_user_id: str):
    """Get user's email, verification status, approval status, language, and is_admin."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT email, verified, approved, code_sent_time, language, is_admin FROM users WHERE telegram_user_id=?", (telegram_user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"email": row[0], "verified": bool(row[1]), "approved": bool(row[2]), "code_sent_time": row[3], "language": row[4], "is_admin": bool(row[5])}
    return None

def get_verification_code(telegram_user_id: str) -> Optional[str]:
    """Get the latest verification code for the user."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT verification_code, code_sent_time FROM users WHERE telegram_user_id=?", (telegram_user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return row[0], row[1]
    return None, None

def verify_code(telegram_user_id: str, code: str, expiry_seconds: int = 3600) -> bool:
    """Verify the code for the user. Returns True if valid and not expired."""
    code_db, sent_time = get_verification_code(telegram_user_id)
    if code_db and code_db == code and (time.time() - sent_time) <= expiry_seconds:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE users SET verified=1 WHERE telegram_user_id=?", (telegram_user_id,))
        conn.commit()
        conn.close()
        return True
    return False

def approve_user(telegram_user_id: str) -> bool:
    """Approve a user for access to restricted features. Returns True if successful."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET approved=1 WHERE telegram_user_id=?", (telegram_user_id,))
    conn.commit()
    conn.close()
    return True

def reject_user(telegram_user_id: str) -> bool:
    """Reject a user's approval request. Returns True if successful."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET approved=0 WHERE telegram_user_id=?", (telegram_user_id,))
    conn.commit()
    conn.close()
    return True

def get_pending_approvals() -> List[Dict[str, Any]]:
    """Get list of users who are verified but not approved."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT telegram_user_id, email, code_sent_time FROM users WHERE verified=1 AND approved=0")
    rows = c.fetchall()
    conn.close()
    return [{"telegram_user_id": row[0], "email": row[1], "code_sent_time": row[2]} for row in rows]

def count_codes_last_hour(telegram_user_id: str) -> int:
    """Count how many codes were sent to this user in the last hour (rate limiting)."""
    one_hour_ago = int(time.time()) - 3600
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM codes WHERE telegram_user_id=? AND sent_time > ?", (telegram_user_id, one_hour_ago))
    count = c.fetchone()[0]
    conn.close()
    return count

# --- ALERTS CRUD ---
def add_alert(user_id: str, ticker: str, price: float, condition: str, email: bool = False) -> int:
    """Add a new alert. Returns alert id."""
    created = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO alerts (ticker, user_id, price, condition, active, email, created) VALUES (?, ?, ?, ?, 1, ?, ?)",
              (ticker, user_id, price, condition, 1 if email else 0, created))
    alert_id = c.lastrowid
    conn.commit()
    conn.close()
    return alert_id

def get_alert(alert_id: int) -> Optional[Dict[str, Any]]:
    """Get alert by id."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM alerts WHERE id=?", (alert_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return dict(zip([d[0] for d in c.description], row))
    return None

def list_alerts(user_id: str) -> List[Dict[str, Any]]:
    """List all alerts for a user."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM alerts WHERE user_id=?", (user_id,))
    rows = c.fetchall()
    conn.close()
    return [dict(zip([d[0] for d in c.description], row)) for row in rows]

def update_alert(alert_id: int, **kwargs) -> bool:
    """Update alert fields by id."""
    if not kwargs:
        return False
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    fields = ", ".join(f"{k}=?" for k in kwargs)
    values = list(kwargs.values()) + [alert_id]
    c.execute(f"UPDATE alerts SET {fields} WHERE id=?", values)
    conn.commit()
    conn.close()
    return True

def delete_alert(alert_id: int) -> bool:
    """Delete alert by id."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM alerts WHERE id=?", (alert_id,))
    conn.commit()
    conn.close()
    return True

def add_indicator_alert(user_id: str, ticker: str, config_json: str, alert_action: str = "notify",
                       timeframe: str = "15m", email: bool = False) -> int:
    """Add a new indicator-based alert. Returns alert id."""
    created = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO alerts
                 (ticker, user_id, alert_type, config_json, alert_action, timeframe, active, email, created)
                 VALUES (?, ?, 'indicator', ?, ?, ?, 1, ?, ?)""",
              (ticker, user_id, config_json, alert_action, timeframe, 1 if email else 0, created))
    alert_id = c.lastrowid
    conn.commit()
    conn.close()
    return alert_id

def get_active_alerts() -> List[Dict[str, Any]]:
    """Get all active alerts (both price and indicator)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM alerts WHERE active=1")
    rows = c.fetchall()
    conn.close()
    return [dict(zip([d[0] for d in c.description], row)) for row in rows]

def get_alerts_by_type(alert_type: str = None) -> List[Dict[str, Any]]:
    """Get alerts filtered by type (price, indicator, or all if None)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if alert_type:
        c.execute("SELECT * FROM alerts WHERE alert_type=?", (alert_type,))
    else:
        c.execute("SELECT * FROM alerts")
    rows = c.fetchall()
    conn.close()
    return [dict(zip([d[0] for d in c.description], row)) for row in rows]

# --- SCHEDULES CRUD ---
def add_schedule(user_id: str, ticker: str, scheduled_time: str, period: str = None, email: int = 0, indicators: str = None, interval: str = None, provider: str = None) -> int:
    """Add a new schedule. Returns schedule id."""
    created = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO schedules (ticker, user_id, scheduled_time, period, active, email, indicators, interval, provider, created, schedule_type, list_type) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, 'report', NULL)",
              (ticker, user_id, scheduled_time, period, email, indicators, interval, provider, created))
    schedule_id = c.lastrowid
    conn.commit()
    conn.close()
    return schedule_id


def create_schedule(schedule_data: Dict[str, Any]) -> int:
    """Create a new schedule with full data dictionary. Returns schedule id."""
    created = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Extract fields from schedule_data
    ticker = schedule_data.get('ticker')
    user_id = schedule_data.get('telegram_user_id')
    scheduled_time = schedule_data.get('scheduled_time')
    period = schedule_data.get('period', 'daily')
    email = 1 if schedule_data.get('email', False) else 0
    indicators = schedule_data.get('indicators')
    interval = schedule_data.get('interval', '1d')
    provider = schedule_data.get('provider', 'yf')
    schedule_type = schedule_data.get('schedule_type', 'report')
    list_type = schedule_data.get('list_type')

    c.execute("""INSERT INTO schedules
                 (ticker, user_id, scheduled_time, period, active, email, indicators, interval, provider, created, schedule_type, list_type)
                 VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)""",
              (ticker, user_id, scheduled_time, period, email, indicators, interval, provider, created, schedule_type, list_type))
    schedule_id = c.lastrowid
    conn.commit()
    conn.close()
    return schedule_id

def get_schedule(schedule_id: int) -> Optional[Dict[str, Any]]:
    """Get schedule by id."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM schedules WHERE id=?", (schedule_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return dict(zip([d[0] for d in c.description], row))
    return None

def get_schedule_by_id(schedule_id: int) -> Optional[Dict[str, Any]]:
    """Get schedule by id (alias for get_schedule)."""
    return get_schedule(schedule_id)

def list_schedules(user_id: str) -> List[Dict[str, Any]]:
    """List all schedules for a user."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM schedules WHERE user_id=?", (user_id,))
    rows = c.fetchall()
    conn.close()
    return [dict(zip([d[0] for d in c.description], row)) for row in rows]

def update_schedule(schedule_id: int, **kwargs) -> bool:
    """Update schedule fields by id."""
    if not kwargs:
        return False
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    fields = ", ".join(f"{k}=?" for k in kwargs)
    values = list(kwargs.values()) + [schedule_id]
    c.execute(f"UPDATE schedules SET {fields} WHERE id=?", values)
    conn.commit()
    conn.close()
    return True

def delete_schedule(schedule_id: int) -> bool:
    """Delete schedule by id."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM schedules WHERE id=?", (schedule_id,))
    conn.commit()
    conn.close()
    return True

def add_json_schedule(user_id: str, config_json: str, schedule_config: str = "advanced") -> int:
    """Add a new JSON-based schedule. Returns schedule id."""
    created = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO schedules
                 (user_id, config_json, schedule_config, active, created)
                 VALUES (?, ?, ?, 1, ?)""",
              (user_id, config_json, schedule_config, created))
    schedule_id = c.lastrowid
    conn.commit()
    conn.close()
    return schedule_id

def get_active_schedules() -> List[Dict[str, Any]]:
    """Get all active schedules."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM schedules WHERE active=1")
    rows = c.fetchall()
    conn.close()
    return [dict(zip([d[0] for d in c.description], row)) for row in rows]

def get_schedules_by_config(schedule_config: str = None) -> List[Dict[str, Any]]:
    """Get schedules filtered by configuration type."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if schedule_config:
        c.execute("SELECT * FROM schedules WHERE schedule_config=?", (schedule_config,))
    else:
        c.execute("SELECT * FROM schedules")
    rows = c.fetchall()
    conn.close()
    return [dict(zip([d[0] for d in c.description], row)) for row in rows]

# --- SETTINGS CRUD ---
def set_setting(key: str, value: str):
    """Set a global setting (key-value)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()

def get_setting(key: str) -> str:
    """Get a global setting value by key."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

# --- USER LIMITS ---
def set_user_limit(telegram_user_id: str, limit_type: str, value: int):
    """Set per-user max_alerts or max_schedules."""
    assert limit_type in ("max_alerts", "max_schedules")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"UPDATE users SET {limit_type}=? WHERE telegram_user_id=?", (value, telegram_user_id))
    conn.commit()
    conn.close()

def get_user_limit(telegram_user_id: str, limit_type: str) -> int:
    """Get per-user max_alerts or max_schedules."""
    assert limit_type in ("max_alerts", "max_schedules")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"SELECT {limit_type} FROM users WHERE telegram_user_id=?", (telegram_user_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row and row[0] is not None else None

# --- USER MANAGEMENT ---
def set_user_max_alerts(telegram_user_id: str, max_alerts: int):
    """Set max alerts for a specific user."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET max_alerts=? WHERE telegram_user_id=?", (max_alerts, telegram_user_id))
    conn.commit()
    conn.close()

def set_user_max_schedules(telegram_user_id: str, max_schedules: int):
    """Set max schedules for a specific user."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET max_schedules=? WHERE telegram_user_id=?", (max_schedules, telegram_user_id))
    conn.commit()
    conn.close()

def set_global_setting(key: str, value: str):
    """Set global setting (alias for set_setting)."""
    set_setting(key, value)

def update_user_email(telegram_user_id: str, email: str):
    """Update user email (set to None to reset)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET email=? WHERE telegram_user_id=?", (email, telegram_user_id))
    conn.commit()
    conn.close()

def update_user_verification(telegram_user_id: str, verified: bool):
    """Update user verification status."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET verified=? WHERE telegram_user_id=?", (1 if verified else 0, telegram_user_id))
    conn.commit()
    conn.close()

# --- USER LISTING ---
def list_users() -> List[Dict[str, Any]]:
    """Return list of all users with full info."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT telegram_user_id, email, verified, approved, language, is_admin, max_alerts, max_schedules FROM users")
    rows = c.fetchall()
    conn.close()
    return [
        {
            "telegram_user_id": row[0],
            "email": row[1],
            "verified": bool(row[2]),
            "approved": bool(row[3]),
            "language": row[4],
            "is_admin": bool(row[5]),
            "max_alerts": row[6],
            "max_schedules": row[7]
        }
        for row in rows
    ]

# --- FEEDBACK/FEATURE REQUESTS CRUD ---
def add_feedback(user_id: str, feedback_type: str, message: str) -> int:
    """Add feedback or feature request. Returns feedback id."""
    created = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO feedback (user_id, type, message, created, status) VALUES (?, ?, ?, ?, 'open')",
              (user_id, feedback_type, message, created))
    feedback_id = c.lastrowid
    conn.commit()
    conn.close()
    return feedback_id

def list_feedback(feedback_type: str = None) -> List[Dict[str, Any]]:
    """List all feedback/feature requests, optionally filtered by type."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if feedback_type:
        c.execute("SELECT * FROM feedback WHERE type=? ORDER BY created DESC", (feedback_type,))
    else:
        c.execute("SELECT * FROM feedback ORDER BY created DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(zip([d[0] for d in c.description], row)) for row in rows]

def update_feedback_status(feedback_id: int, status: str) -> bool:
    """Update feedback status."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE feedback SET status=? WHERE id=?", (status, feedback_id))
    conn.commit()
    conn.close()
    return True

# --- COMMAND AUDIT FUNCTIONS ---

def log_command_audit(telegram_user_id: str, command: str, full_message: str = None,
                     is_registered_user: bool = False, user_email: str = None,
                     success: bool = True, error_message: str = None,
                     response_time_ms: int = None) -> int:
    """Log a command audit entry. Returns audit id."""
    created = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO command_audit
                 (telegram_user_id, command, full_message, is_registered_user, user_email,
                  success, error_message, response_time_ms, created)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (telegram_user_id, command, full_message, 1 if is_registered_user else 0,
               user_email, 1 if success else 0, error_message, response_time_ms, created))
    audit_id = c.lastrowid
    conn.commit()
    conn.close()
    return audit_id

def get_user_command_history(telegram_user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get command history for a specific user."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""SELECT id, command, full_message, is_registered_user, user_email,
                        success, error_message, response_time_ms, created
                 FROM command_audit
                 WHERE telegram_user_id=?
                 ORDER BY created DESC
                 LIMIT ?""", (telegram_user_id, limit))
    rows = c.fetchall()
    conn.close()
    return [
        {
            "id": row[0],
            "command": row[1],
            "full_message": row[2],
            "is_registered_user": bool(row[3]),
            "user_email": row[4],
            "success": bool(row[5]),
            "error_message": row[6],
            "response_time_ms": row[7],
            "created": row[8]
        }
        for row in rows
    ]

def get_all_command_audit(limit: int = 100, offset: int = 0,
                         user_id: str = None, command: str = None,
                         success_only: bool = None,
                         start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
    """Get all command audit entries with filtering options."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Build query with filters
    query = """SELECT id, telegram_user_id, command, full_message, is_registered_user,
                      user_email, success, error_message, response_time_ms, created
               FROM command_audit WHERE 1=1"""
    params = []

    if user_id:
        query += " AND telegram_user_id=?"
        params.append(user_id)

    if command:
        query += " AND command LIKE ?"
        params.append(f"%{command}%")

    if success_only is not None:
        query += " AND success=?"
        params.append(1 if success_only else 0)

    if start_date:
        query += " AND created >= ?"
        params.append(start_date)

    if end_date:
        query += " AND created <= ?"
        params.append(end_date)

    query += " ORDER BY created DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    c.execute(query, params)
    rows = c.fetchall()
    conn.close()

    return [
        {
            "id": row[0],
            "telegram_user_id": row[1],
            "command": row[2],
            "full_message": row[3],
            "is_registered_user": bool(row[4]),
            "user_email": row[5],
            "success": bool(row[6]),
            "error_message": row[7],
            "response_time_ms": row[8],
            "created": row[9]
        }
        for row in rows
    ]

def get_command_audit_stats() -> Dict[str, Any]:
    """Get command audit statistics."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Total commands
    c.execute("SELECT COUNT(*) FROM command_audit")
    total_commands = c.fetchone()[0]

    # Successful commands
    c.execute("SELECT COUNT(*) FROM command_audit WHERE success=1")
    successful_commands = c.fetchone()[0]

    # Failed commands
    c.execute("SELECT COUNT(*) FROM command_audit WHERE success=0")
    failed_commands = c.fetchone()[0]

    # Registered vs non-registered users
    c.execute("SELECT COUNT(DISTINCT telegram_user_id) FROM command_audit WHERE is_registered_user=1")
    registered_users = c.fetchone()[0]

    c.execute("SELECT COUNT(DISTINCT telegram_user_id) FROM command_audit WHERE is_registered_user=0")
    non_registered_users = c.fetchone()[0]

    # Most used commands
    c.execute("""SELECT command, COUNT(*) as count
                 FROM command_audit
                 GROUP BY command
                 ORDER BY count DESC
                 LIMIT 10""")
    top_commands = [{"command": row[0], "count": row[1]} for row in c.fetchall()]

    # Recent activity (last 24 hours)
    c.execute("""SELECT COUNT(*) FROM command_audit
                 WHERE created >= datetime('now', '-1 day')""")
    recent_activity = c.fetchone()[0]

    conn.close()

    return {
        "total_commands": total_commands,
        "successful_commands": successful_commands,
        "failed_commands": failed_commands,
        "registered_users": registered_users,
        "non_registered_users": non_registered_users,
        "top_commands": top_commands,
        "recent_activity_24h": recent_activity,
        "success_rate": (successful_commands / total_commands * 100) if total_commands > 0 else 0
    }

def get_unique_users_command_history() -> List[Dict[str, Any]]:
    """Get list of unique users with their command statistics."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""SELECT telegram_user_id,
                        COUNT(*) as total_commands,
                        COUNT(CASE WHEN success=1 THEN 1 END) as successful_commands,
                        COUNT(CASE WHEN is_registered_user=1 THEN 1 END) as registered_commands,
                        MAX(created) as last_command,
                        MIN(created) as first_command
                 FROM command_audit
                 GROUP BY telegram_user_id
                 ORDER BY total_commands DESC""")

    rows = c.fetchall()
    conn.close()

    return [
        {
            "telegram_user_id": row[0],
            "total_commands": row[1],
            "successful_commands": row[2],
            "registered_commands": row[3],
            "last_command": row[4],
            "first_command": row[5],
            "success_rate": (row[2] / row[1] * 100) if row[1] > 0 else 0
        }
        for row in rows
    ]

def get_admin_user_ids() -> List[str]:
    """Get all admin user IDs from the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT telegram_user_id FROM users WHERE is_admin=1")
        admin_ids = [row[0] for row in c.fetchall()]
        conn.close()
        return admin_ids
    except Exception as e:
        _logger.error("Error getting admin user IDs: %s", e)
        return []
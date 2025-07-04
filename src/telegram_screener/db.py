import sqlite3
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

DB_PATH = "telegram_screener.sqlite3"

# --- DB Schema ---
# users: telegram_user_id TEXT PRIMARY KEY, email TEXT, verification_code TEXT, code_sent_time INTEGER, verified INTEGER, language TEXT, is_admin INTEGER DEFAULT 0
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
        created TEXT
    )''')
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
        created TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
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
    """Get user's email, verification status, language, and is_admin."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT email, verified, code_sent_time, language, is_admin FROM users WHERE telegram_user_id=?", (telegram_user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"email": row[0], "verified": bool(row[1]), "code_sent_time": row[2], "language": row[3], "is_admin": bool(row[4])}
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
def add_alert(user_id: str, ticker: str, price: float, condition: str) -> int:
    """Add a new alert. Returns alert id."""
    created = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO alerts (ticker, user_id, price, condition, active, created) VALUES (?, ?, ?, ?, 1, ?)",
              (ticker, user_id, price, condition, created))
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

# --- SCHEDULES CRUD ---
def add_schedule(user_id: str, ticker: str, scheduled_time: str, period: str = None, email: int = 0, indicators: str = None, interval: str = None, provider: str = None) -> int:
    """Add a new schedule. Returns schedule id."""
    created = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO schedules (ticker, user_id, scheduled_time, period, active, email, indicators, interval, provider, created) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?)",
              (ticker, user_id, scheduled_time, period, email, indicators, interval, provider, created))
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

# --- USER LISTING ---
def list_users() -> list:
    """Return list of (telegram_user_id, email) pairs."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT telegram_user_id, email FROM users")
    rows = c.fetchall()
    conn.close()
    return [(row[0], row[1]) for row in rows]
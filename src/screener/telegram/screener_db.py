import sqlite3
from pathlib import Path

DB_PATH = Path("db/screener.db")

# Ensure DB directory exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def init_db():
    """Initialize the screener database with required tables."""
    conn = get_conn()
    cur = conn.cursor()
    
    # Create users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_user_id TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create tickers table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tickers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            provider TEXT NOT NULL,
            ticker TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            UNIQUE(user_id, provider, ticker)
        )
    """)
    
    conn.commit()
    conn.close()

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_or_create_user(telegram_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO users (telegram_user_id) VALUES (?)", (str(telegram_id),))
    conn.commit()
    cur.execute("SELECT id FROM users WHERE telegram_user_id = ?", (str(telegram_id),))
    user_id = cur.fetchone()["id"]
    conn.close()
    return user_id

def add_ticker(telegram_id, provider, ticker):
    user_id = get_or_create_user(telegram_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO tickers (user_id, provider, ticker) VALUES (?, ?, ?)",
        (user_id, provider, ticker.upper())
    )
    conn.commit()
    conn.close()

def delete_ticker(telegram_id, provider, ticker):
    user_id = get_or_create_user(telegram_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM tickers WHERE user_id = ? AND provider = ? AND ticker = ?",
        (user_id, provider, ticker.upper())
    )
    conn.commit()
    conn.close()

def list_tickers(telegram_id, provider=None):
    user_id = get_or_create_user(telegram_id)
    conn = get_conn()
    cur = conn.cursor()
    if provider:
        cur.execute(
            "SELECT provider, ticker FROM tickers WHERE user_id = ? AND provider = ? ORDER BY provider, ticker",
            (user_id, provider)
        )
    else:
        cur.execute(
            "SELECT provider, ticker FROM tickers WHERE user_id = ? ORDER BY provider, ticker",
            (user_id,)
        )
    rows = cur.fetchall()
    conn.close()
    result = {}
    for row in rows:
        result.setdefault(row["provider"], []).append(row["ticker"])
    return result

def all_tickers_for_status(telegram_id, provider=None):
    """Returns a list of (provider, ticker) tuples for status analysis"""
    tickers_by_provider = list_tickers(telegram_id, provider)
    result = []
    for prov, tlist in tickers_by_provider.items():
        for ticker in tlist:
            result.append((prov, ticker))
    return result

def all_tickers_with_providers_for_status(telegram_id, provider=None):
    """Returns a list of (provider, ticker) tuples for status analysis with provider filter"""
    tickers_by_provider = list_tickers(telegram_id, provider)
    result = []
    for prov, tlist in tickers_by_provider.items():
        for ticker in tlist:
            result.append((prov, ticker))
    return result

def migrate_users_table():
    """Standalone migration: add email and verification fields to users table if not present."""
    conn = get_conn()
    cur = conn.cursor()
    # Add columns if they do not exist
    cur.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in cur.fetchall()]
    if 'email' not in columns:
        cur.execute("ALTER TABLE users ADD COLUMN email TEXT")
    if 'email_verification_sent' not in columns:
        cur.execute("ALTER TABLE users ADD COLUMN email_verification_sent DATETIME")
    if 'email_verification_received' not in columns:
        cur.execute("ALTER TABLE users ADD COLUMN email_verification_received DATETIME")
    if 'email_verification_code' not in columns:
        cur.execute("ALTER TABLE users ADD COLUMN email_verification_code TEXT")
    conn.commit()
    conn.close()

# User email/verification management

def set_user_email(telegram_id, email, code, sent_time):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE users SET email=?, email_verification_code=?, email_verification_sent=?, email_verification_received=NULL WHERE telegram_user_id=?", (email, code, sent_time, str(telegram_id)))
    conn.commit()
    conn.close()

def get_user_email(telegram_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT email FROM users WHERE telegram_user_id=?", (str(telegram_id),))
    row = cur.fetchone()
    conn.close()
    return row["email"] if row else None

def get_user_verification_status(telegram_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT email, email_verification_sent, email_verification_received FROM users WHERE telegram_user_id=?", (str(telegram_id),))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "email": row["email"],
        "verification_sent": row["email_verification_sent"],
        "verification_received": row["email_verification_received"]
    }

def get_user_verification_code(telegram_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT email_verification_code, email_verification_sent FROM users WHERE telegram_user_id=?", (str(telegram_id),))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None, None
    return row["email_verification_code"], row["email_verification_sent"]

def set_user_verified(telegram_id, received_time):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE users SET email_verification_received=? WHERE telegram_user_id=?", (received_time, str(telegram_id)))
    conn.commit()
    conn.close()

# Initialize database when module is imported
init_db()

if __name__ == "__main__":
    print("Running users table migration...")
    migrate_users_table()
    print("Migration complete.") 
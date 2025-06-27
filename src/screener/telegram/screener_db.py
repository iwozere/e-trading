import sqlite3
from pathlib import Path

DB_PATH = Path("db/screener.db")

# Ensure DB directory exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

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
    tickers_by_provider = list_tickers(telegram_id, provider)
    tickers = []
    for tlist in tickers_by_provider.values():
        tickers.extend(tlist)
    return tickers

def all_tickers_with_providers_for_status(telegram_id, provider=None):
    tickers_by_provider = list_tickers(telegram_id, provider)
    result = []
    for prov, tlist in tickers_by_provider.items():
        for ticker in tlist:
            result.append((prov, ticker))
    return result 
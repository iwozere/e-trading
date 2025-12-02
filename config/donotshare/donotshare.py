import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(dotenv_path="config/donotshare/.env")


# Data cache directory
DATA_CACHE_DIR = "c:/data-cache"

# Database configuration
#DB_PATH = "db/trading.db"  # Keep for backward compatibility

# PostgreSQL configuration
POSTGRES_HOST = "localhost"  # Add host configuration
POSTGRES_PORT = 5432
POSTGRES_USER = os.getenv("POSTGRES_USER", "trading_admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE", "trading") # Add database name

REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Database URL - PostgreSQL connection string
DB_URL = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DATABASE}"

TEST_DB_URL="postgresql+psycopg2://test_user:test_password@localhost:5432/e_trading_test"

# 0 - off, 1 - log SQL queries into the log files
SQL_ECHO = "0"

TRADING_WEBGUI_PORT=5002
TRADING_API_PORT=5003

##################################################################
#
# TRADING BROKERS FOR CANDLES ETC.
#
##################################################################

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

BINANCE_KEY = os.getenv("BINANCE_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET")

BINANCE_PAPER_KEY = os.getenv("BINANCE_PAPER_KEY")
BINANCE_PAPER_SECRET = os.getenv("BINANCE_PAPER_SECRET")

BINANCE_FUTURES_PAPER_KEY=os.getenv("BINANCE_FUTURES_PAPER_KEY")
BINANCE_FUTURES_PAPER_SECRET=os.getenv("BINANCE_FUTURES_PAPER_SECRET")


IBKR_HOST = os.getenv("IBKR_HOST")
IBKR_PORT = os.getenv("IBKR_PORT")
IBKR_CLIENT_ID = os.getenv("IBKR_CLIENT_ID")

IBKR_KEY = os.getenv("IBKR_KEY")
IBKR_SECRET = os.getenv("IBKR_SECRET")

IBKR_PAPER_KEY = os.getenv("IBKR_PAPER_KEY")
IBKR_PAPER_SECRET = os.getenv("IBKR_PAPER_SECRET")
IBKR_PAPER_PORT = os.getenv("IBKR_PAPER_PORT")
IBKR_PAPER_CLIENT_ID = os.getenv("IBKR_PAPER_CLIENT_ID")


ALPHA_VANTAGE_KEY=os.getenv("ALPHA_VANTAGE_KEY")
FINNHUB_KEY=os.getenv("FINNHUB_KEY")
TWELVE_DATA_KEY=os.getenv("TWELVE_DATA_KEY")
POLYGON_KEY=os.getenv("POLYGON_KEY")
FMP_API_KEY=os.getenv("FMP_API_KEY")
TIINGO_API_KEY=os.getenv("TIINGO_API_KEY")
FINRA_API_CLIENT=os.getenv("FINRA_API_CLIENT")
FINRA_API_SECRET=os.getenv("FINRA_API_SECRET")

##################################################################
#
# NOTIFICATION SETTINGS
#
##################################################################
SMTP_SERVER=os.getenv("SMTP_SERVER")
SMTP_PORT=os.getenv("SMTP_PORT")
SMTP_USER=os.getenv("SMTP_USER")
SMTP_PASSWORD=os.getenv("SMTP_PASSWORD")


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# DEPRECATED: No longer needed - admin notifications now use HTTP API to send to admin users
# This can be removed from environment variables
#TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


##################################################################
#
# APPLICATION SETTINGS
#
##################################################################

WEBGUI_LOGIN = os.getenv("WEBGUI_LOGIN")
WEBGUI_PASSWORD = os.getenv("WEBGUI_PASSWORD")

API_LOGIN = os.getenv("API_LOGIN")
API_PASSWORD = os.getenv("API_PASSWORD")

# Admin credentials for Flask-Login
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")


##################################################################
#
# SENTIMENT SOURCES
#
##################################################################

TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
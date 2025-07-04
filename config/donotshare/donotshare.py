import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dotenv import load_dotenv

load_dotenv(dotenv_path="config/donotshare/.env")


BINANCE_KEY = os.getenv("BINANCE_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET")

BINANCE_PAPER_KEY = os.getenv("BINANCE_PAPER_KEY")
BINANCE_PAPER_SECRET = os.getenv("BINANCE_PAPER_SECRET")

IBKR_HOST = os.getenv("IBKR_HOST")
IBKR_PORT = os.getenv("IBKR_PORT")
IBKR_CLIENT_ID = os.getenv("IBKR_CLIENT_ID")
IBKR_PAPER_PORT = os.getenv("IBKR_PAPER_PORT")

IBKR_KEY = os.getenv("IBKR_KEY")
IBKR_SECRET = os.getenv("IBKR_SECRET")

IBKR_PAPER_KEY = os.getenv("IBKR_PAPER_KEY")
IBKR_PAPER_SECRET = os.getenv("IBKR_PAPER_SECRET")


ALPHA_VANTAGE_KEY=os.getenv("ALPHA_VANTAGE_KEY")
FINNHUB_KEY=os.getenv("FINNHUB_KEY")
TWELVE_DATA_KEY=os.getenv("TWELVE_DATA_KEY")
POLYGON_KEY=os.getenv("POLYGON_KEY")


#gmail_password = os.getenv("gmail_password")
#gmail_username = os.getenv("gmail_username")
#SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")

SMTP_SERVER=os.getenv("SMTP_SERVER")
SMTP_PORT=os.getenv("SMTP_PORT")
SMTP_USER=os.getenv("SMTP_USER")
SMTP_PASSWORD=os.getenv("SMTP_PASSWORD")


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

WEBGUI_PORT = os.getenv("WEBGUI_PORT")
WEBGUI_LOGIN = os.getenv("WEBGUI_LOGIN")
WEBGUI_PASSWORD = os.getenv("WEBGUI_PASSWORD")

API_PORT = os.getenv("API_PORT")
API_LOGIN = os.getenv("API_LOGIN")
API_PASSWORD = os.getenv("API_PASSWORD")

# Admin credentials for Flask-Login
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

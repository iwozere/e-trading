import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import requests
import yfinance as yf
from config.donotshare.donotshare import ALPHA_VANTAGE_KEY, FINNHUB_KEY, TWELVE_DATA_KEY, POLYGON_KEY

ALPHA_VANTAGE_KEY=os.getenv("ALPHA_VANTAGE_KEY")
FINNHUB_KEY=os.getenv("FINNHUB_KEY")
TWELVE_DATA_KEY=os.getenv("TWELVE_DATA_KEY")
POLYGON_KEY=os.getenv("POLYGON_KEY")


# OPTIONAL: IBKR setup
try:
    from ib_insync import IB, Stock
    ib = IB()
except ImportError:
    ib = None

# Your API keys
API_KEYS = {
    "alpha_vantage": ALPHA_VANTAGE_KEY,
    "finnhub": FINNHUB_KEY,
    "twelve_data": TWELVE_DATA_KEY,
    "polygon": POLYGON_KEY
}

symbol = "AAPL"  # You can dynamically change this
results = {}

# --- Yahoo Finance (yfinance) ---
yf_ticker = yf.Ticker(symbol)
yf_info = yf_ticker.info

results["yfinance"] = {
    "PE": yf_info.get("trailingPE"),
    "PB": yf_info.get("priceToBook"),
    "ROE": yf_info.get("returnOnEquity"),
    "Sector": yf_info.get("sector"),
}

# --- Alpha Vantage ---
def get_alpha_vantage(symbol):
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={API_KEYS['alpha_vantage']}"
    r = requests.get(url)
    return r.json()

alpha = get_alpha_vantage(symbol)
print("alpha")
print(alpha)
results["alpha_vantage"] = {
    "PE": alpha.get("PERatio"),
    "ROE": alpha.get("ReturnOnEquityTTM"),
    "MarketCap": alpha.get("MarketCapitalization")
}

# --- Finnhub ---
def get_finnhub(symbol):
    url = f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={API_KEYS['finnhub']}"
    return requests.get(url).json()

finn = get_finnhub(symbol)
print("finn")
print(finn)
results["finnhub"] = {
    "PE": finn["metric"].get("peNormalizedAnnual"),
    "ROE": finn["metric"].get("roe"),
    "EPS": finn["metric"].get("epsTTM")
}

# --- Twelve Data (EPS only) ---
def get_twelve_data(symbol):
    url = f"https://api.twelvedata.com/earnings?symbol={symbol}&apikey={API_KEYS['twelve_data']}"
    data = requests.get(url).json()
    return data.get("earnings", [{}])[0]

twelve = get_twelve_data(symbol)
print("twelve")
print(twelve)
results["twelve_data"] = {
    "EPS": twelve.get("eps"),
    "Revenue": twelve.get("revenue"),
    "Date": twelve.get("date")
}

# --- Polygon.io ---
def get_polygon(symbol):
    url = f"https://api.polygon.io/v3/reference/tickers/{symbol}?apiKey={API_KEYS['polygon']}"
    r = requests.get(url)
    return r.json()

polygon = get_polygon(symbol)
print("polygon")
print(polygon)
results["polygon"] = {
    "MarketCap": polygon.get("results", {}).get("market_cap"),
    "Name": polygon.get("results", {}).get("name")
}

# --- IBKR (optional) ---
results["ibkr"] = {}
print("ibkr")
if ib:
    try:
        ib.connect("127.0.0.1", 7497, clientId=1)
        contract = Stock(symbol, "SMART", "USD")
        report = ib.reqFundamentalData(contract, "ReportSnapshot")
        if report:
            results["ibkr"]["raw_xml"] = report
        ib.disconnect()
    except Exception as e:
        results["ibkr"]["error"] = str(e)

# --- Final Output ---
from pprint import pprint
pprint(results)

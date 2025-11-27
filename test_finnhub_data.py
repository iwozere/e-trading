"""Quick test to see what Finnhub returns for a sample ticker"""
import requests
from config.donotshare.donotshare import FINNHUB_KEY

# Test with a known liquid ticker
ticker = "AAPL"

# Test profile2 API
profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={FINNHUB_KEY}"
profile_response = requests.get(profile_url, timeout=10)
print(f"\n=== PROFILE2 API for {ticker} ===")
print(f"Status: {profile_response.status_code}")
if profile_response.status_code == 200:
    profile_data = profile_response.json()
    print(f"Market Cap: {profile_data.get('marketCapitalization')}")
    print(f"Shares Outstanding: {profile_data.get('shareOutstanding')}")
    print(f"Full response: {profile_data}")

# Test quote API
quote_url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_KEY}"
quote_response = requests.get(quote_url, timeout=10)
print(f"\n=== QUOTE API for {ticker} ===")
print(f"Status: {quote_response.status_code}")
if quote_response.status_code == 200:
    quote_data = quote_response.json()
    print(f"Volume: {quote_data.get('volume')}")
    print(f"Full response: {quote_data}")

# Test a small cap ticker
small_ticker = "WISH"
profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={small_ticker}&token={FINNHUB_KEY}"
profile_response = requests.get(profile_url, timeout=10)
print(f"\n=== PROFILE2 API for {small_ticker} ===")
print(f"Status: {profile_response.status_code}")
if profile_response.status_code == 200:
    profile_data = profile_response.json()
    print(f"Market Cap: {profile_data.get('marketCapitalization')}")
    print(f"Shares Outstanding: {profile_data.get('shareOutstanding')}")
    print(f"Full response: {profile_data}")

quote_url = f"https://finnhub.io/api/v1/quote?symbol={small_ticker}&token={FINNHUB_KEY}"
quote_response = requests.get(quote_url, timeout=10)
print(f"\n=== QUOTE API for {small_ticker} ===")
print(f"Status: {quote_response.status_code}")
if quote_response.status_code == 200:
    quote_data = quote_response.json()
    print(f"Volume: {quote_data.get('volume')}")
    print(f"Full response: {quote_data}")

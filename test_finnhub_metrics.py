"""Test Finnhub metrics API for volume data"""
import requests
from config.donotshare.donotshare import FINNHUB_KEY

tickers = ["AAPL", "TSLA", "AMD"]

for ticker in tickers:
    # Test metrics2 API
    metrics_url = f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={FINNHUB_KEY}"
    response = requests.get(metrics_url, timeout=10)

    print(f"\n=== METRICS API for {ticker} ===")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()

        # Check for volume-related fields
        if 'metric' in data:
            metric = data['metric']
            print(f"\nVolume-related fields:")
            for key in metric.keys():
                if 'volume' in key.lower() or 'avg' in key.lower():
                    print(f"  {key}: {metric[key]}")

            print(f"\nFloat/Shares fields:")
            for key in metric.keys():
                if 'float' in key.lower() or 'share' in key.lower():
                    print(f"  {key}: {metric[key]}")

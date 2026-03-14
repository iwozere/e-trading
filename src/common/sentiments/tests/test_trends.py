import asyncio
import aiohttp
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.common.sentiments.adapters.async_trends import AsyncTrendsAdapter
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

async def test_single_ticker(ticker="AAPL"):
    print(f"Testing Google Trends adapter for: {ticker}")

    # Use a fresh session to avoid cookie pollution
    async with aiohttp.ClientSession() as session:
        adapter = AsyncTrendsAdapter(session=session)

        try:
            print("--- Establishing Cookies ---")
            success = await adapter._ensure_cookies()
            print(f"Cookie fetch: {'SUCCESS' if success else 'FAILURE'}")
            if success:
                import yarl
                url = yarl.URL('https://trends.google.com')
                cookies = session.cookie_jar.filter_cookies(url)
                print(f"Cookies: {cookies}")

            print("--- Fetching Tokens ---")
            # Intercept the call to see params
            url = "https://trends.google.com/trends/api/explore"
            region = 'US'
            timeframe = 'today 7-d'
            params = {
                'hl': 'en-US',
                'tz': 0, # Use integer
                'req': json.dumps({
                    'comparisonItem': [{'keyword': ticker.upper(), 'geo': region, 'time': timeframe}],
                    'category': 0,
                    'property': ''
                }, separators=(',', ':'))
            }
            print(f"Target URL: {url}")
            print(f"Params: {params}")

            token_success = await adapter._fetch_tokens(ticker)
            print(f"Token fetch results: {'SUCCESS' if token_success else 'FAILURE'}")

            if token_success:
                print(f"Extracted Tokens: {adapter._tokens}")
                print("--- Fetching Summary ---")
                summary = await adapter.fetch_summary(ticker)
                print(f"Summary Result: {summary}")

        except Exception as e:
            print(f"ERROR: Exception occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await adapter.close()

if __name__ == "__main__":
    asyncio.run(test_single_ticker())

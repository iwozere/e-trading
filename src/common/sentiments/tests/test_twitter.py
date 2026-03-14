import asyncio
import aiohttp
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.common.sentiments.adapters.async_twitter import AsyncTwitterAdapter
import config.donotshare.donotshare as secrets
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

async def test_twitter(ticker="AAPL"):
    print(f"Testing Twitter adapter for: {ticker}")
    adapter = AsyncTwitterAdapter()
    if not adapter.bearer_token:
        print("Twitter bearer token NOT found in env/config!")
        return

    try:
        print("--- Fetching Messages ---")
        # We'll override the _get_with_retry slightly in logs or just observe the existing logs
        # But let's actually print headers by monkeypatching or just relying on internal logs if we add them.

        start = time.time()
        messages = await adapter.fetch_messages(ticker, limit=10)
        end = time.time()

        print(f"Request took {end - start:.2f} seconds")
        print(f"Fetched {len(messages)} messages")
        for m in messages:
            print(f"- {m.get('body')[:100]}...")

        print("--- Fetching Summary ---")
        summary = await adapter.fetch_summary(ticker)
        print(f"Summary: {summary}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await adapter.close()

if __name__ == "__main__":
    asyncio.run(test_twitter())

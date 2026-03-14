import asyncio
import aiohttp
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.common.sentiments.adapters.async_reddit import AsyncRedditAdapter
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

async def test_reddit(ticker="AAPL"):
    print(f"Testing Reddit adapter for: {ticker}")
    adapter = AsyncRedditAdapter()
    if not adapter.enabled:
        print("Reddit adapter is NOT enabled (check credentials)")
        return

    try:
        print("--- Ensuring Token ---")
        await adapter._ensure_token()
        print(f"Token: {adapter._token[:10]}... (expires in {adapter._token_expiry - asyncio.get_event_loop().time()} seconds)")

        print("--- Fetching Messages ---")
        messages = await adapter.fetch_messages(ticker, limit=5)
        print(f"Fetched {len(messages)} messages")
        for m in messages:
            print(f"- [{m.get('type')}] {m.get('body')[:100]}...")

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
    asyncio.run(test_reddit())

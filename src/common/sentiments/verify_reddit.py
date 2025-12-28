import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.common.sentiments.adapters.async_reddit import AsyncRedditAdapter

async def test_reddit():
    adapter = AsyncRedditAdapter()
    ticker = "NVDA"
    print(f"Testing Reddit adapter for {ticker}...")

    try:
        summary = await adapter.fetch_summary(ticker)
        print(f"Summary: {summary}")

        messages = await adapter.fetch_messages(ticker, limit=5)
        print(f"Fetched {len(messages)} messages.")
        for msg in messages:
            print(f"- [{msg['type']}] {msg['user']['username']}: {msg['body'][:100]}...")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await adapter.close()

if __name__ == "__main__":
    asyncio.run(test_reddit())

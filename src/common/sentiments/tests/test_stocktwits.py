import asyncio
import aiohttp
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.common.sentiments.adapters.async_stocktwits import AsyncStocktwitsAdapter

async def test_stocktwits(ticker="AAPL"):
    print(f"Testing StockTwits for: {ticker}")
    adapter = AsyncStocktwitsAdapter()
    try:
        messages = await adapter.fetch_messages(ticker, limit=5)
        print(f"Fetched {len(messages)} messages")
        for m in messages:
            print(f"- {m.get('body')[:100]}...")

        summary = await adapter.fetch_summary(ticker)
        print(f"Summary: {summary}")
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        await adapter.close()

if __name__ == "__main__":
    asyncio.run(test_stocktwits("TSLA")) # Trying TSLA as well
    asyncio.run(test_stocktwits("AAPL"))

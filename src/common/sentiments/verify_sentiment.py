
import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader.finnhub_data_downloader import FinnhubDataDownloader
from src.data.downloader.alpha_vantage_data_downloader import AlphaVantageDataDownloader
from src.data.downloader.santiment_data_downloader import SantimentDataDownloader
from src.common.sentiments.adapters.async_news import AsyncNewsAdapter

async def verify_downloaders():
    print("\n=== Verifying Sentiment Downloaders ===\n")

    ticker = "AAPL"

    # 1. Finnhub
    print(f"--- Finnhub ({ticker}) ---")
    fh = FinnhubDataDownloader()
    if fh.api_key:
        try:
            news_sentiment = await fh.get_news_sentiment(ticker)
            print(f"News Sentiment Score: {news_sentiment.sentiment_score if news_sentiment else 'None'}")

            social_sentiment = await fh.get_social_sentiment(ticker)
            print(f"Social Mentions: {social_sentiment.mention_count if social_sentiment else 'None'}")
        except Exception as e:
            print(f"Finnhub Error: {e}")
    else:
        print("Skipping Finnhub (No API Key)")

    # 2. Alpha Vantage
    print(f"\n--- Alpha Vantage ({ticker}) ---")
    av = AlphaVantageDataDownloader()
    if av.api_key:
        try:
            news_data = await av.get_news_sentiment(ticker, limit=5)
            print(f"Article Count: {news_data.article_count if news_data else 'None'}")
            print(f"Sentiment Score: {news_data.sentiment_score if news_data else 'None'}")
        except Exception as e:
            print(f"Alpha Vantage Error: {e}")
    else:
        print("Skipping Alpha Vantage (No API Key)")

    # 3. Santiment
    print(f"\n--- Santiment (BTC) ---")
    san = SantimentDataDownloader()
    try:
        social = await san.get_social_volume("BTC")
        print(f"Social Volume: {social.mention_count if social else 'None'}")
        if social and social.raw_data and "error" in social.raw_data:
             print(f"Warning: {social.raw_data['error']}")
    except Exception as e:
        print(f"Santiment Error: {e}")

async def verify_adapter():
    print("\n=== Verifying AsyncNewsAdapter ===\n")

    adapter = AsyncNewsAdapter()
    try:
        summary = await adapter.fetch_summary("AAPL")
        print(f"Sentiment Summary for AAPL:")
        print(f"  Mentions: {summary.get('mentions')}")
        print(f"  Score: {summary.get('sentiment_score')}")
        print(f"  Bullish/Bearish: {summary.get('bullish')}/{summary.get('bearish')}")
        print(f"  Credibility: {summary.get('avg_credibility'):.2f}")
    except Exception as e:
        print(f"Adapter Error: {e}")
    finally:
        await adapter.close()

async def main():
    await verify_downloaders()
    await verify_adapter()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

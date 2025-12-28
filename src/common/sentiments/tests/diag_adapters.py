import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.common.sentiments.adapters.adapter_manager import register_default_adapters, get_adapter_manager

async def diagnose():
    register_default_adapters()
    manager = get_adapter_manager()
    # Disable global rate limiting for this diagnostic
    manager._global_coordinator = None

    providers = ["stocktwits", "reddit", "twitter", "news", "discord"]
    ticker = "AAPL"
    since_ts = int((datetime.now(timezone.utc) - timedelta(hours=24)).timestamp())

    print(f"--- Diagnosing Adapters for {ticker} (since {since_ts}) ---")

    for provider in providers:
        print(f"\nChecking {provider}...")
        try:
            # We need to add individual adapters
            manager.add_adapter(provider)
            summary = await manager.fetch_summary_from_adapter(provider, ticker, since_ts)
            if summary:
                print(f"SUCCESS: {provider} found {summary.get('mentions', 0)} mentions. Score: {summary.get('sentiment_score', 0.0)}")
            else:
                print(f"FAILURE: {provider} returned None (check logs)")
        except Exception as e:
            print(f"ERROR: {provider} failed with: {e}")

if __name__ == "__main__":
    asyncio.run(diagnose())

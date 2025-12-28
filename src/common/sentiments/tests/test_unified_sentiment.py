import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.common.sentiments.collect_sentiment_async import collect_sentiment_batch

class TestUnifiedSentiment(unittest.IsolatedAsyncioTestCase):
    async def test_collect_all_providers(self):
        # Mocks for all adapters
        mock_summaries = {
            "stocktwits": {"mentions": 10, "sentiment_score": 0.5, "provider": "stocktwits"},
            "reddit": {"mentions": 20, "unique_authors": 15, "sentiment_score": 0.2, "provider": "reddit"},
            "news": {"mentions": 5, "sentiment_score": 0.8, "provider": "news"},
            "trends": {"mentions": 1, "sentiment_score": 0.0, "provider": "trends"},
            "discord": {"mentions": 50, "sentiment_score": -0.1, "provider": "discord"},
            "twitter": {"mentions": 100, "sentiment_score": 0.3, "provider": "twitter"}
        }

        # Configuration for all providers enabled
        config = {
            "providers": {
                "stocktwits": True,
                "reddit": True,
                "news": True,
                "trends": True,
                "discord": True,
                "twitter": True,
                "hf_enabled": False
            },
            "weights": {
                "stocktwits": 1.0,
                "reddit": 1.0,
                "news": 1.0,
                "trends": 1.0,
                "discord": 1.0,
                "twitter": 1.0
            }
        }

        # Mock adapter manager
        with patch('src.common.sentiments.adapters.adapter_manager.get_adapter_manager') as mock_get_manager:
            manager = MagicMock()
            mock_get_manager.return_value = manager

            manager.get_available_adapters.return_value = list(mock_summaries.keys())
            manager.fetch_summary_from_adapter = AsyncMock()
            manager.fetch_summary_from_adapter.side_effect = lambda p, t, s: mock_summaries.get(p)
            manager.close_all = AsyncMock()

            # Call collection
            results = await collect_sentiment_batch(["AAPL"], config=config)

            # Verify results
            self.assertIn("AAPL", results)
            aapl = results["AAPL"]
            self.assertIsNotNone(aapl)

            # Check mention aggregation (10 + 20 + 5 + 1 + 50 + 100 = 186)
            self.assertEqual(aapl.mentions_24h, 186)

            # Check unique authors (only Reddit provides it in this mock)
            self.assertEqual(aapl.unique_authors_24h, 15)

            # Check sentiment score (Average of 0.5, 0.2, 0.8, 0.0, -0.1, 0.3 = 1.7 / 6 = 0.28333)
            self.assertAlmostEqual(aapl.sentiment_score_24h, 1.7 / 6, places=5)

            # Check data quality
            for provider in mock_summaries.keys():
                self.assertEqual(aapl.data_quality[provider], "ok")

            # Check raw payload
            for provider in mock_summaries.keys():
                self.assertIn(provider, aapl.raw_payload)

    async def test_hf_integration(self):
        # Mock summary for aggregation
        mock_summary = {"mentions": 25, "sentiment_score": 0.0, "provider": "stocktwits"}

        # Mock messages for HF
        mock_messages = [
            {"body": "AAPL is great", "likes": 10, "replies": 2, "provider": "stocktwits"},
            {"body": "Short AAPL", "likes": 0, "replies": 5, "provider": "stocktwits"}
        ]

        config = {
            "providers": {
                "stocktwits": True,
                "hf_enabled": True
            },
            "min_mentions_for_hf": 10,
            "weights": {"heuristic_vs_hf": 1.0} # Use only HF score
        }

        with patch('src.common.sentiments.adapters.adapter_manager.get_adapter_manager') as mock_get_manager:
            manager = MagicMock()
            mock_get_manager.return_value = manager
            manager.get_available_adapters.return_value = ["stocktwits", "huggingface"]
            manager.fetch_summary_from_adapter = AsyncMock(return_value=mock_summary)
            manager.fetch_messages_from_adapter = AsyncMock(return_value=mock_messages)
            manager.close_all = AsyncMock()

            # Mock HF adapter
            hf_adapter = AsyncMock()
            manager._adapters = {"huggingface": hf_adapter}
            hf_adapter.predict_batch.return_value = [
                {"label": "POSITIVE", "score": 0.9},
                {"label": "NEGATIVE", "score": 0.8}
            ]

            results = await collect_sentiment_batch(["AAPL"], config=config)

            aapl = results["AAPL"]
            self.assertIsNotNone(aapl)
            self.assertEqual(aapl.data_quality["huggingface"], "ok")

            # Weighted average calculation:
            # Msg 1: POS (1.0). Engagement: 10 + 2*2 = 14. Weight: sqrt(15) = 3.87
            # Msg 2: NEG (-1.0). Engagement: 0 + 2*5 = 10. Weight: sqrt(11) = 3.31
            # Combined score should be slightly positive due to higher engagement on POS message.
            self.assertTrue(aapl.sentiment_score_24h > 0)

if __name__ == "__main__":
    unittest.main()


import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from datetime import datetime
import pandas as pd
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Mock sanpy before importing downloader
sys.modules['san'] = MagicMock()
import san

from src.data.downloader.santiment_data_downloader import SantimentDataDownloader
from src.model.schemas import SentimentData

class TestSantimentDownloader(unittest.TestCase):
    def setUp(self):
        self.downloader = SantimentDataDownloader()
        self.downloader.api_key = "test_key"

    @patch('src.data.downloader.santiment_data_downloader.SANPY_AVAILABLE', True)
    def test_get_social_volume(self):
        # Mock san.get response
        mock_df = pd.DataFrame({
            'value': [100, 200, 150]
        }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))

        # We need to mock the blocking call run_in_executor
        async def run_test():
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_df)

                result = await self.downloader.get_social_volume("BTC")

                self.assertIsInstance(result, SentimentData)
                self.assertEqual(result.mention_count, 450)
                self.assertEqual(result.provider, 'santiment')
                self.assertEqual(result.symbol, 'BTC')

        asyncio.run(run_test())

    @patch('src.data.downloader.santiment_data_downloader.SANPY_AVAILABLE', True)
    def test_get_sentiment_metrics(self):
        # Mock responses
        pos_df = pd.DataFrame({'value': [100]})
        neg_df = pd.DataFrame({'value': [50]})

        async def run_test():
            with patch('asyncio.get_event_loop') as mock_loop:
                 # The method calls run_in_executor once to return (pos, neg)
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=(pos_df, neg_df))

                result = await self.downloader.get_sentiment_metrics("BTC")

                self.assertIsInstance(result, SentimentData)
                self.assertEqual(result.sentiment_score, (100-50)/(100+50)) # 0.333...
                self.assertEqual(result.bullish_score, 100)
                self.assertEqual(result.bearish_score, 50)

        asyncio.run(run_test())

    @patch('src.data.downloader.santiment_data_downloader.SANPY_AVAILABLE', False)
    def test_missing_sanpy(self):
        async def run_test():
            result = await self.downloader.get_social_volume("BTC")
            self.assertIn("error", result.raw_data)
            self.assertEqual(result.raw_data["error"], "sanpy not installed")

        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()

import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import aiohttp
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.common.sentiments.adapters.async_reddit import AsyncRedditAdapter

class TestAsyncRedditAdapter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.adapter = AsyncRedditAdapter(name="test_reddit")
        self.adapter.client_id = "test_id"
        self.adapter.client_secret = "test_secret"

    async def asyncTearDown(self):
        await self.adapter.close()

    @patch('aiohttp.ClientSession.post')
    async def test_ensure_token(self, mock_post):
        # Mock token response
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = {
            "access_token": "fake_token",
            "expires_in": 3600
        }
        mock_post.return_value.__aenter__.return_value = mock_resp

        await self.adapter._ensure_token()
        self.assertEqual(self.adapter._token, "fake_token")
        self.assertTrue(self.adapter._token_expiry > 0)

    @patch('src.common.sentiments.adapters.async_reddit.AsyncRedditAdapter._request_with_retry')
    async def test_fetch_messages(self, mock_request):
        # Mock search response
        mock_request.return_value = {
            "data": {
                "children": [
                    {
                        "data": {
                            "name": "t3_123",
                            "title": "NVDA to the moon 🚀",
                            "selftext": "Thinking of buying more.",
                            "created_utc": 1672531200,
                            "author": "trader1",
                            "author_fullname": "t2_abc",
                            "score": 150,
                            "num_comments": 25,
                            "permalink": "/r/stocks/comments/123"
                        }
                    }
                ]
            }
        }

        messages = await self.adapter.fetch_messages("NVDA")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["id"], "t3_123")
        self.assertIn("NVDA", messages[0]["body"])
        self.assertEqual(messages[0]["likes"], 150)
        self.assertEqual(messages[0]["type"], "submission")

    @patch('src.common.sentiments.adapters.async_reddit.AsyncRedditAdapter.fetch_messages')
    async def test_fetch_summary(self, mock_fetch):
        # Mock messages for summary
        mock_fetch.return_value = [
            {"body": "NVDA moon 🚀", "user": {"id": "u1"}, "likes": 10},
            {"body": "NVDA crash red 📉", "user": {"id": "u2"}, "likes": 5}
        ]

        summary = await self.adapter.fetch_summary("NVDA")
        self.assertEqual(summary["mentions"], 2)
        self.assertEqual(summary["pos"], 1)
        self.assertEqual(summary["neg"], 1)
        self.assertEqual(summary["sentiment_score"], 0.0)
        self.assertEqual(summary["unique_authors"], 2)

if __name__ == "__main__":
    unittest.main()

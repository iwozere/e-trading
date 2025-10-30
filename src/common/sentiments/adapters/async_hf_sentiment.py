# src/common/sentiments/adapters/async_hf_sentiment.py
"""
Async-friendly wrapper around HuggingFace transformers pipeline.

The real HF pipeline is synchronous (CPU/GPU). This wrapper runs HF inference
in a ThreadPoolExecutor to avoid blocking the asyncio loop.

Usage:
    model = AsyncHFSentiment(model_name="cardiffnlp/twitter-roberta-base-sentiment", device=-1)
    results = await model.predict_batch(texts)  # returns list of dicts
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Any
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.common.sentiments.adapters.base_adapter import BaseSentimentAdapter

_logger = setup_logger(__name__)

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

DEFAULT_MODEL = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment")

class AsyncHFSentiment(BaseSentimentAdapter):
    def __init__(self, name: str = "huggingface", model_name: Optional[str] = None,
                 device: int = -1, max_workers: int = 1, concurrency: int = 1, rate_limit_delay: float = 0.1):
        super().__init__(name, concurrency, rate_limit_delay)

        if not HF_AVAILABLE:
            raise RuntimeError("transformers is required for AsyncHFSentiment")

        self.model_name = model_name or DEFAULT_MODEL
        self.device = device  # -1 means CPU
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pipe = None
        self._initialization_error = None

        # Initialize pipeline in background to avoid blocking
        self._init_task = asyncio.create_task(self._initialize_pipeline())

    async def _initialize_pipeline(self) -> None:
        """Initialize the HuggingFace pipeline asynchronously."""
        try:
            _logger.info("Loading HF pipeline %s (device=%s)", self.model_name, self.device)

            # Run pipeline initialization in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self._pipe = await loop.run_in_executor(
                self._executor,
                lambda: pipeline("sentiment-analysis", model=self.model_name, tokenizer=self.model_name, device=self.device)
            )

            _logger.info("HF pipeline loaded successfully")
            self._update_health_success(0.0)  # Mark as healthy after successful initialization

        except Exception as e:
            self._initialization_error = e
            self._update_health_failure(e)
            _logger.error("Failed to initialize HF pipeline: %s", e)

    def _predict_blocking(self, texts: List[str]):
        """Blocking prediction function to run in thread pool."""
        try:
            if self._pipe is None:
                raise RuntimeError("Pipeline not initialized")

            # Validate inputs
            if not texts or not any(text.strip() for text in texts):
                return [{"label": "NEUTRAL", "score": 0.5} for _ in texts]

            # Clean texts
            clean_texts = []
            for text in texts:
                clean_text = str(text).strip()
                if not clean_text:
                    clean_text = "neutral"  # Fallback for empty texts
                clean_texts.append(clean_text)

            results = self._pipe(clean_texts, truncation=True, max_length=512)

            # Ensure results is a list
            if not isinstance(results, list):
                results = [results]

            return results

        except Exception as e:
            _logger.exception("HF blocking predict failed: %s", e)
            # Return neutral fallback for all texts
            return [{"label": "NEUTRAL", "score": 0.5} for _ in texts]

    async def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict sentiment for a batch of texts.

        Args:
            texts: List of text strings to analyze

        Returns:
            List of prediction dictionaries with label and score
        """
        if not texts:
            return []

        # Wait for initialization to complete
        if self._init_task and not self._init_task.done():
            await self._init_task

        # Check if initialization failed
        if self._initialization_error:
            raise RuntimeError(f"HF pipeline initialization failed: {self._initialization_error}")

        start_time = time.time()

        try:
            async with self.semaphore:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(self._executor, self._predict_blocking, texts)

                response_time_ms = (time.time() - start_time) * 1000
                self._update_health_success(response_time_ms)

                out = []
                for r in results:
                    try:
                        label = r.get("label", "NEUTRAL")
                        score = float(r.get("score", 0.5))
                        out.append({"label": label, "score": score, "raw": r})
                    except (ValueError, TypeError) as e:
                        _logger.debug("Error processing HF result: %s", e)
                        out.append({"label": "NEUTRAL", "score": 0.5, "raw": r})

                return out

        except Exception as e:
            self._update_health_failure(e)
            _logger.error("HF predict_batch failed: %s", e)
            raise

    async def predict_single(self, text: str) -> Dict:
        """
        Predict sentiment for a single text.

        Args:
            text: Text string to analyze

        Returns:
            Prediction dictionary with label and score
        """
        if not text or not text.strip():
            return {"label": "NEUTRAL", "score": 0.5}

        try:
            res = await self.predict_batch([text])
            return res[0] if res else {"label": "NEUTRAL", "score": 0.5}
        except Exception as e:
            _logger.error("HF predict_single failed: %s", e)
            return {"label": "NEUTRAL", "score": 0.5}

    async def fetch_messages(self, ticker: str, since_ts: Optional[int] = None, limit: int = 200) -> List[Dict[str, Any]]:
        """
        HuggingFace adapter doesn't fetch messages directly.
        This method is not applicable for this adapter type.
        """
        raise NotImplementedError("HuggingFace adapter doesn't fetch messages - it processes existing text")

    async def fetch_summary(self, ticker: str, since_ts: Optional[int] = None) -> Dict[str, Any]:
        """
        HuggingFace adapter doesn't fetch summaries directly.
        This method is not applicable for this adapter type.
        """
        raise NotImplementedError("HuggingFace adapter doesn't fetch summaries - it processes existing text")

    async def close(self) -> None:
        """Clean up adapter resources."""
        try:
            # Cancel initialization task if still running
            if self._init_task and not self._init_task.done():
                self._init_task.cancel()
                try:
                    await self._init_task
                except asyncio.CancelledError:
                    pass

            # Shutdown executor
            if self._executor:
                self._executor.shutdown(wait=False)

            _logger.debug("HuggingFace adapter closed successfully")

        except Exception as e:
            _logger.warning("Error closing HuggingFace adapter: %s", e)

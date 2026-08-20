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
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.sentiments.adapters.base_adapter import BaseSentimentAdapter
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

try:
    from transformers import pipeline  # type: ignore

    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

DEFAULT_MODEL = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment")


#: HN comments routinely exceed the 512-token window most sentiment checkpoints were trained
#: with. "truncate" (default, matches historic behavior) silently drops everything past the
#: limit; "chunk_mean"/"chunk_max" split the text into word-based chunks, score each
#: independently, and aggregate -- spec §2.5.4 requires this be explicit and recorded, never a
#: silent truncation.
LongTextStrategy = Literal["truncate", "chunk_mean", "chunk_max"]

#: Word-based chunk size used by chunk_mean/chunk_max. Approximates the model's token budget
#: without requiring a tokenizer call up front (~1.3 tokens/word for English is a safe margin
#: under 512 tokens at 350 words).
_CHUNK_WORD_SIZE = 350


class AsyncHFSentiment(BaseSentimentAdapter):
    def __init__(
        self,
        name: str = "huggingface",
        model_name: str | None = None,
        device: int = -1,
        max_workers: int = 1,
        concurrency: int = 1,
        rate_limit_delay: float = 0.1,
        long_text_strategy: LongTextStrategy = "truncate",
    ):
        super().__init__(name, concurrency, rate_limit_delay)

        if not HF_AVAILABLE:
            raise RuntimeError("transformers is required for AsyncHFSentiment")

        self.model_name = model_name or DEFAULT_MODEL
        self.device = device  # -1 means CPU
        self.max_workers = max_workers
        self.long_text_strategy = long_text_strategy
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pipe: Any = None
        self._initialization_error: Optional[Exception] = None

        # Initialize pipeline in background to avoid blocking
        self._init_task = asyncio.create_task(self._initialize_pipeline())

    async def _initialize_pipeline(self) -> None:
        """Initialize the HuggingFace pipeline asynchronously."""
        try:
            _logger.info("Loading HF pipeline %s (device=%s)", self.model_name, self.device)

            # Run pipeline initialization in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            def _build_pipe() -> Any:
                # transformers' overloads mistype this call as returning None
                kwargs: Dict[str, Any] = {"model": self.model_name, "tokenizer": self.model_name, "device": self.device}
                return pipeline("sentiment-analysis", **kwargs)  # type: ignore[func-returns-value]

            self._pipe = await loop.run_in_executor(self._executor, _build_pipe)

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
                return [{"label": "NEUTRAL", "score": 0.5, "long_text_strategy": self.long_text_strategy} for _ in texts]

            # Clean texts
            clean_texts = []
            for text in texts:
                clean_text = text.strip()
                if not clean_text:
                    clean_text = "neutral"  # Fallback for empty texts
                clean_texts.append(clean_text)

            if self.long_text_strategy == "truncate":
                results = self._pipe(clean_texts, truncation=True, max_length=512)
                if not isinstance(results, list):
                    results = [results]
                for r in results:
                    if isinstance(r, dict):
                        r["long_text_strategy"] = "truncate"
                return results

            # chunk_mean / chunk_max: split each text into word-based chunks, score every chunk
            # in one batched pipe() call, then aggregate per text (spec §2.5.4 -- HN comments
            # routinely exceed 512 tokens; silently truncating under-weights the tail of long
            # comments, so the strategy is explicit and recorded on every prediction).
            chunk_lists = [self._chunk_text(t) for t in clean_texts]
            flat_chunks = [c for chunks in chunk_lists for c in chunks]
            flat_results = self._pipe(flat_chunks, truncation=True, max_length=512)
            if not isinstance(flat_results, list):
                flat_results = [flat_results]

            out = []
            idx = 0
            for chunks in chunk_lists:
                n = len(chunks)
                out.append(self._aggregate_chunks(flat_results[idx : idx + n]))
                idx += n
            return out

        except Exception as e:
            _logger.exception("HF blocking predict failed: %s", e)
            # Return neutral fallback for all texts
            return [{"label": "NEUTRAL", "score": 0.5, "long_text_strategy": self.long_text_strategy} for _ in texts]

    @staticmethod
    def _chunk_text(text: str, chunk_words: int = _CHUNK_WORD_SIZE) -> List[str]:
        """Split text into word-based chunks of roughly ``chunk_words`` words each."""
        words = text.split()
        if len(words) <= chunk_words:
            return [text]
        return [" ".join(words[i : i + chunk_words]) for i in range(0, len(words), chunk_words)]

    def _aggregate_chunks(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate per-chunk HF predictions into one prediction per ``self.long_text_strategy``."""
        signed: List[float] = []
        for r in chunk_results:
            label = str(r.get("label", "")).upper()
            score = float(r.get("score", 0.5))
            if "NEG" in label or "LABEL_0" in label:
                signed.append(-score)
            elif "POS" in label or "LABEL_2" in label:
                signed.append(score)
            else:
                signed.append(0.0)

        if not signed:
            return {"label": "NEUTRAL", "score": 0.5, "long_text_strategy": self.long_text_strategy}

        value = max(signed, key=abs) if self.long_text_strategy == "chunk_max" else sum(signed) / len(signed)

        if value > 0.05:
            label = "POSITIVE"
        elif value < -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        return {
            "label": label,
            "score": abs(value),
            "long_text_strategy": self.long_text_strategy,
            "chunk_count": len(chunk_results),
        }

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

    async def fetch_messages(
        self, ticker: str, since_ts: int | None = None, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """
        HuggingFace adapter doesn't fetch messages directly.
        This method is not applicable for this adapter type.
        """
        raise NotImplementedError("HuggingFace adapter doesn't fetch messages - it processes existing text")

    async def fetch_summary(self, ticker: str, since_ts: int | None = None) -> Dict[str, Any]:
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

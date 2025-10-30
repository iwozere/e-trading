# src/common/sentiments/processing/enhanced_hf_analyzer.py
"""
Enhanced HuggingFace sentiment analysis with multiple model support and optimization.

This module provides improved HuggingFace integration with:
- Support for multiple pre-trained sentiment models
- Model selection based on content type
- Batch processing optimization for ML inference
- Fallback mechanisms when ML models fail
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Any, Union
import os
from pathlib import Path
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

class ContentType(Enum):
    """Content type for model selection."""
    SOCIAL_MEDIA = "social_media"
    NEWS = "news"
    FINANCIAL = "financial"
    GENERAL = "general"

@dataclass
class ModelConfig:
    """Configuration for a HuggingFace model."""
    name: str
    model_path: str
    content_types: List[ContentType]
    max_length: int = 512
    batch_size: int = 16
    confidence_threshold: float = 0.6

@dataclass
class SentimentPrediction:
    """Enhanced sentiment prediction result."""
    label: str
    score: float
    confidence: float
    model_used: str
    processing_time_ms: float
    raw_output: Dict[str, Any]

class EnhancedHFAnalyzer:
    """
    Enhanced HuggingFace sentiment analyzer with multiple model support and optimization.

    Features:
    - Multiple pre-trained models for different content types
    - Intelligent model selection based on content
    - Optimized batch processing
    - Robust fallback mechanisms
    - Performance monitoring and caching
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the enhanced HF analyzer.

        Args:
            config: Configuration dictionary with model settings
        """
        if not HF_AVAILABLE:
            raise RuntimeError("transformers and torch are required for EnhancedHFAnalyzer")

        self.config = config or {}
        self.device = self._get_device()
        self.max_workers = self.config.get("max_workers", 2)
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Model configurations
        self.models = self._load_model_configs()
        self._pipelines: Dict[str, Any] = {}
        self._model_stats: Dict[str, Dict] = {}

        # Performance settings
        self.batch_size = self.config.get("batch_size", 16)
        self.enable_caching = self.config.get("enable_caching", True)
        self._cache: Dict[str, SentimentPrediction] = {}
        self.cache_max_size = self.config.get("cache_max_size", 1000)

        # Fallback settings
        self.fallback_enabled = self.config.get("fallback_enabled", True)
        self.fallback_model = self.config.get("fallback_model", "default")

        # Initialize models
        self._initialization_tasks = {}
        self._initialize_models()

    def _get_device(self) -> Union[str, int]:
        """Determine the best device for inference."""
        device_config = self.config.get("device", "auto")

        if device_config == "auto":
            if HF_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available():
                return 0  # Use first GPU
            else:
                return -1  # Use CPU

        return device_config

    def _load_model_configs(self) -> Dict[str, ModelConfig]:
        """Load model configurations for different content types."""
        default_models = {
            "default": ModelConfig(
                name="default",
                model_path="cardiffnlp/twitter-roberta-base-sentiment-latest",
                content_types=[ContentType.GENERAL, ContentType.SOCIAL_MEDIA],
                max_length=512,
                batch_size=16
            ),
            "financial": ModelConfig(
                name="financial",
                model_path="ProsusAI/finbert",
                content_types=[ContentType.FINANCIAL],
                max_length=512,
                batch_size=8
            ),
            "news": ModelConfig(
                name="news",
                model_path="cardiffnlp/twitter-roberta-base-sentiment-latest",
                content_types=[ContentType.NEWS],
                max_length=512,
                batch_size=12
            ),
            "social": ModelConfig(
                name="social",
                model_path="cardiffnlp/twitter-roberta-base-sentiment-latest",
                content_types=[ContentType.SOCIAL_MEDIA],
                max_length=280,
                batch_size=20
            )
        }

        # Load custom models from config
        custom_models = self.config.get("models", {})
        for name, model_config in custom_models.items():
            content_types = [ContentType(ct) for ct in model_config.get("content_types", ["general"])]
            default_models[name] = ModelConfig(
                name=name,
                model_path=model_config["model_path"],
                content_types=content_types,
                max_length=model_config.get("max_length", 512),
                batch_size=model_config.get("batch_size", 16),
                confidence_threshold=model_config.get("confidence_threshold", 0.6)
            )

        return default_models

    def _initialize_models(self) -> None:
        """Initialize model pipelines asynchronously."""
        for model_name, model_config in self.models.items():
            self._initialization_tasks[model_name] = asyncio.create_task(
                self._initialize_single_model(model_name, model_config)
            )

    async def _initialize_single_model(self, model_name: str, model_config: ModelConfig) -> None:
        """Initialize a single model pipeline."""
        try:
            _logger.info("Loading HF model %s: %s", model_name, model_config.model_path)

            start_time = time.time()

            # Run model initialization in thread pool
            loop = asyncio.get_event_loop()
            pipeline_obj = await loop.run_in_executor(
                self._executor,
                self._create_pipeline,
                model_config.model_path,
                model_config.max_length
            )

            self._pipelines[model_name] = pipeline_obj

            # Initialize stats
            self._model_stats[model_name] = {
                "load_time_ms": (time.time() - start_time) * 1000,
                "predictions_count": 0,
                "total_processing_time_ms": 0.0,
                "error_count": 0,
                "last_used": None
            }

            _logger.info("Successfully loaded model %s in %.2f seconds",
                        model_name, time.time() - start_time)

        except Exception as e:
            _logger.error("Failed to initialize model %s: %s", model_name, e)
            self._model_stats[model_name] = {"error": str(e), "load_failed": True}

    def _create_pipeline(self, model_path: str, max_length: int):
        """Create HuggingFace pipeline (blocking operation for thread pool)."""
        return pipeline(
            "sentiment-analysis",
            model=model_path,
            tokenizer=model_path,
            device=self.device,
            truncation=True,
            max_length=max_length,
            return_all_scores=True
        )

    def select_model(self, content_type: ContentType, text_length: Optional[int] = None) -> str:
        """
        Select the best model for given content type and text characteristics.

        Args:
            content_type: Type of content to analyze
            text_length: Length of text (for optimization)

        Returns:
            Model name to use for analysis
        """
        # Find models that support this content type
        suitable_models = []
        for model_name, model_config in self.models.items():
            if content_type in model_config.content_types:
                suitable_models.append((model_name, model_config))

        if not suitable_models:
            # Fallback to default model
            return "default"

        # If only one suitable model, use it
        if len(suitable_models) == 1:
            return suitable_models[0][0]

        # Select based on text length and model performance
        if text_length:
            # Prefer models with appropriate batch sizes for text length
            for model_name, model_config in suitable_models:
                if text_length <= model_config.max_length:
                    return model_name

        # Fallback to first suitable model
        return suitable_models[0][0]

    async def predict_batch(self, texts: List[str], content_type: ContentType = ContentType.GENERAL) -> List[SentimentPrediction]:
        """
        Predict sentiment for a batch of texts with optimized processing.

        Args:
            texts: List of texts to analyze
            content_type: Type of content for model selection

        Returns:
            List of sentiment predictions
        """
        if not texts:
            return []

        # Select appropriate model
        model_name = self.select_model(content_type, max(len(text) for text in texts))

        # Wait for model initialization
        if model_name in self._initialization_tasks:
            await self._initialization_tasks[model_name]

        # Check if model is available
        if model_name not in self._pipelines:
            if self.fallback_enabled and self.fallback_model in self._pipelines:
                _logger.warning("Model %s not available, using fallback %s", model_name, self.fallback_model)
                model_name = self.fallback_model
            else:
                raise RuntimeError(f"Model {model_name} not available and no fallback configured")

        # Process in optimized batches
        model_config = self.models[model_name]
        batch_size = min(model_config.batch_size, self.batch_size)

        all_predictions = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_predictions = await self._process_batch(batch_texts, model_name, model_config)
            all_predictions.extend(batch_predictions)

        return all_predictions

    async def _process_batch(self, texts: List[str], model_name: str, model_config: ModelConfig) -> List[SentimentPrediction]:
        """Process a single batch of texts."""
        start_time = time.time()

        try:
            # Check cache first
            cached_results = []
            uncached_texts = []
            uncached_indices = []

            if self.enable_caching:
                for i, text in enumerate(texts):
                    cache_key = f"{model_name}:{hash(text)}"
                    if cache_key in self._cache:
                        cached_results.append((i, self._cache[cache_key]))
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
            else:
                uncached_texts = texts
                uncached_indices = list(range(len(texts)))

            # Process uncached texts
            new_predictions = []
            if uncached_texts:
                loop = asyncio.get_event_loop()
                pipeline_obj = self._pipelines[model_name]

                raw_results = await loop.run_in_executor(
                    self._executor,
                    self._predict_blocking,
                    pipeline_obj,
                    uncached_texts
                )

                # Convert to SentimentPrediction objects
                processing_time_ms = (time.time() - start_time) * 1000

                for i, (text, raw_result) in enumerate(zip(uncached_texts, raw_results)):
                    prediction = self._convert_raw_result(
                        raw_result, model_name, processing_time_ms / len(uncached_texts)
                    )
                    new_predictions.append((uncached_indices[i], prediction))

                    # Cache result
                    if self.enable_caching:
                        cache_key = f"{model_name}:{hash(text)}"
                        self._cache[cache_key] = prediction

                        # Manage cache size
                        if len(self._cache) > self.cache_max_size:
                            # Remove oldest entries (simple FIFO)
                            keys_to_remove = list(self._cache.keys())[:len(self._cache) - self.cache_max_size + 100]
                            for key in keys_to_remove:
                                del self._cache[key]

            # Combine cached and new results
            all_results = cached_results + new_predictions
            all_results.sort(key=lambda x: x[0])  # Sort by original index

            # Update model stats
            self._update_model_stats(model_name, len(texts), time.time() - start_time)

            return [result[1] for result in all_results]

        except Exception as e:
            _logger.error("Batch processing failed for model %s: %s", model_name, e)
            self._model_stats[model_name]["error_count"] += 1

            # Return fallback predictions
            return [self._create_fallback_prediction(model_name) for _ in texts]

    def _predict_blocking(self, pipeline_obj, texts: List[str]):
        """Blocking prediction function for thread pool execution."""
        try:
            # Clean and validate texts
            clean_texts = []
            for text in texts:
                clean_text = str(text).strip()
                if not clean_text:
                    clean_text = "neutral"
                clean_texts.append(clean_text)

            # Run prediction
            results = pipeline_obj(clean_texts)

            # Ensure results is a list of lists (for return_all_scores=True)
            if not isinstance(results, list):
                results = [results]

            return results

        except Exception as e:
            _logger.exception("HF blocking predict failed: %s", e)
            # Return neutral fallback for all texts
            return [[{"label": "NEUTRAL", "score": 0.5}] for _ in texts]

    def _convert_raw_result(self, raw_result: List[Dict], model_name: str, processing_time_ms: float) -> SentimentPrediction:
        """Convert raw HuggingFace result to SentimentPrediction."""
        try:
            # Find the prediction with highest score
            best_prediction = max(raw_result, key=lambda x: x.get("score", 0))

            label = best_prediction.get("label", "NEUTRAL").upper()
            score = float(best_prediction.get("score", 0.5))

            # Normalize label and convert to sentiment score
            if "POS" in label or "POSITIVE" in label or "LABEL_2" in label:
                sentiment_score = score
                normalized_label = "POSITIVE"
            elif "NEG" in label or "NEGATIVE" in label or "LABEL_0" in label:
                sentiment_score = -score
                normalized_label = "NEGATIVE"
            else:
                sentiment_score = 0.0
                normalized_label = "NEUTRAL"

            return SentimentPrediction(
                label=normalized_label,
                score=sentiment_score,
                confidence=score,
                model_used=model_name,
                processing_time_ms=processing_time_ms,
                raw_output={"all_scores": raw_result, "best": best_prediction}
            )

        except Exception as e:
            _logger.debug("Error converting HF result: %s", e)
            return self._create_fallback_prediction(model_name)

    def _create_fallback_prediction(self, model_name: str) -> SentimentPrediction:
        """Create a neutral fallback prediction."""
        return SentimentPrediction(
            label="NEUTRAL",
            score=0.0,
            confidence=0.0,
            model_used=f"{model_name}_fallback",
            processing_time_ms=0.0,
            raw_output={"fallback": True}
        )

    def _update_model_stats(self, model_name: str, batch_size: int, processing_time: float) -> None:
        """Update model performance statistics."""
        if model_name in self._model_stats and "error" not in self._model_stats[model_name]:
            stats = self._model_stats[model_name]
            stats["predictions_count"] += batch_size
            stats["total_processing_time_ms"] += processing_time * 1000
            stats["last_used"] = datetime.now(timezone.utc).isoformat()

    async def predict_single(self, text: str, content_type: ContentType = ContentType.GENERAL) -> SentimentPrediction:
        """
        Predict sentiment for a single text.

        Args:
            text: Text to analyze
            content_type: Type of content for model selection

        Returns:
            Sentiment prediction
        """
        if not text or not text.strip():
            return self._create_fallback_prediction("none")

        results = await self.predict_batch([text], content_type)
        return results[0] if results else self._create_fallback_prediction("none")

    def get_model_stats(self) -> Dict[str, Dict]:
        """Get performance statistics for all models."""
        stats = {}
        for model_name, model_stats in self._model_stats.items():
            if "error" not in model_stats:
                avg_time = (model_stats["total_processing_time_ms"] /
                           max(1, model_stats["predictions_count"]))
                stats[model_name] = {
                    **model_stats,
                    "avg_processing_time_ms": avg_time,
                    "status": "healthy" if model_name in self._pipelines else "not_loaded"
                }
            else:
                stats[model_name] = {**model_stats, "status": "failed"}

        return stats

    def get_available_models(self) -> List[str]:
        """Get list of successfully loaded models."""
        return list(self._pipelines.keys())

    async def close(self) -> None:
        """Clean up analyzer resources."""
        try:
            # Cancel initialization tasks
            for task in self._initialization_tasks.values():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Clear cache
            self._cache.clear()

            # Shutdown executor
            if self._executor:
                self._executor.shutdown(wait=False)

            _logger.debug("Enhanced HF analyzer closed successfully")

        except Exception as e:
            _logger.warning("Error closing enhanced HF analyzer: %s", e)
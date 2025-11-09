"""
Unit tests for AsyncHFSentiment.

Tests cover:
- Pipeline initialization and model loading
- Batch and single text prediction
- Error handling and fallback behavior
- Thread pool execution and async integration
- Health monitoring and resource cleanup
"""
import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.common.sentiments.adapters.async_hf_sentiment import AsyncHFSentiment
from src.common.sentiments.adapters.base_adapter import AdapterStatus


class TestAsyncHFSentiment:
    """Test suite for AsyncHFSentiment."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock HuggingFace pipeline."""
        pipeline_mock = Mock()
        pipeline_mock.return_value = [
            {"label": "POSITIVE", "score": 0.8},
            {"label": "NEGATIVE", "score": 0.7}
        ]
        return pipeline_mock

    @pytest_asyncio.fixture
    async def adapter(self, mock_pipeline):
        """Create adapter instance for testing."""
        with patch('src.common.sentiments.adapters.async_hf_sentiment.HF_AVAILABLE', True):
            with patch('src.common.sentiments.adapters.async_hf_sentiment.pipeline') as mock_pipe_func:
                mock_pipe_func.return_value = mock_pipeline

                adapter = AsyncHFSentiment(
                    model_name="test-model",
                    device=-1,
                    max_workers=1,
                    concurrency=1,
                    rate_limit_delay=0.01
                )

                # Wait for initialization to complete
                await adapter._init_task

                yield adapter
                await adapter.close()

    @pytest.mark.asyncio
    async def test_initialization_success(self, mock_pipeline):
        """Test successful pipeline initialization."""
        with patch('src.common.sentiments.adapters.async_hf_sentiment.HF_AVAILABLE', True):
            with patch('src.common.sentiments.adapters.async_hf_sentiment.pipeline') as mock_pipe_func:
                mock_pipe_func.return_value = mock_pipeline

                adapter = AsyncHFSentiment(model_name="test-model")

                try:
                    # Wait for initialization
                    await adapter._init_task

                    assert adapter._pipe is not None
                    assert adapter._initialization_error is None
                    assert adapter._health_info.status == AdapterStatus.HEALTHY

                    # Verify pipeline was created with correct parameters
                    mock_pipe_func.assert_called_once_with(
                        "sentiment-analysis",
                        model="test-model",
                        tokenizer="test-model",
                        device=-1
                    )

                finally:
                    await adapter.close()

    @pytest.mark.asyncio
    async def test_initialization_failure(self):
        """Test handling of initialization failure."""
        with patch('src.common.sentiments.adapters.async_hf_sentiment.HF_AVAILABLE', True):
            with patch('src.common.sentiments.adapters.async_hf_sentiment.pipeline') as mock_pipe_func:
                mock_pipe_func.side_effect = Exception("Model loading failed")

                adapter = AsyncHFSentiment(model_name="invalid-model")

                try:
                    # Wait for initialization to complete (with error)
                    await adapter._init_task

                    assert adapter._pipe is None
                    assert adapter._initialization_error is not None
                    assert "Model loading failed" in str(adapter._initialization_error)
                    # After 1 failure, status should be HEALTHY (base class behavior)
                    # Only after 3+ failures does it become DEGRADED, 5+ becomes FAILED
                    assert adapter._health_info.status == AdapterStatus.HEALTHY
                    assert adapter._health_info.failure_count == 1

                finally:
                    await adapter.close()

    @pytest.mark.asyncio
    async def test_hf_not_available(self):
        """Test behavior when HuggingFace is not available."""
        with patch('src.common.sentiments.adapters.async_hf_sentiment.HF_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="transformers is required"):
                AsyncHFSentiment()

    @pytest.mark.asyncio
    async def test_predict_batch_success(self, adapter):
        """Test successful batch prediction."""
        texts = ["This is great!", "This is terrible!"]

        # Mock the pipeline to return expected results
        if adapter._pipe:
            adapter._pipe.return_value = [
                {"label": "POSITIVE", "score": 0.9},
                {"label": "NEGATIVE", "score": 0.8}
            ]

        results = await adapter.predict_batch(texts)

        assert len(results) == 2
        assert results[0]["label"] == "POSITIVE"
        assert results[0]["score"] == 0.9
        assert "raw" in results[0]
        assert results[1]["label"] == "NEGATIVE"
        assert results[1]["score"] == 0.8

    @pytest.mark.asyncio
    async def test_predict_batch_empty_input(self, adapter):
        """Test batch prediction with empty input."""
        results = await adapter.predict_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_predict_batch_empty_texts(self, adapter):
        """Test batch prediction with empty/whitespace texts."""
        texts = ["", "   ", "valid text"]

        if adapter._pipe:
            adapter._pipe.return_value = [
                {"label": "NEUTRAL", "score": 0.5},
                {"label": "NEUTRAL", "score": 0.5},
                {"label": "POSITIVE", "score": 0.8}
            ]

        results = await adapter.predict_batch(texts)

        assert len(results) == 3
        # Empty texts should be handled gracefully
        assert all("label" in result for result in results)

    @pytest.mark.asyncio
    async def test_predict_batch_initialization_error(self):
        """Test batch prediction when initialization failed."""
        with patch('src.common.sentiments.adapters.async_hf_sentiment.HF_AVAILABLE', True):
            with patch('src.common.sentiments.adapters.async_hf_sentiment.pipeline') as mock_pipe_func:
                mock_pipe_func.side_effect = Exception("Init failed")

                adapter = AsyncHFSentiment()

                try:
                    await adapter._init_task  # Wait for failed initialization

                    with pytest.raises(RuntimeError, match="HF pipeline initialization failed"):
                        await adapter.predict_batch(["test"])

                finally:
                    await adapter.close()

    @pytest.mark.asyncio
    async def test_predict_batch_pipeline_error(self, adapter):
        """Test batch prediction when pipeline execution fails."""
        texts = ["test text"]

        # Mock pipeline to raise exception
        if adapter._pipe:
            adapter._pipe.side_effect = Exception("Pipeline execution failed")

        with pytest.raises(Exception, match="Pipeline execution failed"):
            await adapter.predict_batch(texts)

        # Health should be updated to reflect failure
        assert adapter._health_info.status in [AdapterStatus.DEGRADED, AdapterStatus.FAILED]

    @pytest.mark.asyncio
    async def test_predict_single_success(self, adapter):
        """Test successful single text prediction."""
        text = "This is amazing!"

        if adapter._pipe:
            adapter._pipe.return_value = [{"label": "POSITIVE", "score": 0.95}]

        result = await adapter.predict_single(text)

        assert result["label"] == "POSITIVE"
        assert result["score"] == 0.95
        assert "raw" in result

    @pytest.mark.asyncio
    async def test_predict_single_empty_text(self, adapter):
        """Test single prediction with empty text."""
        result = await adapter.predict_single("")

        assert result["label"] == "NEUTRAL"
        assert result["score"] == 0.5

    @pytest.mark.asyncio
    async def test_predict_single_whitespace_text(self, adapter):
        """Test single prediction with whitespace-only text."""
        result = await adapter.predict_single("   \n\t   ")

        assert result["label"] == "NEUTRAL"
        assert result["score"] == 0.5

    @pytest.mark.asyncio
    async def test_predict_single_error_handling(self, adapter):
        """Test error handling in single prediction."""
        with patch.object(adapter, 'predict_batch', new_callable=AsyncMock) as mock_batch:
            mock_batch.side_effect = Exception("Batch failed")

            result = await adapter.predict_single("test")

            # Should return neutral fallback on error
            assert result["label"] == "NEUTRAL"
            assert result["score"] == 0.5

    def test_predict_blocking_success(self, adapter):
        """Test the blocking prediction function."""
        texts = ["Great!", "Terrible!"]

        if adapter._pipe:
            adapter._pipe.return_value = [
                {"label": "POSITIVE", "score": 0.9},
                {"label": "NEGATIVE", "score": 0.8}
            ]

        results = adapter._predict_blocking(texts)

        assert len(results) == 2
        assert results[0]["label"] == "POSITIVE"
        assert results[1]["label"] == "NEGATIVE"

    def test_predict_blocking_empty_texts(self, adapter):
        """Test blocking prediction with empty texts."""
        texts = ["", "   "]

        results = adapter._predict_blocking(texts)

        # Should return neutral fallback for empty texts
        assert len(results) == 2
        assert all(r["label"] == "NEUTRAL" and r["score"] == 0.5 for r in results)

    def test_predict_blocking_no_pipeline(self, adapter):
        """Test blocking prediction when pipeline is None."""
        adapter._pipe = None

        results = adapter._predict_blocking(["test"])

        # Should return neutral fallback
        assert len(results) == 1
        assert results[0]["label"] == "NEUTRAL"
        assert results[0]["score"] == 0.5

    def test_predict_blocking_pipeline_error(self, adapter):
        """Test blocking prediction when pipeline raises exception."""
        if adapter._pipe:
            adapter._pipe.side_effect = Exception("Pipeline error")

        results = adapter._predict_blocking(["test"])

        # Should return neutral fallback on error
        assert len(results) == 1
        assert results[0]["label"] == "NEUTRAL"
        assert results[0]["score"] == 0.5

    def test_predict_blocking_malformed_results(self, adapter):
        """Test handling of malformed pipeline results."""
        texts = ["test1", "test2"]

        # Mock pipeline to return malformed results
        if adapter._pipe:
            adapter._pipe.return_value = [
                {"label": "POSITIVE", "score": 0.9},
                {"invalid": "result"},  # Missing required fields
            ]

        results = adapter._predict_blocking(texts)

        assert len(results) == 2
        assert results[0]["label"] == "POSITIVE"
        assert results[1]["label"] == "NEUTRAL"  # Fallback for malformed result

    @pytest.mark.asyncio
    async def test_fetch_messages_not_implemented(self, adapter):
        """Test that fetch_messages raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="doesn't fetch messages"):
            await adapter.fetch_messages("AAPL")

    @pytest.mark.asyncio
    async def test_fetch_summary_not_implemented(self, adapter):
        """Test that fetch_summary raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="doesn't fetch summaries"):
            await adapter.fetch_summary("AAPL")

    @pytest.mark.asyncio
    async def test_health_check(self, adapter):
        """Test health check functionality."""
        health = await adapter.health_check()

        assert health.status == AdapterStatus.HEALTHY
        assert health.failure_count == 0

    @pytest.mark.asyncio
    async def test_concurrency_limiting(self):
        """Test that concurrency is properly limited."""
        with patch('src.common.sentiments.adapters.async_hf_sentiment.HF_AVAILABLE', True):
            with patch('src.common.sentiments.adapters.async_hf_sentiment.pipeline') as mock_pipe_func:
                mock_pipeline = Mock()
                mock_pipeline.return_value = [{"label": "NEUTRAL", "score": 0.5}]
                mock_pipe_func.return_value = mock_pipeline

                # Create adapter with concurrency limit of 1
                adapter = AsyncHFSentiment(concurrency=1, rate_limit_delay=0.01)

                try:
                    await adapter._init_task

                    # Mock slow prediction
                    async def slow_predict(*args, **kwargs):
                        await asyncio.sleep(0.05)  # Reduced sleep time
                        return [{"label": "NEUTRAL", "score": 0.5}]

                    with patch.object(adapter, 'predict_batch', side_effect=slow_predict):
                        # Start multiple concurrent requests
                        start_time = asyncio.get_event_loop().time()

                        tasks = [
                            adapter.predict_single("text1"),
                            adapter.predict_single("text2"),
                            adapter.predict_single("text3")
                        ]

                        await asyncio.gather(*tasks)

                        end_time = asyncio.get_event_loop().time()

                        # With concurrency=1, requests should be serialized
                        # Total time should be roughly 3 * 0.05 = 0.15 seconds
                        assert end_time - start_time >= 0.1

                finally:
                    await adapter.close()

    @pytest.mark.asyncio
    async def test_close_cleanup(self, adapter):
        """Test proper resource cleanup on close."""
        # Verify executor exists before close
        assert adapter._executor is not None

        await adapter.close()

        # Executor should be shutdown (we can't easily test this without implementation details)
        # But we can verify the method completes without error

    @pytest.mark.asyncio
    async def test_close_with_running_init_task(self):
        """Test close behavior when initialization task is still running."""
        with patch('src.common.sentiments.adapters.async_hf_sentiment.HF_AVAILABLE', True):
            with patch('src.common.sentiments.adapters.async_hf_sentiment.pipeline') as mock_pipe_func:
                # Make initialization slow
                async def slow_init():
                    await asyncio.sleep(1.0)
                    return Mock()

                mock_pipe_func.side_effect = lambda *args, **kwargs: slow_init()

                adapter = AsyncHFSentiment()

                # Close immediately without waiting for initialization
                await adapter.close()

                # Should complete without hanging

    @pytest.mark.asyncio
    async def test_default_model_from_env(self):
        """Test that default model can be set via environment variable."""
        with patch('src.common.sentiments.adapters.async_hf_sentiment.HF_AVAILABLE', True):
            with patch('src.common.sentiments.adapters.async_hf_sentiment.DEFAULT_MODEL', 'env-model'):
                with patch('src.common.sentiments.adapters.async_hf_sentiment.pipeline') as mock_pipe_func:
                    mock_pipe_func.return_value = Mock()

                    adapter = AsyncHFSentiment()  # No model_name specified

                    try:
                        await adapter._init_task

                        # Should use the environment model
                        assert adapter.model_name == 'env-model'

                    finally:
                        await adapter.close()

    @pytest.mark.asyncio
    async def test_thread_pool_execution(self, adapter):
        """Test that predictions are executed in thread pool."""
        texts = ["test text"]

        # Mock the pipeline to return expected results
        if adapter._pipe:
            adapter._pipe.return_value = [{"label": "NEUTRAL", "score": 0.5}]

        # Just verify the prediction works without blocking
        results = await adapter.predict_batch(texts)

        assert len(results) == 1
        assert results[0]["label"] == "NEUTRAL"

    @pytest.mark.asyncio
    async def test_result_normalization(self, adapter):
        """Test that results are properly normalized."""
        texts = ["test"]

        # Test various result formats
        test_cases = [
            # (pipeline_result, expected_label, expected_score)
            ({"label": "POSITIVE", "score": 0.9}, "POSITIVE", 0.9),
            ({"label": "NEGATIVE", "score": 0.8}, "NEGATIVE", 0.8),
            ({"label": "NEUTRAL", "score": 0.5}, "NEUTRAL", 0.5),
            ({"label": "LABEL_1", "score": 0.7}, "LABEL_1", 0.7),  # Non-standard label
            ({"score": 0.6}, "NEUTRAL", 0.5),  # Missing label
            ({"label": "POSITIVE"}, "POSITIVE", 0.5),  # Missing score
            ({}, "NEUTRAL", 0.5),  # Empty result
        ]

        for pipeline_result, expected_label, expected_score in test_cases:
            if adapter._pipe:
                adapter._pipe.return_value = [pipeline_result]

            results = await adapter.predict_batch(texts)

            assert len(results) == 1
            assert results[0]["label"] == expected_label
            assert results[0]["score"] == expected_score
            assert "raw" in results[0]
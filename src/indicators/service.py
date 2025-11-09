# service.py - ENHANCED UNIFIED VERSION
# Standard library
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# Third party
import pandas as pd

# Local application
from src.common.common import get_ohlcv, determine_provider
from src.indicators.utils import coerce_ohlcv, resample_df, validate_indicator_config
from src.indicators.registry import INDICATOR_META, get_canonical_name, get_indicator_meta
from src.indicators.config_manager import get_config_manager
from src.indicators.recommendation_engine import RecommendationEngine
from src.indicators.adapters.ta_lib_adapter import TaLibAdapter
from src.indicators.adapters.pandas_ta_adapter import PandasTaAdapter
from src.indicators.adapters.fundamentals_adapter import FundamentalsAdapter
from src.indicators.models import (
    IndicatorBatchConfig, IndicatorResultSet,
    IndicatorSpec, IndicatorValue, TickerIndicatorsRequest
)
from src.indicators.models import (
    IndicatorResult, IndicatorSet, IndicatorCategory,
    IndicatorCalculationRequest, BatchIndicatorRequest, PerformanceMetrics
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class IndicatorServiceError(Exception):
    """Base exception for indicator service errors."""
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.now()


class ConfigurationError(IndicatorServiceError):
    """Error in indicator configuration."""
    pass


class DataError(IndicatorServiceError):
    """Error related to data availability or quality."""
    pass


class CalculationError(IndicatorServiceError):
    """Error during indicator calculation."""
    pass


class TimeoutError(IndicatorServiceError):
    """Error due to operation timeout."""
    pass


class CircuitBreakerError(IndicatorServiceError):
    """Error when circuit breaker is open."""
    pass


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker."""
    name: str
    is_open: bool = False
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    success_count: int = 0
    total_requests: int = 0
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    half_open_max_calls: int = 3


class CircuitBreaker:
    """Circuit breaker implementation for external dependencies."""

    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.state = CircuitBreakerState(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        async with self._lock:
            # Check if circuit breaker should be opened
            if self._should_open():
                self.state.is_open = True
                self.state.last_failure_time = datetime.now()
                _logger.warning("Circuit breaker %s opened", self.state.name)

            # Check if circuit breaker should transition to half-open
            if self.state.is_open and self._should_attempt_reset():
                _logger.info("Circuit breaker %s attempting reset", self.state.name)
                # Allow limited calls in half-open state
                if self.state.success_count >= self.state.half_open_max_calls:
                    self.state.is_open = False
                    self.state.failure_count = 0
                    self.state.success_count = 0
                    _logger.info("Circuit breaker %s closed", self.state.name)

            # Reject if circuit is open
            if self.state.is_open and not self._should_attempt_reset():
                raise CircuitBreakerError(
                    f"Circuit breaker {self.state.name} is open",
                    error_code="CIRCUIT_BREAKER_OPEN",
                    context={"circuit_breaker": self.state.name}
                )

        # Execute the function
        try:
            self.state.total_requests += 1
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)

            # Record success
            async with self._lock:
                self.state.success_count += 1
                if self.state.is_open:
                    _logger.debug("Circuit breaker %s recorded success in half-open state", self.state.name)

            return result

        except Exception as e:
            # Record failure
            async with self._lock:
                self.state.failure_count += 1
                self.state.last_failure_time = datetime.now()
                _logger.warning("Circuit breaker %s recorded failure: %s", self.state.name, e)
            raise

    def _should_open(self) -> bool:
        """Check if circuit breaker should be opened."""
        return (
            not self.state.is_open and
            self.state.failure_count >= self.state.failure_threshold
        )

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if not self.state.is_open or not self.state.last_failure_time:
            return False

        time_since_failure = (datetime.now() - self.state.last_failure_time).total_seconds()
        return time_since_failure >= self.state.recovery_timeout


class ErrorRecoveryStrategy:
    """Strategy for handling different types of errors."""

    @staticmethod
    def categorize_error(error: Exception) -> str:
        """Categorize an error for appropriate handling."""
        if isinstance(error, (ConnectionError, TimeoutError)):
            return "network"
        elif isinstance(error, (ValueError, TypeError)):
            return "data"
        elif isinstance(error, ConfigurationError):
            return "configuration"
        elif isinstance(error, CalculationError):
            return "calculation"
        elif isinstance(error, CircuitBreakerError):
            return "circuit_breaker"
        else:
            return "unknown"

    @staticmethod
    def should_retry(error: Exception, attempt: int, max_attempts: int) -> bool:
        """Determine if an error should trigger a retry."""
        error_category = ErrorRecoveryStrategy.categorize_error(error)

        # Don't retry configuration errors
        if error_category == "configuration":
            return False

        # Don't retry if circuit breaker is open
        if error_category == "circuit_breaker":
            return False

        # Retry network and calculation errors up to max attempts
        if error_category in ["network", "calculation"] and attempt < max_attempts:
            return True

        # Retry data errors with limited attempts
        if error_category == "data" and attempt < min(max_attempts, 2):
            return True

        return False

    @staticmethod
    def get_retry_delay(error: Exception, attempt: int) -> float:
        """Get delay before retry based on error type and attempt."""
        error_category = ErrorRecoveryStrategy.categorize_error(error)

        base_delays = {
            "network": 2.0,
            "data": 1.0,
            "calculation": 0.5,
            "unknown": 1.0
        }

        base_delay = base_delays.get(error_category, 1.0)
        # Exponential backoff with jitter
        import random
        return base_delay * (2 ** attempt) + random.uniform(0, 1)


@dataclass
class PerformanceTracker:
    """Track performance metrics for operations."""
    operation_times: Dict[str, List[float]] = field(default_factory=dict)
    operation_counts: Dict[str, int] = field(default_factory=dict)
    cache_hits: Dict[str, int] = field(default_factory=dict)
    cache_misses: Dict[str, int] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)

    def record_operation(self, operation: str, duration: float, cache_hit: bool = False, error: bool = False):
        """Record an operation's performance metrics."""
        if operation not in self.operation_times:
            self.operation_times[operation] = []
            self.operation_counts[operation] = 0
            self.cache_hits[operation] = 0
            self.cache_misses[operation] = 0
            self.error_counts[operation] = 0

        self.operation_times[operation].append(duration)
        self.operation_counts[operation] += 1

        if cache_hit:
            self.cache_hits[operation] += 1
        else:
            self.cache_misses[operation] += 1

        if error:
            self.error_counts[operation] += 1

    def get_stats(self, operation: str = None) -> Dict[str, Any]:
        """Get performance statistics."""
        if operation:
            if operation not in self.operation_times:
                return {}

            times = self.operation_times[operation]
            return {
                "operation": operation,
                "count": self.operation_counts[operation],
                "avg_duration": sum(times) / len(times) if times else 0,
                "min_duration": min(times) if times else 0,
                "max_duration": max(times) if times else 0,
                "cache_hit_rate": self.cache_hits[operation] / max(1, self.operation_counts[operation]),
                "error_rate": self.error_counts[operation] / max(1, self.operation_counts[operation]),
                "total_cache_hits": self.cache_hits[operation],
                "total_cache_misses": self.cache_misses[operation],
                "total_errors": self.error_counts[operation]
            }
        else:
            # Return stats for all operations
            stats = {}
            for op in self.operation_times.keys():
                stats[op] = self.get_stats(op)

            # Add overall stats
            total_operations = sum(self.operation_counts.values())
            total_errors = sum(self.error_counts.values())
            total_cache_hits = sum(self.cache_hits.values())
            total_cache_misses = sum(self.cache_misses.values())

            stats["_overall"] = {
                "total_operations": total_operations,
                "total_errors": total_errors,
                "overall_error_rate": total_errors / max(1, total_operations),
                "overall_cache_hit_rate": total_cache_hits / max(1, total_cache_hits + total_cache_misses),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
            }

            return stats


class PerformanceMonitor:
    """Monitor and collect performance metrics."""

    def __init__(self):
        self.tracker = PerformanceTracker()
        self._operation_stack = []

    def start_operation(self, operation: str) -> str:
        """Start timing an operation."""
        operation_id = f"{operation}_{time.time()}"
        self._operation_stack.append({
            "id": operation_id,
            "operation": operation,
            "start_time": time.time()
        })
        return operation_id

    def end_operation(self, operation_id: str, cache_hit: bool = False, error: bool = False):
        """End timing an operation."""
        end_time = time.time()

        # Find the operation in the stack
        for i, op_info in enumerate(self._operation_stack):
            if op_info["id"] == operation_id:
                duration = end_time - op_info["start_time"]
                self.tracker.record_operation(
                    op_info["operation"],
                    duration,
                    cache_hit=cache_hit,
                    error=error
                )
                self._operation_stack.pop(i)
                return duration

        return 0.0

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.tracker.get_stats(),
            "active_operations": len(self._operation_stack)
        }


class BenchmarkRunner:
    """Run benchmarks against legacy implementations."""

    def __init__(self, unified_service: 'UnifiedIndicatorService'):
        self.unified_service = unified_service

    async def benchmark_single_ticker(
        self,
        ticker: str,
        indicators: List[str],
        iterations: int = 10
    ) -> Dict[str, Any]:
        """Benchmark single ticker calculation."""
        results = {
            "ticker": ticker,
            "indicators": indicators,
            "iterations": iterations,
            "unified_service": {"times": [], "errors": 0},
            "comparison": {}
        }

        # Benchmark unified service
        for i in range(iterations):
            try:
                start_time = time.time()

                request = IndicatorCalculationRequest(
                    ticker=ticker,
                    indicators=indicators,
                    timeframe="1d",
                    period="1y"
                )

                await self.unified_service.get_indicators(request)

                duration = time.time() - start_time
                results["unified_service"]["times"].append(duration)

            except Exception as e:
                results["unified_service"]["errors"] += 1
                _logger.error("Benchmark error for %s: %s", ticker, e)

        # Calculate statistics
        times = results["unified_service"]["times"]
        if times:
            results["unified_service"]["avg_time"] = sum(times) / len(times)
            results["unified_service"]["min_time"] = min(times)
            results["unified_service"]["max_time"] = max(times)
            results["unified_service"]["success_rate"] = len(times) / iterations

        return results

    async def benchmark_batch_processing(
        self,
        tickers: List[str],
        indicators: List[str],
        batch_sizes: List[int] = None
    ) -> Dict[str, Any]:
        """Benchmark batch processing with different batch sizes."""
        if batch_sizes is None:
            batch_sizes = [1, 5, 10, 20]

        results = {
            "tickers": tickers,
            "indicators": indicators,
            "batch_results": {}
        }

        for batch_size in batch_sizes:
            try:
                start_time = time.time()

                # Configure batch processing
                batch_config = BatchProcessingConfig(
                    max_concurrent=batch_size,
                    batch_size=batch_size
                )

                request = BatchIndicatorRequest(
                    tickers=tickers,
                    indicators=indicators,
                    max_concurrent=batch_size
                )

                batch_result = await self.unified_service.get_batch_indicators_enhanced(
                    request, batch_config
                )

                duration = time.time() - start_time

                results["batch_results"][batch_size] = {
                    "duration": duration,
                    "success_rate": batch_result.success_rate,
                    "successful_count": len(batch_result.successful),
                    "failed_count": len(batch_result.failed),
                    "throughput": len(tickers) / duration if duration > 0 else 0
                }

            except Exception as e:
                results["batch_results"][batch_size] = {
                    "error": str(e),
                    "duration": 0,
                    "success_rate": 0
                }

        return results


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing operations."""
    max_concurrent: int = 10
    batch_size: int = 50
    timeout_per_ticker: float = 30.0
    retry_attempts: int = 2
    retry_delay: float = 1.0
    partial_results: bool = True
    fail_fast: bool = False


@dataclass
class BatchResult:
    """Result of a batch processing operation."""
    successful: Dict[str, IndicatorSet] = field(default_factory=dict)
    failed: Dict[str, Exception] = field(default_factory=dict)
    partial: Dict[str, IndicatorSet] = field(default_factory=dict)
    performance_metrics: PerformanceMetrics = None
    total_processed: int = 0
    success_rate: float = 0.0


class UnifiedIndicatorService:
    """
    Unified indicator service that consolidates all indicator functionality.

    This service provides a single interface for calculating technical and fundamental
    indicators, with support for multiple backends, configuration management,
    and intelligent recommendations.
    """

    def __init__(self, prefer: Dict[str, int] | None = None, batch_config: BatchProcessingConfig = None):
        # Initialize adapters with graceful handling of missing dependencies
        self.adapters = {}

        # Try to initialize ta-lib adapter
        try:
            self.adapters["ta-lib"] = TaLibAdapter()
            _logger.info("Initialized ta-lib adapter")
        except ImportError as e:
            _logger.warning("Failed to initialize ta-lib adapter: %s", e)

        # Try to initialize pandas-ta adapter
        try:
            self.adapters["pandas-ta"] = PandasTaAdapter()
            _logger.info("Initialized pandas-ta adapter")
        except ImportError as e:
            _logger.warning("Failed to initialize pandas-ta adapter: %s", e)

        # Try to initialize fundamentals adapter
        try:
            self.adapters["fundamentals"] = FundamentalsAdapter()
            _logger.info("Initialized fundamentals adapter")
        except ImportError as e:
            _logger.warning("Failed to initialize fundamentals adapter: %s", e)

        if not self.adapters:
            raise RuntimeError("No indicator adapters available. Install at least one: talib, pandas_ta")

        self.prefer = prefer or {}
        self.config_manager = get_config_manager()
        self.recommendation_engine = RecommendationEngine()
        self.batch_config = batch_config or BatchProcessingConfig()
        self._thread_pool = ThreadPoolExecutor(max_workers=self.batch_config.max_concurrent)

        # Initialize circuit breakers for external dependencies
        self.circuit_breakers = {
            "data_provider": CircuitBreaker("data_provider", failure_threshold=5, recovery_timeout=60.0),
            "fundamentals": CircuitBreaker("fundamentals", failure_threshold=3, recovery_timeout=120.0),
            "ta_lib": CircuitBreaker("ta_lib", failure_threshold=10, recovery_timeout=30.0),
        }

        self.error_recovery = ErrorRecoveryStrategy()
        self.performance_monitor = PerformanceMonitor()
        self.benchmark_runner = BenchmarkRunner(self)

        _logger.info("Initialized UnifiedIndicatorService with %d adapters and %d circuit breakers",
                    len(self.adapters), len(self.circuit_breakers))

    def _select_provider(self, name: str):
        meta = INDICATOR_META.get(name)
        if not meta:
            raise ValueError(f"Unknown indicator: {name}")
        candidates = sorted(meta.providers, key=lambda p: self.prefer.get(p, 0))
        for prov in candidates:
            if self.adapters[prov].supports(name):
                return self.adapters[prov]
        raise RuntimeError(f"No adapter supports {name}")

    def _build_inputs(self, df: pd.DataFrame, name: str, spec: IndicatorSpec | None = None):
        meta = INDICATOR_META[name]
        inputs = {}
        if meta.kind == "tech":
            for key in meta.inputs:
                col = (spec.input_map.get(key) if spec else None) or key
                if col not in df.columns:
                    raise ValueError(f"Missing input column '{col}' for {name}")
                inputs[key] = df[col]
        return inputs

    async def compute(
        self,
        df: pd.DataFrame,
        config: IndicatorBatchConfig,
        fund_params: Dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Async compute supporting both tech and fundamental indicators"""
        validate_indicator_config(config)

        base = coerce_ohlcv(df)
        base = resample_df(base, config.timeframe)
        out = base.copy()

        for spec in config.indicators:
            meta = INDICATOR_META[spec.name]
            adapter = self._select_provider(spec.name)

            # Per-indicator resample (tech only)
            working = (
                out if meta.kind == "fund"
                else resample_df(base, spec.timeframe) if spec.timeframe
                else base
            )

            inputs = self._build_inputs(working, spec.name, spec)
            params = dict(spec.params)

            if meta.kind == "fund":
                params.update(fund_params or {})

            # Await async compute
            res = await adapter.compute(spec.name, working, inputs, params)

            final_names = (
                spec.output if isinstance(spec.output, dict)
                else {"value": spec.output}
            )

            for k, s in res.items():
                cname = final_names.get(k)
                if not cname:
                    continue

                # Align to out index
                if len(s.index) == 1:
                    s = pd.Series(s.iloc[0], index=out.index)
                else:
                    s = s.reindex(out.index).ffill()

                out[cname] = s

        if config.dropna_after:
            out = out.dropna()

        return out

    async def compute_for_ticker(
        self,
        req: TickerIndicatorsRequest
    ) -> IndicatorResultSet:
        """Compute indicators for a ticker (both technical and fundamental)"""
        try:
            provider = req.provider or determine_provider(req.ticker)

            # Fetch OHLCV with circuit breaker protection
            df = await self.circuit_breakers["data_provider"].call(
                asyncio.to_thread,
                get_ohlcv, req.ticker, req.timeframe, req.period, provider
            )

            # Validate data quality
            if df is None or df.empty:
                raise DataError(
                    f"No data available for {req.ticker}",
                    error_code="NO_DATA",
                    context={"ticker": req.ticker, "provider": provider}
                )

            # Check minimum data requirements
            min_required_rows = 20  # Minimum for most technical indicators
            if len(df) < min_required_rows:
                _logger.warning("Insufficient data for %s: %d rows (minimum %d)",
                              req.ticker, len(df), min_required_rows)
                # Continue with available data but log warning

            # Build specs with error handling
            specs: List[IndicatorSpec] = []
            for name in req.indicators:
                try:
                    if name not in INDICATOR_META:
                        raise ConfigurationError(
                            f"Unknown indicator: {name}",
                            error_code="UNKNOWN_INDICATOR",
                            context={"indicator": name, "ticker": req.ticker}
                        )

                    meta = INDICATOR_META[name]
                    if meta.kind == "tech":
                        outmap = (
                            {"value": name} if meta.outputs == ["value"]
                            else {o: f"{name}_{o}" for o in meta.outputs}
                        )
                        specs.append(IndicatorSpec(
                            name=name,
                            output=outmap if len(outmap) > 1 else outmap["value"]
                        ))
                    else:
                        specs.append(IndicatorSpec(name=name, output=name))

                except Exception as e:
                    _logger.error("Error building spec for indicator %s: %s", name, e)
                    # Continue with other indicators if possible
                    continue

            if not specs:
                raise ConfigurationError(
                    "No valid indicators specified",
                    error_code="NO_VALID_INDICATORS",
                    context={"ticker": req.ticker, "requested_indicators": req.indicators}
                )

            cfg = IndicatorBatchConfig(timeframe=req.timeframe, indicators=specs)

            # Compute with error handling
            try:
                df_all = await self.compute(
                    df, cfg,
                    fund_params={"ticker": req.ticker, "provider": provider}
                )
            except Exception as e:
                raise CalculationError(
                    f"Error computing indicators for {req.ticker}: {str(e)}",
                    error_code="COMPUTATION_FAILED",
                    context={"ticker": req.ticker, "indicators": [s.name for s in specs]}
                ) from e

            # Build results with error handling
            tech: Dict[str, IndicatorValue] = {}
            fund: Dict[str, IndicatorValue] = {}

            for name in req.indicators:
                try:
                    if name not in INDICATOR_META:
                        continue  # Skip unknown indicators

                    cols = [c for c in df_all.columns
                           if c == name or c.startswith(f"{name}_")]

                    for c in cols:
                        try:
                            v = df_all[c].iloc[-1] if len(df_all) else None

                            # Validate result value
                            if v is not None and (pd.isna(v) or not pd.isfinite(v)):
                                _logger.warning("Invalid result for %s.%s: %s", req.ticker, c, v)
                                v = None

                            target = tech if INDICATOR_META[name].kind == "tech" else fund
                            target[c] = IndicatorValue(
                                name=c,
                                value=v,
                                source="unified_service"
                            )
                        except Exception as e:
                            _logger.error("Error processing result for %s.%s: %s", req.ticker, c, e)
                            # Continue with other results
                            continue

                except Exception as e:
                    _logger.error("Error processing indicator %s for %s: %s", name, req.ticker, e)
                    continue

            return IndicatorResultSet(
                ticker=req.ticker,
                technical=tech,
                fundamental=fund
            )

        except IndicatorServiceError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise CalculationError(
                f"Unexpected error computing indicators for {req.ticker}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                context={"ticker": req.ticker}
            ) from e

    # Legacy interface compatibility methods
    async def get_indicators(self, request: IndicatorCalculationRequest) -> IndicatorSet:
        """
        Legacy interface: Get indicators for a single ticker.

        Args:
            request: Indicator calculation request

        Returns:
            IndicatorSet with calculated indicators and recommendations
        """
        operation_id = self.performance_monitor.start_operation("get_indicators")
        cache_hit = False
        error_occurred = False

        try:
            # Convert to new request format
            ticker_request = TickerIndicatorsRequest(
                ticker=request.ticker,
                timeframe=request.timeframe,
                period=request.period,
                provider=request.provider,
                indicators=request.indicators,
                include_recommendations=request.include_recommendations,
                force_refresh=request.force_refresh
            )

            # Get results using new interface
            result_set = await self.compute_for_ticker(ticker_request)

            # Convert to legacy format
            indicator_set = IndicatorSet(ticker=request.ticker)

            # Add technical indicators
            for name, indicator_value in result_set.technical.items():
                if indicator_value.value is not None:
                    # Get recommendation if requested
                    recommendation = None
                    if request.include_recommendations:
                        # Build context for contextual recommendations
                        all_values = {}
                        all_values.update({k: v.value for k, v in result_set.technical.items()})
                        all_values.update({k: v.value for k, v in result_set.fundamental.items()})

                        recommendation = self.recommendation_engine.get_contextual_recommendation(
                            name, indicator_value.value, all_values
                        )

                    indicator_result = IndicatorResult(
                        name=name,
                        value=indicator_value.value,
                        recommendation=recommendation,
                        category=IndicatorCategory.TECHNICAL,
                        last_updated=datetime.now(),
                        source=indicator_value.source or "unified_service"
                    )
                    indicator_set.add_indicator(indicator_result)

            # Add fundamental indicators
            for name, indicator_value in result_set.fundamental.items():
                if indicator_value.value is not None:
                    # Get recommendation if requested
                    recommendation = None
                    if request.include_recommendations:
                        recommendation = self.recommendation_engine.get_recommendation(
                            name, indicator_value.value
                        )

                    indicator_result = IndicatorResult(
                        name=name,
                        value=indicator_value.value,
                        recommendation=recommendation,
                        category=IndicatorCategory.FUNDAMENTAL,
                        last_updated=datetime.now(),
                        source=indicator_value.source or "unified_service"
                    )
                    indicator_set.add_indicator(indicator_result)

            # Generate composite recommendation if requested
            if request.include_recommendations and indicator_set.get_all_indicators():
                composite = self.recommendation_engine.get_composite_recommendation(indicator_set)
                indicator_set.overall_recommendation = composite
                indicator_set.composite_score = composite.composite_score

            return indicator_set

        except Exception as e:
            error_occurred = True
            _logger.exception("Error in get_indicators for %s: %s", request.ticker, e)
            return IndicatorSet(ticker=request.ticker)
        finally:
            self.performance_monitor.end_operation(operation_id, cache_hit=cache_hit, error=error_occurred)

    async def get_batch_indicators_enhanced(
        self,
        request: BatchIndicatorRequest,
        config: Optional[BatchProcessingConfig] = None
    ) -> BatchResult:
        """
        Enhanced batch processing with advanced error handling and partial results.

        Args:
            request: Batch indicator calculation request
            config: Optional batch processing configuration

        Returns:
            BatchResult with detailed success/failure information
        """
        operation_id = self.performance_monitor.start_operation("batch_indicators_enhanced")
        start_time = time.time()
        batch_config = config or self.batch_config

        result = BatchResult()
        error_occurred = False

        try:
            _logger.info("Starting batch processing for %d tickers", len(request.tickers))

            # Process tickers in configurable batches
            batch_size = min(batch_config.batch_size, len(request.tickers))

            for i in range(0, len(request.tickers), batch_size):
                batch_tickers = request.tickers[i:i + batch_size]

                # Create semaphore to limit concurrent operations
                semaphore = asyncio.Semaphore(batch_config.max_concurrent)

                # Create tasks for concurrent processing with timeout
                tasks = []
                for ticker in batch_tickers:
                    task = self._process_ticker_with_retry(
                        ticker, request, batch_config, semaphore
                    )
                    tasks.append(task)

                # Execute batch with timeout
                try:
                    batch_results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=batch_config.timeout_per_ticker * len(batch_tickers)
                    )
                except asyncio.TimeoutError:
                    _logger.warning("Batch timeout for tickers %s", batch_tickers)
                    batch_results = [asyncio.TimeoutError("Batch timeout")] * len(batch_tickers)

                # Process batch results
                for ticker, batch_result in zip(batch_tickers, batch_results):
                    result.total_processed += 1

                    if isinstance(batch_result, Exception):
                        result.failed[ticker] = batch_result
                        _logger.error("Failed to process %s: %s", ticker, batch_result)

                        # Create empty result for failed ticker if partial results enabled
                        if batch_config.partial_results:
                            result.partial[ticker] = IndicatorSet(ticker=ticker)
                    else:
                        result.successful[ticker] = batch_result
                        _logger.debug("Successfully processed %s", ticker)

                # Check fail-fast condition
                if batch_config.fail_fast and result.failed:
                    _logger.warning("Fail-fast enabled, stopping batch processing")
                    break

                # Small delay between batches to prevent overwhelming
                if i + batch_size < len(request.tickers):
                    await asyncio.sleep(0.1)

            # Calculate success rate
            total_tickers = len(request.tickers)
            successful_count = len(result.successful)
            result.success_rate = successful_count / total_tickers if total_tickers > 0 else 0.0

            # Create performance metrics
            duration = time.time() - start_time
            result.performance_metrics = PerformanceMetrics(
                operation="batch_indicators",
                duration=duration,
                cache_hit=False,  # TODO: Implement cache hit tracking
                indicators_calculated=len(request.indicators) * successful_count,
                tickers_processed=result.total_processed,
                metadata={
                    "batch_size": batch_size,
                    "max_concurrent": batch_config.max_concurrent,
                    "success_rate": result.success_rate,
                    "failed_count": len(result.failed),
                    "partial_count": len(result.partial)
                }
            )

            _logger.info(
                "Batch processing completed: %d successful, %d failed, %d partial (%.2f%% success rate)",
                len(result.successful), len(result.failed), len(result.partial), result.success_rate * 100
            )

            return result

        except Exception as e:
            error_occurred = True
            _logger.exception("Critical error in batch processing: %s", e)
            result.failed["_batch_error"] = e
            return result
        finally:
            self.performance_monitor.end_operation(operation_id, cache_hit=False, error=error_occurred)

    async def _process_ticker_with_retry(
        self,
        ticker: str,
        request: BatchIndicatorRequest,
        config: BatchProcessingConfig,
        semaphore: asyncio.Semaphore
    ) -> IndicatorSet:
        """Process a single ticker with retry logic and concurrency control."""
        async with semaphore:
            last_exception = None

            for attempt in range(config.retry_attempts + 1):
                try:
                    # Create individual request
                    task_request = IndicatorCalculationRequest(
                        ticker=ticker,
                        indicators=request.indicators,
                        timeframe=request.timeframe,
                        period=request.period,
                        provider=request.provider,
                        force_refresh=request.force_refresh,
                        include_recommendations=request.include_recommendations
                    )

                    # Process with timeout
                    result = await asyncio.wait_for(
                        self.get_indicators(task_request),
                        timeout=config.timeout_per_ticker
                    )

                    return result

                except Exception as e:
                    last_exception = e
                    error_category = self.error_recovery.categorize_error(e)

                    _logger.warning(
                        "Attempt %d failed for %s (category: %s): %s",
                        attempt + 1, ticker, error_category, e
                    )

                    # Check if we should retry
                    if not self.error_recovery.should_retry(e, attempt, config.retry_attempts):
                        _logger.info("Not retrying %s due to error category %s", ticker, error_category)
                        break

                    # Wait before retry with intelligent delay
                    if attempt < config.retry_attempts:
                        delay = self.error_recovery.get_retry_delay(e, attempt)
                        _logger.debug("Waiting %.2f seconds before retry for %s", delay, ticker)
                        await asyncio.sleep(delay)

            # All attempts failed - wrap in appropriate error type
            if isinstance(last_exception, IndicatorServiceError):
                raise last_exception
            else:
                raise CalculationError(
                    f"All retry attempts failed for {ticker}",
                    error_code="RETRY_EXHAUSTED",
                    context={"ticker": ticker, "attempts": config.retry_attempts + 1}
                ) from last_exception

    async def get_batch_indicators(self, request: BatchIndicatorRequest) -> Dict[str, IndicatorSet]:
        """
        Legacy interface: Get indicators for multiple tickers.

        This method maintains backward compatibility while using the enhanced batch processing.

        Args:
            request: Batch indicator calculation request

        Returns:
            Dictionary mapping ticker to IndicatorSet
        """
        try:
            # Use enhanced batch processing
            batch_result = await self.get_batch_indicators_enhanced(request)

            # Convert to legacy format
            results = {}
            results.update(batch_result.successful)

            # Include partial results if available
            if self.batch_config.partial_results:
                results.update(batch_result.partial)

            # For failed tickers, create empty IndicatorSet
            for ticker in batch_result.failed:
                if ticker not in results:
                    results[ticker] = IndicatorSet(ticker=ticker)

            return results

        except Exception as e:
            _logger.exception("Error in get_batch_indicators: %s", e)
            return {}

    def aggregate_batch_results(
        self,
        batch_results: List[BatchResult]
    ) -> BatchResult:
        """
        Aggregate multiple batch results into a single result.

        Args:
            batch_results: List of BatchResult objects to aggregate

        Returns:
            Aggregated BatchResult
        """
        aggregated = BatchResult()
        total_duration = 0.0
        total_indicators = 0

        for batch_result in batch_results:
            # Merge successful results
            aggregated.successful.update(batch_result.successful)

            # Merge failed results
            aggregated.failed.update(batch_result.failed)

            # Merge partial results
            aggregated.partial.update(batch_result.partial)

            # Aggregate metrics
            aggregated.total_processed += batch_result.total_processed

            if batch_result.performance_metrics:
                total_duration += batch_result.performance_metrics.duration
                total_indicators += batch_result.performance_metrics.indicators_calculated

        # Calculate overall success rate
        total_tickers = aggregated.total_processed
        successful_count = len(aggregated.successful)
        aggregated.success_rate = successful_count / total_tickers if total_tickers > 0 else 0.0

        # Create aggregated performance metrics
        aggregated.performance_metrics = PerformanceMetrics(
            operation="aggregated_batch",
            duration=total_duration,
            cache_hit=False,
            indicators_calculated=total_indicators,
            tickers_processed=aggregated.total_processed,
            metadata={
                "batches_aggregated": len(batch_results),
                "success_rate": aggregated.success_rate,
                "total_successful": len(aggregated.successful),
                "total_failed": len(aggregated.failed),
                "total_partial": len(aggregated.partial)
            }
        )

        return aggregated

    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        status = {}
        for name, breaker in self.circuit_breakers.items():
            status[name] = {
                "is_open": breaker.state.is_open,
                "failure_count": breaker.state.failure_count,
                "success_count": breaker.state.success_count,
                "total_requests": breaker.state.total_requests,
                "last_failure_time": breaker.state.last_failure_time.isoformat() if breaker.state.last_failure_time else None,
                "failure_threshold": breaker.state.failure_threshold,
                "recovery_timeout": breaker.state.recovery_timeout
            }
        return status

    def reset_circuit_breaker(self, name: str) -> bool:
        """Manually reset a circuit breaker."""
        if name in self.circuit_breakers:
            breaker = self.circuit_breakers[name]
            breaker.state.is_open = False
            breaker.state.failure_count = 0
            breaker.state.success_count = 0
            breaker.state.last_failure_time = None
            _logger.info("Circuit breaker %s manually reset", name)
            return True
        return False

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of the service."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "circuit_breakers": self.get_circuit_breaker_status()
        }

        # Check adapters
        for name, adapter in self.adapters.items():
            try:
                # Simple health check - try to get supported indicators
                if hasattr(adapter, 'get_supported_indicators'):
                    indicators = adapter.get_supported_indicators()
                    health_status["components"][name] = {
                        "status": "healthy",
                        "supported_indicators": len(indicators) if indicators else 0
                    }
                else:
                    health_status["components"][name] = {"status": "healthy"}
            except Exception as e:
                health_status["components"][name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"

        # Check if any circuit breakers are open
        open_breakers = [name for name, status in health_status["circuit_breakers"].items()
                        if status["is_open"]]
        if open_breakers:
            health_status["status"] = "degraded"
            health_status["open_circuit_breakers"] = open_breakers

        return health_status

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return self.performance_monitor.get_performance_report()

    async def run_benchmark(
        self,
        ticker: str = "AAPL",
        indicators: List[str] = None,
        iterations: int = 10
    ) -> Dict[str, Any]:
        """Run performance benchmark."""
        if indicators is None:
            indicators = ["RSI", "MACD", "BB_UPPER", "BB_MIDDLE", "BB_LOWER"]

        return await self.benchmark_runner.benchmark_single_ticker(
            ticker, indicators, iterations
        )

    async def run_batch_benchmark(
        self,
        tickers: List[str] = None,
        indicators: List[str] = None,
        batch_sizes: List[int] = None
    ) -> Dict[str, Any]:
        """Run batch processing benchmark."""
        if tickers is None:
            tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        if indicators is None:
            indicators = ["RSI", "MACD", "SMA_FAST"]

        return await self.benchmark_runner.benchmark_batch_processing(
            tickers, indicators, batch_sizes
        )

    def reset_performance_metrics(self):
        """Reset all performance metrics."""
        self.performance_monitor = PerformanceMonitor()
        _logger.info("Performance metrics reset")

    def shutdown(self):
        """Shutdown the service and cleanup resources."""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=True)
            _logger.info("Thread pool shutdown completed")

    def __del__(self):
        """Cleanup on object destruction."""
        try:
            self.shutdown()
        except Exception:
            pass  # Ignore errors during cleanup

    def get_available_indicators(self) -> Dict[str, List[str]]:
        """Get list of available indicators by category."""
        technical = []
        fundamental = []

        for name, meta in INDICATOR_META.items():
            if meta.kind == "tech":
                technical.append(name)
                # Add legacy names for compatibility
                if meta.legacy_names:
                    technical.extend(meta.legacy_names)
            else:
                fundamental.append(name)
                # Add legacy names for compatibility
                if meta.legacy_names:
                    fundamental.extend(meta.legacy_names)

        return {
            "technical": sorted(set(technical)),
            "fundamental": sorted(set(fundamental)),
            "all": sorted(set(technical + fundamental))
        }

    def get_indicator_description(self, indicator_name: str) -> Optional[str]:
        """Get description for an indicator."""
        canonical_name = get_canonical_name(indicator_name)
        meta = get_indicator_meta(canonical_name)
        return meta.description if meta else None

    def get_service_info(self) -> Dict[str, Any]:
        """Get service information and statistics."""
        return {
            "service": "Unified Indicator Service",
            "version": "2.0.0",
            "adapters": list(self.adapters.keys()),
            "config_info": self.config_manager.get_config_info(),
            "available_indicators": {
                "technical_count": len([m for m in INDICATOR_META.values() if m.kind == "tech"]),
                "fundamental_count": len([m for m in INDICATOR_META.values() if m.kind == "fund"]),
                "total_count": len(INDICATOR_META)
            },
            "performance_summary": self.performance_monitor.tracker.get_stats("_overall") if hasattr(self.performance_monitor.tracker, 'get_stats') else {},
            "circuit_breaker_status": self.get_circuit_breaker_status(),
            "batch_config": {
                "max_concurrent": self.batch_config.max_concurrent,
                "batch_size": self.batch_config.batch_size,
                "timeout_per_ticker": self.batch_config.timeout_per_ticker,
                "retry_attempts": self.batch_config.retry_attempts
            }
        }


# Maintain backward compatibility with existing class name
class IndicatorService(UnifiedIndicatorService):
    """Backward compatibility alias for UnifiedIndicatorService."""
    pass


# Global instance for easy access
_unified_service = None

def get_unified_indicator_service() -> UnifiedIndicatorService:
    """Get the global unified indicator service instance."""
    global _unified_service
    if _unified_service is None:
        _unified_service = UnifiedIndicatorService()
    return _unified_service
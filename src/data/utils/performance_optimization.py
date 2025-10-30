"""
Performance optimization module for data processing.

This module provides advanced performance optimization features including:
- Data compression (Parquet, Zstandard)
- Lazy loading for large datasets
- Parallel data processing
- Memory optimization
- Performance monitoring
"""

import time
import threading
import multiprocessing
from typing import Any, Optional, Dict, List, Union, Callable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
import logging
import pickle
import gzip
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

_logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for data operations."""
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    memory_usage_mb: float = 0.0
    data_size_mb: float = 0.0
    compression_ratio: float = 1.0
    throughput_mbps: float = 0.0
    cpu_usage_percent: float = 0.0

    def finalize(self):
        """Finalize metrics calculation."""
        if self.end_time is None:
            self.end_time = datetime.now()

        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

        if self.duration_ms > 0:
            self.throughput_mbps = (self.data_size_mb / self.duration_ms) * 1000

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()

    def add_metric(self, name: str, value: Any):
        """Add a custom metric."""
        setattr(self, name, value)


class DataCompressor:
    """Advanced data compression utilities."""

    def __init__(self, compression_level: int = 3, algorithm: str = "auto"):
        """
        Initialize data compressor.

        Args:
            compression_level: Compression level (1-22 for zstd, 1-9 for gzip)
            algorithm: Compression algorithm ('zstd', 'gzip', 'auto')
        """
        self.compression_level = compression_level
        self.algorithm = algorithm

        # Determine best available algorithm
        if algorithm == "auto":
            if ZSTD_AVAILABLE:
                self.algorithm = "zstd"
            else:
                self.algorithm = "gzip"

    def compress_data(self, data: bytes) -> bytes:
        """Compress data using the selected algorithm."""
        if self.algorithm == "zstd" and ZSTD_AVAILABLE:
            return zstd.compress(data, level=self.compression_level)
        else:
            return gzip.compress(data, compresslevel=self.compression_level)

    def decompress_data(self, data: bytes) -> bytes:
        """Decompress data."""
        if self.algorithm == "zstd" and ZSTD_AVAILABLE:
            return zstd.decompress(data)
        else:
            return gzip.decompress(data)

    def get_compression_ratio(self, original: bytes, compressed: bytes) -> float:
        """Calculate compression ratio."""
        return len(compressed) / len(original) if original else 1.0

    def compress_dataframe(
        self,
        df: pd.DataFrame,
        format: str = "parquet",
        compression: str = "snappy"
    ) -> bytes:
        """
        Compress DataFrame to bytes.

        Args:
            df: DataFrame to compress
            format: Output format ('parquet', 'pickle')
            compression: Compression method ('snappy', 'gzip', 'brotli')

        Returns:
            Compressed data as bytes
        """
        if format == "parquet" and PARQUET_AVAILABLE:
            # Use Parquet with compression
            buffer = pa.BufferOutputStream()
            table = pa.Table.from_pandas(df)
            pq.write_table(table, buffer, compression=compression)
            return buffer.getvalue().to_pybytes()
        else:
            # Use pickle with compression
            pickled = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
            return self.compress_data(pickled)

    def decompress_dataframe(self, data: bytes, format: str = "parquet") -> pd.DataFrame:
        """
        Decompress bytes to DataFrame.

        Args:
            data: Compressed data
            format: Input format ('parquet', 'pickle')

        Returns:
            Decompressed DataFrame
        """
        if format == "parquet" and PARQUET_AVAILABLE:
            # Read from Parquet
            buffer = pa.BufferReader(data)
            table = pq.read_table(buffer)
            return table.to_pandas()
        else:
            # Decompress pickle
            decompressed = self.decompress_data(data)
            return pickle.loads(decompressed)


class LazyDataLoader:
    """Lazy loading for large datasets."""

    def __init__(
        self,
        file_path: Union[str, Path],
        chunk_size: int = 10000,
        format: str = "auto"
    ):
        """
        Initialize lazy data loader.

        Args:
            file_path: Path to data file
            chunk_size: Number of rows per chunk
            format: File format ('csv', 'parquet', 'auto')
        """
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.format = format

        # Determine format
        if format == "auto":
            if self.file_path.suffix.lower() == '.parquet':
                self.format = "parquet"
            else:
                self.format = "csv"

        self._total_rows = None
        self._columns = None

    def __len__(self) -> int:
        """Get total number of rows."""
        if self._total_rows is None:
            self._count_rows()
        return self._total_rows

    def _count_rows(self):
        """Count total rows in file."""
        if self.format == "parquet" and PARQUET_AVAILABLE:
            self._total_rows = pq.read_metadata(self.file_path).num_rows
        else:
            # For CSV, we need to count lines (approximate)
            with open(self.file_path, 'r') as f:
                self._total_rows = sum(1 for _ in f) - 1  # Subtract header

    def get_columns(self) -> List[str]:
        """Get column names."""
        if self._columns is None:
            if self.format == "parquet" and PARQUET_AVAILABLE:
                metadata = pq.read_metadata(self.file_path)
                self._columns = [col.name for col in metadata.schema]
            else:
                # Read first line of CSV
                with open(self.file_path, 'r') as f:
                    header = f.readline().strip()
                    self._columns = header.split(',')

        return self._columns

    def iter_chunks(self) -> Iterator[pd.DataFrame]:
        """Iterate over data in chunks."""
        if self.format == "parquet" and PARQUET_AVAILABLE:
            # Use Parquet's built-in chunking
            for chunk in pq.ParquetFile(self.file_path).iter_batches(batch_size=self.chunk_size):
                yield chunk.to_pandas()
        else:
            # Use pandas chunking for CSV
            for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
                yield chunk

    def get_chunk(self, chunk_index: int) -> Optional[pd.DataFrame]:
        """Get specific chunk by index."""
        if self.format == "parquet" and PARQUET_AVAILABLE:
            # Calculate row range for chunk
            start_row = chunk_index * self.chunk_size
            end_row = start_row + self.chunk_size

            # Read specific rows
            table = pq.read_table(self.file_path, row_groups=[chunk_index])
            return table.to_pandas()
        else:
            # For CSV, we need to skip rows
            skip_rows = chunk_index * self.chunk_size + 1  # +1 for header
            try:
                return pd.read_csv(self.file_path, skiprows=range(1, skip_rows), nrows=self.chunk_size)
            except Exception:
                return None

    def filter_chunks(self, filter_func: Callable[[pd.DataFrame], bool]) -> Iterator[pd.DataFrame]:
        """Filter chunks based on a function."""
        for chunk in self.iter_chunks():
            if filter_func(chunk):
                yield chunk

    def map_chunks(self, map_func: Callable[[pd.DataFrame], pd.DataFrame]) -> Iterator[pd.DataFrame]:
        """Apply function to each chunk."""
        for chunk in self.iter_chunks():
            yield map_func(chunk)


class ParallelProcessor:
    """Parallel data processing utilities."""

    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = True,
        chunk_size: int = 1000
    ):
        """
        Initialize parallel processor.

        Args:
            max_workers: Maximum number of workers
            use_processes: Use processes instead of threads
            chunk_size: Size of data chunks for processing
        """
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.use_processes = use_processes
        self.chunk_size = chunk_size

    def process_dataframe(
        self,
        df: pd.DataFrame,
        process_func: Callable[[pd.DataFrame], pd.DataFrame],
        **kwargs
    ) -> pd.DataFrame:
        """
        Process DataFrame in parallel.

        Args:
            df: Input DataFrame
            process_func: Function to apply to each chunk
            **kwargs: Additional arguments for process_func

        Returns:
            Processed DataFrame
        """
        # Split DataFrame into chunks
        chunks = [df[i:i + self.chunk_size] for i in range(0, len(df), self.chunk_size)]

        # Process chunks in parallel
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        with executor_class(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_chunk = {
                executor.submit(process_func, chunk, **kwargs): i
                for i, chunk in enumerate(chunks)
            }

            # Collect results
            results = [None] * len(chunks)
            for future in as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                try:
                    results[chunk_index] = future.result()
                except Exception as e:
                    _logger.exception("Error processing chunk %s:", chunk_index)
                    results[chunk_index] = chunks[chunk_index]  # Return original chunk

        # Combine results
        return pd.concat(results, ignore_index=True)

    def process_lazy_loader(
        self,
        loader: LazyDataLoader,
        process_func: Callable[[pd.DataFrame], pd.DataFrame],
        **kwargs
    ) -> Iterator[pd.DataFrame]:
        """
        Process lazy loader in parallel.

        Args:
            loader: LazyDataLoader instance
            process_func: Function to apply to each chunk
            **kwargs: Additional arguments for process_func

        Returns:
            Iterator of processed chunks
        """
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {}
            for i, chunk in enumerate(loader.iter_chunks()):
                future = executor.submit(process_func, chunk, **kwargs)
                future_to_chunk[future] = i

            # Yield results as they complete
            for future in as_completed(future_to_chunk):
                try:
                    yield future.result()
                except Exception as e:
                    _logger.exception("Error processing chunk:")

    def map_reduce(
        self,
        data: Union[pd.DataFrame, LazyDataLoader],
        map_func: Callable[[pd.DataFrame], Any],
        reduce_func: Callable[[List[Any]], Any],
        **kwargs
    ) -> Any:
        """
        Perform map-reduce operation on data.

        Args:
            data: Input data (DataFrame or LazyDataLoader)
            map_func: Function to apply to each chunk
            reduce_func: Function to combine results
            **kwargs: Additional arguments for map_func

        Returns:
            Reduced result
        """
        if isinstance(data, pd.DataFrame):
            # Process DataFrame
            chunks = [data[i:i + self.chunk_size] for i in range(0, len(data), self.chunk_size)]
        else:
            # Process LazyDataLoader
            chunks = list(data.iter_chunks())

        # Map phase
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        with executor_class(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(map_func, chunk, **kwargs)
                for chunk in chunks
            ]

            # Collect results
            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    _logger.exception("Error in map phase:")

        # Reduce phase
        return reduce_func(results)


class MemoryOptimizer:
    """Memory optimization utilities."""

    def __init__(self):
        """Initialize memory optimizer."""
        self.memory_usage = {}

    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.

        Args:
            df: Input DataFrame

        Returns:
            Memory-optimized DataFrame
        """
        optimized_df = df.copy()

        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype

            # Optimize numeric columns
            if col_type in ['int64', 'float64']:
                if col_type == 'int64':
                    c_min = optimized_df[col].min()
                    c_max = optimized_df[col].max()

                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        optimized_df[col] = optimized_df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        optimized_df[col] = optimized_df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        optimized_df[col] = optimized_df[col].astype(np.int32)

                elif col_type == 'float64':
                    optimized_df[col] = optimized_df[col].astype(np.float32)

            # Optimize object columns
            elif col_type == 'object':
                if optimized_df[col].nunique() / len(optimized_df) < 0.5:
                    optimized_df[col] = optimized_df[col].astype('category')

        return optimized_df

    def get_memory_usage(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get memory usage information for DataFrame."""
        memory_usage = df.memory_usage(deep=True)
        return {
            'total_mb': memory_usage.sum() / 1024 / 1024,
            'per_column': {col: usage / 1024 / 1024 for col, usage in memory_usage.items()},
            'dtypes': df.dtypes.to_dict()
        }

    def estimate_memory_reduction(self, df: pd.DataFrame) -> Dict[str, float]:
        """Estimate potential memory reduction."""
        current_usage = self.get_memory_usage(df)
        optimized_df = self.optimize_dataframe(df)
        optimized_usage = self.get_memory_usage(optimized_df)

        reduction_mb = current_usage['total_mb'] - optimized_usage['total_mb']
        reduction_percent = (reduction_mb / current_usage['total_mb']) * 100

        return {
            'current_mb': current_usage['total_mb'],
            'optimized_mb': optimized_usage['total_mb'],
            'reduction_mb': reduction_mb,
            'reduction_percent': reduction_percent
        }


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: List[PerformanceMetrics] = []
        self._lock = threading.Lock()

    def start_operation(self, operation_name: str) -> PerformanceMetrics:
        """Start monitoring an operation."""
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=datetime.now()
        )

        with self._lock:
            self.metrics.append(metrics)

        return metrics

    def end_operation(self, metrics: PerformanceMetrics, **kwargs):
        """End monitoring an operation."""
        metrics.end_time = datetime.now()

        # Update additional metrics
        for key, value in kwargs.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)

        metrics.finalize()

    def get_operation_metrics(self, operation_name: str) -> List[PerformanceMetrics]:
        """Get metrics for a specific operation."""
        with self._lock:
            return [m for m in self.metrics if m.operation_name == operation_name]

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self._lock:
            if not self.metrics:
                return {}

            # Group by operation
            operation_groups = {}
            for metric in self.metrics:
                if metric.operation_name not in operation_groups:
                    operation_groups[metric.operation_name] = []
                operation_groups[metric.operation_name].append(metric)

            # Calculate statistics
            summary = {}
            for operation, metrics_list in operation_groups.items():
                durations = [m.duration_ms for m in metrics_list if m.duration_ms > 0]
                throughputs = [m.throughput_mbps for m in metrics_list if m.throughput_mbps > 0]

                summary[operation] = {
                    'count': len(metrics_list),
                    'avg_duration_ms': np.mean(durations) if durations else 0,
                    'min_duration_ms': np.min(durations) if durations else 0,
                    'max_duration_ms': np.max(durations) if durations else 0,
                    'avg_throughput_mbps': np.mean(throughputs) if throughputs else 0,
                    'total_data_mb': sum(m.data_size_mb for m in metrics_list)
                }

            # Add backward-compatibility aggregate
            summary['total_operations'] = sum(v['count'] for k, v in summary.items() if isinstance(v, dict) and 'count' in v)

            return summary

    def clear_metrics(self):
        """Clear all metrics."""
        with self._lock:
            self.metrics.clear()

    def get_metrics(self) -> List[PerformanceMetrics]:
        """Get all metrics."""
        with self._lock:
            return self.metrics.copy()


# Global instances
_performance_monitor = PerformanceMonitor()
_memory_optimizer = MemoryOptimizer()
_data_compressor = DataCompressor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    return _performance_monitor


def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer instance."""
    return _memory_optimizer


def get_data_compressor() -> DataCompressor:
    """Get global data compressor instance."""
    return _data_compressor


def optimize_dataframe_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame for better performance."""
    monitor = get_performance_monitor()
    optimizer = get_memory_optimizer()

    # Start monitoring
    metrics = monitor.start_operation("dataframe_optimization")

    try:
        # Get initial memory usage
        initial_usage = optimizer.get_memory_usage(df)
        metrics.data_size_mb = initial_usage['total_mb']

        # Optimize DataFrame
        optimized_df = optimizer.optimize_dataframe(df)

        # Get final memory usage
        final_usage = optimizer.get_memory_usage(optimized_df)
        metrics.memory_usage_mb = final_usage['total_mb']

        # Calculate compression ratio
        metrics.compression_ratio = final_usage['total_mb'] / initial_usage['total_mb']

        return optimized_df

    finally:
        monitor.end_operation(metrics)


def compress_dataframe_efficiently(
    df: pd.DataFrame,
    format: str = "parquet",
    compression: str = "snappy"
) -> bytes:
    """Compress DataFrame efficiently."""
    monitor = get_performance_monitor()
    compressor = get_data_compressor()

    # Start monitoring
    metrics = monitor.start_operation("dataframe_compression")

    try:
        # Get data size
        memory_usage = get_memory_optimizer().get_memory_usage(df)
        metrics.data_size_mb = memory_usage['total_mb']

        # Compress data
        compressed_data = compressor.compress_dataframe(df, format, compression)

        # Calculate compression ratio
        metrics.compression_ratio = len(compressed_data) / (memory_usage['total_mb'] * 1024 * 1024)

        return compressed_data

    finally:
        monitor.end_operation(metrics)

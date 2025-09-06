"""
Data streaming module with WebSocket connection pooling and backpressure handling.

This module provides advanced real-time data streaming capabilities including:
- WebSocket connection pooling
- Backpressure handling
- Stream multiplexing
- Real-time data processing pipelines
"""

import asyncio
import json
import time
import threading
from typing import Dict, List, Optional, Callable, Any, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
import asyncio
import websockets
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

_logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for data streams."""
    url: str
    symbol: str
    interval: str
    max_connections: int = 3
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    ping_interval: int = 20
    ping_timeout: int = 10
    message_queue_size: int = 10000
    backpressure_threshold: int = 1000
    enable_compression: bool = True


@dataclass
class StreamMetrics:
    """Metrics for data streams."""
    messages_received: int = 0
    messages_processed: int = 0
    messages_dropped: int = 0
    connection_attempts: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    reconnections: int = 0
    last_message_time: Optional[datetime] = None
    avg_message_rate: float = 0.0
    queue_size: int = 0
    processing_latency_ms: float = 0.0


class WebSocketConnection:
    """Individual WebSocket connection with reconnection logic."""

    def __init__(
        self,
        url: str,
        on_message: Callable[[str], None],
        on_error: Callable[[Exception], None],
        on_close: Callable[[], None],
        config: StreamConfig
    ):
        """
        Initialize WebSocket connection.

        Args:
            url: WebSocket URL
            on_message: Message handler callback
            on_error: Error handler callback
            on_close: Close handler callback
            config: Stream configuration
        """
        self.url = url
        self.on_message = on_message
        self.on_error = on_error
        self.on_close = on_close
        self.config = config

        self.ws = None
        self.is_connected = False
        self.is_closing = False
        self.reconnect_delay = config.reconnect_delay
        self.last_message_time = None

        self._connection_thread = None
        self._metrics = StreamMetrics()

    def connect(self) -> bool:
        """Connect to WebSocket."""
        try:
            self._metrics.connection_attempts += 1

            # Start async event loop in separate thread
            self.loop = asyncio.new_event_loop()
            self._connection_thread = threading.Thread(
                target=self._run_websocket_loop,
                daemon=True
            )
            self._connection_thread.start()

            # Wait for connection
            for _ in range(50):  # 5 seconds timeout
                if self.is_connected:
                    self._metrics.successful_connections += 1
                    return True
                time.sleep(0.1)

            self._metrics.failed_connections += 1
            return False

        except Exception as e:
            self._metrics.failed_connections += 1
            self.on_error(e)
            return False

    def _run_websocket_loop(self):
        """Run the WebSocket event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._websocket_handler())

    async def _websocket_handler(self):
        """Handle WebSocket connection and messages."""
        try:
            async with websockets.connect(
                self.url,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                close_timeout=10
            ) as websocket:
                self.ws = websocket
                self.is_connected = True
                _logger.info(f"WebSocket connected: {self.url}")

                async for message in websocket:
                    await self._on_message(message)

        except websockets.exceptions.ConnectionClosed:
            _logger.warning(f"WebSocket disconnected: {self.url}")
            self.is_connected = False
            if not self.is_closing:
                self.on_close()
        except Exception as e:
            _logger.error(f"WebSocket error: {e}")
            self.is_connected = False
            self.on_error(e)

    async def _on_message(self, message):
        """WebSocket message received."""
        try:
            self.last_message_time = datetime.now()
            self._metrics.messages_received += 1
            self.on_message(message)
        except Exception as e:
            _logger.error(f"Error handling message: {e}")

    def disconnect(self):
        """Disconnect WebSocket."""
        self.is_closing = True
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.ws:
            self.ws = None
        self.is_connected = False

    def get_metrics(self) -> StreamMetrics:
        """Get connection metrics."""
        return self._metrics


class ConnectionPool:
    """Pool of WebSocket connections for load balancing and redundancy."""

    def __init__(self, config: StreamConfig, message_handler: Callable[[str], None]):
        """
        Initialize connection pool.

        Args:
            config: Stream configuration
            message_handler: Message handler callback
        """
        self.config = config
        self.message_handler = message_handler

        self.connections: List[WebSocketConnection] = []
        self.active_connections: Set[WebSocketConnection] = set()
        self.connection_lock = threading.Lock()

        self._message_queue = queue.Queue(maxsize=config.message_queue_size)
        self._processing_thread = None
        self._is_running = False

        # Initialize connections
        self._initialize_connections()

    def _initialize_connections(self):
        """Initialize WebSocket connections."""
        for i in range(self.config.max_connections):
            connection = WebSocketConnection(
                url=self.config.url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_connection_closed,
                config=self.config
            )
            self.connections.append(connection)

    def _on_message(self, message: str):
        """Handle incoming message."""
        try:
            # Add to processing queue
            self._message_queue.put_nowait(message)
        except queue.Full:
            _logger.warning("Message queue full, dropping message")

    def _on_error(self, error: Exception):
        """Handle connection error."""
        _logger.error(f"Connection pool error: {error}")

    def _on_connection_closed(self):
        """Handle connection closure."""
        with self.connection_lock:
            # Remove from active connections
            for conn in list(self.active_connections):
                if not conn.is_connected:
                    self.active_connections.remove(conn)

            # Attempt to reconnect
            self._reconnect_inactive_connections()

    def _reconnect_inactive_connections(self):
        """Reconnect inactive connections."""
        for connection in self.connections:
            if not connection.is_connected and not connection.is_closing:
                if connection.connect():
                    with self.connection_lock:
                        self.active_connections.add(connection)

    def start(self) -> bool:
        """Start the connection pool."""
        try:
            self._is_running = True

            # Connect all connections
            for connection in self.connections:
                if connection.connect():
                    with self.connection_lock:
                        self.active_connections.add(connection)

            # Start message processing thread
            self._processing_thread = threading.Thread(
                target=self._process_messages,
                daemon=True
            )
            self._processing_thread.start()

            _logger.info(f"Connection pool started with {len(self.active_connections)} connections")
            return True

        except Exception as e:
            _logger.error(f"Failed to start connection pool: {e}")
            return False

    def _process_messages(self):
        """Process messages from the queue."""
        while self._is_running:
            try:
                # Get message with timeout
                message = self._message_queue.get(timeout=1.0)

                # Handle backpressure
                if self._message_queue.qsize() > self.config.backpressure_threshold:
                    _logger.warning("Backpressure detected, processing messages faster")

                # Process message
                start_time = time.time()
                self.message_handler(message)
                processing_time = (time.time() - start_time) * 1000

                # Update metrics
                for conn in self.connections:
                    conn._metrics.messages_processed += 1
                    conn._metrics.processing_latency_ms = (
                        (conn._metrics.processing_latency_ms + processing_time) / 2
                    )

            except queue.Empty:
                continue
            except Exception as e:
                _logger.error(f"Error processing message: {e}")

    def stop(self):
        """Stop the connection pool."""
        self._is_running = False

        # Disconnect all connections
        for connection in self.connections:
            connection.disconnect()

        # Wait for processing thread
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)

        _logger.info("Connection pool stopped")

    def get_metrics(self) -> Dict[str, Any]:
        """Get pool metrics."""
        total_messages = sum(conn._metrics.messages_received for conn in self.connections)
        total_processed = sum(conn._metrics.messages_processed for conn in self.connections)

        return {
            'active_connections': len(self.active_connections),
            'total_connections': len(self.connections),
            'messages_received': total_messages,
            'messages_processed': total_processed,
            'queue_size': self._message_queue.qsize(),
            'is_running': self._is_running
        }


class DataStreamProcessor:
    """Processes and transforms real-time data streams."""

    def __init__(self, config: StreamConfig):
        """
        Initialize data stream processor.

        Args:
            config: Stream configuration
        """
        self.config = config
        self.processors: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = []
        self.filters: List[Callable[[Dict[str, Any]], bool]] = []
        self.subscribers: List[Callable[[Dict[str, Any]], None]] = []

        self._metrics = StreamMetrics()
        self._lock = threading.Lock()

    def add_processor(self, processor: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Add data processor."""
        self.processors.append(processor)

    def add_filter(self, filter_func: Callable[[Dict[str, Any]], bool]):
        """Add data filter."""
        self.filters.append(filter_func)

    def subscribe(self, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to processed data."""
        self.subscribers.append(callback)

    def process_message(self, message: str):
        """Process incoming message."""
        start_time = time.time()

        try:
            # Parse message
            data = json.loads(message)

            # Apply filters
            for filter_func in self.filters:
                if not filter_func(data):
                    return  # Message filtered out

            # Apply processors
            for processor in self.processors:
                data = processor(data)

            # Notify subscribers
            for subscriber in self.subscribers:
                try:
                    subscriber(data)
                except Exception as e:
                    _logger.error(f"Subscriber error: {e}")

            # Update metrics
            with self._lock:
                self._metrics.messages_processed += 1
                self._metrics.last_message_time = datetime.now()
                self._metrics.processing_latency_ms = (
                    (self._metrics.processing_latency_ms + (time.time() - start_time) * 1000) / 2
                )

        except json.JSONDecodeError as e:
            _logger.error(f"Failed to parse message: {e}")
        except Exception as e:
            _logger.error(f"Error processing message: {e}")

    def get_metrics(self) -> StreamMetrics:
        """Get processor metrics."""
        with self._lock:
            return self._metrics


class StreamMultiplexer:
    """Multiplexes multiple data streams into a single interface."""

    def __init__(self):
        """Initialize stream multiplexer."""
        self.streams: Dict[str, ConnectionPool] = {}
        self.processors: Dict[str, DataStreamProcessor] = {}
        self.global_subscribers: List[Callable[[str, Dict[str, Any]], None]] = []

        self._lock = threading.Lock()
        self._is_running = False

    def add_stream(
        self,
        stream_id: str,
        config: StreamConfig,
        processor: Optional[DataStreamProcessor] = None
    ) -> bool:
        """
        Add a new data stream.

        Args:
            stream_id: Unique stream identifier
            config: Stream configuration
            processor: Optional data processor
            **kwargs: Additional configuration

        Returns:
            True if stream added successfully
        """
        try:
            with self._lock:
                if stream_id in self.streams:
                    _logger.warning(f"Stream {stream_id} already exists")
                    return False

                # Create processor if not provided
                if processor is None:
                    processor = DataStreamProcessor(config)

                # Create connection pool
                pool = ConnectionPool(config, processor.process_message)

                self.streams[stream_id] = pool
                self.processors[stream_id] = processor

                # Start stream if multiplexer is running
                if self._is_running:
                    pool.start()

                _logger.info(f"Added stream: {stream_id}")
                return True

        except Exception as e:
            _logger.error(f"Failed to add stream {stream_id}: {e}")
            return False

    def remove_stream(self, stream_id: str) -> bool:
        """Remove a data stream."""
        try:
            with self._lock:
                if stream_id not in self.streams:
                    return False

                # Stop and remove stream
                self.streams[stream_id].stop()
                del self.streams[stream_id]
                del self.processors[stream_id]

                _logger.info(f"Removed stream: {stream_id}")
                return True

        except Exception as e:
            _logger.error(f"Failed to remove stream {stream_id}: {e}")
            return False

    def subscribe(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Subscribe to all streams."""
        self.global_subscribers.append(callback)

        # Subscribe to existing processors
        for stream_id, processor in self.processors.items():
            processor.subscribe(lambda data, sid=stream_id: callback(sid, data))

    def start(self) -> bool:
        """Start all streams."""
        try:
            with self._lock:
                self._is_running = True

                for stream_id, pool in self.streams.items():
                    if not pool.start():
                        _logger.error(f"Failed to start stream: {stream_id}")
                        return False

                _logger.info(f"Started {len(self.streams)} streams")
                return True

        except Exception as e:
            _logger.error(f"Failed to start multiplexer: {e}")
            return False

    def stop(self):
        """Stop all streams."""
        with self._lock:
            self._is_running = False

            for stream_id, pool in self.streams.items():
                pool.stop()

            _logger.info("Stopped all streams")

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all streams."""
        metrics = {
            'total_streams': len(self.streams),
            'active_streams': sum(1 for pool in self.streams.values() if pool._is_running)
        }

        for stream_id, pool in self.streams.items():
            metrics[stream_id] = {
                'pool': pool.get_metrics(),
                'processor': self.processors[stream_id].get_metrics()
            }

        return metrics


class BackpressureHandler:
    """Handles backpressure in data streams."""

    def __init__(self, max_queue_size: int = 10000, drop_threshold: float = 0.8):
        """
        Initialize backpressure handler.

        Args:
            max_queue_size: Maximum queue size
            drop_threshold: Threshold for dropping messages (0.0-1.0)
        """
        self.max_queue_size = max_queue_size
        self.drop_threshold = drop_threshold
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.dropped_messages = 0
        self.total_messages = 0

    def should_drop_message(self) -> bool:
        """Check if message should be dropped due to backpressure."""
        queue_ratio = self.queue.qsize() / self.max_queue_size
        return queue_ratio > self.drop_threshold

    def add_message(self, message: Any) -> bool:
        """
        Add message to queue.

        Returns:
            True if message was added, False if dropped
        """
        self.total_messages += 1

        if self.should_drop_message():
            self.dropped_messages += 1
            return False

        try:
            self.queue.put_nowait(message)
            return True
        except queue.Full:
            self.dropped_messages += 1
            return False

    def get_message(self, timeout: float = 1.0) -> Optional[Any]:
        """Get message from queue."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get backpressure metrics."""
        return {
            'queue_size': self.queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'queue_utilization': self.queue.qsize() / self.max_queue_size,
            'total_messages': self.total_messages,
            'dropped_messages': self.dropped_messages,
            'drop_rate': self.dropped_messages / self.total_messages if self.total_messages > 0 else 0.0
        }


# Global stream multiplexer instance
_stream_multiplexer: Optional[StreamMultiplexer] = None


def get_stream_multiplexer() -> StreamMultiplexer:
    """Get global stream multiplexer instance."""
    global _stream_multiplexer

    if _stream_multiplexer is None:
        _stream_multiplexer = StreamMultiplexer()

    return _stream_multiplexer


def create_stream_config(
    url: str,
    symbol: str,
    interval: str,
    **kwargs
) -> StreamConfig:
    """Create a stream configuration."""
    return StreamConfig(url=url, symbol=symbol, interval=interval, **kwargs)

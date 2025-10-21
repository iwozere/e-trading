"""
Notification Service Message Processor

Asynchronous message processing engine with priority handling, worker pools,
and graceful shutdown. Processes messages from the queue and delivers them
through appropriate channels.
"""

import asyncio
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import signal
import threading
import time

from src.notification.service.message_queue import MessageQueue, QueuedMessage, message_queue
from src.notification.service.config import config
from src.notification.service.fallback_manager import FallbackManager
from src.notification.service.health_monitor import health_monitor
from src.notification.channels.base import channel_registry, MessageContent
from src.data.db.models.model_notification import MessagePriority, MessageStatus
from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


@dataclass
class ProcessingResult:
    """Result of message processing."""
    message_id: int
    success: bool
    error_message: Optional[str] = None
    delivery_results: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[int] = None


class MessageProcessor:
    """
    Asynchronous message processing engine.

    Handles message processing with priority queues, worker pools,
    and graceful shutdown capabilities.
    """

    def __init__(
        self,
        max_workers: int = None,
        batch_size: int = None,
        batch_timeout_seconds: int = None,
        retry_delay_minutes: int = None
    ):
        """
        Initialize the message processor.

        Args:
            max_workers: Maximum number of concurrent workers
            batch_size: Maximum messages per batch
            batch_timeout_seconds: Maximum time to wait for batch completion
            retry_delay_minutes: Minutes to wait before retrying failed messages
        """
        self.max_workers = max_workers or config.processing.max_workers
        self.batch_size = batch_size or config.processing.batch_size
        self.batch_timeout_seconds = batch_timeout_seconds or config.processing.batch_timeout_seconds
        self.retry_delay_minutes = retry_delay_minutes or config.processing.retry_delay_minutes

        self.queue = message_queue
        self._logger = setup_logger(f"{__name__}.MessageProcessor")

        # Initialize fallback manager
        self.fallback_manager = FallbackManager(health_monitor)

        # Channel instances cache
        self._channel_instances = {}

        # Processing state
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._worker_tasks: Set[asyncio.Task] = set()
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Statistics
        self._stats = {
            'messages_processed': 0,
            'messages_delivered': 0,
            'messages_failed': 0,
            'processing_errors': 0,
            'fallback_attempts': 0,
            'dead_letter_messages': 0,
            'start_time': None,
            'last_activity': None
        }

        self._logger.info(
            "Message processor initialized: max_workers=%s, batch_size=%s",
            self.max_workers, self.batch_size
        )

    async def start(self):
        """Start the message processor."""
        if self._running:
            self._logger.warning("Message processor is already running")
            return

        self._running = True
        self._shutdown_event.clear()
        self._stats['start_time'] = datetime.now(timezone.utc)

        self._logger.info("Starting message processor...")

        # Initialize channel instances
        await self._initialize_channel_instances()

        # Start worker tasks
        self._worker_tasks.add(
            asyncio.create_task(self._high_priority_worker())
        )
        self._worker_tasks.add(
            asyncio.create_task(self._normal_priority_worker())
        )
        self._worker_tasks.add(
            asyncio.create_task(self._retry_worker())
        )
        self._worker_tasks.add(
            asyncio.create_task(self._fallback_retry_worker())
        )
        self._worker_tasks.add(
            asyncio.create_task(self._cleanup_worker())
        )

        self._logger.info("Message processor started with %s workers", len(self._worker_tasks))

    async def _initialize_channel_instances(self):
        """Initialize channel instances for delivery."""
        try:
            # Get channel configurations from config
            channel_configs = getattr(config, 'channels', {})

            for channel_name, channel_config in channel_configs.items():
                if channel_config.get('enabled', True):
                    try:
                        # Get channel instance from registry
                        channel_instance = channel_registry.get_channel(channel_name, channel_config)
                        self._channel_instances[channel_name] = channel_instance

                        self._logger.info("Initialized channel: %s", channel_name)
                    except Exception as e:
                        self._logger.error("Failed to initialize channel %s: %s", channel_name, e)

            # Configure default fallback rules
            await self._configure_default_fallback_rules()

        except Exception as e:
            self._logger.error("Error initializing channel instances: %s", e)

    async def _configure_default_fallback_rules(self):
        """Configure default fallback rules for channels."""
        try:
            # Example fallback configuration - this would come from config in real implementation
            from src.notification.service.fallback_manager import FallbackRule, FallbackStrategy

            # Configure Telegram -> Email fallback
            if 'telegram' in self._channel_instances and 'email' in self._channel_instances:
                telegram_fallback = FallbackRule(
                    primary_channel='telegram',
                    fallback_channels=['email'],
                    strategy=FallbackStrategy.PRIORITY_ORDER,
                    max_attempts=2
                )
                self.fallback_manager.configure_fallback_rule(telegram_fallback)

            # Configure Email -> SMS fallback
            if 'email' in self._channel_instances and 'sms' in self._channel_instances:
                email_fallback = FallbackRule(
                    primary_channel='email',
                    fallback_channels=['sms'],
                    strategy=FallbackStrategy.PRIORITY_ORDER,
                    max_attempts=2
                )
                self.fallback_manager.configure_fallback_rule(email_fallback)

            # Set global fallback order
            available_channels = list(self._channel_instances.keys())
            self.fallback_manager.set_global_fallback_channels(available_channels)

        except Exception as e:
            self._logger.error("Error configuring fallback rules: %s", e)

    async def shutdown(self, timeout: int = 30):
        """
        Gracefully shutdown the message processor.

        Args:
            timeout: Maximum time to wait for shutdown in seconds
        """
        if not self._running:
            return

        self._logger.info("Shutting down message processor...")
        self._running = False
        self._shutdown_event.set()

        # Cancel all worker tasks
        for task in self._worker_tasks:
            task.cancel()

        # Wait for tasks to complete or timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._worker_tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self._logger.warning("Shutdown timeout reached, forcing termination")

        # Shutdown executor
        self._executor.shutdown(wait=True, timeout=timeout)

        self._worker_tasks.clear()
        self._logger.info("Message processor shutdown complete")

    async def _high_priority_worker(self):
        """Worker for processing high priority messages."""
        self._logger.info("High priority worker started")

        while self._running:
            try:
                # Check for high priority messages more frequently
                messages = self.queue.dequeue_high_priority(limit=5)

                if messages:
                    self._logger.info("Processing %s high priority messages", len(messages))
                    await self._process_messages_batch(messages, is_high_priority=True)
                else:
                    # Short sleep for high priority worker
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("High priority worker error: %s", e)
                self._stats['processing_errors'] += 1
                await asyncio.sleep(5)  # Back off on error

        self._logger.info("High priority worker stopped")

    async def _normal_priority_worker(self):
        """Worker for processing normal and low priority messages."""
        self._logger.info("Normal priority worker started")

        while self._running:
            try:
                # Process normal and low priority messages in batches
                messages = self.queue.dequeue(limit=self.batch_size)

                if messages:
                    self._logger.info("Processing %s normal priority messages", len(messages))
                    await self._process_messages_batch(messages, is_high_priority=False)
                else:
                    # Longer sleep for normal priority worker
                    await asyncio.sleep(5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("Normal priority worker error: %s", e)
                self._stats['processing_errors'] += 1
                await asyncio.sleep(10)  # Back off on error

        self._logger.info("Normal priority worker stopped")

    async def _retry_worker(self):
        """Worker for processing retry messages."""
        self._logger.info("Retry worker started")

        while self._running:
            try:
                # Check for retry messages less frequently
                messages = self.queue.dequeue_for_retry(
                    limit=10,
                    retry_delay_minutes=self.retry_delay_minutes
                )

                if messages:
                    self._logger.info("Processing %s retry messages", len(messages))
                    await self._process_messages_batch(messages, is_retry=True)
                else:
                    # Longer sleep for retry worker
                    await asyncio.sleep(30)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("Retry worker error: %s", e)
                self._stats['processing_errors'] += 1
                await asyncio.sleep(30)  # Back off on error

        self._logger.info("Retry worker stopped")

    async def _fallback_retry_worker(self):
        """Worker for processing fallback retry queue."""
        self._logger.info("Fallback retry worker started")

        while self._running:
            try:
                # Process retry queue every 30 seconds
                results = await self.fallback_manager.process_retry_queue(self._channel_instances)

                if results["processed"] > 0:
                    self._logger.info(
                        "Processed %s retry messages: %s succeeded, %s failed, %s requeued",
                        results["processed"], results["succeeded"],
                        results["failed"], results["requeued"]
                    )

                    # Update stats
                    self._stats["messages_delivered"] += results["succeeded"]
                    self._stats["messages_failed"] += results["failed"]

                # Sleep between retry processing cycles
                await asyncio.sleep(30)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("Fallback retry worker error: %s", e)
                await asyncio.sleep(30)

        self._logger.info("Fallback retry worker stopped")

    async def _cleanup_worker(self):
        """Worker for periodic cleanup tasks."""
        self._logger.info("Cleanup worker started")

        cleanup_interval = config.processing.cleanup_interval_hours * 3600  # Convert to seconds
        last_cleanup = time.time()

        while self._running:
            try:
                current_time = time.time()

                # Run cleanup if interval has passed
                if current_time - last_cleanup >= cleanup_interval:
                    self._logger.info("Running periodic cleanup...")

                    # Clean up old dead letter messages
                    try:
                        cleaned_count = await self.fallback_manager.cleanup_old_dead_letters()
                        if cleaned_count > 0:
                            self._logger.info("Cleaned up %s old dead letter messages", cleaned_count)
                    except Exception as e:
                        self._logger.error("Error cleaning up dead letters: %s", e)

                    # TODO: Implement additional cleanup tasks
                    # - Clean up old delivered messages
                    # - Update channel health
                    # - Refresh rate limits

                    last_cleanup = current_time
                    self._logger.info("Periodic cleanup completed")

                # Sleep for 1 hour between cleanup checks
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("Cleanup worker error: %s", e)
                await asyncio.sleep(3600)  # Continue on error

        self._logger.info("Cleanup worker stopped")

    async def _process_messages_batch(
        self,
        messages: List[QueuedMessage],
        is_high_priority: bool = False,
        is_retry: bool = False
    ):
        """
        Process a batch of messages.

        Args:
            messages: List of messages to process
            is_high_priority: Whether these are high priority messages
            is_retry: Whether these are retry messages
        """
        if not messages:
            return

        start_time = time.time()

        try:
            # Process messages concurrently
            tasks = []
            for message in messages:
                task = asyncio.create_task(
                    self._process_single_message(message, is_high_priority, is_retry)
                )
                tasks.append(task)

            # Wait for all messages to complete with timeout
            timeout = self.batch_timeout_seconds if not is_high_priority else 10
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )

            # Update statistics
            for result in results:
                if isinstance(result, ProcessingResult):
                    self._stats['messages_processed'] += 1
                    if result.success:
                        self._stats['messages_delivered'] += 1
                    else:
                        self._stats['messages_failed'] += 1
                elif isinstance(result, Exception):
                    self._stats['processing_errors'] += 1
                    self._logger.error("Message processing exception: %s", result)

            processing_time = (time.time() - start_time) * 1000
            self._stats['last_activity'] = datetime.now(timezone.utc)

            self._logger.info(
                "Batch processed: %s messages in %.2fms (high_priority=%s, retry=%s)",
                len(messages), processing_time, is_high_priority, is_retry
            )

        except asyncio.TimeoutError:
            self._logger.error(
                "Batch processing timeout after %ss for %s messages",
                timeout, len(messages)
            )
            # Mark timed-out messages as failed
            for message in messages:
                self.queue.mark_failed(message.id, "Processing timeout")
        except Exception as e:
            self._logger.error("Batch processing error: %s", e)
            # Mark all messages as failed
            for message in messages:
                self.queue.mark_failed(message.id, f"Batch processing error: {str(e)}")

    async def _process_single_message(
        self,
        message: QueuedMessage,
        is_high_priority: bool = False,
        is_retry: bool = False
    ) -> ProcessingResult:
        """
        Process a single message with fallback support.

        Args:
            message: Message to process
            is_high_priority: Whether this is a high priority message
            is_retry: Whether this is a retry message

        Returns:
            ProcessingResult with processing outcome
        """
        start_time = time.time()

        try:
            self._logger.debug(
                "Processing message %s (type=%s, priority=%s, retry=%s)",
                message.id, message.message_type, message.priority.value, is_retry
            )

            # Create message content from queued message
            content = MessageContent(
                text=message.content.get('text', ''),
                subject=message.content.get('subject'),
                html=message.content.get('html'),
                attachments=message.content.get('attachments'),
                metadata=message.metadata
            )

            # Get recipient from metadata or use default
            recipient = message.metadata.get('recipient_id', message.recipient_id)

            # Attempt delivery with fallback
            success, delivery_results, failed_message = await self.fallback_manager.attempt_delivery_with_fallback(
                message_id=message.id,
                channels=message.channels,
                recipient=recipient,
                content=content,
                priority=message.priority.value,
                channel_instances=self._channel_instances
            )

            processing_time_ms = int((time.time() - start_time) * 1000)

            # Update message status in queue
            if success:
                self.queue.mark_delivered(message.id)
                self._logger.info(
                    "Message %s delivered successfully in %sms",
                    message.id, processing_time_ms
                )
            else:
                error_message = failed_message.failure_details if failed_message else "Delivery failed"
                self.queue.mark_failed(message.id, error_message)
                self._logger.warning(
                    "Message %s failed: %s",
                    message.id, error_message
                )

                # Update fallback stats
                if failed_message:
                    if len(failed_message.attempted_channels) > 1:
                        self._stats["fallback_attempts"] += 1

                    if failed_message.message_id in self.fallback_manager._dead_letter_queue:
                        self._stats["dead_letter_messages"] += 1

            # Convert delivery results for response
            result_dict = {}
            for result in delivery_results:
                if hasattr(result, 'metadata') and result.metadata:
                    channel_name = result.metadata.get('channel', 'unknown')
                    result_dict[channel_name] = {
                        "status": result.status.value,
                        "external_id": result.external_id,
                        "response_time_ms": result.response_time_ms,
                        "success": result.success
                    }

            return ProcessingResult(
                message_id=message.id,
                success=success,
                error_message=failed_message.failure_details if failed_message else None,
                delivery_results=result_dict,
                processing_time_ms=processing_time_ms
            )

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            error_message = f"Processing exception: {str(e)}"

            self.queue.mark_failed(message.id, error_message)
            self._logger.error(
                "Message %s processing failed: %s",
                message.id, error_message
            )

            return ProcessingResult(
                message_id=message.id,
                success=False,
                error_message=error_message,
                processing_time_ms=processing_time_ms
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics.

        Returns:
            Dictionary with processor statistics
        """
        stats = self._stats.copy()

        # Add runtime information
        if stats['start_time']:
            uptime_seconds = (datetime.now(timezone.utc) - stats['start_time']).total_seconds()
            stats['uptime_seconds'] = uptime_seconds

        # Add queue statistics
        queue_stats = self.queue.get_queue_stats()
        stats.update(queue_stats)

        # Add fallback statistics
        fallback_stats = self.fallback_manager.get_fallback_statistics()
        stats['fallback_statistics'] = fallback_stats

        # Add retry queue status
        retry_status = self.fallback_manager.get_retry_queue_status()
        stats['retry_queue'] = retry_status

        # Add worker information
        stats['worker_count'] = len(self._worker_tasks)
        stats['running'] = self._running
        stats['max_workers'] = self.max_workers
        stats['batch_size'] = self.batch_size
        stats['channel_count'] = len(self._channel_instances)

        return stats

    @property
    def is_running(self) -> bool:
        """Check if processor is running."""
        return self._running

    @property
    def uptime_seconds(self) -> float:
        """Get processor uptime in seconds."""
        if not self._stats['start_time']:
            return 0.0
        return (datetime.now(timezone.utc) - self._stats['start_time']).total_seconds()


# Global message processor instance
message_processor = MessageProcessor()


# Signal handlers for graceful shutdown
def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        _logger.info("Received signal %s, initiating shutdown...", signum)
        if message_processor.is_running:
            asyncio.create_task(message_processor.shutdown())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
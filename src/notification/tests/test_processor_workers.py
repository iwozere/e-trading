"""Tests for which worker loops MessageProcessor.start() launches."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.notification.service.processor import MessageProcessor


def _started_worker_names(message_queue_workers: bool) -> set[str]:
    async def run() -> set[str]:
        processor = MessageProcessor(queue=MagicMock(), health_mon=MagicMock())
        processor._initialize_channel_instances = AsyncMock()  # type: ignore[method-assign]
        await processor.start(message_queue_workers=message_queue_workers)
        names = {task.get_coro().__name__ for task in processor._worker_tasks}  # type: ignore[union-attr]
        processor._running = False
        for task in processor._worker_tasks:
            task.cancel()
        await asyncio.gather(*processor._worker_tasks, return_exceptions=True)
        return names

    return asyncio.run(run())


def test_start_without_message_queue_workers_skips_whole_message_dequeue():
    """The DB-centric bot delivers via per-channel claims only; no whole-message dequeue loops."""
    names = _started_worker_names(message_queue_workers=False)

    assert names.isdisjoint({"_high_priority_worker", "_normal_priority_worker", "_retry_worker"})
    assert {"_fallback_retry_worker", "_cleanup_worker"} <= names


def test_start_default_keeps_message_queue_workers():
    names = _started_worker_names(message_queue_workers=True)

    assert {"_high_priority_worker", "_normal_priority_worker", "_retry_worker"} <= names

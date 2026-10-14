"""
Keep Windows awake during long runs (plan §9.3): ``fmp run`` (and later ``fmp build``) must not be cut off by
sleep or hibernation on the laptop.

``SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`` keeps the system awake (the display may still turn
off) until the same thread resets it with ``ES_CONTINUOUS``. The state is per thread, so enter the context on the
thread that runs the event loop (``asyncio.run`` runs on the calling thread). On other platforms this is a no-op.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def keep_awake() -> Iterator[None]:
    """Prevent system sleep for the duration of the ``with`` block (Windows only; best effort)."""
    if sys.platform != "win32":
        yield
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32
    if not kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
        _logger.warning("SetThreadExecutionState failed; Windows may sleep during the run")
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)

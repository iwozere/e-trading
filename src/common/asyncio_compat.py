"""
Python 3.14 asyncio compatibility helper.

Python 3.14 removed asyncio's implicit "create an event loop if none
exists" behavior: ``asyncio.get_event_loop()`` now raises ``RuntimeError``
in a thread that never had a loop set, instead of silently creating one
(that fallback was deprecated since 3.10 and removed in 3.14).

Both ``ib_insync`` and its actively maintained successor ``ib_async`` still
assume the old behavior. ``ib_insync``'s ``eventkit`` dependency calls
``get_event_loop()`` at *import* time (``eventkit/util.py``), so merely
importing the library crashes under 3.14 unless a loop already exists for
the current thread — this bit the scheduler, the trading bot, and the web
UI simultaneously the day the Pi's system Python moved to 3.14.

Call :func:`ensure_event_loop` immediately before importing
``ib_async``/``ib_insync`` to avoid this. It is a no-op on Python versions
where a loop is auto-created (<3.14).
"""

import asyncio

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def ensure_event_loop() -> None:
    """
    Register an asyncio event loop for the current thread if none is set.

    Idempotent and cheap — safe to call from every call site that is about
    to import ``ib_async``/``ib_insync``, even repeatedly in the same
    process. A no-op once a loop already exists for the calling thread.
    """
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        _logger.debug(
            "No asyncio event loop for this thread; creating one for ib_async/ib_insync import-time compatibility."
        )
        asyncio.set_event_loop(asyncio.new_event_loop())

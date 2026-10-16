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

A main-thread loop created here (by us on 3.14, or implicitly by
``get_event_loop()`` on 3.12/3.13) is kept referenced and closed at
interpreter exit. Otherwise a later ``asyncio.run()``, which swaps in its own
loop and then unsets it, drops the loop unclosed and Python reports
``ResourceWarning: unclosed event loop``. Worker-thread loops are left to the
thread: closing one while its thread is torn down registers a stray
``_DummyThread`` on Windows (``ProactorEventLoop.close`` calls
``threading.current_thread()``), which would pile up in a long-running process.
"""

import asyncio
import atexit
import threading
import warnings
from typing import Optional

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# The main-thread loop created by ensure_event_loop(), closed at exit.
_main_loop: Optional[asyncio.AbstractEventLoop] = None


def _close_main_loop() -> None:
    """Close the main-thread loop created by ensure_event_loop() unless it is already closed or still running."""
    if _main_loop is not None and not _main_loop.is_closed() and not _main_loop.is_running():
        _main_loop.close()


def _own_if_main_thread(loop: asyncio.AbstractEventLoop) -> None:
    """
    Keep a reference to a loop created for the main thread and close it at exit.

    Args:
        loop: Loop that ensure_event_loop() created (explicitly or implicitly) for the calling thread.
    """
    global _main_loop
    if threading.current_thread() is not threading.main_thread():
        return
    if _main_loop is None:
        atexit.register(_close_main_loop)
    _main_loop = loop


def ensure_event_loop() -> None:
    """
    Register an asyncio event loop for the current thread if none is set.

    Idempotent and cheap — safe to call from every call site that is about
    to import ``ib_async``/``ib_insync``, even repeatedly in the same
    process. A no-op once a loop already exists for the calling thread.
    """
    # On 3.12/3.13 get_event_loop() in the main thread creates a loop itself and says so with a DeprecationWarning;
    # record it so that loop is owned (and closed) like one we create, and so the warning is not shown.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            _logger.debug(
                "No asyncio event loop for this thread; creating one for ib_async/ib_insync import-time compatibility."
            )
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _own_if_main_thread(loop)
            return
    if any(issubclass(w.category, DeprecationWarning) and "no current event loop" in str(w.message) for w in caught):
        _own_if_main_thread(loop)

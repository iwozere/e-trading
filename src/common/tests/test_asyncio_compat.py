"""Unit tests for src.common.asyncio_compat."""

import asyncio
import threading
import unittest

from src.common.asyncio_compat import ensure_event_loop


class TestEnsureEventLoop(unittest.TestCase):
    """Tests for ensure_event_loop()."""

    def test_creates_loop_when_none_set(self):
        """A thread with no event loop registered gets one created for it."""
        result = {}

        def worker():
            try:
                asyncio.get_event_loop()
                result["had_loop_already"] = True
            except RuntimeError:
                result["had_loop_already"] = False

            ensure_event_loop()
            try:
                loop = asyncio.get_event_loop()
                result["loop"] = loop
            except RuntimeError as e:
                result["error"] = e

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        self.assertNotIn("error", result, "ensure_event_loop() should leave a usable event loop in place")
        self.assertIsNotNone(result.get("loop"))

    def test_idempotent_when_loop_already_exists(self):
        """Calling it twice in a thread that already has a loop is a harmless no-op."""

        def worker():
            ensure_event_loop()
            first_loop = asyncio.get_event_loop()
            ensure_event_loop()
            second_loop = asyncio.get_event_loop()
            return first_loop is second_loop

        result = {}

        def target():
            result["unchanged"] = worker()

        thread = threading.Thread(target=target)
        thread.start()
        thread.join()

        self.assertTrue(result["unchanged"], "ensure_event_loop() must not replace an already-set loop")


if __name__ == "__main__":
    unittest.main()

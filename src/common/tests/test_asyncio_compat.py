"""Unit tests for src.common.asyncio_compat."""

import asyncio
import subprocess
import sys
import threading
import unittest
from pathlib import Path

from src.common.asyncio_compat import ensure_event_loop

PROJECT_ROOT = Path(__file__).resolve().parents[3]


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

    def test_loop_dropped_by_asyncio_run_is_not_leaked(self):
        """A fresh interpreter that calls ensure_event_loop() and then asyncio.run() reports no unclosed loop."""
        code = (
            "import asyncio, gc\n"
            "from src.common.asyncio_compat import ensure_event_loop\n"
            "ensure_event_loop()\n"
            "async def noop():\n"
            "    return None\n"
            "asyncio.run(noop())\n"
            "gc.collect()\n"
        )
        result = subprocess.run(
            [sys.executable, "-X", "dev", "-W", "always::ResourceWarning", "-c", code],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("unclosed event loop", result.stderr)
        self.assertNotIn("There is no current event loop", result.stderr)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for src.ml.pipeline.p21_momentum.data.exclusions."""

from __future__ import annotations

import json
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from src.ml.pipeline.p21_momentum.data.exclusions import load_exclusions


class TestLoadExclusions(unittest.TestCase):
    def _write(self, tmpdir: str, payload: dict) -> Path:
        path = Path(tmpdir) / "p21_exclusions.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_missing_file_returns_empty_set(self):
        result = load_exclusions(path=Path("does/not/exist.json"))
        self.assertEqual(result, set())

    def test_active_entry_included(self):
        with TemporaryDirectory() as tmp:
            path = self._write(
                tmp,
                {"exclusions": [{"ticker": "XYZ", "reason": "M&A", "added": "2026-07-15", "expires": "2026-12-31"}]},
            )
            result = load_exclusions(path=path, as_of=date(2026, 8, 22))
        self.assertEqual(result, {"XYZ"})

    def test_expired_entry_excluded(self):
        with TemporaryDirectory() as tmp:
            path = self._write(
                tmp,
                {"exclusions": [{"ticker": "XYZ", "reason": "M&A", "added": "2026-01-01", "expires": "2026-06-30"}]},
            )
            result = load_exclusions(path=path, as_of=date(2026, 8, 22))
        self.assertEqual(result, set())

    def test_entry_without_expires_never_expires(self):
        with TemporaryDirectory() as tmp:
            path = self._write(tmp, {"exclusions": [{"ticker": "ABC", "reason": "delisted"}]})
            result = load_exclusions(path=path, as_of=date(2099, 1, 1))
        self.assertEqual(result, {"ABC"})

    def test_empty_exclusions_list(self):
        with TemporaryDirectory() as tmp:
            path = self._write(tmp, {"exclusions": []})
            result = load_exclusions(path=path)
        self.assertEqual(result, set())


if __name__ == "__main__":
    unittest.main()

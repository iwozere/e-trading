"""
Execution Persistence Service
----------------------------
Handles thread-safe persistence of trade and order execution data to disk.
Provides a centralized way to log trading activity across all bot instances.
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict
from datetime import datetime, timezone

from src.trading.constants import TRADING_LOGS_DIR
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class ExecutionPersistenceError(Exception):
    """Exception raised for errors in execution persistence."""
    pass

class ExecutionPersistenceService:
    """
    Service for persisting trading execution data (orders and trades).
    Ensures thread-safe access to log files and consistent data formatting.
    """
    
    _instance = None
    _lock = threading.Lock()
    _file_locks: Dict[str, threading.RLock] = {}

    def __new__(cls):
        """Singleton pattern implementation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ExecutionPersistenceService, cls).__new__(cls)
            return cls._instance

    def __init__(self):
        """Initialize the service and ensure directories exist."""
        # Only initialize once (singleton)
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.logs_dir = TRADING_LOGS_DIR
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        _logger.info("ExecutionPersistenceService initialized at %s", self.logs_dir)

    def _get_file_lock(self, filename: str) -> threading.RLock:
        """Get or create a reentrant lock for a specific file."""
        with self._lock:
            if filename not in self._file_locks:
                self._file_locks[filename] = threading.RLock()
            return self._file_locks[filename]

    def _append_record(self, file_path: Path, data: Any) -> None:
        """
        Append one record to a JSONL (newline-delimited JSON) file.

        Each call is O(1) — the file is opened in append mode and a single line
        is written.  On a corrupt file the bad file is quarantined (renamed) so
        history is never silently discarded; a fresh file is then started.
        """
        lock = self._get_file_lock(file_path.name)
        with lock:
            if file_path.exists() and file_path.stat().st_size > 0:
                self._quarantine_if_corrupt(file_path)
            try:
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data, default=str))
                    f.write("\n")
            except Exception as e:
                _logger.exception("Error appending record to %s:", file_path)
                raise ExecutionPersistenceError(f"Failed to persist data to {file_path.name}") from e

    def _quarantine_if_corrupt(self, file_path: Path) -> None:
        """Rename a JSONL file that contains an invalid last line to isolate it."""
        try:
            with open(file_path, "rb") as f:
                # Seek to the last non-empty line without reading the whole file.
                f.seek(0, 2)
                end = f.tell()
                if end == 0:
                    return
                # Walk backwards past trailing newlines.
                pos = end - 1
                while pos > 0:
                    f.seek(pos)
                    ch = f.read(1)
                    if ch not in (b"\n", b"\r"):
                        break
                    pos -= 1
                # Find start of that last line.
                while pos > 0:
                    f.seek(pos - 1)
                    if f.read(1) == b"\n":
                        break
                    pos -= 1
                f.seek(pos)
                last_line = f.read().rstrip()
            json.loads(last_line)
        except (json.JSONDecodeError, ValueError):
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            quarantine = file_path.with_name(f"{file_path.stem}.corrupt.{ts}{file_path.suffix}")
            try:
                file_path.rename(quarantine)
                _logger.error(
                    "Corrupt JSONL file quarantined: %s -> %s. A fresh log will be started.",
                    file_path.name, quarantine.name,
                )
            except OSError:
                _logger.exception("Could not quarantine corrupt log file %s:", file_path)
        except OSError:
            _logger.exception("Could not verify integrity of log file %s:", file_path)

    def save_order(self, bot_id: str, order_data: Dict[str, Any]) -> None:
        """
        Persist order details to the orders log (JSONL).

        Args:
            bot_id: Unique bot identifier
            order_data: Dictionary containing order details
        """
        if "bot_id" not in order_data:
            order_data["bot_id"] = bot_id
        if "persisted_at" not in order_data:
            order_data["persisted_at"] = datetime.now(timezone.utc).isoformat()

        path = self.logs_dir / "orders.jsonl"
        self._append_record(path, order_data)
        _logger.debug("Order saved for bot %s", bot_id)

    def save_trade(self, bot_id: str, trade_data: Dict[str, Any]) -> None:
        """
        Persist completed trade details to the trades log (JSONL).

        Args:
            bot_id: Unique bot identifier
            trade_data: Dictionary containing trade details
        """
        if "bot_id" not in trade_data:
            trade_data["bot_id"] = bot_id
        if "persisted_at" not in trade_data:
            trade_data["persisted_at"] = datetime.now(timezone.utc).isoformat()

        path = self.logs_dir / "trades.jsonl"
        self._append_record(path, trade_data)
        _logger.debug("Trade saved for bot %s", bot_id)

# Singleton instance for easy access
execution_persistence = ExecutionPersistenceService()

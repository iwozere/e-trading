"""
Execution Persistence Service
----------------------------
Handles thread-safe persistence of trade and order execution data to disk.
Provides a centralized way to log trading activity across all bot instances.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

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

    def _append_to_json_list(self, file_path: Path, data: Any) -> None:
        """
        Append an item to a JSON list file in a thread-safe manner.
        Creates the file if it doesn't exist.
        """
        lock = self._get_file_lock(file_path.name)
        with lock:
            try:
                all_items = []
                if file_path.exists():
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            if content:
                                all_items = json.loads(content)
                    except (json.JSONDecodeError, Exception) as e:
                        _logger.warning("Failed to load existing log file %s: %s. Starting fresh.", file_path, e)
                        all_items = []

                all_items.append(data)

                # Write to a sibling .tmp file first, then atomically rename over the
                # target.  This prevents a crash mid-write from leaving a partially
                # written (and therefore corrupt) JSON file.  On POSIX the rename is
                # guaranteed atomic; on Windows it is near-atomic (same volume, single
                # FS operation).
                tmp_path = file_path.with_suffix('.tmp')
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(all_items, f, default=str, indent=2)
                tmp_path.replace(file_path)
                    
            except Exception as e:
                _logger.exception("Error appending to JSON list %s: %s", file_path, e)
                raise ExecutionPersistenceError(f"Failed to persist data to {file_path.name}") from e

    def save_order(self, bot_id: str, order_data: Dict[str, Any]) -> None:
        """
        Persist order details to the orders log.
        
        Args:
            bot_id: Unique bot identifier
            order_data: Dictionary containing order details
        """
        # Ensure bot_id is in the data
        if "bot_id" not in order_data:
            order_data["bot_id"] = bot_id
            
        # Add persistence timestamp if not present
        if "persisted_at" not in order_data:
            order_data["persisted_at"] = datetime.now().isoformat()
            
        path = self.logs_dir / "orders.json"
        self._append_to_json_list(path, order_data)
        _logger.debug("Order saved for bot %s", bot_id)

    def save_trade(self, bot_id: str, trade_data: Dict[str, Any]) -> None:
        """
        Persist completed trade details to the trades log.
        
        Args:
            bot_id: Unique bot identifier
            trade_data: Dictionary containing trade details
        """
        # Ensure bot_id is in the data
        if "bot_id" not in trade_data:
            trade_data["bot_id"] = bot_id
            
        # Add persistence timestamp if not present
        if "persisted_at" not in trade_data:
            trade_data["persisted_at"] = datetime.now().isoformat()
            
        path = self.logs_dir / "trades.json"
        self._append_to_json_list(path, trade_data)
        _logger.debug("Trade saved for bot %s", bot_id)

# Singleton instance for easy access
execution_persistence = ExecutionPersistenceService()

"""
Metrics Tracker Service
-----------------------
Provides a centralized way to track and persist performance metrics for all trading bots.
Tracks metrics like PnL, Win Rate, and Max Drawdown.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime

from src.trading.constants import DATA_DIR
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

@dataclass
class PerformanceMetrics:
    """Dataclass for storing performance metrics for a bot."""
    bot_id: str
    symbol: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    peak_balance: float = 0.0
    current_balance: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def update(self, trade_pnl: float, trade_pnl_pct: float, current_balance: float):
        """Update metrics with a new completed trade."""
        self.total_trades += 1
        if trade_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
            
        self.win_rate = (self.winning_trades / self.total_trades) * 100
        self.total_pnl += trade_pnl
        self.total_pnl_pct += trade_pnl_pct
        self.current_balance = current_balance
        
        # Update drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
            self.current_drawdown = 0.0
        else:
            dd = (self.peak_balance - self.current_balance) / self.peak_balance * 100
            self.current_drawdown = dd
            if dd > self.max_drawdown:
                self.max_drawdown = dd
                
        self.last_updated = datetime.now().isoformat()

class MetricsRegistry:
    """
    Registry for managing performance metrics across all active bot instances.
    Handles persistence of metrics to disk.
    """
    
    _instance = None
    _lock = threading.Lock()
    _registry_lock = threading.RLock()

    def __new__(cls):
        """Singleton pattern implementation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MetricsRegistry, cls).__new__(cls)
            return cls._instance

    def __init__(self):
        """Initialize the registry and load existing metrics."""
        # Only initialize once (singleton)
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.metrics_file = DATA_DIR / "metrics.json"
        self.bot_metrics: Dict[str, PerformanceMetrics] = {}
        self._load_metrics()
        self._initialized = True
        _logger.info("MetricsRegistry initialized, tracking %d bots", len(self.bot_metrics))

    def _load_metrics(self) -> None:
        """Load metrics from the JSON file."""
        if not self.metrics_file.exists():
            return
            
        try:
            with open(self.metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for bot_id, metrics_dict in data.items():
                    self.bot_metrics[bot_id] = PerformanceMetrics(**metrics_dict)
        except Exception as e:
            _logger.error("Failed to load metrics from %s: %s", self.metrics_file, e)

    def _save_metrics(self) -> None:
        """Save metrics to the JSON file."""
        try:
            with open(self.metrics_file, "w", encoding="utf-8") as f:
                data = {bot_id: asdict(m) for bot_id, m in self.bot_metrics.items()}
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            _logger.error("Failed to save metrics to %s: %s", self.metrics_file, e)

    def get_metrics(self, bot_id: str, symbol: str, initial_balance: float) -> PerformanceMetrics:
        """Get or create metrics for a specific bot."""
        with self._registry_lock:
            if bot_id not in self.bot_metrics:
                self.bot_metrics[bot_id] = PerformanceMetrics(
                    bot_id=bot_id, 
                    symbol=symbol,
                    current_balance=initial_balance,
                    peak_balance=initial_balance
                )
            return self.bot_metrics[bot_id]

    def record_trade(self, bot_id: str, symbol: str, trade_pnl: float, trade_pnl_pct: float, current_balance: float) -> None:
        """Record a completed trade and update bot metrics."""
        with self._registry_lock:
            metrics = self.get_metrics(bot_id, symbol, current_balance)
            metrics.update(trade_pnl, trade_pnl_pct, current_balance)
            self._save_metrics()
            _logger.info("Metrics updated for bot %s: PnL=%.2f, WinRate=%.1f%%", 
                        bot_id, metrics.total_pnl, metrics.win_rate)

    def get_global_summary(self) -> Dict[str, Any]:
        """Get an aggregated summary of all tracked bots."""
        with self._registry_lock:
            total_pnl = sum(m.total_pnl for m in self.bot_metrics.values())
            total_trades = sum(m.total_trades for m in self.bot_metrics.values())
            
            return {
                "active_bots": len(self.bot_metrics),
                "total_pnl": total_pnl,
                "total_trades": total_trades,
                "bot_breakdown": {bot_id: asdict(m) for bot_id, m in self.bot_metrics.items()}
            }

# Singleton instance for easy access
metrics_registry = MetricsRegistry()

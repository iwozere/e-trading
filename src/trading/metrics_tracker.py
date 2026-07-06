"""
Metrics Tracker Service
-----------------------
Provides a centralized way to track and persist performance metrics for all trading bots.
Tracks metrics like PnL, Win Rate, and Max Drawdown.
"""

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict

from src.notification.logger import setup_logger
from src.trading.constants import DATA_DIR

_logger = setup_logger(__name__)

# Minimum denominator for drawdown % vs peak (avoids div-by-zero / unstable FP when peak is ~0)
_PEAK_DRAWDOWN_EPS = 1e-12


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
    last_updated: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def update(self, trade_pnl: float, trade_pnl_pct: float, current_balance: float):
        """Update metrics with a new completed trade."""
        self.total_trades += 1
        if trade_pnl > 0:
            self.winning_trades += 1
        elif trade_pnl < 0:
            self.losing_trades += 1
        # trade_pnl == 0 (break-even) is counted in total_trades but not
        # in winning_trades or losing_trades — it is a neutral outcome.

        self.win_rate = (self.winning_trades / self.total_trades) * 100
        self.total_pnl += trade_pnl
        self.total_pnl_pct += trade_pnl_pct
        self.current_balance = current_balance

        # Update drawdown (guard peak_balance ~= 0)
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
            self.current_drawdown = 0.0
        else:
            peak = self.peak_balance
            if peak <= _PEAK_DRAWDOWN_EPS:
                self.current_drawdown = 0.0
            else:
                dd = (peak - self.current_balance) / peak * 100.0
                self.current_drawdown = dd
                if dd > self.max_drawdown:
                    self.max_drawdown = dd

        self.last_updated = datetime.now(UTC).isoformat()


class MetricsRegistry:
    """
    Registry for managing performance metrics across all active bot instances.
    Handles persistence of metrics to disk.

    Flush strategy: writes are *deferred* — ``record_trade()`` only marks the
    registry as dirty.  A background daemon thread wakes every
    ``FLUSH_INTERVAL_SECONDS`` (default 30 s) and flushes when dirty.  This
    amortises the disk I/O cost across many consecutive trades instead of
    issuing a full file rewrite on every single trade.
    ``flush()`` can be called explicitly (e.g. on shutdown) to guarantee
    in-flight updates are persisted immediately.
    """

    FLUSH_INTERVAL_SECONDS: int = 30

    _instance = None
    _lock = threading.Lock()
    _registry_lock = threading.RLock()

    def __new__(cls):
        """Singleton pattern implementation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        """Initialize the registry and load existing metrics."""
        # Only initialize once (singleton)
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.metrics_file = DATA_DIR / "metrics.json"
        self.bot_metrics: Dict[str, PerformanceMetrics] = {}
        self._dirty = False
        self._load_metrics()
        self._initialized = True

        # Start background flush thread (daemon so it doesn't block process exit)
        self._flush_thread = threading.Thread(
            target=self._background_flush_loop,
            name="metrics-flush",
            daemon=True,
        )
        self._flush_thread.start()
        _logger.info("MetricsRegistry initialized, tracking %d bots", len(self.bot_metrics))

    def _background_flush_loop(self) -> None:
        """Periodically flush dirty metrics to disk every FLUSH_INTERVAL_SECONDS."""
        while True:
            time.sleep(self.FLUSH_INTERVAL_SECONDS)
            self.flush()

    def _load_metrics(self) -> None:
        """Load metrics from the JSON file."""
        if not self.metrics_file.exists():
            return

        try:
            with open(self.metrics_file, encoding="utf-8") as f:
                data = json.load(f)
                for bot_id, metrics_dict in data.items():
                    self.bot_metrics[bot_id] = PerformanceMetrics(**metrics_dict)
        except Exception as e:
            _logger.error("Failed to load metrics from %s: %s", self.metrics_file, e)

    def flush(self) -> None:
        """
        Flush dirty metrics to disk if any updates are pending.

        Safe to call from any thread at any time (e.g. on shutdown).  Is a
        no-op when the registry has not been updated since the last flush.
        """
        with self._registry_lock:
            if not self._dirty:
                return
            try:
                tmp_path = self.metrics_file.with_suffix(".tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    data = {bot_id: asdict(m) for bot_id, m in self.bot_metrics.items()}
                    json.dump(data, f, indent=2, default=str)
                tmp_path.replace(self.metrics_file)
                self._dirty = False
            except Exception as e:
                _logger.error("Failed to save metrics to %s: %s", self.metrics_file, e)

    # Keep _save_metrics as a private alias so subclasses / tests that call it
    # directly still work.
    def _save_metrics(self) -> None:
        """Mark registry as dirty; the background thread will persist it."""
        self._dirty = True

    def get_metrics(self, bot_id: str, symbol: str, initial_balance: float) -> PerformanceMetrics:
        """Get or create metrics for a specific bot."""
        with self._registry_lock:
            if bot_id not in self.bot_metrics:
                self.bot_metrics[bot_id] = PerformanceMetrics(
                    bot_id=bot_id, symbol=symbol, current_balance=initial_balance, peak_balance=initial_balance
                )
            return self.bot_metrics[bot_id]

    def record_trade(
        self,
        bot_id: str,
        symbol: str,
        trade_pnl: float,
        trade_pnl_pct: float,
        current_balance: float,
    ) -> None:
        """Record a completed trade and mark metrics as dirty for the next flush."""
        with self._registry_lock:
            metrics = self.get_metrics(bot_id, symbol, current_balance)
            metrics.update(trade_pnl, trade_pnl_pct, current_balance)
            self._dirty = True
            _logger.info(
                "Metrics updated for bot %s: PnL=%.2f, WinRate=%.1f%%",
                bot_id,
                metrics.total_pnl,
                metrics.win_rate,
            )

    def get_global_summary(self) -> Dict[str, Any]:
        """Get an aggregated summary of all tracked bots."""
        with self._registry_lock:
            total_pnl = sum(m.total_pnl for m in self.bot_metrics.values())
            total_trades = sum(m.total_trades for m in self.bot_metrics.values())

            return {
                "active_bots": len(self.bot_metrics),
                "total_pnl": total_pnl,
                "total_trades": total_trades,
                "bot_breakdown": {bot_id: asdict(m) for bot_id, m in self.bot_metrics.items()},
            }


# Singleton instance for easy access
metrics_registry = MetricsRegistry()

import asyncio
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Type, Optional

from src.data.downloader.ibkr_downloader import IBKRDownloader
from src.screeners.logic.strategy_bridge import ScreenerStrategyBridge
from src.screeners.discovery.base import IDiscoveryProvider
from src.screeners.logic.notifier import SignalNotifier
from src.strategy.base_strategy import BaseStrategy
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class IBKRScreenerService:
    """
    Main orchestrator for the IBKR Scalable Screener.
    Manages discovery, data gaps, strategy execution, and results.
    """

    def __init__(self,
                 strategy_class: Type[BaseStrategy],
                 strategy_config: Dict[str, Any],
                 discovery_providers: List[IDiscoveryProvider],
                 notifier: Optional[SignalNotifier] = None,
                 interval: str = '1h',
                 concurrency: int = 50,
                 downloader: Optional[IBKRDownloader] = None):
        """
        Initialize the service.

        Args:
            strategy_class: Backtrader strategy class to run on each symbol.
            strategy_config: Configuration dict passed to the strategy.
            discovery_providers: List of symbol-discovery providers.
            notifier: Optional notifier to broadcast signals.
            interval: OHLCV bar interval (e.g. '1h').
            concurrency: Maximum number of symbols processed in parallel.
            downloader: IBKRDownloader instance.  Defaults to ``IBKRDownloader()``
                when *None*.  Pass an explicit instance to inject test doubles or
                pre-configured downloaders.
        """
        self.strategy_class = strategy_class
        self.strategy_config = strategy_config
        self.discovery_providers = discovery_providers
        self.interval = interval
        self.concurrency_limit = asyncio.Semaphore(concurrency)
        self.notifier = notifier

        # Components
        self.downloader = downloader if downloader is not None else IBKRDownloader()
        self.bridge = ScreenerStrategyBridge(strategy_class, strategy_config)

        # In-memory signal de-duplication — maps symbol → fingerprint of last
        # broadcast signal.  Identical signals are suppressed until the
        # fingerprint changes (i.e. price, regime, or indicators change).
        self._seen_signals: Dict[str, str] = {}

        # Thread pool for CPU-bound strategy work — keeps the event loop free.
        # NOTE: ProcessPoolExecutor would give true parallelism but requires all
        # strategy classes to be picklable.  ThreadPoolExecutor is the safe default.
        self._executor = ThreadPoolExecutor(max_workers=min(32, concurrency))

        # Settings — derive from __file__ so the path is always relative to the project root,
        # not the process working directory.
        self.results_dir = Path(__file__).resolve().parents[2] / "results" / "screeners" / "ibkr"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def run_once(self):
        """Executes one full scan of all discovered symbols."""
        _logger.info("Starting IBKR Screener scan loop...")

        # 1. Discover Symbols
        all_symbols = set()
        for provider in self.discovery_providers:
            symbols = await provider.get_symbols()
            all_symbols.update(symbols)

        _logger.info("Total unique symbols discovered: %d", len(all_symbols))

        # 2. Process symbols concurrently
        tasks = [self._process_symbol(symbol) for symbol in all_symbols]
        results = await asyncio.gather(*tasks)

        # 3. Filter and Store Results
        valid_results = [r for r in results if r and "error" not in r]
        self._persist_results(valid_results)

        # 4. De-duplicate signals — only broadcast signals whose content changed
        #    since the last scan.  This prevents identical alerts from flooding
        #    notification channels on every scan interval.
        new_signals = []
        for result in valid_results:
            symbol = result.get('symbol', '')
            fp = self._signal_fingerprint(result)
            if self._seen_signals.get(symbol) != fp:
                self._seen_signals[symbol] = fp
                new_signals.append(result)
            else:
                _logger.debug("Skipping duplicate signal for %s (unchanged)", symbol)

        if self.notifier and new_signals:
            await self.notifier.notify_signals(new_signals)

        _logger.info(
            "Scan loop complete. %d signals generated, %d new (de-duplicated).",
            len(valid_results), len(new_signals)
        )
        return valid_results

    def _signal_fingerprint(self, result: Dict[str, Any]) -> str:
        """
        Compute a stable fingerprint for a signal result dict.

        Volatile, time-varying keys (``timestamp``, ``scan_time``) are excluded
        so that only meaningful data changes trigger a new notification.

        Args:
            result: Signal result dict from :meth:`_harvest_results`.

        Returns:
            Hex MD5 digest of the sorted, serialised result content.
        """
        _VOLATILE_KEYS = frozenset(('timestamp', 'scan_time', 'date'))
        filtered = {k: v for k, v in result.items() if k not in _VOLATILE_KEYS}
        payload = json.dumps(filtered, sort_keys=True, default=str).encode()
        return hashlib.md5(payload).hexdigest()

    async def _process_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Handles the flow for a single symbol with concurrency limiting."""
        async with self.concurrency_limit:
            try:
                _logger.debug("Processing symbol: %s", symbol)

                # A. Download Data
                # We need enough data for the strategy warmup
                warmup = self.strategy_config.get('warmup_period', 100)
                start_date = datetime.now(timezone.utc) - timedelta(days=warmup * 2)

                # Downloader handles caching internally.
                # get_ohlcv is synchronous/I-O-bound — run it in the thread pool so we
                # do not block the event loop while waiting for disk or network I/O.
                loop = asyncio.get_running_loop()
                df = await loop.run_in_executor(
                    self._executor,
                    lambda: self.downloader.get_ohlcv(
                        symbol=symbol,
                        interval=self.interval,
                        start_date=start_date,
                        end_date=datetime.now(timezone.utc)
                    )
                )

                if df.empty or len(df) < 10:
                    _logger.warning("Insufficient data for %s", symbol)
                    return {"symbol": symbol, "error": "Insufficient data"}

                # B. Run Strategy Bridge — CPU-bound Backtrader work in thread pool so
                # the event loop stays responsive for notifications, health checks, etc.
                result = await loop.run_in_executor(self._executor, self.bridge.run, symbol, df)
                return result

            except Exception as e:
                _logger.error("Error processing %s: %s", symbol, e)
                return {"symbol": symbol, "error": str(e)}

    def _persist_results(self, results: List[Dict[str, Any]]):
        """Saves signals to <project_root>/results/screeners/ibkr/signals_{timestamp}.json"""
        if not results:
            return

        now_utc = datetime.now(timezone.utc)
        timestamp = now_utc.strftime("%Y%m%d_%H%M%S")
        file_path = self.results_dir / f"signals_{timestamp}.json"

        try:
            with open(file_path, 'w') as f:
                json.dump({
                    "scan_time": now_utc.isoformat(),
                    "total_symbols": len(results),
                    "signals": results
                }, f, indent=2)
            _logger.info("Saved scan results to %s", file_path)
        except Exception as e:
            _logger.error("Failed to save results: %s", e)

    def shutdown(self) -> None:
        """
        Shutdown the service and release all resources.

        Waits for any in-flight thread-pool tasks to complete before returning.
        Always call this in the ``finally`` block of the entry-point to avoid
        thread leaks on ``KeyboardInterrupt`` or unhandled exceptions.
        """
        self._executor.shutdown(wait=True)
        _logger.info("IBKRScreenerService executor shut down")

    async def start_scheduled_loop(self, run_every_minutes: int = 15):
        """Runs the screener indefinitely on a fixed interval with exponential backoff on retry."""
        _logger.info("Starting scheduled screener loop every %d minutes", run_every_minutes)
        backoff = 1

        while True:
            try:
                await self.run_once()
                backoff = 1 # Reset on success
            except Exception as e:
                _logger.error("Error in screener loop: %s. Retrying in %ds", e, backoff * 60)
                await asyncio.sleep(backoff * 60)
                backoff = min(backoff * 2, 30) # Max 30 min backoff
                continue

            _logger.info("Waiting %d minutes for next scan...", run_every_minutes)
            await asyncio.sleep(run_every_minutes * 60)

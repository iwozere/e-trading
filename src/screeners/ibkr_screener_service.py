import asyncio
import json
import os
from datetime import datetime, timedelta
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
                 concurrency: int = 50):
        """
        Initialize the service.
        """
        self.strategy_class = strategy_class
        self.strategy_config = strategy_config
        self.discovery_providers = discovery_providers
        self.interval = interval
        self.concurrency_limit = asyncio.Semaphore(concurrency)
        self.notifier = notifier

        # Components
        self.downloader = IBKRDownloader()
        self.bridge = ScreenerStrategyBridge(strategy_class, strategy_config)

        # Settings
        self.results_dir = os.path.join('results', 'screeners', 'ibkr')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

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

        # 4. Broadcast Signals
        if self.notifier and valid_results:
            await self.notifier.notify_signals(valid_results)

        _logger.info("Scan loop complete. %d signals generated.", len(valid_results))
        return valid_results

    async def _process_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Handles the flow for a single symbol with concurrency limiting."""
        async with self.concurrency_limit:
            try:
                _logger.debug("Processing symbol: %s", symbol)

                # A. Download Data
                # We need enough data for the strategy warmup
                warmup = self.strategy_config.get('warmup_period', 100)
                start_date = datetime.now() - timedelta(days=warmup * 2) # Heuristic for intraday

                # Downloader handles caching internally
                df = self.downloader.get_ohlcv(
                    symbol=symbol,
                    interval=self.interval,
                    start_date=start_date,
                    end_date=datetime.now()
                )

                if df.empty or len(df) < 10:
                    _logger.warning("Insufficient data for %s", symbol)
                    return {"symbol": symbol, "error": "Insufficient data"}

                # B. Run Strategy Bridge
                result = self.bridge.run(symbol, df)
                return result

            except Exception as e:
                _logger.error("Error processing %s: %s", symbol, e)
                return {"symbol": symbol, "error": str(e)}

    def _persist_results(self, results: List[Dict[str, Any]]):
        """Saves signals to results/screeners/ibkr/signals_{timestamp}.json"""
        if not results:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.results_dir, f"signals_{timestamp}.json")

        try:
            with open(file_path, 'w') as f:
                json.dump({
                    "scan_time": datetime.now().isoformat(),
                    "total_symbols": len(results),
                    "signals": results
                }, f, indent=2)
            _logger.info("Saved scan results to %s", file_path)
        except Exception as e:
            _logger.error("Failed to save results: %s", e)

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

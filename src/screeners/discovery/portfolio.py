import asyncio
from typing import List
from src.screeners.discovery.base import IDiscoveryProvider
from src.trading.broker.ibkr_broker import IBKRBroker
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class PortfolioDiscovery(IDiscoveryProvider):
    """
    Discovers symbols currently held in the IBKR portfolio.
    """

    def __init__(self, broker: IBKRBroker):
        self.broker = broker

    async def get_symbols(self) -> List[str]:
        """Fetches symbols from the broker's active positions."""
        try:
            if not self.broker.is_connected:
                _logger.warning("Broker not connected. Attempting to connect...")
                connected = await self.broker.connect()
                if not connected:
                    _logger.error("Could not connect to broker for portfolio discovery.")
                    return []

            positions = await self.broker.get_positions()
            symbols = list(positions.keys())
            _logger.info("Discovered %d symbols from portfolio.", len(symbols))
            return symbols
        except Exception as e:
            _logger.exception("Error in PortfolioDiscovery: %s", e)
            return []

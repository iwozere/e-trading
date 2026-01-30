import json
import os
from typing import List
from src.screeners.discovery.base import IDiscoveryProvider
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class StaticDiscovery(IDiscoveryProvider):
    """
    Loads a static list of symbols from a JSON file.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    async def get_symbols(self) -> List[str]:
        """Loads symbols from the configured JSON file."""
        if not os.path.exists(self.file_path):
            _logger.warning("Watchlist file %s not found. Returning empty list.", self.file_path)
            return []

        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                symbols = data.get('symbols', [])
                _logger.info("Loaded %d symbols from static watchlist.", len(symbols))
                return symbols
        except Exception as e:
            _logger.exception("Error reading watchlist file %s: %s", self.file_path, e)
            return []

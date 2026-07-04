import json
from pathlib import Path
from typing import List, Union

from src.notification.logger import setup_logger
from src.screeners.discovery.base import IDiscoveryProvider

_logger = setup_logger(__name__)


class StaticDiscovery(IDiscoveryProvider):
    """
    Loads a static list of symbols from a JSON file.
    """

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)

    async def get_symbols(self) -> List[str]:
        """Loads symbols from the configured JSON file."""
        if not self.file_path.exists():
            _logger.warning("Watchlist file %s not found. Returning empty list.", self.file_path)
            return []

        try:
            data = json.loads(self.file_path.read_text(encoding="utf-8"))
            symbols = data.get("symbols", [])
            _logger.info("Loaded %d symbols from static watchlist.", len(symbols))
            return symbols
        except Exception as e:
            _logger.exception("Error reading watchlist file %s: %s", self.file_path, e)
            return []

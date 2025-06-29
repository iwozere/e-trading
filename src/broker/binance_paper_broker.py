import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from src.broker.base_binance_broker import BaseBinanceBroker
from src.notification.logger import setup_logger


_logger = setup_logger(__name__)

"""
Paper trading broker implementation for Binance, simulating trades without executing real orders on the exchange.
"""

class BinancePaperBroker(BaseBinanceBroker):
    """
    Binance paper trading broker (testnet).
    """

    def __init__(self, api_key: str, api_secret: str, cash: float = 10000.0) -> None:
        super().__init__(api_key, api_secret, cash, testnet=True)
        self.broker_name = "Binance Paper"
        self.client.API_URL = "https://testnet.binance.vision/api"  # Testnet URL
        self._cash = cash
        self._value = cash
        self.orders = []
        self.positions = {}
        self.notifs = []
        _logger.info("BinancePaperBroker initialized with testnet.")

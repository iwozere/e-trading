import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Any

import backtrader as bt
from binance.client import Client
from binance.enums import *
from src.broker.base_binance_broker import BaseBinanceBroker
from src.notification.emailer import send_email_alert
from src.notification.logger import _logger
from src.notification.telegram_notifier import send_telegram_alert

from config.donotshare.donotshare import (BINANCE_PAPER_KEY,
                                          BINANCE_PAPER_SECRET)


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

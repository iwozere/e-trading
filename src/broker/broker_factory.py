from typing import Any, Dict

from src.broker.binance_broker import BinanceBroker
from src.broker.binance_paper_broker import BinancePaperBroker
from src.broker.ibkr_broker import IBKRBroker
from src.broker.mock_broker import MockBroker

from config.donotshare.donotshare import (BINANCE_KEY, BINANCE_PAPER_KEY,
                                          BINANCE_PAPER_SECRET, BINANCE_SECRET,
                                          IBKR_CLIENT_ID, IBKR_HOST, IBKR_PORT)


def get_broker(config: Dict[str, Any]):
    """
    Factory function to instantiate the correct broker based on config['type'].
    Supported types: 'binance', 'binance_paper', 'ibkr', 'mock'.
    """
    broker_type = config.get("type", "mock").lower()
    if broker_type == "binance":
        return BinanceBroker(BINANCE_KEY, BINANCE_SECRET, config.get("cash", 1000.0))
    elif broker_type == "binance_paper":
        return BinancePaperBroker(
            BINANCE_PAPER_KEY, BINANCE_PAPER_SECRET, config.get("cash", 1000.0)
        )
    elif broker_type == "ibkr":
        return IBKRBroker(
            IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, config.get("cash", 1000.0)
        )
    elif broker_type == "mock":
        return MockBroker(config.get("cash", 1000.0))
    else:
        raise ValueError(f"Unsupported broker type: {broker_type}")

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.trading.broker.ibkr_broker import IBKRBroker
from src.screeners.ibkr_screener_service import IBKRScreenerService
from src.screeners.discovery.portfolio import PortfolioDiscovery
from src.screeners.discovery.static import StaticDiscovery
from src.screeners.logic.notifier import SignalNotifier
from src.notification.service.client import NotificationServiceClient
from src.strategy.custom_strategy import CustomStrategy
from src.notification.logger import setup_logger
import config.donotshare.donotshare as donotshare

_logger = setup_logger("ibkr_screener_main")

async def main():
    _logger.info("Initializing IBKR Scalable Screener...")

    # 1. Configuration - Load from donotshare.py
    ibkr_host = donotshare.IBKR_HOST or "raspberrypi"
    ibkr_port = donotshare.IBKR_PAPER_PORT or donotshare.IBKR_PORT
    if ibkr_port:
        ibkr_port = int(ibkr_port)
    ibkr_client_id = int(donotshare.IBKR_CLIENT_ID) if donotshare.IBKR_CLIENT_ID else 3

    # 2. Setup Broker
    broker = IBKRBroker(
        host=ibkr_host,
        port=ibkr_port,
        client_id=ibkr_client_id
    )

    # 3. Setup Notification Client
    notif_client = NotificationServiceClient()
    notifier = SignalNotifier(notif_client)

    # 4. Strategy Config (Placeholder - should be dynamic or loaded from file)
    strategy_config = {
        "parameters": {
            "entry_logic": {
                "indicators": [
                    {"name": "RSI", "params": {"timeperiod": 14}, "alias": "rsi_14"}
                ]
            }
        },
        "warmup_period": 100,
        "bot_type": "optimization" # To disable DB logging
    }

    # 5. Discovery Providers
    portfolio_provider = PortfolioDiscovery(broker)
    # watchlist_path = os.path.join(PROJECT_ROOT, "config", "screeners", "watchlist.json")
    # static_provider = StaticDiscovery(watchlist_path)

    # 6. Initialize Service
    service = IBKRScreenerService(
        strategy_class=CustomStrategy,
        strategy_config=strategy_config,
        discovery_providers=[portfolio_provider],
        notifier=notifier,
        interval='1h',
        concurrency=50
    )

    # 7. Start Loop
    try:
        await service.start_scheduled_loop(run_every_minutes=15)
    except KeyboardInterrupt:
        _logger.info("Screener stopped by user.")
    except Exception as e:
        _logger.exception("Screener failed with critical error: %s", e)
    finally:
        await broker.disconnect()
        await notif_client.close()

if __name__ == "__main__":
    asyncio.run(main())

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone

from src.trading.strategy_manager import StrategyManager
from src.trading.instance_service import InstanceService
from src.trading.strategy_instance import StrategyInstance
from src.trading.base_trading_bot import BaseTradingBot
from src.trading.broker.mock_broker import MockBroker

@pytest.mark.asyncio
@pytest.mark.async_timeout(30)
class TestOrchestrationFlow:
    """
    Integration test for the full trading orchestration flow:
    Manager -> Service -> Instance -> Bot -> Broker
    """

    @pytest.fixture
    def mock_notification_client(self):
        client = MagicMock()
        client.send_notification = AsyncMock()
        return client

    @pytest.fixture
    def mock_trading_service(self):
        with patch("src.trading.strategy_instance.trading_service") as mock:
            mock.update_bot_status = MagicMock()
            yield mock

    @pytest.fixture
    def sample_config(self):
        return {
            "name": "TestBot",
            "symbol": "BTCUSDT",
            "enabled": True,
            "broker": {
                "type": "mock",
                "trading_mode": "paper",
                "cash": 10000.0
            },
            "strategy": {
                "type": "CustomStrategy",
                "parameters": {
                    "entry_logic": {"name": "SimpleEntryMixin", "params": {}},
                    "exit_logic": {"name": "SimpleExitMixin", "params": {}}
                }
            }
        }

    @patch("src.trading.strategy_instance.get_broker")
    @patch("src.trading.strategy_instance.DataFeedFactory.create_data_feed")
    @patch("src.trading.strategy_handler.strategy_handler.get_strategy_class")
    async def test_full_orchestration_lifecycle(self, mock_get_strat, mock_feed_factory, mock_get_broker, 
                                                mock_notification_client, mock_trading_service, sample_config):
        # 1. Setup mocks
        mock_broker = MagicMock(spec=MockBroker)
        mock_broker.is_connected = True
        mock_get_broker.return_value = mock_broker
        
        mock_feed = MagicMock()
        mock_feed.get_status.return_value = {"is_connected": True}
        mock_feed_factory.return_value = mock_feed
        
        # Provide a class-like mock for strategy
        mock_strat_class = MagicMock()
        mock_strat_class.__name__ = "MockStrategy"
        mock_get_strat.return_value = mock_strat_class

        # 2. Initialize Manager
        manager = StrategyManager(notification_client=mock_notification_client)
        manager.instance_service.instances = {}
        
        # 3. Load Strategy
        bot_id = "bot_123"
        sample_config["id"] = bot_id
        
        with patch("src.config.configuration_factory.config_factory.load_manifest", side_effect=lambda x: x):
            manager.instance_service.create_instance(bot_id, sample_config)

        # 4. Verify Instance was created
        assert bot_id in manager.instance_service.instances
        instance = manager.instance_service.instances[bot_id]

        # 5. Start Instance
        with patch.object(StrategyInstance, "_run_backtrader_async", new_callable=AsyncMock), \
             patch.object(StrategyInstance, "_start_trading_bot_loop", new_callable=AsyncMock), \
             patch.object(StrategyInstance, "_create_trading_bot", new_callable=AsyncMock, return_value=True):
            
            success = await manager.start_strategy(bot_id)
            if not success:
                print(f"FAILED. last_error: {instance.last_error}")
            assert success is True
            assert instance.status == "running"

        # 6. Stop Instance
        success = await manager.stop_strategy(bot_id)
        assert success is True
        assert instance.status == "stopped"
        
        assert mock_trading_service.update_bot_status.called

"""
Trading Bot Service
------------------

Service layer for trading bots that provides a clean interface
to the domain services. This acts as an adapter between the trading
bot implementation and the domain layer.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.data.db.services import trading_service
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TradingBotService:
    """
    Service for trading bot operations.

    This service provides the interface that trading bots expect
    while using the proper domain services underneath.
    """

    def __init__(self):
        """Initialize the trading bot service."""
        pass

    # ---------- Bot Instance Methods ----------

    def get_bot_instance(self, bot_id: str) -> Optional[Dict[str, Any]]:
        """Get bot instance by ID."""
        try:
            # The trading service doesn't have a direct get_bot method,
            # so we'll need to implement this or use the repository directly
            # For now, return None to indicate not found
            return None
        except Exception as e:
            _logger.exception("Error getting bot instance %s:", bot_id)
            return None

    def create_bot_instance(self, bot_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new bot instance."""
        try:
            return trading_service.upsert_bot(bot_data)
        except Exception as e:
            _logger.exception("Error creating bot instance:")
            raise

    def update_bot_instance(self, bot_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update bot instance."""
        try:
            # Add the bot_id to the update data
            bot_data = {"id": bot_id, **update_data}
            return trading_service.upsert_bot(bot_data)
        except Exception as e:
            _logger.exception("Error updating bot instance %s:", bot_id)
            raise

    def heartbeat(self, bot_id: str) -> None:
        """Update bot heartbeat."""
        try:
            trading_service.heartbeat(bot_id)
        except Exception as e:
            _logger.exception("Error updating heartbeat for bot %s:", bot_id)
            raise

    # ---------- Trade Methods ----------

    def create_trade(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new trade."""
        try:
            return trading_service.add_trade(trade_data)
        except Exception as e:
            _logger.exception("Error creating trade:")
            raise

    def update_trade(self, trade_id: str, update_data: Dict[str, Any]) -> bool:
        """Update trade."""
        try:
            return trading_service.close_trade(trade_id, **update_data)
        except Exception as e:
            _logger.exception("Error updating trade %s:", trade_id)
            raise

    def get_open_trades(self, bot_id: Optional[str] = None, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open trades."""
        try:
            # The current trading service doesn't filter by bot_id directly
            # We'll get all open trades and filter if needed
            trades = trading_service.get_open_trades(symbol=symbol)

            if bot_id:
                trades = [t for t in trades if t.get('bot_id') == bot_id]

            return trades
        except Exception as e:
            _logger.exception("Error getting open trades:")
            return []

    # ---------- Position Methods ----------

    def ensure_open_position(
        self,
        bot_id: str,
        trade_type: str,
        symbol: str,
        direction: str,
        opened_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ensure an open position exists."""
        try:
            return trading_service.ensure_open_position(
                bot_id=bot_id,
                trade_type=trade_type,
                symbol=symbol,
                direction=direction,
                opened_at=opened_at,
                metadata=metadata,
            )
        except Exception as e:
            _logger.exception("Error ensuring open position:")
            raise

    def get_open_positions(
        self,
        bot_id: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get open positions."""
        try:
            return trading_service.get_open_positions(bot_id=bot_id, symbol=symbol)
        except Exception as e:
            _logger.exception("Error getting open positions:")
            return []

    # ---------- Metrics Methods ----------

    def add_metric(self, metric_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add performance metric."""
        try:
            return trading_service.add_metric(metric_data)
        except Exception as e:
            _logger.exception("Error adding metric:")
            raise

    def get_latest_metrics(self, bot_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get latest metrics for bot."""
        try:
            return trading_service.latest_metrics(bot_id, limit=limit)
        except Exception as e:
            _logger.exception("Error getting latest metrics for bot %s:", bot_id)
            return []

    # ---------- Summary Methods ----------

    def get_pnl_summary(self, bot_id: Optional[str] = None) -> Dict[str, Any]:
        """Get PnL summary."""
        try:
            return trading_service.get_pnl_summary(bot_id=bot_id)
        except Exception as e:
            _logger.exception("Error getting PnL summary:")
            return {"net_pnl": 0.0, "n_trades": 0}

    # ---------- Additional Methods for Strategy Compatibility ----------

    def get_trade_by_id(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get trade by ID."""
        try:
            # The current trading service doesn't have a direct get_trade_by_id method
            # This would need to be implemented in the domain service
            # For now, return None
            _logger.warning("get_trade_by_id not implemented in domain service")
            return None
        except Exception as e:
            _logger.exception("Error getting trade by ID %s:", trade_id)
            return None

    def create_partial_exit_trade(self, trade_data: Dict[str, Any], original_trade_id: str) -> Dict[str, Any]:
        """Create partial exit trade."""
        try:
            # Add reference to original trade
            trade_data["original_trade_id"] = original_trade_id
            trade_data["is_partial_exit"] = True
            return trading_service.add_trade(trade_data)
        except Exception as e:
            _logger.exception("Error creating partial exit trade:")
            raise


# Create a singleton instance for easy import
trading_bot_service = TradingBotService()
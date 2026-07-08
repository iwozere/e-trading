"""
Unit tests for BaseStrategy (Backtrader) — pure-logic methods only.

These tests bypass Backtrader's cerebro/broker machinery by constructing
a partially-initialised instance via object.__new__ and manually setting
the attributes each method under test actually uses.  No network calls,
no broker, no data feeds required.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.strategy.base_strategy import BaseStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bare_strategy(**attrs) -> BaseStrategy:
    """
    Return a BaseStrategy instance without running __init__.

    Sets sensible defaults for every attribute referenced by the methods
    under test, then overlays any caller-supplied values.
    """
    obj = object.__new__(BaseStrategy)

    defaults = {
        "symbol": "",
        "timeframe": "",
        "asset_type": "crypto",
        "min_order_value": 0.0,
        "optimization_mode": False,
        "enable_database_logging": False,
        "config": {},
        "base_position_size": 0.1,
        "min_position_size": 0.05,
        "max_position_size": 0.20,
        "trades": [],
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "total_pnl": 0.0,
        "max_drawdown": 0.0,
        "peak_equity": 0.0,
        "entry_price": None,
        "exit_size": None,
        "current_position_size": None,
        "current_trade": None,
        "current_exit_reason": None,
        "current_entry_reason": None,
        "highest_profit": 0.0,
        "executed_exit_price": None,
        "order_refs": {},
        "equity_curve": [],
        "equity_dates": [],
        "indicators": {},
        "indicator_configs": [],
        "warmup_period": 0,
        "current_position_id": None,
        "bot_instance_id": None,
        "trade_repository": None,
        "bot_type": "paper",
    }
    defaults.update(attrs)
    for k, v in defaults.items():
        setattr(obj, k, v)

    return obj


# ---------------------------------------------------------------------------
# _simple_asset_type_detection
# ---------------------------------------------------------------------------


class TestSimpleAssetTypeDetection:
    def test_empty_symbol_returns_crypto(self):
        s = _bare_strategy(symbol="")
        assert s._simple_asset_type_detection() == "crypto"

    def test_btc_symbol_is_crypto(self):
        s = _bare_strategy(symbol="BTCUSDT")
        assert s._simple_asset_type_detection() == "crypto"

    def test_eth_symbol_is_crypto(self):
        s = _bare_strategy(symbol="ETH-USD")
        assert s._simple_asset_type_detection() == "crypto"

    def test_aapl_is_stock(self):
        s = _bare_strategy(symbol="AAPL")
        assert s._simple_asset_type_detection() == "stock"

    def test_msft_is_stock(self):
        s = _bare_strategy(symbol="MSFT")
        assert s._simple_asset_type_detection() == "stock"

    def test_usdt_pair_is_crypto(self):
        s = _bare_strategy(symbol="SOLUSDT")
        assert s._simple_asset_type_detection() == "crypto"


# ---------------------------------------------------------------------------
# _validate_position_size
# ---------------------------------------------------------------------------


class TestValidatePositionSize:
    # --- crypto ---

    def test_crypto_positive_fractional_is_valid(self):
        s = _bare_strategy(asset_type="crypto")
        assert s._validate_position_size(0.001) is True

    def test_crypto_zero_is_invalid(self):
        s = _bare_strategy(asset_type="crypto")
        assert s._validate_position_size(0.0) is False

    def test_crypto_negative_is_invalid(self):
        s = _bare_strategy(asset_type="crypto")
        assert s._validate_position_size(-1.0) is False

    # --- stock ---

    def test_stock_whole_share_is_valid(self):
        s = _bare_strategy(asset_type="stock")
        assert s._validate_position_size(10.0) is True

    def test_stock_fractional_is_invalid(self):
        s = _bare_strategy(asset_type="stock")
        assert s._validate_position_size(1.5) is False

    def test_stock_zero_is_invalid(self):
        s = _bare_strategy(asset_type="stock")
        assert s._validate_position_size(0.0) is False

    # --- unknown asset type ---

    def test_unknown_asset_type_less_than_one_is_invalid(self):
        s = _bare_strategy(asset_type="futures")
        assert s._validate_position_size(0.5) is False

    def test_unknown_asset_type_one_or_more_is_valid(self):
        s = _bare_strategy(asset_type="futures")
        assert s._validate_position_size(1.0) is True


# ---------------------------------------------------------------------------
# _calculate_position_size
# ---------------------------------------------------------------------------


class TestCalculatePositionSize:
    def test_default_returns_base_size(self):
        s = _bare_strategy(base_position_size=0.1, min_position_size=0.05, max_position_size=0.2)
        result = s._calculate_position_size()
        assert result == pytest.approx(0.1)

    def test_confidence_scales_size(self):
        s = _bare_strategy(base_position_size=0.1, min_position_size=0.05, max_position_size=0.2)
        result = s._calculate_position_size(confidence=0.5)
        assert result == pytest.approx(0.05)  # 0.1 * 0.5 = 0.05 (== min_position_size)

    def test_result_clamped_to_min(self):
        s = _bare_strategy(base_position_size=0.1, min_position_size=0.08, max_position_size=0.2)
        result = s._calculate_position_size(confidence=0.5)  # 0.05 < 0.08 → clamped to 0.08
        assert result == pytest.approx(0.08)

    def test_result_clamped_to_max(self):
        s = _bare_strategy(base_position_size=0.1, min_position_size=0.05, max_position_size=0.15)
        result = s._calculate_position_size(risk_multiplier=5.0)  # 0.5 > 0.15 → clamped to 0.15
        assert result == pytest.approx(0.15)

    def test_risk_multiplier_scales_size(self):
        s = _bare_strategy(base_position_size=0.1, min_position_size=0.05, max_position_size=0.5)
        result = s._calculate_position_size(risk_multiplier=2.0)
        assert result == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# _calculate_actual_trade_size
# ---------------------------------------------------------------------------


class TestCalculateActualTradeSize:
    def _make_open_trade(self, size: float) -> MagicMock:
        trade = MagicMock()
        trade.isclosed = False
        trade.size = size
        trade.pnl = 0.0
        trade.price = 100.0
        return trade

    def _make_closed_trade(self, size: float, pnl: float = 0.0, price: float = 100.0) -> MagicMock:
        trade = MagicMock()
        trade.isclosed = True
        trade.size = size
        trade.pnl = pnl
        trade.price = price
        return trade

    def test_open_trade_uses_abs_size(self):
        s = _bare_strategy(exit_size=None, current_position_size=None, entry_price=None)
        trade = self._make_open_trade(size=5.0)
        assert s._calculate_actual_trade_size(trade) == pytest.approx(5.0)

    def test_closed_trade_uses_exit_size_first(self):
        s = _bare_strategy(exit_size=3.0, current_position_size=10.0, entry_price=100.0)
        trade = self._make_closed_trade(size=5.0)
        assert s._calculate_actual_trade_size(trade) == pytest.approx(3.0)

    def test_closed_trade_falls_back_to_current_position_size(self):
        s = _bare_strategy(exit_size=None, current_position_size=7.0, entry_price=None)
        trade = self._make_closed_trade(size=5.0)
        assert s._calculate_actual_trade_size(trade) == pytest.approx(7.0)

    def test_closed_trade_final_fallback_is_one(self):
        s = _bare_strategy(exit_size=None, current_position_size=None, entry_price=None)
        trade = self._make_closed_trade(size=5.0, pnl=0.0)
        assert s._calculate_actual_trade_size(trade) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# get_performance_summary
# ---------------------------------------------------------------------------


class TestGetPerformanceSummary:
    def test_no_trades_returns_zeros(self):
        s = _bare_strategy(
            trades=[], total_trades=0, winning_trades=0, losing_trades=0, total_pnl=0.0, max_drawdown=0.0
        )
        result = s.get_performance_summary()
        assert result["total_trades"] == 0
        assert result["win_rate"] == pytest.approx(0.0)
        assert result["total_pnl"] == pytest.approx(0.0)

    def test_win_rate_calculated_correctly(self):
        s = _bare_strategy(
            trades=[{}],
            total_trades=4,
            winning_trades=3,
            losing_trades=1,
            total_pnl=200.0,
            max_drawdown=0.05,
            peak_equity=10000.0,
        )
        # Mock broker.getvalue() so the method doesn't fail on `self.broker`
        s.broker = MagicMock()
        s.broker.getvalue.return_value = 10200.0

        result = s.get_performance_summary()
        assert result["win_rate"] == pytest.approx(75.0)
        assert result["total_trades"] == 4
        assert result["total_pnl"] == pytest.approx(200.0)

    def test_avg_pnl_calculated_correctly(self):
        s = _bare_strategy(
            trades=[{}],
            total_trades=5,
            winning_trades=3,
            losing_trades=2,
            total_pnl=250.0,
            max_drawdown=0.0,
            peak_equity=0.0,
        )
        s.broker = MagicMock()
        s.broker.getvalue.return_value = 0.0

        result = s.get_performance_summary()
        assert result["avg_pnl"] == pytest.approx(50.0)

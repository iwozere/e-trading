
import pytest
from src.risk.controller import RiskController

@pytest.fixture
def sample_config():
    return {
        'risk_per_trade': 0.01,
        'max_position': 10000,
        'max_portfolio_exposure': 50000,
        'max_correlation': 0.9,
        'trailing_pct': 0.02,
        'account_equity': 100000,
        'target_volatility': 0.15,
        'max_drawdown': 0.2
    }

def test_pre_trade_checks_pass(sample_config):
    controller = RiskController(sample_config)
    size = controller.pre_trade_checks(
        account_equity=100000,
        stop_loss_pct=0.02,
        current_exposures={'BTC': 5000, 'ETH': 10000},
        correlation_matrix=None
    )
    assert size > 0

def test_pre_trade_checks_fail_on_limit(sample_config):
    controller = RiskController(sample_config)
    # intentionally too high exposure
    size = controller.pre_trade_checks(
        account_equity=100000,
        stop_loss_pct=0.02,
        current_exposures={'BTC': 50000, 'ETH': 30000},
        correlation_matrix=None
    )
    assert size == 0

def test_real_time_adjustments(sample_config):
    controller = RiskController(sample_config)
    result = controller.real_time_adjustments(
        entry_price=100,
        current_price=120,
        initial_stop=95,
        returns=[0.01, -0.005, 0.002, 0.015]
    )
    assert 'new_stop' in result
    assert 'volatility_size' in result
    assert result['new_stop'] > 95

def test_drawdown_check_pass(sample_config):
    controller = RiskController(sample_config)
    equity_curve = [100000, 98000, 97000, 96000, 95000]
    assert controller.drawdown_check(equity_curve) is True

def test_drawdown_check_fail(sample_config):
    controller = RiskController(sample_config)
    equity_curve = [100000, 80000]  # 20% drawdown
    assert controller.drawdown_check(equity_curve) is False

def test_post_trade_analysis(sample_config):
    controller = RiskController(sample_config)
    trades = [
        {'symbol': 'BTC', 'pnl': 200},
        {'symbol': 'BTC', 'pnl': -100},
        {'symbol': 'ETH', 'pnl': 50}
    ]
    report = controller.post_trade_analysis(trades)
    assert "RISK REPORT" in report
    assert "BTC" in report
    assert "win_rate" in report

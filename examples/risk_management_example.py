"""
Risk Management Example Usage

Demonstrates how to use the risk management module's main functions.
"""
import numpy as np
from src.risk.pre_trade import position_sizing, exposure_limits, correlation_check
from src.risk.real_time import stop_loss_manager, drawdown_control, volatility_scaling
from src.risk.post_trade import pnl_attribution, trade_analysis, risk_reporting

# --- Pre-Trade ---
# Kelly Criterion
kelly = position_sizing.kelly_criterion(win_prob=0.55, win_loss_ratio=1.5)
print(f"Kelly Criterion fraction: {kelly:.2%}")

# Fixed Fractional
size = position_sizing.fixed_fractional(account_equity=10000, risk_per_trade=0.01, stop_loss_pct=0.02)
print(f"Fixed Fractional position size: {size:.2f}")

# Exposure Limits
within_limit = exposure_limits.check_position_limit(current_position=500, max_position=1000)
print(f"Position within limit: {within_limit}")

portfolio_ok = exposure_limits.check_portfolio_limit({'BTC': 500, 'ETH': 300}, max_portfolio_exposure=1000)
print(f"Portfolio within limit: {portfolio_ok}")

# Correlation Check
corr_matrix = np.array([[1, 0.8], [0.8, 1]])
cor_ok = correlation_check.check_correlation_limit(corr_matrix, threshold=0.85)
print(f"Correlation within limit: {cor_ok}")

# --- Real-Time ---
# Dynamic Stop Loss
new_stop = stop_loss_manager.dynamic_stop_loss(entry_price=100, current_price=110, initial_stop=95, trailing_pct=0.05)
print(f"New stop-loss: {new_stop:.2f}")

# Drawdown Control
equity_curve = [10000, 9800, 9700, 9500, 9400]
dd_ok = drawdown_control.check_drawdown(equity_curve, max_drawdown_pct=0.1)
print(f"Drawdown within limit: {dd_ok}")

# Volatility Scaling
returns = np.random.normal(0, 0.01, 30)
vol_size = volatility_scaling.volatility_scaled_position(account_equity=10000, target_vol=0.15, returns=returns)
print(f"Volatility-scaled position size: {vol_size:.2f}")

# --- Post-Trade ---
# PnL Attribution
trades = [
    {'symbol': 'BTC', 'pnl': 100},
    {'symbol': 'ETH', 'pnl': -50},
    {'symbol': 'BTC', 'pnl': 200},
]
pnl_by_symbol = pnl_attribution.pnl_attribution(trades)
print(f"PnL by symbol: {pnl_by_symbol}")

# Trade Analysis
metrics = trade_analysis.trade_quality_metrics(trades)
print(f"Trade metrics: {metrics}")

# Risk Reporting
report = risk_reporting.generate_risk_report(metrics)
print(report) 
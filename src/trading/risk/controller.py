# src/risk/controller.py

from typing import Dict, List, Optional
from src.trading.risk.pre_trade import position_sizing, exposure_limits, correlation_check
from src.trading.risk.real_time import stop_loss_manager, drawdown_control, volatility_scaling
from src.trading.risk.post_trade import pnl_attribution, trade_analysis, risk_reporting


class RiskController:
    def __init__(self, config: Dict):
        self.config = config

    # --- PRE-TRADE PHASE ---
    def pre_trade_checks(
        self,
        account_equity: float,
        stop_loss_pct: float,
        current_exposures: Dict[str, float],
        entry_price: float = 0.0,
        correlation_matrix=None,
    ) -> float:
        """
        Run position sizing and limit checks before placing a trade.
        Returns position size in asset units if all checks pass; otherwise 0.0.

        Args:
            account_equity: Current account balance.
            stop_loss_pct: Stop loss as a fraction of entry price (e.g., 0.02 for 2%).
            current_exposures: Map of symbol → current notional exposure.
            entry_price: Current asset price used to convert risk dollars to units.
            correlation_matrix: Optional correlation matrix for correlation checks.
        """
        size = position_sizing.fixed_fractional(
            account_equity=account_equity,
            risk_per_trade=self.config.get('risk_per_trade', 0.01),
            stop_loss_pct=stop_loss_pct,
            entry_price=entry_price,
        )

        within_position_limits = exposure_limits.check_position_limit(
            current_position=size,
            max_position=self.config.get('max_position', 1e6)
        )

        within_portfolio_limits = exposure_limits.check_portfolio_limit(
            current_exposures=current_exposures,
            max_portfolio_exposure=self.config.get('max_portfolio_exposure', 1e7)
        )

        correlation_ok = True
        if correlation_matrix is not None:
            correlation_ok = correlation_check.check_correlation_limit(
                correlation_matrix=correlation_matrix,
                threshold=self.config.get('max_correlation', 0.9)
            )

        if within_position_limits and within_portfolio_limits and correlation_ok:
            return size
        return 0.0

    def pre_exit_checks(
        self,
        account_equity: float,
        current_exposures: Dict[str, float],
        symbol: str,
        exit_size: float,
        exit_price: float,
    ) -> bool:
        """
        Light-weight risk gate before closing a position (symmetric with pre-trade).
        Blocks obvious errors (size/price), fat-finger exits vs. recorded exposure,
        and portfolio exposure sanity checks.
        """
        if exit_size <= 0 or exit_price <= 0 or account_equity <= 0:
            return False
        sym_exp = float(current_exposures.get(symbol) or 0.0)
        if sym_exp <= 0:
            return False
        notional = exit_size * exit_price
        if notional > sym_exp * 1.01 + 1e-6:
            return False
        max_frac = float(self.config.get("max_single_trade_notional_fraction", 1.0))
        if max_frac < 1.0 and notional > max_frac * account_equity + 1e-9:
            return False
        return exposure_limits.check_portfolio_limit(
            current_exposures=current_exposures,
            max_portfolio_exposure=self.config.get("max_portfolio_exposure", 1e7),
        )

    # --- REAL-TIME PHASE ---
    def real_time_adjustments(
        self,
        entry_price: float,
        current_price: float,
        initial_stop: float,
        returns: List[float],
        account_equity: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Adjust stop-loss and calculate volatility-adjusted position size.

        Args:
            entry_price: Original trade entry price.
            current_price: Latest market price.
            initial_stop: Initial stop-loss price level.
            returns: Recent return series used for volatility estimation.
            account_equity: Current account balance.  When provided this
                value is used for volatility-scaled sizing so the position
                shrinks after losses.  Falls back to the (stale) value in
                ``self.config`` only when ``None`` is passed — callers
                should always supply the live balance.
        """
        trailing_pct = self.config.get('trailing_pct', 0.02)
        new_stop = stop_loss_manager.dynamic_stop_loss(
            entry_price=entry_price,
            current_price=current_price,
            initial_stop=initial_stop,
            trailing_pct=trailing_pct
        )

        # Prefer the caller-supplied live equity over the stale config value.
        effective_equity = (
            account_equity
            if account_equity is not None
            else self.config.get('account_equity', 100000)
        )
        volatility_size = volatility_scaling.volatility_scaled_position(
            account_equity=effective_equity,
            target_vol=self.config.get('target_volatility', 0.15),
            returns=returns
        )

        return {
            'new_stop': new_stop,
            'volatility_size': volatility_size
        }

    # --- CIRCUIT BREAKER ---
    def drawdown_check(self, equity_curve: List[float]) -> bool:
        """
        Check if max drawdown is within allowed limits.
        """
        max_dd = self.config.get('max_drawdown', 0.2)
        return drawdown_control.check_drawdown(equity_curve, max_dd)

    # --- POST-TRADE ANALYTICS ---
    def post_trade_analysis(self, trades: List[Dict]) -> str:
        """
        Run trade analysis and generate a risk report.
        """
        metrics = trade_analysis.trade_quality_metrics(trades)
        pnl_by_symbol = pnl_attribution.pnl_attribution(trades)
        full_metrics = {**metrics, **pnl_by_symbol}
        return risk_reporting.generate_risk_report(full_metrics)

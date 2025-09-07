"""
Custom Backtrader analyzers for advanced trading performance metrics.

This module provides several custom analyzers for use with Backtrader strategies, including:
- Calmar Ratio
- CAGR (Compound Annual Growth Rate)
- Sortino Ratio
- Profit Factor
- Win Rate
- Consecutive Wins/Losses
- Portfolio Volatility
"""

import math

import backtrader as bt
import numpy as np


class CalmarRatio(bt.Analyzer):
    """
    Analyzer to calculate the Calmar Ratio, which is the CAGR divided by the maximum drawdown.
    Uses CAGR and DrawDown analyzers to compute the metric.
    """
    params = (("riskfreerate", 0.0), ("timeframe", bt.TimeFrame.Years))

    def __init__(self):
        self.cagr_analyzer = CAGR()
        self.drawdown_analyzer = bt.analyzers.DrawDown()

    def stop(self):
        self.rets = {}

        # Получаем результаты от других анализаторов
        cagr = self.cagr_analyzer.get_analysis()["cagr"]
        max_dd = self.drawdown_analyzer.get_analysis()["max"]["drawdown"] / 100.0

        # Рассчитываем Calmar Ratio
        if max_dd != 0:
            calmar = (cagr - self.p.riskfreerate) / abs(max_dd)
        else:
            calmar = float("inf")

        self.rets["calmar"] = calmar

    def get_analysis(self):
        return self.rets


class CAGR(bt.Analyzer):
    """
    Analyzer to calculate the Compound Annual Growth Rate (CAGR) of the strategy's equity curve.
    """
    params = (("timeframe", bt.TimeFrame.Years),)

    def start(self):
        super(CAGR, self).start()
        self.start_value = self.strategy.broker.getvalue()

    def stop(self):
        self.rets = {}
        end_value = self.strategy.broker.getvalue()
        days = len(self.strategy)
        years = days / 365.0

        if years > 0 and self.start_value != 0:
            cagr = (end_value / self.start_value) ** (1.0 / years) - 1
            cagr *= 100  # в процентах
        else:
            cagr = 0.0

        self.rets["cagr"] = cagr

    def get_analysis(self):
        return self.rets


class SortinoRatio(bt.Analyzer):
    """
    Analyzer to calculate the Sortino Ratio, a risk-adjusted return metric that penalizes only downside volatility.
    """
    params = (("riskfreerate", 0.0), ("timeframe", bt.TimeFrame.Days))

    def start(self):
        self.returns = []

    def next(self):
        value = self.strategy.broker.getvalue()
        if len(self.returns) > 0:
            ret = (value / self.prev_value) - 1
            self.returns.append(ret)
        self.prev_value = value

    def stop(self):
        self.rets = {}
        returns = np.array(self.returns)

        if len(returns) == 0:
            self.rets["sortino"] = 0.0
            return

        # Рассчитываем downside deviation
        downside_returns = returns[returns < self.p.riskfreerate]
        if len(downside_returns) > 0:
            downside_dev = np.sqrt(
                np.mean((downside_returns - self.p.riskfreerate) ** 2)
            )
        else:
            # If no downside returns, use the standard deviation of all returns
            # This is a more conservative approach for the Sortino ratio
            downside_dev = np.std(returns)

        mean_return = np.mean(returns)

        if downside_dev > 0:
            sortino = (mean_return - self.p.riskfreerate) / downside_dev
        else:
            # If no volatility at all, return 0
            sortino = 0.0

        self.rets["sortino"] = sortino

    def get_analysis(self):
        return self.rets


class ProfitFactor(bt.Analyzer):
    """
    Analyzer to calculate the Profit Factor, defined as the ratio of gross profit to gross loss from closed trades.
    """
    def start(self):
        self.gross_profit = 0.0
        self.gross_loss = 0.0

    def notify_trade(self, trade):
        if trade.status == trade.Closed:
            if trade.pnl > 0:
                self.gross_profit += trade.pnl
            else:
                self.gross_loss += abs(trade.pnl)

    def stop(self):
        self.rets = {}
        if self.gross_loss != 0:
            pf = self.gross_profit / self.gross_loss
        else:
            pf = float("inf")
        self.rets["profit_factor"] = pf

    def get_analysis(self):
        return self.rets


class WinRate(bt.Analyzer):
    """
    Analyzer to calculate the win rate (percentage of winning trades), average win, and average loss.
    """
    def start(self):
        self.winning_trades = 0
        self.total_trades = 0
        self.win_amounts = []
        self.loss_amounts = []

    def notify_trade(self, trade):
        if trade.status == trade.Closed:
            self.total_trades += 1
            if trade.pnl > 0:
                self.winning_trades += 1
                self.win_amounts.append(trade.pnl)
            else:
                self.loss_amounts.append(abs(trade.pnl))

    def stop(self):
        self.rets = {}
        self.rets["win_rate"] = (
            (self.winning_trades / self.total_trades * 100)
            if self.total_trades > 0
            else 0.0
        )
        self.rets["avg_win"] = np.mean(self.win_amounts) if self.win_amounts else 0.0
        self.rets["avg_loss"] = np.mean(self.loss_amounts) if self.loss_amounts else 0.0

    def get_analysis(self):
        return self.rets


class ConsecutiveWinsLosses(bt.Analyzer):
    """
    Analyzer to track the maximum number of consecutive winning and losing trades.
    """
    def start(self):
        self.current_wins = 0
        self.current_losses = 0
        self.max_wins = 0
        self.max_losses = 0

    def notify_trade(self, trade):
        if trade.status == trade.Closed:
            if trade.pnl > 0:
                self.current_wins += 1
                self.current_losses = 0
                if self.current_wins > self.max_wins:
                    self.max_wins = self.current_wins
            else:
                self.current_losses += 1
                self.current_wins = 0
                if self.current_losses > self.max_losses:
                    self.max_losses = self.current_losses

    def stop(self):
        self.rets = {
            "max_consecutive_wins": self.max_wins,
            "max_consecutive_losses": self.max_losses,
        }

    def get_analysis(self):
        return self.rets


class PortfolioVolatility(bt.Analyzer):
    """
    Analyzer to calculate the volatility of the portfolio's returns, optionally annualized.
    """
    params = (("annualize", True),)

    def start(self):
        self.returns = []
        self.prev_value = None

    def next(self):
        current_value = self.strategy.broker.getvalue()
        if self.prev_value is not None and self.prev_value != 0:
            ret = (current_value - self.prev_value) / self.prev_value
            self.returns.append(ret)
        self.prev_value = current_value

    def stop(self):
        self.rets = {}
        if len(self.returns) > 1:
            volatility = np.std(self.returns)
            if self.p.annualize:
                # Годовая волатильность (предполагая дневные данные)
                volatility *= math.sqrt(252)
            self.rets["volatility"] = volatility
        else:
            self.rets["volatility"] = 0.0

    def get_analysis(self):
        return self.rets

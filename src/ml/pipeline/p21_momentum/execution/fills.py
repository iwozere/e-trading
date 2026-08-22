"""
P21 Momentum — Fill simulation (docs/pipeline-specification.md §10.1).

Fill at the adjusted open of the first trading day of the month, not the
prior close. Slippage always works against the trader: buys higher, sells
lower. Sells execute before buys; if cash is insufficient after that,
remaining buys are scaled down proportionally and WARN_INSUFFICIENT_CASH is
logged. Chatter threshold: skip a trade if |target - current| position
value < MIN_TRADE_USD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

from src.ml.pipeline.p21_momentum.config import (
    COMMISSION_MAX_PCT,
    COMMISSION_MIN_USD,
    COMMISSION_PER_SHARE,
    MIN_TRADE_USD,
    SLIPPAGE_BPS,
)

Side = Literal["BUY", "SELL"]


def simulate_fill(
    side: Side,
    shares: float,
    open_price: float,
    slippage_bps: float = SLIPPAGE_BPS,
    commission_min_usd: float = COMMISSION_MIN_USD,
    commission_per_share: float = COMMISSION_PER_SHARE,
    commission_max_pct: float = COMMISSION_MAX_PCT,
) -> Tuple[float, float]:
    """
    Simulate one fill at the adjusted open, per spec §10.1.

    Args:
        side: "BUY" or "SELL".
        shares: Absolute share quantity to trade (always positive).
        open_price: Adjusted open price for the fill date, from
            YahooDataDownloader.get_ohlcv_batch() (spec §2, §10.1).
        slippage_bps, commission_min_usd, commission_per_share,
            commission_max_pct: Spec §16 execution parameters.

    Returns:
        (fill_price, commission_usd). fill_price already includes slippage
        (buys higher, sells lower); commission_usd is
        min(max(commission_min, commission_per_share * shares), commission_max_pct * gross).
    """
    sign = 1 if side == "BUY" else -1
    fill = open_price * (1 + sign * slippage_bps / 10_000)
    gross = fill * shares
    comm = min(max(commission_min_usd, commission_per_share * shares), commission_max_pct * gross)
    return fill, comm


@dataclass(slots=True)
class TradeIntent:
    """One desired trade, before fill simulation."""

    ticker: str
    side: Side
    shares: float
    current_value_usd: float
    target_value_usd: float


@dataclass(slots=True)
class SimulatedTrade:
    """One trade after fill simulation, ready to become a LedgerEntry."""

    ticker: str
    side: Side
    shares: float
    ref_open: float
    fill_price: float
    slippage_bps: float
    commission_usd: float
    gross_usd: float
    net_usd: float


@dataclass(slots=True)
class ExecutionOutcome:
    """Result of execute_trades() — the simulated trades plus any warnings."""

    trades: List[SimulatedTrade]
    warn_insufficient_cash: bool


def apply_chatter_threshold(intents: List[TradeIntent], min_trade_usd: float = MIN_TRADE_USD) -> List[TradeIntent]:
    """Drop trades whose |target - current| value is below the chatter threshold."""
    return [i for i in intents if abs(i.target_value_usd - i.current_value_usd) >= min_trade_usd]


def execute_trades(
    intents: List[TradeIntent],
    open_prices: Dict[str, float],
    available_cash: float,
    slippage_bps: float = SLIPPAGE_BPS,
) -> ExecutionOutcome:
    """
    Simulate fills for a batch of trades, sells before buys, scaling buys down on insufficient cash.

    Args:
        intents: Desired trades, already past the chatter threshold.
        open_prices: {ticker: adjusted open price for the fill date}.
        available_cash: Cash on hand before this batch (sells replenish it
            as they execute).
        slippage_bps: Spec §16 slippage parameter.

    Returns:
        ExecutionOutcome with the simulated trades (sells first, in input
        order, then buys, in input order) and a WARN_INSUFFICIENT_CASH flag
        if buys had to be scaled down.
    """
    sells = [i for i in intents if i.side == "SELL"]
    buys = [i for i in intents if i.side == "BUY"]

    trades: List[SimulatedTrade] = []
    cash = available_cash

    for intent in sells:
        price = open_prices.get(intent.ticker)
        if price is None or price <= 0:
            continue
        fill, comm = simulate_fill("SELL", intent.shares, price, slippage_bps=slippage_bps)
        gross = fill * intent.shares
        net = gross - comm  # proceeds to cash, net of commission
        cash += net
        trades.append(
            SimulatedTrade(
                ticker=intent.ticker,
                side="SELL",
                shares=intent.shares,
                ref_open=price,
                fill_price=fill,
                slippage_bps=slippage_bps,
                commission_usd=comm,
                gross_usd=gross,
                net_usd=net,
            )
        )

    # Compute total buy cost at full size to see if scaling is needed.
    buy_costs: List[Tuple[TradeIntent, float, float, float]] = []  # (intent, price, fill, gross_est)
    total_buy_cost = 0.0
    for intent in buys:
        price = open_prices.get(intent.ticker)
        if price is None or price <= 0:
            continue
        fill, _ = simulate_fill("BUY", intent.shares, price, slippage_bps=slippage_bps)
        gross_est = fill * intent.shares
        buy_costs.append((intent, price, fill, gross_est))
        total_buy_cost += gross_est

    warn_insufficient_cash = False
    scale = 1.0
    if total_buy_cost > cash and total_buy_cost > 0:
        scale = max(cash / total_buy_cost, 0.0)
        warn_insufficient_cash = True

    for intent, price, fill, _ in buy_costs:
        scaled_shares = intent.shares * scale
        if scaled_shares <= 0:
            continue
        _, comm = simulate_fill("BUY", scaled_shares, price, slippage_bps=slippage_bps)
        gross = fill * scaled_shares
        net = gross + comm  # cost to cash, including commission
        trades.append(
            SimulatedTrade(
                ticker=intent.ticker,
                side="BUY",
                shares=scaled_shares,
                ref_open=price,
                fill_price=fill,
                slippage_bps=slippage_bps,
                commission_usd=comm,
                gross_usd=gross,
                net_usd=net,
            )
        )

    return ExecutionOutcome(trades=trades, warn_insufficient_cash=warn_insufficient_cash)

"""Unit tests for src.ml.pipeline.p21_momentum.execution.fills."""

from __future__ import annotations

import unittest

from src.ml.pipeline.p21_momentum.execution.fills import (
    TradeIntent,
    apply_chatter_threshold,
    execute_trades,
    simulate_fill,
)


class TestSimulateFill(unittest.TestCase):
    def test_buy_slips_up(self):
        fill, _ = simulate_fill("BUY", 10, 100.0, slippage_bps=3.0)
        self.assertAlmostEqual(fill, 100.0 * 1.0003)
        self.assertGreater(fill, 100.0)

    def test_sell_slips_down(self):
        fill, _ = simulate_fill("SELL", 10, 100.0, slippage_bps=3.0)
        self.assertAlmostEqual(fill, 100.0 * 0.9997)
        self.assertLess(fill, 100.0)

    def test_commission_floor_applies_for_small_trade(self):
        _, comm = simulate_fill("BUY", 1, 100.0)
        self.assertEqual(comm, 0.35)  # commission_min_usd floor

    def test_commission_cap_applies_for_low_priced_high_share_count(self):
        # per-share commission (0.0035 * shares) would exceed 1% of gross at
        # a low enough price; commission must be capped at 1% of gross.
        fill, comm = simulate_fill("BUY", 1_000, 0.10)
        gross = fill * 1_000
        self.assertAlmostEqual(comm, gross * 0.01)
        self.assertLess(comm, 0.0035 * 1_000)  # confirms the cap actually bound


class TestApplyChatterThreshold(unittest.TestCase):
    def test_small_delta_dropped(self):
        intents = [TradeIntent("A", "BUY", 1, current_value_usd=1000, target_value_usd=1100)]
        self.assertEqual(apply_chatter_threshold(intents), [])

    def test_large_delta_kept(self):
        intents = [TradeIntent("A", "BUY", 1, current_value_usd=1000, target_value_usd=1500)]
        result = apply_chatter_threshold(intents)
        self.assertEqual(len(result), 1)


class TestExecuteTrades(unittest.TestCase):
    def test_sells_execute_before_buys(self):
        intents = [
            TradeIntent("BUY_T", "BUY", 10, current_value_usd=0, target_value_usd=1000),
            TradeIntent("SELL_T", "SELL", 5, current_value_usd=500, target_value_usd=0),
        ]
        prices = {"BUY_T": 100.0, "SELL_T": 100.0}
        outcome = execute_trades(intents, prices, available_cash=0.0)
        sides = [t.side for t in outcome.trades]
        self.assertEqual(sides[0], "SELL")
        self.assertIn("BUY", sides)

    def test_sufficient_cash_no_warning(self):
        intents = [TradeIntent("A", "BUY", 10, current_value_usd=0, target_value_usd=1000)]
        outcome = execute_trades(intents, {"A": 100.0}, available_cash=10_000.0)
        self.assertFalse(outcome.warn_insufficient_cash)
        self.assertAlmostEqual(outcome.trades[0].shares, 10)

    def test_insufficient_cash_scales_down_buys_and_warns(self):
        intents = [TradeIntent("A", "BUY", 100, current_value_usd=0, target_value_usd=10_000)]
        outcome = execute_trades(intents, {"A": 100.0}, available_cash=1_000.0)
        self.assertTrue(outcome.warn_insufficient_cash)
        self.assertLess(outcome.trades[0].shares, 100)

    def test_sell_proceeds_fund_subsequent_buys(self):
        intents = [
            TradeIntent("SELL_T", "SELL", 50, current_value_usd=5000, target_value_usd=0),
            TradeIntent("BUY_T", "BUY", 40, current_value_usd=0, target_value_usd=4000),
        ]
        prices = {"SELL_T": 100.0, "BUY_T": 100.0}
        outcome = execute_trades(intents, prices, available_cash=0.0)
        buy_trade = next(t for t in outcome.trades if t.side == "BUY")
        self.assertFalse(outcome.warn_insufficient_cash)
        self.assertAlmostEqual(buy_trade.shares, 40)

    def test_missing_price_skips_trade(self):
        intents = [TradeIntent("NOPRICE", "BUY", 10, current_value_usd=0, target_value_usd=1000)]
        outcome = execute_trades(intents, {}, available_cash=10_000.0)
        self.assertEqual(outcome.trades, [])


if __name__ == "__main__":
    unittest.main()

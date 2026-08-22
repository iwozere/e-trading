"""Unit tests for src.ml.pipeline.p21_momentum.quality.gates. One test per §13 row, pass and fail sides."""

from __future__ import annotations

import unittest

from src.ml.pipeline.p21_momentum.quality.gates import (
    GateOutcome,
    GateResult,
    PipelineAbort,
    check_cash_after_execution,
    check_constituent_count,
    check_daily_price_change,
    check_no_weight_exceeds_cap,
    check_position_count,
    check_price_coverage,
    check_regime_inputs_available,
    check_signal_date_is_trading_day,
    check_target_weight_sum,
    run_gates,
)


class TestConstituentCount(unittest.TestCase):
    def test_pass(self):
        self.assertEqual(check_constituent_count(500, min_count=450).outcome, GateOutcome.PASS)

    def test_abort(self):
        self.assertEqual(check_constituent_count(400, min_count=450).outcome, GateOutcome.ABORT)


class TestPriceCoverage(unittest.TestCase):
    def test_pass(self):
        self.assertEqual(check_price_coverage(480, 500, min_pct=0.95).outcome, GateOutcome.PASS)

    def test_abort(self):
        self.assertEqual(check_price_coverage(400, 500, min_pct=0.95).outcome, GateOutcome.ABORT)


class TestRegimeInputsAvailable(unittest.TestCase):
    def test_pass(self):
        self.assertEqual(check_regime_inputs_available(True, True).outcome, GateOutcome.PASS)

    def test_hold_not_abort_on_failure(self):
        result = check_regime_inputs_available(False, True)
        self.assertEqual(result.outcome, GateOutcome.HOLD)


class TestSignalDateIsTradingDay(unittest.TestCase):
    def test_pass(self):
        self.assertEqual(check_signal_date_is_trading_day(True).outcome, GateOutcome.PASS)

    def test_abort(self):
        self.assertEqual(check_signal_date_is_trading_day(False).outcome, GateOutcome.ABORT)


class TestTargetWeightSum(unittest.TestCase):
    def test_pass_within_tolerance(self):
        result = check_target_weight_sum(50_000.5, 50_000.0, tolerance_usd=1.0)
        self.assertEqual(result.outcome, GateOutcome.PASS)

    def test_abort_outside_tolerance(self):
        result = check_target_weight_sum(50_010.0, 50_000.0, tolerance_usd=1.0)
        self.assertEqual(result.outcome, GateOutcome.ABORT)


class TestNoWeightExceedsCap(unittest.TestCase):
    def test_pass(self):
        result = check_no_weight_exceeds_cap(2500.0, nav_total=250_000, max_position_pct=0.01)
        self.assertEqual(result.outcome, GateOutcome.PASS)

    def test_abort(self):
        result = check_no_weight_exceeds_cap(3000.0, nav_total=250_000, max_position_pct=0.01)
        self.assertEqual(result.outcome, GateOutcome.ABORT)


class TestPositionCount(unittest.TestCase):
    def test_pass_at_20(self):
        self.assertEqual(check_position_count(20).outcome, GateOutcome.PASS)

    def test_warn_between_8_and_19(self):
        self.assertEqual(check_position_count(15).outcome, GateOutcome.WARN)

    def test_abort_below_8(self):
        self.assertEqual(check_position_count(5).outcome, GateOutcome.ABORT)


class TestCashAfterExecution(unittest.TestCase):
    def test_pass(self):
        self.assertEqual(check_cash_after_execution(100.0).outcome, GateOutcome.PASS)

    def test_abort_on_negative_cash(self):
        self.assertEqual(check_cash_after_execution(-0.01).outcome, GateOutcome.ABORT)


class TestDailyPriceChange(unittest.TestCase):
    def test_pass(self):
        result = check_daily_price_change("AAPL", 0.05, threshold_pct=0.50)
        self.assertEqual(result.outcome, GateOutcome.PASS)

    def test_warn_not_abort_on_breach(self):
        result = check_daily_price_change("AAPL", 0.60, threshold_pct=0.50)
        self.assertEqual(result.outcome, GateOutcome.WARN)


class TestRunGates(unittest.TestCase):
    def test_no_abort_returns_all(self):
        results = [
            GateResult("A", GateOutcome.PASS, "ok", {}),
            GateResult("B", GateOutcome.WARN, "meh", {}),
        ]
        self.assertEqual(run_gates(results), results)

    def test_abort_raises_pipeline_abort(self):
        results = [
            GateResult("A", GateOutcome.PASS, "ok", {}),
            GateResult("B", GateOutcome.ABORT, "bad", {"x": 1}),
        ]
        with self.assertRaises(PipelineAbort) as ctx:
            run_gates(results)
        self.assertEqual(ctx.exception.check, "B")


if __name__ == "__main__":
    unittest.main()

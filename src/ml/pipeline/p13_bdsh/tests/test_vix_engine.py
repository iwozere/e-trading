import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.ml.pipeline.p13_bdsh.vix_scaling_engine import VIXScalingEngine
from src.ml.pipeline.p13_bdsh.models import P13Config

class TestVIXScalingEngine(unittest.TestCase):
    def setUp(self):
        self.config = P13Config(
            entry_tiers={
                "Tier 1": {"z_threshold": 1.5, "allocation": 0.33},
                "Tier 2": {"z_threshold": 2.5, "allocation": 0.33},
                "Tier 3": {"z_threshold": 3.5, "allocation": 0.34},
            },
            exit_z_threshold=0.0,
            vix_lookback=10,
            initial_capital=100000.0,
            slippage_pct=0.0, # Zero slippage for deterministic testing
            atr_period=5,
            atr_multiplier=2.0
        )
        self.engine = VIXScalingEngine(self.config)

    def test_calculate_vix_zscore(self):
        # Create a series with a jump
        vix_data = [10.0] * 10 + [20.0]
        vix_series = pd.Series(vix_data)
        z_score = self.engine.calculate_vix_zscore(vix_series)
        
        self.assertEqual(len(z_score), 11)
        self.assertTrue(pd.isna(z_score.iloc[0]))
        # Last Z-score should be high because 20 is far from 10 mean
        self.assertGreater(z_score.iloc[-1], 2.0)

    def test_compute_target_exposure(self):
        # Test boundaries
        self.assertEqual(self.engine._compute_target_exposure(-1.0), 0.0)
        self.assertEqual(self.engine._compute_target_exposure(0.5), 0.0)
        self.assertAlmostEqual(self.engine._compute_target_exposure(1.6), 0.33)
        self.assertAlmostEqual(self.engine._compute_target_exposure(2.6), 0.66)
        self.assertAlmostEqual(self.engine._compute_target_exposure(3.6), 1.0)

    def test_calculate_atr(self):
        # Flat data
        data = {
            'high': [102, 102, 102, 102, 102, 102],
            'low': [100, 100, 100, 100, 100, 100],
            'close': [101, 101, 101, 101, 101, 101]
        }
        df = pd.DataFrame(data)
        atr = self.engine.calculate_atr(df, period=5)
        # TR should be 2 for all except first shift? No, TR calculation uses close shift.
        # [102-100, 102-?, 100-?] -> 2, 2, 2...
        # ATR should settle at 2.0
        self.assertAlmostEqual(atr.iloc[-1], 2.0)

    def test_run_backtest_stop_loss(self):
        # Create enough data for ATR (period=5) to warm up
        days = 25
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(days)]
        # Stable prices 100
        prices = [100.0] * 20 + [80.0] + [81.0] * 4
        
        ticker_df = pd.DataFrame({
            'high': prices,
            'low': [p - 1.0 for p in prices],
            'close': prices
        }, index=dates)
        
        # Low VIX for first 10 days, then High VIX
        z_scores = [0.1] * 10 + [3.6] * 15
        z_series = pd.Series(z_scores, index=dates)
        
        results = self.engine.run_backtest(ticker_df, z_series)
        
        # Entry should happen at index 11 (day 12) because prev_z at index 10 is 3.6
        # Stop loss should be triggered at index 20 (price 80)
        self.assertIn(dates[20], self.engine.markers["stop_loss"])
        self.assertTrue(results.iloc[20]["In_Cooldown"])
        self.assertEqual(results.iloc[20]["Target_Exposure"], 0.0)

if __name__ == '__main__':
    unittest.main()

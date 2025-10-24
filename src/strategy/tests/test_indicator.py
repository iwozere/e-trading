import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import unittest
import backtrader as bt
import numpy as np
import pandas as pd

# Import unified indicators from backtrader wrappers
from src.indicators.adapters.backtrader_wrappers import (
    UnifiedRSIIndicator as RsiIndicator,
    UnifiedBollingerBandsIndicator as BollingerBandIndicator,
    UnifiedMACDIndicator as MacdIndicator
)

# Import available unified indicators from backtrader wrappers
from src.indicators.adapters.backtrader_wrappers import (
    UnifiedATRIndicator as AtrIndicator,
    UnifiedSMAIndicator as MaIndicator,
    UnifiedEMAIndicator as EmaIndicator,
)

# Import SuperTrend from unified service
from src.indicators.adapters.backtrader_wrappers import UnifiedSuperTrendIndicator as SuperTrend

# Create placeholder classes for indicators not yet in unified service
class AdrIndicator(bt.Indicator):
    lines = ('adr',)
    params = (('period', 14),)
    def __init__(self):
        self.lines.adr = bt.indicators.SMA(self.data.close, period=self.params.period)

class AdxIndicator(bt.Indicator):
    lines = ('adx',)
    params = (('period', 14),)
    def __init__(self):
        self.lines.adx = bt.indicators.DirectionalMovementIndex(self.data, period=self.params.period)

class CciIndicator(bt.Indicator):
    lines = ('cci',)
    params = (('period', 20),)
    def __init__(self):
        self.lines.cci = bt.indicators.CommodityChannelIndex(self.data, period=self.params.period)

class ObvIndicator(bt.Indicator):
    lines = ('obv',)
    def __init__(self):
        self.lines.obv = bt.indicators.OnBalanceVolume(self.data)

class SarIndicator(bt.Indicator):
    lines = ('sar',)
    params = (('acceleration', 0.02), ('maximum', 0.2))
    def __init__(self):
        self.lines.sar = bt.indicators.ParabolicSAR(self.data, af=self.params.acceleration, afmax=self.params.maximum)

class AroonIndicator(bt.Indicator):
    lines = ('aroonup', 'aroondown')
    params = (('period', 25),)
    def __init__(self):
        aroon = bt.indicators.AroonIndicator(self.data, period=self.params.period)
        self.lines.aroonup = aroon.aroonup
        self.lines.aroondown = aroon.aroondown

class StochasticIndicator(bt.Indicator):
    lines = ('k', 'd')
    params = (('k_period', 14), ('d_period', 3))
    def __init__(self):
        stoch = bt.indicators.Stochastic(self.data, period=self.params.k_period, period_dfast=self.params.d_period)
        self.lines.k = stoch.percK
        self.lines.d = stoch.percD

class IchimokuIndicator(bt.Indicator):
    lines = ('tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou')
    params = (('tenkan_period', 9), ('kijun_period', 26), ('senkou_span_b_period', 52))
    def __init__(self):
        ichimoku = bt.indicators.Ichimoku(self.data,
                                         tenkan=self.params.tenkan_period,
                                         kijun=self.params.kijun_period,
                                         senkou=self.params.senkou_span_b_period)
        self.lines.tenkan = ichimoku.tenkan_sen
        self.lines.kijun = ichimoku.kijun_sen
        self.lines.senkou_a = ichimoku.senkou_span_a
        self.lines.senkou_b = ichimoku.senkou_span_b
        self.lines.chikou = ichimoku.chikou_span

HAS_SUPER_TREND = True  # SuperTrend is available through local implementation


def make_synthetic_df(length=100):
    idx = pd.date_range("2020-01-01", periods=length, freq="D")
    data = {
        "close": 100 + np.sin(np.arange(length) / 5) * 10,
        "open": 100 + np.cos(np.arange(length) / 5) * 10,
        "high": 105 + np.sin(np.arange(length) / 5) * 10,
        "low": 95 + np.sin(np.arange(length) / 5) * 10,
        "volume": 1000 + np.random.randint(0, 100, size=length),
    }
    return pd.DataFrame(data, index=idx)

class IndicatorTestBase(unittest.TestCase):
    def setUp(self):
        self.df = make_synthetic_df()
        self.bt_data = bt.feeds.PandasData(dataname=self.df)

class TestRSIIndicator(IndicatorTestBase):
    def test_rsi_all_types(self):
        # Test unified RSI indicator (simplified interface)
        cerebro = bt.Cerebro()
        cerebro.adddata(self.bt_data)
        cerebro.addindicator(RsiIndicator, period=14)
        try:
            cerebro.run(runonce=False, stdstats=False)
        except Exception as e:
            self.fail(f"RSI indicator raised an exception: {e}")

class TestMAIndicator(IndicatorTestBase):
    def test_ma_all_types(self):
        # Test SMA and EMA separately with unified interface
        cerebro = bt.Cerebro()
        cerebro.adddata(self.bt_data)
        cerebro.addindicator(MaIndicator, period=20)
        try:
            cerebro.run(runonce=False, stdstats=False)
        except Exception as e:
            self.fail(f"SMA indicator raised an exception: {e}")

        cerebro = bt.Cerebro()
        cerebro.adddata(self.bt_data)
        cerebro.addindicator(EmaIndicator, period=20)
        try:
            cerebro.run(runonce=False, stdstats=False)
        except Exception as e:
            self.fail(f"EMA indicator raised an exception: {e}")

class TestADRIndicator(IndicatorTestBase):
    def test_adr_bt(self):
        cerebro = bt.Cerebro()
        cerebro.adddata(self.bt_data)
        cerebro.addindicator(AdrIndicator, period=14)
        try:
            cerebro.run(runonce=False, stdstats=False)
        except Exception as e:
            self.fail(f"ADR indicator raised an exception: {e}")

class TestADXIndicator(IndicatorTestBase):
    def test_adx_all_types(self):
        cerebro = bt.Cerebro()
        cerebro.adddata(self.bt_data)
        cerebro.addindicator(AdxIndicator, period=14)
        try:
            cerebro.run(runonce=False, stdstats=False)
        except Exception as e:
            self.fail(f"ADX indicator raised an exception: {e}")

class TestATRIndicator(IndicatorTestBase):
    def test_atr_all_types(self):
        cerebro = bt.Cerebro()
        cerebro.adddata(self.bt_data)
        cerebro.addindicator(AtrIndicator, period=14)
        try:
            cerebro.run(runonce=False, stdstats=False)
        except Exception as e:
            self.fail(f"ATR indicator raised an exception: {e}")

class TestCCIIndicator(IndicatorTestBase):
    def test_cci_all_types(self):
        cerebro = bt.Cerebro()
        cerebro.adddata(self.bt_data)
        cerebro.addindicator(CciIndicator, period=20)
        try:
            cerebro.run(runonce=False, stdstats=False)
        except Exception as e:
            self.fail(f"CCI indicator raised an exception: {e}")

class TestOBVIndicator(IndicatorTestBase):
    def test_obv_all_types(self):
        cerebro = bt.Cerebro()
        cerebro.adddata(self.bt_data)
        cerebro.addindicator(ObvIndicator)
        try:
            cerebro.run(runonce=False, stdstats=False)
        except Exception as e:
            self.fail(f"OBV indicator raised an exception: {e}")

class TestSARIndicator(IndicatorTestBase):
    def test_sar_all_types(self):
        cerebro = bt.Cerebro()
        cerebro.adddata(self.bt_data)
        cerebro.addindicator(SarIndicator, acceleration=0.02, maximum=0.2)
        try:
            cerebro.run(runonce=False, stdstats=False)
        except Exception as e:
            self.fail(f"SAR indicator raised an exception: {e}")

class TestMACDIndicator(IndicatorTestBase):
    def test_macd_all_types(self):
        cerebro = bt.Cerebro()
        cerebro.adddata(self.bt_data)
        cerebro.addindicator(MacdIndicator, fast_period=12, slow_period=26, signal_period=9)
        try:
            cerebro.run(runonce=False, stdstats=False)
        except Exception as e:
            self.fail(f"MACD indicator raised an exception: {e}")

class TestAroonIndicator(IndicatorTestBase):
    def test_aroon_all_types(self):
        cerebro = bt.Cerebro()
        cerebro.adddata(self.bt_data)
        cerebro.addindicator(AroonIndicator, period=25)
        try:
            cerebro.run(runonce=False, stdstats=False)
        except Exception as e:
            self.fail(f"Aroon indicator raised an exception: {e}")

class TestStochasticIndicator(IndicatorTestBase):
    def test_stochastic_all_types(self):
        cerebro = bt.Cerebro()
        cerebro.adddata(self.bt_data)
        cerebro.addindicator(StochasticIndicator, k_period=14, d_period=3)
        try:
            cerebro.run(runonce=False, stdstats=False)
        except Exception as e:
            self.fail(f"Stochastic indicator raised an exception: {e}")

class TestIchimokuIndicator(IndicatorTestBase):
    def test_ichimoku_all_types(self):
        cerebro = bt.Cerebro()
        cerebro.adddata(self.bt_data)
        cerebro.addindicator(IchimokuIndicator, tenkan_period=9, kijun_period=26, senkou_span_b_period=52)
        try:
            cerebro.run(runonce=False, stdstats=False)
        except Exception as e:
            self.fail(f"Ichimoku indicator raised an exception: {e}")

class TestBollingerBandIndicator(IndicatorTestBase):
    def test_bollinger_band_all_types(self):
        cerebro = bt.Cerebro()
        cerebro.adddata(self.bt_data)
        cerebro.addindicator(BollingerBandIndicator, period=20, devfactor=2.0)
        try:
            cerebro.run(runonce=False, stdstats=False)
        except Exception as e:
            self.fail(f"BollingerBand indicator raised an exception: {e}")

if HAS_SUPER_TREND:
    class TestSuperTrendIndicator(IndicatorTestBase):
        def test_super_trend_bt(self):
            cerebro = bt.Cerebro()
            cerebro.adddata(self.bt_data)
            cerebro.addindicator(SuperTrend, period=10, multiplier=3.0)
            try:
                cerebro.run(runonce=False, stdstats=False)
            except Exception as e:
                self.fail(f"SuperTrend indicator raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()

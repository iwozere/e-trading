import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import unittest
import backtrader as bt
import numpy as np
import pandas as pd

from src.strategy.indicator.rsi import RsiIndicator
from src.strategy.indicator.ma import MaIndicator
from src.strategy.indicator.adr import AdrIndicator
from src.strategy.indicator.adx import AdxIndicator
from src.strategy.indicator.atr import AtrIndicator
from src.strategy.indicator.cci import CciIndicator
from src.strategy.indicator.obv import ObvIndicator
from src.strategy.indicator.sar import SarIndicator
from src.strategy.indicator.macd import MacdIndicator
from src.strategy.indicator.aroon import AroonIndicator
from src.strategy.indicator.stochastic import StochasticIndicator
from src.strategy.indicator.ichimoku import IchimokuIndicator
from src.strategy.indicator.bollinger_band import BollingerBandIndicator
# super_trend is not always present in all installs, but include if available
try:
    from src.strategy.indicator.super_trend import SuperTrend
    HAS_SUPER_TREND = True
except ImportError:
    HAS_SUPER_TREND = False


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
        self.bt_data = bt.feeds.PandasData(self.df)

class TestRSIIndicator(IndicatorTestBase):
    def test_rsi_all_types(self):
        indicator_types = ["bt", "bt-talib", "talib"]
        for ind_type in indicator_types:
            cerebro = bt.Cerebro()
            cerebro.adddata(self.bt_data)
            cerebro.addindicator(RsiIndicator, period=14, indicator_type=ind_type, line_names=("custom_rsi",))
            try:
                cerebro.run(runonce=False, stdstats=False)
            except Exception as e:
                self.fail(f"RSI indicator_type '{ind_type}' raised an exception: {e}")

class TestMAIndicator(IndicatorTestBase):
    def test_ma_all_types(self):
        indicator_types = ["bt", "bt-talib", "talib"]
        ma_types = ["sma", "ema"]
        for ind_type in indicator_types:
            for ma_type in ma_types:
                cerebro = bt.Cerebro()
                cerebro.adddata(self.bt_data)
                cerebro.addindicator(MaIndicator, period=20, ma_type=ma_type, indicator_type=ind_type, line_names=("custom_sma", "custom_ema"))
                try:
                    cerebro.run(runonce=False, stdstats=False)
                except Exception as e:
                    self.fail(f"MA indicator_type '{ind_type}', ma_type '{ma_type}' raised an exception: {e}")

class TestADRIndicator(IndicatorTestBase):
    def test_adr_bt(self):
        cerebro = bt.Cerebro()
        cerebro.adddata(self.bt_data)
        cerebro.addindicator(AdrIndicator, period=14, indicator_type="bt", line_names=("custom_adr",))
        try:
            cerebro.run(runonce=False, stdstats=False)
        except Exception as e:
            self.fail(f"ADR indicator_type 'bt' raised an exception: {e}")

class TestADXIndicator(IndicatorTestBase):
    def test_adx_all_types(self):
        indicator_types = ["bt", "bt-talib", "talib"]
        for ind_type in indicator_types:
            cerebro = bt.Cerebro()
            cerebro.adddata(self.bt_data)
            cerebro.addindicator(AdxIndicator, period=14, indicator_type=ind_type, line_names=("custom_adx",))
            try:
                cerebro.run(runonce=False, stdstats=False)
            except Exception as e:
                self.fail(f"ADX indicator_type '{ind_type}' raised an exception: {e}")

class TestATRIndicator(IndicatorTestBase):
    def test_atr_all_types(self):
        indicator_types = ["bt", "bt-talib", "talib"]
        for ind_type in indicator_types:
            cerebro = bt.Cerebro()
            cerebro.adddata(self.bt_data)
            cerebro.addindicator(AtrIndicator, period=14, indicator_type=ind_type, line_names=("custom_atr",))
            try:
                cerebro.run(runonce=False, stdstats=False)
            except Exception as e:
                self.fail(f"ATR indicator_type '{ind_type}' raised an exception: {e}")

class TestCCIIndicator(IndicatorTestBase):
    def test_cci_all_types(self):
        indicator_types = ["bt", "bt-talib", "talib"]
        for ind_type in indicator_types:
            cerebro = bt.Cerebro()
            cerebro.adddata(self.bt_data)
            cerebro.addindicator(CciIndicator, period=20, indicator_type=ind_type, line_names=("custom_cci",))
            try:
                cerebro.run(runonce=False, stdstats=False)
            except Exception as e:
                self.fail(f"CCI indicator_type '{ind_type}' raised an exception: {e}")

class TestOBVIndicator(IndicatorTestBase):
    def test_obv_all_types(self):
        indicator_types = ["bt-talib", "talib"]
        for ind_type in indicator_types:
            cerebro = bt.Cerebro()
            cerebro.adddata(self.bt_data)
            cerebro.addindicator(ObvIndicator, indicator_type=ind_type, line_names=("custom_obv",))
            try:
                cerebro.run(runonce=False, stdstats=False)
            except Exception as e:
                self.fail(f"OBV indicator_type '{ind_type}' raised an exception: {e}")

class TestSARIndicator(IndicatorTestBase):
    def test_sar_all_types(self):
        indicator_types = ["bt", "bt-talib", "talib"]
        for ind_type in indicator_types:
            cerebro = bt.Cerebro()
            cerebro.adddata(self.bt_data)
            cerebro.addindicator(SarIndicator, acceleration=0.02, maximum=0.2, indicator_type=ind_type, line_names=("custom_sar",))
            try:
                cerebro.run(runonce=False, stdstats=False)
            except Exception as e:
                self.fail(f"SAR indicator_type '{ind_type}' raised an exception: {e}")

class TestMACDIndicator(IndicatorTestBase):
    def test_macd_all_types(self):
        indicator_types = ["bt", "bt-talib", "talib"]
        for ind_type in indicator_types:
            cerebro = bt.Cerebro()
            cerebro.adddata(self.bt_data)
            cerebro.addindicator(MacdIndicator, fast_period=12, slow_period=26, signal_period=9, indicator_type=ind_type, line_names=("custom_macd", "custom_signal", "custom_histo"))
            try:
                cerebro.run(runonce=False, stdstats=False)
            except Exception as e:
                self.fail(f"MACD indicator_type '{ind_type}' raised an exception: {e}")

class TestAroonIndicator(IndicatorTestBase):
    def test_aroon_all_types(self):
        indicator_types = ["bt", "bt-talib", "talib"]
        for ind_type in indicator_types:
            cerebro = bt.Cerebro()
            cerebro.adddata(self.bt_data)
            cerebro.addindicator(AroonIndicator, period=25, indicator_type=ind_type, line_names=("custom_aroonup", "custom_aroondown"))
            try:
                cerebro.run(runonce=False, stdstats=False)
            except Exception as e:
                self.fail(f"Aroon indicator_type '{ind_type}' raised an exception: {e}")

class TestStochasticIndicator(IndicatorTestBase):
    def test_stochastic_all_types(self):
        indicator_types = ["bt", "bt-talib", "talib"]
        for ind_type in indicator_types:
            cerebro = bt.Cerebro()
            cerebro.adddata(self.bt_data)
            cerebro.addindicator(StochasticIndicator, k_period=14, d_period=3, indicator_type=ind_type, line_names=("custom_k", "custom_d"))
            try:
                cerebro.run(runonce=False, stdstats=False)
            except Exception as e:
                self.fail(f"Stochastic indicator_type '{ind_type}' raised an exception: {e}")

class TestIchimokuIndicator(IndicatorTestBase):
    def test_ichimoku_all_types(self):
        indicator_types = ["bt", "bt-talib", "talib"]
        for ind_type in indicator_types:
            cerebro = bt.Cerebro()
            cerebro.adddata(self.bt_data)
            cerebro.addindicator(IchimokuIndicator, tenkan_period=9, kijun_period=26, senkou_span_b_period=52, indicator_type=ind_type, line_names=("custom_tenkan", "custom_kijun", "custom_senkou_a", "custom_senkou_b", "custom_chikou"))
            try:
                cerebro.run(runonce=False, stdstats=False)
            except Exception as e:
                self.fail(f"Ichimoku indicator_type '{ind_type}' raised an exception: {e}")

class TestBollingerBandIndicator(IndicatorTestBase):
    def test_bollinger_band_all_types(self):
        indicator_types = ["bt", "bt-talib", "talib"]
        for ind_type in indicator_types:
            cerebro = bt.Cerebro()
            cerebro.adddata(self.bt_data)
            cerebro.addindicator(BollingerBandIndicator, period=20, devfactor=2.0, indicator_type=ind_type, line_names=("custom_upper", "custom_middle", "custom_lower"))
            try:
                cerebro.run(runonce=False, stdstats=False)
            except Exception as e:
                self.fail(f"BollingerBand indicator_type '{ind_type}' raised an exception: {e}")

if HAS_SUPER_TREND:
    class TestSuperTrendIndicator(IndicatorTestBase):
        def test_super_trend_bt(self):
            cerebro = bt.Cerebro()
            cerebro.adddata(self.bt_data)
            cerebro.addindicator(SuperTrend, period=10, multiplier=3.0, line_names=("custom_super_trend", "custom_direction", "custom_upper_band", "custom_lower_band"))
            try:
                cerebro.run(runonce=False, stdstats=False)
            except Exception as e:
                self.fail(f"SuperTrend indicator raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()

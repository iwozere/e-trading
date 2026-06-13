"""
Unit tests for all 9 entry mixins — signal generation contract.

Tests verify:
- should_enter() returns False when indicators are not ready
- should_enter() returns True on the happy path (all conditions met)
- should_enter() returns False when any single key condition fails
- get_minimum_lookback() returns a positive integer
- get_default_params() contains expected keys

No Backtrader runtime required. Strategy and indicator objects are lightweight mocks.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategy.entry.rsi_bb_entry_mixin import RSIBBEntryMixin
from src.strategy.entry.rsi_or_bb_entry_mixin import RSIOrBBEntryMixin
from src.strategy.entry.rsi_bb_volume_entry_mixin import RSIBBVolumeEntryMixin
from src.strategy.entry.rsi_ichimoku_entry_mixin import RSIIchimokuEntryMixin
from src.strategy.entry.rsi_volume_supertrend_entry_mixin import RSIVolumeSupertrendEntryMixin
from src.strategy.entry.bb_volume_supertrend_entry_mixin import BBVolumeSupertrendEntryMixin
from src.strategy.entry.eom_breakout_entry_mixin import EOMBreakoutEntryMixin
from src.strategy.entry.eom_pullback_entry_mixin import EOMPullbackEntryMixin
from src.strategy.entry.eom_macd_breakout_entry_mixin import EOMMAcdBreakoutEntryMixin
from src.strategy.entry.hmm_lstm_entry_mixin import HMMLSTMEntryMixin


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

class MockLine:
    """Mimics a Backtrader indicator line: supports [0] (current) and [-n] (history)."""

    def __init__(self, current, **history):
        """
        Args:
            current: Value at index 0.
            history: keyword args like neg1=28.0 set index -1, neg26=20.0 sets index -26.
        """
        self._values = {0: current}
        for k, v in history.items():
            if k.startswith("neg"):
                self._values[-int(k[3:])] = v
            else:
                self._values[int(k)] = v

    def __getitem__(self, idx):
        return self._values.get(idx, self._values.get(0, 0.0))


class MockData:
    """Minimal Backtrader data feed mock."""

    def __init__(self, close=100.0, volume=2000.0, open_=95.0, low=90.0, high=110.0,
                 prev_close=98.0):
        self.close = MockLine(close, neg1=prev_close)
        self.volume = MockLine(volume)
        self.open = MockLine(open_)
        self.low = MockLine(low)
        self.high = MockLine(high)
        self._len = 200

    def __len__(self):
        return self._len


class MockStrategy:
    """Minimal Backtrader strategy mock."""

    def __init__(self, indicators=None, data=None):
        self.indicators = indicators or {}
        self.data = data or MockData()


def _attach(mixin, indicators: dict, data=None) -> None:
    """Attach a mock strategy to the mixin, populating given indicators."""
    strategy = MockStrategy(indicators=indicators, data=data or MockData())
    mixin.init_entry(strategy)


# ---------------------------------------------------------------------------
# RSIBBEntryMixin  (RSI AND BB)
# ---------------------------------------------------------------------------

class TestRSIBBEntryMixin:

    def _mixin(self):
        return RSIBBEntryMixin()

    def test_default_params_keys(self):
        params = RSIBBEntryMixin.get_default_params()
        assert "rsi_period" in params
        assert "rsi_oversold" in params
        assert "bb_period" in params

    def test_minimum_lookback_positive(self):
        assert self._mixin().get_minimum_lookback() > 0

    def test_no_entry_when_indicators_missing(self):
        mixin = self._mixin()
        _attach(mixin, {})
        assert mixin.should_enter() is False

    def test_entry_when_rsi_oversold_and_price_below_bb(self):
        mixin = self._mixin()
        _attach(mixin, {
            'entry_rsi':       MockLine(25.0, neg1=28.0),
            'entry_bb_lower':  MockLine(102.0),
            'entry_bb_middle': MockLine(115.0),
        }, data=MockData(close=100.0))  # price (100) <= bb_lower (102)
        assert mixin.should_enter() is True

    def test_no_entry_when_rsi_not_oversold(self):
        mixin = self._mixin()
        _attach(mixin, {
            'entry_rsi':       MockLine(45.0, neg1=48.0),  # above oversold=30
            'entry_bb_lower':  MockLine(102.0),
            'entry_bb_middle': MockLine(115.0),
        }, data=MockData(close=100.0))
        assert mixin.should_enter() is False

    def test_no_entry_when_price_above_bb_lower(self):
        mixin = self._mixin()
        _attach(mixin, {
            'entry_rsi':       MockLine(25.0, neg1=28.0),
            'entry_bb_lower':  MockLine(90.0),   # price (100) > bb_lower (90)
            'entry_bb_middle': MockLine(110.0),
        }, data=MockData(close=100.0))
        assert mixin.should_enter() is False


# ---------------------------------------------------------------------------
# RSIOrBBEntryMixin  (RSI OR BB)
# ---------------------------------------------------------------------------

class TestRSIOrBBEntryMixin:

    def _mixin(self):
        return RSIOrBBEntryMixin()

    def test_default_params_keys(self):
        params = RSIOrBBEntryMixin.get_default_params()
        assert "rsi_oversold" in params

    def test_minimum_lookback_positive(self):
        assert self._mixin().get_minimum_lookback() > 0

    def test_no_entry_when_indicators_missing(self):
        mixin = self._mixin()
        _attach(mixin, {})
        assert mixin.should_enter() is False

    def test_entry_when_only_rsi_oversold(self):
        mixin = self._mixin()
        _attach(mixin, {
            'entry_rsi':       MockLine(25.0, neg1=28.0),
            'entry_bb_lower':  MockLine(85.0),   # price (100) > bb_lower → BB false
            'entry_bb_middle': MockLine(110.0),
        }, data=MockData(close=100.0))
        assert mixin.should_enter() is True  # RSI condition alone is enough

    def test_entry_when_only_price_below_bb(self):
        mixin = self._mixin()
        _attach(mixin, {
            'entry_rsi':       MockLine(50.0, neg1=52.0),  # not oversold
            'entry_bb_lower':  MockLine(105.0),  # price (100) <= bb_lower → BB true
            'entry_bb_middle': MockLine(120.0),
        }, data=MockData(close=100.0))
        assert mixin.should_enter() is True  # BB condition alone is enough

    def test_no_entry_when_neither_condition_met(self):
        mixin = self._mixin()
        _attach(mixin, {
            'entry_rsi':       MockLine(50.0, neg1=52.0),
            'entry_bb_lower':  MockLine(85.0),
            'entry_bb_middle': MockLine(105.0),
        }, data=MockData(close=100.0))
        assert mixin.should_enter() is False


# ---------------------------------------------------------------------------
# RSIBBVolumeEntryMixin  (RSI AND BB AND Volume)
# ---------------------------------------------------------------------------

class TestRSIBBVolumeEntryMixin:

    def _mixin(self):
        return RSIBBVolumeEntryMixin()

    def test_default_params_keys(self):
        params = RSIBBVolumeEntryMixin.get_default_params()
        assert "rsi_period" in params

    def test_no_entry_when_indicators_missing(self):
        mixin = self._mixin()
        _attach(mixin, {})
        assert mixin.should_enter() is False

    def test_entry_on_happy_path(self):
        mixin = self._mixin()
        _attach(mixin, {
            'entry_rsi':      MockLine(25.0),
            'entry_bb_lower': MockLine(102.0),
            'entry_volume_ma': MockLine(1000.0),
        }, data=MockData(close=100.0, volume=1200.0))
        # RSI<=30, price(100)<=bb_lower(102), volume(1200) > vol_ma(1000)*1.1(1100)
        assert mixin.should_enter() is True

    def test_no_entry_when_volume_insufficient(self):
        mixin = self._mixin()
        _attach(mixin, {
            'entry_rsi':      MockLine(25.0),
            'entry_bb_lower': MockLine(102.0),
            'entry_volume_ma': MockLine(2000.0),  # needs >2200, but volume=1200
        }, data=MockData(close=100.0, volume=1200.0))
        assert mixin.should_enter() is False

    def test_minimum_lookback_positive(self):
        assert self._mixin().get_minimum_lookback() > 0


# ---------------------------------------------------------------------------
# RSIIchimokuEntryMixin  (Price > kumo AND RSI oversold AND tenkan crossover)
# ---------------------------------------------------------------------------

class TestRSIIchimokuEntryMixin:

    def _mixin(self):
        return RSIIchimokuEntryMixin()

    def test_default_params_keys(self):
        params = RSIIchimokuEntryMixin.get_default_params()
        assert "rsi_oversold" in params
        assert "kijun_period" in params

    def test_no_entry_when_indicators_missing(self):
        mixin = self._mixin()
        _attach(mixin, {})
        assert mixin.should_enter() is False

    def test_entry_on_happy_path(self):
        mixin = self._mixin()
        # Conditions: price(100) > kumo_top(80), RSI(25)<=30, crossover (prev_price(95)<=prev_tenkan(96), price(100)>tenkan(99))
        _attach(mixin, {
            'entry_rsi':                MockLine(25.0),
            'entry_ichimoku_tenkan':    MockLine(99.0, neg1=96.0),
            'entry_ichimoku_senkou_a':  MockLine(0.0, neg26=75.0),  # kumo 26 bars back
            'entry_ichimoku_senkou_b':  MockLine(0.0, neg26=80.0),
        }, data=MockData(close=100.0, prev_close=95.0))
        assert mixin.should_enter() is True

    def test_no_entry_when_price_inside_kumo(self):
        mixin = self._mixin()
        # kumo_top = max(75, 120) = 120; price=100 < 120 → should not enter
        _attach(mixin, {
            'entry_rsi':                MockLine(25.0),
            'entry_ichimoku_tenkan':    MockLine(99.0, neg1=96.0),
            'entry_ichimoku_senkou_a':  MockLine(0.0, neg26=75.0),
            'entry_ichimoku_senkou_b':  MockLine(0.0, neg26=120.0),
        }, data=MockData(close=100.0, prev_close=95.0))
        assert mixin.should_enter() is False

    def test_no_entry_when_rsi_not_oversold(self):
        mixin = self._mixin()
        _attach(mixin, {
            'entry_rsi':                MockLine(55.0),  # not oversold
            'entry_ichimoku_tenkan':    MockLine(99.0, neg1=96.0),
            'entry_ichimoku_senkou_a':  MockLine(0.0, neg26=75.0),
            'entry_ichimoku_senkou_b':  MockLine(0.0, neg26=80.0),
        }, data=MockData(close=100.0, prev_close=95.0))
        assert mixin.should_enter() is False

    def test_minimum_lookback_positive(self):
        assert self._mixin().get_minimum_lookback() > 0


# ---------------------------------------------------------------------------
# RSIVolumeSupertrendEntryMixin  (RSI AND Volume AND Supertrend=uptrend)
# ---------------------------------------------------------------------------

class TestRSIVolumeSupertrendEntryMixin:

    def _mixin(self):
        return RSIVolumeSupertrendEntryMixin()

    def test_default_params_keys(self):
        params = RSIVolumeSupertrendEntryMixin.get_default_params()
        assert "rsi_oversold" in params

    def test_no_entry_when_indicators_missing(self):
        mixin = self._mixin()
        _attach(mixin, {})
        assert mixin.should_enter() is False

    def test_entry_on_happy_path(self):
        mixin = self._mixin()
        # RSI(25)<=30, volume(3000)>vol_ma(1000)*1.5, supertrend=1 (uptrend)
        _attach(mixin, {
            'entry_rsi':                  MockLine(25.0),
            'entry_volume_ma':            MockLine(1000.0),
            'entry_supertrend_direction': MockLine(1),
        }, data=MockData(volume=3000.0))
        assert mixin.should_enter() is True

    def test_no_entry_when_downtrend(self):
        mixin = self._mixin()
        _attach(mixin, {
            'entry_rsi':                  MockLine(25.0),
            'entry_volume_ma':            MockLine(1000.0),
            'entry_supertrend_direction': MockLine(-1),  # downtrend
        }, data=MockData(volume=3000.0))
        assert mixin.should_enter() is False

    def test_no_entry_when_rsi_not_oversold(self):
        mixin = self._mixin()
        _attach(mixin, {
            'entry_rsi':                  MockLine(50.0),
            'entry_volume_ma':            MockLine(1000.0),
            'entry_supertrend_direction': MockLine(1),
        }, data=MockData(volume=3000.0))
        assert mixin.should_enter() is False

    def test_minimum_lookback_positive(self):
        assert self._mixin().get_minimum_lookback() > 0


# ---------------------------------------------------------------------------
# BBVolumeSupertrendEntryMixin  (BB AND Volume AND Supertrend=uptrend)
# ---------------------------------------------------------------------------

class TestBBVolumeSupertrendEntryMixin:

    def _mixin(self):
        return BBVolumeSupertrendEntryMixin()

    def test_default_params_keys(self):
        params = BBVolumeSupertrendEntryMixin.get_default_params()
        assert "bb_period" in params

    def test_no_entry_when_indicators_missing(self):
        mixin = self._mixin()
        _attach(mixin, {})
        assert mixin.should_enter() is False

    def test_entry_on_happy_path(self):
        mixin = self._mixin()
        # price(100)<=bb_lower(102), volume(2000)>vol_ma(1000)*1.1, supertrend=1
        _attach(mixin, {
            'entry_bb_lower':             MockLine(102.0),
            'entry_volume_ma':            MockLine(1000.0),
            'entry_supertrend_direction': MockLine(1),
        }, data=MockData(close=100.0, volume=2000.0))
        assert mixin.should_enter() is True

    def test_no_entry_when_price_above_bb(self):
        mixin = self._mixin()
        _attach(mixin, {
            'entry_bb_lower':             MockLine(90.0),  # price(100) > bb_lower(90)
            'entry_volume_ma':            MockLine(1000.0),
            'entry_supertrend_direction': MockLine(1),
        }, data=MockData(close=100.0, volume=2000.0))
        assert mixin.should_enter() is False

    def test_no_entry_when_downtrend(self):
        mixin = self._mixin()
        _attach(mixin, {
            'entry_bb_lower':             MockLine(102.0),
            'entry_volume_ma':            MockLine(1000.0),
            'entry_supertrend_direction': MockLine(-1),
        }, data=MockData(close=100.0, volume=2000.0))
        assert mixin.should_enter() is False

    def test_minimum_lookback_positive(self):
        assert self._mixin().get_minimum_lookback() > 0


# ---------------------------------------------------------------------------
# EOMBreakoutEntryMixin  (Breakout + EOM + Volume + RSI not overbought)
# ---------------------------------------------------------------------------

class TestEOMBreakoutEntryMixin:

    def _mixin(self):
        return EOMBreakoutEntryMixin(params={"use_atr_filter": False})

    def test_default_params_keys(self):
        params = EOMBreakoutEntryMixin.get_default_params()
        assert "breakout_threshold" in params
        assert "rsi_overbought" in params

    def test_no_entry_when_indicators_missing(self):
        mixin = self._mixin()
        _attach(mixin, {})
        assert mixin.should_enter() is False

    def test_entry_on_happy_path(self):
        mixin = self._mixin()
        resistance = 98.0
        breakout_threshold = 0.002
        close = resistance * (1 + breakout_threshold) + 0.5  # clearly above breakout level
        _attach(mixin, {
            'entry_resistance':  MockLine(resistance),
            'entry_eom':         MockLine(0.5, neg1=0.3),   # EOM>0 and rising
            'entry_volume_sma':  MockLine(1000.0),
            'entry_rsi':         MockLine(50.0),             # not overbought (<70)
        }, data=MockData(close=close, volume=1500.0))
        assert mixin.should_enter() is True

    def test_no_entry_when_no_breakout(self):
        mixin = self._mixin()
        _attach(mixin, {
            'entry_resistance':  MockLine(100.0),
            'entry_eom':         MockLine(0.5, neg1=0.3),
            'entry_volume_sma':  MockLine(1000.0),
            'entry_rsi':         MockLine(50.0),
        }, data=MockData(close=95.0, volume=1500.0))  # no breakout
        assert mixin.should_enter() is False

    def test_no_entry_when_eom_negative(self):
        mixin = self._mixin()
        resistance = 98.0
        close = resistance * 1.003 + 0.5
        _attach(mixin, {
            'entry_resistance':  MockLine(resistance),
            'entry_eom':         MockLine(-0.1, neg1=0.3),  # EOM negative → not bullish
            'entry_volume_sma':  MockLine(1000.0),
            'entry_rsi':         MockLine(50.0),
        }, data=MockData(close=close, volume=1500.0))
        assert mixin.should_enter() is False

    def test_no_entry_when_rsi_overbought(self):
        mixin = self._mixin()
        resistance = 98.0
        close = resistance * 1.003 + 0.5
        _attach(mixin, {
            'entry_resistance':  MockLine(resistance),
            'entry_eom':         MockLine(0.5, neg1=0.3),
            'entry_volume_sma':  MockLine(1000.0),
            'entry_rsi':         MockLine(75.0),  # overbought
        }, data=MockData(close=close, volume=1500.0))
        assert mixin.should_enter() is False

    def test_minimum_lookback_positive(self):
        assert self._mixin().get_minimum_lookback() > 0


# ---------------------------------------------------------------------------
# EOMPullbackEntryMixin  (Support bounce + EOM cross-up + RSI recovery + ATR)
# ---------------------------------------------------------------------------

class TestEOMPullbackEntryMixin:

    def _mixin(self):
        return EOMPullbackEntryMixin()

    def test_default_params_keys(self):
        params = EOMPullbackEntryMixin.get_default_params()
        assert "rsi_oversold" in params
        assert "support_threshold" in params

    def test_no_entry_when_indicators_missing(self):
        mixin = self._mixin()
        _attach(mixin, {})
        assert mixin.should_enter() is False

    def test_entry_on_happy_path(self):
        mixin = self._mixin()
        support = 90.0
        # low(91) <= support*(1+0.005)=90.45 — fails! Need low <= support_level
        # support_level = 90 * 1.005 = 90.45 → low must be <= 90.45
        low = 90.0
        _attach(mixin, {
            'entry_support':  MockLine(support),
            'entry_eom':      MockLine(0.1, neg1=-0.05),  # crosses above 0
            'entry_rsi':      MockLine(35.0, neg1=33.0),  # oversold(<40) and rising
            'entry_atr':      MockLine(2.0),
            'entry_atr_sma':  MockLine(1.5),              # atr(2)>atr_sma(1.5)*0.9
        }, data=MockData(close=95.0, open_=89.0, low=low))  # close>open (reversal candle)
        assert mixin.should_enter() is True

    def test_no_entry_when_eom_not_crossing_up(self):
        mixin = self._mixin()
        support = 90.0
        _attach(mixin, {
            'entry_support':  MockLine(support),
            'entry_eom':      MockLine(0.1, neg1=0.2),   # already positive — no cross
            'entry_rsi':      MockLine(35.0, neg1=33.0),
            'entry_atr':      MockLine(2.0),
            'entry_atr_sma':  MockLine(1.5),
        }, data=MockData(close=95.0, open_=89.0, low=90.0))
        assert mixin.should_enter() is False

    def test_no_entry_when_rsi_falling(self):
        mixin = self._mixin()
        support = 90.0
        _attach(mixin, {
            'entry_support':  MockLine(support),
            'entry_eom':      MockLine(0.1, neg1=-0.05),
            'entry_rsi':      MockLine(35.0, neg1=38.0),  # falling (35 < 38)
            'entry_atr':      MockLine(2.0),
            'entry_atr_sma':  MockLine(1.5),
        }, data=MockData(close=95.0, open_=89.0, low=90.0))
        assert mixin.should_enter() is False

    def test_minimum_lookback_positive(self):
        assert self._mixin().get_minimum_lookback() > 0


# ---------------------------------------------------------------------------
# EOMMAcdBreakoutEntryMixin  (MACD crossover + near resistance + EOM>0 + volume)
# ---------------------------------------------------------------------------

class TestEOMMAcdBreakoutEntryMixin:

    def _mixin(self):
        return EOMMAcdBreakoutEntryMixin()

    def test_default_params_keys(self):
        params = EOMMAcdBreakoutEntryMixin.get_default_params()
        assert "resistance_range_low" in params
        assert "volume_threshold" in params

    def test_no_entry_when_indicators_missing(self):
        mixin = self._mixin()
        _attach(mixin, {})
        assert mixin.should_enter() is False

    def test_entry_on_happy_path(self):
        mixin = self._mixin()
        resistance = 100.0
        # near resistance: close in [100*0.995=99.5, 100*1.002=100.2]
        close = 100.0
        _attach(mixin, {
            'entry_resistance':   MockLine(resistance),
            'entry_macd':         MockLine(0.5, neg1=-0.1),     # MACD crossover above signal
            'entry_macd_signal':  MockLine(0.3, neg1=0.0),      # was equal/below
            'entry_macd_hist':    MockLine(0.2, neg1=0.1),      # histogram rising
            'entry_eom':          MockLine(0.3),                 # EOM positive
            'entry_volume_sma':   MockLine(1000.0),
        }, data=MockData(close=close, volume=900.0))  # volume(900)>=vol_floor(0.8*1000=800)
        assert mixin.should_enter() is True

    def test_no_entry_when_no_macd_crossover(self):
        mixin = self._mixin()
        resistance = 100.0
        _attach(mixin, {
            'entry_resistance':   MockLine(resistance),
            'entry_macd':         MockLine(0.5, neg1=0.6),   # MACD above signal but wasn't crossing (prev also above)
            'entry_macd_signal':  MockLine(0.3, neg1=0.2),
            'entry_macd_hist':    MockLine(0.2, neg1=0.1),
            'entry_eom':          MockLine(0.3),
            'entry_volume_sma':   MockLine(1000.0),
        }, data=MockData(close=100.0, volume=900.0))
        # macd_prev(0.6) > macd_signal_prev(0.2) → already crossed, not a fresh crossover
        assert mixin.should_enter() is False

    def test_no_entry_when_price_not_near_resistance(self):
        mixin = self._mixin()
        resistance = 100.0
        _attach(mixin, {
            'entry_resistance':   MockLine(resistance),
            'entry_macd':         MockLine(0.5, neg1=-0.1),
            'entry_macd_signal':  MockLine(0.3, neg1=0.0),
            'entry_macd_hist':    MockLine(0.2, neg1=0.1),
            'entry_eom':          MockLine(0.3),
            'entry_volume_sma':   MockLine(1000.0),
        }, data=MockData(close=85.0, volume=900.0))  # far from resistance
        assert mixin.should_enter() is False

    def test_minimum_lookback_positive(self):
        assert self._mixin().get_minimum_lookback() > 0


# ---------------------------------------------------------------------------
# HMMLSTMEntryMixin  (ML-model based — graceful fallback when models absent)
# ---------------------------------------------------------------------------

class TestHMMLSTMEntryMixin:

    def test_default_params_keys(self):
        params = HMMLSTMEntryMixin.get_default_params()
        assert "prediction_threshold" in params
        assert "regime_confidence_threshold" in params

    def test_instantiates_without_models(self):
        mixin = HMMLSTMEntryMixin()
        assert mixin is not None
        assert mixin.hmm_model is None
        assert mixin.lstm_model is None

    def test_no_entry_when_models_not_loaded(self):
        mixin = HMMLSTMEntryMixin()
        strategy = MockStrategy(indicators={}, data=MockData())
        mixin.init_entry(strategy)
        # is_initialized=False → are_indicators_ready()=False → should_enter()=False
        assert mixin.should_enter() is False

    def test_minimum_lookback_positive(self):
        mixin = HMMLSTMEntryMixin()
        assert mixin.get_minimum_lookback() >= 1

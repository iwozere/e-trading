import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.shared.indicators.adapters import RSI, BBANDS

# Create a combined indicator factory for the strategy logic
# This allows us to broadcast multiple parameters across multiple assets in one go.

class SignalFactory:
    """
    Orchestrates the generation of entry and exit signals using vectorbt IndicatorFactory.
    """

    @staticmethod
    def get_signals(
        data: pd.DataFrame,
        rsi_window: int = 14,
        rsi_lower: float = 30,
        rsi_upper: float = 70,
        bb_window: int = 20,
        bb_std: float = 2.0
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculates signals based on RSI and Bollinger Bands.
        Returns entries and exits DataFrames.
        """
        # Vectorbt expects close prices for RSI and BBands
        close = data.vbt.ohlcv.close

        # 1. RSI
        rsi = RSI.compute(close, window=rsi_window)

        # 2. Bollinger Bands
        bb = BBANDS.compute(close, window=bb_window, nbdevup=bb_std, nbdevdn=bb_std)
        upper = bb['upperband']
        lower = bb['lowerband']

        # Strategy Logic:
        # Long Entry: RSI < rsi_lower AND close < lower_band
        # Short Entry: RSI > rsi_upper AND close > upper_band
        # Exit: RSI cross 50 or close cross middle band?
        # (For simplicity let's use fixed exits or just opposite signals)

        entries = (rsi < rsi_lower) & (close < lower)
        short_entries = (rsi > rsi_upper) & (close > upper)

        # Exits - for now just opposite signals or simple cross
        exits = (rsi > 50)
        short_exits = (rsi < 50)

        return {
            "entries": entries,
            "exits": exits,
            "short_entries": short_entries,
            "short_exits": short_exits
        }

# We want to wrap the logic into a vectorbt Indicator so we can use vbt.Param for optimization
def run_strategy_logic(
    close,
    rsi_window=14,
    rsi_lower=30,
    rsi_upper=70,
    bb_window=20,
    bb_std=2.0
):
    # This function will be called by IndicatorFactory
    # We must treat close as a pd.Series or pd.DataFrame depending on broadcasting
    # Shared adapters handle values/index conversion
    rsi = RSI.compute(close, window=rsi_window)
    bb = BBANDS.compute(close, window=bb_window, nbdevup=bb_std, nbdevdn=bb_std)

    entries = (rsi < rsi_lower) & (close < bb['lowerband'])
    short_entries = (rsi > rsi_upper) & (close > bb['upperband'])

    # Simple exit logic for demonstration
    exits = (rsi > 50)
    short_exits = (rsi < 50)

    # Ensure they have the same columns and index as input close
    # entries = entries.vbt.align(close) # Vectorbt align helper

    return entries, exits, short_entries, short_exits

# Create the full wrapper
StrategyInd = vbt.IndicatorFactory(
    class_name='StrategyInd',
    short_name='strat',
    input_names=['close'],
    param_names=['rsi_window', 'rsi_lower', 'rsi_upper', 'bb_window', 'bb_std'],
    output_names=['entries', 'exits', 'short_entries', 'short_exits']
).from_apply_func(run_strategy_logic, keep_pd=True)
# keep_pd=True ensures we get DataFrames back with indices preserved

if __name__ == "__main__":
    # Test broadcasting
    import logging
    from src.vectorbt.data.loader import DataLoader

    logging.basicConfig(level=logging.INFO)
    loader = DataLoader(data_dir="data")
    data = loader.load_all_symbols("1h", start_date="2024-01-01", end_date="2024-01-10")
    close = data.vbt.ohlcv.close

    # Example of broadcasting across multiple rsi_windows
    strat = StrategyInd.run(
        close,
        rsi_window=[14, 21],
        rsi_lower=30,
        rsi_upper=70,
        bb_window=20,
        bb_std=2.0
    )

    print("Entries shape:", strat.entries.shape)
    print("Broadcasting test successful.")

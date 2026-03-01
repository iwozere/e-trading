import vectorbt as vbt
import pandas as pd
import numpy as np

def debug_terminal_equity():
    # 1. Setup Data: Price drops, then last bar is a tiny tick up
    # Scenario: Long trade entered in middle, never exited by signal.
    # VectorBT should close it at last index.
    close = pd.Series([100, 100, 100, 110, 110, 115, 115], name='close')
    entries = pd.Series([False, False, True, False, False, False, False])
    exits = pd.Series([False, False, False, False, False, False, False])

    pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=100)

    print("\n--- Portfolio Records ---")
    print(pf.trades.records_readable)

    print("\n--- Realized PnL Series (pf.trades.pnl.to_pd()) ---")
    pnl_series = pf.trades.pnl.to_pd()
    print(pnl_series)

    print("\n--- Final Calculations used in Visualizer ---")
    realized_pnl = pnl_series.fillna(0.0).cumsum()
    realized_equity = realized_pnl + pf.init_cash
    print("Realized Equity Curve:")
    print(realized_equity)

    print("\n--- Diagnostics ---")
    last_idx = close.index[-1]
    last_pnl = pnl_series.loc[last_idx] if last_idx in pnl_series.index else 0
    print(f"PnL recorded at last index ({last_idx}): {last_pnl}")

    if last_pnl != 0:
        print("CONFIRMED: VectorBT includes terminal exit PnL at the last index, causing the jump.")

if __name__ == "__main__":
    debug_terminal_equity()

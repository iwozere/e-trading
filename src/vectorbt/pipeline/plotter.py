import vectorbt as vbt
import pandas as pd
import argparse
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.vectorbt.data.loader import DataLoader
from src.vectorbt.pipeline.engine import StrategyEngine
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class AdvancedPlotter:
    """
    Utility for generating comprehensive visual backtest reports.
    """

    def __init__(self, strategy_path: str):
        with open(strategy_path, 'r') as f:
            self.strategy_config = json.load(f)
        self.engine = StrategyEngine(self.strategy_config)

    def plot_strategy(self, symbol: str, interval: str, params: Optional[Dict[str, Any]] = None):
        """
        Calculates signals and generates a multi-panel Plotly figure.
        """
        # 1. Load Data
        loader = DataLoader(symbols=[symbol])
        data = loader.load_all_symbols(interval)
        if data is None or data.empty:
            _logger.error(f"No data found for {symbol} at {interval}")
            return

        close = data.xs('Close', level='column', axis=1)
        ohlcv = data.xs(symbol, level='symbol', axis=1) # Get OHLCV for the specific symbol

        # 2. Run Engine
        # If no params provided, we try to use defaults or dummy values for visualization
        if params is None:
            # Simple heuristic: use min values from space if not provided
            params = {"leverage": 1.0}
            for ind_id, ind_cfg in self.strategy_config.get("indicators", {}).items():
                for p_name, p_cfg in ind_cfg.get("space", {}).items():
                    params[f"{ind_id}_{p_name}"] = p_cfg.get("min", 10)

        _logger.info(f"Running strategy engine for plotting {symbol}...")
        res = self.engine.run(close, params)

        # 3. Create Multi-Panel Figure
        # Rows: 1: Price+Signals, 2: Equity, 3+: Indicators
        indicators = list(self.strategy_config.get("indicators", {}).keys())
        # Filter indicators that are NOT price overlays
        main_indicators = [i for i in indicators if i.startswith("bb")] # Crude check for BBANDS
        sub_indicators = [i for i in indicators if not i.startswith("bb")]

        n_rows = 2 + len(sub_indicators)
        row_heights = [0.4, 0.2] + [0.4/len(sub_indicators)] * len(sub_indicators) if sub_indicators else [0.6, 0.4]

        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=(f"{symbol} Price & Signals", "Equity Curve") + tuple(ind.upper() for ind in sub_indicators),
            row_heights=row_heights
        )

        # --- Subplot 1: Price & Signals ---
        # Plot OHLC
        fig.add_trace(go.Candlestick(
            x=ohlcv.index,
            open=ohlcv['Open'],
            high=ohlcv['High'],
            low=ohlcv['Low'],
            close=ohlcv['Close'],
            name="OHLC"
        ), row=1, col=1)

        # Plot Entries/Exits (Vectorbt style)
        entries = res['entries'][symbol]
        exits = res['exits'][symbol]

        entry_idx = entries[entries].index
        exit_idx = exits[exits].index

        fig.add_trace(go.Scatter(
            x=entry_idx, y=ohlcv.loc[entry_idx, 'Low'] * 0.99,
            mode='markers', name='Long Entry',
            marker=dict(symbol='triangle-up', size=10, color='green')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=exit_idx, y=ohlcv.loc[exit_idx, 'High'] * 1.01,
            mode='markers', name='Long Exit',
            marker=dict(symbol='triangle-down', size=10, color='red')
        ), row=1, col=1)

        # Overlay BBANDS if present
        for ind_id in main_indicators:
            ind_res = self.engine.run(close, params) # Re-run or cache... for now we use internal state logic
            # Actually StrategyEngine.run returns signals, not indicator data.
            # We need to expose raw indicator results from StrategyEngine or re-calculate.
            # Let's simplify and re-calculate for visual confirmation if needed.
            pass

        # --- Subplot 2: Equity Curve ---
        # Simple cumulative PnL proxy
        pf = vbt.Portfolio.from_signals(close, entries=res['entries'], exits=res['exits'], init_cash=1000)
        equity = pf.value()[symbol] if isinstance(pf.value(), pd.DataFrame) else pf.value()

        fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Equity", line=dict(color='royalblue')), row=2, col=1)

        # --- Subplots 3+: Indicators ---
        # Note: In this version, plotter re-calculates to show values
        for i, ind_id in enumerate(sub_indicators):
            # For simplicity in this script, we'll use StrategyEngine's logic to get results
            # but StrategyEngine.run only returns signals.
            # In a production version, we would modify StrategyEngine to return 'results' too.
            # For now, let's just implement a minimal Plotly RSI view if it exists.
            if "rsi" in ind_id.lower():
                # Re-calculate RSI for plotting
                import talib
                rsi_val = talib.RSI(close[symbol].values, timeperiod=params.get(f"{ind_id}_window", 14))
                fig.add_trace(go.Scatter(x=close.index, y=rsi_val, name="RSI", line=dict(color='purple')), row=3+i, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="gray", row=3+i, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="gray", row=3+i, col=1)

        fig.update_layout(height=400 * n_rows, title_text=f"Advanced Backtest: {symbol} ({interval})", showlegend=True, xaxis_rangeslider_visible=False)

        # 4. Save
        output_dir = Path(f"results/vectorbt/{symbol}/{interval}")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "visual_backtest.html"
        fig.write_html(str(output_path))
        _logger.info(f"✅ Visual backtest saved to {output_path.absolute()}")

def main():
    parser = argparse.ArgumentParser(description="Advanced Strategy Plotter")
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--interval", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="src/vectorbt/configs/default_strategy.json")
    # Add optional parameter injection from a trial JSON
    parser.add_argument("--trial-json", type=str, help="Load parameters from a specific trial JSON")

    args = parser.parse_args()

    params = None
    if args.trial_json and os.path.exists(args.trial_json):
        with open(args.trial_json, 'r') as f:
            trial_data = json.load(f)
            params = trial_data.get('params')

    plotter = AdvancedPlotter(args.strategy)
    plotter.plot_strategy(args.symbol, args.interval, params)

if __name__ == "__main__":
    main()

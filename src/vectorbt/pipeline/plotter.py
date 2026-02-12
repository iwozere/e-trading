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

        # 2. Run Engine
        if params is None:
            params = {"leverage": 1.0}
            for ind_id, ind_cfg in self.strategy_config.get("indicators", {}).items():
                for p_name, p_cfg in ind_cfg.get("space", {}).items():
                    params[f"{ind_id}_{p_name}"] = p_cfg.get("min", 10)

        _logger.info(f"Running strategy engine for plotting {symbol}...")
        res_full = self.engine.run(data, params)
        res = res_full["signals"]

        # 3. Simulate Portfolio & Plot using NATIVE vbt methods
        pf = vbt.Portfolio.from_signals(
            close,
            entries=res['entries'],
            exits=res['exits'],
            short_entries=res['short_entries'],
            short_exits=res['short_exits'],
            fees=0.0004,
            init_cash=1000
        )

        _logger.info(f"Generating native Plotly report for {symbol}...")
        fig = pf.plot(
            subplots=[
                'cum_returns',
                'underwater',
                'orders',
                'net_exposure'
            ],
            column=symbol,
            group_by=False
        )

        fig.update_layout(
            height=1000,
            title_text=f"Vectorbt Native Report: {symbol} ({interval})",
            showlegend=True,
            xaxis_rangeslider_visible=False
        )

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

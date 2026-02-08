"""
Simulation Plotter Module

This module provides functionality to create plots from simulation results.
It reuses logic from the original run_plotter.py but adapted for batch simulation reports.
"""

import json
import os
import sys
from typing import Dict, List, Tuple
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class SimulationPlotter:
    """Class for creating plots from simulation reports"""

    def __init__(self, config_path: str = "config/optimizer/optimizer.json"):
        """
        Initialize the plotter
        """
        self.config = self._load_config(config_path)
        self.mixin_config = self._load_mixin_config()
        self.vis_settings = self.config.get("visualization_settings", {})

    def _load_config(self, config_path: str) -> Dict:
        """Load optimizer configuration for visualization settings"""
        try:
            full_path = PROJECT_ROOT / config_path
            with open(full_path, "r") as f:
                return json.load(f)
        except Exception:
            _logger.exception("Error loading config from %s", config_path)
            return {}

    def _load_mixin_config(self) -> Dict:
        """Load mixin indicators configuration"""
        try:
            full_path = PROJECT_ROOT / "config/indicators.json"
            with open(full_path, "r") as f:
                config = json.load(f)
                return config.get("plotter_config", {})
        except Exception as e:
            _logger.exception("Error loading mixin config: %s", e)
            return {"entry_mixins": {}, "exit_mixins": {}}

    def load_report_data(self, json_file: str) -> Dict:
        """Load data from JSON report file"""
        try:
            with open(json_file, "r") as f:
                return json.load(f)
        except Exception:
            _logger.exception("Error loading report file %s", json_file)
            return {}

    def load_price_data(self, data_file: str) -> pd.DataFrame:
        """Load price data from CSV file"""
        try:
            # Check if it's an absolute path already
            if os.path.isabs(data_file):
                csv_path = data_file
            else:
                csv_path = os.path.join(PROJECT_ROOT, "data", data_file)

            df = pd.read_csv(csv_path)

            # Identify timestamp column
            ts_col = None
            for col in ["timestamp", "close_time", "time", "date"]:
                if col in df.columns:
                    ts_col = col
                    break

            if ts_col:
                # Handle millisecond timestamps if necessary
                if df[ts_col].dtype in ['int64', 'float64'] and df[ts_col].max() > 1e11:
                    df["datetime"] = pd.to_datetime(df[ts_col], unit='ms', utc=True)
                else:
                    df["datetime"] = pd.to_datetime(df[ts_col], utc=True)
            elif isinstance(df.index, pd.DatetimeIndex):
                df["datetime"] = df.index
            else:
                _logger.warning("No timestamp column found in %s. Using default range.", data_file)
                df["datetime"] = pd.date_range(start="2020-01-01", periods=len(df), freq="15min")

            df = df.sort_values("datetime", ascending=True)
            df.set_index("datetime", inplace=True)

            # Standardize columns
            df = df[["open", "high", "low", "close", "volume"]]
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df.ffill(inplace=True)
            df.bfill(inplace=True)
            return df
        except Exception:
            _logger.exception("Error loading price data from %s", data_file)
            return pd.DataFrame()

    def get_indicators_for_strategy(self, report_data: Dict) -> List[str]:
        """Determine indicators based on strategy mixins"""
        indicators = set()
        strategy_config = report_data.get("strategy_config", {})

        # Check if we have nested parameters (from CustomStrategy)
        params = strategy_config.get("parameters", strategy_config)
        entry_logic = params.get("entry_logic", {})
        exit_logic = params.get("exit_logic", {})

        entry_mixin = entry_logic.get("name", "")
        exit_mixin = exit_logic.get("name", "")

        mixins = self.mixin_config.get("entry_mixins", {})
        if entry_mixin in mixins:
            indicators.update(mixins[entry_mixin].get("indicators", []))

        mixins = self.mixin_config.get("exit_mixins", {})
        if exit_mixin in mixins:
            indicators.update(mixins[exit_mixin].get("indicators", []))

        return list(indicators)

    def get_subplot_layout(self, indicators: List[str]) -> Dict[str, str]:
        """Hardcoded or dynamic subplot layout"""
        layout = {}
        for indicator in indicators:
            if indicator in ["rsi", "volume", "atr"]:
                layout[indicator] = "separate"
            else:
                layout[indicator] = "overlay"
        return layout

    def calculate_indicators(self, df: pd.DataFrame, indicators: List[str], strategy_params: Dict) -> Dict:
        """Calculation engine for indicators"""
        calculated = {}
        # Simple helper to get param values
        def get_p(name, default):
            p = strategy_params.get("parameters", strategy_params)
            for logic in ["entry_logic", "exit_logic"]:
                logic_params = p.get(logic, {}).get("params", {})
                if name in logic_params: return logic_params[name]
            return default

        for ind in indicators:
            try:
                if ind == "rsi":
                    period = get_p("rsi_period", get_p("e_rsi_period", get_p("x_rsi_period", 14)))
                    calculated["rsi"] = self._calculate_rsi(df["close"], period)
                elif ind == "bbands":
                    period = get_p("bb_period", get_p("e_bb_period", get_p("x_bb_period", 20)))
                    dev = get_p("bb_dev", get_p("e_bb_dev", get_p("x_bb_dev", 2.0)))
                    calculated["bbands"] = self._calculate_bbands(df["close"], period, dev)
                elif ind == "supertrend":
                    period = get_p("st_period", get_p("e_st_period", 10))
                    mult = get_p("st_multiplier", get_p("e_st_multiplier", 3.0))
                    calculated["supertrend"] = self._calculate_supertrend(df, period, mult)
                elif ind == "volume":
                    calculated["volume"] = df["volume"]
                elif ind == "atr":
                    period = get_p("atr_period", get_p("x_atr_period", 14))
                    calculated["atr"] = self._calculate_atr(df, period)
                elif ind == "ichimoku":
                    # Basic defaults
                    calculated["ichimoku"] = self._calculate_ichimoku(df, 9, 26, 52)
            except Exception as e:
                _logger.warning("Failed calculating %s: %s", ind, e)
        return calculated

    def _calculate_rsi(self, s, p):
        delta = s.diff()
        gain = (delta.where(delta > 0, 0)).rolling(p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(p).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_bbands(self, s, p, d):
        sma = s.rolling(p).mean()
        std = s.rolling(p).std()
        return {"upper": sma + (std * d), "middle": sma, "lower": sma - (std * d)}

    def _calculate_atr(self, df, p):
        tr = pd.concat([df['high'] - df['low'],
                       (df['high'] - df['close'].shift()).abs(),
                       (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(p).mean()

    def _calculate_supertrend(self, df, period, multiplier):
        import numpy as np
        high, low, close = df['high'], df['low'], df['close']
        atr = self._calculate_atr(df, period)
        hl2 = (high + low) / 2
        upper, lower = hl2 + (multiplier * atr), hl2 - (multiplier * atr)

        upper_arr = upper.values
        lower_arr = lower.values
        close_arr = close.values
        st = np.zeros(len(df))
        dir_arr = np.ones(len(df))

        for i in range(1, len(df)):
            if close_arr[i] > upper_arr[i-1]: dir_arr[i] = 1
            elif close_arr[i] < lower_arr[i-1]: dir_arr[i] = -1
            else: dir_arr[i] = dir_arr[i-1]

            if dir_arr[i] == 1 and lower_arr[i] < lower_arr[i-1]: lower_arr[i] = lower_arr[i-1]
            if dir_arr[i] == -1 and upper_arr[i] > upper_arr[i-1]: upper_arr[i] = upper_arr[i-1]
            st[i] = lower_arr[i] if dir_arr[i] == 1 else upper_arr[i]
        return pd.Series(st, index=df.index)

    def _calculate_ichimoku(self, df, tp, kp, sbp):
        h, l = df['high'], df['low']
        tenkan = (h.rolling(tp).max() + l.rolling(tp).min()) / 2
        kijun = (h.rolling(kp).max() + l.rolling(kp).min()) / 2
        span_a = ((tenkan + kijun) / 2).shift(kp)
        span_b = ((h.rolling(sbp).max() + l.rolling(sbp).min()) / 2).shift(kp)
        return {"tenkan": tenkan, "kijun": kijun, "span_a": span_a, "span_b": span_b}

    def calculate_equity_curve(self, trades: List[Dict], initial: float) -> pd.Series:
        if not trades: return pd.Series()
        sorted_trades = sorted(trades, key=lambda x: x.get("exit_time", ""))
        equity = [initial]
        times = [pd.Timestamp(sorted_trades[0].get("entry_time")) - pd.Timedelta(days=1)]
        curr = initial
        for t in sorted_trades:
            if t.get("net_pnl") is not None:
                curr += t["net_pnl"]
                equity.append(curr)
                times.append(pd.Timestamp(t["exit_time"]))
        return pd.Series(equity, index=times)

    def create_plot(self, df: pd.DataFrame, indicators: Dict, trades: List[Dict],
                    equity: pd.Series, output_file: str, report_data: Dict) -> None:
        """Core plotting engine using matplotlib"""
        strategy_inds = self.get_indicators_for_strategy(report_data)
        layout = self.get_subplot_layout(strategy_inds)

        n_subs = 1
        for ind in strategy_inds:
            if layout.get(ind) == "separate": n_subs += 1
        if self.vis_settings.get("show_equity_curve", True): n_subs += 1

        size = self.vis_settings.get("plot_size", [15, 10])
        fig, axes = plt.subplots(n_subs, 1, figsize=size,
                                 gridspec_kw={"height_ratios": [3] + [1] * (n_subs - 1)})
        if n_subs == 1: axes = [axes]

        title = f"Simulation: {report_data.get('bot_id', 'Unknown')}\nFinal PnL: {report_data.get('total_pnl', 0):.2f}%"
        fig.suptitle(title, fontsize=self.vis_settings.get("font_size", 12))

        # Price Plot
        ax_p = axes[0]
        ax_p.plot(df.index, df["close"], label="Close", color="black", alpha=0.8)

        for name, data in indicators.items():
            if layout.get(name) == "overlay":
                if name == "bbands":
                    ax_p.plot(df.index, data["upper"], 'r--', alpha=0.5, label="BB Upper")
                    ax_p.plot(df.index, data["lower"], 'g--', alpha=0.5, label="BB Lower")
                elif name == "supertrend":
                    ax_p.plot(df.index, data, 'm-', alpha=0.7, label="SuperTrend")

        # Trades
        self._plot_trades(ax_p, trades)
        ax_p.legend(loc='upper left')
        ax_p.grid(True, alpha=0.3)

        # Separate subplots
        idx = 1
        for name, data in indicators.items():
            if layout.get(name) == "separate" and idx < len(axes):
                ax = axes[idx]
                if name == "rsi":
                    ax.plot(df.index, data, color="purple", label="RSI")
                    ax.axhline(70, color='r', ls='--', alpha=0.5)
                    ax.axhline(30, color='g', ls='--', alpha=0.5)
                    ax.set_ylim(0, 100)
                elif name == "volume":
                    ax.bar(df.index, data, color="blue", alpha=0.6, label="Volume")
                elif name == "atr":
                    ax.plot(df.index, data, color="orange", label="ATR")
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)
                idx += 1

        # Equity
        if self.vis_settings.get("show_equity_curve", True) and not equity.empty:
            ax_e = axes[-1]
            ax_e.plot(equity.index, equity.values, color="green", lw=2, label="Equity")
            ax_e.grid(True, alpha=0.3)
            ax_e.legend(loc='upper left')

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(output_file, dpi=self.vis_settings.get("plot_dpi", 100))
        plt.close()

    def _plot_trades(self, ax, trades):
        for t in trades:
            if all(k in t for k in ["entry_time", "exit_time", "entry_price", "exit_price"]):
                ax.scatter(pd.Timestamp(t["entry_time"]), t["entry_price"], color="green", marker="^", s=100)
                ax.scatter(pd.Timestamp(t["exit_time"]), t["exit_price"], color="red", marker="v", s=100)

    def plot_report(self, report_path: str) -> str:
        """Process a report file and save plot"""
        data = self.load_report_data(report_path)
        if not data: return ""

        data_file = data.get("data_file")
        if not data_file:
            _logger.warning("No data_file in report %s", report_path)
            return ""

        df = self.load_price_data(data_file)
        if df.empty: return ""

        # Limit data for plotting if it's too large (e.g., > 10,000 bars)
        plot_limit = self.vis_settings.get("max_plot_bars", 10000)
        if len(df) > plot_limit:
            _logger.info("Data too large (%d bars), limiting plot to last %d bars", len(df), plot_limit)
            df = df.tail(plot_limit)

        inds_to_calc = self.get_indicators_for_strategy(data)
        indicators = self.calculate_indicators(df, inds_to_calc, data.get("strategy_config", {}))

        equity = self.calculate_equity_curve(data.get("trades", []), data.get("initial_balance", 10000))

        output_file = report_path.replace(".json", ".png")
        self.create_plot(df, indicators, data.get("trades", []), equity, output_file, data)
        return output_file


def plot_result_file(report_path: str):
    """Main entry point for integration"""
    plotter = SimulationPlotter()
    return plotter.plot_report(report_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_result_file(sys.argv[1])
    else:
        print("Usage: python plot_simulation.py <report_json_path>")

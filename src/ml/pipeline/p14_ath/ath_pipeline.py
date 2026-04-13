"""
ATH Pipeline Orchestrator

Implements sequential ATH and Drawdown analysis logic.
"""

from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

from src.data.data_manager import DataManager
from src.notification.logger import setup_logger
from .config import ATHPipelineConfig

_logger = setup_logger(__name__)


def _calendar_date(ts) -> date:
    """UTC calendar date for matching OHLC index to CSV event dates (tz-safe)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC")
    return t.date()


def _fmt_usd_axis(y: float, _pos: int) -> str:
    """Y-axis tick labels as whole USD amounts (no scientific offset)."""
    if not np.isfinite(y):
        return ""
    return f"${y:,.0f}"


def _simulate_ath_dd_equity(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    start_usd: float,
) -> pd.Series:
    """
    Equity curve: invest start_usd at first close, then sell full position at each
    sequential ATH (at ATH_Price on ATH_Date) and reinvest full cash at the next
    trough (at Max_Drop_Price on Max_Drop_Date). Between events, stock is marked
    to market on close; cash is flat in USD.
    """
    close = df["close"].astype(float)
    idx = df.index
    if close.empty:
        return pd.Series(dtype=float)

    cash = 0.0
    shares = start_usd / close.iloc[0]

    events: List[Tuple[pd.Timestamp, str, float]] = []
    if not results_df.empty:
        ordered = results_df.sort_values("ATH_Date")
        for _, r in ordered.iterrows():
            events.append((pd.to_datetime(r["ATH_Date"]), "sell", float(r["ATH_Price"])))
            events.append((pd.to_datetime(r["Max_Drop_Date"]), "buy", float(r["Max_Drop_Price"])))
        events.sort(key=lambda x: (_calendar_date(x[0]), 0 if x[1] == "sell" else 1))

    equity = np.empty(len(idx), dtype=float)
    ev_i = 0

    for i, dt in enumerate(idx):
        bar_d = _calendar_date(dt)
        while ev_i < len(events):
            ev_dt, ev_type, ev_px = events[ev_i]
            if bar_d < _calendar_date(ev_dt):
                break
            if ev_type == "sell" and shares > 0:
                cash = shares * ev_px
                shares = 0.0
            elif ev_type == "buy" and cash > 0 and ev_px > 0:
                shares = cash / ev_px
                cash = 0.0
            ev_i += 1

        equity[i] = shares * close.iloc[i] if shares > 0 else cash

    return pd.Series(equity, index=idx)


def _bollinger_bands(close: pd.Series, period: int = 14, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """SMA middle band and upper/lower at ± num_std × rolling std of close."""
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Wilder / RMA smoothing), range ~0–100."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out


class ATHPipeline:
    """
    Pipeline for Sequential ATH & Drawdown Analysis.
    """

    def __init__(self, config: Optional[ATHPipelineConfig] = None):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration.
        """
        self.config = config or ATHPipelineConfig.create_default()
        self.data_manager = DataManager()

        # Results directory dated for today
        self.results_dir = self.config.results_dir / datetime.now().strftime("%Y-%m-%d")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        _logger.info("ATH Pipeline initialized (results_dir: %s)", self.results_dir)

    def analyze_ticker(self, ticker: str, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Analyze a single ticker for sequential ATHs and drawdowns.

        Args:
            ticker: Stock symbol.
            df: Optional pre-fetched OHLCV data.

        Returns:
            DataFrame with analysis results.
        """
        _logger.info("Analyzing ticker: %s", ticker)

        # Fetch historical data using DataManager if not provided
        if df is None:
            # Calculate dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * self.config.lookback_years)

            try:
                df = self.data_manager.get_ohlcv(
                    symbol=ticker,
                    timeframe=self.config.interval,
                    start_date=start_date,
                    end_date=end_date
                )
            except Exception as e:
                _logger.error("Failed to fetch data for %s: %s", ticker, e)
                return pd.DataFrame()

        if df is None or df.empty:
            _logger.warning("No data found for %s", ticker)
            return pd.DataFrame()

        # Core Logic: Greedy Peak-Trough Algorithm
        # Ensure data is sorted by date
        df = df.sort_index()
        prices = df['close']
        results = []

        global_ath_price = -1
        current_ath_date = None
        current_ath_price = -1

        drawdown_min_price = float('inf')
        drawdown_min_date = None

        for bar_dt, price in prices.items():
            if price > global_ath_price:
                # We found a NEW High that breaks the previous ATH
                # Record result for previous window ONLY if we found a drawdown on a DIFFERENT day
                # AND it must be more than 1% as requested to reduce clutter
                if current_ath_date is not None and drawdown_min_date is not None:
                    drop_pct = round(((drawdown_min_price - current_ath_price) / current_ath_price) * 100, 2)
                    if drop_pct < -1.0: # Restriction 1: More than 1% drop
                        results.append({
                            'Ticker': ticker,
                            'ATH_Date': current_ath_date.strftime('%Y-%m-%d'),
                            'ATH_Price': round(float(current_ath_price), 2),
                            'Max_Drop_Date': drawdown_min_date.strftime('%Y-%m-%d'),
                            'Max_Drop_Price': round(float(drawdown_min_price), 2),
                            'Drop_Percent': drop_pct,
                            'Days_To_Drop': (drawdown_min_date - current_ath_date).days
                        })

                # Reset for the new window starting at this new ATH
                global_ath_price = price
                current_ath_date = bar_dt
                current_ath_price = price
                drawdown_min_price = float('inf')
                drawdown_min_date = None
            else:
                # Continue monitoring drawdown in the current window
                if price < drawdown_min_price:
                    drawdown_min_price = price
                    drawdown_min_date = bar_dt

        # Record the final ATH window if it exists, found on different day, and > 1% drop
        if current_ath_date is not None and drawdown_min_date is not None:
            drop_pct = round(((drawdown_min_price - current_ath_price) / current_ath_price) * 100, 2)
            if drop_pct < -1.0: # Restriction 1: More than 1% drop
                results.append({
                    'Ticker': ticker,
                    'ATH_Date': current_ath_date.strftime('%Y-%m-%d'),
                    'ATH_Price': round(float(current_ath_price), 2),
                    'Max_Drop_Date': drawdown_min_date.strftime('%Y-%m-%d'),
                    'Max_Drop_Price': round(float(drawdown_min_price), 2),
                    'Drop_Percent': drop_pct,
                    'Days_To_Drop': (drawdown_min_date - current_ath_date).days
                })

        results_df = pd.DataFrame(results)

        # Save individual ticker results to CSV
        if not results_df.empty:
            ticker_csv_path = self.results_dir / f"{ticker}_ath_analysis.csv"
            results_df.to_csv(ticker_csv_path, index=False)
            _logger.info("Saved individual results for %s to %s", ticker, ticker_csv_path)

        # Visualization
        if self.config.generate_plots and not results_df.empty:
            self._plot_results(ticker, df, results_df)

        return results_df

    def _plot_results(self, ticker: str, df: pd.DataFrame, results_df: pd.DataFrame):
        """Generate and save analysis plot."""
        try:
            bb_period, bb_std = 14, 2.0
            rsi_period = 14

            close = df["close"].astype(float)
            bb_mid, bb_upper, bb_lower = _bollinger_bands(close, period=bb_period, num_std=bb_std)
            rsi = _rsi_wilder(close, period=rsi_period)

            # Increase resolution 4x for each axis: 14*4=56, 7*4=28
            fig, (ax_price, ax_rsi, ax_eq) = plt.subplots(
                3, 1, figsize=(56, 44), sharex=True, gridspec_kw={"height_ratios": [2.4, 1.0, 1.0]}
            )

            ax_price.plot(df.index, df['close'], label='Close Price', color='royalblue', alpha=0.7)
            ax_price.plot(df.index, bb_upper, label=f'BB upper ({bb_period}, σ={bb_std})', color='steelblue', linewidth=1.0, alpha=0.85)
            ax_price.plot(df.index, bb_mid, label=f'BB mid ({bb_period})', color='gray', linewidth=1.0, linestyle='--', alpha=0.8)
            ax_price.plot(df.index, bb_lower, label=f'BB lower ({bb_period}, σ={bb_std})', color='steelblue', linewidth=1.0, alpha=0.85)
            ax_price.fill_between(df.index, bb_lower, bb_upper, color='steelblue', alpha=0.08)

            if self.config.plot_markers:
                # Convert date strings back to datetime for matching index
                ath_dates = pd.to_datetime(results_df['ATH_Date'])
                drop_dates = pd.to_datetime(results_df['Max_Drop_Date'])

                # Markers (triangles) kept at s=100 as requested (not scaled)
                ax_price.scatter(ath_dates, results_df['ATH_Price'],
                            marker='^', color='green', s=100, label='Sequential ATH', zorder=5)
                ax_price.scatter(drop_dates, results_df['Max_Drop_Price'],
                            marker='v', color='red', s=100, label='Max Drawdown Trough', zorder=5)

            ax_price.set_title(f"{ticker} Sequential ATH & Drawdown Analysis (10-Year View)", fontsize=48)
            ax_price.set_ylabel("Price (USD)", fontsize=36)
            ax_price.tick_params(axis='both', labelsize=28)

            if self.config.log_scale:
                ax_price.set_yscale('log')

            ax_price.grid(True, which="both", ls="--", alpha=0.3)
            ax_price.legend(loc='best', fontsize=22)

            ax_rsi.plot(rsi.index, rsi.values, color='purple', linewidth=1.5, label=f'RSI ({rsi_period})')
            ax_rsi.axhline(70.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
            ax_rsi.axhline(30.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
            ax_rsi.set_ylabel("RSI", fontsize=36)
            ax_rsi.set_ylim(0.0, 100.0)
            ax_rsi.tick_params(axis='both', labelsize=28)
            ax_rsi.grid(True, which="both", ls="--", alpha=0.3)
            ax_rsi.legend(loc='best', fontsize=22)

            start_usd = float(self.config.initial_equity_usd)
            eq_series = _simulate_ath_dd_equity(df, results_df, start_usd=start_usd)
            ax_eq.plot(eq_series.index, eq_series.values, color='darkorange', linewidth=2.0,
                       label=f'ATH / trough strategy (${start_usd:,.0f} start)')
            ax_eq.set_ylabel("Portfolio (USD)", fontsize=36)
            ax_eq.set_xlabel("Date", fontsize=36)
            ax_eq.tick_params(axis='both', labelsize=28)
            ax_eq.grid(True, which="both", ls="--", alpha=0.3)

            eq_clean = eq_series.replace([np.inf, -np.inf], np.nan).dropna()
            eq_pos = eq_clean[eq_clean > 0]
            ratio = float(eq_pos.max() / eq_pos.min()) if len(eq_pos) else 1.0
            use_log = (
                self.config.equity_log_scale
                and len(eq_pos) > 0
                and float(eq_pos.min()) > 0
                and ratio > 3.0
            )
            if use_log:
                ax_eq.set_yscale("log")
                ax_eq.set_ylim(
                    bottom=max(float(eq_pos.min()) * 0.88, 1.0),
                    top=float(eq_pos.max()) * 1.08,
                )
            else:
                ax_eq.set_yscale("linear")
                lo, hi = float(eq_clean.min()), float(eq_clean.max())
                span = hi - lo if hi > lo else max(abs(hi), 1.0) * 0.05
                ax_eq.set_ylim(lo - 0.06 * span, hi + 0.06 * span)
                ax_eq.yaxis.set_major_locator(MaxNLocator(nbins=8))

            ax_eq.yaxis.set_major_formatter(FuncFormatter(_fmt_usd_axis))

            ax_eq.legend(loc='best', fontsize=28)

            plt.tight_layout()

            plot_path = self.results_dir / f"{ticker}_ath_analysis.png"
            # Increased DPI for even better resolution (4x axes + high DPI)
            plt.savefig(plot_path, dpi=300)
            plt.close()
            _logger.info("Saved plot for %s to %s", ticker, plot_path)
        except Exception as e:
            _logger.error("Failed to generate plot for %s: %s", ticker, e)

    def run(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Run the pipeline for a list of tickers.

        Args:
            tickers: Optional list of tickers to override config.

        Returns:
            Aggregated results DataFrame.
        """
        tickers_to_process = tickers or self.config.tickers
        _logger.info("Starting ATH Pipeline run for %d tickers", len(tickers_to_process))

        # Calculate dates for batch fetch
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.config.lookback_years)

        # 1. Batch fetch all data at once (optimizes for deltas and Yahoo batching)
        _logger.info("Prefetching data for %d symbols in batch...", len(tickers_to_process))
        all_data = self.data_manager.get_ohlcv_batch(
            symbols=tickers_to_process,
            timeframe=self.config.interval,
            start_date=start_date,
            end_date=end_date
        )

        all_results = []

        # 2. Process each ticker using prefetched data
        for ticker in tickers_to_process:
            try:
                ticker_df = all_data.get(ticker)
                res = self.analyze_ticker(ticker, df=ticker_df)
                if not res.empty:
                    all_results.append(res)
            except Exception:
                _logger.exception("Unexpected error analyzing %s:", ticker)

        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            output_path = self.results_dir / self.config.output_csv
            final_df.to_csv(output_path, index=False)
            _logger.info("Pipeline run complete. Saved results to %s", output_path)
            return final_df
        else:
            _logger.warning("No results generated from the pipeline run.")
            return pd.DataFrame()

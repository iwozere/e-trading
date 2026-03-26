"""
ATH Pipeline Orchestrator

Implements sequential ATH and Drawdown analysis logic.
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt

from src.data.data_manager import DataManager
from src.notification.logger import setup_logger
from .config import ATHPipelineConfig

_logger = setup_logger(__name__)

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
        
        for date, price in prices.items():
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
                current_ath_date = date
                current_ath_price = price
                drawdown_min_price = float('inf')
                drawdown_min_date = None
            else:
                # Continue monitoring drawdown in the current window
                if price < drawdown_min_price:
                    drawdown_min_price = price
                    drawdown_min_date = date
                    
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
            # Increase resolution 4x for each axis: 14*4=56, 7*4=28
            plt.figure(figsize=(56, 28))
            plt.plot(df.index, df['close'], label='Close Price', color='royalblue', alpha=0.7)
            
            if self.config.plot_markers:
                # Convert date strings back to datetime for matching index
                ath_dates = pd.to_datetime(results_df['ATH_Date'])
                drop_dates = pd.to_datetime(results_df['Max_Drop_Date'])
                
                # Markers (triangles) kept at s=100 as requested (not scaled)
                plt.scatter(ath_dates, results_df['ATH_Price'], 
                            marker='^', color='green', s=100, label='Sequential ATH', zorder=5)
                plt.scatter(drop_dates, results_df['Max_Drop_Price'], 
                            marker='v', color='red', s=100, label='Max Drawdown Trough', zorder=5)
                
            plt.title(f"{ticker} Sequential ATH & Drawdown Analysis (10-Year View)", fontsize=48)
            plt.xlabel("Date", fontsize=36)
            plt.ylabel("Price (USD)", fontsize=36)
            plt.tick_params(axis='both', labelsize=28)
            
            if self.config.log_scale:
                plt.yscale('log')
                
            plt.grid(True, which="both", ls="--", alpha=0.3)
            plt.legend(loc='best', fontsize=28)
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
            except Exception as e:
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

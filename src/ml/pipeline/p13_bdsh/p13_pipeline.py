import os
import logging
import pandas as pd
from typing import List
from . import config
from .data_loader import download_data, preprocess_data
from .vix_scaling_engine import VIXScalingEngine, P13Plotter
from .models import P13Config

logger = logging.getLogger(__name__)

class P13Pipeline:
    def __init__(self, tickers: List[str], start_date: str = None, end_date: str = None):
        self.tickers = tickers
        self.start_date = start_date or config.DEFAULT_START_DATE
        self.end_date = end_date or config.DEFAULT_END_DATE
        self.vix_symbol = config.VIX_SYMBOL
        
        # Ensure results directory exists - moved from config.py side effect
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        
        # Use typed config model
        self.p13_config = P13Config.from_module(config)
        self.engine = VIXScalingEngine(self.p13_config)
        self.results_summary = []

    def run(self):
        """Orchestrates the pipeline execution."""
        logger.info(f"Starting Pipeline p13 for tickers: {self.tickers}")
        
        # 1. Ingestion
        raw_data = download_data(self.tickers, self.vix_symbol, self.start_date, self.end_date)
        if raw_data is None or raw_data.empty:
            logger.error("Pipeline failed: No data available.")
            return

        # 2. Preprocessing
        clean_data = preprocess_data(raw_data, self.tickers, self.vix_symbol)
        # Extract VIX Close for Z-score calculation
        vix_series = clean_data[self.vix_symbol]['close']
        
        # 3. Process each ticker
        all_states = self.load_state()
        
        for ticker in self.tickers:
            if ticker not in clean_data.columns.get_level_values(0):
                logger.warning(f"Ticker {ticker} not found in cleaned data. Skipping.")
                continue
                
            logger.info(f"Processing ticker: {ticker}")
            # ticker_df contains open, high, low, close
            ticker_df = clean_data[ticker]
            
            # Step-by-step processing
            # 3.1 Z-Score
            z_score = self.engine.calculate_vix_zscore(vix_series)
            
            # 3.2 Backtest (Integrated loop with signals)
            results = self.engine.run_backtest(ticker_df, z_score)
            
            # 3.3 Metrics
            metrics = self.engine.calculate_metrics(results)
            metrics["Ticker"] = ticker
            self.results_summary.append(metrics)
            
            # 3.4 Update State for Production
            last_row = results.iloc[-1]
            all_states[ticker] = {
                "current_vix_z": float(last_row["Z_Score"]),
                "active_exposure": float(last_row["Target_Exposure"]),
                "avg_entry_price": float(last_row["Avg_Entry_Price"]),
                "in_cooldown": bool(last_row["In_Cooldown"]),
                "stop_loss_price": float(last_row["Stop_Loss_Price"]),
                "last_updated": str(results.index[-1])
            }
            
            # 3.5 Saving Artifacts
            logs_path = os.path.join(config.RESULTS_DIR, f"{ticker}_trade_logs.csv")
            results.to_csv(logs_path)
            
            chart_path = os.path.join(config.RESULTS_DIR, f"{ticker}_performance.png")
            P13Plotter.plot_results(results, self.engine.markers, ticker, self.p13_config.entry_tiers, chart_path)

        self.save_state(all_states)

        # 4. Final Reporting
        if self.results_summary:
            summary_df = pd.DataFrame(self.results_summary)
            summary_path = os.path.join(config.RESULTS_DIR, "overall_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            logger.info("\n--- Final Results Summary ---")
            logger.info(f"\n{summary_df.to_string(index=False)}")
            logger.info(f"\nOverall summary saved to {summary_path}")

    def load_state(self) -> dict:
        """Loads state from JSON file."""
        import json
        if os.path.exists(config.STATE_FILE):
            try:
                with open(config.STATE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
        return {}

    def save_state(self, state: dict):
        """Saves state to JSON file."""
        import json
        try:
            with open(config.STATE_FILE, 'w') as f:
                json.dump(state, f, indent=4)
            logger.info(f"Saved production state to {config.STATE_FILE}")
        except Exception as e:
            logger.error(f"Could not save state: {e}")

        if not self.results_summary:
            logger.warning("No tickers were successfully processed.")

if __name__ == "__main__":
    # Test pipeline
    pipeline = P13Pipeline(["SPY", "TLT"])
    pipeline.run()

import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import sys

# Ensure project root is in path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager

logger = logging.getLogger(__name__)

class PipelineSummaryGenerator:
    def __init__(self, data_manager: Optional[DataManager] = None):
        """
        Initialize the generator. 
        Args:
            data_manager: Optional DataManager instance. If None, a new one will be created.
        """
        self.dm = data_manager or DataManager()

    def generate_historical_summary(self, results_base_dir: Path):
        """
        Scans all YYYY-MM-DD subfolders in results_base_dir,
        finds winners from 09_final_universe.csv,
        gets detection price from 02_fundamental_raw_data.csv,
        fetches today's price, and calculates $1000 investment growth.
        """
        if not results_base_dir.exists():
            logger.error(f"Results directory {results_base_dir} does not exist.")
            return

        all_winners = {} # ticker -> {detection_date, detection_price}

        # 1. Scan folders for detections
        folders = sorted([f for f in results_base_dir.iterdir() if f.is_dir() and self._is_date_folder(f.name)])
        
        for folder in folders:
            date_str = folder.name
            winners_file = folder / "09_final_universe.csv"
            raw_data_file = folder / "02_fundamental_raw_data.csv"
            
            if not winners_file.exists() or not raw_data_file.exists():
                continue
                
            try:
                winners_df = pd.read_csv(winners_file)
                raw_df = pd.read_csv(raw_data_file)
                
                if winners_df.empty:
                    continue
                    
                # Map ticker to price
                # Try 'current_price' first, then 'Price'
                price_col = 'current_price' if 'current_price' in raw_df.columns else 'Price'
                if price_col not in raw_df.columns:
                    logger.warning(f"Price column not found in {raw_data_file}")
                    continue
                    
                price_map = dict(zip(raw_df['ticker'], raw_df[price_col]))
                
                for ticker in winners_df['ticker']:
                    if ticker not in all_winners:
                        price = price_map.get(ticker)
                        if price and not pd.isna(price):
                            all_winners[ticker] = {
                                "detection_date": date_str,
                                "detection_price": float(price)
                            }
            except Exception as e:
                logger.warning(f"Error processing folder {date_str}: {e}")

        if not all_winners:
            logger.info("No winners found to summarize.")
            return

        # 2. Get today's prices and calculate returns
        summary_data = []
        now = datetime.now()
        today_str = now.strftime('%Y-%m-%d')
        
        print(f"Updating historical performance for {len(all_winners)} candidates...")
        
        for ticker, data in all_winners.items():
            try:
                # Fetch latest price
                # Use a 5-day window to ensure we get at least one trading day
                start_date = now - timedelta(days=5)
                ohlcv = self.dm.get_ohlcv(ticker, timeframe="1d", start_date=start_date, end_date=now)
                if ohlcv is not None and not ohlcv.empty:
                    today_price = float(ohlcv['close'].iloc[-1])
                    det_price = data['detection_price']
                    current_value = 1000 * (today_price / det_price)
                    
                    summary_data.append({
                        "ticker": ticker,
                        "detection_date": data['detection_date'],
                        "detection_price": round(det_price, 2),
                        "today_date": today_str,
                        "today_price": round(today_price, 2),
                        "current_value": round(current_value, 2)
                    })
                else:
                    logger.warning(f"Could not fetch today's price for {ticker}")
            except Exception as e:
                logger.warning(f"Error fetching today's price for {ticker}: {e}")

        if not summary_data:
            logger.info("No summary data generated.")
            return

        # 3. Save to summary.csv in the root results folder
        summary_df = pd.DataFrame(summary_data)
        # Sort by current value descending
        summary_df = summary_df.sort_values(by="current_value", ascending=False)
        
        output_file = results_base_dir / "summary.csv"
        summary_df.to_csv(output_file, index=False)
        
        logger.info(f"Historical summary saved to {output_file}")
        print(f"\n[OK] Historical summary updated: {output_file}")
        print(summary_df.head(10).to_string(index=False))

    def _is_date_folder(self, name: str) -> bool:
        try:
            datetime.strptime(name, '%Y-%m-%d')
            return True
        except ValueError:
            return False

if __name__ == "__main__":
    # Allow running as a standalone script for quick testing
    import argparse
    parser = argparse.ArgumentParser(description="Generate pipeline winners summary.")
    parser.add_argument("results_dir", type=str, help="Base results directory (e.g. results/p06_emps2)")
    args = parser.parse_args()
    
    generator = PipelineSummaryGenerator()
    generator.generate_historical_summary(Path(args.results_dir))

"""
ATH Pipeline Execution Script

Run the Sequential ATH & Drawdown Analysis pipeline from the command line.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p14_ath.config import ATHPipelineConfig
from src.ml.pipeline.p14_ath.ath_pipeline import ATHPipeline
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Sequential ATH & Drawdown Analysis Pipeline")
    parser.add_argument("--tickers", nargs="+", default="SPY,VT,ORCL,QQQ,IWM,TLT,GLD,SLV,VXX,TLT,USO,UNG,DXY,BTC-USD,ETH-USD,XAUUSD=X,XAGUSD=X,XRP-USD,LTC-USD,BCH-USD,ADA-USD,XLM-USD", help="List of tickers to analyze")
    parser.add_argument("--lookback", type=int, default=10, help="Lookback period in years (default: 10)")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")

    args = parser.parse_args()

    # Create configuration
    config = ATHPipelineConfig.create_default()
    if args.tickers:
        config.tickers = args.tickers
    config.lookback_years = args.lookback
    if args.no_plots:
        config.generate_plots = False

    # Initialize and run pipeline
    _logger.info("Initializing ATH Pipeline...")
    pipeline = ATHPipeline(config)

    results = pipeline.run()

    if not results.empty:
        print("\nAnalysis Summary:")
        print(f"Total entries: {len(results)}")
        print(f"Unique tickers: {results['Ticker'].nunique()}")
        print(f"Results saved to: {pipeline.results_dir}")
    else:
        print("\nNo results generated. Check the logs for details.")

if __name__ == "__main__":
    main()

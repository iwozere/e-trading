import argparse
import sys
import os
import re
from pathlib import Path
from typing import List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p09_arbitrage.config import ArbitrageConfig
from src.ml.pipeline.p09_arbitrage.arbitrage_pipeline import ArbitragePipeline
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

def discover_timeframes(data_dir: Path) -> List[str]:
    """
    Scans the data directory for all unique timeframes.
    Looks for patterns like symbol_TF_date.csv or symbol_TF.csv
    """
    timeframes = set()
    # Pattern to match timeframe (e.g., 1h, 15m, 1d)
    # Typically it's the second or third part of the filename: BTCUSDT_1h_...
    pattern = re.compile(r'^[A-Z0-9]+_([0-9]+[a-z]+)')
    
    files = list(data_dir.glob("*.csv"))
    
    for f in files:
        match = pattern.match(f.name)
        if match:
            timeframes.add(match.group(1))
            
    return sorted(list(timeframes))

def main():
    parser = argparse.ArgumentParser(description="P09 Arbitrage Pipeline: Statistical Pairs Trading")
    parser.add_argument("--tf", type=str, help="Specific timeframe to analyze (e.g., 1h, 15m). If omitted, runs all detected.")
    parser.add_argument("--entry", type=float, help="Z-score entry threshold")
    parser.add_argument("--exit", type=float, help="Z-score exit threshold")
    parser.add_argument("--p-value", type=float, help="Cointegration p-value threshold")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing OHLCV data")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        _logger.error(f"Data directory {data_dir} does not exist.")
        sys.exit(1)

    # Initialize config
    config = ArbitrageConfig()
    if args.entry:
        config.zscore_entry_threshold = args.entry
    if args.exit:
        config.zscore_exit_threshold = args.exit
    if args.p_value:
        config.min_cointegration_p_value = args.p_value
        
    # Determine timeframes to run
    if args.tf:
        timeframes = [args.tf]
    else:
        _logger.info("No timeframe provided. Scanning for available timeframes...")
        timeframes = discover_timeframes(data_dir)
        if not timeframes:
            _logger.warning("No timeframes discovered in data directory.")
            sys.exit(0)
        _logger.info(f"Discovered timeframes: {', '.join(timeframes)}")

    _logger.info("--- P09 Arbitrage Pipeline Initiative ---")
    _logger.info("Thresholds: Entry=%.1f, Exit=%.1f", 
                 config.zscore_entry_threshold, config.zscore_exit_threshold)
    
    # Run Pipeline for each timeframe
    pipeline = ArbitragePipeline(config, data_dir=str(data_dir))
    
    for tf in timeframes:
        _logger.info(f"Running analysis for timeframe: {tf}")
        try:
            pipeline.run(timeframe=tf)
        except Exception as e:
            _logger.error(f"Pipeline failed for timeframe {tf}: {e}")
    
    _logger.info("--- P09 Pipeline Multi-Timeframe Execution Success ---")

if __name__ == "__main__":
    main()

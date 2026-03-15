import argparse
import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p09_arbitrage.config import ArbitrageConfig
from src.ml.pipeline.p09_arbitrage.arbitrage_pipeline import ArbitragePipeline
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="P09 Arbitrage Pipeline: Statistical Pairs Trading")
    parser.add_argument("--tf", type=str, default="1h", help="Timeframe to analyze (e.g., 1h, 15m)")
    parser.add_argument("--entry", type=float, help="Z-score entry threshold")
    parser.add_argument("--exit", type=float, help="Z-score exit threshold")
    parser.add_argument("--p-value", type=float, help="Cointegration p-value threshold")
    
    args = parser.parse_args()
    
    # Initialize config
    config = ArbitrageConfig()
    if args.entry:
        config.zscore_entry_threshold = args.entry
    if args.exit:
        config.zscore_exit_threshold = args.exit
    if args.p_value:
        config.min_cointegration_p_value = args.p_value
        
    _logger.info("--- P09 Arbitrage Pipeline Initiative ---")
    _logger.info("Timeframe: %s", args.tf)
    _logger.info("Thresholds: Entry=%.1f, Exit=%.1f", 
                 config.zscore_entry_threshold, config.zscore_exit_threshold)
    
    # Run Pipeline
    pipeline = ArbitragePipeline(config)
    pipeline.run(timeframe=args.tf)
    
    _logger.info("--- P09 Pipeline Execution Success ---")

if __name__ == "__main__":
    main()

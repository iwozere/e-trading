import argparse
import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p12_order_flow.config import OrderFlowConfig
from src.ml.pipeline.p12_order_flow.order_flow_pipeline import OrderFlowPipeline
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="P12 Order Flow Pipeline: Crypto Microstructure Analysis")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)")
    parser.add_argument("--days", type=int, help="Lookback days for historical analysis")
    parser.add_argument("--tf", type=str, help="Timeframe (e.g., 1h, 15m)")
    
    args = parser.parse_args()
    
    # Initialize config
    config = OrderFlowConfig()
    
    if args.symbols:
        config.symbols = [s.strip() for s in args.symbols.split(",")]
    if args.days:
        config.lookback_days = args.days
    if args.tf:
        config.timeframe = args.tf
        
    _logger.info("--- P12 Order Flow Pipeline Initiative ---")
    _logger.info("Symbols: %s", config.symbols)
    _logger.info("Lookback: %d days", config.lookback_days)
    _logger.info("Timeframe: %s", config.timeframe)
    
    # Run Pipeline
    pipeline = OrderFlowPipeline(config)
    pipeline.run()
    
    _logger.info("--- P12 Pipeline Execution Success ---")

if __name__ == "__main__":
    main()

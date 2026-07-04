from datetime import datetime
from pathlib import Path
from typing import List

from src.ml.pipeline.p12_order_flow.config import OrderFlowConfig
from src.ml.pipeline.p12_order_flow.data_ingestion import OrderFlowDataIngestor
from src.ml.pipeline.p12_order_flow.microstructure_analyzer import MicrostructureAnalyzer
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class OrderFlowPipeline:
    """
    Orchestrates the P12 Order Flow analysis for multiple symbols.
    """

    def __init__(self, config: OrderFlowConfig | None = None):
        self.config = config or OrderFlowConfig()
        self.ingestor = OrderFlowDataIngestor(self.config)
        self.analyzer = MicrostructureAnalyzer(self.config)

    def run(self, symbols: List[str] | None = None):
        """
        Runs the full pipeline for specified symbols (or all from config).
        """
        target_symbols = symbols or self.config.symbols
        _logger.info("Starting P12 Order Flow Pipeline for %d symbols", len(target_symbols))

        # Result directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        res_dir = Path(self.config.result_root) / timestamp
        res_dir.mkdir(parents=True, exist_ok=True)

        all_results = {}

        for symbol in target_symbols:
            try:
                _logger.info("--- Processing %s ---", symbol)

                # 1. Fetch & Unify
                unified_df = self.ingestor.fetch_unified_data(symbol)
                if unified_df.empty:
                    _logger.warning("Skipping %s due to empty data.", symbol)
                    continue

                # 2. Analyze
                analyzed_df = self.analyzer.analyze(unified_df)

                # 3. Save per-symbol artifacts
                symbol_dir = res_dir / symbol
                symbol_dir.mkdir(parents=True, exist_ok=True)

                output_path = symbol_dir / "order_flow_analysis.csv"
                analyzed_df.to_csv(output_path)

                # Save just the signals for quick review
                signals_cols = ["long_flush", "short_squeeze", "crowded_longs", "crowded_shorts"]
                signals_df = analyzed_df[analyzed_df[signals_cols].any(axis=1)][signals_cols]
                if not signals_df.empty:
                    signals_df.to_csv(symbol_dir / "signals_only.csv")
                    _logger.info("Found %d signal events for %s", len(signals_df), symbol)

                all_results[symbol] = analyzed_df

            except Exception as e:
                _logger.error("Error in P12 Pipeline for %s: %s", symbol, str(e), exc_info=True)

        _logger.info("P12 Pipeline Complete. Results saved in %s", res_dir)
        return all_results

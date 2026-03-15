import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import itertools
from src.ml.pipeline.p09_arbitrage.config import ArbitrageConfig
from src.ml.pipeline.p09_arbitrage.cointegration_analyzer import CointegrationAnalyzer
from src.ml.pipeline.p09_arbitrage.spread_tracker import SpreadTracker
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class ArbitragePipeline:
    """
    Orchestrator for P09 Arbitrage Pipeline.
    Handles pair discovery, cointegration testing, signal generation, and validation.
    """
    def __init__(self, config: ArbitrageConfig, data_dir: str = "data"):
        self.config = config
        self.data_dir = Path(data_dir)
        self.analyzer = CointegrationAnalyzer()
        self.tracker = SpreadTracker(config)

    def discover_pairs(self, timeframe: str) -> List[tuple]:
        """
        Finds all available symbol pairs for a given timeframe.
        Returns a list of (symbol_a, symbol_b, path_a, path_b).
        Prioritizes files in _full directory.
        """
        search_dirs = [
            self.data_dir / "_full" / "2020-2025",
            self.data_dir
        ]
        
        found_files = []
        for d in search_dirs:
            if d.exists():
                found_files.extend(list(d.glob(f"*_{timeframe}*.csv")))
        
        symbol_map = {}
        for f in found_files:
            sym = f.name.split('_')[0]
            if sym not in symbol_map:
                symbol_map[sym] = f
        
        symbols = sorted(list(symbol_map.keys()))
        pairs = []
        for s1, s2 in itertools.combinations(symbols, 2):
            pairs.append((s1, s2, symbol_map[s1], symbol_map[s2]))
            
        return pairs

    def run(self, timeframe: str = "1h"):
        """Executes the pipeline for a specific timeframe."""
        _logger.info(f"--- P09 Arbitrage Pipeline Starting ---")
        _logger.info(f"Timeframe: {timeframe}")
        pair_info = self.discover_pairs(timeframe)
        _logger.info(f"Found {len(pair_info)} unique pairs from symbol set.")
        
        results_summary = []
        date_str = datetime.now().strftime("%Y%m%d")
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Base results directory
        base_res_dir = Path("results/p09_arbitrage")
        # Global run directory for the summary report
        run_dir = base_res_dir / "_runs" / timestamp_str
        run_dir.mkdir(parents=True, exist_ok=True)

        for sym_a, sym_b, file_a, file_b in pair_info:
            pair_name = f"{sym_a}_{sym_b}"
            _logger.info(f"Testing Pair: {pair_name}")
            
            try:
                df_a = pd.read_csv(file_a, index_col=0, parse_dates=True)
                df_b = pd.read_csv(file_b, index_col=0, parse_dates=True)
                
                if df_a.empty or df_b.empty:
                    _logger.warning(f"Empty data for {pair_name}, skipping.")
                    continue
                    
                test_res = self.analyzer.test_cointegration(df_a['close'], df_b['close'])
                p_value = test_res.get('p_value', 1.0)
                
                if p_value < self.config.min_cointegration_p_value:
                    _logger.info(f"✅ Pair Cointegrated! p-value: {p_value:.4f}, beta: {test_res['beta']:.4f}")
                    
                    spread_df = self.tracker.calculate_signals(df_a, df_b, test_res['beta'], self.config)
                    
                    # New pair-centric structure: results/p09_arbitrage/PAIR_NAME/TIMEFRAME/YYYYMMDD/
                    pair_dir = base_res_dir / pair_name / timeframe / date_str
                    pair_dir.mkdir(parents=True, exist_ok=True)
                    
                    signal_path = pair_dir / "arbitrage_signals.csv"
                    spread_df.to_csv(signal_path)
                    
                    bt_res = None
                    try:
                        from src.ml.pipeline.p09_arbitrage.run_backtrader_p09 import run_p09_backtest
                        bt_res = run_p09_backtest(signal_path, sym_a, sym_b, test_res['beta'])
                        
                        # Visualization
                        if bt_res and "equity_curve" in bt_res:
                            from src.ml.pipeline.p09_arbitrage.p09_visualizer import plot_p09_results
                            trades_path = pair_dir / "trades.csv"
                            trades_df = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
                            plot_path = pair_dir / "arbitrage_plot.png"
                            
                            plot_p09_results(
                                signal_df=spread_df,
                                trades_df=trades_df,
                                equity_curve=bt_res["equity_curve"],
                                pair_name=pair_name,
                                timeframe=timeframe,
                                output_path=plot_path
                            )
                            bt_res.pop("equity_curve") # Remove Series before logging/summary
                            
                    except Exception as e:
                        _logger.error(f"Backtest/Visualization failed for {pair_name}: {e}")

                    test_res.update({
                        "symbol_a": sym_a,
                        "symbol_b": sym_b,
                        "pair_name": pair_name,
                        "signals_count": int((spread_df['signal'] != 0).sum())
                    })
                    if bt_res:
                        test_res.update(bt_res)
                        
                    results_summary.append(test_res)
                else:
                    _logger.debug(f"Pair not cointegrated (p={p_value:.4f})")
            except Exception as e:
                _logger.error(f"Error processing pair {pair_name}: {e}")
                
        if results_summary:
            summary_df = pd.DataFrame(results_summary)
            summary_df = summary_df.fillna(0)
            # Save global summary in the timestamped run folder
            summary_df.to_csv(run_dir / "cointegration_report.csv", index=False)
            _logger.info(f"--- P09 Pipeline Execution Success ---")
            _logger.info(f"Summary saved to: {run_dir / 'cointegration_report.csv'}")
        else:
            _logger.warning("No cointegrated pairs found with current threshold.")
        
        _logger.info(f"Pipeline Complete. Found {len(results_summary)} cointegrated pairs.")

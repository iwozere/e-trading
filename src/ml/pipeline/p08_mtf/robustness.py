import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from src.ml.pipeline.p08_mtf.data_loader import P08DataLoader
from src.ml.pipeline.p08_mtf.evaluator import P08Evaluator
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class P08RobustnessEngine:
    """
    Stress-tests winning strategies on foreign tickers and timeframes.
    Helps distinguish between 'Universal Patterns' and 'Overfit Flukes'.
    """

    def __init__(self, data_root: Path = Path("data")):
        self.loader = P08DataLoader(data_root)
        self.data_root = data_root

    def test_on_target(self, ticker: str, timeframe: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Runs a blind test of params on a specific ticker/timeframe."""
        _logger.info("Testing robustness on %s %s", ticker, timeframe)

        # 1. Gather all files for this target to get a wide range
        files = list(self.data_root.glob(f"{ticker}_{timeframe}_*.csv"))
        if not files:
            return {"error": f"No data found for {ticker} {timeframe}"}

        # 2. Load and merge segments (P08 MTF style)
        dfs = []
        for f in sorted(files):
            try:
                df_mtf = self.loader.get_mtf_dataset(f)
                dfs.append(df_mtf)
            except Exception as e:
                _logger.warning("Failed to load segment %s: %s", f.name, e)

        if not dfs:
            return {"error": "No valid data segments loaded"}

        # 3. Evaluate (blind run - no training split needed ideally, but we'll use run_evaluation)
        # We can treat the whole thing as 'test' data if we want a pure forward test
        res = P08Evaluator.run_evaluation(dfs, params, timeframe=timeframe)

        if "error" in res:
            return res

        pf = res["pf"]
        return {
            "Sharpe": pf.sharpe_ratio(),
            "Trades": pf.trades.count().sum(),
            "Return [%]": pf.total_return() * 100,
            "Win Rate [%]": pf.trades.win_rate() * 100
        }

    def generate_matrix(self, winners_per_ticker_tf: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Creates a DataFrame showing how each winner performs on every other ticker/tf.
        winners_per_ticker_tf: { 'ETHUSDT_30m': {...best_params...}, ... }
        """
        results = []

        # Determine all unique ticker/tf combinations available in data/
        all_pairs = set()
        for f in self.data_root.glob("*_*_*.csv"):
            ticker, tf, _, _ = self.loader.parse_filename(f)
            if ticker:
                all_pairs.add((ticker, tf))

        for source_name, params in winners_per_ticker_tf.items():
            for target_ticker, target_tf in sorted(all_pairs):
                res = self.test_on_target(target_ticker, target_tf, params)

                results.append({
                    "Source Strategy": source_name,
                    "Target Ticker": target_ticker,
                    "Target TF": target_tf,
                    "Sharpe": res.get("Sharpe", np.nan),
                    "Trades": res.get("Trades", 0),
                    "Return [%]": res.get("Return [%]", 0),
                    "Is Winner": 1 if res.get("Sharpe", 0) > 1.0 and res.get("Trades", 0) > 10 else 0
                })

        return pd.DataFrame(results)

if __name__ == "__main__":
    # Test stub
    engine = P08RobustnessEngine()
    # Mock some params (from your ETH 30m winner ideally)
    sample_params = {
        'rsi_period': 14, 'bb_period': 20, 'pt_mult': 2.0, 'sl_mult': 1.0,
        'tpl_hours': 24.0, 'buy_prob_min': 0.5, 'sell_prob_min': 0.5
    }
    # df_matrix = engine.generate_matrix({"Sample_Winner": sample_params})
    # print(df_matrix)

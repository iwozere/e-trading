import sys
import pandas as pd
import numpy as np
import backtrader as bt
from pathlib import Path
from typing import Dict, Any, List
import optuna

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p08_mtf.pipeline import P08Pipeline
from src.ml.pipeline.p08_mtf.data_loader import P08DataLoader
from src.ml.pipeline.p08_mtf.features import P08FeatureEngine
from src.ml.pipeline.p07_combined.models import P07XGBModel
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class P08BTStrategy(bt.Strategy):
    """
    Backtrader Strategy for P08 MTF Models.
    Realism Features:
    - Next-Bar Open Execution: Signals at T close are executed at T+1 open.
    - Uses pre-calculated features for speed while maintaining point-in-time safety.
    """
    params = (
        ('model', None),
        ('feature_columns', []),
        ('thresholds', {'buy_prob_min': 0.5, 'sell_prob_min': 0.5}),
    )

    def __init__(self):
        self.model = self.p.model
        self.feature_cols = self.p.feature_columns
        self.thresholds = self.p.thresholds
        
        # Keep track of active order
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                _logger.debug(f"BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}")
            else:
                _logger.debug(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            _logger.warning(f"Order Rejected/Canceled: {order.status}")
        
        self.order = None

    def next(self):
        # We don't execute if an order is pending
        if self.order:
            return

        # 1. Extract features for THIS bar
        # We assume the data feed contains the features as 'lines' or we access them from the dataframe
        # In this implementation, we'll access them from the underlying data feed
        current_features = []
        for col in self.feature_cols:
            # Backtrader lines are accessed by names or indices
            current_features.append(getattr(self.datas[0], col)[0])
        
        # Convert to DataFrame for model (expects 2D)
        X = pd.DataFrame([current_features], columns=self.feature_cols)
        
        # 2. Predict Probabilities
        probs = self.model.predict_proba(X)[0] # returns [Sell=0, Hold=1, Buy=2]
        
        # 3. Logic: Signal at close of Bar T -> Market order executed at Open of Bar T+1
        if probs[2] > self.thresholds['buy_prob_min']:
            if not self.position:
                self.order = self.buy()
        elif probs[0] > self.thresholds['sell_prob_min']:
            if self.position:
                self.order = self.close()

def run_bt_simulation(ticker: str, timeframe: str, df_full: pd.DataFrame, model: Any, params: Dict[str, Any]):
    """Orchestrates the Backtrader run for a single winner."""
    
    # 1. Feature Engineering (Full Range)
    # Using P08FeatureEngine directly
    X = P08FeatureEngine.build_features(df_full, params)
    
    # Keep only common indices
    common_idx = df_full.index.intersection(X.index)
    df_sim = df_full.loc[common_idx].copy()
    X_sim = X.loc[common_idx]
    
    # Join features back to the main dataframe for Backtrader
    df_bt = df_sim[['open', 'high', 'low', 'close', 'volume']].copy()
    for col in X_sim.columns:
        df_bt[col] = X_sim[col]

    # 2. Initialize Cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100.0)
    
    # 95% of cash to allow some room for fees/slippage
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)
    
    # Precise Binance-like configuration
    cerebro.broker.setcommission(commission=0.001) # 0.1% fee
    cerebro.broker.set_slippage_perc(0.001) # 0.1% slippage impact
    
    # 3. Add Data
    # Feed all columns to BT
    class P08PandasData(bt.feeds.PandasData):
        lines = tuple(X_sim.columns.tolist())
        params = tuple([(col, -1) for col in X_sim.columns])

    # Note: We need to set the index mapping correctly for all custom lines
    custom_params = {}
    for i, col in enumerate(X_sim.columns):
        custom_params[col] = i + 5 # standard OHLCV are first 5

    data_feed = P08PandasData(dataname=df_bt, **custom_params)
    cerebro.adddata(data_feed)
    
    # 4. Add Strategy
    cerebro.addstrategy(P08BTStrategy, 
                        model=model, 
                        feature_columns=X_sim.columns.tolist(),
                        thresholds={'buy_prob_min': 0.5, 'sell_prob_min': 0.5})
    
    # 5. Run
    _logger.info(f"Starting Backtrader Simulation for {ticker} {timeframe}...")
    results = cerebro.run()
    strat = results[0]
    
    # 6. Extract Metrics
    final_value = cerebro.broker.getvalue()
    total_return = ((final_value / 100.0) - 1) * 100.0
    
    # Use PyFolio or manual calculation if needed, but for now let's use basics
    return {
        "final_value": final_value,
        "total_return_pct": total_return,
        "ticker": ticker,
        "timeframe": timeframe
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="P08 Backtrader Winner Simulation")
    parser.add_argument("--ticker", type=str, help="Specific ticker to run")
    parser.add_argument("--tf", type=str, help="Specific timeframe to run")
    args = parser.parse_args()

    pipeline = P08Pipeline()
    data_loader = P08DataLoader()

    # 1. Load Winners
    candidates_path = Path("results/p08_mtf/p08_robustness_candidates.csv")
    if not candidates_path.exists():
        _logger.error("No candidates file found. Run pipeline or select_candidates first.")
        return
    
    winners_df = pd.read_csv(candidates_path)
    if args.ticker and args.tf:
        winners_df = winners_df[(winners_df['ticker'] == args.ticker) & (winners_df['timeframe'] == args.tf)]

    if winners_df.empty:
        _logger.warning("No winners to simulate.")
        return

    all_bt_results = []

    for _, winner in winners_df.iterrows():
        ticker = winner['ticker']
        tf = winner['timeframe']
        
        _logger.info(f"--- Processing Backtrader Winner: {ticker} {tf} ---")
        
        # A. Find Best Params and Model
        all_studies = optuna.get_all_study_summaries(storage=pipeline.db_url)
        study_name = next((s.study_name for s in all_studies if s.study_name.startswith(f"p08_{ticker}_{tf}")), None)
        if not study_name:
            _logger.error(f"No study found for {ticker} {tf}")
            continue
        
        study = optuna.load_study(study_name=study_name, storage=pipeline.db_url)
        params = study.best_params
        
        # Load Model
        res_base = Path("results/p08_mtf") / ticker / tf
        model_paths = list(res_base.glob("**/best_model.json"))
        if not model_paths:
            _logger.error(f"No model found for {ticker} {tf}")
            continue
        model_paths.sort(key=lambda x: x.parent.name, reverse=True)
        model = P07XGBModel()
        model.load_model(str(model_paths[0]))

        # B. Load Full Data (2020-2025)
        all_files = sorted(list(Path("data").glob(f"{ticker}_{tf}_*.csv")))
        if not all_files:
            _logger.error(f"No data files for {ticker} {tf}")
            next
        
        dfs = []
        for f in all_files:
            try:
                dfs.append(data_loader.get_mtf_dataset(f))
            except Exception as e:
                _logger.warning(f"Failed to load {f.name}: {e}")
        
        if not dfs: continue
        df_full = pd.concat(dfs).sort_index()
        df_full = df_full.loc[~df_full.index.duplicated(keep='last')]
        
        # C. Run Simulation
        try:
            res = run_bt_simulation(ticker, tf, df_full, model, params)
            all_bt_results.append(res)
        except Exception as e:
            _logger.error(f"Simulation failed for {ticker} {tf}: {e}", exc_info=True)

    # 2. Save Summary
    if all_bt_results:
        summary_df = pd.DataFrame(all_bt_results)
        output_path = Path("results/p08_mtf/backtrader_winners_summary.csv")
        summary_df.to_csv(output_path, index=False)
        _logger.info(f"Successfully saved Backtrader summary to {output_path}")
        print(summary_df.to_string())

if __name__ == "__main__":
    main()

import argparse
from typing import Any
import sys
from pathlib import Path

import optuna
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p07_combined.data_loader import P07DataLoader
from src.ml.pipeline.p07_combined.evaluator import P07Evaluator
from src.ml.pipeline.p07_combined.models import P07XGBModel
from src.ml.pipeline.p07_combined.pipeline import P07Pipeline
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run_generalization(source_ticker: str, source_tf: str, merge: bool = False):
    """
    Evaluates a specific trained model against ALL available data files.
    """
    pipeline = P07Pipeline()
    data_loader = P07DataLoader()

    # 1. Find the best model and params for the source
    all_studies = optuna.get_all_study_summaries(storage=pipeline.db_url)
    study_name = None
    for s in all_studies:
        if s.study_name.startswith(f"p07_{source_ticker}_{source_tf}"):
            study_name = s.study_name
            break

    if not study_name:
        _logger.error(f"No study found for {source_ticker} {source_tf}")
        return

    study = optuna.load_study(study_name=study_name, storage=pipeline.db_url)
    best_params = study.best_params

    # Locate the model file.
    # Usually in results/p07_combined/ticker/tf/[date_range or latest]/best_model.json
    res_base = Path("results/p07_combined") / source_ticker / source_tf
    model_paths = list(res_base.glob("**/best_model.json"))
    if not model_paths:
        _logger.error(f"No best_model.json found in {res_base}")
        return

    # Sort to get the latest (by folder name or modified time)
    model_paths.sort(key=lambda x: x.parent.name, reverse=True)
    source_model_path = model_paths[0]
    _logger.info(f"Using source model: {source_model_path}")

    model = P07XGBModel()
    model.load_model(str(source_model_path))

    # 2. Find and group data files
    data_dir = Path("data")
    all_data_files = list(data_dir.glob("*_*_*.csv"))

    groups: dict[Any, Any] = {}
    if merge:
        for f in all_data_files:
            ticker, tf, _, _ = data_loader.parse_filename(f)
            if not ticker:
                continue
            key = (ticker, tf)
            if key not in groups:
                groups[key] = []
            groups[key].append(f)
        _logger.info(f"Merged mode: Grouped into {len(groups)} ticker/timeframe combinations.")
    else:
        for f in all_data_files:
            ticker, tf, start, end = data_loader.parse_filename(f)
            if not ticker:
                continue
            groups[(ticker, tf, start, end)] = [f]
        _logger.info(f"Segment mode: {len(groups)} segments to test.")

    results = []

    for key, file_list in groups.items():
        if merge:
            ticker, tf = key
            _logger.info(f"Testing on MERGED {ticker} {tf} ({len(file_list)} files)...")
        else:
            ticker, tf, start, end = key
            _logger.info(f"Testing on {ticker} {tf} ({start} to {end})...")

        try:
            # Load and enrich data
            dfs = []
            for df_path in file_list:
                dfs.append(data_loader.get_merged_dataset(df_path))

            ohlcv = pd.concat(dfs).sort_index()
            ohlcv = ohlcv.loc[~ohlcv.index.duplicated(keep="last")]
            ohlcv = pipeline.enrich_data(ohlcv)

            # Evaluate model on this segment
            eval_res = P07Evaluator.evaluate_model(model, ohlcv, best_params, timeframe=tf, init_cash=100.0)

            if "error" in eval_res:
                _logger.warning(f"Skipping {key}: {eval_res['error']}")
                continue

            metrics = eval_res["metrics"]

            # Simple conclusion logic
            conclusion = "PASS" if metrics.get("Sharpe Ratio", -1) > 0.5 and eval_res["num_trades"] > 5 else "FAIL"

            results.append(
                {
                    "ticker": ticker,
                    "timeframe": tf,
                    "data_start": eval_res["test_start"],
                    "data_end": eval_res["test_end"],
                    "num_trades": eval_res["num_trades"],
                    "sharpe": metrics.get("Sharpe Ratio"),
                    "total_return_pct": metrics.get("Total Return [%]"),
                    "win_rate": metrics.get("Win Rate [%]"),
                    "max_drawdown": metrics.get("Max Drawdown [%]"),
                    "conclusion": conclusion,
                    "source_model": f"{source_ticker}_{source_tf}",
                    "mode": "MERGED" if merge else "SEGMENT",
                    **{f"param_{k}": v for k, v in best_params.items()},
                }
            )

        except Exception as e:
            _logger.error(f"Failed to evaluate {df_path.name}: {e}")

    # 3. Save aggregated results
    if results:
        res_df = pd.DataFrame(results)
        suffix = "merged" if merge else "segments"
        output_file = Path("results/p07_combined") / f"p07_generalization_{source_ticker}_{source_tf}_{suffix}.csv"
        res_df.to_csv(output_file, index=False)
        _logger.info(f"Generalization results saved to {output_file}")
        print(f"\nSummary of results for {source_ticker}_{source_tf} model ({suffix}):")
        print(res_df[["ticker", "timeframe", "sharpe", "conclusion"]])
    else:
        _logger.warning("No results to save.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P07 Cross-Robustness (Generalization) Test")
    parser.add_argument("--ticker", type=str, default="ETHUSDT", help="Source model ticker")
    parser.add_argument("--tf", type=str, default="4h", help="Source model timeframe")
    parser.add_argument(
        "--candidates",
        default="results/p07_combined/p07_robustness_candidates.csv",
        type=str,
        help="Path to candidates CSV (ticker,timeframe)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_false",
        dest="merge",
        help="Run tests on individual file segments instead of merging",
    )
    parser.set_defaults(merge=True)

    args = parser.parse_args()

    if args.candidates:
        candidates_path = Path(args.candidates)
        if not candidates_path.exists():
            _logger.error(f"Candidates file not found: {candidates_path}")
            sys.exit(1)

        candidates_df = pd.read_csv(candidates_path)
        _logger.info(f"Loaded {len(candidates_df)} candidates from {candidates_path}")

        for _, row in candidates_df.iterrows():
            ticker = row["ticker"]
            tf = row["timeframe"]
            _logger.info(f"\n{'=' * 50}\nStarting generalization for {ticker} {tf}\n{'=' * 50}")
            run_generalization(ticker, tf, merge=args.merge)
    else:
        run_generalization(args.ticker, args.tf, merge=args.merge)

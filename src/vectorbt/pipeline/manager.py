import optuna
import os
import sqlite3
import vectorbt as vbt
from typing import Optional, List
from src.vectorbt.pipeline.objective import Objective
from src.vectorbt.data.loader import DataLoader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class StudyManager:
    """
    Manages Optuna studies with optimized SQLite storage and concurrency guardrails.
    """

    def __init__(self, db_path: str = "results/vectorbt/db/optimization_study.db"):
        self.db_path = db_path
        self.storage_url = f"sqlite:///{db_path}"
        self._ensure_db_dir()
        self._setup_sqlite_performance()

    def _ensure_db_dir(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def _setup_sqlite_performance(self):
        """
        Applies PRAGMAs for high-concurrency and performance on low-power hardware (Pi).
        """
        try:
            conn = sqlite3.connect(self.db_path)
            # Senior Architect Recommendations:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.close()
            _logger.info("SQLite performance PRAGMAs applied.")
        except Exception as e:
            _logger.error(f"Failed to tune SQLite: {e}")

    def _get_symbols_slug(self, symbols: List[str]) -> str:
        """Create a clean string slug for symbols (e.g. BTC-ETH)."""
        return "-".join(sorted([s.upper() for s in symbols]))

    def run_optimization(
        self,
        interval: str,
        n_trials: int = 100,
        n_jobs: Optional[int] = None,
        study_name: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        strategy_path: str = "src/vectorbt/configs/default_strategy.json"
    ):
        """
        Runs the Optuna study with concurrency guardrails and rolling splits.
        """
        import json
        with open(strategy_path, "r") as f:
            strategy_config = json.load(f)

        print(f"DEBUG: Loaded strategy_config indicators: {list(strategy_config.get('indicators', {}).keys())}")

        if n_jobs is None:
            # Senior Architect Recommendation: Cap parallelism for SQLite sanity
            n_jobs = min(os.cpu_count() - 1, 6)

        # 0. Handle Isolation Logic
        symbol_slug = self._get_symbols_slug(symbols) if symbols else "PORTFOLIO"
        report_dir = f"results/vectorbt/{symbol_slug}/{interval}"

        if study_name is None:
            study_name = f"optimization_{symbol_slug}_{interval}"

        # Update storage path to isolate this study inside the hierarchy
        self.db_path = os.path.join(report_dir, "optimization.db")
        self.storage_url = f"sqlite:///{self.db_path}"
        self._ensure_db_dir()
        self._setup_sqlite_performance()

        # 1. Load data
        loader = DataLoader(data_dir="data", symbols=symbols)
        data = loader.load_all_symbols(interval)
        if data is None or data.empty:
            _logger.error(f"No data for interval {interval}")
            return None

        # 2. Create rolling splits (WFO)
        # Using 3-split rolling window (e.g. 70% train, 30% test? No, Optuna runs on whole split)
        # For simplicity, let's just create 3 overlapping 1-year windows if we have enough data
        # Or use vbt.rolling_split
        try:
            # Let's use simple pandas splitting for 3 segments
            total_len = len(data)
            segment_len = total_len // 3
            data_splits = [
                data.iloc[0 : 2*segment_len],       # First 2/3
                data.iloc[segment_len : total_len], # Last 2/3
                data.iloc[segment_len//2 : total_len - segment_len//2] # Middle 2/3
            ]
            _logger.info(f"Created {len(data_splits)} rolling splits for stability scoring.")
        except Exception as e:
            _logger.error(f"Failed to create splits: {e}")
            data_splits = [data]

        # 3. Create objective
        objective = Objective(data_splits, strategy_config=strategy_config)

        # 4. Create or load study
        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage_url,
            direction="maximize",
            load_if_exists=True
        )

        # 5. Run optimization
        _logger.info(f"Starting optimization: {study_name} with {n_trials} trials...")
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

        _logger.info(f"Optimization finished. Best trial: {study.best_trial.params}")
        _logger.info(f"Best score: {study.best_value}")

        # 6. Post-optimization: Generate Report for best trial
        # Extract BTC as benchmark if available
        benchmark_close = None
        if 'BTC' in data.columns.get_level_values('symbol'):
            benchmark_close = data.xs(('BTC', 'Close'), level=('symbol', 'column'), axis=1)

        self.generate_report(study.best_trial, data, report_dir, strategy_config, benchmark_close)

        # 7. Generate Study Summary
        from src.vectorbt.pipeline.reporter import Reporter
        reporter = Reporter(output_dir=report_dir)
        reporter.generate_study_summary(study)

        return study

    def generate_report(self, trial, data, report_dir, strategy_config, benchmark_close=None):
        """
        Generates detailed report using the Reporter class.
        """
        try:
            from src.vectorbt.pipeline.engine import StrategyEngine
            from src.vectorbt.pipeline.reporter import Reporter

            close = data.xs('Close', level='column', axis=1)

            # Re-run best params via dynamic engine
            _logger.info(f"Generating report with trial params: {trial.params}")
            engine = StrategyEngine(strategy_config)
            res = engine.run(close, trial.params)

            # Simulate (vbt-core compatible Value-based sizing)
            leverage = trial.params.get('leverage', 1.0)
            init_cash = 1000.0
            n_assets = len(close.columns)
            target_value = (init_cash * leverage) / n_assets

            pf = vbt.Portfolio.from_signals(
                close,
                entries=res['entries'],
                exits=res['exits'],
                short_entries=res['short_entries'],
                short_exits=res['short_exits'],
                fees=0.0004,
                size=target_value,
                size_type='Value',
                cash_sharing=True,
                init_cash=init_cash,
                freq='1h'
            )

            # Use modular reporter
            reporter = Reporter(output_dir=report_dir)
            reporter.generate_full_report(
                pf=pf,
                trial_id=trial.number,
                params=trial.params,
                benchmark_close=benchmark_close
            )
        except Exception as e:
            import traceback
            _logger.error(f"Failed to generate report: {e}")
            _logger.error(traceback.format_exc())

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = StudyManager()
    # Run a small test optimization
    manager.run_optimization(interval="1h", n_trials=5, n_jobs=1)

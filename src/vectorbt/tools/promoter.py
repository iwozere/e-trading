"""
Strategy Promoter

Promotes best-performing strategies from Optuna SQLite storage to PostgreSQL
production database for live trading. Reads top trials from optimization studies,
validates their performance metrics, and inserts them into the trading_bots table.
"""

import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# Add project root to sys.path to allow absolute imports from 'src'
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import optuna

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class StrategyPromoter:
    """
    Promotes top-performing strategies from Optuna SQLite to PostgreSQL.
    """

    def __init__(
        self,
        sqlite_db_path: str = "src/vectorbt/db/optimization_study.db",
        min_calmar: float = 1.5,
        max_drawdown: float = 0.4,
        min_trades: int = 20,
    ):
        """
        Initialize the promoter.

        Args:
            sqlite_db_path: Path to Optuna SQLite database
            min_calmar: Minimum Calmar ratio required for promotion
            max_drawdown: Maximum drawdown allowed (as decimal, e.g., 0.4 = 40%)
            min_trades: Minimum number of trades required
        """
        self.sqlite_db_path = sqlite_db_path
        self.storage_url = f"sqlite:///{sqlite_db_path}"
        self.min_calmar = min_calmar
        self.max_drawdown = max_drawdown
        self.min_trades = min_trades

    def get_best_trials(
        self, study_name: str, top_n: int = 5
    ) -> List[optuna.trial.FrozenTrial]:
        """
        Retrieve the top N trials from a study.

        Args:
            study_name: Name of the Optuna study
            top_n: Number of top trials to retrieve

        Returns:
            List of FrozenTrial objects
        """
        try:
            study = optuna.load_study(
                study_name=study_name, storage=self.storage_url
            )

            # Get all trials sorted by value (descending)
            trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
            trials_sorted = sorted(trials, key=lambda t: t.value, reverse=True)

            return trials_sorted[:top_n]
        except Exception as e:
            _logger.error(f"Failed to load trials from study '{study_name}': {e}")
            return []

    def validate_trial(self, trial: optuna.trial.FrozenTrial) -> tuple[bool, str]:
        """
        Validate a trial against promotion criteria.

        Args:
            trial: FrozenTrial to validate

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check if trial has required user attributes
        if "total_trades" not in trial.user_attrs:
            return False, "Missing 'total_trades' attribute"

        if "avg_max_drawdown" not in trial.user_attrs:
            return False, "Missing 'avg_max_drawdown' attribute"

        # Check minimum trades
        total_trades = trial.user_attrs.get("total_trades", 0)
        if total_trades < self.min_trades:
            return False, f"Insufficient trades: {total_trades} < {self.min_trades}"

        # Check max drawdown
        avg_max_dd = trial.user_attrs.get("avg_max_drawdown", 1.0)
        if avg_max_dd > self.max_drawdown:
            return False, f"Excessive drawdown: {avg_max_dd:.2%} > {self.max_drawdown:.2%}"

        # Check if we can load the corresponding JSON report for metrics
        # We'll try to validate Calmar from the report if available
        # For now, we can compute it from the trial value if needed

        return True, "Valid"

    def load_trial_metrics(
        self, study_name: str, trial_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Load metrics from the JSON report for a trial.

        Args:
            study_name: Name of the study
            trial_id: Trial number

        Returns:
            Dictionary of metrics or None if not found
        """
        report_path = f"src/vectorbt/reports/{study_name}/trial_{trial_id}.json"

        if not os.path.exists(report_path):
            _logger.warning(f"Report not found: {report_path}")
            return None

        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
            return report.get("metrics", {})
        except Exception as e:
            _logger.error(f"Failed to load report {report_path}: {e}")
            return None

    def promote_trial(
        self,
        study_name: str,
        trial: optuna.trial.FrozenTrial,
        user_id: int = 1,
        description: Optional[str] = None,
    ) -> Optional[int]:
        """
        Promote a single trial to PostgreSQL.

        Args:
            study_name: Name of the optimization study
            trial: FrozenTrial to promote
            user_id: User ID to associate with the bot
            description: Optional description for the bot

        Returns:
            BotInstance ID if successful, None otherwise
        """
        # Validate trial
        is_valid, reason = self.validate_trial(trial)
        if not is_valid:
            _logger.warning(f"Trial {trial.number} failed validation: {reason}")
            return None

        # Load metrics from report
        metrics = self.load_trial_metrics(study_name, trial.number)
        if metrics is None:
            _logger.warning(
                f"Could not load metrics for trial {trial.number}. Proceeding with user_attrs only."
            )
            metrics = {}

        # Additional Calmar check if available
        calmar = metrics.get("calmar_ratio", 0.0)
        if calmar > 0 and calmar < self.min_calmar:
            _logger.warning(
                f"Trial {trial.number} below Calmar threshold: {calmar:.2f} < {self.min_calmar}"
            )
            return None

        # Build configuration JSON
        config = {
            "study_name": study_name,
            "trial_number": trial.number,
            "trial_value": trial.value,
            "params": trial.params,
            "metrics": metrics,
            "user_attrs": trial.user_attrs,
            "promoted_at": datetime.utcnow().isoformat(),
        }

        # Default description
        if description is None:
            leverage = trial.params.get("leverage", "N/A")
            description = (
                f"Strategy from {study_name} - Trial #{trial.number} "
                f"(Leverage: {leverage}x, Score: {trial.value:.4f})"
            )

        # Insert into PostgreSQL using TradingService
        try:
            from src.data.db.services.trading_service import trading_service

            bot_data = {
                "user_id": user_id,
                "status": "promoted",  # Custom status for promoted strategies
                "config": config,
                "description": description,
                "current_balance": None,
                "total_pnl": None,
                "error_count": 0,
                "started_at": None,
                "last_heartbeat": None,
            }

            bot_dict = trading_service.upsert_bot(bot_data)
            bot_id = bot_dict["id"]

            _logger.info(f"✅ Promoted trial {trial.number} as bot #{bot_id}")
            return bot_id

        except Exception as e:
            _logger.error(f"Failed to promote trial {trial.number}: {e}")
            return None

    def promote_top_trials(
        self,
        study_name: str,
        top_n: int = 5,
        user_id: int = 1,
    ) -> List[int]:
        """
        Promote the top N trials from a study.

        Args:
            study_name: Name of the study
            top_n: Number of trials to promote
            user_id: User ID to associate with bots

        Returns:
            List of promoted BotInstance IDs
        """
        _logger.info(f"Promoting top {top_n} trials from study '{study_name}'...")

        trials = self.get_best_trials(study_name, top_n=top_n * 2)  # Get extra for filtering
        if not trials:
            _logger.warning(f"No trials found in study '{study_name}'")
            return []

        promoted_ids = []
        for trial in trials:
            if len(promoted_ids) >= top_n:
                break

            bot_id = self.promote_trial(study_name, trial, user_id=user_id)
            if bot_id is not None:
                promoted_ids.append(bot_id)

        _logger.info(
            f"🎉 Promotion complete: {len(promoted_ids)}/{top_n} strategies promoted."
        )
        return promoted_ids

    def list_studies(self) -> List[str]:
        """
        List all available studies in the SQLite database.

        Returns:
            List of study names
        """
        try:
            summaries = optuna.get_all_study_summaries(storage=self.storage_url)
            return [s.study_name for s in summaries]
        except Exception as e:
            _logger.error(f"Failed to list studies: {e}")
            return []


def main():
    """CLI entry point for strategy promotion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Promote optimized strategies from SQLite to PostgreSQL"
    )
    parser.add_argument(
        "study_name", type=str, help="Name of the Optuna study to promote from"
    )
    parser.add_argument(
        "--top-n", type=int, default=5, help="Number of top trials to promote (default: 5)"
    )
    parser.add_argument(
        "--user-id", type=int, default=1, help="User ID to associate with bots (default: 1)"
    )
    parser.add_argument(
        "--min-calmar", type=float, default=1.5, help="Minimum Calmar ratio (default: 1.5)"
    )
    parser.add_argument(
        "--max-drawdown", type=float, default=0.4, help="Max drawdown threshold (default: 0.4)"
    )
    parser.add_argument(
        "--min-trades", type=int, default=20, help="Minimum trades required (default: 20)"
    )
    parser.add_argument(
        "--list-studies", action="store_true", help="List all available studies"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    promoter = StrategyPromoter(
        min_calmar=args.min_calmar,
        max_drawdown=args.max_drawdown,
        min_trades=args.min_trades,
    )

    if args.list_studies:
        studies = promoter.list_studies()
        print("\n📚 Available studies:")
        for study in studies:
            print(f"  - {study}")
        return

    promoted_ids = promoter.promote_top_trials(
        args.study_name, top_n=args.top_n, user_id=args.user_id
    )

    print(f"\n✅ Promoted {len(promoted_ids)} strategies:")
    for bot_id in promoted_ids:
        print(f"  - BotInstance ID: {bot_id}")


if __name__ == "__main__":
    main()

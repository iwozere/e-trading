"""
P22 job — weekly walk-forward backtest, live learned model and results email (2026-11-09).

1. `--labels-through` is worked out from the review queue (`labels.labels_complete_through`: the
   day before the oldest pending deal candidate's filing), unless given.
2. Runs the same backtest as `cli/backtest_cli.py` from `config.BACKTEST_START_YEAR` and writes
   `results/p22_biotech_ma/YYYY-MM-DD/backtest/`.
3. Trains the live learned model on every backtest row (`learned.fit_final_model`) and saves it to
   `config.LEARNED_MODEL_PATH` (plus a copy in the backtest folder); daily scoring ranks with it.
4. Emails lift@25/50/200 of the learned ranking against the baselines and the composite, with the
   change since the previous run (`outputs/backtest_report.py`).

    python src/ml/pipeline/p22_biotech_ma/jobs/run_backtest.py [--labels-through 2026-11-06] [--no-email]

Read-only on the database. Takes about an hour on the Pi.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.ml.pipeline.p22_biotech_ma.backtest.labels import labels_complete_through
from src.ml.pipeline.p22_biotech_ma.backtest.learned import fit_final_model, save_model
from src.ml.pipeline.p22_biotech_ma.backtest.metrics import DEFAULT_KS
from src.ml.pipeline.p22_biotech_ma.cli.backtest_cli import execute
from src.ml.pipeline.p22_biotech_ma.config import BACKTEST_START_YEAR, LEARNED_MODEL_PATH, RESULTS_DIR
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.ml.pipeline.p22_biotech_ma.outputs.backtest_report import HEADLINE_K, build_backtest_report, lifts, previous_summary
from src.ml.pipeline.p22_biotech_ma.outputs.email_notify import queue_email
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run(labels_through: Optional[date] = None, send_email: bool = True, user_id: Optional[int] = None,
        today: Optional[date] = None) -> Dict[str, Any]:
    """Run the weekly backtest; returns the scheduler summary (`success` False when it could not run)."""
    results_dir = setup_run_logging()
    today = today or date.today()
    with DatabaseService().uow() as uow:
        pending = uow.p22.get_pending_review_items(item_type="deal_candidate")
    through = labels_through or labels_complete_through(pending, today)
    _logger.info("Weekly backtest: labels complete through %s (%d deal candidates pending review)", through, len(pending))

    out_dir = results_dir / "backtest"
    outcome = execute(date(BACKTEST_START_YEAR, 1, 1), None, through, 1, DEFAULT_KS, out_dir)
    if outcome is None:
        return {"success": False, "error": "no landed DERA quarters", "labels_complete_through": through.isoformat()}
    result, summary = outcome

    model_info: Dict[str, Any] = {"saved": False}
    model = fit_final_model(result.score_rows, through)
    if model is not None:
        save_model(model, LEARNED_MODEL_PATH)
        shutil.copyfile(LEARNED_MODEL_PATH, out_dir / LEARNED_MODEL_PATH.name)
        model_info = {"saved": True, "train_rows": model.fitted.train_rows, "train_positives": model.fitted.train_positives,
                      "path": str(LEARNED_MODEL_PATH)}

    title, text, body = build_backtest_report(summary, previous_summary(RESULTS_DIR, results_dir), model_info)
    email_queued = queue_email(title, text, body, user_id) if send_email else False

    window = lifts(summary.get("metrics_learned_window") or [])
    learned50 = window.get("learned", {}).get(HEADLINE_K)
    naive50 = window.get("naive_informed", {}).get(HEADLINE_K)
    report = {
        "success": True,
        "labels_complete_through": through.isoformat(),
        "deal_candidates_pending": len(pending),
        "snapshots": summary.get("snapshots"),
        "learned_lift_at_50": round(learned50, 3) if learned50 is not None else None,
        "naive_lift_at_50": round(naive50, 3) if naive50 is not None else None,
        "beats_naive_at_50": learned50 is not None and naive50 is not None and learned50 > naive50,
        "model_saved": model_info["saved"],
        "email_queued": email_queued,
    }
    _logger.info("Weekly backtest complete: %s", report)
    return report


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description="P22 weekly backtest + live learned model + email")
    parser.add_argument("--labels-through", dest="labels_through", type=date.fromisoformat, default=None,
                        help="YYYY-MM-DD (default: day before the oldest pending deal candidate, else today)")
    parser.add_argument("--no-email", dest="send_email", action="store_false")
    parser.add_argument("--user-id", dest="user_id", type=int, default=None, help="Email recipient (default: scheduler system user)")
    args = parser.parse_args(argv)
    return run(args.labels_through, args.send_email, args.user_id)


if __name__ == "__main__":
    outcome = main()
    print(f"__SCHEDULER_RESULT__:{json.dumps(outcome, default=str)}")
    sys.exit(0 if outcome.get("success") else 1)

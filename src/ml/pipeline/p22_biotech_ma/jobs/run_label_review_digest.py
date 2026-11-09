"""
P22 job — weekly label-review digest email (2026-11-09).

Lists the deal candidates queued for review in the last `config.LABEL_REVIEW_DIGEST_DAYS` days,
the oldest still pending (they hold back the backtest's label cut-off) and the depth of the other
review queues (`outputs/label_review_digest.py`). Sends nothing when every queue is empty.

    python src/ml/pipeline/p22_biotech_ma/jobs/run_label_review_digest.py [--no-email]

Read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.ml.pipeline.p22_biotech_ma.config import LABEL_REVIEW_DIGEST_DAYS
from src.ml.pipeline.p22_biotech_ma.jobs.run_common import setup_run_logging
from src.ml.pipeline.p22_biotech_ma.outputs.email_notify import queue_email
from src.ml.pipeline.p22_biotech_ma.outputs.label_review_digest import build_digest, should_send
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def run(send_email: bool = True, user_id: Optional[int] = None, now: Optional[datetime] = None) -> Dict[str, Any]:
    """Build and queue the digest; returns the scheduler summary."""
    setup_run_logging()
    now = now or datetime.now(timezone.utc)
    with DatabaseService().uow() as uow:
        pending = uow.p22.get_pending_review_items()
        companies = {c["company_id"]: c for c in uow.p22.list_companies_full()}
    title, text, body, stats = build_digest(pending, companies, now, LABEL_REVIEW_DIGEST_DAYS)
    email_queued = queue_email(title, text, body, user_id) if send_email and should_send(stats) else False
    report = {"success": True, **stats, "email_queued": email_queued}
    _logger.info("Label-review digest: %s", report)
    return report


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description="P22 weekly label-review digest")
    parser.add_argument("--no-email", dest="send_email", action="store_false")
    parser.add_argument("--user-id", dest="user_id", type=int, default=None, help="Email recipient (default: scheduler system user)")
    args = parser.parse_args(argv)
    return run(args.send_email, args.user_id)


if __name__ == "__main__":
    print(f"__SCHEDULER_RESULT__:{json.dumps(main(), default=str)}")

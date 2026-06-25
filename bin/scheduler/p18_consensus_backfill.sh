#!/usr/bin/env bash
#
# P18 quarterly consensus backfill — Linux crontab wrapper.
#
# WHY THIS EXISTS
#   The P18 daily scan only *loads* the quarterly 13F consensus cache; it never
#   builds it. The full rebuild (thousands of filers × EDGAR rate-limit + FIGI
#   CUSIP mapping) takes hours and therefore MUST NOT run through the project
#   scheduler (job_schedules / APScheduler), whose `timeout_seconds` would
#   SIGKILL it mid-run. This wrapper runs it directly from Linux cron instead,
#   with no time limit, so it can complete.
#
#   It is idempotent and self-targeting: `--auto-quarter` picks the most
#   recently completed quarter and `--if-missing` makes the run a no-op once a
#   non-empty consensus cache exists. So you can schedule it generously during
#   each filing window and forget about it — it builds the cache once filings
#   are available, then quietly does nothing on subsequent runs.
#
# INSTALL (run `crontab -e` on the box that owns DATA_CACHE_DIR, e.g. the Pi):
#   # 13F-HR are due ~45 days after quarter-end (mid-Feb/May/Aug/Nov). Run daily
#   # at 04:00 UTC across the back half of each filing month; --if-missing makes
#   # every run after the first success an instant no-op.
#   0 4 16-28 2,5,8,11 * /opt/apps/e-trading/bin/scheduler/p18_consensus_backfill.sh
#
# LOG: results/p18_institutional_flow/consensus_backfill_cron.log
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Activate the project virtualenv (try the common locations).
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/venv/bin/activate"
fi

LOG_DIR="$PROJECT_ROOT/results/p18_institutional_flow"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/consensus_backfill_cron.log"
LOCK_FILE="/tmp/p18_consensus_backfill.lock"

# Prevent a new run from piling on top of a still-running heavy backfill.
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "$(date -u +%FT%TZ) another consensus backfill is still running — skipping" >> "$LOG_FILE"
    exit 0
fi

echo "$(date -u +%FT%TZ) starting idempotent consensus backfill (--auto-quarter --if-missing)" >> "$LOG_FILE"
python src/ml/pipeline/p18_institutional_flow_tracker/backfill_consensus.py \
    --auto-quarter --if-missing >> "$LOG_FILE" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) finished (exit $rc)" >> "$LOG_FILE"
exit $rc

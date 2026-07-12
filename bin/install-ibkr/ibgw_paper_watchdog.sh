#!/usr/bin/env bash
# PAPER gateway only. Detects the ib-gateway-paper stuck-restart-loop (socat /
# Xvfb bind failures after IBC's nightly in-place Gateway restart) and force-
# restarts that one container to clear it. See CHEAT_SHEET.md for background.
#
# Deliberately does not touch the LIVE gateway (manual-start-only, not
# subject to the nightly auto-restart that triggers this). Confirm the real
# live container name on the host (`docker ps -a`) before writing an
# equivalent live-side watchdog — docs currently disagree with observed
# container naming (docs say ibgw-live, paper was observed as
# ib-gateway-paper rather than the documented ibgw-paper).
#
# Deployed as ibgw-paper-watchdog.timer (runs every 5 min) — see the
# .service/.timer units in this directory.
set -euo pipefail

PAPER_CONTAINER="${IBGW_PAPER_CONTAINER:-ib-gateway-paper}"
WINDOW="${IBGW_WATCHDOG_WINDOW:-5m}"
THRESHOLD="${IBGW_WATCHDOG_THRESHOLD:-5}"

hits=$(docker logs --since "${WINDOW}" "${PAPER_CONTAINER}" 2>&1 \
    | grep -E -c "Address already in use|Fatal server error" || true)

if [ "${hits}" -ge "${THRESHOLD}" ]; then
    # Deliberately avoid the word "error" in the benign branch below — Vector's
    # log-monitoring filter (docs/monitoring-setup.md) matches any journald line
    # containing "error", so a routine "0 errors, no action" line on every 5-min
    # run was tripping a Telegram alert every ~30 min (dedup window) for nothing.
    echo "ibgw-paper-watchdog: ${hits} stuck-bind/Xvfb errors in last ${WINDOW} on ${PAPER_CONTAINER} — restarting container"
    docker restart "${PAPER_CONTAINER}"
    echo "ibgw-paper-watchdog: restart issued"
fi
# No output on the healthy path (hits < threshold) — nothing to report, and any
# stdout here matching Vector's error-keyword filter would page as a false alert.

#!/bin/sh
set -eu

#
# Script to run the Telegram Admin Panel
#
# This script is used to run the Telegram Admin Panel. It is used to run the Telegram Admin Panel in the foreground and being invoked by systemd.
#


# Resolve /opt/apps/e-trading as project root (robust, no bash needed)
SCRIPT_PATH="$0"
# Follow symlinks if any
while [ -L "$SCRIPT_PATH" ]; do
  LINK_TARGET=$(readlink "$SCRIPT_PATH")
  case "$LINK_TARGET" in
    /*) SCRIPT_PATH="$LINK_TARGET" ;;
    *)  SCRIPT_PATH="$(dirname "$SCRIPT_PATH")/$LINK_TARGET" ;;
  esac
done
SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_PATH")" && pwd -P)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# Work inside project
cd "$PROJECT_ROOT"

# Ensure log directory exists (optional if you want file logs)
mkdir -p "$PROJECT_ROOT/logs/log"

# Activate venv (portable in /bin/sh)
. "$PROJECT_ROOT/.venv/bin/activate"

# IMPORTANT: run in the foreground (no nohup, no &)
# Let systemd capture stdout/stderr (journalctl). If you also want a file, keep the redirection BELOW.
# Plain journal only:
# exec "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/src/frontend/telegram/screener/admin_panel.py"

# Journal + file (optional):
exec "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/src/frontend/telegram/screener/admin_panel.py" \
  >>"$PROJECT_ROOT/logs/log/telegram_admin_panel.out" 2>&1
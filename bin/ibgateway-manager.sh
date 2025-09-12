#!/bin/sh
# POSIX sh (dash) compatible manager for IB Gateway (PAPER & LIVE)
# - Install headless IB Gateway (paper/live)
# - Create systemd service
# - Configure UFW rule for your LAN
# - Remove installs (service + files + UFW rules)
#
# Examples:
#   sh ibgateway-manager.sh                # interactive menu
#   sh ibgateway-manager.sh --install paper --verbose
#   sh ibgateway-manager.sh --install both --assume-yes --verbose
#   sh ibgateway-manager.sh --remove live
#   sh ibgateway-manager.sh --remove both --assume-yes
#
# Notes:
# - Ports: LIVE=4001, PAPER=4002
# - Install dirs: $HOME/ibgw_live, $HOME/ibgw_paper
# - Services: ibgateway-live, ibgateway-paper

set -e

# Constants
LIVE_PORT=4001
PAPER_PORT=4002
LIVE_DIR="$HOME/ibgw_live"
PAPER_DIR="$HOME/ibgw_paper"
LIVE_CFG="$LIVE_DIR/IBGateway/config/ibgateway.live.xml"
PAPER_CFG="$PAPER_DIR/IBGateway/config/ibgateway.paper.xml"
LIVE_SERVICE="ibgateway-live"
PAPER_SERVICE="ibgateway-paper"
INSTALLER_URL="https://download.interactivebrokers.com/installers/ibgateway/latest-standalone/ibgateway-latest-standalone-linux-x64.sh"

# Flags
ASK=1
VERBOSE=0
ACTION=""       # install / remove
TARGET=""       # live / paper / both

say() { echo "$*"; }
info() { [ "$VERBOSE" -eq 1 ] && echo "[INFO] $*"; }

usage() {
  cat <<EOF
IB Gateway manager (install/remove PAPER & LIVE)

Usage:
  $0                           # interactive menu
  $0 --install [paper|live|both] [--verbose] [--assume-yes]
  $0 --remove  [paper|live|both] [--verbose] [--assume-yes]

Options:
  --install X     Install 'paper', 'live', or 'both'
  --remove  X     Remove  'paper', 'live', or 'both'
  --assume-yes    Non-interactive (auto-confirm)
  --verbose       Step-by-step logging
  -h, --help      Show this help

Defaults:
  LIVE:  port $LIVE_PORT, dir $LIVE_DIR, service $LIVE_SERVICE
  PAPER: port $PAPER_PORT, dir $PAPER_DIR, service $PAPER_SERVICE
EOF
}

confirm() {
  if [ "$ASK" -eq 0 ]; then
    info "Auto-confirm: $1"
    return 0
  fi
  printf "%s [y/N]: " "$1"
  read ans
  case "$ans" in
    y|Y|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

prompt() {
  # prompt VAR "Question: " secret_flag(0/1)
  var="$1"; q="$2"; secret="$3"
  eval cur=\${$var:-}
  if [ -n "$cur" ]; then return 0; fi
  printf "%s" "$q"
  if [ "$secret" = "1" ]; then
    stty -echo; read val; stty echo; echo
  else
    read val
  fi
  eval $var=\$val
}

ensure_pkg() {
  # ensure package exists; apt install if needed
  pkg="$1"
  if ! dpkg -s "$pkg" >/dev/null 2>&1; then
    say "Installing $pkg..."
    sudo apt update -y
    sudo apt install -y "$pkg"
  fi
}

ensure_prereqs() {
  ensure_pkg default-jre-headless
  ensure_pkg wget
}

download_installer() {
  # returns /tmp/ibgw.sh
  dest="/tmp/ibgw.sh"
  say "Downloading IB Gateway headless installer..."
  wget -O "$dest" "$INSTALLER_URL"
  chmod +x "$dest"
  echo "$dest"
}

enable_ufw_rule() {
  cidr="$1"; port="$2"
  say "UFW: allow $cidr -> port $port/tcp"
  if ! command -v ufw >/dev/null 2>&1; then
    say "Installing ufw..."
    sudo apt update -y >/dev/null 2>&1 || true
    sudo apt install -y ufw >/dev/null 2>&1 || true
  fi
  sudo ufw allow from "$cidr" to any port "$port" proto tcp || true
  if ! sudo ufw status | grep -q "Status: active"; then
    say "Enabling ufw..."
    sudo ufw --force enable
  fi
}

create_service() {
  service="$1"; workdir="$2"; cfg="$3"
  say "Creating systemd service: $service"
  sudo sh -c "cat > /etc/systemd/system/$service.service" <<EOF
[Unit]
Description=$service (IBKR Gateway headless)
After=network.target

[Service]
Type=simple
User=$(id -un)
Group=$(id -gn)
WorkingDirectory=$workdir
ExecStart=$workdir/IBGateway/ibgateway.sh -inline -cfg $cfg
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable --now "$service"
}

write_cfg() {
  mode="$1"    # live/paper
  cfg="$2"
  user_var="$3"
  pass_var="$4"
  eval u=\${$user_var}
  eval p=\${$pass_var}
  tm="$mode"
  [ "$mode" = "paper" ] && tm="paper"
  [ "$mode" = "live" ] && tm="live"
  say "Writing $mode headless config: $cfg"
  mkdir -p "$(dirname "$cfg")"
  cat > "$cfg" <<EOF
<ibgateway>
  <Settings>
    <IbLoginId>$u</IbLoginId>
    <IbPassword>$p</IbPassword>
    <TradingMode>$tm</TradingMode>
    <ApiOnly>true</ApiOnly>
    <ReadOnlyApi>false</ReadOnlyApi>
    <UseSSL>true</UseSSL>
    <AcceptNonBrokerageWarning>true</AcceptNonBrokerageWarning>
  </Settings>
</ibgateway>
EOF
  chmod 600 "$cfg"
}

install_one() {
  mode="$1"             # "paper" or "live"
  port="$2"
  dir="$3"
  cfg="$4"
  service="$5"

  say "=== Install $mode ==="
  ensure_prereqs
  inst="$(download_installer)"

  say "Installing to $dir ..."
  mkdir -p "$dir"
  "$inst" -q -dir "$dir"

  # creds + LAN
  if [ "$mode" = "paper" ]; then
    prompt PAPER_USER "IBKR PAPER username: " 0
    prompt PAPER_PASS "IBKR PAPER password: " 1
    write_cfg "paper" "$cfg" PAPER_USER PAPER_PASS
  else
    prompt LIVE_USER "IBKR LIVE username: " 0
    prompt LIVE_PASS "IBKR LIVE password: " 1
    write_cfg "live" "$cfg" LIVE_USER LIVE_PASS
  fi
  prompt LAN_CIDR "Your LAN CIDR (e.g. 192.168.1.0/24): " 0

  enable_ufw_rule "$LAN_CIDR" "$port"
  create_service "$service" "$dir" "$cfg"

  say "Done installing $mode."
  say "Status:      sudo systemctl status $service"
  say "Logs:        journalctl -u $service -f"
  say "Port:        $port/tcp"
  say
  say "If client cannot connect:"
  say "  1) Log in once via GUI with the $mode account;"
  say "  2) Enable API/socket clients, confirm port $port, add Trusted IPs ($LAN_CIDR);"
  say "  3) Restart service: sudo systemctl restart $service"
}

delete_ufw_rules_for_port() {
  port="$1"
  say "Removing UFW rules for port $port ..."
  if ! command -v ufw >/dev/null 2>&1; then
    info "ufw not installed; skipping."
    return
  fi
  while true; do
    RULES=$(sudo ufw status numbered | grep "$port" || true)
    [ -z "$RULES" ] && break
    NUM=$(echo "$RULES" | head -n1 | sed 's/^\[\([0-9]\+\)\].*/\1/')
    if [ -n "$NUM" ]; then
      echo y | sudo ufw delete "$NUM" >/dev/null
    else
      break
    fi
  done
}

stop_disable_service() {
  service="$1"
  if systemctl list-unit-files | grep -q "^$service.service"; then
    say "Stopping and disabling $service ..."
    sudo systemctl stop "$service" || true
    sudo systemctl disable "$service" || true
    sudo rm -f "/etc/systemd/system/$service.service"
  else
    say "Service $service not found; skipping."
  fi
}

remove_dir() {
  d="$1"
  if [ -d "$d" ]; then
    say "Deleting $d ..."
    rm -rf "$d"
  fi
}

remove_one() {
  mode="$1"   # paper/live
  if [ "$mode" = "paper" ]; then
    say "=== Remove paper ==="
    confirm "Remove service $PAPER_SERVICE?" && stop_disable_service "$PAPER_SERVICE"
    confirm "Delete directory $PAPER_DIR?" && remove_dir "$PAPER_DIR"
    confirm "Delete UFW rules for port $PAPER_PORT?" && delete_ufw_rules_for_port "$PAPER_PORT"
  else
    say "=== Remove live ==="
    confirm "Remove service $LIVE_SERVICE?" && stop_disable_service "$LIVE_SERVICE"
    confirm "Delete directory $LIVE_DIR?" && remove_dir "$LIVE_DIR"
    confirm "Delete UFW rules for port $LIVE_PORT?" && delete_ufw_rules_for_port "$LIVE_PORT"
  fi
  sudo systemctl daemon-reload
  sudo systemctl reset-failed || true
  say "Done removing $mode."
}

interactive_menu() {
  say "=== IB Gateway Manager ==="
  say "1) Install PAPER"
  say "2) Install LIVE"
  say "3) Install BOTH (PAPER + LIVE)"
  say "4) Remove PAPER"
  say "5) Remove LIVE"
  say "6) Remove BOTH"
  say "7) Quit"
  printf "Choose [1-7]: "
  read ch
  case "$ch" in
    1) install_one "paper" $PAPER_PORT "$PAPER_DIR" "$PAPER_CFG" "$PAPER_SERVICE" ;;
    2) install_one "live"  $LIVE_PORT  "$LIVE_DIR"  "$LIVE_CFG"  "$LIVE_SERVICE" ;;
    3) install_one "paper" $PAPER_PORT "$PAPER_DIR" "$PAPER_CFG" "$PAPER_SERVICE"
       install_one "live"  $LIVE_PORT  "$LIVE_DIR"  "$LIVE_CFG"  "$LIVE_SERVICE" ;;
    4) remove_one "paper" ;;
    5) remove_one "live" ;;
    6) remove_one "paper"; remove_one "live" ;;
    *) say "Bye."; exit 0 ;;
  esac
}

# Parse args
while [ $# -gt 0 ]; do
  case "$1" in
    --install) ACTION="install"; shift; TARGET="${1:-}"; shift || true ;;
    --remove)  ACTION="remove";  shift; TARGET="${1:-}"; shift || true ;;
    --assume-yes) ASK=0; shift ;;
    --verbose) VERBOSE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) say "Unknown option: $1"; usage; exit 1 ;;
  esac
done

# Dispatch
if [ -z "$ACTION" ]; then
  interactive_menu
  exit 0
fi

# Validate target
case "$TARGET" in
  live|paper|both) ;;
  *) say "Error: specify target: live | paper | both"; usage; exit 1 ;;
esac

if [ "$ACTION" = "install" ]; then
  [ "$TARGET" = "paper" ] && install_one "paper" $PAPER_PORT "$PAPER_DIR" "$PAPER_CFG" "$PAPER_SERVICE"
  [ "$TARGET" = "live" ]  && install_one "live"  $LIVE_PORT  "$LIVE_DIR"  "$LIVE_CFG"  "$LIVE_SERVICE"
  if [ "$TARGET" = "both" ]; then
    install_one "paper" $PAPER_PORT "$PAPER_DIR" "$PAPER_CFG" "$PAPER_SERVICE"
    install_one "live"  $LIVE_PORT  "$LIVE_DIR"  "$LIVE_CFG"  "$LIVE_SERVICE"
  fi
elif [ "$ACTION" = "remove" ]; then
  [ "$TARGET" = "paper" ] && remove_one "paper"
  [ "$TARGET" = "live" ]  && remove_one "live"
  [ "$TARGET" = "both" ]  && { remove_one "paper"; remove_one "live"; }
else
  say "Unknown action: $ACTION"
  usage
  exit 1
fi

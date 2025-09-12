#!/bin/sh
# IB Gateway manager (POSIX sh). Auto-detects CPU:
# - ARM64 → Docker (arm64 image)
# - x86_64 → Native installer
#
# Usage:
#   sh ibgateway-manager.sh                    # interactive menu
#   sh ibgateway-manager.sh --install paper    # install only PAPER
#   sh ibgateway-manager.sh --install both --verbose
#   sh ibgateway-manager.sh --remove live --assume-yes
#
# Options:
#   --install [paper|live|both]
#   --remove  [paper|live|both]
#   --assume-yes              (no prompts)
#   --verbose                 (extra logs)
#   --with-systemd            (Docker mode: add systemd unit to auto-start)
#   -h | --help

set -e

# Shared defaults
LIVE_PORT=4001
PAPER_PORT=4002

# Native (x86_64) paths
LIVE_DIR="$HOME/ibgw_live"
PAPER_DIR="$HOME/ibgw_paper"
LIVE_CFG="$LIVE_DIR/IBGateway/config/ibgateway.live.xml"
PAPER_CFG="$PAPER_DIR/IBGateway/config/ibgateway.paper.xml"
LIVE_SERVICE="ibgateway-live"
PAPER_SERVICE="ibgateway-paper"

# Docker (ARM64) paths
DC_ROOT="$HOME/ibkr"
DC_FILE="$DC_ROOT/docker-compose.yml"
ENV_FILE="$DC_ROOT/.env"
DC_UNIT="ibgateway-docker.service"
DC_IMAGE="ghcr.io/gnzsnz/ib-gateway:10.40.1a"  # arm64-ready image

ASK=1
VERBOSE=0
WITH_SYSTEMD=0
ACTION=""
TARGET=""
ARCH="$(uname -m)"

say() { echo "$*"; }
info() { [ "$VERBOSE" -eq 1 ] && echo "[INFO] $*"; }

usage() {
  cat <<EOF
IB Gateway manager (auto ARM-docker / x86-native)

Usage:
  $0
  $0 --install [paper|live|both] [--verbose] [--assume-yes] [--with-systemd]
  $0 --remove  [paper|live|both] [--verbose] [--assume-yes]

Ports:
  LIVE  = $LIVE_PORT
  PAPER = $PAPER_PORT

ARM64 (Pi): Docker compose in $DC_ROOT, systemd unit $DC_UNIT
x86_64: native install to $LIVE_DIR / $PAPER_DIR with systemd services

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
  pkg="$1"
  if ! dpkg -s "$pkg" >/dev/null 2>&1; then
    say "Installing $pkg..."
    sudo apt update -y
    sudo apt install -y "$pkg"
  fi
}

# -------------------------
# x86_64 (native) functions
# -------------------------

find_installer_url() {
  for u in \
    "https://download.interactivebrokers.com/installers/ibgateway/latest-standalone/ibgateway-latest-standalone-linux-x64.sh" \
    "https://download2.interactivebrokers.com/installers/ibgateway/latest-standalone/ibgateway-latest-standalone-linux-x64.sh" \
    "https://download.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh" \
    "https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh"
  do
    if wget --spider -q "$u"; then echo "$u"; return 0; fi
  done
  return 1
}

download_installer() {
  dest="/tmp/ibgw.sh"
  say "Locating IB Gateway installer..."
  url="$(find_installer_url)" || { say "Could not find a working installer URL."; exit 1; }
  say "Downloading: $url"
  wget -O "$dest" "$url"
  chmod +x "$dest"
  echo "$dest"
}

write_native_cfg() {
  mode="$1"; cfg="$2"; user_var="$3"; pass_var="$4"
  eval u=\${$user_var}
  eval p=\${$pass_var}
  say "Writing $mode headless config: $cfg"
  mkdir -p "$(dirname "$cfg")"
  cat > "$cfg" <<EOF
<ibgateway>
  <Settings>
    <IbLoginId>$u</IbLoginId>
    <IbPassword>$p</IbPassword>
    <TradingMode>$mode</TradingMode>
    <ApiOnly>true</ApiOnly>
    <ReadOnlyApi>false</ReadOnlyApi>
    <UseSSL>true</UseSSL>
    <AcceptNonBrokerageWarning>true</AcceptNonBrokerageWarning>
  </Settings>
</ibgateway>
EOF
  chmod 600 "$cfg"
}

create_native_service() {
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

native_install_one() {
  mode="$1"  # paper/live
  port="$2"
  dir="$3"
  cfg="$4"
  service="$5"

  say "=== Native install: $mode ==="
  ensure_pkg default-jre-headless
  ensure_pkg wget

  inst="$(download_installer)"
  say "Installing to $dir ..."
  mkdir -p "$dir"
  "$inst" -q -dir "$dir"

  if [ "$mode" = "paper" ]; then
    prompt PAPER_USER "IBKR PAPER username: " 0
    prompt PAPER_PASS "IBKR PAPER password: " 1
    write_native_cfg "paper" "$cfg" PAPER_USER PAPER_PASS
  else
    prompt LIVE_USER "IBKR LIVE username: " 0
    prompt LIVE_PASS "IBKR LIVE password: " 1
    write_native_cfg "live" "$cfg" LIVE_USER LIVE_PASS
  fi
  prompt LAN_CIDR "Your LAN CIDR (e.g. 192.168.1.0/24): " 0

  enable_ufw_rule "$LAN_CIDR" "$port"
  create_native_service "$service" "$dir" "$cfg"

  say "Done installing $mode."
  say "Status: sudo systemctl status $service"
  say "Logs:   journalctl -u $service -f"
}

native_delete_ufw_rules_for_port() {
  port="$1"
  say "Removing UFW rules for port $port ..."
  if ! command -v ufw >/dev/null 2>&1; then
    return
  fi
  while true; do
    RULES=$(sudo ufw status numbered | grep "$port" || true)
    [ -z "$RULES" ] && break
    NUM=$(echo "$RULES" | head -n1 | sed 's/^\[\([0-9]\+\)\].*/\1/')
    [ -n "$NUM" ] || break
    echo y | sudo ufw delete "$NUM" >/dev/null
  done
}

native_stop_disable_service() {
  service="$1"
  if systemctl list-unit-files | grep -q "^$service.service"; then
    say "Stopping/disabling $service ..."
    sudo systemctl stop "$service" || true
    sudo systemctl disable "$service" || true
    sudo rm -f "/etc/systemd/system/$service.service"
  else
    say "Service $service not found; skipping."
  fi
}

native_remove_dir() {
  d="$1"
  if [ -d "$d" ]; then
    say "Deleting $d ..."
    rm -rf "$d"
  fi
}

native_remove_one() {
  mode="$1"
  if [ "$mode" = "paper" ]; then
    confirm "Remove service $PAPER_SERVICE?" && native_stop_disable_service "$PAPER_SERVICE"
    confirm "Delete directory $PAPER_DIR?" && native_remove_dir "$PAPER_DIR"
    confirm "Delete UFW rules for $PAPER_PORT?" && native_delete_ufw_rules_for_port "$PAPER_PORT"
  else
    confirm "Remove service $LIVE_SERVICE?" && native_stop_disable_service "$LIVE_SERVICE"
    confirm "Delete directory $LIVE_DIR?" && native_remove_dir "$LIVE_DIR"
    confirm "Delete UFW rules for $LIVE_PORT?" && native_delete_ufw_rules_for_port "$LIVE_PORT"
  fi
  sudo systemctl daemon-reload
  sudo systemctl reset-failed || true
  say "Removed $mode."
}

# -------------------------
# ARM64 (Docker) functions
# -------------------------

docker_ensure() {
  if ! command -v docker >/dev/null 2>&1; then
    say "Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y docker.io docker-compose-plugin
    sudo usermod -aG docker "$USER" || true
  fi
}

docker_write_env() {
  mkdir -p "$DC_ROOT"
  [ -f "$ENV_FILE" ] || touch "$ENV_FILE"
  # add keys if missing
  grep -q '^READONLY_API=' "$ENV_FILE" 2>/dev/null || echo "READONLY_API=false" >> "$ENV_FILE"
  grep -q '^PAPER_USER='   "$ENV_FILE" 2>/dev/null || echo "PAPER_USER=" >> "$ENV_FILE"
  grep -q '^PAPER_PASS='   "$ENV_FILE" 2>/dev/null || echo "PAPER_PASS=" >> "$ENV_FILE"
  grep -q '^LIVE_USER='    "$ENV_FILE" 2>/dev/null || echo "LIVE_USER=" >> "$ENV_FILE"
  grep -q '^LIVE_PASS='    "$ENV_FILE" 2>/dev/null || echo "LIVE_PASS=" >> "$ENV_FILE"
}

docker_set_env_kv() {
  key="$1"; val="$2"
  if grep -q "^$key=" "$ENV_FILE"; then
    # escape slashes
    esc="$(printf '%s\n' "$val" | sed 's/[\/&]/\\&/g')"
    sed -i "s|^$key=.*|$key=$esc|" "$ENV_FILE"
  else
    echo "$key=$val" >> "$ENV_FILE"
  fi
}

docker_write_compose() {
  say "Writing $DC_FILE"
  mkdir -p "$DC_ROOT"
  cat > "$DC_FILE" <<EOF
services:
  ibgw-paper:
    image: $DC_IMAGE
    container_name: ibgw-paper
    restart: unless-stopped
    environment:
      - IBG_MODE=paper
      - IBG_USER=\${PAPER_USER}
      - IBG_PASSWORD=\${PAPER_PASS}
      - IBC_TRADING_MODE=paper
      - IBC_READONLY_API=\${READONLY_API}
    ports: ["$PAPER_PORT:$PAPER_PORT"]

  ibgw-live:
    image: $DC_IMAGE
    container_name: ibgw-live
    restart: unless-stopped
    environment:
      - IBG_MODE=live
      - IBG_USER=\${LIVE_USER}
      - IBG_PASSWORD=\${LIVE_PASS}
      - IBC_TRADING_MODE=live
      - IBC_READONLY_API=\${READONLY_API}
    ports: ["$LIVE_PORT:$LIVE_PORT"]
EOF
}

docker_up_targets() {
  cd "$DC_ROOT"
  if [ "$TARGET" = "paper" ]; then
    docker compose up -d ibgw-paper
  elif [ "$TARGET" = "live" ]; then
    docker compose up -d ibgw-live
  else
    docker compose up -d
  fi
}

docker_down_targets() {
  cd "$DC_ROOT"
  if [ "$TARGET" = "paper" ]; then
    docker compose stop ibgw-paper || true
    docker compose rm -f ibgw-paper || true
  elif [ "$TARGET" = "live" ]; then
    docker compose stop ibgw-live || true
    docker compose rm -f ibgw-live || true
  else
    docker compose down || true
  fi
}

docker_create_systemd() {
  say "Creating systemd unit: $DC_UNIT"
  sudo tee "/etc/systemd/system/$DC_UNIT" >/dev/null <<EOF
[Unit]
Description=IB Gateway (paper+live) via Docker Compose
After=docker.service network.target
Requires=docker.service

[Service]
Type=oneshot
WorkingDirectory=$DC_ROOT
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
RemainAfterExit=yes
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable --now "$DC_UNIT"
}

docker_remove_systemd_if_both() {
  # Remove unit only if removing BOTH (otherwise keep it)
  if [ "$TARGET" = "both" ] && [ -f "/etc/systemd/system/$DC_UNIT" ]; then
    say "Disabling/removing $DC_UNIT"
    sudo systemctl disable --now "$DC_UNIT" || true
    sudo rm -f "/etc/systemd/system/$DC_UNIT"
    sudo systemctl daemon-reload
  fi
}

docker_install_flow() {
  say "=== Docker install on ARM64 ==="
  docker_ensure
  mkdir -p "$DC_ROOT"
  docker_write_env

  if [ "$TARGET" = "paper" ]; then
    prompt PAPER_USER "IBKR PAPER username: " 0
    prompt PAPER_PASS "IBKR PAPER password: " 1
    docker_set_env_kv "PAPER_USER" "$PAPER_USER"
    docker_set_env_kv "PAPER_PASS" "$PAPER_PASS"
  elif [ "$TARGET" = "live" ]; then
    prompt LIVE_USER "IBKR LIVE username: " 0
    prompt LIVE_PASS "IBKR LIVE password: " 1
    docker_set_env_kv "LIVE_USER" "$LIVE_USER"
    docker_set_env_kv "LIVE_PASS" "$LIVE_PASS"
  else
    prompt PAPER_USER "IBKR PAPER username: " 0
    prompt PAPER_PASS "IBKR PAPER password: " 1
    prompt LIVE_USER  "IBKR LIVE username: " 0
    prompt LIVE_PASS  "IBKR LIVE password: " 1
    docker_set_env_kv "PAPER_USER" "$PAPER_USER"
    docker_set_env_kv "PAPER_PASS" "$PAPER_PASS"
    docker_set_env_kv "LIVE_USER"  "$LIVE_USER"
    docker_set_env_kv "LIVE_PASS"  "$LIVE_PASS"
  fi

  [ -f "$DC_FILE" ] || docker_write_compose

  docker_up_targets

  say "Docker services up."
  say "Paper: port $PAPER_PORT  Live: port $LIVE_PORT"
  say "Compose dir: $DC_ROOT"
  say "Logs: docker compose logs -f ibgw-paper|ibgw-live"

  if [ "$WITH_SYSTEMD" -eq 1 ]; then
    docker_create_systemd
  else
    say "(Tip) To auto-start on boot: re-run with --with-systemd"
  fi
}

docker_remove_flow() {
  say "=== Docker remove on ARM64 ==="
  if [ ! -d "$DC_ROOT" ]; then
    say "$DC_ROOT not found; nothing to remove."
  else
    docker_ensure
    docker_down_targets
    docker_remove_systemd_if_both
  fi
  say "Done."
}

# -------------------------
# Interactive menu
# -------------------------

interactive_menu() {
  say "=== IB Gateway Manager (ARCH: $ARCH) ==="
  say "1) Install PAPER"
  say "2) Install LIVE"
  say "3) Install BOTH"
  say "4) Remove PAPER"
  say "5) Remove LIVE"
  say "6) Remove BOTH"
  say "7) Quit"
  printf "Choose [1-7]: "
  read ch
  case "$ch" in
    1) ACTION="install"; TARGET="paper" ;;
    2) ACTION="install"; TARGET="live" ;;
    3) ACTION="install"; TARGET="both" ;;
    4) ACTION="remove";  TARGET="paper" ;;
    5) ACTION="remove";  TARGET="live" ;;
    6) ACTION="remove";  TARGET="both" ;;
    *) say "Bye."; exit 0 ;;
  esac
}

# -------------------------
# Parse args
# -------------------------

while [ $# -gt 0 ]; do
  case "$1" in
    --install) ACTION="install"; shift; TARGET="${1:-}"; shift || true ;;
    --remove)  ACTION="remove";  shift; TARGET="${1:-}"; shift || true ;;
    --assume-yes) ASK=0; shift ;;
    --verbose) VERBOSE=1; shift ;;
    --with-systemd) WITH_SYSTEMD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) say "Unknown option: $1"; usage; exit 1 ;;
  esac
done

# Dispatch
if [ -z "$ACTION" ]; then
  interactive_menu
fi

case "$TARGET" in
  live|paper|both) ;;
  *) say "Error: target must be one of: paper | live | both"; usage; exit 1 ;;
esac

# Route by architecture
case "$ARCH" in
  aarch64|arm64)
    if [ "$ACTION" = "install" ]; then
      docker_install_flow
    else
      docker_remove_flow
    fi
    ;;
  x86_64|amd64)
    if [ "$ACTION" = "install" ]; then
      [ "$TARGET" = "paper" ] && native_install_one "paper" $PAPER_PORT "$PAPER_DIR" "$PAPER_CFG" "$PAPER_SERVICE"
      [ "$TARGET" = "live" ]  && native_install_one "live"  $LIVE_PORT  "$LIVE_DIR"  "$LIVE_CFG"  "$LIVE_SERVICE"
      if [ "$TARGET" = "both" ]; then
        native_install_one "paper" $PAPER_PORT "$PAPER_DIR" "$PAPER_CFG" "$PAPER_SERVICE"
        native_install_one "live"  $LIVE_PORT  "$LIVE_DIR"  "$LIVE_CFG"  "$LIVE_SERVICE"
      fi
    else
      [ "$TARGET" = "paper" ] && native_remove_one "paper"
      [ "$TARGET" = "live" ]  && native_remove_one "live"
      [ "$TARGET" = "both" ]  && { native_remove_one "paper"; native_remove_one "live"; }
    fi
    ;;
  *)
    say "Unsupported arch: $ARCH. Exiting."
    exit 1
    ;;
esac

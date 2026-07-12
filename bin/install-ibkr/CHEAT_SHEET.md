# 📑 IBKR Gateway on Raspberry Pi – Cheat Sheet

## 🖥 System Overview

* **Software:** **Interactive Brokers Gateway** (not TWS)
* **Host:** Raspberry Pi (raspberrypi)
* **Paper container:** `ib-gateway-paper` — `~/ibkr/ibgateway-paper/docker-compose.yml`, `network_mode: host`, `restart: always`
* **Live container:** `ib-gateway-live` — `~/ibkr/ibgateway-live/docker-compose.yml`, bridge networking with explicit port mapping, `restart: always`
* **No systemd wrapper unit exists.** There is no `ibgateway-docker.service` — auto-start-on-boot comes purely from each container's own `restart: always` policy plus the Docker daemon (`docker.service`) starting on boot. Use `docker`/`docker compose` commands directly, not `systemctl`.
* **Both gateways currently run continuously** (both have `restart: always`). Live's `READ_ONLY_API=yes` blocks order placement as the safety net — there is no "live is off by default" behavior in the current config.

> [!WARNING]
> An earlier, now-abandoned setup at the top-level `~/ibkr/docker-compose.yml` defined a container named `ibgw-paper` (pinned image `ghcr.io/gnzsnz/ib-gateway:10.40.1a`, also `network_mode: host`). That compose file was removed when the project moved to the per-container layout above, but the container itself was never torn down and can still be running. Because it shares host networking with `ib-gateway-paper`, **the two fight over the same host ports** and this is the likely root cause of the "Address already in use" stuck-restart incidents. Check for it and remove if still present:
> ```bash
> docker ps -a --filter name=ibgw-paper
> docker stop ibgw-paper && docker rm ibgw-paper
> ```

---

## 🔌 Port Reference

| Trading Mode | IBGateway internal bind | Host/LAN-reachable port | TWS (Standard) |
| --- | --- | --- | --- |
| **Paper** (`ib-gateway-paper`) | `127.0.0.1:4002` (loopback only) | **4004** — via `socat`, offset +2 to avoid colliding with IBGateway's own loopback bind in the shared host netns | 7497 |
| **Live** (`ib-gateway-live`) | container `4001` | **4001** — direct bridge port-publish, no offset needed (isolated netns, no collision) | 7496 |

> [!IMPORTANT]
> Paper uses `network_mode: host`, so IBGateway's loopback-only API bind (`127.0.0.1:4002`) is *not* reachable from other machines — only from the pi itself. The image's `socat` proxy re-exposes it on `0.0.0.0:4004`. **Any client connecting from a different machine must use port 4004 for paper, not 4002.** Bots running directly on the pi can still use 4002 via loopback. `src/trading/broker/check_ibkr_conn.py` already documents this bridge; `broker_factory.py`'s `IBKR_PAPER_PORT` default (`4002`) is only correct for same-host connections — confirm where your trading bots actually run.
>
> Live has no such offset — it's mapped normally (`4001:4001`) since it isn't host-networked.

### VNC

| Gateway | Host VNC port |
| --- | --- |
| Live | **5901** → container `5900` (explicit `ports:` mapping) |
| Paper | Not published as a mapped port — `network_mode: host` means the container's own VNC listener (default `5900`) is directly on the host if enabled. Not independently confirmed; check `ss -tuln \| grep 5900` on the host if you need VNC into paper. |

---

## 🔑 Credentials (`.env`)

Each gateway has its own `.env` in its own directory (`~/ibkr/ibgateway-paper/.env`, `~/ibkr/ibgateway-live/.env`):

```env
TWS_USERID=your_username
TWS_PASSWORD=your_password
VNC_SERVER_PASSWORD=your_vnc_password
```

> [!IMPORTANT]
> Settings persist under each directory's `config/` (mounted to `/home/ibgateway/Jts/persistent`). Ensure `TrustedAddr` in `jts.ini` includes your LAN subnet to allow remote connections.

---

## 🚀 Daily Operations

### ✅ Check Status

```bash
docker ps --filter name=ib-gateway-paper --filter name=ib-gateway-live
```

### 🔄 Restart

```bash
# Paper
docker restart ib-gateway-paper
# or, from its compose project directory:
cd ~/ibkr/ibgateway-paper && docker compose restart

# Live
docker restart ib-gateway-live
# or:
cd ~/ibkr/ibgateway-live && docker compose restart
```

### 📜 View Logs

```bash
docker logs -f ib-gateway-paper
docker logs -f ib-gateway-live
```

Look for:

```
IBC: Login has completed
```

---

## 🧪 Connectivity Test

### From Pi

```bash
ss -tuln | grep -E '4001|4002|4004'
```

### From Windows / Bot

```python
from ib_insync import IB

# Paper — port 4004 if connecting remotely, 4002 if running on the pi itself
ib_paper = IB()
ib_paper.connect('10.0.0.4', 4004, clientId=101)
print("Paper connected:", ib_paper.isConnected())

# Live
ib_live = IB()
ib_live.connect('10.0.0.4', 4001, clientId=201)
print("Live connected:", ib_live.isConnected())
```

Use **unique `clientId` per bot/process**.

---

## 🔎 Connectivity Diagnostics

Several scripts are available in `src/trading/broker/` to troubleshoot connection issues:

### 1. General Connectivity Check
```bash
python src/trading/broker/check_ibkr_conn.py
```
*   Checks ports **4001** (Live), **4002** (Paper loopback), and **4004** (Paper socat bridge).
*   Attempts an `ib_insync` handshake to verify API availability.
*   Provides specific recommendations if the connection fails.

### 2. Live & Paper Data Test
```bash
python src/trading/broker/check_ibkr_conn_live_paper.py
```
*   Verifies connection to both instances.
*   Fetches account management info and current positions.
*   Tests market data retrieval for sample tickers (e.g., MSFT).

### 3. Read-Only Protection Verification
```bash
python src/trading/broker/check_ibkr_live_readonly.py
```
*   **Safety Check:** Connects to the Live instance (4001) and attempts to place a dummy market order.
*   Confirms that the order is blocked by the Gateway's Read-Only settings.

---

## 🛠 Maintenance

### Stop

```bash
docker stop ib-gateway-paper
docker stop ib-gateway-live
```

### Edit Credentials

```bash
nano ~/ibkr/ibgateway-paper/.env   # or ~/ibkr/ibgateway-live/.env
cd ~/ibkr/ibgateway-paper && docker compose up -d   # recreate with new .env
```

---

## 🩹 Watchdog (auto-restart on stuck bind loop) — PAPER only

IBC restarts the Gateway process in-place periodically (e.g. nightly). If a second
process is squatting on the same host port (see the abandoned `ibgw-paper` warning
above) — or occasionally even without one, if `socat` doesn't release the port in
time — the new `socat` fork loops forever logging `Address already in use`
(sometimes preceded by an Xvfb `Fatal server error:`). This wedges the paper API
until the container is restarted — it does not self-heal.

`ibgw_paper_watchdog.sh` checks recent logs of the `ib-gateway-paper` container for
that signature and, if it fires repeatedly within a short window, runs
`docker restart ib-gateway-paper` directly — scoped to that one container only, so it
never touches the live gateway. Runs every 5 minutes via `ibgw-paper-watchdog.timer`,
so an outage is bounded to ~5-10 minutes instead of persisting for hours.

Treat this as a safety net, not the fix — removing the abandoned `ibgw-paper`
container (above) addresses the actual root cause; the watchdog just bounds the
damage if the stuck-bind loop recurs for any other reason.

There is no live-side watchdog: `ib-gateway-live` isn't host-networked, so it isn't
exposed to this port-collision failure mode.

### Deploy

```bash
sudo cp ibgw_paper_watchdog.sh /opt/apps/e-trading/bin/install-ibkr/ibgw_paper_watchdog.sh
sudo chmod +x /opt/apps/e-trading/bin/install-ibkr/ibgw_paper_watchdog.sh
sudo cp ibgw-paper-watchdog.service ibgw-paper-watchdog.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ibgw-paper-watchdog.timer
```

### Verify

```bash
sudo systemctl list-timers ibgw-paper-watchdog.timer
sudo systemctl start ibgw-paper-watchdog.service   # run once immediately, check output
journalctl -u ibgw-paper-watchdog.service -n 20 --no-pager
```

---

## 🔐 Security / Best Practices

* **UFW firewall:**
  Allow access from LAN only:

  ```bash
  sudo ufw allow from 10.0.0.0/24 to any port 4001 proto tcp
  sudo ufw allow from 10.0.0.0/24 to any port 4004 proto tcp
  ```

* **Live runs continuously** (`restart: always`) with `READ_ONLY_API=yes` as the
  order-placement safety net. If you intended live to be started only on demand,
  that's a config change needed in `~/ibkr/ibgateway-live/docker-compose.yml`
  (`restart: always` → e.g. `restart: "no"`), not the current behavior.

* **Persistent Session:** already configured — `./config` is mounted to
  `/home/ibgateway/Jts/persistent` in both compose files, preserving tokens and
  reducing 2FA prompts across restarts.

---

## 🧾 Quick Commands Table

| Action | Command |
| --- | --- |
| Status (both) | `docker ps --filter name=ib-gateway-paper --filter name=ib-gateway-live` |
| Restart paper | `docker restart ib-gateway-paper` |
| Restart live | `docker restart ib-gateway-live` |
| Stop live | `docker stop ib-gateway-live` |
| Follow logs (paper) | `docker logs -f ib-gateway-paper` |
| Follow logs (live) | `docker logs -f ib-gateway-live` |
| Verify ports open | `ss -tuln \| grep -E '4001\|4002\|4004'` |
| Test connection (Py) | `IB().connect('10.0.0.4', 4004, clientId=101)` (paper, remote)<br>`IB().connect('10.0.0.4', 4001, clientId=201)` (live) |

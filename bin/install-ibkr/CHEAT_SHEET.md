Perfect — here’s the **extended cheat sheet** with **live gateway manual controls** included.

---

# 📑 IBKR Gateway on Raspberry Pi – Cheat Sheet

## 🖥 System Overview

* **Software:** **Interactive Brokers Gateway** (not TWS)
* **Host:** Raspberry Pi 5 (Ubuntu Server)
* **Paper Service:** `ibgw-paper` (port **4002**, auto-start on boot)
* **Live Service:** `ibgw-live` (port **4001**, manual start only)
* **Compose File:** `~/ibkr/docker-compose.yml`
* **Systemd Unit:** `ibgateway-docker.service` (starts paper only)

---

## 🔌 Port Reference

| Trading Mode      | **IB Gateway (Docker)** | **TWS (Standard)** |
| ----------------- | ----------------------- | ------------------ |
| **Paper Trading** | **4002**                | 7497               |
| **Live Trading**  | **4001**                | 7496               |

> [!NOTE]
> This project defaults to IB Gateway ports (**4001/4002**) to match the Raspberry Pi Docker deployment. If you switch to using TWS on your local machine, remember to update the ports.

### 📄 Live vs Paper Configuration

| Feature               | **Live Gateway** (`ibgateway-live`) | **Paper Gateway** (`ibgateway-paper`) |
| --------------------- | ----------------------------------- | ------------------------------------- |
| **API Port**          | **4001**                            | **4002**                              |
| **VNC Port**          | **5901**                            | **5902**                              |
| **Trading Mode**      | `live`                              | `paper`                               |
| **API Access**        | **Read-Only** (`yes`)               | **Trade/Read/Write** (`no`)           |
| **Docker Image**      | `gnzsnz/ib-gateway:stable`          | `ghcr.io/gnzsnz/ib-gateway:stable`    |
| **Settings Volume**   | `./config` -> `/persistent`         | `./config` -> `/persistent`           |
| **Environment File**  | `.env`                              | `.env`                                |

---

## 🔑 Credentials (`.env`)

Both gateways use a `.env` file in their respective folders for credentials. 

```env
TWS_USERID=your_username
TWS_PASSWORD=your_password
VNC_SERVER_PASSWORD=your_vnc_password
```

> [!IMPORTANT]
> The `jts.ini` configuration in `config/` is pre-configured for the respective trading modes and API access levels. Ensure `TrustedAddr` includes your local subnet (e.g., `10.0.0.13`) to allow remote connections.

---

## 🚀 Daily Operations

### ✅ Check Status

```bash
sudo systemctl status ibgateway-docker --no-pager -l
docker compose -f ~/ibkr/docker-compose.yml ps
```

### 🔄 Restart Paper Gateway

```bash
sudo systemctl restart ibgateway-docker
```

### 📜 View Logs

```bash
docker compose -f ~/ibkr/docker-compose.yml logs -f ibgw-paper
# For live:
docker compose -f ~/ibkr/docker-compose.yml logs -f ibgw-live
```

Look for:

```
IBC: Login has completed
```

---

## 🧪 Connectivity Test

### From Pi

```bash
ss -tuln | grep -E '4001|4002'
# Expect both ports LISTEN if live is started manually
```

### From Windows / Bot

```python
from ib_insync import IB

# Paper
ib_paper = IB()
ib_paper.connect('10.0.0.4', 4002, clientId=101)
print("Paper connected:", ib_paper.isConnected())

# Live (if started)
ib_live = IB()
ib_live.connect('10.0.0.4', 4001, clientId=201)
print("Live connected:", ib_live.isConnected())
```

Use **unique `clientId` per bot/process**.

---

## � Connectivity Diagnostics

Several scripts are available in `src/trading/broker/` to troubleshoot connection issues:

### 1. General Connectivity Check
```bash
python src/trading/broker/check_ibkr_conn.py
```
*   Checks if ports **4001** (Live) and **4002** (Paper) are open on the Pi.
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

## �🛠 Maintenance

### Stop Paper Gateway

```bash
sudo systemctl stop ibgateway-docker
```

### Manual Start/Stop (Compose)

| Action | Paper                             | Live                             |
| ------ | --------------------------------- | -------------------------------- |
| Start  | `docker compose up -d ibgw-paper` | `docker compose up -d ibgw-live` |
| Stop   | `docker compose stop ibgw-paper`  | `docker compose stop ibgw-live`  |

### Edit Credentials

```bash
nano ~/ibkr/.env
# Update PAPER_USER / PAPER_PASS / LIVE_USER / LIVE_PASS
docker compose up -d
```

---

## 🔐 Security / Best Practices

* **UFW firewall:**
  Allow access from LAN only:

  ```bash
  sudo ufw allow from 10.0.0.0/24 to any port 4001 proto tcp
  sudo ufw allow from 10.0.0.0/24 to any port 4002 proto tcp
  ```

* **Keep live gateway off by default** to avoid accidental orders.
  Only start it when you intend to trade live.

* **Persistent Session (optional):**

  ```yaml
  volumes:
    - ./jts:/home/ibgateway/Jts
  ```

  Saves tokens and reduces 2FA prompts across restarts.

---

## 🧾 Quick Commands Table

| Action                | Command                                                                                                         |               |         |
| --------------------- | --------------------------------------------------------------------------------------------------------------- | ------------- | ------- |
| Status (all)          | `docker compose -f ~/ibkr/docker-compose.yml ps`                                                                |               |         |
| Restart paper gateway | `sudo systemctl restart ibgateway-docker`                                                                       |               |         |
| Start live gateway    | `docker compose up -d ibgw-live`                                                                                |               |         |
| Stop live gateway     | `docker compose stop ibgw-live`                                                                                 |               |         |
| Follow logs (paper)   | `docker compose logs -f ibgw-paper`                                                                             |               |         |
| Follow logs (live)    | `docker compose logs -f ibgw-live`                                                                              |               |         |
| Verify ports open     | \`ss -tuln                                                                                                      | grep -E '4001 | 4002'\` |
| Test connection (Py)  | `IB().connect('10.0.0.4', 4002, clientId=101)` (paper)<br>`IB().connect('10.0.0.4', 4001, clientId=201)` (live) |               |         |

---

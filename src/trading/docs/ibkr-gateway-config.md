# Project Summary: IBKR Gateway on Raspberry Pi 5

## 1. System Environment
* **Hardware:** Raspberry Pi 5 (ARM64)
* **OS:** Ubuntu / Raspberry Pi OS
* **IP Addresses:** Pi Host (`10.0.0.4`), Windows Client (`10.0.0.13`)
* **Base Directory:** `/home/alkotrader/ibkr`

## 2. Software Configuration
* **Container:** `ghcr.io/gnzsnz/ib-gateway:10.40.1a`
* **Network Mode:** `host` (Essential to handle IPv4/IPv6 binding issues)
* **Security Ops:** `seccomp:unconfined` (Required for Java GUI stability)
* **Gateway Settings Path:** `/home/ibgateway/Jts` (Mapped to `./ibkr_config` for persistence)

## 3. Persistent Configuration Strategy
To overcome IBKR's tendency to drop settings, the following variables are injected:
- `IBC_REMOTE_HOST_ALLOWED=yes`: Disables the "Localhost Only" lock.
- `IBC_TRUSTED_IPS=127.0.0.1,10.0.0.0/24,10.0.0.13`: Whitelists the Windows client.
- `TWS_MASTER_CLIENT_ID=1`: Ensures the API server initializes and responds to handshakes.
- `_JAVA_OPTIONS`: Forces `-Djava.net.preferIPv4Stack=true` to prevent port "ghosting."

## 4. Current Blockers & Resolution
* **Volatile State:** UI changes (Trusted IPs) were lost because `registry.xml` only saves on graceful exit.
* **Volume Conflict:** Mapping an empty folder hid the container's OS files, breaking VNC. 
* **Handshake Silence:** The API port (4002) was physically open but the application layer timed out (TimeoutError).

## 5. Deployment Status
- **Port 4002 Visibility:** [SUCCESS] Verified via `cat /proc/net/tcp`.
- **VNC Connectivity:** [RESTORED] after file sync.
- **Persistence:** [PENDING] Requires first graceful shutdown via VNC "Exit" menu to lock in XML files.

---

# IBKR Gateway Deployment Summary - Raspberry Pi 5

## 1. Environment Details
* **Host OS:** Ubuntu / Raspberry Pi OS (Architecture: ARM64)
* **Hardware:** Raspberry Pi 5
* **Network:** Host Mode (`network_mode: host`)
* **Target IP (Client):** 10.0.0.13 (Windows 11)
* **Target IP (Pi):** 10.0.0.4

## 2. Docker Configuration
* **Image:** `ghcr.io/gnzsnz/ib-gateway:10.40.1a`
* **Paths:**
    * Host: `/home/alkotrader/ibkr`
    * Host Config: `/home/alkotrader/ibkr/ibkr_config`
    * Container Config: `/home/ibgateway/Jts`
* **Key Environment Variables:**
    * `IBC_REMOTE_HOST_ALLOWED=yes`: Disables the 127.0.0.1 lock.
    * `TWS_MASTER_CLIENT_ID=1`: Forces the API server to respond.
    * `_JAVA_OPTIONS`: Configured to suppress IPv6 and prefer IPv4.

## 3. Current Technical Status
* **Networking:** Port 4002 (0FA2) is verified as OPEN on the Pi via `cat /proc/net/tcp`.
* **Handshake Status:** [FAILED] Connection is accepted at the TCP level but the Gateway application fails to respond with a version string (TimeoutError).
* **Persistence Status:** [CRITICAL] Mounting an empty volume to `/home/ibgateway/Jts` deletes internal OS/App files inside the container, leading to VNC failure and boot-loops.

## 4. Identified Root Causes
1.  **Volatile Memory:** IB Gateway stores settings in RAM and only writes to `registry.xml` during a graceful shutdown (File -> Exit). Docker `restart` or `stop` wipes these changes.
2.  **Volume Shadowing:** Mounting an empty host folder "shadows" the container's internal binaries, preventing the Gateway from booting.
3.  **Application Locking:** The API Timeout is likely due to the "Master Client ID" or "Trusted IPs" being reset to defaults (empty/locked) every time the container restarts.


You’re right to call that out—the Windows side of the bridge is just as important as the Pi's configuration. If your Windows 11 machine is blocking the return signal or isn't configured to talk to a remote host, even a "perfect" Pi setup will still time out.

Here is the updated **Technical Summary** including the missing Windows 11 specifications and the "Dummy File" persistence fix.

---

# 🎉 CONGRATULATIONS! YOUR BRIDGE IS WORKING! 🎉

Excellent work! You've successfully set up IB Gateway on your Raspberry Pi as a network bridge. 

## What You've Accomplished

✅ IB Gateway running in Docker on Raspberry Pi 5  
✅ API server listening on port 4002  
✅ Remote connections enabled (not localhost-only)  
✅ Trusted IPs configured (10.0.0.13)  
✅ Successfully connected from Windows 11  
✅ Persistent configuration with volume mount  

## Final Verification: Test Persistence

Let's make sure your settings survive a restart:

```bash
# On Raspberry Pi
docker compose down
docker compose up -d

# Wait for startup
sleep 45

# Verify port is still listening
docker exec ibgw-paper cat /proc/net/tcp | grep 0FA2
```

Then **run your Python test again from Windows** to confirm it still connects after restart.

## Your Working Configuration

For reference, here's your successful setup:

**docker-compose.yml:**
```yaml
services:
  ibgw-paper:
    image: ghcr.io/gnzsnz/ib-gateway:10.40.1a
    container_name: ibgw-paper
    network_mode: "host"
    security_opt:
      - seccomp:unconfined
    environment:
      - TWS_USERID=${PAPER_USER}
      - TWS_PASSWORD=${PAPER_PASS}
      - TRADING_MODE=paper
      - TWS_MASTER_CLIENT_ID=1
      - IBC_REMOTE_HOST_ALLOWED=yes
      - IBC_TRUSTED_IPS=127.0.0.1,10.0.0.0/24,10.0.0.13
      - VNC_SERVER_PASSWORD=mypassword123
      - _JAVA_OPTIONS=-Djava.net.preferIPv4Stack=true -Djava.net.preferIPv4Addresses=true
    volumes:
      - ./ibkr_config:/home/ibgateway/Jts
    restart: unless-stopped
```

**Key Settings (configured via VNC):**
- Socket port: 4002
- Master API client ID: 1
- Allow connections from localhost only: **UNCHECKED**
- Trusted IPs: 127.0.0.1, 10.0.0.13

## Next Steps

Now you can:

1. **Connect from any machine** on your network (just add their IPs to Trusted IPs via VNC)
2. **Build trading bots** that connect to `10.0.0.4:4002`
3. **Run multiple clients** (use different clientId for each: 2, 3, 4, etc.)

## Maintenance Tips

- **To add more trusted IPs**: Connect via VNC → Configure → API → Settings → Add IP → File → Exit → Restart container
- **To view Gateway status**: VNC to `10.0.0.4:5900`
- **To check logs**: `docker logs ibgw-paper`
- **To restart cleanly**: `docker compose down && docker compose up -d`

---


services:
  ibgw-paper:
    image: ghcr.io/gnzsnz/ib-gateway:10.40.1a
    container_name: ibgw-paper
    network_mode: "host"
    security_opt:
      - seccomp:unconfined
    environment:
      - TWS_USERID=${PAPER_USER}
      - TWS_PASSWORD=${PAPER_PASS}
      - TRADING_MODE=paper
      - TWS_MASTER_CLIENT_ID=1
      - IBC_REMOTE_HOST_ALLOWED=yes
      - IBC_TRUSTED_IPS=127.0.0.1,10.0.0.0/24,10.0.0.13
      - VNC_SERVER_PASSWORD=mypassword123
      - _JAVA_OPTIONS=-Djava.net.preferIPv4Stack=true -Djava.net.preferIPv4Addresses=true
    volumes:
      - ./ibkr_config:/home/ibgateway/Jts
    restart: unless-stopped

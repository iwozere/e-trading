# IBKR Gateway Config (Raspberry Pi / Remote API)

This is the current, clean setup reference for running IBKR Gateway in Docker and connecting from another machine.

## Target topology

- IBKR Gateway host: Raspberry Pi (Linux, ARM64)
- Remote client: trading runtime on another machine (Windows/Linux/macOS)
- Gateway API port: `4002` (paper) or your configured port

## Docker compose example

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
      - IBC_TRUSTED_IPS=127.0.0.1,10.0.0.0/24
      - _JAVA_OPTIONS=-Djava.net.preferIPv4Stack=true -Djava.net.preferIPv4Addresses=true
    volumes:
      - ./ibkr_config:/home/ibgateway/Jts
    restart: unless-stopped
```

## Required gateway UI settings

Via VNC/API settings in IB Gateway:

- API socket port matches your bot config (example: `4002`)
- "Allow connections from localhost only" is disabled
- Trusted IP list includes your trading host IP
- Master API client ID configured (example: `1`)

Important: make one graceful Gateway exit after changes to persist config.

## Connectivity checks

On gateway host:

```bash
docker logs ibgw-paper
```

From trading host:

- verify TCP reachability to gateway host + port
- start bot in paper mode first

## Trading config alignment

Your bot config must point to the same endpoint and mode:

- broker type configured for IBKR adapter
- host/IP points to gateway host
- port matches gateway API port
- `trading_mode` is `paper` for initial rollout

## Troubleshooting quick list

- Port open but handshake timeout: check trusted IP and localhost-only flag
- Settings reset after restart: ensure persisted volume and graceful exit
- Intermittent network issues: prefer host networking for gateway container

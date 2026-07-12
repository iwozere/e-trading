# IBKR Gateway Config (Raspberry Pi / Remote API)

This is the current, clean setup reference for running IBKR Gateway in Docker and connecting from another machine.

See `bin/install-ibkr/CHEAT_SHEET.md` for day-to-day operations (restart/logs/watchdog); this doc focuses on the connection topology.

## Target topology

- IBKR Gateway host: Raspberry Pi (Linux, ARM64), containers `ib-gateway-paper` and `ib-gateway-live`
- Remote client: trading runtime on another machine (Windows/Linux/macOS)
- Paper is `network_mode: host`; live uses standard bridge port-publishing

## Actual deployed compose (per-container, not a single shared file)

```yaml
# ~/ibkr/ibgateway-paper/docker-compose.yml
services:
  ib-gateway:
    image: ghcr.io/gnzsnz/ib-gateway:stable
    container_name: ib-gateway-paper
    restart: always
    env_file: .env
    environment:
      - TRADING_MODE=paper
      - READ_ONLY_API=no
      - TWS_ACCEPT_INCOMING_CONNECTION_FROM_ANYWHERE=yes
      - TWS_SETTINGS_PATH=/home/ibgateway/Jts/persistent
    volumes:
      - ./config:/home/ibgateway/Jts/persistent
    network_mode: host
```

```yaml
# ~/ibkr/ibgateway-live/docker-compose.yml
services:
  ib-gateway:
    image: gnzsnz/ib-gateway:stable
    container_name: ib-gateway-live
    restart: always
    env_file: .env
    environment:
      - TRADING_MODE=live
      - READ_ONLY_API=yes
      - TWS_ACCEPT_INCOMING_CONNECTION_FROM_ANYWHERE=yes
      - TWS_SETTINGS_PATH=/home/ibgateway/Jts/persistent
    volumes:
      - ./config:/home/ibgateway/Jts/persistent
    ports:
      - "4001:4001"   # API port for live
      - "5901:5900"   # VNC port for live
```

Both run continuously (`restart: always`); live's `READ_ONLY_API=yes` is the only
thing preventing accidental live orders, not a manual-start gate.

## Ports — read this before connecting remotely

| | Paper | Live |
| --- | --- | --- |
| IBGateway's own bind | `127.0.0.1:4002` (loopback, host netns) | container `4001` |
| Reachable from other machines on | **4004** | **4001** |

Paper's `network_mode: host` means IBGateway's loopback-only API bind sits directly
in the *host's* network namespace — reachable only from the pi itself, not from other
machines, regardless of firewall rules. The image runs a `socat` proxy to work around
this, re-exposing the API on `0.0.0.0:4004` (offset +2 from 4002, chosen to avoid
binding the same port IBGateway itself already holds in that shared namespace).
**Any remote client must connect to paper on port 4004, not 4002.** Only bots running
locally on the pi (via `127.0.0.1`) can use 4002 directly.

Live isn't host-networked, so its bridge port-publish (`4001:4001`) needs no offset —
connect on 4001 as expected.

`src/trading/broker/check_ibkr_conn.py` already checks all three ports (4001, 4002,
4004) and documents this bridge. `broker_factory.py`'s `IBKR_PAPER_PORT` env default
is `4002` — correct only for same-host connections; set it to `4004` if your trading
runtime connects to paper from a different machine.

## Required gateway UI settings

Via VNC/API settings in IB Gateway:

- API socket port matches your bot config (paper: 4002 internal / 4004 external; live: 4001)
- "Allow connections from localhost only" is disabled
- Trusted IP list includes your trading host IP
- Master API client ID configured (example: `1`)

Important: make one graceful Gateway exit after changes to persist config.

## Connectivity checks

On gateway host:

```bash
docker logs ib-gateway-paper
docker logs ib-gateway-live
```

From trading host:

- verify TCP reachability to gateway host on the correct port (4004 for paper if remote, 4001 for live)
- start bot in paper mode first

## Trading config alignment

Your bot config must point to the same endpoint and mode:

- broker type configured for IBKR adapter
- host/IP points to gateway host
- port matches gateway API port (see table above — paper is NOT 4002 for remote clients)
- `trading_mode` is `paper` for initial rollout

## Troubleshooting quick list

- Port open but handshake timeout: check trusted IP and localhost-only flag
- Connecting to paper from a remote machine but nothing responds on 4002: use 4004 instead (see Ports section)
- Settings reset after restart: ensure persisted volume and graceful exit
- Paper stuck logging `Address already in use` after a gateway restart and never recovers: see the Watchdog section in `bin/install-ibkr/CHEAT_SHEET.md` — check first for an orphaned duplicate host-networked container squatting on the same port (`docker ps -a`)

# Log Monitoring Setup — Raspberry Pi Production Server

## Overview

This document describes how to set up automated log monitoring for all Linux systemd services
and Docker containers on the Raspberry Pi production server. The setup uses **Vector** to collect,
filter, and deduplicate error logs, forwarding alerts through the existing notification pipeline
(PostgreSQL → `notification-bot.service` → Telegram).

**What it does:**
- Monitors all systemd services and Docker containers in real time
- Filters log lines matching `ERROR`, `Exception`, `CRITICAL`, `Traceback`, `fatal`
- Deduplicates similar errors — at most one alert per unique error fingerprint per 30 minutes
- Delivers alerts via the existing Telegram notification channel

**What gets deployed:**
- One new internal FastAPI route added to `trading-api.service` (`src/api/internal_routes.py`)
- `trading-api.service` deployed and enabled on the Pi (currently not configured there)
- Vector (single Rust binary, ~15–30 MB RAM) running as a systemd service

---

## Architecture

```
systemd services (journald) ──┐
                               ├──► Vector ──► normalize ──► filter ──► fingerprint ──► throttle ──► HTTP POST
Docker containers             ──┘                                                       (30 min)          │
                                                                                                 localhost:5003
                                                                                                          │
                                                                                           POST /internal/log-alert
                                                                                                          │
                                                                                           trading-api.service
                                                                                                          │
                                                                                           PostgreSQL msg_messages
                                                                                                          │
                                                                                           notification-bot.service
                                                                                                          │
                                                                                                   Telegram alert
```

---

## Prerequisites

Verify the following on the Raspberry Pi before starting:

- Raspberry Pi OS Bookworm (64-bit) or Ubuntu 22.04+
- Python venv at `/opt/apps/e-trading/.venv`
- PostgreSQL running: `sudo systemctl status postgresql`
- Docker installed (required for container log monitoring)
- `/opt/apps/e-trading/config/donotshare/.env` present and populated

---

## Part 1: Code Changes (Development Machine)

Two minimal changes are required. Commit them and deploy to the Pi via `git pull`.

### 1.1 Create `src/api/internal_routes.py`

This is a dedicated FastAPI router for internal system-to-system calls. It has no
authentication but enforces localhost-only access at the application level.

```python
"""Internal routes for system-to-system communication — no auth, localhost only."""
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from src.data.db.services.notification_service import NotificationService
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

router = APIRouter(prefix="/internal", tags=["internal"])

_LOCALHOST = {"127.0.0.1", "::1"}


class LogAlertRequest(BaseModel):
    text: str
    source: str


@router.post("/log-alert", include_in_schema=False)
async def receive_log_alert(request: Request, body: LogAlertRequest) -> dict:
    """Receive a log error alert from Vector. Restricted to localhost."""
    if request.client.host not in _LOCALHOST:
        raise HTTPException(status_code=403, detail="Forbidden")

    svc = NotificationService()
    svc.create_message({
        "message_type": "system_alert",
        "channels": ["telegram"],
        "recipient_id": "REPLACE_WITH_ADMIN_USER_ID",  # see Part 2, Step 4
        "content": {"message": body.text, "source": body.source},
        "priority": "HIGH",
    })
    _logger.info("Log alert queued: source=%s", body.source)
    return {"ok": True}
```

### 1.2 Register the Router in `src/api/main.py`

Add alongside the other router imports (around line 71):

```python
from src.api.internal_routes import router as internal_router
```

Add alongside the other `include_router` calls (around line 221):

```python
app.include_router(internal_router)
```

### 1.3 Commit and Push

```bash
git add src/api/internal_routes.py src/api/main.py
git commit -m "Add internal log-alert endpoint for Vector monitoring"
git push
```

---

## Part 2: Ensure `trading-webui.service` is running on the Pi

The `/internal/log-alert` endpoint is part of the main FastAPI app (`src.api.main`),
which is served by `trading-webui.service` on port 5003. No separate service is needed.

### Step 1 — Pull the latest code

```bash
cd /opt/apps/e-trading
git pull
```

### Step 2 — Restart the web UI service to load the new internal router

```bash
sudo systemctl restart trading-webui.service
sudo systemctl status trading-webui.service
```

### Step 3 — Verify

```bash
curl -s http://localhost:5003/api/health | python3 -m json.tool
```

The health endpoint should return `{"status": "healthy", ...}`.

### Step 4 — Test the internal endpoint

```bash
curl -s -X POST http://localhost:5003/internal/log-alert \
  -H "Content-Type: application/json" \
  -d '{"text": "[systemd/manual] Test alert", "source": "manual-test"}' \
  | python3 -m json.tool
```

Expected: `{"ok": true}`

The health endpoint should return `{"status": "healthy", ...}`.

---

## Part 3: Install Vector

Vector is distributed as a single Rust binary. The official apt repository supports
Raspberry Pi OS (arm64 and armhf).

### Step 1 — Add the Vector apt repository

```bash
curl -1sLf 'https://repositories.timber.io/public/vector/cfg/setup/bash.deb.sh' | sudo -E bash
```

### Step 2 — Install Vector

```bash
sudo apt-get install -y vector
```

This installs:

| Path | Purpose |
|------|---------|
| `/usr/bin/vector` | The binary |
| `/etc/vector/vector.toml` | Main config file |
| `/lib/systemd/system/vector.service` | Systemd unit (pre-created) |
| `/var/lib/vector/` | Internal state (throttle windows, etc.) |

### Step 3 — Verify

```bash
vector --version
```

---

## Part 4: Configure Vector

### Step 1 — Grant Vector access to log sources

```bash
# Read journald logs
sudo usermod -a -G systemd-journal vector

# Read Docker container logs
sudo usermod -a -G docker vector
```

### Step 2 — Write the config file

Replace the contents of `/etc/vector/vector.toml`:

```toml
# ─── Sources ───────────────────────────────────────────────────────────────────

[sources.journald]
type = "journald"
# Monitors all systemd units. To restrict to specific services:
# units = ["trading-api.service", "nginx.service", "postgresql.service"]

[sources.docker]
type = "docker_logs"
# Monitors all containers. To restrict to specific containers:
# include_containers = ["my-app", "my-worker"]

# ─── Transforms ────────────────────────────────────────────────────────────────

# Normalise field names: journald and Docker use different field names
[transforms.normalize]
type = "remap"
inputs = ["journald", "docker"]
source = '''
  if exists(.container_name) {
    .source_type = "docker"
    .source_name = string!(.container_name)
  } else {
    .source_type = "systemd"
    .source_name = string(._SYSTEMD_UNIT) ?? string(.SYSLOG_IDENTIFIER) ?? "unknown"
  }
  .message = string(.MESSAGE) ?? string!(.message)
  . = { "source_type": .source_type, "source_name": .source_name, "message": .message }
'''

# Keep only lines that look like errors
[transforms.errors_only]
type = "filter"
inputs = ["normalize"]
condition = 'match(string!(.message), r"(?i)(error|exception|critical|traceback|fatal)")'

# Build a stable fingerprint by stripping variable parts (numbers, UUIDs,
# timestamps) so "Error on item 42" and "Error on item 99" count as one error
[transforms.fingerprint]
type = "remap"
inputs = ["errors_only"]
source = '''
  fp = replace(string!(.message), r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\S*", "<TS>")
  fp = replace(fp, r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "<UUID>")
  fp = replace(fp, r"\b\d+\b", "<N>")
  .dedup_key = .source_name + "|" + slice!(fp, 0, 180)
'''

# Cooldown: max 1 alert per unique error fingerprint per 30 minutes.
# After the window expires the same error will fire again if it recurs.
[transforms.cooldown]
type = "throttle"
inputs = ["fingerprint"]
threshold = 1
window_secs = 1800    # 30 minutes — change here to adjust cooldown
key_field = "dedup_key"

# Shape the payload to match the LogAlertRequest Pydantic model
[transforms.final_shape]
type = "remap"
inputs = ["cooldown"]
source = '''
  . = {
    "text": "[" + .source_type + "/" + .source_name + "] " + .message,
    "source": .source_name,
  }
'''

# ─── Sink ──────────────────────────────────────────────────────────────────────

[sinks.notify]
type = "http"
inputs = ["final_shape"]
uri = "http://127.0.0.1:5003/internal/log-alert"
method = "post"
encoding.codec = "json"

[sinks.notify.request]
retry_attempts = 5
retry_initial_backoff_secs = 1
retry_max_duration_secs = 30
```

### Step 3 — Enable and restart Vector

```bash
sudo systemctl enable vector
sudo systemctl restart vector
sudo systemctl status vector
```

### Step 4 — Watch startup logs

```bash
sudo journalctl -u vector -f --no-pager
```

Vector should report sources connected and no errors. Press `Ctrl+C` when satisfied.

---

## Part 5: Testing

### Test 1 — Endpoint responds correctly

Send a manual alert from localhost:

```bash
curl -s -X POST http://localhost:5003/internal/log-alert \
  -H "Content-Type: application/json" \
  -d '{"text": "[systemd/manual] Test alert from monitoring setup", "source": "manual-test"}' \
  | python3 -m json.tool
```

Expected: `{"ok": true}`

Verify it appeared in the database:

```bash
psql -U your_db_user -d your_db_name \
  -c "SELECT id, status, content FROM msg_messages ORDER BY id DESC LIMIT 3;"
```

### Test 2 — End-to-end via journald

Inject a fake error into the journal:

```bash
systemd-cat -t test-service -p err echo "ERROR: test exception injected by monitoring setup"
```

Within a few seconds, Vector should detect the line and POST it to the endpoint.
Check the database as above; a new `msg_messages` row with `message_type = 'system_alert'`
should appear.

### Test 3 — 30-minute cooldown / deduplication

Inject the same error twice in quick succession:

```bash
systemd-cat -t test-service -p err echo "ERROR: repeated connection error on port 5432"
systemd-cat -t test-service -p err echo "ERROR: repeated connection error on port 9999"
```

Both lines normalise to fingerprint `"test-service|ERROR: repeated connection error on port <N>"`.
Only **one** row should appear in `msg_messages` — the second is suppressed by the throttle.

---

## Part 6: Maintenance

### Adjust the cooldown window

Edit `/etc/vector/vector.toml`, find `window_secs` under `[transforms.cooldown]`, then:

```bash
sudo systemctl restart vector
```

### Restrict to specific services or containers

In `/etc/vector/vector.toml`:

```toml
[sources.journald]
type = "journald"
units = ["trading-api.service", "notification-bot.service", "nginx.service"]

[sources.docker]
type = "docker_logs"
include_containers = ["my-app", "postgres"]
```

### Exclude known noisy patterns

Add an extra filter transform between `errors_only` and `fingerprint`:

```toml
[transforms.exclude_noise]
type = "filter"
inputs = ["errors_only"]
condition = '!match(string!(.message), r"ExpectedDisconnect|HealthCheckTimeout")'
```

Then change `[transforms.fingerprint]` to `inputs = ["exclude_noise"]`.

### Add more error patterns

Edit the `condition` in `[transforms.errors_only]`:

```toml
condition = 'match(string!(.message), r"(?i)(error|exception|critical|traceback|fatal|your-pattern)")'
```

---

## Troubleshooting

| Symptom | Action |
|---------|--------|
| Vector not starting | `sudo journalctl -u vector -n 50` |
| No alerts arriving | `sudo journalctl -u vector -f` — look for HTTP 4xx/5xx or connection errors |
| `403 Forbidden` from endpoint | Vector is not connecting from 127.0.0.1 — verify `uri` in `vector.toml` |
| `Connection refused` on port 5003 | `trading-api.service` is not running — `sudo systemctl status trading-api` |
| Endpoint returns 500 | `recipient_id` is invalid — verify user ID in `users` table |
| Journald access denied | `sudo usermod -a -G systemd-journal vector && sudo systemctl restart vector` |
| Docker logs not appearing | `sudo usermod -a -G docker vector && sudo systemctl restart vector` |
| Alert storm (too many) | Increase `window_secs` in `[transforms.cooldown]` |
| Errors not detected | Widen the regex in `[transforms.errors_only]` |
| Vector consuming too much CPU | Restrict sources to specific units/containers (see Maintenance section) |

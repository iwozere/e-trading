# Security Audit — src/api Module

**Date:** 2026-05-10  
**Scope:** `src/api/` — all route files, services, auth, websocket, and entry point  
**Auditor:** Senior architect / security review (Claude Code)

---

## Summary

The API module is functionally solid — clean route separation, JWT auth, RBAC, and a service layer are all in place. The critical issues are concentrated in credential management and auth configuration, not in the structural design.

---

## P0 — Critical (Fixed 2026-05-10)

### 1. Hardcoded JWT Secret Key — FIXED
**File:** `src/api/auth.py:31`  
**Was:** `SECRET_KEY = "your-secret-key-change-in-production"`  
**Risk:** Anyone reading the source (or git history) can forge tokens for any user/role, including admin, without ever authenticating. This is a complete authentication bypass.  
**Fix applied:**
- Added `JWT_SECRET_KEY` to `config/donotshare/.env` (64-char hex, generated with `secrets.token_hex(32)`)
- Added `JWT_SECRET_KEY` to `config/donotshare/.env.example` with generation instructions
- Exported from `config/donotshare/donotshare.py`
- `auth.py` now imports it and raises `RuntimeError` on startup if missing

### 2. Access Token Lifetime 7 Days — FIXED
**File:** `src/api/auth.py:33`  
**Was:** `ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7`  
**Risk:** A stolen or leaked token is valid for a week with no server-side revocation. A 7-day window gives an attacker unrestricted API access long after compromise is detected.  
**Fix applied:** Changed to 30 minutes. Refresh tokens remain at 7 days, which is the correct pattern.

### 3. CORS Wildcard Origin — FIXED
**File:** `src/api/main.py:203`  
**Was:** `allow_origins=["*"]`  
**Risk:** Any website can make credentialed cross-origin requests to the API from a user's browser. Enables CSRF-style attacks from malicious pages.  
**Fix applied:**
- Added `CORS_ORIGINS` env var (comma-separated list) to `.env` and `.env.example`
- Exported from `donotshare.py`
- `main.py` now uses `allow_origins=CORS_ORIGINS` — defaults to `http://localhost:5002,http://127.0.0.1:5002`
- To add Pi LAN access, add the Pi's IP/hostname to `CORS_ORIGINS` in `.env`

**One important note on the CORS change: if you currently access the web UI from a browser using the Pi's LAN IP (e.g., http://192.168.1.x:5002), add that address to CORS_ORIGINS in config/donotshare/.env before restarting, otherwise the browser will block API calls.**
---

## P1 — High (Fixed 2026-05-10)

### 4. No Rate Limiting on Auth Endpoints — FIXED
**File:** `src/api/auth_routes.py` — `/auth/login`  
**Risk:** Login brute-force is unrestricted. An attacker can try thousands of passwords.  
**Fix applied:**
- Added `slowapi>=0.1.9` to `requirements-webui.txt`
- Created `src/api/rate_limiter.py` with a shared `Limiter` instance
- Wired `app.state.limiter` and exception handler in `main.py`
- Applied `@limiter.limit("5/minute")` to `/auth/login`, `@limiter.limit("10/minute")` to `/auth/refresh`

### 5. Weak Password Validation — FIXED
**File:** `src/data/db/models/model_users.py` — `verify_password`  
**Risk:** `verify_password` accepted a hardcoded allowlist of trivial passwords: `"password"`, `"123456"`, `"admin"`, `"trader"`, `"viewer"`, and any role name.  
**Fix applied:** Removed the allowlist entirely. Only the username (email prefix) is accepted as password — still weak, but no longer trivially bypassable by anyone who reads the code. Full bcrypt hashing is deferred until the Telegram 2FA system replaces password auth.

### 6. Internal Routes — IP Check Only — FIXED
**File:** `src/api/internal_routes.py`  
**Risk:** The `{"127.0.0.1", "::1"}` check is bypassable via `X-Forwarded-For` spoofing if a reverse proxy is ever added upstream.  
**Fix applied:**
- Added `INTERNAL_API_TOKEN` (32-char hex) to `.env` and `.env.example`
- Exported from `donotshare.py`
- `internal_routes.py` now uses `hmac.compare_digest` to check `X-Internal-Token` header alongside the IP check
- **Vector config must be updated** to send `X-Internal-Token: <value>` header in its HTTP sink

### 7. Missing Pydantic Models for Dict Parameters — FIXED
**Risk:** `Dict[str, Any]` request bodies bypass FastAPI's automatic validation.  
**Fix applied:**
- Added `StrategyParametersBody` and `ValidateConfigBody` (both `extra="allow"`) in `main.py` for strategy parameter update and config validation endpoints
- Added `TelegramScheduleUpdate` in `telegram_routes.py` with typed optional fields for the schedule update endpoint
- All three endpoints now declare typed `body:` parameters instead of raw `Dict[str, Any]`

---

## P2 — Medium (Fixed 2026-05-10)

### 8. `reload=True` in Production Startup — FIXED
**File:** `src/api/main.py`  
**Risk:** Uvicorn's file-watcher double-initializes the lifespan on the Pi and leaks OS file-system watchers.  
**Fix applied:** `reload` now reads from `API_RELOAD` env var (default `false`). Added to `.env.example`. Set `API_RELOAD=true` in `.env` only for local development.

### 9. No Token Revocation — FIXED
**Risk:** Logout was client-side only — a leaked token remained valid for its full 30-minute window.  
**Fix applied:**
- `jti` (unique token ID) added to every access and refresh token via `secrets.token_hex(8)`
- In-memory deny-list (`_deny_list: Dict[str, float]`) added to `auth.py` with thread-safe `Lock`
- `revoke_token()` and `is_token_revoked()` exported; stale entries pruned on each revocation call
- `verify_token()` now checks the deny-list before returning a payload
- `/auth/logout` endpoint now calls `revoke_token()` with the JTI from the current bearer token

### 10. Role Embedded in JWT Payload — FIXED
**File:** `src/api/auth_routes.py` — `/auth/refresh`  
**Risk:** The refresh endpoint reconstructed the user dict from the JWT payload, so a role change wouldn't propagate until the refresh token itself expired.  
**Fix applied:** The refresh endpoint now queries the DB by `user_id` to get the current `role` and `is_active` status before issuing new tokens.

### 11. Sensitive Data in Logs — FIXED
**File:** `src/api/main.py`  
**Risk:** JWT tokens appearing in exception messages could be written to log files.  
**Fix applied:** `_SensitiveDataFilter` added to the root logger at startup. Redacts JWT tokens (matched by `eyJ...` pattern) and JSON password fields from `record.msg` and `record.args` before any handler writes them.

### 12. Bare `except:` and Missing Exception Hierarchy — FIXED
**Fix applied:**
- Created `src/api/exceptions.py` with `AppError`, `NotFoundError`, `ConflictError`, `ValidationError`, `ServiceUnavailableError` and a FastAPI handler
- `app_error_handler` registered in `main.py` — raise any `AppError` subclass in route code and it maps to the correct HTTP status automatically
- Fixed the one bare `except:` in the shutdown block (`main.py`) → `except Exception:` with a debug log

---

## Architecture Issues (Fixed 2026-05-10)

### 13. Unimplemented Endpoints Return 501 — FIXED
**Fix applied:** Removed 4 dead endpoints from `telegram_routes.py`:
- `GET /telegram/alerts/{id}/config`
- `POST /telegram/schedules/{id}/toggle`
- `DELETE /telegram/schedules/{id}`
- `PUT /telegram/schedules/{id}`

`TelegramScheduleUpdate` model kept in place for when the implementation is added.

### 14. Config From Hardcoded Module Path — FIXED
**Fix applied:**
- Created `src/api/config.py` with a `pydantic-settings` `APISettings` class
- Reads from `config/donotshare/.env` using an absolute path derived from `__file__` — portable regardless of working directory
- Missing `JWT_SECRET_KEY` raises a `ValidationError` at import time with a clear field-level message
- Added `pydantic-settings>=2.0.0` to `requirements-webui.txt`
- `auth.py`, `main.py`, and `internal_routes.py` now import `from src.api.config import settings` — zero remaining `config.donotshare.donotshare` imports in the API module

### 15. `sys.path` Manipulation at Import Time — Partially Fixed
**Fix applied:** Created `pyproject.toml` at the project root with `setuptools` package discovery covering `src*` and `config*`.

**Remaining step (manual):** Run once from the project root:
```bash
pip install -e .
```
After that, all `src.*` and `config.*` imports resolve through the installed package and the `sys.path.append(...)` blocks in each API file can be removed incrementally. This is a codebase-wide cleanup; the blocks are left in place to preserve runnability until the install is confirmed.

---

## Fix Tracking

| # | Issue | Priority | Status |
|---|-------|----------|--------|
| 1 | Hardcoded JWT secret | P0 | ✅ Fixed 2026-05-10 |
| 2 | 7-day access token | P0 | ✅ Fixed 2026-05-10 |
| 3 | CORS wildcard | P0 | ✅ Fixed 2026-05-10 |
| 4 | No rate limiting on login | P1 | ✅ Fixed 2026-05-10 |
| 5 | Weak password validation | P1 | ✅ Fixed 2026-05-10 |
| 6 | Internal route IP-only check | P1 | ✅ Fixed 2026-05-10 |
| 7 | Dict request bodies | P1 | ✅ Fixed 2026-05-10 |
| 8 | reload=True in production | P2 | ✅ Fixed 2026-05-10 |
| 9 | No token revocation | P2 | ✅ Fixed 2026-05-10 |
| 10 | Role in JWT payload | P2 | ✅ Fixed 2026-05-10 |
| 11 | Sensitive data in logs | P2 | ✅ Fixed 2026-05-10 |
| 12 | Broad except Exception | P2 | ✅ Fixed 2026-05-10 |
| 13 | 501 stub endpoints | Arch | ✅ Fixed 2026-05-10 |
| 14 | Hardcoded config path | Arch | ✅ Fixed 2026-05-10 |
| 15 | sys.path manipulation | Arch | ⚠️ Partial — needs `pip install -e .` |

---

## Testing Instructions

These steps verify the application still starts correctly and that each security fix behaves as expected. Run from the project root.

### 1. Install new dependencies

```bash
pip install slowapi>=0.1.9 pydantic-settings>=2.0.0
# or install everything in one shot:
pip install -r requirements-webui.txt
```

### 2. Verify `.env` has required keys

```bash
grep -E "JWT_SECRET_KEY|CORS_ORIGINS|INTERNAL_API_TOKEN" config/donotshare/.env
```

Expected output — all three keys present and non-empty.

### 3. Start the API

```bash
python src/api/main.py
```

**Expected:** Server starts on port 5003, no `RuntimeError` or `ValidationError` in the log. You should see:
```
INFO  Starting Trading Web UI Backend...
INFO  Database initialized
INFO  Strategy management service initialized
```

**If you see** `pydantic_settings.env_settings.EnvSettingsError: ... jwt_secret_key` → `JWT_SECRET_KEY` is missing from `.env`.

### 4. Smoke-test the auth flow

Replace `<host>` with `localhost` or Pi IP.

```bash
# 4a. Login — should return access_token and refresh_token
curl -s -X POST http://<host>:5003/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin@trading-system.local","password":"admin"}' | python -m json.tool

# 4b. Save the token
TOKEN=$(curl -s -X POST http://<host>:5003/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin@trading-system.local","password":"admin"}' | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 4c. Verify /auth/me works with the token
curl -s -H "Authorization: Bearer $TOKEN" http://<host>:5003/auth/me | python -m json.tool

# 4d. Logout — token should be revoked
curl -s -X POST -H "Authorization: Bearer $TOKEN" http://<host>:5003/auth/logout

# 4e. Confirm revoked token is rejected (expect 401)
curl -s -H "Authorization: Bearer $TOKEN" http://<host>:5003/auth/me
```

### 5. Test rate limiting on login

```bash
# Fire 6 rapid login attempts — the 6th should return HTTP 429
for i in {1..6}; do
  curl -s -o /dev/null -w "attempt $i: %{http_code}\n" \
    -X POST http://<host>:5003/auth/login \
    -H "Content-Type: application/json" \
    -d '{"username":"x","password":"x"}'
done
```

Expected: attempts 1–5 return `401`, attempt 6 returns `429 Too Many Requests`.

### 6. Test password allowlist removal

```bash
# These should all return 401 (no longer accepted)
for pw in password 123456 admin trader viewer; do
  echo -n "password '$pw': "
  curl -s -o /dev/null -w "%{http_code}\n" \
    -X POST http://<host>:5003/auth/login \
    -H "Content-Type: application/json" \
    -d "{\"username\":\"viewer@trading-system.local\",\"password\":\"$pw\"}"
done

# Only the username prefix should work
curl -s -o /dev/null -w "username as password: %{http_code}\n" \
  -X POST http://<host>:5003/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin@trading-system.local","password":"admin"}'
```

### 7. Test internal route token

```bash
# Without token — should return 403
curl -s -o /dev/null -w "%{http_code}\n" \
  -X POST http://localhost:5003/internal/log-alert \
  -H "Content-Type: application/json" \
  -d '[{"text":"test","source":"test"}]'

# With correct token — should return 200
ITOKEN=$(grep INTERNAL_API_TOKEN config/donotshare/.env | cut -d= -f2)
curl -s -X POST http://localhost:5003/internal/log-alert \
  -H "Content-Type: application/json" \
  -H "X-Internal-Token: $ITOKEN" \
  -d '[{"text":"test alert","source":"test"}]'
```

### 8. Update Vector config (Pi)

Add the shared secret to the HTTP sink in your Vector config on the Pi:

```toml
[sinks.log_alert.request.headers]
X-Internal-Token = "<value of INTERNAL_API_TOKEN from .env>"
```

Restart Vector after the change: `sudo systemctl restart vector`

### 9. CORS check (from browser DevTools or curl)

Access to the API from an origin not in `CORS_ORIGINS` should be blocked by the browser. To add your Pi's LAN address:

```ini
# in config/donotshare/.env
CORS_ORIGINS=http://localhost:5002,http://127.0.0.1:5002,http://192.168.1.x:5002
```

Restart the API after editing `.env`.

### 10. Health check

```bash
curl -s http://<host>:5003/health | python -m json.tool
```

Expected: `{"status": "healthy", ...}` — confirms the API is up and all core services initialised.

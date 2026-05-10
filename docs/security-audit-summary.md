# Security Audit Summary — E-Trading Platform

**Date:** 2026-05-10  
**Scope:** All modules under `src/`  
**Auditor:** Claude Code (Senior Architect & Security Engineer perspective)  
**Branch:** `main` @ `1ae11cd`

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Severity Overview](#severity-overview)
3. [Modules Requiring Immediate Attention](#modules-requiring-immediate-attention)
4. [Critical Findings](#critical-findings)
5. [High Severity Findings](#high-severity-findings)
6. [Medium Severity Findings](#medium-severity-findings)
7. [Low Severity Findings](#low-severity-findings)
8. [Architectural Concerns](#architectural-concerns)
9. [Remediation Roadmap](#remediation-roadmap)

---

## Executive Summary

A full-codebase security and architecture audit was performed across all 21 modules under `src/`. The audit covered injection vulnerabilities, insecure deserialization, authentication gaps, cryptographic weaknesses, path traversal, sensitive data exposure, race conditions, and resource exhaustion.

**Total findings: 48**

| Severity | Count |
|----------|-------|
| CRITICAL | 6 |
| HIGH | 16 |
| MEDIUM | 18 |
| LOW | 8 |

The most severe risks are:
- **Arbitrary code execution** via `pickle.load()` and `torch.load()` on model files across the ML pipeline
- **Authentication bypass** via trivially weak password verification (username == password)
- **Path traversal** exposing arbitrary filesystem files through the SPA catch-all route and email attachments
- **Credential management** relying on a `config.donotshare` Python module that leaks secrets via import
- **Command execution** from database-sourced schedule parameters without validation

---

## Severity Overview

```
CRITICAL  ██████  6
HIGH      ████████████████  16
MEDIUM    ██████████████████  18
LOW       ████████  8
```

---

## Modules Requiring Immediate Attention

Ordered by risk exposure:

| Priority | Module | Reason |
|----------|--------|--------|
| 🔴 P1 | `src/api` | Path traversal, auth bypass, CORS issues |
| 🔴 P1 | `src/data/db/models` | Trivial password bypass — entire auth is broken |
| 🔴 P1 | `src/ml` | `pickle.load()` + `torch.load()` → arbitrary code execution |
| 🔴 P1 | `src/strategy` | Same insecure deserialization as ML module |
| 🔴 P1 | `src/scheduler` | Path traversal + command injection from DB-controlled args |
| 🟠 P2 | `src/notification` | TLS disabled, path traversal in attachments, SMTP creds in source |
| 🟠 P2 | `src/telegram` | Weak random for verification codes, missing admin auth guards |
| 🟠 P2 | `src/config` | Credential templates encourage hardcoded secrets in config |
| 🟡 P3 | `src/common` | MD5 for hashing, bare except clauses, global singleton race |
| 🟡 P3 | `src/web_ui` | `shell=True` subprocess, bare except, warning suppression |
| 🟡 P3 | `src/error_handling` | Bare `except:` swallowing critical errors |

---

## Critical Findings

### C-01 — Path Traversal via SPA Catch-All Route
**Module:** `src/api`  
**File:** `src/api/main.py` ~L735  
**Category:** Path Traversal (CWE-22)

```python
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    dist_path = PROJECT_ROOT / "src/web_ui/frontend/dist"
    local_path = dist_path / full_path   # ← unvalidated user input
    if local_path.is_file():
        return FileResponse(local_path)
```

`full_path` is never resolved or checked against `dist_path`. A request to `GET /../../../../config/donotshare/donotshare.py` serves the credentials file directly.

**Fix:**
```python
requested = (dist_path / full_path).resolve()
if not str(requested).startswith(str(dist_path.resolve())):
    raise HTTPException(status_code=404)
```

---

### C-02 — Trivial Authentication Bypass (Password == Username)
**Module:** `src/data`  
**File:** `src/data/db/models/model_users.py` ~L33  
**Category:** Broken Authentication (CWE-287)

```python
def verify_password(self, password: str) -> bool:
    username = self.email.split('@')[0]
    return password == username  # anyone knowing the email can log in
```

The comment says "Temporary implementation" — this is in production code. Any user's email prefix is a valid password.

**Fix:** Use `passlib` with bcrypt:
```python
from passlib.context import CryptContext
_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(self, password: str) -> bool:
    return _pwd_ctx.verify(password, self.password_hash)
```

---

### C-03 — Insecure Deserialization: `pickle.load()` on Model Files
**Module:** `src/ml`, `src/strategy`, `src/backtester`  
**Category:** Insecure Deserialization (CWE-502)

Affected files (representative):
- `src/strategy/hmm_lstm_strategy.py` ~L224
- `src/strategy/entry/hmm_lstm_entry_mixin.py` ~L132, L142
- `src/strategy/cnn_xgboost_strategy.py` ~L252, L269, L275
- `src/backtester/optimizer/hmm_lstm.py` ~L131, L135
- `src/ml/pipeline/p01_hmm_lstm/x_03_apply_hmm.py` ~L96
- `src/ml/pipeline/p03_cnn_xgboost/x_06_train_xgboost.py` ~L94
- `src/ml/pipeline/p03_cnn_xgboost/x_03_generate_embeddings.py` ~L192
- `src/ml/future/feature_engineering_pipeline.py` ~L757
- `src/ml/future/mlflow_integration.py` ~L476
- `src/data/utils/performance_optimization.py` ~L168
- `src/data/utils/advanced_caching.py` ~L204
- `src/common/sentiments/caching/redis_cache.py` ~L170

```python
with open(hmm_file, 'rb') as f:
    hmm_package = pickle.load(f)  # arbitrary code execution if file is replaced
```

A compromised or injected `.pkl` file executes arbitrary Python at load time. Models are auto-discovered via `glob()` — an attacker dropping a file in the models dir triggers this.

**Fix (short-term):** Add HMAC signature verification before unpickling.  
**Fix (long-term):** Migrate to ONNX for PyTorch models and JSON/numpy `.npy` for feature data.

---

### C-04 — Unsafe `torch.load()` Without `weights_only=True`
**Module:** `src/ml`, `src/strategy`  
**Category:** Insecure Deserialization (CWE-502)

Affected files:
- `src/ml/future/helformer_optuna_train.py` ~L154
- `src/ml/future/nn_regime_detector.py` ~L122
- `src/ml/pipeline/p01_hmm_lstm/x_06_train_lstm.py` ~L153
- `src/ml/pipeline/p02_cnn_lstm_xgboost/x_04_train_cnn_lstm.py` ~L184
- `src/ml/pipeline/p02_cnn_lstm_xgboost/x_05_extract_features.py` ~L124
- `src/ml/pipeline/p03_cnn_xgboost/x_03_generate_embeddings.py` ~L207
- `src/strategy/cnn_xgboost_strategy.py` ~L245
- `src/strategy/future/hybrid_nn_core.py` ~L72

```python
checkpoint = torch.load("helformer_best_model.pt")   # ← executes pickle
```

`torch.load()` uses pickle internally. PyTorch ≥1.13 introduced `weights_only=True` to restrict loading to tensors.

**Fix:**
```python
checkpoint = torch.load(model_path, weights_only=True, map_location="cpu")
```

---

### C-05 — Path Traversal + Command Injection via Scheduler Script Execution
**Module:** `src/scheduler`  
**File:** `src/scheduler/scheduler_service.py` ~L646, L651  
**Category:** Path Traversal (CWE-22) + Command Injection (CWE-78)

```python
script_full_path = Path(PROJECT_ROOT) / script_path   # ← from DB, not validated
cmd = [python_executable, str(script_full_path)] + script_args  # ← args from DB
result = subprocess.run(cmd, ...)
```

`script_path` and `script_args` come directly from the database. A database compromise (or malicious admin) can execute arbitrary Python scripts anywhere on the filesystem.

**Fix:**
```python
ALLOWED_SCRIPTS_DIR = (PROJECT_ROOT / "src").resolve()

def _validate_script_path(script_path: str) -> Path:
    resolved = (PROJECT_ROOT / script_path).resolve()
    if not str(resolved).startswith(str(ALLOWED_SCRIPTS_DIR)):
        raise ValueError(f"Script path outside allowed directory: {script_path}")
    if not resolved.suffix == ".py":
        raise ValueError("Only .py scripts are allowed")
    return resolved
```

---

### C-06 — Path Traversal in Email Attachments
**Module:** `src/notification`  
**File:** `src/notification/channels/email_channel.py` ~L407  
**Category:** Path Traversal (CWE-22)

```python
for filename, attachment_data in attachments.items():
    if isinstance(attachment_data, (str, Path)):
        file_path = Path(attachment_data)
        if file_path.exists():
            with open(file_path, "rb") as f:
                file_data = f.read()
```

Attachment paths are not validated. `../../etc/passwd` as an attachment path reads and emails an arbitrary file.

**Fix:**
```python
ALLOWED_ATTACHMENT_DIR = Path("/var/e-trading/attachments").resolve()

def _safe_attachment_path(path: str) -> Path:
    resolved = Path(path).resolve()
    if not str(resolved).startswith(str(ALLOWED_ATTACHMENT_DIR)):
        raise ValueError(f"Attachment path outside allowed directory: {path}")
    return resolved
```

---

## High Severity Findings

### H-01 — Internal API Token Validation is Optional
**Module:** `src/api`  
**File:** `src/api/internal_routes.py`  
**Category:** Missing Authentication (CWE-306)

```python
if settings.internal_api_token:          # ← skipped when token is empty/None
    if not hmac.compare_digest(...):
        raise HTTPException(403)
```

If `INTERNAL_API_TOKEN` is unset, all internal routes are accessible with no credentials.

**Fix:** Make token presence mandatory and fail at startup if absent.

---

### H-02 — Overly Permissive CORS Configuration
**Module:** `src/api`  
**File:** `src/api/main.py` ~L237  
**Category:** CORS Misconfiguration (CWE-942)

```python
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
```

Combined with `allow_credentials=True`, any origin in `cors_origins_list` can make authenticated cross-origin requests with all methods and headers. If that list ever contains a wildcard or a compromised domain, full credential theft is possible.

**Fix:** Use explicit methods and headers:
```python
allow_methods=["GET", "POST", "PUT", "DELETE"],
allow_headers=["content-type", "authorization"],
```

---

### H-03 — Partial Hardcoding and Fragile `.env` Loading in `config.donotshare`
**Module:** `src/config`, `src/notification`, `src/telegram`  
**Category:** Hardcoded Values / Fragile Configuration (CWE-798 partial)

`config/donotshare/donotshare.py` is correctly implemented as a facade over `.env` — all API keys and passwords are read via `os.getenv()`. However, three real issues remain:

**Issue 1 — Hardcoded infrastructure defaults baked into source:**
```python
POSTGRES_HOST = "localhost"   # not from env — cannot change without code edit
POSTGRES_PORT = 5432          # same
POSTGRES_USER = os.getenv("POSTGRES_USER", "trading_admin")  # hardcoded default
```
If the DB is moved or the username changes, this silently uses stale values.

**Issue 2 — Test credentials hardcoded:**
```python
TEST_DB_URL = "postgresql+psycopg2://test_user:test_password@localhost:5432/e_trading_test"
```
Hardcoded test credentials in source code. If test DB is ever network-accessible, this is a credential leak.

**Issue 3 — Relative `dotenv_path` silently fails:**
```python
load_dotenv(dotenv_path="config/donotshare/.env")
```
This is a relative path resolved against CWD. If any entrypoint is launched from a directory other than the project root, no `.env` is loaded, all `os.getenv()` calls return `None`, and the application starts with all credentials as `None` — potentially bypassing checks that only guard against wrong values, not absent ones.

**Fix:**
```python
# Use absolute path anchored to this file
_ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE)

# Move infra config to env
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))

# Remove hardcoded test URL
TEST_DB_URL = os.getenv("TEST_DB_URL")  # must be set in test .env
```

---

### H-04 — TLS Certificate Validation Can Be Disabled
**Module:** `src/notification`  
**File:** `src/notification/channels/email_channel.py` ~L479  
**Category:** Improper Certificate Validation (CWE-295)

```python
if not self.config.get("validate_certs", True):
    smtp_kwargs["tls_context"].check_hostname = False
    smtp_kwargs["tls_context"].verify_mode = ssl.CERT_NONE
```

An attacker on the network path can MITM SMTP sessions when this is enabled. Configuration flags should never disable TLS verification in production.

**Fix:** Remove this configuration option. Use a test-only fixture for integration tests.

---

### H-05 — Missing Admin Authorization Guard in Telegram Handler
**Module:** `src/telegram`  
**File:** `src/telegram/handlers/admin.py` ~L24  
**Category:** Missing Authorization (CWE-863)

The `/admin` command dispatches to `process_admin_command` without first verifying the sender is an admin. Authorization is delegated to the downstream function, but if that function has a gap, any Telegram user can issue admin commands.

**Fix:** Add an explicit guard at the handler entry point:
```python
async def cmd_admin(msg: Message):
    if not telegram_svc.is_admin(str(msg.from_user.id)):
        await msg.reply("Unauthorized.")
        return
    ...
```

---

### H-06 — Weak Verification Code Generation (`random` instead of `secrets`)
**Module:** `src/telegram`  
**File:** `src/telegram/handlers/account.py` ~L94  
**Category:** Weak Randomness (CWE-330)

```python
verification_code = f"{random.randint(100000, 999999):06d}"
```

Python's `random` module is not cryptographically secure. A 6-digit numeric code has only ~900,000 possibilities regardless.

**Fix:**
```python
import secrets
verification_code = f"{secrets.randbelow(900000) + 100000:06d}"
```

---

### H-07 — Health Check Leaks Internal Metrics Without Auth
**Module:** `src/api`  
**File:** `src/api/health_routes.py` ~L45  
**Category:** Information Disclosure (CWE-200)

```python
return {
    "status": "healthy",
    "database": "connected",
    "total_messages": message_count   # ← leaks operational detail
}
```

Public health endpoint leaks message counts and has no rate limiting, making it a free DoS vector.

**Fix:** Return only `{"status": "healthy"}` from the public endpoint; move detailed metrics to an authenticated `/api/health/detailed` route.

---

### H-08 — No Input Validation on DB-Sourced Schedule Parameters
**Module:** `src/scheduler`  
**File:** `src/scheduler/scheduler_service.py` ~L633  
**Category:** Improper Input Validation (CWE-20)

`script_path`, `script_args`, `cron`, and `task_params` all come from the database without schema validation. A compromised DB record is equivalent to RCE.

**Fix:** Define a strict `ScheduleParams` Pydantic model and validate all records before execution.

---

### H-09 — Subprocess Timeout Does Not Wait for Process Death
**Module:** `src/scheduler`  
**File:** `src/scheduler/scheduler_service.py` ~L674  
**Category:** Resource Exhaustion

```python
except asyncio.TimeoutError:
    process.kill()
    # ← missing: await process.wait()
```

After `kill()`, the process is not awaited. Zombie processes accumulate, eventually exhausting system PIDs.

**Fix:**
```python
except asyncio.TimeoutError:
    process.kill()
    await process.wait()
```

---

### H-10 — Bare `except:` Clauses (Multiple Modules)
**Modules:** `src/error_handling`, `src/ml`, `src/common`, `src/web_ui`  
**Category:** Improper Exception Handling (CWE-390)

Representative occurrences:
- `src/error_handling/error_monitor.py` ~L126
- `src/ml/future/automated_training_pipeline.py` ~L342
- `src/ml/future/mlflow_integration.py` ~L169, L203, L210, L217, L224, L231
- `src/ml/pipeline/p07_combined/regime_model.py` ~L69, L110
- `src/common/sentiments/adapters/async_twitter.py` ~L189
- `src/analytics/advanced_analytics.py` ~L558

```python
except:           # catches KeyboardInterrupt, SystemExit, GeneratorExit
    pass
```

Bare `except:` prevents graceful shutdown and silently swallows trade-critical errors.

**Fix:** Replace all occurrences with `except Exception as e:` (minimum) or a specific exception type.

---

### H-11 — Rate Limit on Login Endpoint Is Insufficient
**Module:** `src/api`  
**File:** `src/api/auth_routes.py` ~L80  
**Category:** Insufficient Rate Limiting (CWE-307)

```python
@limiter.limit("5/minute")
```

5 attempts/minute = 7,200/day. A distributed attacker can brute-force 6-character passwords in days.

**Fix:** Use compound limits: `"3/minute;20/hour;100/day"` plus account lockout after N consecutive failures.

---

### H-12 — Markdown Injection in Alert Notifications
**Module:** `src/scheduler`  
**File:** `src/scheduler/scheduler_service.py` ~L1080  
**Category:** Content Injection (CWE-74)

```python
title = f"🚨 {alert_name}: {ticker} ({timeframe})"
message_parts = [f"**Alert Triggered: {alert_name}**", ...]
```

`alert_name` and `ticker` from the DB are embedded in Markdown without escaping. A DB compromise can inject `[click here](http://evil.com)` into Telegram messages sent to users.

**Fix:** Escape Markdown special characters before interpolation:
```python
import re
def escape_md(text: str) -> str:
    return re.sub(r'([*_\[\]()#+-=~|<>])', r'\\\1', text)
```

---

### H-13 — MD5 Used for Hashing Alert/Cache Keys
**Modules:** `src/common`, `src/ml`  
**Files:** `src/common/alerts/alert_evaluator.py` ~L988, `src/common/sentiments/caching/cache_manager.py` ~L98  
**Category:** Weak Cryptography (CWE-328)

```python
return hashlib.md5(payload.encode()).hexdigest()
```

MD5 is collision-vulnerable. Two different alert configurations could generate the same key, causing missed alerts or state corruption.

**Fix:** Replace all MD5 usages with SHA-256.

---

### H-14 — Unbounded Scheduler Thread With No Shutdown Hook
**Module:** `src/ml`  
**File:** `src/ml/future/automated_training_pipeline.py` ~L546  
**Category:** Resource Exhaustion / Improper Shutdown

```python
while True:
    schedule.run_pending()
    time.sleep(60)

scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
```

Daemon threads are killed abruptly at interpreter exit — no cleanup, no checkpoint flush. On unhandled exception, thread silently dies and scheduling stops.

**Fix:** Use a `threading.Event` stop signal, add exception handling inside the loop, and register `atexit` cleanup.

---

### H-15 — Insecure Email Recipient Resolution
**Module:** `src/notification`  
**File:** `src/notification/channels/email_channel.py` ~L194  
**Category:** Broken Object-Level Authorization (CWE-639)

The notification system resolves an arbitrary user ID/telegram ID into an email address without checking whether the caller is authorized to send to that recipient, enabling unsolicited email to any registered user.

**Fix:** Add an authorization check before resolving cross-user recipients.

---

### H-16 — No Validation of Financial Data for NaN/Inf/Negative Prices
**Module:** `src/backtester`, `src/indicators`  
**File:** `src/backtester/optimizer/base_optimizer.py` ~L141  
**Category:** Data Integrity (CWE-20)

```python
df = df.dropna()   # NaN removed, but Inf, negative prices, and price logic violations ignored
```

Corrupted OHLCV data (e.g., close > high, negative volume, Inf values) silently propagates through the ML training pipeline, producing invalid models.

**Fix:** Implement a strict `validate_ohlcv(df)` function called before any pipeline step.

---

## Medium Severity Findings

### M-01 — Exception Messages Returned to API Clients
**Module:** `src/api`  
**File:** `src/api/main.py` (multiple endpoints)

```python
raise HTTPException(status_code=500, detail=str(e))
```

Stack traces or internal paths can appear in API responses. Use generic messages for 500 errors; log details server-side only.

---

### M-02 — JWT Secret Key Has No Minimum Length Enforcement
**Module:** `src/api`  
**File:** `src/api/config.py`

`jwt_secret_key: str` — Pydantic accepts `""` or `"a"`, producing trivially crackable JWTs.

**Fix:** `jwt_secret_key: str = Field(..., min_length=32)`

---

### M-03 — Race Condition in User Registration (TOCTOU)
**Module:** `src/telegram`  
**File:** `src/telegram/handlers/account.py` ~L97  

Check-then-set pattern without locking allows duplicate registrations under concurrent requests from the same user.

**Fix:** Use a DB-level unique constraint and `INSERT ON CONFLICT DO UPDATE`.

---

### M-04 — Telegram Bot Token Partial Logging
**Module:** `src/telegram`  
**File:** `src/telegram/telegram_bot.py` ~L59`

```python
_logger.info("Bot token: %s…", TELEGRAM_BOT_TOKEN[:10])
```

Reveals the token prefix and confirms the format. Remove entirely.

---

### M-05 — SMTP Credentials Fall Back to Hardcoded Values
**Module:** `src/notification`  
**File:** `src/notification/service/config.py` ~L157`

```python
"smtp_password": os.getenv("SMTP_PASSWORD", SMTP_PASSWORD),  # fallback = hardcoded
```

If the env var is absent, the hardcoded value from `config.donotshare` is silently used.

**Fix:** Remove the fallback; raise `ValueError` if the env var is absent.

---

### M-06 — Database URLs Partially Logged
**Module:** `src/scheduler`  
**File:** `src/scheduler/scheduler_service.py` ~L164`

Even the host:port portion of a DB URL leaks network topology to anyone with log access.

**Fix:** Log only `"connected"` or a redacted tag; never any portion of the connection string.

---

### M-07 — `random` Used for Retry Jitter
**Module:** `src/error_handling`  
**File:** `src/error_handling/retry_manager.py` ~L156`

Not security-critical for jitter, but consistent use of `secrets.SystemRandom()` across the codebase reduces risk of misuse in security contexts.

---

### M-08 — Unvalidated Alert Index Parameter
**Module:** `src/api`  
**File:** `src/api/main.py` ~L696`

Negative index values accepted, allowing Python's list-from-end indexing to access unintended alerts.

**Fix:** `if alert_index < 0: raise HTTPException(400)`

---

### M-09 — Email Validation Regex Insufficient
**Module:** `src/telegram`  
**File:** `src/telegram/handlers/account.py` ~L81`

Current regex accepts `test@test..com` and doesn't check 254-char RFC limit. Use `email-validator` library.

---

### M-10 — Alert State JSON Size Limit Allows Silent Data Loss
**Module:** `src/common`  
**File:** `src/common/alerts/alert_evaluator.py` ~L1372`

When state exceeds 10KB, critical crossing-detection state is silently truncated. This can cause missed alerts.

**Fix:** Enforce size limits earlier in the pipeline; log a warning and fail loudly if truncation would occur.

---

### M-11 — No SSRF Protection on Alert Data Fetches
**Module:** `src/common`  
**File:** `src/common/alerts/alert_evaluator.py` ~L349`

`ticker` from alert config is used in data fetch calls without validation. A carefully crafted symbol could trigger internal network requests.

**Fix:** Validate tickers against an alphanumeric whitelist before use.

---

### M-12 — Global Health Monitor Singleton is Not Thread-Safe
**Module:** `src/common`  
**File:** `src/common/health_monitor.py` ~L481`

Double-instantiation possible under concurrent first access. Use a `threading.Lock()` guard.

---

### M-13 — Missing Rate Limiting on Alert Evaluation
**Module:** `src/common`  
**File:** `src/common/alerts/alert_evaluator.py`

No per-user/per-alert rate limit. A user with many alerts can saturate CPU.

---

### M-14 — Unvalidated Config Path Construction
**Module:** `src/ml`  
**File:** `src/ml/pipeline/p04_short_squeeze/config/config_manager.py`

Config file path from caller is not validated against an allowed directory before loading.

---

### M-15 — No Cryptographic Validation on Loaded JSON Indicator Files
**Module:** `src/strategy`  
**File:** `src/strategy/hmm_lstm_strategy.py` ~L267`

JSON files loaded without schema validation; missing keys raise bare `KeyError` caught by outer handler.

---

### M-16 — NaN Propagation in Feature Engineering
**Module:** `src/ml`  
**File:** `src/ml/future/feature_engineering_pipeline.py` ~L482`

```python
X_numeric.fillna(X_numeric.mean())   # fails silently if entire column is NaN
```

An all-NaN column propagates NaN through training.

---

### M-17 — No Rate Limiting on External API Calls in Downloaders
**Module:** `src/data`

Bulk downloads can exhaust broker API rate limits and trigger IP bans.

---

### M-18 — Missing Audit Logging on Sensitive Operations
**Module:** `src/api`

Configuration changes, strategy deletion, and permission changes lack structured audit log entries.

---

## Low Severity Findings

### L-01 — Hardcoded Windows Path as Fallback
**Module:** `src/common`  
**File:** `src/common/common.py` ~L112`

```python
DATA_CACHE_DIR = "c:/data-cache"   # breaks on Linux/macOS
```

Use `Path.home() / ".cache" / "e-trading"` or an environment variable.

---

### L-02 — Cache Files Stored Without Encryption
**Module:** `src/data`  
**File:** `src/data/utils/caching.py`

Parquet/CSV cache files on disk contain market data and strategy signals without encryption. Risk if disk is shared or stolen.

---

### L-03 — Race Condition in Cache Metadata File Writes
**Module:** `src/data`  
**File:** `src/data/utils/caching.py` ~L87`

No file lock around metadata JSON writes; concurrent processes can corrupt the metadata file.

**Fix:** Use `filelock.FileLock`.

---

### L-04 — `shell=True` on Windows in Web UI Subprocess
**Module:** `src/web_ui`  
**File:** `src/web_ui/run_web_ui.py` ~L84`

```python
subprocess.run(['node', '--version'], shell=os.name == 'nt')
```

`shell=True` on Windows enables metacharacter interpretation. Remove it; list-form subprocess does not need it.

---

### L-05 — Warnings Suppressed Globally
**Module:** `src/analytics`  
**File:** `src/analytics/advanced_analytics.py` ~L21`

```python
warnings.filterwarnings('ignore')
```

Hides deprecation notices from dependencies. Scope the filter to a specific category and module.

---

### L-06 — Error Messages Expose Internal Exception Details
**Module:** `src/common`  
**File:** `src/common/health_monitor.py` ~L104`

```python
error_message=f"Database error: {str(e)}"
```

May expose driver errors, table names, or connection details. Use a generic user-facing string.

---

### L-07 — Telegram Chat ID Not Type-Validated
**Module:** `src/notification`  
**File:** `src/notification/channels/telegram_channel.py` ~L143`

`chat_id` from DB is passed directly to Telegram API without checking it is a valid integer or `@username`.

---

### L-08 — Strategy Alert Logging Reveals Signal Details
**Module:** `src/ml`  
**File:** `src/ml/pipeline/p04_short_squeeze/core/alert_engine.py` ~L138`

`reason` field logged at INFO level exposes internal strategy signal rationale. Move to DEBUG.

---

## Architectural Concerns

### A-01 — `config.donotshare` as a Credential Store
All secrets (API keys, DB password, Telegram token, SMTP password, IBKR credentials) are stored in a Python module. This makes secret rotation a code deployment, prevents separate access control for secrets vs. code, and risks accidental commit of the file. Migrate to environment variables or a secrets manager (e.g., HashiCorp Vault, AWS Secrets Manager, Azure Key Vault).

### A-02 — Pickle as Serialization Format for Traded Models
The entire ML inference path depends on pickle for model loading at runtime. This creates a permanent code execution risk tied to the models directory. Adopt ONNX for neural networks and joblib with HMAC signing for scikit-learn/XGBoost models as a migration target.

### A-03 — No Automated Static Security Analysis in CI
There is no evidence of `bandit`, `semgrep`, or equivalent SAST tooling in the pipeline. All of the above findings would be caught automatically with basic SAST rules.

### A-04 — Inconsistent Input Validation Patterns
Some API endpoints use Pydantic `Field(ge=..., le=...)` constraints; others validate inline with `if/raise`; others perform no validation at all. Standardize on Pydantic models for all endpoint parameters.

### A-05 — Missing Security-Focused Unit Tests
No test files audit path traversal, injection, or auth bypass scenarios. Add a `tests/security/` suite with parameterized tests for known-bad inputs.

---

## Remediation Roadmap

### Sprint 1 — Immediate (Week 1)

| ID | Action | Owner |
|----|--------|-------|
| C-01 | Fix path traversal in SPA catch-all | Backend |
| C-02 | Implement bcrypt password hashing | Backend |
| C-03 | Add HMAC signing to all `pickle.load()` calls | ML |
| C-04 | Add `weights_only=True` to all `torch.load()` calls | ML |
| C-05 | Add script path whitelist to scheduler | Infra |
| C-06 | Add path resolution guard to email attachments | Backend |
| H-01 | Make internal API token mandatory | Backend |
| H-03 | Fix relative `.env` path; move hardcoded infra defaults to env; remove hardcoded `TEST_DB_URL` | All |
| H-10 | Replace all bare `except:` clauses | All |

### Sprint 2 — High Priority (Weeks 2–3)

| ID | Action | Owner |
|----|--------|-------|
| H-02 | Tighten CORS to explicit methods/headers | Backend |
| H-04 | Remove TLS disable option from notification config | Backend |
| H-05 | Add admin guard at Telegram handler entry point | Backend |
| H-06 | Replace `random.randint` with `secrets.randbelow` | Backend |
| H-08 | Add Pydantic schema validation for schedule params | Infra |
| H-09 | Add `await process.wait()` after `process.kill()` | Infra |
| H-11 | Tighten login rate limit + add lockout | Backend |
| H-12 | Escape Markdown in all notification builders | Backend |
| H-13 | Replace MD5 with SHA-256 everywhere | All |
| H-16 | Implement `validate_ohlcv()` in data pipeline | Data |

### Sprint 3 — Medium Priority (Month 2)

| ID | Action |
|----|--------|
| M-01–M-18 | Address all medium findings above |
| A-03 | Add `bandit` and `semgrep` to CI pipeline |
| A-04 | Standardize all endpoint params on Pydantic Field constraints |
| A-05 | Create `tests/security/` suite |

### Long-Term (Quarter)

| Action |
|--------|
| A-02: Migrate ML models to ONNX + joblib with HMAC |
| A-01: Adopt a secrets manager for all credentials |
| Penetration test by external party |
| SOC 2 / trading regulatory compliance review |

---

*Generated by Claude Code security audit — 2026-05-10*

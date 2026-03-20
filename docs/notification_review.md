# Notification Module — Architecture Review

## Overview

The notification module implements a **database-centric delivery pipeline**: clients write messages to the DB, and a background service polls and dispatches them through pluggable channel drivers (Telegram, Email, SMS).  The design correctly separates concerns into three layers:

```
Callers → NotificationServiceClient → DB
                                       ↑
                              MessagePoller (polls)
                                       ↓
                             MessageProcessor → Channel Plugins
```

Overall the module is **well above average** in sophistication: it has retry logic, circuit breakers, health reporting, delivery tracking, analytics, archival, and a proper plugin ABC.  The issues below are real but none are show-stoppers for the current scale.

---

## Strengths

| Area | What's good |
|---|---|
| **Plugin ABC** | [NotificationChannel](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py#113-357) in [channels/base.py](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py) is a clean interface: [send_message](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py#149-170), [check_health](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py#171-180), [get_rate_limit](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py#181-190), [supports_feature](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py#191-203) are all correctly abstract. [send_message_with_retry](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py#275-349) is a solid base-class concrete helper. |
| **Delivery result model** | [DeliveryResult](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py#35-63) with [__post_init__](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py#46-53) validation, [is_successful](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py#54-58), [is_retryable](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py#59-63) properties is a well-designed value object. |
| **DB-centric queue** | Writing messages to DB first and polling is the correct pattern for durability at this scale. |
| **Sensitive data filter** | [SensitiveDataFilter](file:///c:/dev/cursor/e-trading/src/notification/logger.py#43-97) in [logger.py](file:///c:/dev/cursor/e-trading/src/notification/logger.py) correctly handles pre-rendered strings (urllib3 pattern), dictionary args, and tuple args. |
| **Graceful shutdown** | [startup()](file:///c:/dev/cursor/e-trading/src/notification/notification_db_centric_bot.py#301-337) / [shutdown()](file:///c:/dev/cursor/e-trading/src/notification/service/main.py#330-349) hooks with `signal.SIGTERM` handling are present. |

---

## Issues — Prioritised

### P1 · Critical

#### P1.1 — Duplicate entry points: [service/main.py](file:///c:/dev/cursor/e-trading/src/notification/service/main.py) vs [notification_db_centric_bot.py](file:///c:/dev/cursor/e-trading/src/notification/notification_db_centric_bot.py)

Both files contain **identical** [MessagePoller](file:///c:/dev/cursor/e-trading/src/notification/notification_db_centric_bot.py#31-141), [HealthReporter](file:///c:/dev/cursor/e-trading/src/notification/service/main.py#139-276), [_register_channel_plugins](file:///c:/dev/cursor/e-trading/src/notification/notification_db_centric_bot.py#282-299), [startup()](file:///c:/dev/cursor/e-trading/src/notification/notification_db_centric_bot.py#301-337), [shutdown()](file:///c:/dev/cursor/e-trading/src/notification/service/main.py#330-349), and [main()](file:///c:/dev/cursor/e-trading/src/data/tests/test_data_downloader_factory.py#89-110) functions.  Any bug fix must be applied twice and the two will inevitably drift.

> **Fix:** Delete [notification_db_centric_bot.py](file:///c:/dev/cursor/e-trading/src/notification/notification_db_centric_bot.py).  It is a stale copy of [service/main.py](file:///c:/dev/cursor/e-trading/src/notification/service/main.py).

---

#### P1.2 — [CircuitBreaker](file:///c:/dev/cursor/e-trading/src/notification/service/client.py#61-104) is not thread-safe / async-safe

```python
# service/client.py  line 71
def call(self, func, *args, **kwargs):
    if self.state == CircuitBreakerState.OPEN:
        if self._should_attempt_reset():
            self.state = CircuitBreakerState.HALF_OPEN   # ← race
```

`failure_count` and [state](file:///c:/dev/cursor/e-trading/src/data/downloader/fmp_data_downloader.py#466-482) are plain instance attributes mutated without a lock.  Under concurrent asyncio tasks this is a data race.  Also [call()](file:///c:/dev/cursor/e-trading/src/notification/service/client.py#71-86) is synchronous but the client is async — it cannot wrap coroutines.

> **Fix:** Add `asyncio.Lock`, make [call()](file:///c:/dev/cursor/e-trading/src/notification/service/client.py#71-86) async, use `await func(...)`.

---

#### P1.3 — Signal handler calls `asyncio.create_task` from a sync context

```python
# service/main.py  line 355
def signal_handler(signum, frame):
    asyncio.create_task(shutdown())   # ← wrong: may have no running loop
```

[signal_handler](file:///c:/dev/cursor/e-trading/src/notification/service/main.py#353-356) is a normal Python signal handler (called in the main thread context, not in the event loop).  `asyncio.create_task` requires a running loop and will raise `RuntimeError` if called this way.

> **Fix:** Use `loop.call_soon_threadsafe(loop.create_task, shutdown())` or `asyncio.get_event_loop().call_soon_threadsafe(...)`.

---

### P2 · Important

#### P2.1 — `ChannelRegistry.get_channel` creates a new instance on every call

```python
# channels/base.py  line 421–425
channel_class = self._channels[channel_name]
instance = channel_class(channel_name, config)   # new instance every time
self._instances[channel_name] = instance         # stored but never reused
```

The comment says "Create new instance each time to ensure fresh config" but this defeats the `_instances` cache entirely.  For channels that open persistent connections (Telegram bot, SMTP), this is expensive and incorrect.

> **Fix:** Check `_instances` first; only create a new instance if the config changed or the instance is not present.  Pass a config hash to detect changes.

---

#### P2.2 — [logger.py](file:///c:/dev/cursor/e-trading/src/notification/logger.py) rotation settings are extreme

```python
MAX_BYTES = 500 * 1024 * 1024  # 500 MB
BACKUP_COUNT = 99
```

This allows up to **50 GB of log files** on disk before the oldest is dropped.  On a VPS or cloud instance this will silently fill the disk.

> **Fix:** Reduce to 10–50 MB / 5–10 backups, or add an external log-rotation policy (logrotate, systemd journal).

---

#### P2.3 — [MessagePoller](file:///c:/dev/cursor/e-trading/src/notification/notification_db_centric_bot.py#31-141) reads module-level `message_processor` global

```python
# line 79 / 130
if message_processor:
    await self._process_messages(messages)
```

[MessagePoller](file:///c:/dev/cursor/e-trading/src/notification/notification_db_centric_bot.py#31-141) communicates with the processor via a module-level global written by [startup()](file:///c:/dev/cursor/e-trading/src/notification/notification_db_centric_bot.py#301-337).  This makes [MessagePoller](file:///c:/dev/cursor/e-trading/src/notification/notification_db_centric_bot.py#31-141) impossible to unit-test in isolation and fragile to import-order issues.

> **Fix:** Pass the processor as a constructor argument: [MessagePoller(processor=mp, ...)](file:///c:/dev/cursor/e-trading/src/notification/notification_db_centric_bot.py#31-141).

---

#### P2.4 — [NotificationServiceClient](file:///c:/dev/cursor/e-trading/src/notification/service/client.py#125-751) hardcodes port re-routing logic

```python
# client.py  lines 165–174
if ":8080" in service_url or ":5003" in service_url:
    service_url = "http://localhost:5003"
if not service_url.endswith(":5003") and "localhost" in service_url:
    service_url = "http://localhost:5003"
```

Silently rewriting the caller's URL is surprising and will cause hard-to-debug failures in non-standard deployments.

> **Fix:** Log a deprecation warning and let the caller decide; or remove if those ports are truly obsolete.

---

### P3 · Minor

#### P3.1 — [validate_config()](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py#136-148) called in `NotificationChannel.__init__` before subclass [__init__](file:///c:/dev/cursor/e-trading/src/data/data_manager.py#883-900)

```python
class NotificationChannel(ABC):
    def __init__(self, channel_name, config):
        ...
        self.validate_config(config)   # called on partially-constructed subclass
```

If a subclass [validate_config](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py#136-148) accesses `self.some_attr` set later in its own [__init__](file:///c:/dev/cursor/e-trading/src/data/data_manager.py#883-900), it will raise `AttributeError`.

> **Fix:** Document clearly that [validate_config](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py#136-148) must not access instance state, or call it at the end of the subclass [__init__](file:///c:/dev/cursor/e-trading/src/data/data_manager.py#883-900).

---

#### P3.2 — `MessageContent.__post_init__` rejects empty [text](file:///c:/dev/cursor/e-trading/src/notification/logger.py#515-567) even when [html](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py#107-111) is present

```python
if not self.text and not self.html:
    raise ValueError(...)
```

This is correct.  However, [split_long_message](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py#228-274) only checks [len(content.text)](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py#219-227), so an HTML-only message always returns `[content]` unchanged — the HTML is never split.  This is a latent bug for email channels with long HTML bodies.

> **Fix:** In [split_long_message](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py#228-274), fall back to splitting [html](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py#107-111) when [text](file:///c:/dev/cursor/e-trading/src/notification/logger.py#515-567) is empty.

---

#### P3.3 — `sys.path` manipulation in multiple files

[client.py](file:///c:/dev/cursor/e-trading/src/notification/service/client.py), [main.py](file:///c:/dev/cursor/e-trading/src/screeners/main.py), and [notification_db_centric_bot.py](file:///c:/dev/cursor/e-trading/src/notification/notification_db_centric_bot.py) all do:
```python
sys.path.append(str(PROJECT_ROOT))
```

This is fragile (appends instead of prepending, may add duplicates) and unnecessary when running as an installed package or with `PYTHONPATH` set.

> **Fix:** Remove these lines; configure `PYTHONPATH` or `pyproject.toml` entry points instead.

---

## Summary Table

| ID | Severity | File | Issue |
|---|---|---|---|
| P1.1 | 🔴 Critical | [notification_db_centric_bot.py](file:///c:/dev/cursor/e-trading/src/notification/notification_db_centric_bot.py) | Exact duplicate of [service/main.py](file:///c:/dev/cursor/e-trading/src/notification/service/main.py) — delete it |
| P1.2 | 🔴 Critical | [service/client.py](file:///c:/dev/cursor/e-trading/src/notification/service/client.py) | [CircuitBreaker](file:///c:/dev/cursor/e-trading/src/notification/service/client.py#61-104) not thread/async-safe |
| P1.3 | 🔴 Critical | [service/main.py](file:///c:/dev/cursor/e-trading/src/notification/service/main.py) | Signal handler calls `create_task` incorrectly |
| P2.1 | 🟠 Important | [channels/base.py](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py) | [ChannelRegistry](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py#359-471) recreates channel instances on every call |
| P2.2 | 🟠 Important | [logger.py](file:///c:/dev/cursor/e-trading/src/notification/logger.py) | 500 MB × 99 files = possible 50 GB disk usage |
| P2.3 | 🟠 Important | [service/main.py](file:///c:/dev/cursor/e-trading/src/notification/service/main.py) | [MessagePoller](file:///c:/dev/cursor/e-trading/src/notification/notification_db_centric_bot.py#31-141) depends on module-level global |
| P2.4 | 🟠 Important | [service/client.py](file:///c:/dev/cursor/e-trading/src/notification/service/client.py) | Silent URL rewriting is surprising |
| P3.1 | 🟡 Minor | [channels/base.py](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py) | [validate_config](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py#136-148) called before subclass init completes |
| P3.2 | 🟡 Minor | [channels/base.py](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py) | [split_long_message](file:///c:/dev/cursor/e-trading/src/notification/channels/base.py#228-274) ignores HTML-only messages |
| P3.3 | 🟡 Minor | Multiple files | `sys.path` mutation at module level |

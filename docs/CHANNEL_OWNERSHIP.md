# Channel Ownership Architecture

## Overview

This document defines the clear separation of responsibilities between the Telegram Bot and Notification Service for message delivery across different channels.

## Architecture Principles

**Single Responsibility**: Each service owns specific channels exclusively
**No Overlap**: Channels are never shared between services
**Database-Centric**: Shared message queue for audit trail and work distribution

---

## Service Responsibilities

### Telegram Bot

**Owns**: Telegram channel **exclusively**

**Responsibilities**:
- Receives all user commands via Telegram
- Handles ALL Telegram message sending (instant + queued)
- Routes email/SMS requests to Notification Service
- Uses database for audit trail and queueing heavy work

**Sending Patterns**:

1. **Instant Commands** (Direct via aiogram)
   - `/start`, `/help`, `/alerts`, `/schedules`, `/feedback`
   - Sends immediately via aiogram API
   - Response time: < 1 second

2. **Queued Commands** (Via database queue)
   - `/report`, `/screener` (heavy processing)
   - Queues message to database with `channels=["telegram"]`
   - Background queue processor polls database and sends via aiogram
   - Processing time: 5-10 seconds

**Implementation**:
- Main file: `src/telegram/telegram_bot.py`
- Queue processor: `src/telegram/services/telegram_queue_processor.py`
- Polls database every 5 seconds for pending Telegram messages

---

### Notification Service

**Owns**: Email and SMS channels **exclusively**

**Responsibilities**:
- Polls database for pending notifications
- Handles Email/SMS delivery ONLY
- **NEVER** sends Telegram messages

**Configuration**:
```python
# src/notification/service/processor.py
enabled_channels = ['email']  # Telegram owned by telegram bot
```

**Email Flow Example**:
1. User runs `/report vt -email`
2. Telegram bot sends Telegram message directly
3. Telegram bot queues email notification with `channels=["email"]`
4. Notification service polls database, finds email notification
5. Notification service sends email via SMTP

**Implementation**:
- Main file: `src/notification/notification_db_centric_bot.py`
- Processor: `src/notification/service/processor.py`
- Channel handlers: `src/notification/channels/`

---

## Database Queue Schema

**Message Structure**:
```json
{
  "id": "msg_123",
  "channels": ["telegram"],  // or ["email"] or ["sms"] or ["telegram", "email"]
  "status": "PENDING",
  "recipient_id": "859865894",
  "content": {
    "title": "VT Report",
    "message": "Analysis results...",
    "attachments": {...}
  },
  "message_metadata": {
    "telegram_chat_id": 859865894,
    "reply_to_message_id": null
  }
}
```

**Channel Filtering**:
- **Telegram Bot**: Queries `WHERE 'telegram' = ANY(channels)`
- **Notification Service**: Queries `WHERE 'email' = ANY(channels) OR 'sms' = ANY(channels)`

---

## Message Flow Diagrams

### Instant Command Flow
```
User → /help
  ↓
Telegram Bot (receives)
  ↓
aiogram.send_message() [Direct]
  ↓
User receives response (< 1s)
```

### Queued Telegram Command Flow
```
User → /report vt
  ↓
Telegram Bot (receives)
  ↓
Heavy processing (analysis, chart generation)
  ↓
Queue to database: channels=["telegram"]
  ↓
Telegram Queue Processor (polls every 5s)
  ↓
aiogram.send_message()
  ↓
Mark as DELIVERED in database
  ↓
User receives report (5-10s)
```

### Multi-Channel Flow (Telegram + Email)
```
User → /report vt -email
  ↓
Telegram Bot (receives)
  ↓
Heavy processing
  ↓
├─ Send Telegram (direct via aiogram) → User receives Telegram
└─ Queue to database: channels=["email"]
     ↓
     Notification Service (polls)
     ↓
     SMTP send
     ↓
     Mark as DELIVERED
     ↓
     User receives email
```

---

## Adding New Channels

### To Add a Channel to Telegram Bot:
1. **Not Recommended** - Telegram bot should only handle Telegram
2. Email/SMS should go through Notification Service

### To Add a Channel to Notification Service:
1. Create channel handler in `src/notification/channels/`
2. Implement `BaseChannel` interface
3. Add to `enabled_channels` in processor
4. Configure in `src/notification/service/config.py`

Example:
```python
# src/notification/channels/sms_channel.py
class SMSChannel(BaseChannel):
    async def send_message(self, recipient, content, message_id, priority):
        # SMS sending logic via Twilio/etc
        pass
```

---

## Troubleshooting

### Duplicate Messages

**Symptom**: User receives same message twice

**Cause**: Both services sending to same channel

**Fix**: Check `enabled_channels` configuration
```bash
# Should show:
# Telegram Bot: ["telegram"]
# Notification Service: ["email"]
```

**Verify**:
```python
# src/notification/service/processor.py
enabled_channels = ['email']  # Should NOT include 'telegram'
```

---

### Messages Not Delivered

**For Telegram Messages**:
1. Check Telegram bot is running
2. Check queue processor is polling: `src/telegram/services/telegram_queue_processor.py`
3. Query database for stuck messages:
   ```sql
   SELECT * FROM messages
   WHERE 'telegram' = ANY(channels)
   AND status = 'PENDING'
   AND created_at < NOW() - INTERVAL '1 minute';
   ```

**For Email Messages**:
1. Check Notification Service is running
2. Check `enabled_channels` includes 'email'
3. Check SMTP configuration
4. Query database for stuck messages:
   ```sql
   SELECT * FROM messages
   WHERE 'email' = ANY(channels)
   AND status = 'PENDING'
   AND created_at < NOW() - INTERVAL '1 minute';
   ```

---

### Slow Delivery

**Telegram**:
- Instant commands should be < 1s
- Queued commands should be < 10s
- Check queue processor polling interval (default: 5s)

**Email**:
- Should be < 30s
- Check notification service polling interval
- Check SMTP server response time

---

## Configuration Reference

### Telegram Bot
```python
# src/telegram/telegram_bot.py

# Direct sending for instant commands
await bot.send_message(chat_id=chat_id, text=text)

# Queueing for heavy commands
message_queue_client.create_message(
    recipient_id=str(chat_id),
    channels=["telegram"],  # Telegram only
    content={"title": title, "message": body}
)
```

### Notification Service
```python
# src/notification/service/processor.py

enabled_channels = ['email']  # Email only (no telegram)

# Notification service filters automatically:
# - Ignores messages with only 'telegram' channel
# - Processes messages with 'email' or 'sms' channels
```

---

## Migration History

### Previous Architecture (Duplicate Messages)
- Both services sent Telegram messages
- Result: Users received duplicates

### Current Architecture (No Duplicates)
- **Telegram Bot**: Owns Telegram exclusively
- **Notification Service**: Owns Email/SMS exclusively
- **Result**: No duplicates, clear ownership

### Migration completed: Phase 1 and Phase 4 of MIGRATION_PLAN.md

---

## References

- **MIGRATION_PLAN.md**: Full migration plan and rationale
- **Message Queue Client**: `src/notification/service/message_queue_client.py`
- **Telegram Queue Processor**: `src/telegram/services/telegram_queue_processor.py`
- **Notification Processor**: `src/notification/service/processor.py`

---

## Summary

| Service | Owned Channels | Sending Method | Use Case |
|---------|---------------|----------------|----------|
| **Telegram Bot** | Telegram | Direct (aiogram) | All Telegram messages |
| **Telegram Bot** | - | Queue → Poll → Send | Heavy processing commands |
| **Notification Service** | Email, SMS | Poll queue → Send | Email/SMS notifications |

**Key Rule**: If it's Telegram, the Telegram Bot handles it. If it's Email/SMS, the Notification Service handles it.

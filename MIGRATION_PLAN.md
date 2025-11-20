# Migration Plan: Hybrid Architecture with Proper Channel Separation

## Overview
Implement Option 3 - clear separation where Telegram Bot owns all Telegram delivery and Notification Service handles Email/SMS only.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER COMMANDS                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      TELEGRAM BOT                                │
│  • Receives all user commands                                   │
│  • Handles ALL Telegram sending (instant + queued)              │
│  • Routes email/SMS to Notification Service                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────┴─────────┐
                    │                   │
              ┌─────▼──────┐    ┌──────▼──────┐
              │  INSTANT   │    │   QUEUED    │
              │  COMMANDS  │    │  COMMANDS   │
              └─────┬──────┘    └──────┬──────┘
                    │                  │
                    │                  ↓
                    │         ┌────────────────┐
                    │         │   DATABASE     │
                    │         │  • Audit logs  │
                    │         │  • Queue jobs  │
                    │         └────────┬───────┘
                    │                  │
                    ↓                  ↓
              ┌──────────────────────────────┐
              │    aiogram (Telegram API)    │
              │  • Direct sends (instant)    │
              │  • Queued sends (processed)  │
              └──────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  NOTIFICATION SERVICE                            │
│  • Polls database for pending notifications                     │
│  • Handles ONLY Email + SMS channels                            │
│  • NO Telegram sending                                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────┴─────────┐
                    │                   │
              ┌─────▼──────┐    ┌──────▼──────┐
              │   EMAIL    │    │     SMS     │
              │  CHANNEL   │    │   CHANNEL   │
              └────────────┘    └─────────────┘
```

## Task Breakdown

### Phase 1: Disable Telegram in Notification Service (Quick Fix)

**Goal**: Stop duplicate messages immediately

**Tasks**:

1. **Update notification service processor initialization**
   - File: `src/notification/notification_db_centric_bot.py`
   - Change: Only initialize email/SMS channels
   - Lines to modify: Where `enabled_channels` is defined

   ```python
   # Change from:
   enabled_channels = ["telegram", "email", "sms"]

   # To:
   enabled_channels = ["email"]  # Only email for now (SMS when ready)
   ```

2. **Test immediately**
   - Restart notification service
   - Run `/report vt` command
   - Verify: Only ONE message received (from telegram bot)
   - Expected result: No duplicates

**Estimated time**: 5 minutes
**Priority**: CRITICAL - do this first

---

### Phase 2: Document Current Behavior (Establish Baseline)

**Goal**: Understand exactly which commands use which paths

**Tasks**:

3. **Audit all command handlers in telegram bot**
   - File: `src/telegram/telegram_bot.py`
   - Create a mapping document of:
     - Commands that send directly (instant)
     - Commands that use notification queue
     - Commands that support `-email` flag

   **Expected mapping**:
   ```
   INSTANT (Direct aiogram):
   - /start
   - /help
   - /register
   - /verify
   - /alerts (list)
   - /schedules (list)
   - /feedback
   - /feature

   QUEUED (Via notification service):
   - /report (heavy processing, supports -email)
   - /screener (heavy processing, supports -email)

   SCHEDULED (Notification service):
   - Alert triggers (from database schedules)
   - Report schedules
   ```

4. **Document email flow**
   - Map how `-email` flag currently works
   - Identify where email notifications are queued
   - Expected flow:
     - User runs `/report vt -email`
     - Report generates (telegram bot)
     - Telegram result sent directly
     - Email queued to database with channels=["email"]
     - Notification service picks up and sends email

**Estimated time**: 30 minutes
**Priority**: HIGH - understand before changing

---

### Phase 3: Configuration Updates

**Goal**: Formalize channel ownership in configuration

**Tasks**:

5. **Create channel configuration documentation**
   - File: Create `docs/CHANNEL_OWNERSHIP.md`
   - Content:

   ```markdown
   # Channel Ownership Architecture

   ## Telegram Bot
   **Owns**: Telegram channel exclusively
   - All Telegram messages sent via aiogram
   - Direct sends for instant commands
   - Queued sends for heavy commands (after processing)
   - Uses database for audit trail and queueing work

   ## Notification Service
   **Owns**: Email and SMS channels exclusively
   - Polls database for pending notifications
   - Handles Email/SMS delivery
   - NEVER sends Telegram messages

   ## Database
   - Shared queue for all channels
   - Messages marked with channels: ["telegram"] | ["email"] | ["sms"] | ["email", "telegram"]
   - Each service filters to its owned channels
   ```

6. **Update notification service config**
   - File: `src/notification/service/config.py`
   - Add documentation comments clarifying channel usage:

   ```python
   class ChannelConfig(BaseSettings):
       """Channel configuration.

       NOTE: Telegram channel is configured here for potential future use,
       but currently ONLY the Telegram Bot should send Telegram messages.
       The notification service uses ONLY email/SMS channels.
       """
   ```

7. **Add environment variable for channel selection**
   - File: `src/notification/service/config.py`
   - Add new field:

   ```python
   enabled_channels: List[str] = Field(
       default_factory=lambda: ["email"],  # Telegram bot handles telegram
       env="NOTIFICATION_ENABLED_CHANNELS",
       description="Channels enabled for this notification service instance"
   )
   ```

**Estimated time**: 20 minutes
**Priority**: MEDIUM - helps prevent future confusion

---

### Phase 4: Enhance Telegram Bot Queue Processing

**Goal**: Make telegram bot properly handle queued messages from database

**Tasks**:

8. **Check if telegram bot polls for pending Telegram messages**
   - File: `src/telegram/telegram_bot.py`
   - Search for: database polling, message queue checking
   - **Current status**: Unknown - need to verify

9. **If NO polling exists, add queue processor**
   - Create: `src/telegram/services/telegram_queue_processor.py`
   - Responsibilities:
     - Poll database every 5 seconds for messages with channels=["telegram"]
     - Filter to PENDING status
     - Send via aiogram
     - Mark as DELIVERED/FAILED in database

   ```python
   class TelegramQueueProcessor:
       """
       Processes Telegram messages from database queue.
       Used for heavy commands like /report and /screener.
       """

       async def poll_and_process(self):
           """Poll database for pending Telegram messages."""
           # Get messages where channels contains "telegram"
           # Send via aiogram
           # Update status in database
   ```

10. **Integrate queue processor into telegram bot**
    - File: `src/telegram/telegram_bot.py`
    - Add background task:

    ```python
    async def start_queue_processor():
        processor = TelegramQueueProcessor()
        asyncio.create_task(processor.poll_and_process())
    ```

11. **Update /report and /screener commands**
    - **Current**: Queue to database with channels=["telegram"]
    - **Change**: After queueing, telegram bot's queue processor will handle delivery
    - **No change needed if already queueing correctly**

**Estimated time**: 2 hours
**Priority**: HIGH - needed for proper separation

---

### Phase 5: Testing & Validation

**Goal**: Verify all paths work correctly without duplicates

**Tasks**:

12. **Test instant commands**
    - Commands: `/start`, `/help`, `/alerts`, `/schedules`
    - Expected: Immediate response via aiogram
    - Expected: No database queue entry (unless for audit)
    - Verify: Single message received

13. **Test queued Telegram commands**
    - Commands: `/report vt`, `/screener SP500`
    - Expected: Processing message, then result message
    - Expected: Database entry created with channels=["telegram"]
    - Expected: Telegram bot polls and sends
    - Verify: Single message received (no duplicate)

14. **Test email flag**
    - Commands: `/report vt -email`, `/screener SP500 -email`
    - Expected: Telegram message sent by telegram bot
    - Expected: Email queued with channels=["email"]
    - Expected: Notification service sends email
    - Verify: One Telegram message, one email

15. **Test multi-channel**
    - If implemented: channels=["telegram", "email"]
    - Expected: Telegram bot sends Telegram
    - Expected: Notification service sends email
    - Verify: Both received, no duplicates

16. **Test scheduled alerts**
    - Create test alert with notification
    - Expected: Notification service handles (if email)
    - Expected: Alert system triggers correctly
    - Verify: Single notification per channel

17. **Load testing**
    - Send 10 `/report` commands rapidly
    - Verify: All processed without duplicates
    - Check: Database queue handling
    - Check: No race conditions

**Estimated time**: 1.5 hours
**Priority**: CRITICAL - must verify before production

---

### Phase 6: Monitoring & Observability

**Goal**: Add visibility into channel routing

**Tasks**:

18. **Add channel routing metrics**
    - File: `src/telegram/telegram_bot.py`
    - Add logging:

    ```python
    _logger.info("Routing message %s to channels: %s", msg_id, channels)
    ```

19. **Add notification service channel filter logging**
    - File: `src/notification/notification_db_centric_bot.py`
    - Log which messages are skipped:

    ```python
    _logger.debug("Skipping message %s - channel %s not in enabled_channels %s",
                  msg_id, msg_channels, enabled_channels)
    ```

20. **Create dashboard query**
    - SQL query to show message distribution:

    ```sql
    SELECT
        channels,
        status,
        COUNT(*) as count,
        AVG(EXTRACT(EPOCH FROM (delivered_at - created_at))) as avg_delivery_time_sec
    FROM messages
    WHERE created_at > NOW() - INTERVAL '24 hours'
    GROUP BY channels, status
    ORDER BY channels, status;
    ```

21. **Add health check endpoint**
    - Verify both services are running
    - Show enabled channels per service
    - Endpoint: `/health/channels`

    ```json
    {
      "telegram_bot": {
        "status": "healthy",
        "channels": ["telegram"],
        "queue_size": 3
      },
      "notification_service": {
        "status": "healthy",
        "channels": ["email"],
        "queue_size": 1
      }
    }
    ```

**Estimated time**: 1 hour
**Priority**: MEDIUM - helpful for operations

---

### Phase 7: Documentation & Handoff

**Goal**: Document the new architecture for future developers

**Tasks**:

22. **Create architecture documentation**
    - File: `docs/NOTIFICATION_ARCHITECTURE.md`
    - Sections:
      - Overview of hybrid approach
      - Channel ownership table
      - Message flow diagrams
      - When to use each path
      - Adding new channels
      - Troubleshooting duplicates

23. **Update README**
    - File: `README.md` (project root)
    - Add section on notification architecture
    - Link to detailed docs

24. **Create runbook**
    - File: `docs/runbooks/NOTIFICATION_ISSUES.md`
    - Common issues:
      - Duplicate messages → Check enabled_channels
      - Messages not delivered → Check service ownership
      - Slow delivery → Check queue processor
    - Debug steps
    - Log locations

25. **Code comments**
    - Add comments to key decision points:
      - Where channels are selected
      - Why telegram bot handles telegram
      - Why notification service skips telegram

**Estimated time**: 1.5 hours
**Priority**: MEDIUM - important for maintainability

---

## Summary Timeline

| Phase | Description | Time | Priority |
|-------|-------------|------|----------|
| 1 | Disable Telegram in Notification Service | 5 min | CRITICAL |
| 2 | Document Current Behavior | 30 min | HIGH |
| 3 | Configuration Updates | 20 min | MEDIUM |
| 4 | Enhance Telegram Bot Queue Processing | 2 hrs | HIGH |
| 5 | Testing & Validation | 1.5 hrs | CRITICAL |
| 6 | Monitoring & Observability | 1 hr | MEDIUM |
| 7 | Documentation & Handoff | 1.5 hrs | MEDIUM |
| **TOTAL** | | **~7 hours** | |

## Rollout Strategy

### Immediate (Do Now)
- ✅ Phase 1, Task 1-2: Disable Telegram in notification service (5 min)
- Test: Verify no duplicates

### Short-term (This Week)
- Phase 2: Document current behavior
- Phase 4: Add telegram queue processor (if needed)
- Phase 5: Full testing

### Medium-term (Next Week)
- Phase 3: Configuration cleanup
- Phase 6: Monitoring
- Phase 7: Documentation

## Rollback Plan

If issues occur:

1. **Rollback Phase 1**: Re-enable Telegram in notification service
   - Set `enabled_channels = ["telegram", "email"]`
   - Accept duplicates temporarily

2. **Database cleanup**: Clear stuck messages
   ```sql
   UPDATE messages
   SET status = 'FAILED'
   WHERE status = 'PENDING'
   AND created_at < NOW() - INTERVAL '1 hour';
   ```

3. **Service restart**: Restart both services with original config

## Success Criteria

✅ **Zero duplicate messages** for all commands
✅ **Instant commands respond < 1 second**
✅ **Queued commands process within 10 seconds**
✅ **Email notifications delivered via notification service**
✅ **All tests passing**
✅ **Clear ownership documented**
✅ **Monitoring shows correct routing**

---

## Decision Rationale

### Why Option 3 (Hybrid with Separation)?

1. **User Experience First**: Interactive commands must be instant. A 5-10 second queue delay for `/help` would feel broken.

2. **Right Tool for the Job**:
   - Telegram bot already runs aiogram - use it for Telegram
   - Notification service specializes in email/SMS

3. **Original Design Intent**: "Interactive commands sent by telegram bot to be instant" - this is correct architectural thinking.

4. **The Duplicate Issue**: Not a design flaw, but a configuration issue. Notification service shouldn't send Telegram at all.

5. **Sophistication**: Hybrid approach is MORE sophisticated than "one pipe for everything" - it's optimized for different operation types.

---

## Current Status

### Fixed Issues
- ✅ Missing `telegram_chat_id` parameter - now passed correctly
- ✅ Base64 attachment decoding - now handles dict format from database
- ✅ Comprehensive logging added for debugging

### Remaining Issue
- ❌ Duplicate messages - notification service also sending Telegram
- **Fix**: Phase 1, Task 1 (5 minutes)

---

## Next Steps

**START HERE**: Implement Phase 1, Task 1-2 to stop duplicates immediately.

This is a 5-minute fix that solves the immediate problem while planning the full migration.

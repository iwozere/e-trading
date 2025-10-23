# Email Flag Feature Documentation

## Overview

The Telegram bot now supports an `-email` flag that can be added to any interactive command to receive the response via email in addition to the immediate Telegram response. The system uses a hybrid approach for reliability and supports attachments.

## How It Works

1. **Immediate Response**: The command is processed immediately and the response is sent to Telegram as usual
2. **Email Notification**: If the `-email` flag is present, the same response is also sent to the user's registered email address via the notification service
3. **User Verification**: The user must have a verified email address registered with the bot for email notifications to work

## Supported Commands

All interactive commands now support the `-email` flag:

### Basic Commands
- `/help -email` - Send help text to email
- `/info -email` - Send user info to email
- `/start -email` - Send welcome message to email

### User Management Commands
- `/register email@example.com -email` - Send registration confirmation to email
- `/verify CODE -email` - Send verification result to email
- `/language en -email` - Send language change confirmation to email
- `/request_approval -email` - Send approval request confirmation to email

### Alert Management Commands
- `/alerts -email` - Send alerts list to email
- `/alerts add BTCUSDT 50000 above -email` - Send alert creation confirmation to email
- `/alerts edit 1 60000 below -email` - Send alert edit confirmation to email
- `/alerts delete 1 -email` - Send alert deletion confirmation to email

### Schedule Management Commands
- `/schedules -email` - Send schedules list to email
- `/schedules add AAPL 09:00 -email` - Send schedule creation confirmation to email
- `/schedules edit 1 10:00 -email` - Send schedule edit confirmation to email
- `/schedules delete 1 -email` - Send schedule deletion confirmation to email

### Admin Commands
- `/admin users -email` - Send user list to email
- `/admin approve user123 -email` - Send approval confirmation to email
- `/admin broadcast "message" -email` - Send broadcast confirmation to email

### Feedback Commands
- `/feedback "Great bot!" -email` - Send feedback confirmation to email
- `/feature "Add dark mode" -email` - Send feature request confirmation to email

## Technical Implementation

### Command Parsing
The `-email` flag is parsed by the `EnterpriseCommandParser` and stored in the `parsed.args["email"]` field as a boolean value.

### Hybrid Email Notification Flow
The system uses a hybrid approach implemented in the `NotificationServiceClient` for maximum reliability:

1. **Command Execution**: Command is executed immediately and response sent to Telegram
2. **Email Flag Check**: If `-email` flag is present, email notification process begins
3. **User Validation**: User's email address is retrieved from the database
4. **NotificationServiceClient Call**: Single call to `notification_client.send_notification()`
5. **Automatic Fallback**: Client handles HTTP API → Direct DB fallback transparently
6. **Error Handling**: If both methods fail, error is logged but main command succeeds

### Service Startup Resilience
**Problem Solved**: Previously, if the Telegram bot started before the notification service API, email notifications would fail silently with no database entries.

**Solution**: `NotificationServiceClient` implements automatic fallback:
- **HTTP API Available**: Uses standard notification service API (preferred)
- **HTTP API Unavailable**: Automatically falls back to direct database insertion
- **Transparent to Consumers**: All services using the client benefit from this resilience
- **Both Methods**: Notification processor will eventually process the queued messages

### Attachment Support
The system now supports sending pictures and documents via email:

#### Attachment Handling Flow
1. **Extraction**: `extract_attachments_from_telegram_message()` extracts files from Telegram
2. **Processing**: Files are converted to base64 for JSON storage or stored as file paths
3. **Database Storage**: Attachments stored in `content.attachments` field as JSON
4. **Email Delivery**: Notification processor converts back to binary for email delivery

#### Supported Attachment Types
- **Photos**: JPEG, PNG, WebP images from Telegram
- **Documents**: Any file type uploaded to Telegram
- **Stickers**: Converted to WebP format
- **File Paths**: References to files on disk

### Error Handling & Resilience
- **Service Unavailable**: Automatic fallback to direct database insertion
- **Attachment Failures**: Logged but don't prevent text notification
- **Database Failures**: Comprehensive error logging and graceful degradation
- **User Errors**: Helpful messages when email is not registered

## Usage Examples

### Basic Usage
```
/help -email
```
Response: Help text sent immediately to Telegram + email notification sent to registered email

### Complex Commands
```
/alerts add BTCUSDT 50000 above -email
```
Response: Alert created and confirmation sent to both Telegram and email

### Admin Commands
```
/admin users -email
```
Response: User list sent to Telegram + email notification with user list

## Email Notification Format

Email notifications are sent with:
- **Subject**: "Telegram Bot - [COMMAND] Command Response"
- **Content**: The same text that was sent to Telegram
- **Metadata**: Command name, user ID, timestamp, source information

## Prerequisites

For email notifications to work:
1. User must be registered: `/register email@example.com`
2. Email must be verified: `/verify CODE`
3. Notification service must be running and accessible

## Error Messages

### No Verified Email
```
📧 Email notification requested but no verified email found. Use /register to set up email notifications.
```

### Service Unavailable
Email notification failures are logged but don't generate user-visible errors to avoid disrupting the main command flow.

## Configuration

The feature uses the existing notification service configuration:
- SMTP settings for email delivery
- Notification service URL and credentials
- Email templates and formatting

## Testing

Use the test script to verify command parsing:
```bash
python src/telegram/test_email_flag.py
```

## Implementation Files

### Core Files
- `src/telegram/telegram_bot.py` - Main bot with simplified email notification calls
- `src/telegram/screener/immediate_handlers.py` - Command handlers with email support
- `src/telegram/command_parser.py` - Command parsing with email flag support

### Supporting Files
- `src/notification/service/client.py` - **Notification service client with automatic fallback logic**
- `src/data/db/services/notification_service.py` - Direct database service used by fallback
- `src/data/db/repos/repo_notification.py` - Database repository for notifications

## Future Enhancements

1. **Email Templates**: Custom email templates for different command types
2. **Email Preferences**: User preferences for email notification frequency
3. **Rich Email Content**: HTML emails with formatting and attachments
4. **Batch Notifications**: Grouping multiple notifications into digest emails
5. **Email-Only Commands**: Commands that only send responses via email

## Service Startup Scenarios

### Scenario 1: Normal Startup (Both Services Running)
```
1. Start notification service API
2. Start Telegram bot
3. User runs `/help -email`
Result: ✅ HTTP API used, immediate processing
```

### Scenario 2: Telegram Bot Starts First
```
1. Start Telegram bot (notification service not running)
2. Wait 5 minutes
3. Start notification service API
4. Wait 10 minutes
5. User runs `/help -email`
Result: ✅ HTTP API fails, fallback to direct DB insertion
```

### Scenario 3: Database Issues
```
1. Both services running
2. Database connection issues
3. User runs `/help -email`
Result: ⚠️ Both methods fail, error logged, Telegram command succeeds
```

## Attachment Examples

### Sending Command with Picture
```
User sends: Photo + "/report AAPL -email"
Result: 
- Telegram: Immediate report response
- Email: Report text + attached photo
- Database: Entry in msg_messages with base64 encoded image
```

### Supported Attachment Scenarios
```python
# Photo attachment
attachments = {
    "chart_AAPL.jpg": b"<binary_image_data>"
}

# Document attachment  
attachments = {
    "report.pdf": "/path/to/report.pdf"
}

# Multiple attachments
attachments = {
    "screenshot.png": b"<image_data>",
    "data.csv": "/path/to/data.csv"
}
```

## Database Schema Impact

### msg_messages Table
```sql
-- Attachments stored in content JSON field
content: {
    "title": "Telegram Bot - HELP Command Response",
    "message": "Welcome to the bot...",
    "attachments": {
        "photo_123.jpg": {
            "data": "base64_encoded_data",
            "type": "base64",
            "size": 45678
        }
    }
}

-- Metadata includes attachment info
message_metadata: {
    "command": "help",
    "has_attachments": true,
    "attachment_count": 1,
    "fallback_method": "direct_db"  // Only present if fallback was used
}
```

## Troubleshooting

### Email Not Received
1. Check if user email is verified: `/info`
2. Verify notification service is running
3. Check SMTP configuration
4. Review notification service logs
5. **Check msg_messages table** for queued entries

### Service Startup Issues
1. **Check Database**: Look for entries in `msg_messages` table
2. **Fallback Indicators**: Look for `fallback_method: "direct_db"` in metadata
3. **HTTP API Logs**: Check notification service client logs for connection errors
4. **Database Logs**: Check for direct insertion success/failure

### Attachment Issues
1. **Size Limits**: Check if attachments exceed database JSON limits
2. **Format Support**: Verify attachment type is supported
3. **Storage Space**: Ensure sufficient database storage for base64 data
4. **Processing Logs**: Check notification processor logs for attachment handling

### Command Not Parsing Email Flag
1. Ensure command is defined in `COMMAND_SPECS` with email parameter
2. Check command syntax (space before `-email`)
3. Verify command parser is working with test script

### Service Integration Issues
1. Check notification service connectivity
2. Verify service authentication
3. Review error logs for specific failures
4. **New**: Check for fallback database entries when HTTP API fails
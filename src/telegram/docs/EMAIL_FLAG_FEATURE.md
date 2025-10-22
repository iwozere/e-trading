# Email Flag Feature Documentation

## Overview

The Telegram bot now supports an `-email` flag that can be added to any interactive command to receive the response via email in addition to the immediate Telegram response.

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

### Email Notification Flow
1. Command is executed immediately and response sent to Telegram
2. If `-email` flag is present, `_send_email_notification_for_command()` is called
3. User's email address is retrieved from the database
4. Email notification is sent via the notification service
5. If user has no verified email, a Telegram message is sent explaining how to register

### Error Handling
- Email notification failures do not affect the main command execution
- If notification service is unavailable, the error is logged but the command succeeds
- If user has no verified email, a helpful message is sent via Telegram

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
- `src/telegram/telegram_bot.py` - Main bot with email notification utility
- `src/telegram/screener/immediate_handlers.py` - Command handlers with email support
- `src/telegram/command_parser.py` - Command parsing with email flag support

### Supporting Files
- `src/notification/service/client.py` - Notification service client
- `src/telegram/test_email_flag.py` - Test script for email flag parsing

## Future Enhancements

1. **Email Templates**: Custom email templates for different command types
2. **Email Preferences**: User preferences for email notification frequency
3. **Rich Email Content**: HTML emails with formatting and attachments
4. **Batch Notifications**: Grouping multiple notifications into digest emails
5. **Email-Only Commands**: Commands that only send responses via email

## Troubleshooting

### Email Not Received
1. Check if user email is verified: `/info`
2. Verify notification service is running
3. Check SMTP configuration
4. Review notification service logs

### Command Not Parsing Email Flag
1. Ensure command is defined in `COMMAND_SPECS` with email parameter
2. Check command syntax (space before `-email`)
3. Verify command parser is working with test script

### Service Integration Issues
1. Check notification service connectivity
2. Verify service authentication
3. Review error logs for specific failures
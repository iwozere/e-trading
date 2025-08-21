# Manual Testing Guide for Telegram Screener Bot

This guide provides step-by-step instructions for manually testing all features of the Telegram Screener Bot. Follow these instructions to thoroughly test the bot's functionality.

## Quick Start - Required Scripts

**Before starting any testing, launch these scripts:**

```bash
# Terminal 1: Start main bot (REQUIRED for all testing)
cd /path/to/e-trading
python src/frontend/telegram/bot.py

# Terminal 2: Start admin panel (REQUIRED for Phase 6 testing)
cd /path/to/e-trading
python src/frontend/telegram/screener/admin_panel.py

# Terminal 3: Create admin user (ONE-TIME setup)
cd /path/to/e-trading
python src/util/create_admin.py YOUR_TELEGRAM_USER_ID your.email@example.com
```

**Service Status Check:**
- Bot API: http://localhost:8080/api/test
- Admin Panel: http://localhost:5000/login

## Prerequisites

### 1. Environment Setup
Before testing, ensure you have:

1. **Bot Token**: A valid Telegram bot token from [@BotFather](https://t.me/botfather)
2. **Environment Variables**: Set up in your `.env` file or environment:
   ```bash
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   SMTP_USER=your_email@gmail.com
   SMTP_PASSWORD=your_app_password
   WEBGUI_LOGIN=admin
   WEBGUI_PASSWORD=your_admin_password
   WEBGUI_PORT=5000
   ```
3. **Database**: SQLite database will be auto-created on first run
4. **Dependencies**: All Python dependencies installed

### 2. Bot Setup
1. Create a bot with [@BotFather](https://t.me/botfather)
2. Get the bot token and add it to your environment variables
3. Start a conversation with your bot in Telegram

## Required Scripts and Services

### Core Services to Launch

#### 1. Main Telegram Bot (Required for all testing)
```bash
# Start the main Telegram bot
cd /path/to/e-trading
python src/frontend/telegram/bot.py
```
**Purpose**: Handles all Telegram commands and user interactions
**Port**: 8080 (HTTP API for admin panel integration)
**Status**: Must be running for all Telegram bot testing

#### 2. Web Admin Panel (Required for Phase 6 testing)
```bash
# Start the web-based admin panel
cd /path/to/e-trading
python src/frontend/telegram/screener/admin_panel.py
```
**Purpose**: Web interface for admin functions, user management, and system monitoring
**Port**: 5000 (configurable via WEBGUI_PORT)
**Status**: Required for Phase 6 (Web Admin Panel) testing

#### 3. Admin User Creation Script (One-time setup)
```bash
# Create admin user (run once before testing)
cd /path/to/e-trading
python src/util/create_admin.py YOUR_TELEGRAM_USER_ID your.email@example.com
```
**Purpose**: Creates admin user with full privileges
**Status**: Run once before testing admin features

### Optional Testing Scripts

#### 4. Fundamental Screener Test Script
```bash
# Test screener functionality independently
cd /path/to/e-trading
python src/frontend/telegram/screener/tests/test_screener.py
```
**Purpose**: Test screener functionality without full bot
**Status**: Optional - for debugging screener issues

#### 5. Notification Commands Test Script
```bash
# Run automated tests for notification commands
cd /path/to/e-trading
python -m pytest src/frontend/telegram/screener/tests/test_notifications_commands.py -v
```
**Purpose**: Automated testing of notification command processing
**Status**: Optional - for regression testing

### Service Management

#### Starting All Services
```bash
# Terminal 1: Start main bot
cd /path/to/e-trading
python src/frontend/telegram/bot.py

# Terminal 2: Start admin panel (for Phase 6 testing)
cd /path/to/e-trading
python src/frontend/telegram/screener/admin_panel.py
```

#### Service Dependencies
- **Bot depends on**: Database (auto-created), environment variables
- **Admin Panel depends on**: Bot API (port 8080), database, admin credentials
- **Both services share**: Same SQLite database file

#### Health Checks
```bash
# Check if bot API is running
curl http://localhost:8080/api/test

# Check if admin panel is running
curl http://localhost:5000/login
```

### Troubleshooting Service Startup

#### Common Issues:
1. **Port already in use**:
   ```bash
   # Check what's using the port
   netstat -tulpn | grep :8080  # For bot API
   netstat -tulpn | grep :5000  # For admin panel
   
   # Kill process if needed
   kill -9 <PID>
   ```

2. **Database permissions**:
   ```bash
   # Ensure write permissions to db directory
   chmod 755 db/
   chmod 644 db/*.sqlite3
   ```

3. **Environment variables not loaded**:
   ```bash
   # Check if variables are set
   echo $TELEGRAM_BOT_TOKEN
   echo $SMTP_USER
   echo $WEBGUI_LOGIN
   ```

#### Service Logs
- **Bot logs**: Check console output for errors
- **Admin panel logs**: Check console output for Flask errors
- **Database logs**: Check for SQLite errors in console

## Testing Structure

### Phase 1: Basic Bot Functionality
### Phase 2: User Registration and Verification
### Phase 3: Core Commands (Public)
### Phase 4: Restricted Commands (Requires Approval)
### Phase 5: Admin Commands
### Phase 6: Web Admin Panel
### Phase 7: Advanced Features
### Phase 8: Error Handling and Edge Cases

---

## Phase 1: Basic Bot Functionality

### Test 1.1: Bot Startup and Basic Commands

**Objective**: Verify the bot starts correctly and responds to basic commands.

**Steps**:
1. **Start the bot** (if not already running):
   ```bash
   cd /path/to/e-trading
   python src/frontend/telegram/bot.py
   ```
   **Note**: The bot must be running for all Telegram testing. See "Required Scripts and Services" section above.

2. **Test `/start` command**:
   - Send `/start` to your bot
   - **Expected**: Welcome message with bot description
   - **Verify**: Message appears in your chat (not admin chat)

3. **Test `/help` command**:
   - Send `/help` to your bot
   - **Expected**: Comprehensive help message with all available commands
   - **Verify**: All command categories are listed (Basic, Report, Alert, Schedule, Admin)

4. **Test case-insensitive commands**:
   - Send `/START`, `/Start`, `/start`
   - Send `/HELP`, `/Help`, `/help`
   - **Expected**: All variations work the same way

**Success Criteria**: Bot responds to all basic commands with appropriate messages.

### Test 1.2: Unknown Command Handling

**Objective**: Verify the bot handles unknown commands gracefully.

**Steps**:
1. Send `/unknowncommand` to your bot
2. Send `/nonexistent` to your bot
3. Send `/` (just slash) to your bot

**Expected**: Bot responds with helpful message explaining the command is unknown and suggests using `/help`

**Success Criteria**: Bot provides helpful error messages for unknown commands.

---

## Phase 2: User Registration and Verification

### Test 2.1: User Registration

**Objective**: Test the email registration process.

**Steps**:
1. **Test registration with valid email**:
   - Send `/register your.email@example.com` to your bot
   - **Expected**: Confirmation message that verification code was sent
   - **Verify**: Check your email for a 6-digit verification code

2. **Test registration with invalid email**:
   - Send `/register invalid-email` to your bot
   - **Expected**: Error message about invalid email format

3. **Test registration without email**:
   - Send `/register` to your bot
   - **Expected**: Error message asking for email address

4. **Test duplicate registration**:
   - Register the same email again
   - **Expected**: New verification code sent, old code invalidated

**Success Criteria**: Registration process works correctly with proper validation.

### Test 2.2: Email Verification

**Objective**: Test the email verification process.

**Steps**:
1. **Test verification with correct code**:
   - Use the verification code from your email
   - Send `/verify 123456` (replace with actual code)
   - **Expected**: Success message confirming email verification

2. **Test verification with incorrect code**:
   - Send `/verify 000000` (wrong code)
   - **Expected**: Error message about invalid code

3. **Test verification without code**:
   - Send `/verify` to your bot
   - **Expected**: Error message asking for verification code

4. **Test expired code**:
   - Wait 1 hour after receiving code
   - Try to verify with expired code
   - **Expected**: Error message about expired code

**Success Criteria**: Verification process works correctly with proper error handling.

### Test 2.3: User Information

**Objective**: Test the user information display.

**Steps**:
1. **Before registration**:
   - Send `/info` to your bot
   - **Expected**: Shows unregistered status

2. **After registration but before verification**:
   - Register with email but don't verify
   - Send `/info` to your bot
   - **Expected**: Shows registered but unverified status

3. **After verification**:
   - Verify your email
   - Send `/info` to your bot
   - **Expected**: Shows verified status but not approved

**Success Criteria**: User information displays correctly at each stage.

---

## Phase 3: Core Commands (Public)

### Test 3.1: Language Settings

**Objective**: Test language preference settings.

**Steps**:
1. **Test language change**:
   - Send `/language en` to your bot
   - **Expected**: Confirmation message about language change

2. **Test invalid language**:
   - Send `/language invalid` to your bot
   - **Expected**: Error message about unsupported language

3. **Test language without parameter**:
   - Send `/language` to your bot
   - **Expected**: Error message asking for language code

**Success Criteria**: Language settings work correctly.

### Test 3.2: Feedback and Feature Requests

**Objective**: Test feedback and feature request functionality.

**Steps**:
1. **Test feedback submission**:
   - Send `/feedback This bot is great!` to your bot
   - **Expected**: Confirmation message thanking for feedback

2. **Test feature request**:
   - Send `/feature Add dark mode support` to your bot
   - **Expected**: Confirmation message about feature request received

3. **Test empty feedback**:
   - Send `/feedback` to your bot
   - **Expected**: Error message asking for feedback text

**Success Criteria**: Feedback and feature requests are properly recorded.

---

## Phase 4: Restricted Commands (Requires Approval)

### Test 4.1: Approval Request Process

**Objective**: Test the admin approval workflow.

**Steps**:
1. **Request approval**:
   - Send `/request_approval` to your bot
   - **Expected**: Confirmation message that approval request was sent

2. **Test restricted commands before approval**:
   - Try `/report AAPL` (should fail)
   - Try `/alerts` (should fail)
   - Try `/schedules` (should fail)
   - **Expected**: Error messages about needing approval

**Success Criteria**: Restricted commands are properly blocked before approval.

### Test 4.2: Admin Approval (Requires Admin Access)

**Objective**: Test admin approval functionality.

**Steps**:
1. **Create admin user** (if not already done):
   ```bash
   cd /path/to/e-trading
   python src/util/create_admin.py YOUR_TELEGRAM_USER_ID your.email@example.com
   ```
   **Note**: This script creates an admin user with full privileges. Run this once before testing admin features.

2. **Approve user**:
   - Send `/admin approve YOUR_TELEGRAM_USER_ID` to your bot
   - **Expected**: Confirmation message about user approval

3. **Test approved user access**:
   - Send `/info` to your bot
   - **Expected**: Shows approved status

**Success Criteria**: User approval process works correctly.

### Test 4.3: Report Generation

**Objective**: Test the report generation functionality.

**Steps**:
1. **Basic report**:
   - Send `/report AAPL` to your bot
   - **Expected**: Comprehensive report with fundamental and technical analysis

2. **Multiple tickers**:
   - Send `/report AAPL MSFT GOOGL` to your bot
   - **Expected**: Reports for all three tickers

3. **Report with flags**:
   - Send `/report AAPL -email -period=1y -indicators=RSI,MACD` to your bot
   - **Expected**: Report sent to both Telegram and email

4. **Invalid ticker**:
   - Send `/report INVALID` to your bot
   - **Expected**: Error message about invalid ticker

5. **Case-insensitive tickers**:
   - Send `/report aapl` to your bot
   - **Expected**: Same report as `/report AAPL`

**Success Criteria**: Report generation works with various parameters and flags.

### Test 4.4: Price Alerts

**Objective**: Test the price alert functionality.

**Steps**:
1. **List alerts**:
   - Send `/alerts` to your bot
   - **Expected**: List of current alerts (empty initially)

2. **Add alert**:
   - Send `/alerts add AAPL 150 above` to your bot
   - **Expected**: Confirmation message about alert creation

3. **Add alert with email**:
   - Send `/alerts add MSFT 300 below -email` to your bot
   - **Expected**: Alert created with email notification

4. **Edit alert**:
   - Send `/alerts edit 1 160 above` to your bot
   - **Expected**: Confirmation message about alert update

5. **Pause/Resume alert**:
   - Send `/alerts pause 1` to your bot
   - Send `/alerts resume 1` to your bot
   - **Expected**: Confirmation messages for pause/resume

6. **Delete alert**:
   - Send `/alerts delete 1` to your bot
   - **Expected**: Confirmation message about alert deletion

7. **Test invalid alert parameters**:
   - Send `/alerts add INVALID 100 above` to your bot
   - Send `/alerts add AAPL invalid above` to your bot
   - **Expected**: Appropriate error messages

**Success Criteria**: All alert operations work correctly.

### Test 4.5: Scheduled Reports

**Objective**: Test the scheduled report functionality.

**Steps**:
1. **List schedules**:
   - Send `/schedules` to your bot
   - **Expected**: List of current schedules (empty initially)

2. **Add daily schedule**:
   - Send `/schedules add AAPL 09:00 daily` to your bot
   - **Expected**: Confirmation message about schedule creation

3. **Add weekly schedule**:
   - Send `/schedules add MSFT 10:00 weekly -email` to your bot
   - **Expected**: Schedule created with email delivery

4. **Add schedule with indicators**:
   - Send `/schedules add GOOGL 11:00 daily -indicators=RSI,MACD,PE` to your bot
   - **Expected**: Schedule created with custom indicators

5. **Edit schedule**:
   - Send `/schedules edit 1 10:00 daily` to your bot
   - **Expected**: Confirmation message about schedule update

6. **Pause/Resume schedule**:
   - Send `/schedules pause 1` to your bot
   - Send `/schedules resume 1` to your bot
   - **Expected**: Confirmation messages for pause/resume

7. **Delete schedule**:
   - Send `/schedules delete 1` to your bot
   - **Expected**: Confirmation message about schedule deletion

**Success Criteria**: All schedule operations work correctly.

---

## Phase 5: Admin Commands

### Test 5.1: Admin Help

**Objective**: Test admin command help system.

**Steps**:
1. **Admin help**:
   - Send `/admin` to your bot
   - **Expected**: List of available admin commands

2. **Test admin access without admin privileges**:
   - Create a non-admin user
   - Try `/admin users` with non-admin user
   - **Expected**: Error message about admin access required

**Success Criteria**: Admin help displays correctly and access control works.

### Test 5.2: User Management

**Objective**: Test admin user management commands.

**Steps**:
1. **List users**:
   - Send `/admin users` to your bot
   - **Expected**: List of all registered users with status

2. **List pending approvals**:
   - Send `/admin pending` to your bot
   - **Expected**: List of users waiting for approval

3. **Approve user**:
   - Send `/admin approve USER_ID` to your bot
   - **Expected**: Confirmation message about user approval

4. **Reject user**:
   - Send `/admin reject USER_ID` to your bot
   - **Expected**: Confirmation message about user rejection

5. **Verify user manually**:
   - Send `/admin verify USER_ID` to your bot
   - **Expected**: Confirmation message about manual verification

6. **Reset user email**:
   - Send `/admin resetemail USER_ID` to your bot
   - **Expected**: Confirmation message about email reset

**Success Criteria**: All user management commands work correctly.

### Test 5.3: System Settings

**Objective**: Test admin system configuration commands.

**Steps**:
1. **Set global limits**:
   - Send `/admin setlimit alerts 10` to your bot
   - Send `/admin setlimit schedules 5` to your bot
   - **Expected**: Confirmation messages about limit updates

2. **Set user-specific limits**:
   - Send `/admin setlimit alerts 15 USER_ID` to your bot
   - **Expected**: Confirmation message about user-specific limit

**Success Criteria**: System settings commands work correctly.

### Test 5.4: Broadcast Messaging

**Objective**: Test admin broadcast functionality.

**Steps**:
1. **Send broadcast**:
   - Send `/admin broadcast Hello everyone! This is a test message.` to your bot
   - **Expected**: Confirmation message about broadcast sent

2. **Test broadcast to multiple users**:
   - Create multiple test users
   - Send broadcast message
   - **Expected**: All users receive the broadcast

**Success Criteria**: Broadcast messages are delivered to all users.

---

## Phase 6: Web Admin Panel

### Test 6.1: Admin Panel Setup

**Objective**: Test the web-based admin panel.

**Steps**:
1. **Start admin panel** (if not already running):
   ```bash
   cd /path/to/e-trading
   python src/frontend/telegram/screener/admin_panel.py
   ```
   **Note**: The admin panel must be running for Phase 6 testing. See "Required Scripts and Services" section above.

2. **Access admin panel**:
   - Open browser and go to `http://localhost:5000`
   - **Expected**: Login page

3. **Login**:
   - Use admin credentials (WEBGUI_LOGIN/WEBGUI_PASSWORD)
   - **Expected**: Dashboard page

**Success Criteria**: Admin panel starts and login works.

### Test 6.2: Dashboard Functionality

**Objective**: Test the admin panel dashboard.

**Steps**:
1. **View dashboard statistics**:
   - Check user counts (total, verified, approved)
   - Check alert and schedule counts
   - Check recent activity
   - **Expected**: All statistics display correctly

2. **Navigate using stat cards**:
   - Click on user count cards
   - Click on alert/schedule count cards
   - **Expected**: Navigation to respective pages

**Success Criteria**: Dashboard displays accurate statistics and navigation works.

### Test 6.3: User Management Interface

**Objective**: Test user management through web interface.

**Steps**:
1. **View user list**:
   - Go to `/users` page
   - **Expected**: List of all users with status information

2. **Filter users**:
   - Use filter buttons (All, Verified, Approved, Pending)
   - **Expected**: User list filters correctly

3. **Approve user**:
   - Click "Approve" button for a pending user
   - **Expected**: User status changes to approved

4. **Reject user**:
   - Click "Reject" button for a pending user
   - **Expected**: User status changes to rejected

5. **Verify user manually**:
   - Click "Verify" button for an unverified user
   - **Expected**: User status changes to verified

6. **Reset user email**:
   - Click "Reset Email" button for a user
   - **Expected**: User email is reset

**Success Criteria**: All user management actions work through web interface.

### Test 6.4: Alert and Schedule Management

**Objective**: Test alert and schedule management through web interface.

**Steps**:
1. **View alerts**:
   - Go to `/alerts` page
   - **Expected**: List of all user alerts

2. **Manage alerts**:
   - Toggle alert status (active/inactive)
   - Delete alerts
   - **Expected**: Alert status changes and deletions work

3. **View schedules**:
   - Go to `/schedules` page
   - **Expected**: List of all user schedules

4. **Manage schedules**:
   - Toggle schedule status (active/inactive)
   - Delete schedules
   - **Expected**: Schedule status changes and deletions work

**Success Criteria**: Alert and schedule management works through web interface.

### Test 6.5: Feedback Management

**Objective**: Test feedback management through web interface.

**Steps**:
1. **View feedback**:
   - Go to `/feedback` page
   - **Expected**: List of user feedback and feature requests

2. **Filter feedback**:
   - Use filter buttons (All, Feedback, Feature Requests)
   - **Expected**: Feedback list filters correctly

**Success Criteria**: Feedback management interface works correctly.

### Test 6.6: Broadcast Messaging

**Objective**: Test broadcast messaging through web interface.

**Steps**:
1. **Send broadcast**:
   - Go to `/broadcast` page
   - Enter message text
   - Click "Send Broadcast"
   - **Expected**: Confirmation message and all users receive broadcast

**Success Criteria**: Broadcast messaging works through web interface.

---

## Phase 7: Advanced Features

### Test 7.1: Fundamental Screener

**Objective**: Test the fundamental screener functionality.

**Steps**:
1. **Test screener with small cap stocks**:
   - Send `/schedules screener us_small_cap 09:00 daily` to your bot
   - **Expected**: Screener runs and generates report

2. **Test screener with medium cap stocks**:
   - Send `/schedules screener us_medium_cap 10:00 daily` to your bot
   - **Expected**: Screener runs and generates report

3. **Test screener with large cap stocks**:
   - Send `/schedules screener us_large_cap 11:00 daily` to your bot
   - **Expected**: Screener runs and generates report

4. **Test screener with Swiss shares**:
   - Send `/schedules screener swiss_shares 12:00 daily` to your bot
   - **Expected**: Screener runs and generates report

**Success Criteria**: All screener types work correctly.

### Test 7.2: Email Integration

**Objective**: Test email delivery functionality.

**Steps**:
1. **Test email report delivery**:
   - Send `/report AAPL -email` to your bot
   - **Expected**: Report sent to registered email address

2. **Test email alert delivery**:
   - Create alert with `-email` flag
   - **Expected**: Alert notifications sent to email

3. **Test email schedule delivery**:
   - Create schedule with `-email` flag
   - **Expected**: Scheduled reports sent to email

**Success Criteria**: Email delivery works for all features.

### Test 7.3: Command Audit System

**Objective**: Test the command audit functionality.

**Steps**:
1. **View audit logs**:
   - Go to `/audit` page in admin panel
   - **Expected**: List of all command executions

2. **Filter audit logs**:
   - Use time filters (24h, custom range)
   - Use user type filters (registered/non-registered)
   - Use success/failure filters
   - **Expected**: Audit logs filter correctly

3. **View user-specific audit**:
   - Click on user ID in audit list
   - **Expected**: Detailed command history for that user

**Success Criteria**: Audit system tracks all commands correctly.

---

## Phase 8: Error Handling and Edge Cases

### Test 8.1: Network and API Errors

**Objective**: Test error handling for network and API issues.

**Steps**:
1. **Test with invalid ticker**:
   - Send `/report INVALIDTICKER123` to your bot
   - **Expected**: Graceful error message

2. **Test with unavailable data**:
   - Try to get report for very obscure ticker
   - **Expected**: Error message about data unavailability

3. **Test rate limiting**:
   - Send multiple rapid commands
   - **Expected**: Bot handles rate limiting gracefully

**Success Criteria**: Bot handles errors gracefully with user-friendly messages.

### Test 8.2: Database and Storage

**Objective**: Test database operations and storage limits.

**Steps**:
1. **Test user limits**:
   - Try to create more alerts than allowed
   - Try to create more schedules than allowed
   - **Expected**: Error messages about limits exceeded

2. **Test database corruption handling**:
   - Manually corrupt database file
   - Restart bot
   - **Expected**: Bot handles corruption gracefully

**Success Criteria**: Database operations work correctly and limits are enforced.

### Test 8.3: Bot Restart and Recovery

**Objective**: Test bot restart and state recovery.

**Steps**:
1. **Restart bot**:
   - Stop the bot process
   - Restart the bot
   - **Expected**: Bot starts successfully and maintains state

2. **Test scheduled tasks after restart**:
   - Create schedules
   - Restart bot
   - **Expected**: Schedules continue to work after restart

**Success Criteria**: Bot recovers properly after restart.

---

## Performance Testing

### Test 9.1: Load Testing

**Objective**: Test bot performance under load.

**Steps**:
1. **Multiple concurrent users**:
   - Simulate multiple users sending commands simultaneously
   - **Expected**: Bot handles concurrent requests without errors

2. **Large data processing**:
   - Request reports for multiple tickers simultaneously
   - **Expected**: Bot processes requests efficiently

**Success Criteria**: Bot performs well under load.

### Test 9.2: Memory and Resource Usage

**Objective**: Monitor resource usage during operation.

**Steps**:
1. **Monitor memory usage**:
   - Run bot for extended period
   - Monitor memory consumption
   - **Expected**: Memory usage remains stable

2. **Monitor CPU usage**:
   - Monitor CPU usage during heavy operations
   - **Expected**: CPU usage is reasonable

**Success Criteria**: Resource usage remains within acceptable limits.

---

## Security Testing

### Test 10.1: Access Control

**Objective**: Test security and access control.

**Steps**:
1. **Test admin access control**:
   - Try admin commands with non-admin user
   - **Expected**: Access denied messages

2. **Test user data isolation**:
   - Create multiple users
   - Verify users can only access their own data
   - **Expected**: Proper data isolation

**Success Criteria**: Security measures work correctly.

### Test 10.2: Input Validation

**Objective**: Test input validation and sanitization.

**Steps**:
1. **Test malicious input**:
   - Try commands with special characters
   - Try SQL injection attempts
   - **Expected**: Input properly validated and sanitized

**Success Criteria**: All inputs are properly validated.

---

## Final Verification

### Test 11.1: End-to-End Workflow

**Objective**: Test complete user workflow from registration to advanced usage.

**Steps**:
1. **Complete user journey**:
   - Register new user
   - Verify email
   - Request approval
   - Get approved
   - Use all restricted features
   - **Expected**: Complete workflow works seamlessly

2. **Admin workflow**:
   - Use admin panel to manage users
   - Send broadcasts
   - Monitor system
   - **Expected**: Admin workflow works correctly

**Success Criteria**: Complete system works end-to-end.

---

## Troubleshooting Guide

### Common Issues and Solutions

1. **Bot not responding**:
   - Check bot token is correct
   - Verify bot is running
   - Check logs for errors

2. **Email not received**:
   - Check SMTP settings
   - Verify email address is correct
   - Check spam folder

3. **Admin panel not accessible**:
   - Check port is not in use
   - Verify admin credentials
   - Check firewall settings

4. **Database errors**:
   - Check database file permissions
   - Verify database schema
   - Check for corruption

### Log Analysis

Monitor logs for:
- Error messages
- Performance issues
- Security events
- User activity patterns

---

## Test Completion Checklist

- [ ] Basic bot functionality tested
- [ ] User registration and verification tested
- [ ] All public commands tested
- [ ] All restricted commands tested
- [ ] All admin commands tested
- [ ] Web admin panel tested
- [ ] Advanced features tested
- [ ] Error handling tested
- [ ] Performance tested
- [ ] Security tested
- [ ] End-to-end workflow tested

## Reporting

After completing all tests, document:
1. Test results and any issues found
2. Performance metrics
3. Security findings
4. Recommendations for improvements
5. Any bugs or unexpected behavior

This comprehensive testing guide ensures thorough validation of all Telegram bot features and functionality.

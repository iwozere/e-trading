import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

import pytest
from unittest.mock import AsyncMock, patch
from src.telegram.screener.notifications import (
    # process_start_command,  # REMOVED, see notification manager refactor
    process_help_command, process_info_command, process_register_command, process_verify_command, process_language_command, process_admin_command, process_alerts_command, process_schedules_command, process_feedback_command, process_feature_command, process_unknown_command, process_report_command
)

@pytest.mark.asyncio
@pytest.mark.parametrize("func,command,extra_args,result_key,result_value,expected_type,expected_title,expected_message", [
    (process_help_command, "start", None, "help_text", "This is help text.", "INFO", "Help", "This is help text."),
    (process_help_command, "help", None, "help_text", "This is help text.", "INFO", "Help", "This is help text."),
    (process_info_command, "info", None, "message", "User info", "INFO", "Info", "User info"),
])
async def test_simple_commands_success(mocker, func, command, extra_args, result_key, result_value, expected_type, expected_title, expected_message):
    message = mocker.Mock()
    message.chat.id = 1
    message.message_id = 2
    telegram_user_id = "3"
    notification_manager = mocker.Mock()
    notification_manager.send_notification = AsyncMock()
    with patch("src.telegram.screener.notifications.handle_command") as mock_handle_command:
        mock_handle_command.return_value = {
            "status": "ok",
            result_key: result_value,
            "title": expected_title
        }
        if extra_args is not None:
            await func(message, telegram_user_id, extra_args, notification_manager)
        else:
            await func(message, telegram_user_id, notification_manager)
    notification_manager.send_notification.assert_awaited_once()
    args, kwargs = notification_manager.send_notification.call_args
    assert kwargs["notification_type"] == expected_type
    assert kwargs["title"] == expected_title
    assert kwargs["message"] == expected_message

@pytest.mark.asyncio
@pytest.mark.parametrize("func,command,extra_args,result_key,result_value,expected_type,expected_title,expected_message", [
    (process_help_command, "start", None, "message", "Something went wrong.", "ERROR", "Help", "Something went wrong."),
    (process_help_command, "help", None, "message", "Something went wrong.", "ERROR", "Help", "Something went wrong."),
    (process_info_command, "info", None, "message", "Info error", "ERROR", "Info", "Info error"),
])
async def test_simple_commands_error(mocker, func, command, extra_args, result_key, result_value, expected_type, expected_title, expected_message):
    message = mocker.Mock()
    message.chat.id = 1
    message.message_id = 2
    telegram_user_id = "3"
    notification_manager = mocker.Mock()
    notification_manager.send_notification = AsyncMock()
    with patch("src.telegram.screener.notifications.handle_command") as mock_handle_command:
        mock_handle_command.return_value = {
            "status": "error",
            result_key: result_value,
            "title": expected_title
        }
        if extra_args is not None:
            await func(message, telegram_user_id, extra_args, notification_manager)
        else:
            await func(message, telegram_user_id, notification_manager)
    notification_manager.send_notification.assert_awaited_once()
    args, kwargs = notification_manager.send_notification.call_args
    assert kwargs["notification_type"] == expected_type
    assert kwargs["title"] == expected_title
    assert kwargs["message"] == expected_message

@pytest.mark.asyncio
@pytest.mark.parametrize("func,command,args,result_key,result_value,expected_type,expected_title,expected_message", [
    (process_register_command, "register", ["register", "user@example.com", "en"], "message", "Registered!", "INFO", "Register", "Registered!"),
    (process_register_command, "register", ["register"], "message", "Missing email.", "ERROR", "Register", "Missing email."),
    (process_verify_command, "verify", ["verify", "1234"], "message", "Verified!", "INFO", "Verify", "Verified!"),
    (process_verify_command, "verify", ["verify"], "message", "Missing code.", "ERROR", "Verify", "Missing code."),
    (process_language_command, "language", ["language", "en"], "message", "Language set!", "INFO", "Language", "Language set!"),
    (process_language_command, "language", ["language"], "message", "Missing language.", "ERROR", "Language", "Missing language."),
    (process_admin_command, "admin", ["admin", "arg1"], "message", "Admin ok", "INFO", "Admin", "Admin ok"),
    (process_alerts_command, "alerts", ["alerts"], "message", "Alerts ok", "INFO", "Alerts", "Alerts ok"),
    (process_schedules_command, "schedules", ["schedules"], "message", "Schedules ok", "INFO", "Schedules", "Schedules ok"),
    (process_feedback_command, "feedback", ["feedback", "Great bot!"], "message", "Thanks!", "INFO", "Feedback", "Thanks!"),
    (process_feature_command, "feature", ["feature", "Add dark mode"], "message", "Feature received!", "INFO", "Feature Request", "Feature received!"),
])
async def test_commands_with_args(mocker, func, command, args, result_key, result_value, expected_type, expected_title, expected_message):
    message = mocker.Mock()
    message.chat.id = 1
    message.message_id = 2
    telegram_user_id = "3"
    notification_manager = mocker.Mock()
    notification_manager.send_notification = AsyncMock()
    with patch("src.telegram.screener.notifications.handle_command") as mock_handle_command:
        # Use 'ok' for success, 'error' for error
        status = "ok" if expected_type == "INFO" else "error"
        mock_handle_command.return_value = {
            "status": status,
            result_key: result_value,
            "title": expected_title
        }
        await func(message, telegram_user_id, args, notification_manager)
    notification_manager.send_notification.assert_awaited_once()
    args_, kwargs = notification_manager.send_notification.call_args
    assert kwargs["notification_type"].upper() == expected_type
    assert kwargs["title"] == expected_title
    assert kwargs["message"] == expected_message

@pytest.mark.asyncio
def test_process_unknown_command(mocker):
    message = mocker.Mock()
    message.chat.id = 1
    message.message_id = 2
    telegram_user_id = "3"
    notification_manager = mocker.Mock()
    notification_manager.send_notification = AsyncMock()
    help_text = "Help text here."
    # Should send error with help text
    import asyncio
    asyncio.run(process_unknown_command(message, telegram_user_id, notification_manager, help_text))
    notification_manager.send_notification.assert_called_once()
    args, kwargs = notification_manager.send_notification.call_args
    assert kwargs["notification_type"] == "ERROR"
    assert kwargs["title"] == "Unknown Command"
    assert kwargs["message"] == help_text

@pytest.mark.asyncio
@pytest.mark.parametrize("args,result,expected_calls", [
    # 1. /report mrns (non-existent ticker)
    (["report", "mrns"],
     {
         "status": "ok",
         "reports": [
             {"ticker": "mrns", "error": "Ticker not found", "message": "", "chart_bytes": None}
         ],
         "email": False,
         "user_email": None
     },
     [
         {"notification_type": "ERROR", "title": "Report Error for mrns", "channels": ["telegram"]}
     ]),
    # 2. /report mtns vti (one non-existent, one existent)
    (["report", "mtns", "vti"],
     {
         "status": "ok",
         "reports": [
             {"ticker": "mtns", "error": "Ticker not found", "message": "", "chart_bytes": None},
             {"ticker": "vti", "error": None, "message": "VTI report", "chart_bytes": b"bytes"}
         ],
         "email": False,
         "user_email": None
     },
     [
         {"notification_type": "ERROR", "title": "Report Error for mtns", "channels": ["telegram"]},
         {"notification_type": "INFO", "title": "Report for vti", "channels": ["telegram"]}
     ]),
    # 3. /report vt vti (both existent)
    (["report", "vt", "vti"],
     {
         "status": "ok",
         "reports": [
             {"ticker": "vt", "error": None, "message": "VT report", "chart_bytes": b"bytes"},
             {"ticker": "vti", "error": None, "message": "VTI report", "chart_bytes": b"bytes"}
         ],
         "email": False,
         "user_email": None
     },
     [
         {"notification_type": "INFO", "title": "Report for vt", "channels": ["telegram"]},
         {"notification_type": "INFO", "title": "Report for vti", "channels": ["telegram"]}
     ]),
    # 4. /report mrns -email (expect both email and telegram)
    (["report", "mrns", "-email"],
     {
         "status": "ok",
         "reports": [
             {"ticker": "mrns", "error": "Ticker not found", "message": "", "chart_bytes": None}
         ],
         "email": True,
         "user_email": "user@example.com"
     },
     [
         {"notification_type": "ERROR", "title": "Report Error for mrns", "channels": ["email"]},
         {"notification_type": "ERROR", "title": "Report Error for mrns", "channels": ["telegram"]}
     ]),
    # 5. /report mtns vti -email (expect both email and telegram for each)
    (["report", "mtns", "vti", "-email"],
     {
         "status": "ok",
         "reports": [
             {"ticker": "mtns", "error": "Ticker not found", "message": "", "chart_bytes": None},
             {"ticker": "vti", "error": None, "message": "VTI report", "chart_bytes": b"bytes"}
         ],
         "email": True,
         "user_email": "user@example.com"
     },
     [
         {"notification_type": "ERROR", "title": "Report Error for mtns", "channels": ["email"]},
         {"notification_type": "INFO", "title": "Report for vti", "channels": ["email"]},
         {"notification_type": "ERROR", "title": "Report Error for mtns", "channels": ["telegram"]},
         {"notification_type": "INFO", "title": "Report for vti", "channels": ["telegram"]}
     ]),
    # 6. /report vt vti -email (expect both email and telegram for each)
    (["report", "vt", "vti", "-email"],
     {
         "status": "ok",
         "reports": [
             {"ticker": "vt", "error": None, "message": "VT report", "chart_bytes": b"bytes"},
             {"ticker": "vti", "error": None, "message": "VTI report", "chart_bytes": b"bytes"}
         ],
         "email": True,
         "user_email": "user@example.com"
     },
     [
         {"notification_type": "INFO", "title": "Report for vt", "channels": ["email"]},
         {"notification_type": "INFO", "title": "Report for vti", "channels": ["email"]},
         {"notification_type": "INFO", "title": "Report for vt", "channels": ["telegram"]},
         {"notification_type": "INFO", "title": "Report for vti", "channels": ["telegram"]}
     ]),
])
async def test_process_report_command_basic(mocker, args, result, expected_calls):
    message = mocker.Mock()
    message.chat.id = 1
    message.message_id = 2
    telegram_user_id = "3"
    notification_manager = mocker.Mock()
    notification_manager.send_notification = AsyncMock()
    with patch("src.telegram.screener.notifications.handle_command") as mock_handle_command:
        mock_handle_command.return_value = result
        await process_report_command(message, telegram_user_id, args, notification_manager)
    # Check calls
    calls = notification_manager.send_notification.await_args_list
    assert len(calls) == len(expected_calls)
    for call, expected in zip(calls, expected_calls):
        kwargs = call.kwargs
        assert kwargs["notification_type"] == expected["notification_type"]
        assert kwargs["title"] == expected["title"]
        for ch in expected["channels"]:
            assert ch in kwargs["channels"]

@pytest.mark.asyncio
@pytest.mark.parametrize("provider,period,interval", [
    ("yf", "2y", "1d"),
    ("av", "1y", "1wk"),
    ("bnc", "6mo", "1h"),
])
async def test_process_report_command_with_params(mocker, provider, period, interval):
    # Simulate /report vt vti -email --provider --period --interval
    args = ["report", "vt", "vti", "-email", f"--provider={provider}", f"--period={period}", f"--interval={interval}"]
    result = {
        "status": "ok",
        "reports": [
            {"ticker": "vt", "error": None, "message": f"VT report {provider} {period} {interval}", "chart_bytes": b"bytes"},
            {"ticker": "vti", "error": None, "message": f"VTI report {provider} {period} {interval}", "chart_bytes": b"bytes"}
        ],
        "email": True,
        "user_email": "user@example.com"
    }
    expected_calls = [
        {"notification_type": "INFO", "title": "Report for vt", "channels": ["email"]},
        {"notification_type": "INFO", "title": "Report for vti", "channels": ["email"]},
        {"notification_type": "INFO", "title": "Report for vt", "channels": ["telegram"]},
        {"notification_type": "INFO", "title": "Report for vti", "channels": ["telegram"]},
    ]
    message = mocker.Mock()
    message.chat.id = 1
    message.message_id = 2
    telegram_user_id = "3"
    notification_manager = mocker.Mock()
    notification_manager.send_notification = AsyncMock()
    with patch("src.telegram.screener.notifications.handle_command") as mock_handle_command:
        mock_handle_command.return_value = result
        await process_report_command(message, telegram_user_id, args, notification_manager)
    calls = notification_manager.send_notification.await_args_list
    assert len(calls) == len(expected_calls)
    for call, expected in zip(calls, expected_calls):
        kwargs = call.kwargs
        assert kwargs["notification_type"] == expected["notification_type"]
        assert kwargs["title"] == expected["title"]
        for ch in expected["channels"]:
            assert ch in kwargs["channels"]

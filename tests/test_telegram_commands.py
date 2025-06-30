import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from src.screener.telegram.telegram_screener_bot import (
    handle_add, handle_delete, handle_list, DEFAULT_PERIOD, DEFAULT_INTERVAL,
    handle_register, handle_info, handle_verify, analyze_command_core,
)

class TestTelegramCommandLogic(unittest.TestCase):
    @patch('src.screener.telegram.telegram_screener_bot.add_ticker')
    @patch('src.screener.telegram.telegram_screener_bot.get_ticker_settings')
    def test_handle_add_success(self, mock_get_ticker_settings, mock_add_ticker):
        mock_get_ticker_settings.return_value = (DEFAULT_PERIOD, DEFAULT_INTERVAL)
        user_id = 'user1'
        args = ['/add', '-yf', 'AAPL,MSFT']
        success, msg = handle_add(user_id, args)
        self.assertTrue(success)
        self.assertIn('AAPL', msg)
        self.assertIn('MSFT', msg)
        self.assertIn('yf', msg)
        self.assertEqual(mock_add_ticker.call_count, 2)

    def test_handle_add_invalid_args(self):
        user_id = 'user1'
        args = ['/add', 'AAPL']
        success, msg = handle_add(user_id, args)
        self.assertFalse(success)
        self.assertIn('Usage', msg)

    @patch('src.screener.telegram.telegram_screener_bot.delete_ticker')
    def test_handle_delete_success(self, mock_delete_ticker):
        user_id = 'user1'
        args = ['/delete', '-yf', 'AAPL,MSFT']
        success, msg = handle_delete(user_id, args)
        self.assertTrue(success)
        self.assertIn('AAPL', msg)
        self.assertIn('MSFT', msg)
        self.assertIn('yf', msg)
        self.assertEqual(mock_delete_ticker.call_count, 2)

    def test_handle_delete_invalid_args(self):
        user_id = 'user1'
        args = ['/delete', 'AAPL']
        success, msg = handle_delete(user_id, args)
        self.assertFalse(success)
        self.assertIn('Usage', msg)

    @patch('src.screener.telegram.telegram_screener_bot.list_tickers')
    def test_handle_list_success(self, mock_list_tickers):
        user_id = 'user1'
        provider = 'yf'
        mock_list_tickers.return_value = {
            'yf': [
                {'ticker': 'AAPL', 'period': DEFAULT_PERIOD, 'interval': DEFAULT_INTERVAL},
                {'ticker': 'MSFT', 'period': DEFAULT_PERIOD, 'interval': DEFAULT_INTERVAL}
            ]
        }
        success, msg = handle_list(user_id, provider)
        self.assertTrue(success)
        self.assertIn('AAPL', msg)
        self.assertIn('MSFT', msg)
        self.assertIn('YF', msg)

    @patch('src.screener.telegram.telegram_screener_bot.list_tickers')
    def test_handle_list_empty(self, mock_list_tickers):
        user_id = 'user1'
        provider = 'yf'
        mock_list_tickers.return_value = {'yf': []}
        success, msg = handle_list(user_id, provider)
        self.assertFalse(success)
        self.assertIn('empty', msg)

    @patch('src.screener.telegram.telegram_screener_bot.set_user_email')
    def test_handle_register_success(self, mock_set_user_email):
        telegram_id = 'user1'
        email = 'test@example.com'
        success, result = handle_register(telegram_id, email)
        self.assertTrue(success)
        self.assertEqual(result[0], email)
        self.assertIn('e-Trading Email Verification', result[2])

    def test_handle_register_invalid_email(self):
        telegram_id = 'user1'
        email = 'notanemail'
        success, msg = handle_register(telegram_id, email)
        self.assertFalse(success)
        self.assertIn('Usage', msg)

    @patch('src.screener.telegram.telegram_screener_bot.get_user_verification_status')
    def test_handle_info_success(self, mock_get_status):
        telegram_id = 'user1'
        mock_get_status.return_value = {
            'email': 'test@example.com',
            'verification_received': True,
            'verification_sent': '2024-01-01',
        }
        success, msg = handle_info(telegram_id)
        self.assertTrue(success)
        self.assertIn('test@example.com', msg)
        self.assertIn('✅', msg)

    @patch('src.screener.telegram.telegram_screener_bot.get_user_verification_status')
    def test_handle_info_no_email(self, mock_get_status):
        telegram_id = 'user1'
        mock_get_status.return_value = None
        success, msg = handle_info(telegram_id)
        self.assertFalse(success)
        self.assertIn('No email registered', msg)

    @patch('src.screener.telegram.telegram_screener_bot.get_user_verification_code')
    @patch('src.screener.telegram.telegram_screener_bot.set_user_verified')
    def test_handle_verify_success(self, mock_set_verified, mock_get_code):
        telegram_id = 'user1'
        code = '123456'
        from datetime import datetime, timedelta, timezone
        sent_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
        mock_get_code.return_value = (code, sent_time)
        success, msg = handle_verify(telegram_id, code)
        self.assertTrue(success)
        self.assertIn('verified', msg)

    @patch('src.screener.telegram.telegram_screener_bot.get_user_verification_code')
    def test_handle_verify_invalid_code(self, mock_get_code):
        telegram_id = 'user1'
        code = '123456'
        sent_time = '2024-01-01 00:00:00'
        mock_get_code.return_value = ('654321', sent_time)
        success, msg = handle_verify(telegram_id, code)
        self.assertFalse(success)
        self.assertIn('Invalid verification code', msg)

    @patch('src.screener.telegram.telegram_screener_bot.get_user_verification_code')
    def test_handle_verify_expired(self, mock_get_code):
        telegram_id = 'user1'
        code = '123456'
        from datetime import datetime, timedelta, timezone
        sent_time = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')
        mock_get_code.return_value = (code, sent_time)
        success, msg = handle_verify(telegram_id, code)
        self.assertFalse(success)
        self.assertIn('expired', msg)

class TestAnalyzeCommandLogic(unittest.TestCase):
    def setUp(self):
        self.user_id = 'user1'
        self.notification_manager = MagicMock()
        self.get_ticker_settings = MagicMock(return_value=(DEFAULT_PERIOD, DEFAULT_INTERVAL))
        self.format_comprehensive_analysis = MagicMock(return_value='YF ANALYSIS')
        self.get_user_verification_status = MagicMock(return_value={'email': 'test@example.com', 'verification_received': True})
        self.bot = MagicMock()

    def test_analyze_command_core_real(self):
        """Test analyze_command_core with real dependencies and various message_text inputs."""
        from src.screener.telegram.telegram_screener_bot import (
            notification_manager, get_ticker_settings,
            format_comprehensive_analysis,
            get_user_verification_status, bot
        )
        user_id = "123456"
        test_cases = [
            "/analyze -yf AAPL",
            "/analyze -bnc BTCUSDT",
            "/analyze -yf AAPL -email",
            "/analyze",
            "/analyze -email",
            "/analyze INVALIDTICKER",
            "/analyze -bnc INVALIDCOIN",
        ]
        for msg in test_cases:
            print(f"\n--- Testing: {msg} ---")
            actions = analyze_command_core(
                user_id=user_id,
                message_text=msg,
                notification_manager=notification_manager,
                get_ticker_settings=get_ticker_settings,
                format_comprehensive_analysis=format_comprehensive_analysis,
                get_user_verification_status=get_user_verification_status,
                bot=bot
            )
            for action in actions:
                print(action)

if __name__ == '__main__':
    unittest.main()
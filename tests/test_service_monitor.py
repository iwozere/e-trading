import unittest
from unittest.mock import MagicMock, patch
from src.notification.service_monitor import ServiceMonitor

class TestServiceMonitor(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_mq = MagicMock()
        with patch('src.notification.service_monitor.MessageQueueClient', return_value=self.mock_mq):
            self.monitor = ServiceMonitor(admin_user_id=1)

    def test_check_service_logs_with_error(self):
        """Test that errors are correctly identified in logs."""
        # Mock subprocess.run
        mock_result = MagicMock()
        mock_result.stdout = "Jan 14 10:00:00 server python[123]: Traceback (most recent call last):\n  File \"app.py\", line 10, in <module>\n    raise ValueError(\"Test error\")\nValueError: Test error"

        with patch('subprocess.run', return_value=mock_result):
            errors = self.monitor.check_service_logs("test.service")
            self.assertTrue(len(errors) > 0)
            self.assertIn("Traceback", errors[0])

    def test_check_service_logs_clean(self):
        """Test that a clean log returns no errors."""
        mock_result = MagicMock()
        mock_result.stdout = "Jan 14 10:00:00 server systemd[1]: Started Test Service."

        with patch('subprocess.run', return_value=mock_result):
            errors = self.monitor.check_service_logs("test.service")
            self.assertEqual(len(errors), 0)

    def test_check_service_status_active(self):
        """Test active status detection."""
        mock_result = MagicMock()
        mock_result.stdout = "active\n"

        with patch('subprocess.run', return_value=mock_result):
            is_active, status = self.monitor.check_service_status("test.service")
            self.assertTrue(is_active)
            self.assertEqual(status, "active")

    def test_check_service_status_inactive(self):
        """Test inactive status detection."""
        mock_result = MagicMock()
        mock_result.stdout = "inactive\n"

        with patch('subprocess.run', return_value=mock_result):
            is_active, status = self.monitor.check_service_status("test.service")
            self.assertFalse(is_active)
            self.assertEqual(status, "inactive")

if __name__ == '__main__':
    unittest.main()

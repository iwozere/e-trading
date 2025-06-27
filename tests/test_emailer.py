import unittest
from unittest.mock import patch, MagicMock, mock_open, call
from src.notification.emailer import EmailNotifier, send_email_alert

to_addr = "akossyrev@gmail.com"

class TestEmailNotifier(unittest.TestCase):
    @patch("smtplib.SMTP")
    def test_send_email_success(self, mock_smtp):
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        notifier = EmailNotifier()
        notifier.send_email(to_addr, "Test Subject", "Test Body")
        mock_smtp.assert_called_with("smtp.office365.com", 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.send_message.assert_called_once()

    @patch("smtplib.SMTP")
    def test_send_email_error(self, mock_smtp):
        mock_smtp.side_effect = Exception("SMTP error")
        notifier = EmailNotifier()
        # Should not raise, just log error
        notifier.send_email(to_addr, "Test Subject", "Test Body")

    @patch("smtplib.SMTP")
    def test_send_email_alert(self, mock_smtp):
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        send_email_alert(to_addr, "Alert Subject", "Alert Body")
        mock_smtp.assert_called_with("smtp.office365.com", 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.send_message.assert_called_once()

    @patch("smtplib.SMTP")
    @patch("builtins.open", new_callable=mock_open, read_data=b"filecontent")
    def test_send_email_with_attachments(self, mock_file, mock_smtp):
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        notifier = EmailNotifier()
        attachments = ["/fake/path/file1.txt", "/fake/path/file2.pdf"]
        notifier.send_email(to_addr, "Test Subject", "Test Body", attachments=attachments)
        # open should be called for each attachment
        calls = [call(attachments[0], "rb"), call(attachments[1], "rb")]
        mock_file.assert_has_calls(calls, any_order=True)
        mock_server.send_message.assert_called_once()

if __name__ == "__main__":
    unittest.main() 
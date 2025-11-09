"""
Unit tests for the alert engine.

Tests alert threshold evaluation, cooldown logic, and notification integration.
"""

import unittest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys
from unittest.mock import Mock, AsyncMock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p04_short_squeeze.core.alert_engine import AlertEngine
from src.ml.pipeline.p04_short_squeeze.core.models import (
    StructuralMetrics, TransientMetrics, Candidate, ScoredCandidate, Alert
)
from src.ml.pipeline.p04_short_squeeze.config.data_classes import (
    AlertConfig, AlertThresholds, AlertThreshold, AlertCooldown, AlertChannels
)
from src.data.db.models.model_short_squeeze import AlertLevel, CandidateSource


class TestAlertEngine(unittest.TestCase):
    """Test cases for AlertEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create alert configuration
        self.alert_config = AlertConfig(
            thresholds=AlertThresholds(
                high=AlertThreshold(
                    squeeze_score=0.8,
                    min_si_percent=0.25,
                    min_volume_spike=4.0,
                    min_sentiment=0.6
                ),
                medium=AlertThreshold(
                    squeeze_score=0.6,
                    min_si_percent=0.20,
                    min_volume_spike=3.0,
                    min_sentiment=0.5
                ),
                low=AlertThreshold(
                    squeeze_score=0.4,
                    min_si_percent=0.15,
                    min_volume_spike=2.0,
                    min_sentiment=0.4
                )
            ),
            cooldown=AlertCooldown(
                high_alert_days=7,
                medium_alert_days=5,
                low_alert_days=3
            ),
            channels=AlertChannels(
                telegram_enabled=True,
                telegram_chat_ids=['@test_alerts'],
                email_enabled=True,
                email_recipients=['test@example.com']
            )
        )

        # Mock notification client
        self.mock_notification_client = Mock()
        self.mock_notification_client.send_notification = AsyncMock(return_value=True)

        # Create alert engine
        self.alert_engine = AlertEngine(self.alert_config, self.mock_notification_client)

        # Sample metrics and candidates
        self.structural_metrics = StructuralMetrics(
            short_interest_pct=0.30,
            days_to_cover=8.5,
            float_shares=50_000_000,
            avg_volume_14d=1_000_000,
            market_cap=500_000_000
        )

        self.transient_metrics_high = TransientMetrics(
            volume_spike=5.0,
            call_put_ratio=2.5,
            sentiment_24h=0.7,
            borrow_fee_pct=0.20
        )

        self.transient_metrics_medium = TransientMetrics(
            volume_spike=3.2,
            call_put_ratio=1.8,
            sentiment_24h=0.55,
            borrow_fee_pct=0.15
        )

        self.transient_metrics_low = TransientMetrics(
            volume_spike=2.1,
            call_put_ratio=1.2,
            sentiment_24h=0.45,
            borrow_fee_pct=0.10
        )

        self.candidate = Candidate(
            ticker="TSLA",
            screener_score=0.75,
            structural_metrics=self.structural_metrics,
            last_updated=datetime.now(),
            source=CandidateSource.SCREENER
        )

    def test_alert_engine_initialization(self):
        """Test alert engine initialization and validation."""
        # Valid configuration should work
        engine = AlertEngine(self.alert_config, self.mock_notification_client)
        self.assertIsNotNone(engine)

        # Invalid threshold ordering should raise error
        invalid_config = AlertConfig(
            thresholds=AlertThresholds(
                high=AlertThreshold(squeeze_score=0.5, min_si_percent=0.25, min_volume_spike=4.0, min_sentiment=0.6),
                medium=AlertThreshold(squeeze_score=0.6, min_si_percent=0.20, min_volume_spike=3.0, min_sentiment=0.5),
                low=AlertThreshold(squeeze_score=0.8, min_si_percent=0.15, min_volume_spike=2.0, min_sentiment=0.4)
            ),
            cooldown=self.alert_config.cooldown,
            channels=self.alert_config.channels
        )

        with self.assertRaises(ValueError) as context:
            AlertEngine(invalid_config, self.mock_notification_client)

        self.assertIn("must be ordered", str(context.exception))

    def test_determine_alert_level_high(self):
        """Test high alert level determination."""
        alert_level = self.alert_engine._determine_alert_level(
            squeeze_score=0.85,
            structural=self.structural_metrics,
            transient=self.transient_metrics_high
        )

        self.assertEqual(alert_level, AlertLevel.HIGH)

    def test_determine_alert_level_medium(self):
        """Test medium alert level determination."""
        alert_level = self.alert_engine._determine_alert_level(
            squeeze_score=0.65,
            structural=self.structural_metrics,
            transient=self.transient_metrics_medium
        )

        self.assertEqual(alert_level, AlertLevel.MEDIUM)

    def test_determine_alert_level_low(self):
        """Test low alert level determination."""
        alert_level = self.alert_engine._determine_alert_level(
            squeeze_score=0.45,
            structural=self.structural_metrics,
            transient=self.transient_metrics_low
        )

        self.assertEqual(alert_level, AlertLevel.LOW)

    def test_determine_alert_level_none(self):
        """Test no alert level when thresholds not met."""
        # Score too low
        alert_level = self.alert_engine._determine_alert_level(
            squeeze_score=0.3,
            structural=self.structural_metrics,
            transient=self.transient_metrics_low
        )

        self.assertIsNone(alert_level)

        # Sentiment too low for high alert
        low_sentiment_transient = TransientMetrics(
            volume_spike=5.0,
            call_put_ratio=2.5,
            sentiment_24h=0.3,  # Below high threshold
            borrow_fee_pct=0.20
        )

        alert_level = self.alert_engine._determine_alert_level(
            squeeze_score=0.85,
            structural=self.structural_metrics,
            transient=low_sentiment_transient
        )

        # Should not be high level due to low sentiment
        self.assertNotEqual(alert_level, AlertLevel.HIGH)

    def test_generate_alert_reason(self):
        """Test alert reason generation."""
        scored_candidate = ScoredCandidate(
            candidate=self.candidate,
            transient_metrics=self.transient_metrics_high,
            squeeze_score=0.85
        )

        reason = self.alert_engine._generate_alert_reason(scored_candidate, AlertLevel.HIGH)

        # Check that reason contains key information
        self.assertIn("HIGH squeeze alert", reason)
        self.assertIn("TSLA", reason)
        self.assertIn("85.0%", reason)  # Squeeze score as percentage
        self.assertIn("30.0%", reason)  # Short interest
        self.assertIn("8.5", reason)    # Days to cover
        self.assertIn("5.0x", reason)   # Volume spike
        self.assertIn("0.70", reason)   # Sentiment

    def test_calculate_cooldown_expiration(self):
        """Test cooldown expiration calculation."""
        now = datetime.now()

        # High alert cooldown
        high_expiry = self.alert_engine._calculate_cooldown_expiration(AlertLevel.HIGH)
        expected_high = now + timedelta(days=7)
        self.assertAlmostEqual(
            high_expiry.timestamp(),
            expected_high.timestamp(),
            delta=60  # Within 1 minute
        )

        # Medium alert cooldown
        medium_expiry = self.alert_engine._calculate_cooldown_expiration(AlertLevel.MEDIUM)
        expected_medium = now + timedelta(days=5)
        self.assertAlmostEqual(
            medium_expiry.timestamp(),
            expected_medium.timestamp(),
            delta=60
        )

        # Low alert cooldown
        low_expiry = self.alert_engine._calculate_cooldown_expiration(AlertLevel.LOW)
        expected_low = now + timedelta(days=3)
        self.assertAlmostEqual(
            low_expiry.timestamp(),
            expected_low.timestamp(),
            delta=60
        )

    @patch('src.ml.pipeline.p04_short_squeeze.core.alert_engine.session_scope')
    def test_is_in_cooldown_false(self, mock_session_scope):
        """Test cooldown check when not in cooldown."""
        # Mock database session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_service.repo.alerts.check_cooldown.return_value = False

        mock_session_scope.return_value.__enter__.return_value = mock_session

        with patch('src.ml.pipeline.p04_short_squeeze.core.alert_engine.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service

            result = self.alert_engine._is_in_cooldown("TSLA", AlertLevel.HIGH)

            self.assertFalse(result)
            mock_service.repo.alerts.check_cooldown.assert_called_once_with("TSLA", AlertLevel.HIGH)

    @patch('src.ml.pipeline.p04_short_squeeze.core.alert_engine.session_scope')
    def test_is_in_cooldown_true(self, mock_session_scope):
        """Test cooldown check when in cooldown."""
        # Mock database session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_service.repo.alerts.check_cooldown.return_value = True

        mock_session_scope.return_value.__enter__.return_value = mock_session

        with patch('src.ml.pipeline.p04_short_squeeze.core.alert_engine.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service

            result = self.alert_engine._is_in_cooldown("TSLA", AlertLevel.HIGH)

            self.assertTrue(result)

    @patch('src.ml.pipeline.p04_short_squeeze.core.alert_engine.session_scope')
    def test_evaluate_candidate_alert_success(self, mock_session_scope):
        """Test successful alert evaluation for a candidate."""
        # Mock database to return no cooldown
        mock_session = Mock()
        mock_service = Mock()
        mock_service.repo.alerts.check_cooldown.return_value = False

        mock_session_scope.return_value.__enter__.return_value = mock_session

        with patch('src.ml.pipeline.p04_short_squeeze.core.alert_engine.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service

            scored_candidate = ScoredCandidate(
                candidate=self.candidate,
                transient_metrics=self.transient_metrics_high,
                squeeze_score=0.85
            )

            alert = self.alert_engine._evaluate_candidate_alert(scored_candidate)

            self.assertIsNotNone(alert)
            self.assertEqual(alert.ticker, "TSLA")
            self.assertEqual(alert.alert_level, AlertLevel.HIGH)
            self.assertEqual(alert.squeeze_score, 0.85)
            self.assertIsNotNone(alert.cooldown_expires)

    @patch('src.ml.pipeline.p04_short_squeeze.core.alert_engine.session_scope')
    def test_evaluate_candidate_alert_cooldown(self, mock_session_scope):
        """Test alert evaluation when candidate is in cooldown."""
        # Mock database to return cooldown active
        mock_session = Mock()
        mock_service = Mock()
        mock_service.repo.alerts.check_cooldown.return_value = True

        mock_session_scope.return_value.__enter__.return_value = mock_session

        with patch('src.ml.pipeline.p04_short_squeeze.core.alert_engine.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service

            scored_candidate = ScoredCandidate(
                candidate=self.candidate,
                transient_metrics=self.transient_metrics_high,
                squeeze_score=0.85
            )

            alert = self.alert_engine._evaluate_candidate_alert(scored_candidate)

            self.assertIsNone(alert)

    def test_evaluate_alerts_multiple_candidates(self):
        """Test evaluating alerts for multiple candidates."""
        with patch.object(self.alert_engine, '_evaluate_candidate_alert') as mock_evaluate:
            # Mock return values
            mock_alert1 = Mock()
            mock_alert2 = None  # No alert for second candidate
            mock_evaluate.side_effect = [mock_alert1, mock_alert2]

            scored_candidates = [
                ScoredCandidate(
                    candidate=self.candidate,
                    transient_metrics=self.transient_metrics_high,
                    squeeze_score=0.85
                ),
                ScoredCandidate(
                    candidate=Candidate(
                        ticker="AAPL",
                        screener_score=0.5,
                        structural_metrics=self.structural_metrics,
                        last_updated=datetime.now()
                    ),
                    transient_metrics=self.transient_metrics_low,
                    squeeze_score=0.3
                )
            ]

            alerts = self.alert_engine.evaluate_alerts(scored_candidates)

            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0], mock_alert1)

    async def test_send_alert_success(self):
        """Test successful alert sending."""
        alert = Alert(
            ticker="TSLA",
            alert_level=AlertLevel.HIGH,
            reason="Test alert reason",
            squeeze_score=0.85,
            timestamp=datetime.now(),
            cooldown_expires=datetime.now() + timedelta(days=7)
        )

        with patch.object(self.alert_engine, '_mark_alert_sent') as mock_mark_sent:
            mock_mark_sent.return_value = None

            result = await self.alert_engine.send_alert(alert)

            self.assertTrue(result)
            self.mock_notification_client.send_notification.assert_called_once()

            # Check notification call arguments
            call_args = self.mock_notification_client.send_notification.call_args
            self.assertIn("HIGH Short Squeeze Alert: TSLA", call_args.kwargs['title'])
            self.assertIn("Test alert reason", call_args.kwargs['message'])
            self.assertEqual(call_args.kwargs['channels'], ['telegram', 'email'])

    async def test_send_alert_failure(self):
        """Test alert sending failure."""
        self.mock_notification_client.send_notification.return_value = False

        alert = Alert(
            ticker="TSLA",
            alert_level=AlertLevel.HIGH,
            reason="Test alert reason",
            squeeze_score=0.85
        )

        result = await self.alert_engine.send_alert(alert)

        self.assertFalse(result)

    @patch('src.ml.pipeline.p04_short_squeeze.core.alert_engine.session_scope')
    async def test_mark_alert_sent(self, mock_session_scope):
        """Test marking alert as sent in database."""
        # Mock database session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_service.create_alert.return_value = 123  # Mock alert ID
        mock_service.mark_alert_sent.return_value = True

        mock_session_scope.return_value.__enter__.return_value = mock_session

        with patch('src.ml.pipeline.p04_short_squeeze.core.alert_engine.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service

            alert = Alert(
                ticker="TSLA",
                alert_level=AlertLevel.HIGH,
                reason="Test alert reason",
                squeeze_score=0.85
            )

            await self.alert_engine._mark_alert_sent(alert, "notification_123")

            mock_service.create_alert.assert_called_once()
            mock_service.mark_alert_sent.assert_called_once_with(123, "notification_123")
            self.assertTrue(alert.sent)
            self.assertEqual(alert.notification_id, "notification_123")

    async def test_process_alerts_complete_workflow(self):
        """Test complete alert processing workflow."""
        scored_candidates = [
            ScoredCandidate(
                candidate=self.candidate,
                transient_metrics=self.transient_metrics_high,
                squeeze_score=0.85
            )
        ]

        with patch.object(self.alert_engine, 'evaluate_alerts') as mock_evaluate, \
             patch.object(self.alert_engine, 'send_alert') as mock_send:

            # Mock alert generation
            mock_alert = Alert(
                ticker="TSLA",
                alert_level=AlertLevel.HIGH,
                reason="Test alert",
                squeeze_score=0.85
            )
            mock_evaluate.return_value = [mock_alert]

            # Mock successful sending
            mock_send.return_value = True
            mock_alert.sent = True

            results = await self.alert_engine.process_alerts(scored_candidates)

            self.assertEqual(results['candidates_processed'], 1)
            self.assertEqual(results['alerts_generated'], 1)
            self.assertEqual(results['alerts_sent'], 1)
            self.assertEqual(results['alerts_failed'], 0)
            self.assertEqual(len(results['alert_details']), 1)

    @patch('src.ml.pipeline.p04_short_squeeze.core.alert_engine.session_scope')
    def test_get_alert_statistics(self, mock_session_scope):
        """Test getting alert statistics."""
        # Mock database session and service
        mock_session = Mock()
        mock_service = Mock()

        # Mock recent alerts
        mock_alerts = [
            Mock(alert_level="HIGH", sent=True),
            Mock(alert_level="MEDIUM", sent=True),
            Mock(alert_level="LOW", sent=False)
        ]
        mock_service.repo.alerts.get_recent_alerts.return_value = mock_alerts

        mock_session_scope.return_value.__enter__.return_value = mock_session

        with patch('src.ml.pipeline.p04_short_squeeze.core.alert_engine.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service

            stats = self.alert_engine.get_alert_statistics(7)

            self.assertEqual(stats['period_days'], 7)
            self.assertEqual(stats['total_alerts'], 3)
            self.assertEqual(stats['alerts_sent'], 2)
            self.assertEqual(stats['alerts_pending'], 1)
            self.assertEqual(stats['by_level']['HIGH'], 1)
            self.assertEqual(stats['by_level']['MEDIUM'], 1)
            self.assertEqual(stats['by_level']['LOW'], 1)
            self.assertAlmostEqual(stats['success_rate'], 2/3, places=2)

    def test_get_cooldown_days(self):
        """Test getting cooldown days for different alert levels."""
        self.assertEqual(self.alert_engine._get_cooldown_days(AlertLevel.HIGH), 7)
        self.assertEqual(self.alert_engine._get_cooldown_days(AlertLevel.MEDIUM), 5)
        self.assertEqual(self.alert_engine._get_cooldown_days(AlertLevel.LOW), 3)


class AsyncTestCase(unittest.TestCase):
    """Base class for async test cases."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def async_test(self, coro):
        return self.loop.run_until_complete(coro)


class TestAlertEngineAsync(AsyncTestCase):
    """Async test cases for AlertEngine."""

    def setUp(self):
        super().setUp()

        # Create alert configuration
        self.alert_config = AlertConfig(
            thresholds=AlertThresholds(
                high=AlertThreshold(squeeze_score=0.8, min_si_percent=0.25, min_volume_spike=4.0, min_sentiment=0.6),
                medium=AlertThreshold(squeeze_score=0.6, min_si_percent=0.20, min_volume_spike=3.0, min_sentiment=0.5),
                low=AlertThreshold(squeeze_score=0.4, min_si_percent=0.15, min_volume_spike=2.0, min_sentiment=0.4)
            ),
            cooldown=AlertCooldown(high_alert_days=7, medium_alert_days=5, low_alert_days=3),
            channels=AlertChannels(telegram_enabled=True, email_enabled=True)
        )

        # Mock notification client
        self.mock_notification_client = Mock()
        self.mock_notification_client.send_notification = AsyncMock(return_value=True)
        self.mock_notification_client.close = AsyncMock()

        # Create alert engine
        self.alert_engine = AlertEngine(self.alert_config, self.mock_notification_client)

    def test_send_alert_async(self):
        """Test async alert sending."""
        alert = Alert(
            ticker="TSLA",
            alert_level=AlertLevel.HIGH,
            reason="Test alert reason",
            squeeze_score=0.85
        )

        with patch.object(self.alert_engine, '_mark_alert_sent') as mock_mark_sent:
            mock_mark_sent.return_value = None

            result = self.async_test(self.alert_engine.send_alert(alert))

            self.assertTrue(result)

    def test_close_async(self):
        """Test async close method."""
        self.async_test(self.alert_engine.close())
        self.mock_notification_client.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
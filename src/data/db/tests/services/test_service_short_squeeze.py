"""
Basic tests for ShortSqueezeService.

Tests cover core functionality:
- Candidate management
- Alert creation
- Data retrieval
- Statistics
"""
from datetime import date

from src.data.db.services.short_squeeze_service import ShortSqueezeService
from src.data.db.models.model_short_squeeze import AlertLevel


class TestShortSqueezeServiceCandidates:
    """Tests for candidate management."""

    def test_add_adhoc_candidate(self, mock_database_service, db_session):
        """Test adding an ad-hoc candidate."""
        service = ShortSqueezeService(db_service=mock_database_service)

        success = service.add_adhoc_candidate(
            ticker="AAPL",
            reason="High short interest detected",
            ttl_days=7
        )

        assert isinstance(success, bool)

    def test_remove_adhoc_candidate(self, mock_database_service, db_session):
        """Test removing an ad-hoc candidate."""
        service = ShortSqueezeService(db_service=mock_database_service)

        # Add first
        service.add_adhoc_candidate(ticker="AAPL", reason="Test", ttl_days=7)

        # Remove
        success = service.remove_adhoc_candidate(ticker="AAPL")

        assert isinstance(success, bool)

    def test_get_active_adhoc_candidates(self, mock_database_service, db_session):
        """Test getting active ad-hoc candidates."""
        service = ShortSqueezeService(db_service=mock_database_service)

        # Add a candidate
        service.add_adhoc_candidate(ticker="TSLA", reason="Test", ttl_days=7)

        # Get active candidates
        candidates = service.get_active_adhoc_candidates()

        assert isinstance(candidates, list)


class TestShortSqueezeServiceAlerts:
    """Tests for alert operations."""

    def test_create_alert(self, mock_database_service, db_session):
        """Test creating an alert."""
        service = ShortSqueezeService(db_service=mock_database_service)

        alert_id = service.create_alert(
            ticker="AAPL",
            alert_level=AlertLevel.MEDIUM,
            reason="Short interest spike",
            analysis_data={"short_interest": 25.5}
        )

        assert alert_id is not None
        assert isinstance(alert_id, int)

    def test_mark_alert_sent(self, mock_database_service, db_session):
        """Test marking alert as sent."""
        service = ShortSqueezeService(db_service=mock_database_service)

        # Create alert first
        alert_id = service.create_alert(
            ticker="AAPL",
            alert_level=AlertLevel.HIGH,
            reason="Test alert",
            analysis_data={}
        )

        # Mark as sent
        success = service.mark_alert_sent(
            alert_id=alert_id,
            notification_id="notif_123"
        )

        assert isinstance(success, bool)


class TestShortSqueezeServiceData:
    """Tests for data retrieval operations."""

    def test_get_top_candidates_by_screener_score(self, mock_database_service, db_session):
        """Test getting top candidates by screener score."""
        service = ShortSqueezeService(db_service=mock_database_service)

        candidates = service.get_top_candidates_by_screener_score(limit=10)

        assert isinstance(candidates, list)

    def test_get_top_squeeze_scores(self, mock_database_service, db_session):
        """Test getting top squeeze scores."""
        service = ShortSqueezeService(db_service=mock_database_service)

        scores = service.get_top_squeeze_scores(limit=10)

        assert isinstance(scores, list)

    def test_get_ticker_analysis(self, mock_database_service, db_session):
        """Test getting ticker analysis."""
        service = ShortSqueezeService(db_service=mock_database_service)

        analysis = service.get_ticker_analysis(ticker="AAPL", days=30)

        assert isinstance(analysis, dict)


class TestShortSqueezeServiceFINRA:
    """Tests for FINRA data operations."""

    def test_store_finra_data(self, mock_database_service, db_session):
        """Test storing FINRA data."""
        service = ShortSqueezeService(db_service=mock_database_service)

        finra_data = [
            {
                "ticker": "AAPL",
                "settlement_date": date.today(),
                "short_interest": 1000000,
                "avg_daily_volume": 50000000
            }
        ]

        count = service.store_finra_data(finra_data_list=finra_data)

        assert isinstance(count, int)
        assert count >= 0

    def test_get_latest_finra_short_interest(self, mock_database_service, db_session):
        """Test getting latest FINRA short interest."""
        service = ShortSqueezeService(db_service=mock_database_service)

        # Store some data first
        finra_data = [{
            "ticker": "AAPL",
            "settlement_date": date.today(),
            "short_interest": 1000000,
            "avg_daily_volume": 50000000
        }]
        service.store_finra_data(finra_data_list=finra_data)

        # Get latest
        data = service.get_latest_finra_short_interest(ticker="AAPL")

        # May be None if not found, or dict if found
        assert data is None or isinstance(data, dict)

    def test_get_finra_data_count_for_date(self, mock_database_service, db_session):
        """Test getting FINRA data count for specific date."""
        service = ShortSqueezeService(db_service=mock_database_service)

        count = service.get_finra_data_count_for_date(settlement_date=date.today())

        assert isinstance(count, int)
        assert count >= 0


class TestShortSqueezeServiceStatistics:
    """Tests for statistics and reporting."""

    def test_get_pipeline_statistics(self, mock_database_service, db_session):
        """Test getting pipeline statistics."""
        service = ShortSqueezeService(db_service=mock_database_service)

        stats = service.get_pipeline_statistics()

        assert isinstance(stats, dict)

    def test_get_finra_data_freshness_report(self, mock_database_service, db_session):
        """Test getting FINRA data freshness report."""
        service = ShortSqueezeService(db_service=mock_database_service)

        report = service.get_finra_data_freshness_report()

        assert isinstance(report, dict)

    def test_cleanup_old_data(self, mock_database_service, db_session):
        """Test cleaning up old data."""
        service = ShortSqueezeService(db_service=mock_database_service)

        result = service.cleanup_old_data(days_to_keep=90)

        assert isinstance(result, dict)

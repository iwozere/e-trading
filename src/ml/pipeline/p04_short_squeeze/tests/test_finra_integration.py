#!/usr/bin/env python3
"""
Test script for FINRA integration and volume-based squeeze detection.

This script tests the complete pipeline with FINRA data and volume analysis.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.downloader.fmp_data_downloader import FMPDataDownloader
from src.data.downloader.finra_data_downloader import create_finra_downloader
from src.ml.pipeline.p04_short_squeeze.core.volume_squeeze_detector import create_volume_squeeze_detector
# FINRA service functionality is now part of ShortSqueezeService

_logger = setup_logger(__name__)


def test_finra_connection():
    """Test FINRA data downloader connection."""
    print("🧪 Testing FINRA Connection...")

    try:
        finra = create_finra_downloader()

        if finra.test_connection():
            print("✅ FINRA connection successful")

            # Get available dates
            dates = finra.get_available_dates()
            if dates:
                print(f"✅ Found {len(dates)} available report dates")
                print(f"   Most recent: {dates[-1].strftime('%Y-%m-%d')}")

                # Test getting data for specific symbols
                test_symbols = ['AAPL', 'TSLA', 'GME']
                bulk_data = finra.get_bulk_short_interest(test_symbols)

                print(f"✅ Bulk data test: {len(bulk_data)}/{len(test_symbols)} symbols found")

                for symbol in test_symbols:
                    if symbol in bulk_data:
                        data = bulk_data[symbol]
                        print(f"   {symbol}: Short Volume = {data.get('ShortVolume', 'N/A')}")
                    else:
                        print(f"   {symbol}: No data available")

                return True
            else:
                print("❌ No available dates found")
                return False
        else:
            print("❌ FINRA connection failed")
            return False

    except Exception as e:
        print(f"❌ FINRA test error: {e}")
        return False


def test_volume_detection():
    """Test volume-based squeeze detection."""
    print("\n🧪 Testing Volume-Based Squeeze Detection...")

    try:
        fmp = FMPDataDownloader()

        if not fmp.test_connection():
            print("❌ FMP connection failed")
            return False

        print("✅ FMP connection successful")

        # Create volume detector
        detector = create_volume_squeeze_detector(fmp)

        # Test with known volatile stocks
        test_tickers = ['GME', 'AMC', 'TSLA', 'AAPL', 'NVDA']

        print(f"Testing volume detection on {len(test_tickers)} tickers...")

        results = []
        for ticker in test_tickers:
            analysis = detector.analyze_ticker(ticker)
            if analysis:
                candidate, indicators = analysis
                results.append((ticker, indicators))
                print(f"✅ {ticker}: Score={indicators.combined_score:.3f}, "
                      f"Probability={indicators.squeeze_probability}")
            else:
                print(f"❌ {ticker}: Analysis failed")

        if results:
            print(f"\n📊 Volume Detection Summary:")
            results.sort(key=lambda x: x[1].combined_score, reverse=True)
            for ticker, indicators in results[:3]:
                print(f"   Top: {ticker} - Score: {indicators.combined_score:.3f}, "
                      f"Volume: {indicators.volume_score:.3f}, "
                      f"Momentum: {indicators.momentum_score:.3f}")

        return len(results) > 0

    except Exception as e:
        print(f"❌ Volume detection test error: {e}")
        return False


def test_database_integration():
    """Test database integration for FINRA data."""
    print("\n🧪 Testing Database Integration...")

    try:
        # Run migration first
        print("Running database migration...")
        from src.data.db.migrations.add_finra_short_interest_table import run_migration

        if not run_migration():
            print("❌ Database migration failed")
            return False

        print("✅ Database migration successful")

        # Test FINRA service (now part of ShortSqueezeService)
        from src.data.db.core.database import session_scope
        from src.data.db.services.short_squeeze_service import ShortSqueezeService

        with session_scope() as session:
            service = ShortSqueezeService(session)

            # Get data freshness report
            report = service.get_finra_data_freshness_report()
        print(f"📊 Data Freshness Report:")
        print(f"   Unique symbols: {report.get('unique_symbols', 0)}")
        print(f"   Total records: {report.get('total_records', 0)}")
        print(f"   Latest report: {report.get('latest_report_date', 'None')}")
        print(f"   Data age: {report.get('data_age_days', 'N/A')} days")

        # Test getting high short interest candidates
        candidates = service.get_high_short_interest_candidates(
            min_short_ratio=0.1, limit=10
        )

        print(f"✅ Found {len(candidates)} high short interest candidates")

        return True

    except Exception as e:
        print(f"❌ Database integration test error: {e}")
        return False


def test_hybrid_screening():
    """Test the complete hybrid screening approach."""
    print("\n🧪 Testing Hybrid Screening (FINRA + Volume)...")

    try:
        from src.ml.pipeline.p04_short_squeeze.config.config_manager import ConfigManager
        from src.ml.pipeline.p04_short_squeeze.core.weekly_screener import create_weekly_screener
        from src.ml.pipeline.p04_short_squeeze.core.universe_loader import create_universe_loader

        # Initialize components
        fmp = FMPDataDownloader()
        config_manager = ConfigManager()
        config = config_manager.load_config()

        # Create universe loader and screener
        universe_loader = create_universe_loader(fmp, config.screener.universe)
        screener = create_weekly_screener(fmp, config.screener)

        # Load small test universe
        print("Loading test universe...")
        full_universe = universe_loader.load_universe()

        if not full_universe:
            print("❌ Failed to load universe")
            return False

        # Use small subset for testing
        test_universe = full_universe[:20]  # Test with 20 stocks
        print(f"✅ Loaded test universe: {len(test_universe)} stocks")

        # Run hybrid screening
        print("Running hybrid screening...")
        results = screener.run_screener(test_universe)

        print(f"📊 Hybrid Screening Results:")
        print(f"   Universe size: {results.total_universe}")
        print(f"   Candidates found: {results.candidates_found}")
        print(f"   Top candidates: {len(results.top_candidates)}")
        print(f"   Runtime: {results.runtime_metrics.get('duration_seconds', 0):.2f} seconds")

        # Show top candidates
        if results.top_candidates:
            print(f"\n🎯 Top Candidates:")
            for i, candidate in enumerate(results.top_candidates[:5], 1):
                print(f"   {i}. {candidate.ticker}: "
                      f"Score={candidate.screener_score:.3f}, "
                      f"SI={candidate.structural_metrics.short_interest_pct*100:.1f}%, "
                      f"Source={candidate.source.value}")

        return results.candidates_found > 0

    except Exception as e:
        print(f"❌ Hybrid screening test error: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting FINRA Integration Tests")
    print("=" * 60)

    tests = [
        ("FINRA Connection", test_finra_connection),
        ("Volume Detection", test_volume_detection),
        ("Database Integration", test_database_integration),
        ("Hybrid Screening", test_hybrid_screening),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            if test_func():
                print(f"✅ {test_name} - PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! FINRA integration is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
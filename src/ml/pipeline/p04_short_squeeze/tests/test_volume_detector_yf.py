#!/usr/bin/env python3
"""
Quick test of the yfinance volume detector
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p04_short_squeeze.core.volume_squeeze_detector_yf import create_volume_squeeze_detector_yf

def test_volume_detector():
    """Test the volume detector with a few tickers."""
    print("Testing Volume Detector with yfinance...")
    
    # Create detector
    detector = create_volume_squeeze_detector_yf()
    print("✓ Detector created successfully")
    
    # Test with a few popular tickers
    test_tickers = ['AAPL', 'TSLA', 'NVDA', 'GME', 'AMC']
    
    print(f"\nTesting {len(test_tickers)} tickers...")
    for ticker in test_tickers:
        print(f"\n--- Testing {ticker} ---")
        
        # Test company info
        info = detector.get_company_info_yf(ticker)
        if info:
            print(f"  Market Cap: ${info.get('market_cap', 0):,.0f}")
            print(f"  Float Shares: {info.get('float_shares', 0):,.0f}")
        
        # Test volume metrics
        volume_metrics = detector.calculate_volume_metrics(ticker)
        if volume_metrics:
            print(f"  Volume Spike: {volume_metrics.volume_spike_ratio:.2f}x")
            print(f"  Avg Volume (14d): {volume_metrics.avg_volume_14d:,.0f}")
        
        # Test momentum metrics
        momentum_metrics = detector.calculate_momentum_metrics(ticker)
        if momentum_metrics:
            print(f"  Momentum Score: {momentum_metrics.momentum_score:.3f}")
            print(f"  1d Change: {momentum_metrics.price_change_1d:.2%}")
        
        # Test full analysis
        result = detector.analyze_ticker(ticker)
        if result:
            candidate, indicators = result
            print(f"  ✓ CANDIDATE: Score={indicators.combined_score:.3f}, Probability={indicators.squeeze_probability}")
        else:
            print(f"  ✗ Not a candidate")
    
    print("\n" + "="*50)
    print("Test completed successfully!")
    print("="*50)

if __name__ == "__main__":
    test_volume_detector()

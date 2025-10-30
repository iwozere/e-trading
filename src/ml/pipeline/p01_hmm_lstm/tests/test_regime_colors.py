#!/usr/bin/env python3
"""
Test script to verify regime color mapping fixes.

This script helps test the improved regime labeling and color mapping
to ensure we get proper distribution of Bearish (red), Sideways (blue),
and Bullish (green) regimes.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from src.ml.pipeline.p01_hmm_lstm.x_03_train_hmm import HMMTrainer

def test_regime_labeling():
    """Test the regime labeling logic with different scenarios."""

    trainer = HMMTrainer()

    # Test scenarios
    test_cases = [
        {
            'name': 'Good Separation - All Positive',
            'returns': [0.001, 0.002, 0.003],  # Good range, all positive
            'expected': ['Bearish', 'Sideways', 'Bullish']
        },
        {
            'name': 'Poor Separation - All Positive',
            'returns': [0.0001, 0.0002, 0.0003],  # Small range, all positive
            'expected': ['Sideways', 'Sideways', 'Bullish']
        },
        {
            'name': 'Mixed Returns',
            'returns': [-0.001, 0.0001, 0.002],  # Negative, near zero, positive
            'expected': ['Bearish', 'Sideways', 'Bullish']
        },
        {
            'name': 'All Negative',
            'returns': [-0.003, -0.002, -0.001],  # All negative
            'expected': ['Bearish', 'Sideways', 'Bullish']
        }
    ]

    print("Testing Regime Labeling Logic")
    print("=" * 50)

    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print(f"Returns: {test_case['returns']}")

        # Create mock data
        df = pd.DataFrame({
            'log_return': test_case['returns'] * 1000,  # Scale up for more realistic values
            'close': [100 + i for i in range(len(test_case['returns']))]
        })

        # Create mock regimes (0, 1, 2)
        regimes = np.array([0, 1, 2])

        # Test the labeling logic
        labels = trainer.analyze_regime_characteristics(df, regimes, "test")

        print(f"Expected: {test_case['expected']}")
        print(f"Actual:   {labels}")

        # Test color mapping
        color_mapping = {}
        used_colors = set()

        for i, label in enumerate(labels):
            if 'bearish' in label.lower():
                color_mapping[i] = 'red'
                used_colors.add('red')
            elif 'bullish' in label.lower():
                color_mapping[i] = 'green'
                used_colors.add('green')
            elif 'sideways' in label.lower():
                color_mapping[i] = 'blue'
                used_colors.add('blue')

        print(f"Colors used: {used_colors}")
        print(f"Color mapping: {color_mapping}")

        # Check if we have all three colors
        if len(used_colors) == 3:
            print("✅ All three colors present")
        else:
            print(f"⚠️  Missing colors: {set(['red', 'green', 'blue']) - used_colors}")

def test_4h_scenario():
    """Test the specific 4h timeframe scenario."""

    print("\n" + "=" * 50)
    print("Testing 4h Timeframe Scenario")
    print("=" * 50)

    # Simulate 4h data characteristics
    # 4h timeframes often have all positive returns with small differences
    returns_4h = [0.0001, 0.0002, 0.0003]  # Typical 4h scenario

    df = pd.DataFrame({
        'log_return': returns_4h * 1000,
        'close': [100 + i for i in range(len(returns_4h))]
    })

    regimes = np.array([0, 1, 2])

    trainer = HMMTrainer()
    labels = trainer.analyze_regime_characteristics(df, regimes, "4h")

    print(f"4h Returns: {returns_4h}")
    print(f"Labels: {labels}")

    # Test color mapping
    color_mapping = {}
    for i, label in enumerate(labels):
        if 'bearish' in label.lower():
            color_mapping[i] = 'red'
        elif 'bullish' in label.lower():
            color_mapping[i] = 'green'
        elif 'sideways' in label.lower():
            color_mapping[i] = 'blue'

    print(f"Color mapping: {color_mapping}")

    # Check for red color
    if 'red' in color_mapping.values():
        print("✅ Red color present")
    else:
        print("❌ Red color missing")

if __name__ == "__main__":
    test_regime_labeling()
    test_4h_scenario()

    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    print("The improved logic should:")
    print("1. Use relative positioning when regimes are well-separated")
    print("2. Use absolute thresholds when regimes are poorly separated")
    print("3. Ensure all three colors (red, blue, green) are used")
    print("4. Handle 4h timeframes with small positive returns")

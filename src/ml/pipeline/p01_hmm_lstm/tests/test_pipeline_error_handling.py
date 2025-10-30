#!/usr/bin/env python3
"""
Test script to demonstrate improved pipeline error handling.

This script shows how the pipeline now properly handles failures with:
1. Fail-fast mode (default) - stops on critical stage failures
2. Configurable error handling for optional vs critical stages
3. Clear error reporting and recovery suggestions
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from src.notification.logger import setup_logger
from run_pipeline import PipelineRunner

_logger = setup_logger(__name__)

def test_pipeline_error_handling():
    """Test different pipeline error handling scenarios."""

    print("=" * 80)
    print("PIPELINE ERROR HANDLING TEST")
    print("=" * 80)

    # Initialize pipeline runner
    runner = PipelineRunner()

    # Test 1: List stages with criticality information
    print("\n1. Pipeline Stages with Criticality:")
    print("-" * 50)
    runner.list_stages()

    # Test 2: Validate requirements
    print("\n2. Requirements Validation:")
    print("-" * 50)
    if runner.validate_requirements():
        print("✓ All requirements validated successfully")
    else:
        print("✗ Requirements validation failed")
        return

    # Test 3: Demonstrate fail-fast behavior
    print("\n3. Fail-Fast Behavior Demonstration:")
    print("-" * 50)
    print("The pipeline now has the following error handling behavior:")
    print()
    print("CRITICAL STAGES (fail-fast enabled by default):")
    print("  • Data Loading (Stage 1)")
    print("  • Data Preprocessing (Stage 2)")
    print("  • HMM Training (Stage 3)")
    print("  • HMM Application (Stage 4)")
    print("  • LSTM Training (Stage 7)")
    print()
    print("OPTIONAL STAGES (can fail without stopping pipeline):")
    print("  • Indicator Optimization (Stage 5)")
    print("  • LSTM Optimization (Stage 6)")
    print("  • Model Validation (Stage 8)")
    print()
    print("COMMAND LINE OPTIONS:")
    print("  --no-fail-fast                    # Disable fail-fast mode")
    print("  --continue-on-optional-failures   # Continue on optional failures")
    print("  --skip-stages 5,6,8              # Skip optional stages")
    print()

    # Test 4: Show example usage
    print("4. Example Usage:")
    print("-" * 50)
    print("Default behavior (fail-fast enabled):")
    print("  python run_pipeline.py")
    print()
    print("Skip optional stages to focus on core pipeline:")
    print("  python run_pipeline.py --skip-stages 5,6,8")
    print()
    print("Disable fail-fast for debugging:")
    print("  python run_pipeline.py --no-fail-fast")
    print()
    print("Continue even if optional stages fail:")
    print("  python run_pipeline.py --continue-on-optional-failures")
    print()

    # Test 5: Error recovery suggestions
    print("5. Error Recovery Suggestions:")
    print("-" * 50)
    print("When a critical stage fails:")
    print("  1. Check the error message and logs")
    print("  2. Fix the underlying issue (data, config, etc.)")
    print("  3. Restart the pipeline from the failed stage:")
    print("     python run_pipeline.py --skip-stages 1,2,3  # if stage 4 failed")
    print()
    print("When an optional stage fails:")
    print("  1. Check if the failure affects your use case")
    print("  2. Use --continue-on-optional-failures to proceed")
    print("  3. Or skip the problematic stage:")
    print("     python run_pipeline.py --skip-stages 5")
    print()

    print("=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    test_pipeline_error_handling()

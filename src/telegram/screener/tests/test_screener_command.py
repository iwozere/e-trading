#!/usr/bin/env python3
"""
Test script for the new /screener command functionality.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.telegram.command_parser import parse_command

def test_screener_command_parsing():
    """Test that the screener command is parsed correctly."""
    print("Testing /screener command parsing...")

    # Test basic screener command
    result = parse_command('/screener {"screener_type":"hybrid","list_type":"us_medium_cap","max_results":5}')
    print(f"Command: {result.command}")
    print(f"Args: {result.args}")
    print(f"Positionals: {result.positionals}")

    # Test screener command with email flag
    result2 = parse_command('/screener {"screener_type":"fundamental","list_type":"us_small_cap","max_results":10} -email')
    print(f"\nCommand with email: {result2.command}")
    print(f"Args: {result2.args}")
    print(f"Positionals: {result2.positionals}")

    print("\n✅ Screener command parsing test completed!")

if __name__ == "__main__":
    test_screener_command_parsing()

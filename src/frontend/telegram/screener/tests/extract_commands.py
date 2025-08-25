#!/usr/bin/env python3
"""
Extract screener commands from screener_commands.json and format them for easy copy-paste.
"""

import json
from pathlib import Path

def extract_commands():
    """Extract and format screener commands for copy-paste use."""

    # Load the commands file
    commands_file = Path(__file__).parent / "screener_commands.json"

    with open(commands_file, 'r') as f:
        commands = json.load(f)

    print("🎯 SCREENER COMMANDS FOR COPY-PASTE")
    print("=" * 80)
    print("Use these commands with your Telegram bot:")
    print("Format: /screener <JSON_CONFIG> [-email]")
    print("=" * 80)

    for screener_name, screener_data in commands.items():
        print(f"\n📊 {screener_data['description'].upper()}")
        print("-" * 60)

        # Convert the config to a compact JSON string
        config_json = json.dumps(screener_data['config'], separators=(',', ':'))

        # Create the command
        command = f"/screener {config_json}"

        print("Command (without email):")
        print(command)
        print()

        print("Command (with email):")
        print(f"{command} -email")
        print()

        print("JSON Config (for manual formatting):")
        print(json.dumps(screener_data['config'], indent=2))
        print("-" * 60)

    print("\n📋 USAGE INSTRUCTIONS:")
    print("1. Copy the JSON config from above")
    print("2. Use with /screener command in Telegram")
    print("3. Add -email flag to receive results via email")
    print("4. Each screener will find up to 100 stocks")
    print("5. Results include fundamental and technical analysis")
    print("6. FMP pre-filters stocks before detailed analysis")

    print("\n🎯 QUICK COMMANDS:")
    print("=" * 40)

    for screener_name, screener_data in commands.items():
        config_json = json.dumps(screener_data['config'], separators=(',', ':'))
        print(f"{screener_name}: /screener {config_json} -email")

if __name__ == "__main__":
    extract_commands()

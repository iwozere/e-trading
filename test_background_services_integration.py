#!/usr/bin/env python3
"""
Test script for Background Services Integration with Bot HTTP API
"""

import asyncio
from src.frontend.telegram.screener.http_api_client import BotHttpApiClient, send_notification_via_api
from src.frontend.telegram.screener.alert_monitor import AlertMonitor
from src.frontend.telegram.screener.schedule_processor import ScheduleProcessor

async def test_background_services_integration():
    """Test the integration between background services and bot HTTP API."""
    print("Testing Background Services Integration with Bot HTTP API...")

    # Test 1: HTTP API Client
    print("\n1. Testing HTTP API Client...")
    async with BotHttpApiClient() as client:
        # Test connection
        if await client.test_connection():
            print("✅ Bot API connection successful")
        else:
            print("❌ Bot API connection failed (bot may not be running)")
            print("   This is expected if the bot is not currently running")

        # Test status endpoint
        status = await client.get_status()
        if status.get('success'):
            print(f"✅ Status endpoint working: {status}")
        else:
            print(f"❌ Status endpoint failed: {status}")

    # Test 2: Direct notification sending
    print("\n2. Testing direct notification sending...")
    # This will fail if bot is not running, which is expected
    success = await send_notification_via_api(
        user_id="12345",  # Test user ID
        message="Test notification from background services",
        title="Background Services Test"
    )
    if success:
        print("✅ Direct notification sending successful")
    else:
        print("❌ Direct notification sending failed (expected if bot not running)")

    # Test 3: AlertMonitor initialization
    print("\n3. Testing AlertMonitor initialization...")
    try:
        api_client = BotHttpApiClient()
        alert_monitor = AlertMonitor(api_client)
        print("✅ AlertMonitor initialized successfully")
    except Exception as e:
        print(f"❌ AlertMonitor initialization failed: {e}")

    # Test 4: ScheduleProcessor initialization
    print("\n4. Testing ScheduleProcessor initialization...")
    try:
        api_client = BotHttpApiClient()
        schedule_processor = ScheduleProcessor(api_client)
        print("✅ ScheduleProcessor initialized successfully")
    except Exception as e:
        print(f"❌ ScheduleProcessor initialization failed: {e}")

    print("\n🎉 Background Services Integration Test Complete!")
    print("\nTo run the complete system:")
    print("1. Start the bot: python src/frontend/telegram/bot.py")
    print("2. Start background services: python src/frontend/telegram/screener/background_services.py")
    print("3. Start admin panel: python src/frontend/telegram/screener/admin_panel.py")

if __name__ == "__main__":
    asyncio.run(test_background_services_integration())

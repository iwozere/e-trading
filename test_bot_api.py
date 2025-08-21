#!/usr/bin/env python3
"""
Test script for Bot HTTP API
"""

import asyncio
import aiohttp
import json

async def test_bot_api():
    """Test the bot API endpoints"""
    base_url = "http://localhost:8080"

    print("Testing Bot HTTP API...")

    async with aiohttp.ClientSession() as session:
        # Test 1: Health check
        print("\n1. Testing health check...")
        try:
            async with session.get(f"{base_url}/api/test") as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Health check passed: {result}")
                else:
                    print(f"❌ Health check failed: HTTP {response.status}")
        except Exception as e:
            print(f"❌ Health check error: {e}")

        # Test 2: Status endpoint
        print("\n2. Testing status endpoint...")
        try:
            async with session.get(f"{base_url}/api/status") as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Status check passed: {result}")
                else:
                    print(f"❌ Status check failed: HTTP {response.status}")
        except Exception as e:
            print(f"❌ Status check error: {e}")

        # Test 3: Broadcast endpoint (should fail without bot running)
        print("\n3. Testing broadcast endpoint...")
        try:
            async with session.post(f"{base_url}/api/broadcast", json={
                "message": "Test broadcast message",
                "title": "Test Announcement"
            }) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Broadcast test passed: {result}")
                else:
                    error_text = await response.text()
                    print(f"❌ Broadcast test failed: HTTP {response.status} - {error_text}")
        except Exception as e:
            print(f"❌ Broadcast test error: {e}")

if __name__ == "__main__":
    asyncio.run(test_bot_api())

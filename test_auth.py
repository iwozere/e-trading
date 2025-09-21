#!/usr/bin/env python3
"""
Test Authentication API
----------------------

Simple script to test the authentication endpoints.
"""

import requests
import json

API_BASE = "http://localhost:5003"

def test_health():
    """Test health endpoint."""
    try:
        response = requests.get(f"{API_BASE}/api/health")
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_login():
    """Test login endpoint."""
    try:
        login_data = {
            "username": "admin",
            "password": "admin"
        }

        response = requests.post(
            f"{API_BASE}/auth/login",
            json=login_data,
            headers={"Content-Type": "application/json"}
        )

        print(f"Login: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Login successful! Token: {data['access_token'][:20]}...")
            print(f"User: {data['user']}")
            return data['access_token']
        else:
            print(f"Login failed: {response.text}")
            return None

    except Exception as e:
        print(f"Login test failed: {e}")
        return None

def test_protected_endpoint(token):
    """Test protected endpoint with token."""
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        response = requests.get(f"{API_BASE}/api/test-auth", headers=headers)
        print(f"Protected endpoint: {response.status_code}")

        if response.status_code == 200:
            print(f"Auth test successful: {response.json()}")
            return True
        else:
            print(f"Auth test failed: {response.text}")
            return False

    except Exception as e:
        print(f"Protected endpoint test failed: {e}")
        return False

def test_strategies_endpoint(token):
    """Test strategies endpoint."""
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        response = requests.get(f"{API_BASE}/api/strategies", headers=headers)
        print(f"Strategies endpoint: {response.status_code}")

        if response.status_code == 200:
            strategies = response.json()
            print(f"Strategies: {len(strategies)} found")
            return True
        else:
            print(f"Strategies test failed: {response.text}")
            return False

    except Exception as e:
        print(f"Strategies endpoint test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Trading Web UI Authentication")
    print("=" * 50)

    # Test health
    if not test_health():
        print("❌ Health check failed - is the backend running?")
        return

    print()

    # Test login
    token = test_login()
    if not token:
        print("❌ Login failed")
        return

    print()

    # Test protected endpoint
    if not test_protected_endpoint(token):
        print("❌ Protected endpoint test failed")
        return

    print()

    # Test strategies endpoint
    if not test_strategies_endpoint(token):
        print("❌ Strategies endpoint test failed")
        return

    print()
    print("✅ All authentication tests passed!")

if __name__ == "__main__":
    main()
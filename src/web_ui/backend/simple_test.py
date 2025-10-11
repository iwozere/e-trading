#!/usr/bin/env python3
"""
Simple Backend Test Runner
-------------------------

A basic test to verify the backend code works without complex pytest setup.
This avoids PostgreSQL and other dependency issues.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

def test_imports():
    """Test that we can import the main backend modules."""
    try:
        from src.web_ui.backend.auth import create_access_token, verify_token
        from src.web_ui.backend.services.webui_app_service import WebUIAppService
        print("✅ Backend imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_jwt_token_creation():
    """Test JWT token creation and verification."""
    try:
        from src.web_ui.backend.auth import create_access_token, verify_token

        # Create a token
        data = {"sub": "test_user", "username": "testuser"}
        token = create_access_token(data)

        # Verify the token
        payload = verify_token(token)

        assert payload["sub"] == "test_user"
        assert payload["username"] == "testuser"

        print("✅ JWT token creation and verification works")
        return True
    except Exception as e:
        print(f"❌ JWT test failed: {e}")
        return False

def test_webui_service():
    """Test WebUI service initialization."""
    try:
        from src.web_ui.backend.services.webui_app_service import WebUIAppService

        service = WebUIAppService()
        assert service is not None

        print("✅ WebUI service initialization works")
        return True
    except Exception as e:
        print(f"❌ WebUI service test failed: {e}")
        return False

def main():
    """Run all simple tests."""
    print("Running Simple Backend Tests...")
    print("=" * 50)

    tests = [
        test_imports,
        test_jwt_token_creation,
        test_webui_service,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")

    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All simple tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
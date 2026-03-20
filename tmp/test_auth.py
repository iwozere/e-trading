
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Imports
from src.data.db.services.users_service import users_service

def test_auth():
    print("Testing authentication for admin...")
    # admin@trading-system.local / admin
    result = users_service.authenticate_user_by_email("admin@trading-system.local", "admin")
    if result:
        print(f"Auth Success: {result['email']} (Role: {result['role']})")
    else:
        print("Auth Failed!")

    print("\nTesting authentication for trader...")
    # trader / trader (via username lookup)
    result = users_service.authenticate_user_by_username("trader", "trader")
    if result:
        print(f"Auth Success: {result['email']} (Role: {result['role']})")
    else:
        print("Auth Failed!")

if __name__ == "__main__":
    try:
        test_auth()
    except Exception as e:
        print(f"Error testing auth: {e}")
        import traceback
        traceback.print_exc()

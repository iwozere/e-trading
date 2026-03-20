
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Imports
from sqlalchemy import select
from sqlalchemy.orm import Session
from src.data.db.core.database import make_engine
from src.data.db.models.model_users import User

def check_users():
    print("Checking users in database...")
    engine = make_engine()
    with Session(engine) as session:
        users = session.execute(select(User)).scalars().all()
        if not users:
            print("No users found in database!")
        for user in users:
            print(f"ID: {user.id}, Email: {user.email}, Role: {user.role}, Active: {user.is_active}")

if __name__ == "__main__":
    try:
        check_users()
    except Exception as e:
        print(f"Error checking users: {e}")
        import traceback
        traceback.print_exc()

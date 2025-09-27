#!/usr/bin/env python3
"""
Admin Setup Script
==================

This script helps set up admin users for the Telegram bot.
It registers the user, verifies their email, and assigns admin privileges.

Usage:
    python src/util/create_admin.py <telegram_user_id> <email>

Example:
    python src/util/create_admin.py 123456789 admin@example.com
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.data.db.services import telegram_service as db
from src.notification.logger import setup_logger

logger = setup_logger("create_admin")

def generate_code():
    """Generate a 6-digit verification code"""
    import random
    return f"{random.randint(100000, 999999):06d}"

def create_admin(telegram_user_id: str, email: str):
    """
    Create an admin user with the given telegram_user_id and email.

    Args:
        telegram_user_id: The Telegram user ID
        email: The email address for the admin

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize database
        db.init_db()

        # Check if user already exists
        existing_status = db.get_user_status(telegram_user_id)
        if existing_status:
            logger.info("User %s already exists. Current status: %s", telegram_user_id, existing_status)

            # If user is already admin, no need to do anything
            if existing_status.get("is_admin", False):
                logger.info("User %s is already an admin.", telegram_user_id)
                return True

            # If user exists but is not admin, update to admin
            logger.info("Updating existing user %s to admin role.", telegram_user_id)

            # Generate new verification code and set as verified and approved
            code = generate_code()
            sent_time = int(time.time())

            # Update user with admin privileges
            db.set_user_email(telegram_user_id, email, code, sent_time, is_admin=True)

            # Manually verify and approve the user
            conn = db.sqlite3.connect(db.DB_PATH)
            c = conn.cursor()
            c.execute("UPDATE users SET verified=1, approved=1, is_admin=1 WHERE telegram_user_id=?", (telegram_user_id,))
            conn.commit()
            conn.close()

            logger.info("Successfully updated user %s to admin role.", telegram_user_id)
            return True

        # Create new admin user
        logger.info("Creating new admin user: %s with email: %s", telegram_user_id, email)

        # Generate verification code
        code = generate_code()
        sent_time = int(time.time())

        # Create user with admin privileges
        db.set_user_email(telegram_user_id, email, code, sent_time, is_admin=True)

        # Manually verify and approve the user
        conn = db.sqlite3.connect(db.DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE users SET verified=1, approved=1, is_admin=1 WHERE telegram_user_id=?", (telegram_user_id,))
        conn.commit()
        conn.close()

        logger.info("Successfully created admin user %s with email %s", telegram_user_id, email)
        logger.info("Verification code: %s (not needed since user is auto-verified)", code)

        return True

    except Exception as e:
        logger.exception("Error creating admin user: ")
        return False

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) != 3:
        print("Usage: python src/util/create_admin.py <telegram_user_id> <email>")
        print("Example: python src/util/create_admin.py 123456789 admin@example.com")
        sys.exit(1)

    telegram_user_id = sys.argv[1]
    email = sys.argv[2]

    # Validate inputs
    if not telegram_user_id.isdigit():
        print("Error: telegram_user_id must be a number")
        sys.exit(1)

    if "@" not in email or "." not in email:
        print("Error: email must be a valid email address")
        sys.exit(1)

    print(f"Setting up admin user:")
    print(f"  Telegram User ID: {telegram_user_id}")
    print(f"  Email: {email}")
    print()

    # Create admin user
    success = create_admin(telegram_user_id, email)

    if success:
        print("✅ Admin user created successfully!")
        print(f"User {telegram_user_id} can now use admin commands in the Telegram bot.")
        print("Admin commands include:")
        print("  /admin users - List all users")
        print("  /admin approve USER_ID - Approve a user")
        print("  /admin reject USER_ID - Reject a user")
        print("  /admin pending - List pending approvals")
        print("  /admin broadcast MESSAGE - Send broadcast message")
    else:
        print("❌ Failed to create admin user. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()

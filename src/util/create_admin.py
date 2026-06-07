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

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.data.db.services.telegram_service import telegram_service
from src.data.db.services.users_service import users_service
from src.notification.logger import setup_logger

logger = setup_logger("create_admin")

def generate_code() -> str:
    """Generate a cryptographically secure 6-digit verification code."""
    import secrets
    return f"{secrets.randbelow(900000) + 100000:06d}"

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
        # Check if user already exists
        existing_status = telegram_service.get_user_status(telegram_user_id)
        if existing_status:
            logger.info("User %s already exists. Current status: %s", telegram_user_id, existing_status)

            # If user is already admin, no need to do anything
            if existing_status.get("is_admin", False):
                logger.info("User %s is already an admin.", telegram_user_id)
                return True

            # If user exists but is not admin, update to admin
            logger.info("Updating existing user %s to admin role.", telegram_user_id)

            # Generate verification code
            code = generate_code()
            sent_time = int(time.time())

            # Update user info and set as admin
            users_service.update_user(
                telegram_user_id=telegram_user_id,
                email=email,
                is_admin=True,
                is_verified=True,
                is_approved=True
            )

            logger.info("Successfully updated user %s to admin role.", telegram_user_id)
            return True

        # Create new admin user
        logger.info("Creating new admin user: %s with email: %s", telegram_user_id, email)

        # Create user with admin privileges
        users_service.create_user(
            telegram_user_id=telegram_user_id,
            email=email,
            is_admin=True,
            is_verified=True,
            is_approved=True
        )

        logger.info("Successfully created admin user %s with email %s", telegram_user_id, email)
        return True

    except Exception:
        logger.exception("Error creating admin user: ")
        return False

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) != 3:
        logger.error("Usage: python src/util/create_admin.py <telegram_user_id> <email>")
        logger.error("Example: python src/util/create_admin.py 123456789 admin@example.com")
        sys.exit(1)

    telegram_user_id = sys.argv[1]
    email = sys.argv[2]

    # Validate inputs
    if not telegram_user_id.isdigit():
        logger.error("telegram_user_id must be a number, got: %s", telegram_user_id)
        sys.exit(1)

    if "@" not in email or "." not in email:
        logger.error("email must be a valid email address, got: %s", email)
        sys.exit(1)

    logger.info("Setting up admin user: telegram_user_id=%s email=%s", telegram_user_id, email)

    # Create admin user
    success = create_admin(telegram_user_id, email)

    if success:
        logger.info("Admin user created successfully for %s", telegram_user_id)
        logger.info("User can now use admin commands: /admin users|approve|reject|pending")
    else:
        logger.error("Failed to create admin user for %s. Check logs above for details.", telegram_user_id)
        sys.exit(1)

if __name__ == "__main__":
    main()

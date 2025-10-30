"""
Authentication and Authorization Utilities
-----------------------------------------

JWT token management, password hashing, and role-based
access control utilities.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.api.services.webui_app_service import webui_app_service
from src.data.db.models.model_users import User
from src.data.db.models.model_webui import WebUIAuditLog
from src.data.db.services.database_service import get_database_service
from src.data.db.services import users_service
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# JWT Configuration
SECRET_KEY = "your-secret-key-change-in-production"  # TODO: Move to environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Security scheme
security = HTTPBearer()


class AuthenticationError(Exception):
    """Custom authentication error."""
    pass


class AuthorizationError(Exception):
    """Custom authorization error."""
    pass


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token.

    Args:
        data: Token payload data
        expires_delta: Token expiration time

    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """
    Create JWT refresh token.

    Args:
        data: Token payload data

    Returns:
        str: Encoded JWT refresh token
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode JWT token.

    Args:
        token: JWT token to verify

    Returns:
        Dict: Decoded token payload

    Raises:
        AuthenticationError: If token is invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.PyJWTError:
        raise AuthenticationError("Invalid token")


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email."""
    return db.query(User).filter(User.email == email).first()

def get_user_by_telegram_id(telegram_user_id: str) -> Optional[User]:
    """Get user by telegram ID - wrapper for service call."""
    return users_service.get_user_by_telegram_id(telegram_user_id)

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """
    Authenticate user with email or telegram_user_id and password.

    Args:
        db: Database session
        username: Email or telegram_user_id
        password: Plain text password (temporary: empty or "temp")

    Returns:
        User: Authenticated user or None
    """
    # Try to find user by email first
    user = get_user_by_email(db, username)

    # If not found by email, try by telegram_user_id using the service
    if not user:
        # Check if username looks like telegram format or is numeric
        if username.startswith("telegram_"):
            telegram_id = username.replace("telegram_", "")
            user = get_user_by_telegram_id(telegram_id)
        elif username.isdigit():
            user = get_user_by_telegram_id(username)

    if not user:
        return None

    if not user.is_active:
        return None

    if not user.verify_password(password):
        return None

    # Update last login
    user.last_login = datetime.now(timezone.utc)
    db.commit()

    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    Get current authenticated user from JWT token.

    Args:
        credentials: HTTP authorization credentials

    Returns:
        User: Current authenticated user

    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Verify token
        payload = verify_token(credentials.credentials)

        # Check token type
        if payload.get("type") != "access":
            raise AuthenticationError("Invalid token type")

        # Get user information from token
        user_id = payload.get("sub")
        username = payload.get("username")
        if user_id is None:
            raise AuthenticationError("Invalid token payload")

        # Get user from database to ensure it exists and has proper foreign key
        db_service = get_database_service()

        with db_service.uow() as r:
            user = r.s.query(User).filter(User.id == int(user_id)).first()

            if not user:
                # If user doesn't exist by ID, try to find by email/username
                email = f"{username}@trading-system.local" if username else None
                if email:
                    user = r.s.query(User).filter(User.email == email).first()

                if not user:
                    raise AuthenticationError("User not found in database")

            if not user.is_active:
                raise AuthenticationError("User account is inactive")

            # Create a new User object with all the data loaded
            # This avoids session-related issues by creating a detached object
            detached_user = User(
                id=user.id,
                email=user.email,
                role=user.role,
                is_active=user.is_active,
                created_at=user.created_at,
                updated_at=user.updated_at,
                last_login=user.last_login
            )

            return detached_user

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        _logger.exception("Authentication error:")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_role(required_roles: list[str]):
    """
    Decorator to require specific roles for endpoint access.

    Args:
        required_roles: List of required roles

    Returns:
        Dependency function
    """
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {', '.join(required_roles)}"
            )
        return current_user

    return role_checker


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require admin role."""
    return require_role(["admin"])(current_user)


def require_trader_or_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require trader or admin role."""
    return require_role(["trader", "admin"])(current_user)


def log_user_action(
    user: User,
    action: str,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
):
    """
    Log user action for audit purposes.

    Args:
        user: User performing the action
        action: Action description
        resource_type: Type of resource being acted upon
        resource_id: ID of the resource
        details: Additional action details
        ip_address: User's IP address
        user_agent: User's browser/client info
    """
    try:
        webui_app_service.log_user_action(
            user_id=user.id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent
        )

    except Exception as e:
        _logger.exception("Failed to log user action:")
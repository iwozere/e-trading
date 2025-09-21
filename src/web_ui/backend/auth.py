"""
Authentication and Authorization Utilities
-----------------------------------------

JWT token management, password hashing, and role-based
access control utilities.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.web_ui.backend.database import get_db
from src.web_ui.backend.models import User, AuditLog
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
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

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
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

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
    except jwt.JWTError:
        raise AuthenticationError("Invalid token")


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
    user = db.query(User).filter(User.email == username).first()

    # If not found by email, try by telegram_user_id
    if not user:
        # Check if username looks like telegram format or is numeric
        if username.startswith("telegram_"):
            telegram_id = username.replace("telegram_", "")
            user = db.query(User).filter(User.telegram_user_id == telegram_id).first()
        elif username.isdigit():
            user = db.query(User).filter(User.telegram_user_id == username).first()

    if not user:
        return None

    if not user.is_active:
        return None

    if not user.verify_password(password):
        return None

    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()

    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token.

    Args:
        credentials: HTTP authorization credentials
        db: Database session

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

        # Get user from database
        user_id = payload.get("sub")
        if user_id is None:
            raise AuthenticationError("Invalid token payload")

        user = db.query(User).filter(User.id == int(user_id)).first()
        if user is None:
            raise AuthenticationError("User not found")

        if not user.is_active:
            raise AuthenticationError("User account is disabled")

        return user

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        _logger.error("Authentication error: %s", e)
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
    db: Session,
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
        db: Database session
        user: User performing the action
        action: Action description
        resource_type: Type of resource being acted upon
        resource_id: ID of the resource
        details: Additional action details
        ip_address: User's IP address
        user_agent: User's browser/client info
    """
    try:
        audit_log = AuditLog(
            user_id=user.id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent
        )

        db.add(audit_log)
        db.commit()

    except Exception as e:
        _logger.error("Failed to log user action: %s", e)
        db.rollback()
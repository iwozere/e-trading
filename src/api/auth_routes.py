"""
Authentication API Routes
------------------------

FastAPI routes for user authentication, registration,
and token management.
"""

from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.api.services.webui_app_service import webui_app_service
from src.data.db.models.model_users import User
from src.api.auth import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    verify_token,
    get_current_user,
    log_user_action,
    AuthenticationError,
    security
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Create router
router = APIRouter(prefix="/auth", tags=["authentication"])


# Pydantic models
class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""
    refresh_token: str


class UserResponse(BaseModel):
    """User response model."""
    id: int
    username: str
    email: Optional[str]
    role: str
    is_active: bool
    created_at: Optional[str]
    last_login: Optional[str]


# TODO: Add 2FA verification models
# class SendVerificationCodeRequest(BaseModel):
# class VerifyCodeRequest(BaseModel)


# Authentication endpoints

@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    login_data: LoginRequest
):
    """
    Authenticate user and return JWT tokens.

    Args:
        request: FastAPI request object
        login_data: Login credentials
        db: Database session

    Returns:
        LoginResponse: JWT tokens and user info

    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Authenticate user
        user = webui_app_service.authenticate_user(login_data.username, login_data.password)

        if not user:
            # Log failed login attempt
            _logger.warning(
                "Failed login attempt for username: %s from IP: %s",
                login_data.username,
                request.client.host if request.client else "unknown"
            )

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Create tokens
        token_data = {"sub": str(user["id"]), "username": user.get("username") or user.get("email", ""), "role": user["role"]}
        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)

        # Log successful login
        webui_app_service.log_user_action(
            user_id=user["id"],
            action="login",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )

        _logger.info("User %s logged in successfully", user.get("username") or user.get("email", "unknown"))

        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=30 * 60,  # 30 minutes in seconds
            user=user
        )

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Login error:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/refresh", response_model=LoginResponse)
async def refresh_token(
    request: Request,
    refresh_data: RefreshTokenRequest
):
    """
    Refresh access token using refresh token.

    Args:
        request: FastAPI request object
        refresh_data: Refresh token data
        db: Database session

    Returns:
        LoginResponse: New JWT tokens

    Raises:
        HTTPException: If refresh fails
    """
    try:
        # Verify refresh token
        payload = verify_token(refresh_data.refresh_token)

        # Check token type
        if payload.get("type") != "refresh":
            raise AuthenticationError("Invalid token type")

        # Get user
        user_id = payload.get("sub")
        if not user_id:
            raise AuthenticationError("Invalid token payload")

        # For now, we'll use a simple approach since we don't have user lookup by ID
        # In a real implementation, this would use the users service
        user = {
            "id": int(user_id),
            "username": payload.get("username", "unknown"),
            "role": payload.get("role", "viewer"),
            "is_active": True
        }

        # Create new tokens
        token_data = {"sub": str(user["id"]), "username": user.get("username") or user.get("email", ""), "role": user["role"]}
        access_token = create_access_token(token_data)
        new_refresh_token = create_refresh_token(token_data)

        # Log token refresh
        webui_app_service.log_user_action(
            user_id=user["id"],
            action="token_refresh",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )

        return LoginResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=30 * 60,  # 30 minutes in seconds
            user=user
        )

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        _logger.exception("Token refresh error:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/logout")
async def logout(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """
    Logout user (invalidate tokens on client side).

    Args:
        request: FastAPI request object
        current_user: Current authenticated user
        db: Database session

    Returns:
        dict: Success message
    """
    try:
        # Log logout action
        webui_app_service.log_user_action(
            user_id=current_user.id,
            action="logout",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )

        _logger.info("User %s logged out", current_user.username or current_user.email or f"ID:{current_user.id}")

        return {"message": "Successfully logged out"}

    except Exception as e:
        _logger.exception("Logout error:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current user information.

    Args:
        current_user: Current authenticated user

    Returns:
        UserResponse: Current user info
    """
    return UserResponse(**current_user.to_dict())


# TODO: Implement 2FA endpoints for email/telegram verification
# @router.post("/send-verification-code")
# @router.post("/verify-code")
# These will replace password-based authentication
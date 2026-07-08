"""
Authentication API Routes
------------------------

FastAPI routes for user authentication, registration,
and token management.
"""

from datetime import datetime, timezone
import secrets
import sys
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.auth import (
    AuthenticationError,
    create_access_token,
    create_refresh_token,
    get_current_user,
    revoke_token,
    security,
    verify_token,
)
from src.api.rate_limiter import limiter
from src.api.services.webui_app_service import webui_app_service
from src.data.db.models.model_users import AuthIdentity, User, VerificationCode
from src.data.db.services.database_service import get_database_service
from src.notification.logger import setup_logger
from src.notification.service.client import NotificationServiceClient

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
    email: str | None
    role: str
    is_active: bool
    created_at: str | None
    last_login: str | None


_CODE_TTL_SECONDS = 600  # 10-minute window for verification codes


class ChangePasswordRequest(BaseModel):
    """Request model for password change."""

    current_password: str
    new_password: str


class ResetPasswordRequest(BaseModel):
    """Request model for requesting password reset code."""

    identity: str


class ConfirmResetRequest(BaseModel):
    """Request model for confirming password reset."""

    identity: str
    code: str
    new_password: str


class SendCodeRequest(BaseModel):
    """Request body for POST /auth/2fa/send."""

    channel: Literal["telegram", "email"]


class VerifyCodeRequest(BaseModel):
    """Request body for POST /auth/2fa/verify."""

    code: str


class VerifyCodeResponse(BaseModel):
    """Response returned after successful 2FA verification."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


# Authentication endpoints


@router.post("/login", response_model=LoginResponse)
@limiter.limit("5/minute")
async def login(request: Request, login_data: LoginRequest):
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
                request.client.host if request.client else "unknown",
            )

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Create tokens
        token_data = {
            "sub": str(user["id"]),
            "username": user.get("username") or user.get("email", ""),
            "role": user["role"],
        }
        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)

        # Log successful login
        webui_app_service.log_user_action(
            user_id=user["id"],
            action="login",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

        _logger.info("User %s logged in successfully", user.get("username") or user.get("email", "unknown"))

        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=30 * 60,  # 30 minutes in seconds
            user=user,
        )

    except HTTPException:
        raise
    except Exception:
        _logger.exception("Login error:")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post("/refresh", response_model=LoginResponse)
@limiter.limit("10/minute")
async def refresh_token(request: Request, refresh_data: RefreshTokenRequest):
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

        # Look up the user in the DB to get the current role (not the stale JWT role)
        db_service = get_database_service()
        with db_service.uow() as r:
            db_user = r.s.query(User).filter(User.id == int(user_id)).first()

        if not db_user or not db_user.is_active:
            raise AuthenticationError("User not found or inactive")

        user = {
            "id": db_user.id,
            "username": db_user.username,
            "email": db_user.email,
            "role": db_user.role,
            "is_active": db_user.is_active,
        }

        # Create new tokens
        token_data = {
            "sub": str(user["id"]),
            "username": user.get("username") or user.get("email", ""),
            "role": user["role"],
        }
        access_token = create_access_token(token_data)
        new_refresh_token = create_refresh_token(token_data)

        # Log token refresh
        webui_app_service.log_user_action(
            user_id=user["id"],
            action="token_refresh",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

        return LoginResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=30 * 60,  # 30 minutes in seconds
            user=user,
        )

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception:
        _logger.exception("Token refresh error:")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post("/logout")
async def logout(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    current_user: User = Depends(get_current_user),
):
    """
    Logout user — revokes the current access token server-side.

    Args:
        request: FastAPI request object
        credentials: Raw bearer credentials (used to revoke the token)
        current_user: Current authenticated user

    Returns:
        dict: Success message
    """
    try:
        # Revoke the current access token so it cannot be reused
        try:
            payload = verify_token(credentials.credentials)
            jti = payload.get("jti")
            exp = payload.get("exp")
            if jti and exp:
                revoke_token(jti, float(exp))
        except AuthenticationError:
            pass  # Already expired — nothing to revoke

        # Log logout action
        webui_app_service.log_user_action(
            user_id=current_user.id,
            action="logout",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

        _logger.info("User %s logged out", current_user.username or current_user.email or f"ID:{current_user.id}")

        return {"message": "Successfully logged out"}

    except Exception:
        _logger.exception("Logout error:")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


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


@router.post("/2fa/send")
@limiter.limit("3 per 15 minutes")
async def send_2fa_code(
    request: Request,
    body: SendCodeRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Generate and dispatch a 2FA verification code via email or Telegram.

    Args:
        request: FastAPI request (required by rate limiter)
        body: Channel selection — "telegram" or "email"
        current_user: Currently authenticated user

    Returns:
        Confirmation message

    Raises:
        HTTPException 400: No linked account for the requested channel
        HTTPException 500: Notification delivery failed
    """
    try:
        db_service = get_database_service()
        telegram_chat_id: int | None = None

        # Step 1: read — validate that the requested channel is available for this user
        with db_service.uow() as r:
            if body.channel == "telegram":
                ident = (
                    r.s.query(AuthIdentity)
                    .filter(
                        AuthIdentity.user_id == current_user.id,
                        AuthIdentity.provider == "telegram",
                    )
                    .first()
                )
                if ident is None:
                    telegram_chat_id = None
                else:
                    telegram_chat_id = int(ident.external_id)

        if body.channel == "telegram" and telegram_chat_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No Telegram account linked to this user",
            )
        if body.channel == "email" and not current_user.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No email address on file for this user",
            )

        # Step 2: write — atomically replace any existing code and store the new one
        code = f"{secrets.randbelow(10**6):06d}"
        with db_service.uow() as r:
            r.s.query(VerificationCode).filter(
                VerificationCode.user_id == current_user.id,
                VerificationCode.provider == body.channel,
            ).delete(synchronize_session=False)
            r.s.add(
                VerificationCode(
                    user_id=current_user.id,
                    code=code,
                    sent_time=int(time.time()),
                    provider=body.channel,
                )
            )

        # Step 3: dispatch notification (queues into msg_messages for the processor)
        async with NotificationServiceClient(service_url="database://") as notify_client:
            await notify_client.send_notification(
                notification_type="system",
                title="Your verification code",
                message=f"Your 2FA code is: {code}\nValid for 10 minutes.",
                channels=[body.channel],
                telegram_chat_id=telegram_chat_id,
                email_receiver=current_user.email if body.channel == "email" else None,
                recipient_id=str(current_user.id),
            )

        _logger.info("2FA code sent to user %s via %s", current_user.id, body.channel)
        return {"detail": "Code sent"}

    except HTTPException:
        raise
    except Exception:
        _logger.exception("Error sending 2FA code for user %s:", current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send verification code",
        )


@router.post("/2fa/verify", response_model=VerifyCodeResponse)
@limiter.limit("5 per 15 minutes")
async def verify_2fa_code(
    request: Request,
    body: VerifyCodeRequest,
    current_user: User = Depends(get_current_user),
) -> VerifyCodeResponse:
    """
    Validate a 2FA code and return a JWT with the 2fa_verified claim.

    The code is consumed on first successful use (replay prevention).
    Expired codes and wrong codes both return the same 400 error to
    prevent timing-based oracle attacks.

    Args:
        request: FastAPI request (required by rate limiter)
        body: The verification code entered by the user
        current_user: Currently authenticated user

    Returns:
        New access token with 2fa_verified=True in the payload

    Raises:
        HTTPException 400: Code is invalid, expired, or already used
        HTTPException 500: Unexpected server error
    """
    try:
        db_service = get_database_service()
        valid = False

        with db_service.uow() as r:
            code_row = (
                r.s.query(VerificationCode)
                .filter(VerificationCode.user_id == current_user.id)
                .order_by(VerificationCode.sent_time.desc())
                .first()
            )
            if code_row is not None:
                now = int(time.time())
                not_expired = (now - code_row.sent_time) <= _CODE_TTL_SECONDS
                valid = (code_row.code == body.code) and not_expired
                if valid:
                    r.s.delete(code_row)  # consume on first use; prevents replay

        if not valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired verification code",
            )

        token_data = {
            "sub": str(current_user.id),
            "username": current_user.username or current_user.email or f"user_{current_user.id}",
            "role": current_user.role,
            "2fa_verified": True,
        }
        access_token = create_access_token(token_data)

        _logger.info("2FA verified successfully for user %s", current_user.id)
        return VerifyCodeResponse(access_token=access_token, expires_in=30 * 60)

    except HTTPException:
        raise
    except Exception:
        _logger.exception("Error verifying 2FA code for user %s:", current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/change-password")
async def change_password(
    request: Request,
    body: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Allow logged-in users to change their password from the Web UI.
    """
    special_emails = {"admin@trading-system.local", "trader@trading-system.local", "viewer@trading-system.local"}
    if current_user.email in special_emails:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Password changes are not permitted for system-level accounts.",
        )

    db_service = get_database_service()
    with db_service.uow() as r:
        db_user = r.s.query(User).filter(User.id == current_user.id).first()
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        if db_user.password_hash:
            if not db_user.verify_password(body.current_password):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Incorrect current password",
                )

        db_user.set_password(body.new_password)
        db_user.updated_at = datetime.now(timezone.utc)  # type: ignore

        webui_app_service.log_user_action(
            user_id=db_user.id,
            action="change_password",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

    _logger.info("User %s changed password successfully", current_user.email or f"ID:{current_user.id}")
    return {"detail": "Password updated successfully"}


@router.post("/reset-password/request")
@limiter.limit("5/minute")
async def request_password_reset(
    request: Request,
    body: ResetPasswordRequest,
) -> dict:
    """
    Request a password reset verification code via Email or Telegram.
    """
    identity = body.identity.strip()
    if not identity:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Identity identifier is required",
        )

    clean_identity = identity
    if clean_identity.startswith("telegram_"):
        clean_identity = clean_identity.replace("telegram_", "")

    db_service = get_database_service()
    user = None
    telegram_chat_id = None

    with db_service.uow() as r:
        user = r.s.query(User).filter(User.email == clean_identity).first()

        if not user:
            ident = r.s.query(AuthIdentity).filter(
                AuthIdentity.provider == "telegram",
                AuthIdentity.external_id == clean_identity
            ).first()
            if ident:
                user = r.s.query(User).filter(User.id == ident.user_id).first()
                telegram_chat_id = int(ident.external_id)
        else:
            ident = r.s.query(AuthIdentity).filter(
                AuthIdentity.user_id == user.id,
                AuthIdentity.provider == "telegram"
            ).first()
            if ident:
                telegram_chat_id = int(ident.external_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No account found with that email or Telegram ID",
        )

    special_emails = {"admin@trading-system.local", "trader@trading-system.local", "viewer@trading-system.local"}
    if user.email in special_emails:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password reset is not permitted for default system accounts.",
        )

    channel = None
    if "@" in identity:
        channel = "email"
    elif identity.isdigit() or identity.startswith("telegram_"):
        if telegram_chat_id is not None:
            channel = "telegram"

    if not channel:
        if telegram_chat_id is not None:
            channel = "telegram"
        elif user.email:
            channel = "email"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User has no linked email or Telegram channel to send reset code",
            )

    if channel == "telegram" and telegram_chat_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No Telegram chat ID linked to this account",
        )
    if channel == "email" and not user.email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No email address linked to this account",
        )

    code = f"{secrets.randbelow(10**6):06d}"

    with db_service.uow() as r:
        r.s.query(VerificationCode).filter(
            VerificationCode.user_id == user.id,
            VerificationCode.provider == f"reset_{channel}",
        ).delete(synchronize_session=False)

        r.s.add(
            VerificationCode(
                user_id=user.id,
                code=code,
                sent_time=int(time.time()),
                provider=f"reset_{channel}",
            )
        )

    try:
        async with NotificationServiceClient(service_url="database://") as notify_client:
            await notify_client.send_notification(
                notification_type="system",
                title="Your password reset code",
                message=f"Your password reset verification code is: {code}\nValid for 10 minutes.",
                channels=[channel],
                telegram_chat_id=telegram_chat_id,
                email_receiver=user.email if channel == "email" else None,
                recipient_id=str(user.id),
            )
    except Exception as exc:
        _logger.error("Failed to send reset notification: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send verification code. Please try again later.",
        )

    _logger.info("Password reset code sent to user %s via %s", user.id, channel)
    return {
        "detail": "Verification code sent successfully",
        "channel": channel,
        "recipient": user.email if channel == "email" else f"Telegram ({telegram_chat_id})",
    }


@router.post("/reset-password/confirm")
@limiter.limit("5/minute")
async def confirm_password_reset(
    request: Request,
    body: ConfirmResetRequest,
) -> dict:
    """
    Validate a password reset verification code and update password.
    """
    identity = body.identity.strip()
    if not identity:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Identity identifier is required",
        )

    clean_identity = identity
    if clean_identity.startswith("telegram_"):
        clean_identity = clean_identity.replace("telegram_", "")

    db_service = get_database_service()
    user = None

    with db_service.uow() as r:
        user = r.s.query(User).filter(User.email == clean_identity).first()

        if not user:
            ident = r.s.query(AuthIdentity).filter(
                AuthIdentity.provider == "telegram",
                AuthIdentity.external_id == clean_identity
            ).first()
            if ident:
                user = r.s.query(User).filter(User.id == ident.user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No account found with that email or Telegram ID",
        )

    special_emails = {"admin@trading-system.local", "trader@trading-system.local", "viewer@trading-system.local"}
    if user.email in special_emails:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password reset is not permitted for default system accounts.",
        )

    valid = False
    with db_service.uow() as r:
        code_rows = (
            r.s.query(VerificationCode)
            .filter(
                VerificationCode.user_id == user.id,
                VerificationCode.provider.like("reset_%")
            )
            .order_by(VerificationCode.sent_time.desc())
            .all()
        )

        now = int(time.time())
        matched_row = None
        for code_row in code_rows:
            if (now - code_row.sent_time) <= _CODE_TTL_SECONDS:
                if code_row.code == body.code.strip():
                    valid = True
                    matched_row = code_row
                    break

        if valid and matched_row:
            r.s.delete(matched_row)

            db_user = r.s.query(User).filter(User.id == user.id).first()
            if db_user:
                db_user.set_password(body.new_password)
                db_user.updated_at = datetime.now(timezone.utc)  # type: ignore

                webui_app_service.log_user_action(
                    user_id=db_user.id,
                    action="reset_password",
                    ip_address=request.client.host if request.client else None,
                    user_agent=request.headers.get("user-agent"),
                )

    if not valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification code",
        )

    _logger.info("Password reset successfully for user %s", user.id)
    return {"detail": "Password reset successfully. You can now log in with your new password."}

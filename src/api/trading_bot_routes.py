"""
Trading Bot Management API Routes (FastAPI)
-------------------------------------------

FastAPI routes for managing trading bots through the web UI.
Provides CRUD operations for bot configurations and status management.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.services import trading_service
from src.trading.services.bot_config_validator import (
    validate_database_bot_record
)
from src.web_ui.config.trading_bot_config import (
    get_entry_mixin_parameters,
    get_exit_mixin_parameters,
    get_available_entry_mixins,
    get_available_exit_mixins,
    parameter_definition_to_dict,
    BROKER_TYPES,
    TRADING_MODES,
    DATA_SOURCES,
    INTERVALS,
    SYMBOLS
)
from src.api.auth import get_current_user
from src.data.db.models.model_users import User
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Create router
router = APIRouter(prefix="/api/trading", tags=["trading-bots"])


# Pydantic models for request/response
class BotConfigRequest(BaseModel):
    """Request model for bot configuration."""
    name: str = Field(..., description="Bot name")
    symbol: str = Field(..., description="Trading symbol")
    description: Optional[str] = Field(None, description="Bot description")
    enabled: bool = Field(True, description="Whether bot is enabled")
    broker: Dict[str, Any] = Field(..., description="Broker configuration")
    strategy: Dict[str, Any] = Field(..., description="Strategy configuration")
    data: Dict[str, Any] = Field(..., description="Data configuration")
    trading: Dict[str, Any] = Field(..., description="Trading configuration")
    risk_management: Dict[str, Any] = Field(..., description="Risk management configuration")
    notifications: Dict[str, Any] = Field(..., description="Notification configuration")


class BotStatusRequest(BaseModel):
    """Request model for bot status update."""
    action: str = Field(..., description="Action to perform", pattern="^(start|stop|restart)$")


class BotResponse(BaseModel):
    """Response model for bot operations."""
    success: bool
    bot: Optional[Dict[str, Any]] = None
    bots: Optional[List[Dict[str, Any]]] = None
    count: Optional[int] = None
    message: Optional[str] = None
    error: Optional[str] = None
    details: Optional[str] = None
    validation_errors: Optional[List[str]] = None
    validation_warnings: Optional[List[str]] = None


class ConfigOptionsResponse(BaseModel):
    """Response model for configuration options."""
    success: bool
    options: Optional[Dict[str, Any]] = None
    entry_mixins: Optional[List[Dict[str, Any]]] = None
    exit_mixins: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class ValidationResponse(BaseModel):
    """Response model for validation results."""
    success: bool
    validation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.get("/bots", response_model=BotResponse)
async def get_trading_bots(current_user: User = Depends(get_current_user)):
    """
    Get all trading bots for the current user.

    Returns:
        JSON list of bot configurations with status information
    """
    try:
        user_id = current_user.id

        # Get bots for user
        bots = trading_service.get_enabled_bots(user_id)

        # Add additional status information
        for bot in bots:
            # Calculate runtime if bot is running
            if bot['status'] == 'running' and bot['started_at']:
                runtime_seconds = (datetime.now() - bot['started_at']).total_seconds()
                bot['runtime_hours'] = round(runtime_seconds / 3600, 1)
            else:
                bot['runtime_hours'] = 0

            # Format currency values
            if bot['current_balance']:
                bot['current_balance_formatted'] = f"${bot['current_balance']:,.2f}"
            if bot['total_pnl']:
                bot['total_pnl_formatted'] = f"${bot['total_pnl']:+,.2f}"
                bot['pnl_color'] = 'green' if bot['total_pnl'] >= 0 else 'red'

        return BotResponse(
            success=True,
            bots=bots,
            count=len(bots)
        )

    except Exception as e:
        _logger.exception("Error getting trading bots")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve trading bots: {str(e)}"
        )


@router.get("/bots/{bot_id}", response_model=BotResponse)
async def get_trading_bot(bot_id: str, current_user: User = Depends(get_current_user)):
    """
    Get a specific trading bot by ID.

    Args:
        bot_id: Bot ID

    Returns:
        JSON bot configuration
    """
    try:
        bot = trading_service.get_bot_by_id(bot_id)

        if not bot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Bot not found"
            )

        # Verify ownership
        if bot['user_id'] != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        return BotResponse(success=True, bot=bot)

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error getting trading bot %s", bot_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve trading bot: {str(e)}"
        )


@router.post("/bots", response_model=BotResponse)
async def create_trading_bot(
    config: BotConfigRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Create a new trading bot.

    Returns:
        JSON with created bot information
    """
    try:
        user_id = current_user.id
        config_dict = config.dict()

        # Generate bot ID if not provided
        if 'id' not in config_dict:
            config_dict['id'] = f"bot_{user_id}_{int(datetime.now().timestamp())}"

        # Validate configuration
        bot_record = {
            'id': None,  # Will be auto-generated by database
            'user_id': user_id,
            'type': config_dict.get('broker', {}).get('trading_mode', 'paper'),
            'status': 'stopped',
            'config': config_dict,
            'description': config_dict.get('description', ''),
            'started_at': None,
            'last_heartbeat': None,
            'error_count': 0,
            'current_balance': None,
            'total_pnl': None,
            'extra_metadata': None,
            'created_at': datetime.now().isoformat(),
            'updated_at': None
        }

        is_valid, errors, warnings = validate_database_bot_record(bot_record)

        if not is_valid:
            return BotResponse(
                success=False,
                error="Invalid bot configuration",
                validation_errors=errors,
                validation_warnings=warnings
            )

        # Create bot in database
        bot_data = {
            'user_id': user_id,
            'type': config_dict.get('broker', {}).get('trading_mode', 'paper'),
            'status': 'stopped',
            'config': config_dict,
            'description': config_dict.get('description', '')
        }

        result = trading_service.upsert_bot(bot_data)

        _logger.info("Created trading bot: %s for user %s", config_dict.get('name'), user_id)

        return BotResponse(
            success=True,
            bot=result,
            validation_warnings=warnings
        )

    except Exception as e:
        _logger.exception("Error creating trading bot")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create trading bot: {str(e)}"
        )


@router.put("/bots/{bot_id}", response_model=BotResponse)
async def update_trading_bot(
    bot_id: str,
    config: BotConfigRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Update an existing trading bot configuration.

    Args:
        bot_id: Bot ID to update

    Returns:
        JSON with updated bot information
    """
    try:
        user_id = current_user.id
        config_dict = config.dict()

        # Get existing bot to verify ownership
        existing_bot = trading_service.get_bot_by_id(bot_id)
        if not existing_bot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Bot not found"
            )

        if existing_bot['user_id'] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        # Validate new configuration
        bot_record = {
            'id': bot_id,
            'user_id': user_id,
            'type': config_dict.get('broker', {}).get('trading_mode', 'paper'),
            'status': existing_bot['status'],  # Keep current status
            'config': config_dict,
            'description': config_dict.get('description', ''),
            'started_at': existing_bot['started_at'],
            'last_heartbeat': existing_bot['last_heartbeat'],
            'error_count': existing_bot['error_count'],
            'current_balance': existing_bot['current_balance'],
            'total_pnl': existing_bot['total_pnl'],
            'extra_metadata': existing_bot['extra_metadata'],
            'created_at': existing_bot['created_at'],
            'updated_at': datetime.now().isoformat()
        }

        is_valid, errors, warnings = validate_database_bot_record(bot_record)

        if not is_valid:
            return BotResponse(
                success=False,
                error="Invalid bot configuration",
                validation_errors=errors,
                validation_warnings=warnings
            )

        # Update bot in database
        bot_data = {
            'id': bot_id,
            'user_id': user_id,
            'type': config_dict.get('broker', {}).get('trading_mode', 'paper'),
            'status': existing_bot['status'],
            'config': config_dict,
            'description': config_dict.get('description', '')
        }

        result = trading_service.upsert_bot(bot_data)

        _logger.info("Updated trading bot %s for user %s", bot_id, user_id)

        return BotResponse(
            success=True,
            bot=result,
            validation_warnings=warnings
        )

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error updating trading bot %s", bot_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update trading bot: {str(e)}"
        )


@router.put("/bots/{bot_id}/status", response_model=BotResponse)
async def update_bot_status(
    bot_id: str,
    status_request: BotStatusRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Update bot status (start/stop/restart).

    Args:
        bot_id: Bot ID
        status_request: Status update request

    Returns:
        JSON with operation result
    """
    try:
        user_id = current_user.id
        action = status_request.action

        # Verify bot ownership
        bot = trading_service.get_bot_by_id(bot_id)
        if not bot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Bot not found"
            )

        if bot['user_id'] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        # Update status based on action
        if action == 'start':
            if bot['status'] == 'running':
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Bot is already running"
                )

            success = trading_service.update_bot_status(bot_id, 'starting')
            # TODO: Actually start the bot via enhanced trading service

        elif action == 'stop':
            if bot['status'] == 'stopped':
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Bot is already stopped"
                )

            success = trading_service.update_bot_status(bot_id, 'stopping')
            # TODO: Actually stop the bot via enhanced trading service

        elif action == 'restart':
            success = trading_service.update_bot_status(bot_id, 'restarting')
            # TODO: Actually restart the bot via enhanced trading service

        if success:
            _logger.info("Bot %s status updated to %s by user %s", bot_id, action, user_id)
            return BotResponse(
                success=True,
                message=f'Bot {action} initiated successfully'
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to {action} bot"
            )

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error updating bot %s status", bot_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update bot status: {str(e)}"
        )


@router.delete("/bots/{bot_id}", response_model=BotResponse)
async def delete_trading_bot(bot_id: str, current_user: User = Depends(get_current_user)):
    """
    Delete a trading bot.

    Args:
        bot_id: Bot ID to delete

    Returns:
        JSON with operation result
    """
    try:
        user_id = current_user.id

        # Verify bot ownership
        bot = trading_service.get_bot_by_id(bot_id)
        if not bot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Bot not found"
            )

        if bot['user_id'] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        # Don't allow deletion of running bots
        if bot['status'] == 'running':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete running bot. Stop it first."
            )

        # Mark bot as disabled instead of actual deletion
        success = trading_service.update_bot_status(bot_id, 'disabled')

        if success:
            _logger.info("Bot %s deleted by user %s", bot_id, user_id)
            return BotResponse(
                success=True,
                message='Bot deleted successfully'
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete bot"
            )

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error deleting bot %s", bot_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete bot: {str(e)}"
        )


@router.get("/config/mixins", response_model=ConfigOptionsResponse)
async def get_mixin_configurations():
    """
    Get available mixins and their parameter definitions.

    Returns:
        JSON with entry and exit mixins and their parameters
    """
    try:
        entry_mixins = get_available_entry_mixins()
        exit_mixins = get_available_exit_mixins()

        # Add parameter definitions for each mixin
        for mixin in entry_mixins:
            mixin_name = mixin['value']
            params = get_entry_mixin_parameters(mixin_name)
            mixin['parameters'] = [parameter_definition_to_dict(p) for p in params]

        for mixin in exit_mixins:
            mixin_name = mixin['value']
            params = get_exit_mixin_parameters(mixin_name)
            mixin['parameters'] = [parameter_definition_to_dict(p) for p in params]

        return ConfigOptionsResponse(
            success=True,
            entry_mixins=entry_mixins,
            exit_mixins=exit_mixins
        )

    except Exception as e:
        _logger.exception("Error getting mixin configurations")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get mixin configurations: {str(e)}"
        )


@router.get("/config/options", response_model=ConfigOptionsResponse)
async def get_configuration_options():
    """
    Get configuration options for dropdowns and selects.

    Returns:
        JSON with all configuration options
    """
    try:
        return ConfigOptionsResponse(
            success=True,
            options={
                'broker_types': BROKER_TYPES,
                'trading_modes': TRADING_MODES,
                'data_sources': DATA_SOURCES,
                'intervals': INTERVALS,
                'symbols': SYMBOLS,
                'entry_mixins': get_available_entry_mixins(),
                'exit_mixins': get_available_exit_mixins()
            }
        )

    except Exception as e:
        _logger.exception("Error getting configuration options")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get configuration options: {str(e)}"
        )


@router.post("/bots/{bot_id}/validate", response_model=ValidationResponse)
async def validate_bot_configuration(
    bot_id: str,
    config: BotConfigRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Validate a bot configuration without saving.

    Args:
        bot_id: Bot ID (for existing bots) or 0 for new bots
        config: Bot configuration to validate

    Returns:
        JSON with validation results
    """
    try:
        user_id = current_user.id
        config_dict = config.dict()

        # Create validation record
        bot_record = {
            'id': bot_id if bot_id > 0 else None,
            'user_id': user_id,
            'type': config_dict.get('broker', {}).get('trading_mode', 'paper'),
            'status': 'stopped',
            'config': config_dict,
            'description': config_dict.get('description', ''),
            'started_at': None,
            'last_heartbeat': None,
            'error_count': 0,
            'current_balance': None,
            'total_pnl': None,
            'extra_metadata': None,
            'created_at': datetime.now().isoformat(),
            'updated_at': None
        }

        is_valid, errors, warnings = validate_database_bot_record(bot_record)

        return ValidationResponse(
            success=True,
            validation={
                'is_valid': is_valid,
                'errors': errors,
                'warnings': warnings
            }
        )

    except Exception as e:
        _logger.exception("Error validating bot configuration")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate configuration: {str(e)}"
        )
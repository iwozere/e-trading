#!/usr/bin/env python3
"""
Trading Web UI Backend - FastAPI Application
-------------------------------------------

This is the main FastAPI application for the trading web UI backend.
It provides REST API endpoints and WebSocket communication for managing
the enhanced multi-strategy trading system.

Features:
- Strategy management API (CRUD operations)
- Real-time monitoring with WebSocket
- System administration endpoints
- Authentication and authorization
- Integration with enhanced trading system
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from contextlib import asynccontextmanager
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Setup logger first
from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

# Make trading system import optional for testing
try:
    from src.trading.strategy_manager import StrategyManager
    TRADING_SYSTEM_AVAILABLE = True
except ImportError as e:
    _logger.warning("Trading system not available: %s", e)
    StrategyManager = None
    TRADING_SYSTEM_AVAILABLE = False
from config.donotshare.donotshare import TRADING_API_PORT, TRADING_WEBGUI_PORT
from src.api.services.webui_app_service import webui_app_service
from src.api.auth_routes import router as auth_router
from src.api.telegram_routes import router as telegram_router
from src.api.jobs_routes import router as jobs_router
from src.api.notification_routes import router as notification_router
from src.api.auth import get_current_user, require_trader_or_admin
from src.data.db.models.model_users import User
from src.api.services import (
    StrategyManagementService, StrategyValidationError, StrategyOperationError,
    SystemMonitoringService
)

# Global strategy manager and service instances
strategy_manager: Optional[StrategyManager] = None
strategy_service: Optional[StrategyManagementService] = None
monitoring_service: Optional[SystemMonitoringService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global strategy_manager, strategy_service, monitoring_service

    # Startup
    _logger.info("Starting Trading Web UI Backend...")

    # Initialize database
    webui_app_service.init_database()
    _logger.info("Database initialized")

    # Initialize strategy manager if available
    if TRADING_SYSTEM_AVAILABLE:
        strategy_manager = StrategyManager()

        # Load existing strategies if config exists
        config_file = "config/enhanced_trading/raspberry_pi_multi_strategy.json"
        if Path(config_file).exists():
            await strategy_manager.load_strategies_from_config(config_file)
            _logger.info("Loaded existing strategy configurations")
    else:
        strategy_manager = None
        _logger.warning("Trading system not available - running in API-only mode")

    # Initialize strategy service
    strategy_service = StrategyManagementService(strategy_manager)
    _logger.info("Strategy management service initialized")

    # Initialize monitoring service
    monitoring_service = SystemMonitoringService()
    _logger.info("System monitoring service initialized")

    # Initialize heartbeat manager
    _logger.info("Initializing heartbeat manager...")
    try:
        from src.common.heartbeat_manager import HeartbeatManager

        def api_service_health_check():
            """Health check function for API service."""
            try:
                # Check if services are initialized
                strategy_service_healthy = strategy_service is not None
                monitoring_service_healthy = monitoring_service is not None
                trading_system_healthy = TRADING_SYSTEM_AVAILABLE and strategy_manager is not None

                if strategy_service_healthy and monitoring_service_healthy:
                    status = 'HEALTHY' if trading_system_healthy else 'DEGRADED'
                    error_msg = None if trading_system_healthy else 'Trading system not available'

                    return {
                        'status': status,
                        'error_message': error_msg,
                        'metadata': {
                            'strategy_service_initialized': strategy_service_healthy,
                            'monitoring_service_initialized': monitoring_service_healthy,
                            'trading_system_available': TRADING_SYSTEM_AVAILABLE,
                            'strategy_manager_initialized': strategy_manager is not None,
                            'api_port': TRADING_API_PORT,
                            'webgui_port': TRADING_WEBGUI_PORT
                        }
                    }
                else:
                    return {
                        'status': 'DOWN',
                        'error_message': 'Core services not initialized',
                        'metadata': {
                            'strategy_service_initialized': strategy_service_healthy,
                            'monitoring_service_initialized': monitoring_service_healthy,
                            'trading_system_available': TRADING_SYSTEM_AVAILABLE
                        }
                    }
            except Exception as e:
                return {
                    'status': 'DOWN',
                    'error_message': f'Health check failed: {str(e)}'
                }

        # Create and start heartbeat manager
        api_heartbeat_manager = HeartbeatManager(
            system='api_service',
            interval_seconds=30
        )
        api_heartbeat_manager.set_health_check_function(api_service_health_check)
        api_heartbeat_manager.start_heartbeat()

        _logger.info("Heartbeat manager started for API service")

    except Exception:
        _logger.exception("Failed to initialize heartbeat manager:")

    yield

    # Shutdown
    _logger.info("Shutting down Trading Web UI Backend...")

    # Stop heartbeat
    try:
        api_heartbeat_manager.stop_heartbeat()
        _logger.info("Stopped API service heartbeat")
    except:
        pass

    if strategy_manager:
        await strategy_manager.shutdown()

# Create FastAPI application
app = FastAPI(
    title="Trading Web UI API",
    description="REST API for managing multi-strategy trading system",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5002"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include authentication routes
app.include_router(auth_router)

# Include Telegram bot management routes
app.include_router(telegram_router)

# Include jobs and scheduling routes
app.include_router(jobs_router)

# Include notification management routes
app.include_router(notification_router)

# Include trading bot management routes
from src.api.trading_bot_routes import router as trading_bot_router
app.include_router(trading_bot_router)

# Note: Static files and HTML templates removed since using React frontend

# Add unified analytics endpoints
from src.api.services.unified_analytics_service import unified_analytics_service

@app.get("/api/analytics")
async def get_available_analytics(current_user: User = Depends(get_current_user)):
    """Get information about available analytics methods."""
    try:
        return unified_analytics_service.get_available_analytics()
    except Exception:
        _logger.exception("Error getting available analytics:")
        raise HTTPException(status_code=500, detail="Failed to get available analytics")

@app.get("/api/analytics/dashboard")
async def get_unified_dashboard_data(
    days: int = 30,
    current_user: User = Depends(get_current_user)
):
    """Get unified dashboard data combining notifications and trading analytics."""
    try:
        if days < 1 or days > 365:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 365")

        dashboard_data = await unified_analytics_service.get_unified_dashboard_data(days=days)
        return dashboard_data
    except HTTPException:
        raise
    except Exception:
        _logger.exception("Error getting unified dashboard data:")
        raise HTTPException(status_code=500, detail="Failed to get unified dashboard data")

@app.get("/api/analytics/correlation")
async def get_correlation_analysis(
    notification_metric: str = "success_rate",
    trading_metric: str = "win_rate",
    days: int = 30,
    current_user: User = Depends(get_current_user)
):
    """Analyze correlations between notification and trading metrics."""
    try:
        if days < 1 or days > 365:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 365")

        correlation_data = await unified_analytics_service.get_correlation_analysis(
            notification_metric=notification_metric,
            trading_metric=trading_metric,
            days=days
        )
        return correlation_data
    except HTTPException:
        raise
    except Exception:
        _logger.exception("Error getting correlation analysis:")
        raise HTTPException(status_code=500, detail="Failed to get correlation analysis")

# Security
security = HTTPBearer()

# Pydantic models for API
from pydantic import BaseModel, Field

class StrategyConfig(BaseModel):
    """Strategy configuration model."""
    id: str = Field(..., description="Unique strategy identifier")
    name: str = Field(..., description="Human-readable strategy name")
    enabled: bool = Field(True, description="Whether strategy is enabled")
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    broker: Dict[str, Any] = Field(..., description="Broker configuration")
    strategy: Dict[str, Any] = Field(..., description="Strategy parameters")
    data: Dict[str, Any] = Field(default_factory=dict, description="Data configuration")
    trading: Dict[str, Any] = Field(default_factory=dict, description="Trading settings")
    risk_management: Dict[str, Any] = Field(default_factory=dict, description="Risk management settings")
    notifications: Dict[str, Any] = Field(default_factory=dict, description="Notification settings")

class StrategyStatus(BaseModel):
    """Strategy status model."""
    instance_id: str
    name: str
    status: str
    uptime_seconds: float
    error_count: int
    last_error: Optional[str]
    broker_type: Optional[str]
    trading_mode: Optional[str]
    symbol: Optional[str]
    strategy_type: Optional[str]

class SystemStatus(BaseModel):
    """System status model."""
    service_name: str
    version: str
    status: str
    uptime_seconds: float
    active_strategies: int
    total_strategies: int
    system_metrics: Dict[str, Any]

class StrategyAction(BaseModel):
    """Strategy action model."""
    action: str = Field(..., description="Action to perform (start, stop, restart)")
    confirm_live_trading: bool = Field(False, description="Confirmation for live trading")

# Remove old authentication dependency - now using proper JWT auth from auth.py

# API Routes

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Trading Web UI API", "version": "1.0.0"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": asyncio.get_event_loop().time(),
        "trading_system_available": TRADING_SYSTEM_AVAILABLE
    }

@app.get("/api/test-auth")
async def test_auth(current_user: User = Depends(get_current_user)):
    """Test authentication endpoint."""
    return {
        "message": "Authentication successful",
        "user": current_user.to_dict()
    }

# Strategy Management Endpoints

@app.get("/api/strategies", response_model=List[StrategyStatus])
async def list_strategies(current_user: User = Depends(get_current_user)):
    """List all configured strategies."""
    try:
        strategies = strategy_service.get_all_strategies_status()
        return [StrategyStatus(**strategy) for strategy in strategies]
    except Exception as e:
        _logger.exception("Error listing strategies:")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategies", response_model=Dict[str, str])
async def create_strategy(
    strategy_config: StrategyConfig,
    current_user: User = Depends(require_trader_or_admin)
):
    """Create a new strategy."""
    try:
        # Convert Pydantic model to dict
        config_dict = strategy_config.model_dump()

        # Create strategy using service
        result = await strategy_service.create_strategy(config_dict)

        return {
            "message": result["message"],
            "strategy_id": result["strategy_id"]
        }

    except StrategyValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except StrategyOperationError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        _logger.exception("Error creating strategy:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/strategies/{strategy_id}", response_model=StrategyStatus)
async def get_strategy(strategy_id: str, current_user: User = Depends(get_current_user)):
    """Get details of a specific strategy."""
    try:
        status = strategy_service.get_strategy_status(strategy_id)
        if not status:
            raise HTTPException(status_code=404, detail="Strategy not found")

        return StrategyStatus(**status)

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error getting strategy %s:", strategy_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/strategies/{strategy_id}", response_model=Dict[str, str])
async def update_strategy(
    strategy_id: str,
    strategy_config: StrategyConfig,
    current_user: User = Depends(require_trader_or_admin)
):
    """Update an existing strategy."""
    try:
        # Convert Pydantic model to dict
        config_dict = strategy_config.model_dump()

        # Update strategy using service
        result = await strategy_service.update_strategy(strategy_id, config_dict)

        return {
            "message": result["message"],
            "strategy_id": result["strategy_id"]
        }

    except StrategyValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except StrategyOperationError as e:
        raise HTTPException(status_code=404 if "not found" in str(e).lower() else 503, detail=str(e))
    except Exception as e:
        _logger.exception("Error updating strategy %s:", strategy_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/strategies/{strategy_id}", response_model=Dict[str, str])
async def delete_strategy(strategy_id: str, current_user: User = Depends(require_trader_or_admin)):
    """Delete a strategy."""
    try:
        # Delete strategy using service
        result = await strategy_service.delete_strategy(strategy_id)

        return {
            "message": result["message"],
            "strategy_id": result["strategy_id"]
        }

    except StrategyOperationError as e:
        raise HTTPException(status_code=404 if "not found" in str(e).lower() else 400, detail=str(e))
    except Exception as e:
        _logger.exception("Error deleting strategy %s:", strategy_id)
        raise HTTPException(status_code=500, detail=str(e))

# Strategy Lifecycle Endpoints

@app.post("/api/strategies/{strategy_id}/start", response_model=Dict[str, str])
async def start_strategy(
    strategy_id: str,
    action: StrategyAction,
    current_user: User = Depends(require_trader_or_admin)
):
    """Start a strategy."""
    try:
        # Start strategy using service
        result = await strategy_service.start_strategy(
            strategy_id,
            confirm_live_trading=action.confirm_live_trading
        )

        return {
            "message": result["message"],
            "strategy_id": result["strategy_id"]
        }

    except StrategyOperationError as e:
        if "confirmation" in str(e).lower():
            raise HTTPException(status_code=400, detail=str(e))
        elif "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        _logger.exception("Error starting strategy %s:", strategy_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategies/{strategy_id}/stop", response_model=Dict[str, str])
async def stop_strategy(strategy_id: str, current_user: User = Depends(require_trader_or_admin)):
    """Stop a strategy."""
    try:
        # Stop strategy using service
        result = await strategy_service.stop_strategy(strategy_id)

        return {
            "message": result["message"],
            "strategy_id": result["strategy_id"]
        }

    except StrategyOperationError as e:
        raise HTTPException(status_code=404 if "not found" in str(e).lower() else 503, detail=str(e))
    except Exception as e:
        _logger.exception("Error stopping strategy %s:", strategy_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategies/{strategy_id}/restart", response_model=Dict[str, str])
async def restart_strategy(
    strategy_id: str,
    action: StrategyAction,
    current_user: User = Depends(require_trader_or_admin)
):
    """Restart a strategy."""
    try:
        # Restart strategy using service
        result = await strategy_service.restart_strategy(
            strategy_id,
            confirm_live_trading=action.confirm_live_trading
        )

        return {
            "message": result["message"],
            "strategy_id": result["strategy_id"]
        }

    except StrategyOperationError as e:
        if "confirmation" in str(e).lower():
            raise HTTPException(status_code=400, detail=str(e))
        elif "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        _logger.exception("Error restarting strategy %s:", strategy_id)
        raise HTTPException(status_code=500, detail=str(e))

# System Monitoring Endpoints

@app.get("/api/system/status", response_model=SystemStatus)
async def get_system_status(current_user: User = Depends(get_current_user)):
    """Get overall system status."""
    try:
        # Get service status
        service_status_info = strategy_service.get_service_status()

        # Get real system metrics
        metrics = monitoring_service.get_comprehensive_metrics()
        system_metrics = {
            "cpu_percent": metrics['cpu']['usage_percent'],
            "memory_percent": metrics['memory']['usage_percent'],
            "temperature_c": metrics['temperature']['average_celsius'] or 0.0,
            "disk_usage_percent": max([
                disk['usage_percent'] for disk in metrics['disk']['partitions'].values()
            ], default=0.0)
        }

        return SystemStatus(
            service_name="Enhanced Multi-Strategy Trading System",
            version="2.0.0",
            status="running" if service_status_info["available"] else "unavailable",
            uptime_seconds=0.0,  # TODO: Calculate actual uptime
            active_strategies=service_status_info["active_strategies"],
            total_strategies=service_status_info["total_strategies"],
            system_metrics=system_metrics
        )

    except Exception as e:
        _logger.exception("Error getting system status:")
        raise HTTPException(status_code=500, detail=str(e))

# Configuration Management Endpoints

# Strategy parameter update endpoint
@app.put("/api/strategies/{strategy_id}/parameters", response_model=Dict[str, Any])
async def update_strategy_parameters(
    strategy_id: str,
    parameters: Dict[str, Any],
    current_user: User = Depends(require_trader_or_admin)
):
    """Update strategy parameters while running."""
    try:
        # Update parameters using service
        result = await strategy_service.update_strategy_parameters(strategy_id, parameters)

        return result

    except StrategyOperationError as e:
        raise HTTPException(status_code=404 if "not found" in str(e).lower() else 503, detail=str(e))
    except Exception as e:
        _logger.exception("Error updating strategy parameters %s:", strategy_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config/templates")
async def get_strategy_templates(current_user: User = Depends(get_current_user)):
    """Get available strategy templates."""
    try:
        templates = strategy_service.get_strategy_templates()
        return {"templates": templates}

    except Exception as e:
        _logger.exception("Error loading templates:")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config/validate")
async def validate_configuration(
    config: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Validate a strategy configuration."""
    try:
        # Use strategy service for validation
        strategy_service.validate_strategy_config(config)
        return {"valid": True, "errors": []}

    except StrategyValidationError as e:
        return {"valid": False, "errors": [str(e)]}
    except Exception as e:
        _logger.exception("Error validating configuration:")
        raise HTTPException(status_code=500, detail=str(e))

# System monitoring endpoints
@app.get("/api/monitoring/metrics")
async def get_system_metrics(current_user: User = Depends(get_current_user)):
    """Get comprehensive system metrics."""
    try:
        metrics = monitoring_service.get_comprehensive_metrics()
        return metrics
    except Exception as e:
        _logger.exception("Error getting system metrics:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/monitoring/alerts")
async def get_system_alerts(
    unacknowledged_only: bool = False,
    current_user: User = Depends(get_current_user)
):
    """Get system alerts."""
    try:
        alerts = monitoring_service.get_alerts(unacknowledged_only)
        return {"alerts": alerts}
    except Exception as e:
        _logger.exception("Error getting system alerts:")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/monitoring/alerts/{alert_index}/acknowledge")
async def acknowledge_alert(
    alert_index: int,
    current_user: User = Depends(require_trader_or_admin)
):
    """Acknowledge a system alert."""
    try:
        success = monitoring_service.acknowledge_alert(alert_index)
        if success:
            return {"message": "Alert acknowledged successfully"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error acknowledging alert:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/monitoring/history")
async def get_performance_history(
    hours: int = 1,
    current_user: User = Depends(get_current_user)
):
    """Get performance history."""
    try:
        if hours < 1 or hours > 24:
            raise HTTPException(status_code=400, detail="Hours must be between 1 and 24")

        history = monitoring_service.get_performance_history(hours)
        return {"history": history}
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Error getting performance history:")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    # Get port from config, default to 8000 if not set
    port = int(TRADING_API_PORT) if TRADING_API_PORT else 8000

    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )

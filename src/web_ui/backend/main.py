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

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.trading.enhanced_strategy_manager import EnhancedStrategyManager
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Global strategy manager instance
strategy_manager: Optional[EnhancedStrategyManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global strategy_manager

    # Startup
    _logger.info("Starting Trading Web UI Backend...")
    strategy_manager = EnhancedStrategyManager()

    # Load existing strategies if config exists
    config_file = "config/enhanced_trading/raspberry_pi_multi_strategy.json"
    if Path(config_file).exists():
        await strategy_manager.load_strategies_from_config(config_file)
        _logger.info("Loaded existing strategy configurations")

    yield

    # Shutdown
    _logger.info("Shutting down Trading Web UI Backend...")
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
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Pydantic models for API
from pydantic import BaseModel, Field
from typing import Union

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

# Authentication dependency (simplified for now)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user (placeholder implementation)."""
    # TODO: Implement proper JWT authentication
    # For now, accept any token for development
    if not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"username": "admin", "role": "admin"}  # Placeholder user

# API Routes

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Trading Web UI API", "version": "1.0.0"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": asyncio.get_event_loop().time()}

# Strategy Management Endpoints

@app.get("/api/strategies", response_model=List[StrategyStatus])
async def list_strategies(user: dict = Depends(get_current_user)):
    """List all configured strategies."""
    if not strategy_manager:
        raise HTTPException(status_code=503, detail="Strategy manager not available")

    try:
        strategies = strategy_manager.get_all_status()
        return [StrategyStatus(**strategy) for strategy in strategies]
    except Exception as e:
        _logger.error(f"Error listing strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategies", response_model=Dict[str, str])
async def create_strategy(
    strategy_config: StrategyConfig,
    user: dict = Depends(get_current_user)
):
    """Create a new strategy."""
    if not strategy_manager:
        raise HTTPException(status_code=503, detail="Strategy manager not available")

    try:
        # Convert Pydantic model to dict
        config_dict = strategy_config.dict()

        # Create strategy instance
        from src.trading.enhanced_strategy_manager import StrategyInstance
        instance = StrategyInstance(strategy_config.id, config_dict)
        strategy_manager.strategy_instances[strategy_config.id] = instance

        _logger.info(f"Created strategy: {strategy_config.name}")
        return {"message": "Strategy created successfully", "strategy_id": strategy_config.id}

    except Exception as e:
        _logger.error(f"Error creating strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/strategies/{strategy_id}", response_model=StrategyStatus)
async def get_strategy(strategy_id: str, user: dict = Depends(get_current_user)):
    """Get details of a specific strategy."""
    if not strategy_manager:
        raise HTTPException(status_code=503, detail="Strategy manager not available")

    try:
        status = strategy_manager.get_strategy_status(strategy_id)
        if not status:
            raise HTTPException(status_code=404, detail="Strategy not found")

        return StrategyStatus(**status)

    except HTTPException:
        raise
    except Exception as e:
        _logger.error(f"Error getting strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/strategies/{strategy_id}", response_model=Dict[str, str])
async def update_strategy(
    strategy_id: str,
    strategy_config: StrategyConfig,
    user: dict = Depends(get_current_user)
):
    """Update an existing strategy."""
    if not strategy_manager:
        raise HTTPException(status_code=503, detail="Strategy manager not available")

    try:
        if strategy_id not in strategy_manager.strategy_instances:
            raise HTTPException(status_code=404, detail="Strategy not found")

        # Update strategy configuration
        config_dict = strategy_config.dict()
        strategy_manager.strategy_instances[strategy_id].config = config_dict

        _logger.info(f"Updated strategy: {strategy_config.name}")
        return {"message": "Strategy updated successfully", "strategy_id": strategy_id}

    except HTTPException:
        raise
    except Exception as e:
        _logger.error(f"Error updating strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/strategies/{strategy_id}", response_model=Dict[str, str])
async def delete_strategy(strategy_id: str, user: dict = Depends(get_current_user)):
    """Delete a strategy."""
    if not strategy_manager:
        raise HTTPException(status_code=503, detail="Strategy manager not available")

    try:
        if strategy_id not in strategy_manager.strategy_instances:
            raise HTTPException(status_code=404, detail="Strategy not found")

        # Stop strategy if running
        instance = strategy_manager.strategy_instances[strategy_id]
        if instance.status == 'running':
            await strategy_manager.stop_strategy(strategy_id)

        # Remove from manager
        del strategy_manager.strategy_instances[strategy_id]

        _logger.info(f"Deleted strategy: {strategy_id}")
        return {"message": "Strategy deleted successfully", "strategy_id": strategy_id}

    except HTTPException:
        raise
    except Exception as e:
        _logger.error(f"Error deleting strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Strategy Lifecycle Endpoints

@app.post("/api/strategies/{strategy_id}/start", response_model=Dict[str, str])
async def start_strategy(
    strategy_id: str,
    action: StrategyAction,
    user: dict = Depends(get_current_user)
):
    """Start a strategy."""
    if not strategy_manager:
        raise HTTPException(status_code=503, detail="Strategy manager not available")

    try:
        if strategy_id not in strategy_manager.strategy_instances:
            raise HTTPException(status_code=404, detail="Strategy not found")

        instance = strategy_manager.strategy_instances[strategy_id]

        # Check for live trading confirmation
        if (instance.config.get('broker', {}).get('trading_mode') == 'live' and
            not action.confirm_live_trading):
            raise HTTPException(
                status_code=400,
                detail="Live trading requires explicit confirmation"
            )

        success = await strategy_manager.start_strategy(strategy_id)

        if success:
            return {"message": "Strategy started successfully", "strategy_id": strategy_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to start strategy")

    except HTTPException:
        raise
    except Exception as e:
        _logger.error(f"Error starting strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategies/{strategy_id}/stop", response_model=Dict[str, str])
async def stop_strategy(strategy_id: str, user: dict = Depends(get_current_user)):
    """Stop a strategy."""
    if not strategy_manager:
        raise HTTPException(status_code=503, detail="Strategy manager not available")

    try:
        success = await strategy_manager.stop_strategy(strategy_id)

        if success:
            return {"message": "Strategy stopped successfully", "strategy_id": strategy_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to stop strategy")

    except Exception as e:
        _logger.error(f"Error stopping strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategies/{strategy_id}/restart", response_model=Dict[str, str])
async def restart_strategy(
    strategy_id: str,
    action: StrategyAction,
    user: dict = Depends(get_current_user)
):
    """Restart a strategy."""
    if not strategy_manager:
        raise HTTPException(status_code=503, detail="Strategy manager not available")

    try:
        if strategy_id not in strategy_manager.strategy_instances:
            raise HTTPException(status_code=404, detail="Strategy not found")

        instance = strategy_manager.strategy_instances[strategy_id]

        # Check for live trading confirmation
        if (instance.config.get('broker', {}).get('trading_mode') == 'live' and
            not action.confirm_live_trading):
            raise HTTPException(
                status_code=400,
                detail="Live trading requires explicit confirmation"
            )

        success = await strategy_manager.restart_strategy(strategy_id)

        if success:
            return {"message": "Strategy restarted successfully", "strategy_id": strategy_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to restart strategy")

    except HTTPException:
        raise
    except Exception as e:
        _logger.error(f"Error restarting strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System Monitoring Endpoints

@app.get("/api/system/status", response_model=SystemStatus)
async def get_system_status(user: dict = Depends(get_current_user)):
    """Get overall system status."""
    if not strategy_manager:
        raise HTTPException(status_code=503, detail="Strategy manager not available")

    try:
        strategies = strategy_manager.get_all_status()
        active_strategies = sum(1 for s in strategies if s['status'] == 'running')

        # Get system metrics (placeholder)
        system_metrics = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "temperature_c": 0.0,
            "disk_usage_percent": 0.0
        }

        return SystemStatus(
            service_name="Enhanced Multi-Strategy Trading System",
            version="2.0.0",
            status="running" if strategy_manager.is_running else "stopped",
            uptime_seconds=0.0,  # TODO: Calculate actual uptime
            active_strategies=active_strategies,
            total_strategies=len(strategies),
            system_metrics=system_metrics
        )

    except Exception as e:
        _logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configuration Management Endpoints

@app.get("/api/config/templates")
async def get_strategy_templates(user: dict = Depends(get_current_user)):
    """Get available strategy templates."""
    try:
        # Load templates from config examples
        templates_file = PROJECT_ROOT / "config/examples/enhanced_trading_config_examples.json"

        if templates_file.exists():
            with open(templates_file, 'r') as f:
                templates = json.load(f)
            return {"templates": templates}
        else:
            return {"templates": {}}

    except Exception as e:
        _logger.error(f"Error loading templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config/validate")
async def validate_configuration(
    config: Dict[str, Any],
    user: dict = Depends(get_current_user)
):
    """Validate a strategy configuration."""
    try:
        # Basic validation
        required_fields = ['id', 'name', 'symbol', 'broker', 'strategy']
        missing_fields = [field for field in required_fields if field not in config]

        if missing_fields:
            return {
                "valid": False,
                "errors": [f"Missing required field: {field}" for field in missing_fields]
            }

        # Additional validation can be added here
        return {"valid": True, "errors": []}

    except Exception as e:
        _logger.error(f"Error validating configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
"""
Report Worker

Dramatiq worker for executing report generation jobs.
"""

import os
import json
from datetime import datetime
from uuid import UUID
from typing import Dict, Any, Optional
from pathlib import Path
import dramatiq
from sqlalchemy.orm import sessionmaker

from .dramatiq_config import broker
from src.data.db.services.database_service import get_database_service
from src.data.db.services.jobs_service import JobsService
from src.data.db.models.model_jobs import RunStatus
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dramatiq.actor(broker=broker, queue_name="reports", max_retries=3, time_limit=3600000)  # 1 hour time limit
def run_report(run_id: str) -> Dict[str, Any]:
    """
    Execute a report generation job.

    Args:
        run_id: UUID string of the run to execute

    Returns:
        Dictionary with execution results
    """
    run_uuid = UUID(run_id)
    worker_id = f"report_worker_{os.getpid()}_{datetime.utcnow().timestamp()}"

    _logger.info(f"Starting report execution for run: {run_uuid}")

    try:
        # Get database service and create session
        db_service = get_database_service()

        with db_service.uow() as uow:
            jobs_service = JobsService(uow.session)

            # Claim the run atomically
            run = jobs_service.claim_run(run_uuid, worker_id)
            if not run:
                _logger.warning(f"Run {run_uuid} could not be claimed - may already be running or completed")
                return {"status": "skipped", "reason": "run_already_claimed"}

            _logger.info(f"Claimed run {run_uuid} for execution")

            # Get job snapshot
            job_snapshot = run.job_snapshot
            report_type = job_snapshot.get("report_type", "unknown")
            parameters = job_snapshot.get("parameters", {})

            _logger.info(f"Executing report: {report_type} with parameters: {parameters}")

            # Execute the report
            result = _execute_report(report_type, parameters, run_uuid)

            # Update run with results
            update_data = {
                "status": RunStatus.COMPLETED.value,
                "finished_at": datetime.utcnow(),
                "result": result
            }

            jobs_service.update_run(run_uuid, update_data)
            uow.commit()

            _logger.info(f"Successfully completed report run: {run_uuid}")
            return {"status": "completed", "result": result}

    except Exception as e:
        _logger.error(f"Failed to execute report run {run_uuid}: {e}")

        # Update run with error
        try:
            with db_service.uow() as uow:
                jobs_service = JobsService(uow.session)
                update_data = {
                    "status": RunStatus.FAILED.value,
                    "finished_at": datetime.utcnow(),
                    "error": str(e)
                }
                jobs_service.update_run(run_uuid, update_data)
                uow.commit()
        except Exception as update_error:
            _logger.error(f"Failed to update run status after error: {update_error}")

        # Re-raise the exception to trigger retry mechanism
        raise


def _execute_report(report_type: str, parameters: Dict[str, Any], run_id: UUID) -> Dict[str, Any]:
    """
    Execute a specific report type.

    Args:
        report_type: Type of report to generate
        parameters: Report parameters
        run_id: Run ID for artifact storage

    Returns:
        Dictionary with report results
    """
    logger.info(f"Executing report type: {report_type}")

    # Create artifacts directory for this run
    artifacts_dir = Path("artifacts") / str(run_id)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        if report_type == "system_status":
            return _generate_system_status_report(parameters, artifacts_dir)
        elif report_type == "strategy_performance":
            return _generate_strategy_performance_report(parameters, artifacts_dir)
        elif report_type == "portfolio_summary":
            return _generate_portfolio_summary_report(parameters, artifacts_dir)
        elif report_type == "risk_analysis":
            return _generate_risk_analysis_report(parameters, artifacts_dir)
        elif report_type == "market_overview":
            return _generate_market_overview_report(parameters, artifacts_dir)
        else:
            raise ValueError(f"Unknown report type: {report_type}")

    except Exception as e:
        logger.error(f"Failed to execute report {report_type}: {e}")
        raise


def _generate_system_status_report(parameters: Dict[str, Any], artifacts_dir: Path) -> Dict[str, Any]:
    """Generate system status report."""
    _logger.info("Generating system status report")

    # Mock implementation - replace with actual system status logic
    report_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "system_status": "operational",
        "active_strategies": 3,
        "total_trades": 150,
        "system_uptime": "7 days, 12 hours",
        "memory_usage": "45%",
        "cpu_usage": "23%",
        "disk_usage": "67%"
    }

    # Save report to file
    report_file = artifacts_dir / "system_status.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)

    return {
        "report_type": "system_status",
        "status": "completed",
        "artifacts": [str(report_file)],
        "data": report_data
    }


def _generate_strategy_performance_report(parameters: Dict[str, Any], artifacts_dir: Path) -> Dict[str, Any]:
    """Generate strategy performance report."""
    _logger.info("Generating strategy performance report")

    # Mock implementation - replace with actual strategy performance logic
    report_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "strategies": [
            {
                "name": "BTCUSDT_Strategy_1",
                "total_return": 12.5,
                "sharpe_ratio": 1.8,
                "max_drawdown": -5.2,
                "win_rate": 0.65,
                "total_trades": 45
            },
            {
                "name": "ETHUSDT_Strategy_1",
                "total_return": 8.3,
                "sharpe_ratio": 1.4,
                "max_drawdown": -7.1,
                "win_rate": 0.58,
                "total_trades": 32
            }
        ]
    }

    # Save report to file
    report_file = artifacts_dir / "strategy_performance.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)

    return {
        "report_type": "strategy_performance",
        "status": "completed",
        "artifacts": [str(report_file)],
        "data": report_data
    }


def _generate_portfolio_summary_report(parameters: Dict[str, Any], artifacts_dir: Path) -> Dict[str, Any]:
    """Generate portfolio summary report."""
    _logger.info("Generating portfolio summary report")

    # Mock implementation - replace with actual portfolio logic
    report_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_value": 125000.50,
        "total_return": 8.5,
        "daily_pnl": 1250.75,
        "positions": [
            {"symbol": "BTCUSDT", "quantity": 0.5, "value": 45000.25},
            {"symbol": "ETHUSDT", "quantity": 2.0, "value": 80000.00}
        ]
    }

    # Save report to file
    report_file = artifacts_dir / "portfolio_summary.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)

    return {
        "report_type": "portfolio_summary",
        "status": "completed",
        "artifacts": [str(report_file)],
        "data": report_data
    }


def _generate_risk_analysis_report(parameters: Dict[str, Any], artifacts_dir: Path) -> Dict[str, Any]:
    """Generate risk analysis report."""
    _logger.info("Generating risk analysis report")

    # Mock implementation - replace with actual risk analysis logic
    report_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "var_95": -2500.00,
        "var_99": -4500.00,
        "max_drawdown": -8500.00,
        "correlation_matrix": {
            "BTCUSDT": {"ETHUSDT": 0.75, "LTCUSDT": 0.65},
            "ETHUSDT": {"BTCUSDT": 0.75, "LTCUSDT": 0.70},
            "LTCUSDT": {"BTCUSDT": 0.65, "ETHUSDT": 0.70}
        }
    }

    # Save report to file
    report_file = artifacts_dir / "risk_analysis.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)

    return {
        "report_type": "risk_analysis",
        "status": "completed",
        "artifacts": [str(report_file)],
        "data": report_data
    }


def _generate_market_overview_report(parameters: Dict[str, Any], artifacts_dir: Path) -> Dict[str, Any]:
    """Generate market overview report."""
    _logger.info("Generating market overview report")

    # Mock implementation - replace with actual market data logic
    report_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "market_summary": {
            "total_market_cap": 2500000000000,
            "market_change_24h": 2.5,
            "fear_greed_index": 65
        },
        "top_movers": [
            {"symbol": "BTCUSDT", "change_24h": 3.2, "volume": 1500000000},
            {"symbol": "ETHUSDT", "change_24h": 2.8, "volume": 1200000000}
        ]
    }

    # Save report to file
    report_file = artifacts_dir / "market_overview.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)

    return {
        "report_type": "market_overview",
        "status": "completed",
        "artifacts": [str(report_file)],
        "data": report_data
    }


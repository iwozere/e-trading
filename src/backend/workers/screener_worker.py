"""
Screener Worker

Dramatiq worker for executing screener jobs.
"""

import os
import json
import pandas as pd
from datetime import datetime
from uuid import UUID
from typing import Dict, Any, List, Optional
from pathlib import Path
import dramatiq
from sqlalchemy.orm import sessionmaker

from .dramatiq_config import broker
from src.data.db.services.database_service import get_database_service
from src.data.db.services.jobs_service import JobsService
from src.data.db.models.model_jobs import RunStatus
from src.backend.config_loader import get_screener_config
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dramatiq.actor(broker=broker, queue_name="screeners", max_retries=3, time_limit=1800000)  # 30 minutes time limit
def run_screener(run_id: str) -> Dict[str, Any]:
    """
    Execute a screener job.

    Args:
        run_id: UUID string of the run to execute

    Returns:
        Dictionary with execution results
    """
    run_uuid = UUID(run_id)
    worker_id = f"screener_worker_{os.getpid()}_{datetime.utcnow().timestamp()}"

    _logger.info(f"Starting screener execution for run: {run_uuid}")

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
            tickers = job_snapshot.get("tickers", [])
            filter_criteria = job_snapshot.get("filter_criteria", {})
            top_n = job_snapshot.get("top_n", 50)
            screener_set = job_snapshot.get("screener_set")

            _logger.info(f"Executing screener for {len(tickers)} tickers with criteria: {filter_criteria}")

            # Execute the screener
            result = _execute_screener(tickers, filter_criteria, top_n, run_uuid)

            # Update run with results
            update_data = {
                "status": RunStatus.COMPLETED.value,
                "finished_at": datetime.utcnow(),
                "result": result
            }

            jobs_service.update_run(run_uuid, update_data)
            uow.commit()

            _logger.info(f"Successfully completed screener run: {run_uuid}")
            return {"status": "completed", "result": result}

    except Exception as e:
        _logger.error(f"Failed to execute screener run {run_uuid}: {e}")

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


def _execute_screener(
    tickers: List[str],
    filter_criteria: Dict[str, Any],
    top_n: int,
    run_id: UUID
) -> Dict[str, Any]:
    """
    Execute screener logic on the provided tickers.

    Args:
        tickers: List of ticker symbols to screen
        filter_criteria: Filter criteria to apply
        top_n: Number of top results to return
        run_id: Run ID for artifact storage

    Returns:
        Dictionary with screener results
    """
    _logger.info(f"Executing screener for {len(tickers)} tickers")

    # Create artifacts directory for this run
    artifacts_dir = Path("artifacts") / str(run_id)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Get market data for tickers
        market_data = _get_market_data(tickers)

        if market_data.empty:
            _logger.warning("No market data available for screening")
            return {
                "status": "completed",
                "results": [],
                "total_screened": 0,
                "artifacts": []
            }

        # Apply filter criteria
        filtered_data = _apply_filters(market_data, filter_criteria)

        # Sort and get top N results
        top_results = _get_top_results(filtered_data, top_n)

        # Save results to files
        artifacts = _save_screener_results(top_results, artifacts_dir)

        # Prepare result summary
        result = {
            "status": "completed",
            "total_screened": len(market_data),
            "total_passed": len(filtered_data),
            "top_results_count": len(top_results),
            "filter_criteria": filter_criteria,
            "artifacts": artifacts,
            "results": top_results.to_dict('records') if not top_results.empty else []
        }

        _logger.info(f"Screener completed: {len(top_results)} results from {len(market_data)} tickers")
        return result

    except Exception as e:
        _logger.error(f"Failed to execute screener: {e}")
        raise


def _get_market_data(tickers: List[str]) -> pd.DataFrame:
    """
    Get market data for the specified tickers.

    Args:
        tickers: List of ticker symbols

    Returns:
        DataFrame with market data
    """
    _logger.info(f"Fetching market data for {len(tickers)} tickers")

    # Mock implementation - replace with actual data source
    # This would typically fetch from your data source (CSV files, API, etc.)

    mock_data = []
    for ticker in tickers:
        # Generate mock market data
        import random

        mock_data.append({
            "symbol": ticker,
            "price": round(random.uniform(10, 1000), 2),
            "market_cap": round(random.uniform(1000000000, 1000000000000), 0),
            "pe_ratio": round(random.uniform(5, 50), 2),
            "pb_ratio": round(random.uniform(0.5, 5.0), 2),
            "debt_to_equity": round(random.uniform(0.1, 2.0), 2),
            "return_on_equity": round(random.uniform(0.05, 0.25), 3),
            "revenue_growth": round(random.uniform(-0.1, 0.3), 3),
            "profit_margin": round(random.uniform(0.02, 0.20), 3),
            "dividend_yield": round(random.uniform(0.0, 0.05), 3),
            "beta": round(random.uniform(0.5, 2.0), 2),
            "volume": round(random.uniform(1000000, 100000000), 0),
            "change_1d": round(random.uniform(-0.1, 0.1), 3),
            "change_1w": round(random.uniform(-0.2, 0.2), 3),
            "change_1m": round(random.uniform(-0.3, 0.3), 3)
        })

    return pd.DataFrame(mock_data)


def _apply_filters(data: pd.DataFrame, filter_criteria: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply filter criteria to the market data.

    Args:
        data: DataFrame with market data
        filter_criteria: Dictionary with filter criteria

    Returns:
        Filtered DataFrame
    """
    _logger.info(f"Applying filters: {filter_criteria}")

    filtered_data = data.copy()

    # Apply common filters
    if "market_cap_min" in filter_criteria:
        filtered_data = filtered_data[filtered_data["market_cap"] >= filter_criteria["market_cap_min"]]

    if "market_cap_max" in filter_criteria:
        filtered_data = filtered_data[filtered_data["market_cap"] <= filter_criteria["market_cap_max"]]

    if "pe_ratio_max" in filter_criteria:
        filtered_data = filtered_data[filtered_data["pe_ratio"] <= filter_criteria["pe_ratio_max"]]

    if "pe_ratio_min" in filter_criteria:
        filtered_data = filtered_data[filtered_data["pe_ratio"] >= filter_criteria["pe_ratio_min"]]

    if "pb_ratio_max" in filter_criteria:
        filtered_data = filtered_data[filtered_data["pb_ratio"] <= filter_criteria["pb_ratio_max"]]

    if "debt_to_equity_max" in filter_criteria:
        filtered_data = filtered_data[filtered_data["debt_to_equity"] <= filter_criteria["debt_to_equity_max"]]

    if "return_on_equity_min" in filter_criteria:
        filtered_data = filtered_data[filtered_data["return_on_equity"] >= filter_criteria["return_on_equity_min"]]

    if "revenue_growth_min" in filter_criteria:
        filtered_data = filtered_data[filtered_data["revenue_growth"] >= filter_criteria["revenue_growth_min"]]

    if "profit_margin_min" in filter_criteria:
        filtered_data = filtered_data[filtered_data["profit_margin"] >= filter_criteria["profit_margin_min"]]

    if "dividend_yield_min" in filter_criteria:
        filtered_data = filtered_data[filtered_data["dividend_yield"] >= filter_criteria["dividend_yield_min"]]

    if "beta_max" in filter_criteria:
        filtered_data = filtered_data[filtered_data["beta"] <= filter_criteria["beta_max"]]

    if "volume_min" in filter_criteria:
        filtered_data = filtered_data[filtered_data["volume"] >= filter_criteria["volume_min"]]

    _logger.info(f"Filtered from {len(data)} to {len(filtered_data)} tickers")
    return filtered_data


def _get_top_results(data: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Get top N results based on a scoring system.

    Args:
        data: Filtered DataFrame
        top_n: Number of top results to return

    Returns:
        DataFrame with top results
    """
    if data.empty:
        return data

    # Create a simple scoring system
    # This is a mock implementation - replace with your actual scoring logic

    # Normalize metrics (higher is better for most metrics)
    data["score"] = 0.0

    # Market cap score (log scale)
    data["score"] += (data["market_cap"].apply(lambda x: 0 if x <= 0 else min(10, (x / 1000000000) ** 0.3)))

    # ROE score
    data["score"] += data["return_on_equity"] * 20

    # Revenue growth score
    data["score"] += data["revenue_growth"] * 15

    # Profit margin score
    data["score"] += data["profit_margin"] * 25

    # Dividend yield score
    data["score"] += data["dividend_yield"] * 10

    # Penalize high PE ratio
    data["score"] -= (data["pe_ratio"] - 15).clip(lower=0) * 0.1

    # Penalize high debt-to-equity
    data["score"] -= (data["debt_to_equity"] - 0.5).clip(lower=0) * 2

    # Sort by score and return top N
    top_results = data.nlargest(top_n, "score")

    _logger.info(f"Selected top {len(top_results)} results")
    return top_results


def _save_screener_results(results: pd.DataFrame, artifacts_dir: Path) -> List[str]:
    """
    Save screener results to files.

    Args:
        results: DataFrame with results
        artifacts_dir: Directory to save artifacts

    Returns:
        List of artifact file paths
    """
    artifacts = []

    if results.empty:
        return artifacts

    # Save as CSV
    csv_file = artifacts_dir / "screener_results.csv"
    results.to_csv(csv_file, index=False)
    artifacts.append(str(csv_file))

    # Save as JSON
    json_file = artifacts_dir / "screener_results.json"
    results.to_json(json_file, orient="records", indent=2)
    artifacts.append(str(json_file))

    # Save summary
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_results": len(results),
        "top_symbols": results["symbol"].tolist()[:10],  # Top 10 symbols
        "average_score": results["score"].mean(),
        "score_range": {
            "min": results["score"].min(),
            "max": results["score"].max()
        }
    }

    summary_file = artifacts_dir / "screener_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    artifacts.append(str(summary_file))

    _logger.info(f"Saved screener results to {len(artifacts)} files")
    return artifacts


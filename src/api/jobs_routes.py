"""
Jobs API Routes

FastAPI routes for job scheduling and execution operations.
Provides endpoints for managing schedules and runs.
"""

from typing import List, Optional
# UUID import removed - using integer IDs
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from src.api.auth import get_current_user, require_trader_or_admin
from src.api.services.webui_app_service import webui_app_service
from src.data.db.services.jobs_service import JobsService
from src.data.db.models.model_jobs import (
    JobType, RunStatus,
    ScheduleCreate, ScheduleUpdate, ScheduleResponse,
    ScheduleRunResponse, ReportRequest, ScreenerRequest, ScreenerSetInfo
)
from src.notification.logger import setup_logger

logger = setup_logger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["jobs"])


def get_jobs_service(session: Session = Depends(webui_app_service.get_db_session)) -> JobsService:
    """Get jobs service with database session."""
    return JobsService(session)


# ---------- Ad-hoc Execution Endpoints ----------

@router.post("/reports/run", response_model=ScheduleRunResponse, status_code=status.HTTP_201_CREATED)
async def run_report(
    request: ReportRequest,
    current_user = Depends(get_current_user),
    jobs_service: JobsService = Depends(get_jobs_service)
):
    """
    Run a report immediately.

    Creates a new run for report execution with the specified parameters.
    """
    try:
        run = jobs_service.create_report_run(
            user_id=current_user.id,
            report_type=request.report_type,
            parameters=request.parameters,
            scheduled_for=request.scheduled_for
        )

        _logger.info("Created report run: %s for user %s", run.id, current_user.id)
        return ScheduleRunResponse.from_orm(run)

    except Exception as e:
        _logger.exception("Failed to create report run:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create report run: {str(e)}"
        )


@router.post("/screeners/run", response_model=ScheduleRunResponse, status_code=status.HTTP_201_CREATED)
async def run_screener(
    request: ScreenerRequest,
    current_user = Depends(get_current_user),
    jobs_service: JobsService = Depends(get_jobs_service)
):
    """
    Run a screener immediately.

    Creates a new run for screener execution with the specified parameters.
    """
    try:
        run = jobs_service.create_screener_run(
            user_id=current_user.id,
            screener_set=request.screener_set,
            tickers=request.tickers,
            filter_criteria=request.filter_criteria,
            top_n=request.top_n,
            scheduled_for=request.scheduled_for
        )

        _logger.info("Created screener run: %s for user %s", run.id, current_user.id)
        return ScheduleRunResponse.from_orm(run)

    except ValueError as e:
        _logger.exception("Invalid screener request:")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        _logger.exception("Failed to create screener run:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create screener run: {str(e)}"
        )


@router.get("/runs/{run_id}", response_model=ScheduleRunResponse)
async def get_run(
    run_id: int,
    current_user = Depends(get_current_user),
    jobs_service: JobsService = Depends(get_jobs_service)
):
    """
    Get run status and details.

    Returns the current status and details of a specific run.
    """
    run = jobs_service.get_run(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found"
        )

    # Check if user has access to this run
    if run.user_id != current_user.id and current_user.role not in ["admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    return ScheduleRunResponse.from_orm(run)


@router.get("/runs", response_model=List[ScheduleRunResponse])
async def list_runs(
    job_type: Optional[JobType] = Query(None, description="Filter by job type"),
    status: Optional[RunStatus] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    order_by: str = Query("scheduled_for", description="Field to order by"),
    order_desc: bool = Query(True, description="Order in descending order"),
    current_user = Depends(get_current_user),
    jobs_service: JobsService = Depends(get_jobs_service)
):
    """
    List runs with optional filtering.

    Returns a list of runs with optional filtering by job type, status, etc.
    """
    # For non-admin users, only show their own runs
    user_id = current_user.id if current_user.role != "admin" else None

    runs = jobs_service.list_runs(
        user_id=user_id,
        job_type=job_type,
        status=status,
        limit=limit,
        offset=offset,
        order_by=order_by,
        order_desc=order_desc
    )

    return [ScheduleRunResponse.from_orm(run) for run in runs]


@router.delete("/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_run(
    run_id: int,
    current_user = Depends(get_current_user),
    jobs_service: JobsService = Depends(get_jobs_service)
):
    """
    Cancel a pending run.

    Cancels a run that is still in pending status.
    """
    run = jobs_service.get_run(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found"
        )

    # Check if user has access to this run
    if run.user_id != current_user.id and current_user.role not in ["admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    success = jobs_service.cancel_run(run_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot cancel run - it may not exist or is already running/completed"
        )


# ---------- Scheduling Endpoints ----------

@router.post("/schedules", response_model=ScheduleResponse, status_code=status.HTTP_201_CREATED)
async def create_schedule(
    schedule_data: ScheduleCreate,
    current_user = Depends(require_trader_or_admin),
    jobs_service: JobsService = Depends(get_jobs_service)
):
    """
    Create a new schedule.

    Creates a new schedule for recurring job execution.
    """
    try:
        schedule = jobs_service.create_schedule(current_user.id, schedule_data)
        _logger.info("Created schedule: %s for user %s", schedule.name, current_user.id)
        return ScheduleResponse.from_orm(schedule)

    except ValueError as e:
        _logger.exception("Invalid schedule data:")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        _logger.exception("Failed to create schedule:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create schedule: {str(e)}"
        )


@router.get("/schedules", response_model=List[ScheduleResponse])
async def list_schedules(
    job_type: Optional[JobType] = Query(None, description="Filter by job type"),
    enabled: Optional[bool] = Query(None, description="Filter by enabled status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    current_user = Depends(get_current_user),
    jobs_service: JobsService = Depends(get_jobs_service)
):
    """
    List schedules with optional filtering.

    Returns a list of schedules with optional filtering.
    """
    # For non-admin users, only show their own schedules
    user_id = current_user.id if current_user.role != "admin" else None

    schedules = jobs_service.list_schedules(
        user_id=user_id,
        job_type=job_type,
        enabled=enabled,
        limit=limit,
        offset=offset
    )

    return [ScheduleResponse.from_orm(schedule) for schedule in schedules]


@router.get("/schedules/{schedule_id}", response_model=ScheduleResponse)
async def get_schedule(
    schedule_id: int,
    current_user = Depends(get_current_user),
    jobs_service: JobsService = Depends(get_jobs_service)
):
    """
    Get schedule details.

    Returns the details of a specific schedule.
    """
    schedule = jobs_service.get_schedule(schedule_id)
    if not schedule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schedule not found"
        )

    # Check if user has access to this schedule
    if schedule.user_id != current_user.id and current_user.role not in ["admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    return ScheduleResponse.from_orm(schedule)


@router.put("/schedules/{schedule_id}", response_model=ScheduleResponse)
async def update_schedule(
    schedule_id: int,
    update_data: ScheduleUpdate,
    current_user = Depends(require_trader_or_admin),
    jobs_service: JobsService = Depends(get_jobs_service)
):
    """
    Update a schedule.

    Updates an existing schedule with new parameters.
    """
    schedule = jobs_service.get_schedule(schedule_id)
    if not schedule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schedule not found"
        )

    # Check if user has access to this schedule
    if schedule.user_id != current_user.id and current_user.role not in ["admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    try:
        updated_schedule = jobs_service.update_schedule(schedule_id, update_data)
        if not updated_schedule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Schedule not found"
            )

        _logger.info("Updated schedule: %s (ID: %s)", updated_schedule.name, schedule_id)
        return ScheduleResponse.from_orm(updated_schedule)

    except ValueError as e:
        _logger.exception("Invalid schedule update data:")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        _logger.exception("Failed to update schedule %s:", schedule_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update schedule: {str(e)}"
        )


@router.delete("/schedules/{schedule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_schedule(
    schedule_id: int,
    current_user = Depends(require_trader_or_admin),
    jobs_service: JobsService = Depends(get_jobs_service)
):
    """
    Delete a schedule.

    Deletes an existing schedule.
    """
    schedule = jobs_service.get_schedule(schedule_id)
    if not schedule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schedule not found"
        )

    # Check if user has access to this schedule
    if schedule.user_id != current_user.id and current_user.role not in ["admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    success = jobs_service.delete_schedule(schedule_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schedule not found"
        )


@router.post("/schedules/{schedule_id}/trigger", response_model=ScheduleRunResponse, status_code=status.HTTP_201_CREATED)
async def trigger_schedule(
    schedule_id: int,
    current_user = Depends(require_trader_or_admin),
    jobs_service: JobsService = Depends(get_jobs_service)
):
    """
    Manually trigger a schedule.

    Creates a new run for immediate execution of a scheduled job.
    """
    schedule = jobs_service.get_schedule(schedule_id)
    if not schedule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schedule not found"
        )

    # Check if user has access to this schedule
    if schedule.user_id != current_user.id and current_user.role not in ["admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    try:
        run = jobs_service.trigger_schedule(schedule_id)
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Schedule not found"
            )

        _logger.info("Triggered schedule: %s (ID: %s)", schedule.name, schedule_id)
        return ScheduleRunResponse.from_orm(run)

    except ValueError as e:
        _logger.exception("Cannot trigger schedule:")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        _logger.exception("Failed to trigger schedule %s:", schedule_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger schedule: {str(e)}"
        )


# ---------- Screener Sets Endpoints ----------

@router.get("/screener-sets", response_model=List[ScreenerSetInfo])
async def list_screener_sets(
    category: Optional[str] = Query(None, description="Filter by category"),
    search: Optional[str] = Query(None, description="Search in names and descriptions"),
    current_user = Depends(get_current_user)
):
    """
    List available screener sets.

    Returns a list of available screener sets with optional filtering.
    """
    try:
        screener_config = get_screener_config()

        if search:
            set_names = screener_config.search_sets(search)
        elif category:
            set_names = screener_config.list_sets_by_category(category)
        else:
            set_names = screener_config.list_available_sets()

        screener_sets = []
        for set_name in set_names:
            set_info = screener_config.get_set_info(set_name)
            screener_sets.append(ScreenerSetInfo(
                name=set_name,
                description=set_info.get("description", ""),
                ticker_count=set_info.get("ticker_count", 0),
                tickers=set_info.get("tickers", []),
                categories=set_info.get("categories", [])
            ))

        return screener_sets

    except Exception as e:
        _logger.exception("Failed to list screener sets:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list screener sets: {str(e)}"
        )


@router.get("/screener-sets/{set_name}", response_model=ScreenerSetInfo)
async def get_screener_set(
    set_name: str,
    current_user = Depends(get_current_user)
):
    """
    Get screener set details.

    Returns the details of a specific screener set.
    """
    try:
        screener_config = get_screener_config()
        set_info = screener_config.get_set_info(set_name)

        return ScreenerSetInfo(
            name=set_name,
            description=set_info.get("description", ""),
            ticker_count=set_info.get("ticker_count", 0),
            tickers=set_info.get("tickers", []),
            categories=set_info.get("categories", [])
        )

    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Screener set not found: {set_name}"
        )
    except Exception as e:
        _logger.exception("Failed to get screener set %s:", set_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get screener set: {str(e)}"
        )


# ---------- Statistics Endpoints ----------

@router.get("/runs/statistics")
async def get_run_statistics(
    job_type: Optional[JobType] = Query(None, description="Filter by job type"),
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    current_user = Depends(get_current_user),
    jobs_service: JobsService = Depends(get_jobs_service)
):
    """
    Get run statistics.

    Returns statistics about runs for the specified time period.
    """
    # For non-admin users, only show their own statistics
    user_id = current_user.id if current_user.role != "admin" else None

    try:
        stats = jobs_service.get_run_statistics(
            user_id=user_id,
            job_type=job_type,
            days=days
        )
        return stats

    except Exception as e:
        _logger.exception("Failed to get run statistics:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get run statistics: {str(e)}"
        )


# ---------- Admin Endpoints ----------

@router.post("/admin/cleanup-runs", status_code=status.HTTP_200_OK)
async def cleanup_old_runs(
    days_to_keep: int = Query(90, ge=1, le=365, description="Number of days of runs to keep"),
    current_user = Depends(require_trader_or_admin),
    jobs_service: JobsService = Depends(get_jobs_service)
):
    """
    Clean up old runs (Admin only).

    Deletes old completed and failed runs to free up database space.
    """
    try:
        deleted_count = jobs_service.cleanup_old_runs(days_to_keep)
        _logger.info("Cleaned up %s old runs", deleted_count)
        return {"deleted_count": deleted_count, "days_to_keep": days_to_keep}

    except Exception as e:
        _logger.exception("Failed to cleanup old runs:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup old runs: {str(e)}"
        )



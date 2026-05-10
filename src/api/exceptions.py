"""
Application exception hierarchy for the API layer.

Raise these in service/route code; the handlers in main.py convert them
to the correct HTTP responses automatically.
"""
from fastapi import Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Base class for all application-level errors."""
    status_code: int = 500

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class NotFoundError(AppError):
    """Resource does not exist."""
    status_code = 404


class ConflictError(AppError):
    """Operation would produce a duplicate or conflicting state."""
    status_code = 409


class ValidationError(AppError):
    """Input failed business-logic validation (distinct from Pydantic schema errors)."""
    status_code = 422


class ServiceUnavailableError(AppError):
    """A required downstream service is not available."""
    status_code = 503


# ---------------------------------------------------------------------------
# FastAPI exception handlers — register these in main.py
# ---------------------------------------------------------------------------

async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

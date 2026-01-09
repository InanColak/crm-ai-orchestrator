"""
API v1 Router Aggregator
========================
Collects and exports all v1 API routers.
"""

from fastapi import APIRouter

from backend.app.api.v1.workflow import router as workflow_router
from backend.app.api.v1.approvals import router as approvals_router
from backend.app.api.v1.oauth import router as oauth_router
from backend.app.api.v1.documents import router as documents_router
from backend.app.api.v1.settings import router as settings_router

# Main API router - aggregates all v1 routes
api_router = APIRouter()

# Register sub-routers
api_router.include_router(
    workflow_router,
    prefix="/workflows",
    tags=["Workflows"],
)

api_router.include_router(
    approvals_router,
    prefix="/approvals",
    tags=["Approvals"],
)

api_router.include_router(
    oauth_router,
    prefix="/oauth",
    tags=["OAuth"],
)

api_router.include_router(
    documents_router,
    prefix="/documents",
    tags=["Documents"],
)

api_router.include_router(
    settings_router,
    prefix="/settings",
    tags=["Settings"],
)

__all__ = ["api_router"]

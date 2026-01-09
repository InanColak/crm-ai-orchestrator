"""
Settings API Endpoints
======================
Provides API endpoints for checking configuration status
and managing application settings.

Note: API keys are provider-managed (not customer-specific).
Only CRM integrations and usage are exposed to clients.
"""

from typing import Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.app.core.config import Settings, get_settings

router = APIRouter()


class IntegrationStatus(BaseModel):
    """Status of a single integration."""
    name: str
    status: str  # 'connected', 'not_configured'
    status_color: str  # 'green', 'gray'
    details: Optional[str] = None


class SettingsSection(BaseModel):
    """A section of settings with multiple items."""
    id: str
    title: str
    description: str
    icon: str
    items: list[IntegrationStatus]


class UsageInfo(BaseModel):
    """Usage information for a metric."""
    used: int
    limit: int
    percentage: float


class UsageData(BaseModel):
    """Usage data for the current billing period."""
    ai_operations: UsageInfo
    documents: UsageInfo
    searches: UsageInfo
    period_start: str
    period_end: str


class SettingsStatusResponse(BaseModel):
    """Complete settings status response."""
    sections: list[SettingsSection]
    usage: Optional[UsageData] = None


def _get_status_color(status: str) -> str:
    """Map status to color."""
    return 'green' if status == 'connected' else 'gray'


def _check_integration_status(
    name: str,
    is_configured: bool,
    is_connected: bool = False,
    details: Optional[str] = None
) -> IntegrationStatus:
    """Helper to create integration status."""
    if is_connected:
        status = 'connected'
    elif is_configured:
        status = 'configured'
    else:
        status = 'not_configured'

    return IntegrationStatus(
        name=name,
        status=status,
        status_color=_get_status_color(status),
        details=details
    )


@router.get("/status", response_model=SettingsStatusResponse)
async def get_settings_status(
    settings: Settings = Depends(get_settings)
) -> SettingsStatusResponse:
    """
    Get current settings and integration status.

    Returns the configuration status of CRM integrations.
    API keys are provider-managed and not exposed to clients.
    """
    sections = []

    # 1. Integrations Section (CRM platforms only)
    integrations_items = [
        _check_integration_status(
            name="HubSpot",
            is_configured=bool(settings.hubspot_client_id and settings.hubspot_client_secret),
            is_connected=bool(settings.hubspot_access_token),
            details="OAuth connected" if settings.hubspot_access_token else None
        ),
        _check_integration_status(
            name="Salesforce",
            is_configured=bool(settings.salesforce_client_id and settings.salesforce_client_secret),
            is_connected=bool(settings.salesforce_instance_url),
            details=None
        ),
    ]

    sections.append(SettingsSection(
        id="integrations",
        title="Integrations",
        description="Connect your CRM platforms",
        icon="Link2",
        items=integrations_items
    ))

    # Usage data (demo values - will be replaced with real tracking)
    # TODO: Implement real usage tracking service
    from datetime import datetime, timedelta
    now = datetime.now()
    period_start = now.replace(day=1).strftime("%Y-%m-%d")
    next_month = (now.replace(day=28) + timedelta(days=4)).replace(day=1)
    period_end = (next_month - timedelta(days=1)).strftime("%Y-%m-%d")

    usage = UsageData(
        ai_operations=UsageInfo(used=1234, limit=10000, percentage=12.34),
        documents=UsageInfo(used=23, limit=100, percentage=23.0),
        searches=UsageInfo(used=456, limit=5000, percentage=9.12),
        period_start=period_start,
        period_end=period_end
    )

    return SettingsStatusResponse(sections=sections, usage=usage)


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    environment: str
    integrations: dict[str, bool]


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    settings: Settings = Depends(get_settings)
) -> HealthCheckResponse:
    """
    Quick health check endpoint for settings.

    Returns a simplified view of critical configuration status.
    Note: API key status is not exposed (provider-managed).
    """
    return HealthCheckResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment,
        integrations={
            "hubspot": bool(settings.hubspot_access_token or settings.hubspot_client_id),
            "salesforce": bool(settings.salesforce_client_id),
        }
    )

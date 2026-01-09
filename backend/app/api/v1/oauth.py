"""
OAuth API Endpoints
===================
Endpoints for CRM OAuth integration (HubSpot, Salesforce).
Handles authorization flows, callbacks, and connection management.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from backend.app.core.dependencies import ClientContextDep, SettingsDep
from backend.services.oauth_service import (
    CRMConnection,
    CRMProvider,
    ConnectionStatus,
    OAuthError,
    OAuthNotConnectedError,
    OAuthStateError,
    OAuthTokenError,
    get_oauth_service,
)


router = APIRouter()


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class AuthorizeResponse(BaseModel):
    """Response for authorization URL request."""
    authorization_url: str = Field(
        ...,
        description="URL to redirect user for OAuth authorization"
    )
    provider: str = Field(..., description="CRM provider")


class ConnectionStatusResponse(BaseModel):
    """Response for connection status."""
    provider: str
    is_connected: bool
    connection_status: str
    portal_id: Optional[str] = None
    scopes: list[str] = []
    expires_at: Optional[str] = None
    connected_at: Optional[str] = None
    last_used_at: Optional[str] = None


class DisconnectResponse(BaseModel):
    """Response for disconnect operation."""
    provider: str
    disconnected: bool
    message: str


class OAuthCallbackResponse(BaseModel):
    """Response after successful OAuth callback."""
    provider: str
    connection_status: str
    portal_id: Optional[str] = None
    message: str


# =============================================================================
# HUBSPOT OAUTH ENDPOINTS
# =============================================================================

@router.get(
    "/hubspot/authorize",
    response_model=AuthorizeResponse,
    summary="Get HubSpot authorization URL",
)
async def hubspot_authorize(
    client: ClientContextDep,
    settings: SettingsDep,
) -> AuthorizeResponse:
    """
    Get the HubSpot OAuth authorization URL.

    Redirect the user to this URL to initiate the OAuth flow.
    After authorization, HubSpot will redirect back to the callback URL.

    ## Usage
    1. Call this endpoint to get the authorization URL
    2. Redirect the user to the URL in their browser
    3. User logs into HubSpot and authorizes the app
    4. HubSpot redirects to `/oauth/hubspot/callback`
    5. Our app exchanges the code for tokens automatically
    """
    try:
        oauth_service = await get_oauth_service()
        auth_url = await oauth_service.get_authorization_url(
            client_id=UUID(client.client_id),
            provider=CRMProvider.HUBSPOT,
            initiated_by=client.client_name
        )

        return AuthorizeResponse(
            authorization_url=auth_url,
            provider="hubspot"
        )

    except OAuthError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get(
    "/hubspot/callback",
    summary="HubSpot OAuth callback",
    response_class=RedirectResponse,
    responses={
        302: {"description": "Redirect to frontend with status"},
        400: {"description": "OAuth error"}
    }
)
async def hubspot_callback(
    code: str = Query(..., description="Authorization code from HubSpot"),
    state: str = Query(..., description="State token for CSRF validation"),
    settings: SettingsDep = None,
) -> RedirectResponse:
    """
    Handle HubSpot OAuth callback.

    This endpoint is called by HubSpot after user authorization.
    It exchanges the authorization code for access tokens and stores them securely.

    **Note:** This endpoint is called directly by HubSpot, not by the frontend.
    After processing, it redirects to the frontend with the connection status.
    """
    try:
        oauth_service = await get_oauth_service()
        connection = await oauth_service.handle_callback(
            state=state,
            code=code,
            provider=CRMProvider.HUBSPOT
        )

        # Redirect to frontend with success
        # In production, this URL comes from settings
        frontend_url = "http://localhost:3000/settings/integrations"
        return RedirectResponse(
            url=f"{frontend_url}?status=connected&provider=hubspot&portal_id={connection.portal_id}",
            status_code=status.HTTP_302_FOUND
        )

    except OAuthStateError as e:
        frontend_url = "http://localhost:3000/settings/integrations"
        return RedirectResponse(
            url=f"{frontend_url}?status=error&provider=hubspot&error=invalid_state",
            status_code=status.HTTP_302_FOUND
        )

    except OAuthTokenError as e:
        frontend_url = "http://localhost:3000/settings/integrations"
        return RedirectResponse(
            url=f"{frontend_url}?status=error&provider=hubspot&error=token_exchange_failed",
            status_code=status.HTTP_302_FOUND
        )


@router.post(
    "/hubspot/refresh",
    response_model=ConnectionStatusResponse,
    summary="Manually refresh HubSpot token",
)
async def hubspot_refresh_token(
    client: ClientContextDep,
) -> ConnectionStatusResponse:
    """
    Manually refresh the HubSpot access token.

    Normally, tokens are refreshed automatically when needed.
    Use this endpoint to force a refresh.
    """
    try:
        oauth_service = await get_oauth_service()

        # This will trigger refresh if needed
        await oauth_service.get_access_token(
            client_id=UUID(client.client_id),
            provider=CRMProvider.HUBSPOT
        )

        # Get updated status
        connection = await oauth_service.get_connection_status(
            client_id=UUID(client.client_id),
            provider=CRMProvider.HUBSPOT
        )

        return _connection_to_response(connection)

    except OAuthNotConnectedError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="HubSpot is not connected for this client"
        )
    except OAuthTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Token refresh failed: {e}"
        )


@router.delete(
    "/hubspot/disconnect",
    response_model=DisconnectResponse,
    summary="Disconnect HubSpot integration",
)
async def hubspot_disconnect(
    client: ClientContextDep,
) -> DisconnectResponse:
    """
    Disconnect the HubSpot integration.

    This revokes the stored tokens and marks the connection as disconnected.
    The user will need to re-authorize to reconnect.
    """
    try:
        oauth_service = await get_oauth_service()
        success = await oauth_service.disconnect(
            client_id=UUID(client.client_id),
            provider=CRMProvider.HUBSPOT,
            disconnected_by=client.client_name
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="HubSpot is not connected for this client"
            )

        return DisconnectResponse(
            provider="hubspot",
            disconnected=True,
            message="HubSpot integration has been disconnected"
        )

    except OAuthError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# =============================================================================
# GENERIC STATUS ENDPOINT
# =============================================================================

@router.get(
    "/status",
    response_model=dict[str, ConnectionStatusResponse],
    summary="Get all CRM connection statuses",
)
async def get_all_connection_status(
    client: ClientContextDep,
) -> dict[str, ConnectionStatusResponse]:
    """
    Get connection status for all CRM integrations.

    Returns status for HubSpot and Salesforce (when implemented).
    """
    oauth_service = await get_oauth_service()

    statuses = {}

    # HubSpot status
    hubspot_connection = await oauth_service.get_connection_status(
        client_id=UUID(client.client_id),
        provider=CRMProvider.HUBSPOT
    )
    statuses["hubspot"] = _connection_to_response(hubspot_connection)

    # Salesforce status (placeholder for future)
    statuses["salesforce"] = ConnectionStatusResponse(
        provider="salesforce",
        is_connected=False,
        connection_status="disconnected",
        scopes=[]
    )

    return statuses


@router.get(
    "/status/{provider}",
    response_model=ConnectionStatusResponse,
    summary="Get CRM connection status",
)
async def get_connection_status(
    provider: str,
    client: ClientContextDep,
) -> ConnectionStatusResponse:
    """
    Get connection status for a specific CRM provider.
    """
    try:
        crm_provider = CRMProvider(provider.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid provider: {provider}. Supported: hubspot, salesforce"
        )

    oauth_service = await get_oauth_service()
    connection = await oauth_service.get_connection_status(
        client_id=UUID(client.client_id),
        provider=crm_provider
    )

    return _connection_to_response(connection)


# =============================================================================
# HELPERS
# =============================================================================

def _connection_to_response(connection: CRMConnection) -> ConnectionStatusResponse:
    """Convert CRMConnection to API response."""
    return ConnectionStatusResponse(
        provider=connection.provider.value,
        is_connected=connection.is_connected,
        connection_status=connection.connection_status.value,
        portal_id=connection.portal_id,
        scopes=connection.scopes,
        expires_at=connection.expires_at.isoformat() if connection.expires_at else None,
        connected_at=connection.connected_at.isoformat() if connection.connected_at else None,
        last_used_at=connection.last_used_at.isoformat() if connection.last_used_at else None
    )

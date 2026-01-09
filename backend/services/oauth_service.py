"""
OAuth Service
=============
Handles OAuth 2.0 flows for CRM integrations (HubSpot, Salesforce).
Manages authorization, token exchange, refresh, and secure storage.
"""

import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional
from uuid import UUID

import httpx
from pydantic import BaseModel

from backend.app.core.config import Settings, get_settings
from backend.services.encryption_service import (
    EncryptionService,
    get_encryption_service,
)
from backend.services.supabase_client import SupabaseService, get_supabase


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class CRMProvider(str, Enum):
    """Supported CRM providers."""
    HUBSPOT = "hubspot"
    SALESFORCE = "salesforce"


class ConnectionStatus(str, Enum):
    """CRM connection status values."""
    CONNECTED = "connected"
    EXPIRED = "expired"
    REFRESH_FAILED = "refresh_failed"
    REVOKED = "revoked"
    DISCONNECTED = "disconnected"


# HubSpot OAuth endpoints
HUBSPOT_AUTHORIZE_URL = "https://app.hubspot.com/oauth/authorize"
HUBSPOT_TOKEN_URL = "https://api.hubapi.com/oauth/v1/token"
HUBSPOT_TOKEN_INFO_URL = "https://api.hubapi.com/oauth/v1/access-tokens"


# =============================================================================
# EXCEPTIONS
# =============================================================================

class OAuthError(Exception):
    """Base exception for OAuth operations."""
    pass


class OAuthStateError(OAuthError):
    """Invalid or expired OAuth state."""
    pass


class OAuthTokenError(OAuthError):
    """Token exchange or refresh failed."""
    pass


class OAuthNotConnectedError(OAuthError):
    """CRM is not connected for this client."""
    pass


# =============================================================================
# DATA MODELS
# =============================================================================

class OAuthTokens(BaseModel):
    """OAuth token response data."""
    access_token: str
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None  # seconds
    token_type: str = "Bearer"
    scopes: list[str] = []

    # Provider-specific
    portal_id: Optional[str] = None  # HubSpot
    hub_id: Optional[str] = None  # HubSpot
    instance_url: Optional[str] = None  # Salesforce


class CRMConnection(BaseModel):
    """CRM connection status for a client."""
    client_id: UUID
    provider: CRMProvider
    is_connected: bool
    connection_status: ConnectionStatus
    portal_id: Optional[str] = None
    scopes: list[str] = []
    expires_at: Optional[datetime] = None
    connected_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None


# =============================================================================
# OAUTH SERVICE
# =============================================================================

class OAuthService:
    """
    OAuth 2.0 Service for CRM Integrations.

    Handles the complete OAuth lifecycle:
    - Authorization URL generation with CSRF protection
    - Token exchange and secure storage
    - Automatic token refresh
    - Connection status management

    Usage:
        >>> service = await OAuthService.create()
        >>> auth_url = await service.get_authorization_url(client_id, "hubspot")
        >>> # User visits auth_url and authorizes
        >>> tokens = await service.handle_callback(state, code)
    """

    def __init__(
        self,
        db: SupabaseService,
        encryption: EncryptionService,
        settings: Settings
    ):
        self._db = db
        self._encryption = encryption
        self._settings = settings
        self._http: Optional[httpx.AsyncClient] = None

    @classmethod
    async def create(cls) -> "OAuthService":
        """Factory method to create OAuthService with dependencies."""
        db = await get_supabase()
        encryption = get_encryption_service()
        settings = get_settings()
        return cls(db, encryption, settings)

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=30.0)
        return self._http

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http:
            await self._http.aclose()
            self._http = None

    # =========================================================================
    # AUTHORIZATION
    # =========================================================================

    async def get_authorization_url(
        self,
        client_id: UUID,
        provider: CRMProvider,
        initiated_by: Optional[str] = None
    ) -> str:
        """
        Generate OAuth authorization URL.

        Creates a secure state token and stores it for CSRF validation.

        Args:
            client_id: Client UUID initiating the connection
            provider: CRM provider (hubspot, salesforce)
            initiated_by: User email/name who initiated

        Returns:
            Authorization URL to redirect user to

        Raises:
            OAuthError: If provider is not supported
        """
        if provider == CRMProvider.HUBSPOT:
            return await self._get_hubspot_auth_url(client_id, initiated_by)
        elif provider == CRMProvider.SALESFORCE:
            raise OAuthError("Salesforce OAuth not yet implemented")
        else:
            raise OAuthError(f"Unsupported provider: {provider}")

    async def _get_hubspot_auth_url(
        self,
        client_id: UUID,
        initiated_by: Optional[str]
    ) -> str:
        """Generate HubSpot OAuth authorization URL."""
        # Validate configuration
        if not self._settings.hubspot_client_id:
            raise OAuthError("HUBSPOT_CLIENT_ID is not configured")

        # Generate secure state token
        state_token = secrets.token_urlsafe(32)

        # Store state for validation
        await self._db.insert("oauth_states", {
            "client_id": str(client_id),
            "provider": CRMProvider.HUBSPOT.value,
            "state_token": state_token,
            "redirect_uri": self._settings.hubspot_oauth_redirect_uri,
            "requested_scopes": self._settings.hubspot_oauth_scopes,
            "initiated_by": initiated_by,
            "expires_at": (
                datetime.now(timezone.utc) + timedelta(minutes=10)
            ).isoformat()
        })

        # Build authorization URL
        scopes = " ".join(self._settings.hubspot_oauth_scopes)
        params = {
            "client_id": self._settings.hubspot_client_id,
            "redirect_uri": self._settings.hubspot_oauth_redirect_uri,
            "scope": scopes,
            "state": state_token
        }

        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{HUBSPOT_AUTHORIZE_URL}?{query}"

    # =========================================================================
    # TOKEN EXCHANGE
    # =========================================================================

    async def handle_callback(
        self,
        state: str,
        code: str,
        provider: CRMProvider = CRMProvider.HUBSPOT
    ) -> CRMConnection:
        """
        Handle OAuth callback and exchange code for tokens.

        Args:
            state: State token from callback
            code: Authorization code from callback
            provider: CRM provider

        Returns:
            CRMConnection status

        Raises:
            OAuthStateError: If state is invalid or expired
            OAuthTokenError: If token exchange fails
        """
        # Validate state
        oauth_state = await self._validate_state(state, provider)
        client_id = UUID(oauth_state["client_id"])

        # Exchange code for tokens
        if provider == CRMProvider.HUBSPOT:
            tokens = await self._exchange_hubspot_code(code)
        else:
            raise OAuthError(f"Unsupported provider: {provider}")

        # Store encrypted tokens
        await self._store_tokens(client_id, provider, tokens)

        # Mark state as used
        await self._db.update(
            "oauth_states",
            oauth_state["id"],
            {"is_used": True, "used_at": datetime.now(timezone.utc).isoformat()}
        )

        # Log the connection
        await self._log_credential_event(
            client_id=client_id,
            provider=provider,
            operation="token_created",
            performed_by=oauth_state.get("initiated_by"),
            details={"scopes": tokens.scopes, "portal_id": tokens.portal_id}
        )

        return CRMConnection(
            client_id=client_id,
            provider=provider,
            is_connected=True,
            connection_status=ConnectionStatus.CONNECTED,
            portal_id=tokens.portal_id,
            scopes=tokens.scopes,
            expires_at=self._calculate_expiry(tokens.expires_in),
            connected_at=datetime.now(timezone.utc)
        )

    async def _validate_state(
        self,
        state: str,
        provider: CRMProvider
    ) -> dict:
        """Validate OAuth state token."""
        results = await self._db.fetch_many(
            "oauth_states",
            filters={
                "state_token": state,
                "provider": provider.value,
                "is_used": False
            },
            limit=1
        )

        if not results:
            raise OAuthStateError("Invalid or expired OAuth state")

        oauth_state = results[0]

        # Check expiry
        expires_at = datetime.fromisoformat(
            oauth_state["expires_at"].replace("Z", "+00:00")
        )
        if expires_at < datetime.now(timezone.utc):
            raise OAuthStateError("OAuth state has expired")

        return oauth_state

    async def _exchange_hubspot_code(self, code: str) -> OAuthTokens:
        """Exchange authorization code for HubSpot tokens."""
        http = await self._get_http_client()

        response = await http.post(
            HUBSPOT_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": self._settings.hubspot_client_id,
                "client_secret": self._settings.hubspot_client_secret,
                "redirect_uri": self._settings.hubspot_oauth_redirect_uri,
                "code": code
            }
        )

        if response.status_code != 200:
            raise OAuthTokenError(
                f"Token exchange failed: {response.status_code} - {response.text}"
            )

        data = response.json()

        # Get token info for portal_id
        token_info = await self._get_hubspot_token_info(data["access_token"])

        return OAuthTokens(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            expires_in=data.get("expires_in"),
            token_type=data.get("token_type", "Bearer"),
            scopes=token_info.get("scopes", []),
            portal_id=str(token_info.get("hub_id")),
            hub_id=str(token_info.get("hub_id"))
        )

    async def _get_hubspot_token_info(self, access_token: str) -> dict:
        """Get HubSpot token info (scopes, hub_id)."""
        http = await self._get_http_client()

        response = await http.get(
            f"{HUBSPOT_TOKEN_INFO_URL}/{access_token}"
        )

        if response.status_code != 200:
            return {}

        return response.json()

    # =========================================================================
    # TOKEN STORAGE
    # =========================================================================

    async def _store_tokens(
        self,
        client_id: UUID,
        provider: CRMProvider,
        tokens: OAuthTokens
    ) -> None:
        """Store encrypted OAuth tokens in database."""
        # Encrypt tokens
        access_encrypted = self._encryption.encrypt(tokens.access_token)
        refresh_encrypted = None
        if tokens.refresh_token:
            refresh_encrypted = self._encryption.encrypt(tokens.refresh_token)

        # Check if existing credential exists
        existing = await self._db.fetch_many(
            "crm_credentials",
            filters={"client_id": str(client_id), "provider": provider.value},
            limit=1
        )

        credential_data = {
            "client_id": str(client_id),
            "provider": provider.value,
            "access_token_encrypted": access_encrypted.ciphertext,
            "encryption_iv": access_encrypted.iv,
            "encryption_key_version": access_encrypted.key_version,
            "expires_at": self._calculate_expiry(tokens.expires_in).isoformat() if tokens.expires_in else None,
            "token_type": tokens.token_type,
            "scopes": tokens.scopes,
            "provider_metadata": {
                "portal_id": tokens.portal_id,
                "hub_id": tokens.hub_id,
                "instance_url": tokens.instance_url
            },
            "is_active": True,
            "connection_status": ConnectionStatus.CONNECTED.value,
            "connected_at": datetime.now(timezone.utc).isoformat()
        }

        if refresh_encrypted:
            credential_data["refresh_token_encrypted"] = refresh_encrypted.ciphertext
            # Note: Using same IV for simplicity, production should use separate IVs

        if existing:
            await self._db.update("crm_credentials", existing[0]["id"], credential_data)
        else:
            await self._db.insert("crm_credentials", credential_data)

    # =========================================================================
    # TOKEN RETRIEVAL
    # =========================================================================

    async def get_access_token(
        self,
        client_id: UUID,
        provider: CRMProvider = CRMProvider.HUBSPOT
    ) -> str:
        """
        Get decrypted access token for a client.

        Automatically refreshes if token is expired.

        Args:
            client_id: Client UUID
            provider: CRM provider

        Returns:
            Decrypted access token

        Raises:
            OAuthNotConnectedError: If client is not connected
            OAuthTokenError: If token refresh fails
        """
        credential = await self._get_credential(client_id, provider)

        if not credential:
            raise OAuthNotConnectedError(
                f"No {provider.value} connection for client {client_id}"
            )

        # Check if refresh needed
        if self._needs_refresh(credential):
            credential = await self._refresh_token(client_id, provider, credential)

        # Decrypt and return
        return self._encryption.decrypt(
            credential["access_token_encrypted"],
            credential["encryption_iv"],
            credential.get("encryption_key_version", 1)
        )

    async def _get_credential(
        self,
        client_id: UUID,
        provider: CRMProvider
    ) -> Optional[dict]:
        """Get credential record from database."""
        results = await self._db.fetch_many(
            "crm_credentials",
            filters={
                "client_id": str(client_id),
                "provider": provider.value,
                "is_active": True
            },
            limit=1
        )
        return results[0] if results else None

    def _needs_refresh(self, credential: dict) -> bool:
        """Check if token needs refresh (within 5 minutes of expiry)."""
        if not credential.get("expires_at"):
            return False

        expires_at = datetime.fromisoformat(
            credential["expires_at"].replace("Z", "+00:00")
        )
        buffer = timedelta(minutes=5)
        return datetime.now(timezone.utc) + buffer >= expires_at

    # =========================================================================
    # TOKEN REFRESH
    # =========================================================================

    async def _refresh_token(
        self,
        client_id: UUID,
        provider: CRMProvider,
        credential: dict
    ) -> dict:
        """Refresh expired token."""
        if not credential.get("refresh_token_encrypted"):
            await self._update_connection_status(
                credential["id"],
                ConnectionStatus.EXPIRED
            )
            raise OAuthTokenError("No refresh token available")

        try:
            # Decrypt refresh token
            refresh_token = self._encryption.decrypt(
                credential["refresh_token_encrypted"],
                credential["encryption_iv"],
                credential.get("encryption_key_version", 1)
            )

            # Refresh based on provider
            if provider == CRMProvider.HUBSPOT:
                new_tokens = await self._refresh_hubspot_token(refresh_token)
            else:
                raise OAuthError(f"Refresh not supported for {provider}")

            # Store new tokens
            await self._store_tokens(client_id, provider, new_tokens)

            # Log refresh
            await self._log_credential_event(
                client_id=client_id,
                provider=provider,
                operation="token_refreshed"
            )

            # Return updated credential
            return await self._get_credential(client_id, provider)

        except Exception as e:
            await self._update_connection_status(
                credential["id"],
                ConnectionStatus.REFRESH_FAILED
            )
            await self._log_credential_event(
                client_id=client_id,
                provider=provider,
                operation="refresh_failed",
                error_message=str(e)
            )
            raise OAuthTokenError(f"Token refresh failed: {e}")

    async def _refresh_hubspot_token(self, refresh_token: str) -> OAuthTokens:
        """Refresh HubSpot access token."""
        http = await self._get_http_client()

        response = await http.post(
            HUBSPOT_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "client_id": self._settings.hubspot_client_id,
                "client_secret": self._settings.hubspot_client_secret,
                "refresh_token": refresh_token
            }
        )

        if response.status_code != 200:
            raise OAuthTokenError(
                f"Token refresh failed: {response.status_code} - {response.text}"
            )

        data = response.json()
        token_info = await self._get_hubspot_token_info(data["access_token"])

        return OAuthTokens(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", refresh_token),
            expires_in=data.get("expires_in"),
            token_type=data.get("token_type", "Bearer"),
            scopes=token_info.get("scopes", []),
            portal_id=str(token_info.get("hub_id")),
            hub_id=str(token_info.get("hub_id"))
        )

    # =========================================================================
    # DISCONNECTION
    # =========================================================================

    async def disconnect(
        self,
        client_id: UUID,
        provider: CRMProvider,
        disconnected_by: Optional[str] = None
    ) -> bool:
        """
        Disconnect CRM integration.

        Args:
            client_id: Client UUID
            provider: CRM provider
            disconnected_by: User who initiated disconnect

        Returns:
            True if disconnected successfully
        """
        credential = await self._get_credential(client_id, provider)
        if not credential:
            return False

        await self._db.update("crm_credentials", credential["id"], {
            "is_active": False,
            "connection_status": ConnectionStatus.DISCONNECTED.value,
            "disconnected_by": disconnected_by,
            "disconnected_at": datetime.now(timezone.utc).isoformat()
        })

        await self._log_credential_event(
            client_id=client_id,
            provider=provider,
            operation="token_disconnected",
            performed_by=disconnected_by
        )

        return True

    # =========================================================================
    # STATUS
    # =========================================================================

    async def get_connection_status(
        self,
        client_id: UUID,
        provider: CRMProvider = CRMProvider.HUBSPOT
    ) -> CRMConnection:
        """Get current connection status for a client."""
        credential = await self._get_credential(client_id, provider)

        if not credential:
            return CRMConnection(
                client_id=client_id,
                provider=provider,
                is_connected=False,
                connection_status=ConnectionStatus.DISCONNECTED
            )

        metadata = credential.get("provider_metadata", {})

        return CRMConnection(
            client_id=client_id,
            provider=provider,
            is_connected=credential["is_active"],
            connection_status=ConnectionStatus(credential["connection_status"]),
            portal_id=metadata.get("portal_id"),
            scopes=credential.get("scopes", []),
            expires_at=datetime.fromisoformat(
                credential["expires_at"].replace("Z", "+00:00")
            ) if credential.get("expires_at") else None,
            connected_at=datetime.fromisoformat(
                credential["connected_at"].replace("Z", "+00:00")
            ) if credential.get("connected_at") else None,
            last_used_at=datetime.fromisoformat(
                credential["last_used_at"].replace("Z", "+00:00")
            ) if credential.get("last_used_at") else None
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _calculate_expiry(self, expires_in: Optional[int]) -> datetime:
        """Calculate token expiry datetime."""
        if expires_in:
            return datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        # Default to 6 hours for HubSpot
        return datetime.now(timezone.utc) + timedelta(hours=6)

    async def _update_connection_status(
        self,
        credential_id: str,
        status: ConnectionStatus
    ) -> None:
        """Update connection status in database."""
        await self._db.update("crm_credentials", credential_id, {
            "connection_status": status.value
        })

    async def _log_credential_event(
        self,
        client_id: UUID,
        provider: CRMProvider,
        operation: str,
        performed_by: Optional[str] = None,
        details: Optional[dict] = None,
        error_message: Optional[str] = None
    ) -> None:
        """Log credential operation for audit."""
        # Get credential_id if exists
        credential = await self._get_credential(client_id, provider)

        await self._db.insert("crm_credential_audit", {
            "credential_id": credential["id"] if credential else None,
            "client_id": str(client_id),
            "operation": operation,
            "provider": provider.value,
            "performed_by": performed_by,
            "details": details or {},
            "error_message": error_message
        })


# =============================================================================
# SINGLETON & DEPENDENCY INJECTION
# =============================================================================

_oauth_service: Optional[OAuthService] = None


async def get_oauth_service() -> OAuthService:
    """
    Get singleton OAuthService instance.

    Returns:
        OAuthService instance
    """
    global _oauth_service
    if _oauth_service is None:
        _oauth_service = await OAuthService.create()
    return _oauth_service


async def reset_oauth_service() -> None:
    """Reset singleton (for testing)."""
    global _oauth_service
    if _oauth_service:
        await _oauth_service.close()
    _oauth_service = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "CRMProvider",
    "ConnectionStatus",
    # Exceptions
    "OAuthError",
    "OAuthStateError",
    "OAuthTokenError",
    "OAuthNotConnectedError",
    # Models
    "OAuthTokens",
    "CRMConnection",
    # Service
    "OAuthService",
    # DI
    "get_oauth_service",
    "reset_oauth_service",
]

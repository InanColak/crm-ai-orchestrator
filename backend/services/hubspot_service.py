"""
HubSpot Service - CRM SDK Integration
=====================================
Centralized service for all HubSpot CRM operations.

Features:
- Async wrapper around HubSpot Python SDK
- Rate limiting with token bucket
- Retry logic with exponential backoff
- Full audit logging support
- Type-safe operations with Pydantic validation

FLOW Methodology:
- Function: All HubSpot CRUD operations
- Level: Production-ready with error handling
- Output: Type-safe responses
- Win Metric: Zero unhandled exceptions, <200ms response time

ADR-014 Compliance:
- All write operations return data for approval system
- No direct mutations - everything goes through HITL
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, TypeVar

from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from backend.app.core.config import get_settings
from backend.app.schemas.hubspot import (
    HubSpotObjectType,
    Contact,
    ContactCreate,
    ContactUpdate,
    Company,
    CompanyCreate,
    Deal,
    DealCreate,
    DealUpdate,
    Task,
    TaskCreate,
    TaskUpdate,
    Note,
    NoteCreate,
    SearchRequest,
    SearchResult,
    Owner,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class HubSpotError(Exception):
    """Base exception for HubSpot operations."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: dict | None = None,
        retryable: bool = True,
    ):
        self.message = message
        self.status_code = status_code
        self.response_body = response_body
        self.retryable = retryable
        super().__init__(self.message)


class HubSpotRateLimitError(HubSpotError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: int | None = None, **kwargs):
        super().__init__(message, retryable=True, **kwargs)
        self.retry_after = retry_after


class HubSpotNotFoundError(HubSpotError):
    """Resource not found."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, status_code=404, retryable=False, **kwargs)


class HubSpotValidationError(HubSpotError):
    """Validation error from HubSpot API."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, status_code=400, retryable=False, **kwargs)


class HubSpotAuthError(HubSpotError):
    """Authentication/authorization error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, status_code=401, retryable=False, **kwargs)


# =============================================================================
# RATE LIMITER
# =============================================================================

class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for HubSpot API calls.

    HubSpot limits:
    - Private apps: 100 requests per 10 seconds
    - OAuth apps: 100 requests per 10 seconds
    """

    def __init__(self, requests_per_second: int = 10, max_burst: int | None = None):
        self.requests_per_second = requests_per_second
        self.max_burst = max_burst or requests_per_second * 2
        self._tokens = float(self.max_burst)
        self._last_update = datetime.now(timezone.utc)
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens, returns True if successful."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            elapsed = (now - self._last_update).total_seconds()

            # Refill tokens
            refill = elapsed * self.requests_per_second
            self._tokens = min(self.max_burst, self._tokens + refill)
            self._last_update = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            return False

    async def wait_for_token(self, timeout: float = 30.0) -> bool:
        """Wait until a token is available."""
        start = datetime.now(timezone.utc)

        while True:
            if await self.acquire():
                return True

            elapsed = (datetime.now(timezone.utc) - start).total_seconds()
            if elapsed >= timeout:
                return False

            # Wait a bit before retrying
            await asyncio.sleep(0.1)


# =============================================================================
# HUBSPOT CLIENT FACTORY (Multi-Tenant)
# =============================================================================

class HubSpotClientFactory:
    """
    Factory for creating per-client HubSpot SDK instances.

    Supports multi-tenant architecture where each customer has their own
    HubSpot portal and OAuth tokens.

    Usage:
        >>> factory = HubSpotClientFactory.get_instance()
        >>> service = await factory.get_service(client_id)
        >>> contact = await service.get_contact("123")
    """

    _instance: HubSpotClientFactory | None = None

    # Cache of HubSpot clients per client_id
    _clients: dict[str, Any] = {}
    _rate_limiters: dict[str, TokenBucketRateLimiter] = {}

    def __init__(self):
        self._settings = get_settings()

    @classmethod
    def get_instance(cls) -> HubSpotClientFactory:
        """Get singleton factory instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def get_client(self, client_id: str) -> Any:
        """
        Get or create HubSpot SDK client for a customer.

        Args:
            client_id: Customer's client_id

        Returns:
            Configured HubSpot SDK client

        Raises:
            HubSpotAuthError: If no valid token exists for client
        """
        # Check cache first
        if client_id in self._clients:
            return self._clients[client_id]

        # Get token from OAuth service
        try:
            from backend.services.oauth_service import (
                CRMProvider,
                get_oauth_service,
                OAuthNotConnectedError,
            )
            from uuid import UUID

            oauth_service = await get_oauth_service()
            access_token = await oauth_service.get_access_token(
                client_id=UUID(client_id),
                provider=CRMProvider.HUBSPOT
            )

            # Create SDK client
            from hubspot import HubSpot
            client = HubSpot(access_token=access_token)

            # Cache it
            self._clients[client_id] = client
            self._rate_limiters[client_id] = TokenBucketRateLimiter(
                requests_per_second=self._settings.hubspot_rate_limit_per_second
            )

            logger.info(f"Created HubSpot client for client {client_id}")
            return client

        except ImportError:
            raise HubSpotAuthError("hubspot-api-client not installed")
        except OAuthNotConnectedError:
            raise HubSpotAuthError(
                f"HubSpot not connected for client {client_id}. "
                "Please connect HubSpot via OAuth."
            )
        except Exception as e:
            raise HubSpotAuthError(f"Failed to get HubSpot client: {e}")

    def get_rate_limiter(self, client_id: str) -> TokenBucketRateLimiter | None:
        """Get rate limiter for a client."""
        return self._rate_limiters.get(client_id)

    def invalidate_client(self, client_id: str) -> None:
        """Remove cached client (e.g., after token refresh or disconnect)."""
        self._clients.pop(client_id, None)
        self._rate_limiters.pop(client_id, None)
        logger.info(f"Invalidated HubSpot client cache for {client_id}")

    async def get_service(self, client_id: str) -> "HubSpotService":
        """
        Get a HubSpotService instance configured for a specific client.

        Args:
            client_id: Customer's client_id

        Returns:
            HubSpotService configured with client's credentials
        """
        client = await self.get_client(client_id)
        rate_limiter = self.get_rate_limiter(client_id)
        return HubSpotService(
            client=client,
            rate_limiter=rate_limiter,
            client_id=client_id
        )


# =============================================================================
# HUBSPOT SERVICE
# =============================================================================

class HubSpotService:
    """
    Centralized HubSpot CRM Service.

    Provides type-safe, rate-limited, retry-enabled access to HubSpot API.
    All operations are async and return Pydantic models.

    Multi-Tenant Usage (Recommended):
        >>> factory = HubSpotClientFactory.get_instance()
        >>> service = await factory.get_service(client_id)
        >>> contact = await service.get_contact("123")

    Legacy Usage (Development with static token):
        >>> service = HubSpotService.get_instance()
        >>> contact = await service.get_contact("123")
    """

    _instance: HubSpotService | None = None

    def __init__(
        self,
        client: Any = None,
        rate_limiter: TokenBucketRateLimiter | None = None,
        client_id: str | None = None
    ):
        """
        Initialize HubSpotService.

        Args:
            client: Pre-configured HubSpot SDK client (for multi-tenant)
            rate_limiter: Rate limiter instance (for multi-tenant)
            client_id: Customer client_id (for logging/tracking)
        """
        self._settings = get_settings()
        self._client: Any = client
        self._rate_limiter: TokenBucketRateLimiter | None = rate_limiter
        self._client_id = client_id
        self._initialized = client is not None

    @classmethod
    def get_instance(cls) -> HubSpotService:
        """
        Get singleton instance (legacy/development mode).

        Uses the static HUBSPOT_ACCESS_TOKEN from environment.
        For production multi-tenant, use HubSpotClientFactory instead.
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._initialize_legacy()
        return cls._instance

    def _initialize_legacy(self) -> None:
        """Initialize HubSpot client with legacy static token."""
        if self._initialized:
            return

        settings = self._settings

        if not settings.hubspot_access_token:
            logger.warning("HubSpot access token not configured (legacy mode)")
            self._initialized = True
            return

        try:
            from hubspot import HubSpot

            self._client = HubSpot(access_token=settings.hubspot_access_token)
            self._rate_limiter = TokenBucketRateLimiter(
                requests_per_second=settings.hubspot_rate_limit_per_second
            )
            self._client_id = "legacy-static-token"

            logger.info("HubSpot client initialized (legacy mode)")

        except ImportError:
            logger.error("hubspot-api-client not installed")
        except Exception as e:
            logger.error(f"Failed to initialize HubSpot client: {e}")

        self._initialized = True

    @property
    def is_configured(self) -> bool:
        """Check if HubSpot is properly configured."""
        return self._client is not None

    @property
    def client_id(self) -> str | None:
        """Get the client_id this service is configured for."""
        return self._client_id

    def _ensure_configured(self) -> None:
        """Raise error if not configured."""
        if not self.is_configured:
            raise HubSpotAuthError(
                "HubSpot is not configured. "
                "Use HubSpotClientFactory.get_service(client_id) for multi-tenant, "
                "or set HUBSPOT_ACCESS_TOKEN for legacy mode."
            )

    async def _rate_limit(self) -> None:
        """Apply rate limiting."""
        if self._rate_limiter:
            if not await self._rate_limiter.wait_for_token():
                raise HubSpotRateLimitError("Rate limit exceeded, timeout waiting for token")

    def _handle_error(self, e: Exception, operation: str) -> None:
        """Convert SDK exceptions to our exceptions."""
        error_str = str(e).lower()
        status_code = getattr(e, "status", None) or getattr(e, "status_code", None)

        if status_code == 404 or "not found" in error_str:
            raise HubSpotNotFoundError(f"{operation}: Resource not found") from e

        if status_code == 401 or "unauthorized" in error_str:
            raise HubSpotAuthError(f"{operation}: Authentication failed") from e

        if status_code == 429 or "rate limit" in error_str:
            retry_after = getattr(e, "retry_after", None)
            raise HubSpotRateLimitError(
                f"{operation}: Rate limit exceeded",
                retry_after=retry_after
            ) from e

        if status_code == 400 or "validation" in error_str:
            raise HubSpotValidationError(f"{operation}: {e}") from e

        raise HubSpotError(f"{operation}: {e}", status_code=status_code) from e

    # =========================================================================
    # CONTACTS
    # =========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((HubSpotRateLimitError,)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def get_contact(self, contact_id: str, properties: list[str] | None = None) -> Contact:
        """
        Get a contact by ID.

        Args:
            contact_id: HubSpot contact ID
            properties: Optional list of properties to fetch

        Returns:
            Contact model
        """
        self._ensure_configured()
        await self._rate_limit()

        try:
            properties = properties or [
                "email", "firstname", "lastname", "phone",
                "company", "jobtitle", "lifecyclestage"
            ]

            response = await asyncio.to_thread(
                self._client.crm.contacts.basic_api.get_by_id,
                contact_id=contact_id,
                properties=properties,
            )

            return Contact(
                id=response.id,
                properties=response.properties,
                created_at=response.created_at,
                updated_at=response.updated_at,
                archived=response.archived or False,
            )

        except Exception as e:
            self._handle_error(e, "get_contact")

    async def search_contacts(
        self,
        email: str | None = None,
        firstname: str | None = None,
        lastname: str | None = None,
        company: str | None = None,
        limit: int = 10,
    ) -> list[Contact]:
        """
        Search contacts by various criteria.

        Args:
            email: Filter by email
            firstname: Filter by first name
            lastname: Filter by last name
            company: Filter by company name
            limit: Max results (1-100)

        Returns:
            List of matching contacts
        """
        self._ensure_configured()
        await self._rate_limit()

        try:
            filters = []

            if email:
                filters.append({
                    "propertyName": "email",
                    "operator": "EQ",
                    "value": email
                })
            if firstname:
                filters.append({
                    "propertyName": "firstname",
                    "operator": "CONTAINS_TOKEN",
                    "value": firstname
                })
            if lastname:
                filters.append({
                    "propertyName": "lastname",
                    "operator": "CONTAINS_TOKEN",
                    "value": lastname
                })
            if company:
                filters.append({
                    "propertyName": "company",
                    "operator": "CONTAINS_TOKEN",
                    "value": company
                })

            search_request = {
                "filterGroups": [{"filters": filters}] if filters else [],
                "properties": [
                    "email", "firstname", "lastname", "phone",
                    "company", "jobtitle", "lifecyclestage"
                ],
                "limit": min(limit, 100),
            }

            response = await asyncio.to_thread(
                self._client.crm.contacts.search_api.do_search,
                public_object_search_request=search_request,
            )

            return [
                Contact(
                    id=r.id,
                    properties=r.properties,
                    created_at=r.created_at,
                    updated_at=r.updated_at,
                    archived=r.archived or False,
                )
                for r in response.results
            ]

        except Exception as e:
            self._handle_error(e, "search_contacts")

    async def create_contact(self, data: ContactCreate) -> Contact:
        """
        Create a new contact.

        Args:
            data: Contact creation data

        Returns:
            Created contact
        """
        self._ensure_configured()
        await self._rate_limit()

        try:
            properties = data.properties.model_dump(exclude_none=True)

            response = await asyncio.to_thread(
                self._client.crm.contacts.basic_api.create,
                simple_public_object_input_for_create={"properties": properties},
            )

            logger.info(f"Created contact: {response.id}")

            return Contact(
                id=response.id,
                properties=response.properties,
                created_at=response.created_at,
                updated_at=response.updated_at,
            )

        except Exception as e:
            self._handle_error(e, "create_contact")

    async def update_contact(self, contact_id: str, data: ContactUpdate) -> Contact:
        """
        Update an existing contact.

        Args:
            contact_id: HubSpot contact ID
            data: Contact update data

        Returns:
            Updated contact
        """
        self._ensure_configured()
        await self._rate_limit()

        try:
            properties = data.properties.model_dump(exclude_none=True)

            response = await asyncio.to_thread(
                self._client.crm.contacts.basic_api.update,
                contact_id=contact_id,
                simple_public_object_input={"properties": properties},
            )

            logger.info(f"Updated contact: {contact_id}")

            return Contact(
                id=response.id,
                properties=response.properties,
                created_at=response.created_at,
                updated_at=response.updated_at,
            )

        except Exception as e:
            self._handle_error(e, "update_contact")

    # =========================================================================
    # DEALS
    # =========================================================================

    async def get_deal(self, deal_id: str, properties: list[str] | None = None) -> Deal:
        """Get a deal by ID."""
        self._ensure_configured()
        await self._rate_limit()

        try:
            properties = properties or [
                "dealname", "amount", "dealstage", "pipeline",
                "closedate", "hubspot_owner_id", "description"
            ]

            response = await asyncio.to_thread(
                self._client.crm.deals.basic_api.get_by_id,
                deal_id=deal_id,
                properties=properties,
            )

            return Deal(
                id=response.id,
                properties=response.properties,
                created_at=response.created_at,
                updated_at=response.updated_at,
                archived=response.archived or False,
            )

        except Exception as e:
            self._handle_error(e, "get_deal")

    async def search_deals(
        self,
        dealname: str | None = None,
        dealstage: str | None = None,
        pipeline: str | None = None,
        owner_id: str | None = None,
        limit: int = 10,
    ) -> list[Deal]:
        """Search deals by various criteria."""
        self._ensure_configured()
        await self._rate_limit()

        try:
            filters = []

            if dealname:
                filters.append({
                    "propertyName": "dealname",
                    "operator": "CONTAINS_TOKEN",
                    "value": dealname
                })
            if dealstage:
                filters.append({
                    "propertyName": "dealstage",
                    "operator": "EQ",
                    "value": dealstage
                })
            if pipeline:
                filters.append({
                    "propertyName": "pipeline",
                    "operator": "EQ",
                    "value": pipeline
                })
            if owner_id:
                filters.append({
                    "propertyName": "hubspot_owner_id",
                    "operator": "EQ",
                    "value": owner_id
                })

            search_request = {
                "filterGroups": [{"filters": filters}] if filters else [],
                "properties": [
                    "dealname", "amount", "dealstage", "pipeline",
                    "closedate", "hubspot_owner_id"
                ],
                "limit": min(limit, 100),
            }

            response = await asyncio.to_thread(
                self._client.crm.deals.search_api.do_search,
                public_object_search_request=search_request,
            )

            return [
                Deal(
                    id=r.id,
                    properties=r.properties,
                    created_at=r.created_at,
                    updated_at=r.updated_at,
                    archived=r.archived or False,
                )
                for r in response.results
            ]

        except Exception as e:
            self._handle_error(e, "search_deals")

    async def create_deal(self, data: DealCreate) -> Deal:
        """Create a new deal."""
        self._ensure_configured()
        await self._rate_limit()

        try:
            properties = data.properties.model_dump(exclude_none=True)
            create_input = {"properties": properties}

            if data.associations:
                create_input["associations"] = data.associations

            response = await asyncio.to_thread(
                self._client.crm.deals.basic_api.create,
                simple_public_object_input_for_create=create_input,
            )

            logger.info(f"Created deal: {response.id}")

            return Deal(
                id=response.id,
                properties=response.properties,
                created_at=response.created_at,
                updated_at=response.updated_at,
            )

        except Exception as e:
            self._handle_error(e, "create_deal")

    async def update_deal(self, deal_id: str, data: DealUpdate) -> Deal:
        """Update an existing deal."""
        self._ensure_configured()
        await self._rate_limit()

        try:
            properties = data.properties.model_dump(exclude_none=True)

            response = await asyncio.to_thread(
                self._client.crm.deals.basic_api.update,
                deal_id=deal_id,
                simple_public_object_input={"properties": properties},
            )

            logger.info(f"Updated deal: {deal_id}")

            return Deal(
                id=response.id,
                properties=response.properties,
                created_at=response.created_at,
                updated_at=response.updated_at,
            )

        except Exception as e:
            self._handle_error(e, "update_deal")

    # =========================================================================
    # TASKS
    # =========================================================================

    async def get_task(self, task_id: str) -> Task:
        """Get a task by ID."""
        self._ensure_configured()
        await self._rate_limit()

        try:
            response = await asyncio.to_thread(
                self._client.crm.objects.tasks.basic_api.get_by_id,
                task_id=task_id,
                properties=[
                    "hs_task_subject", "hs_task_body", "hs_task_status",
                    "hs_task_priority", "hs_task_type", "hs_timestamp",
                    "hubspot_owner_id"
                ],
            )

            return Task(
                id=response.id,
                properties=response.properties,
                created_at=response.created_at,
                updated_at=response.updated_at,
                archived=response.archived or False,
            )

        except Exception as e:
            self._handle_error(e, "get_task")

    async def create_task(self, data: TaskCreate) -> Task:
        """Create a new task."""
        self._ensure_configured()
        await self._rate_limit()

        try:
            properties = data.properties.model_dump(exclude_none=True, mode="json")
            create_input = {"properties": properties}

            if data.associations:
                create_input["associations"] = data.associations

            response = await asyncio.to_thread(
                self._client.crm.objects.tasks.basic_api.create,
                simple_public_object_input_for_create=create_input,
            )

            logger.info(f"Created task: {response.id}")

            return Task(
                id=response.id,
                properties=response.properties,
                created_at=response.created_at,
                updated_at=response.updated_at,
            )

        except Exception as e:
            self._handle_error(e, "create_task")

    async def update_task(self, task_id: str, data: TaskUpdate) -> Task:
        """Update an existing task."""
        self._ensure_configured()
        await self._rate_limit()

        try:
            properties = data.properties.model_dump(exclude_none=True, mode="json")

            response = await asyncio.to_thread(
                self._client.crm.objects.tasks.basic_api.update,
                task_id=task_id,
                simple_public_object_input={"properties": properties},
            )

            logger.info(f"Updated task: {task_id}")

            return Task(
                id=response.id,
                properties=response.properties,
                created_at=response.created_at,
                updated_at=response.updated_at,
            )

        except Exception as e:
            self._handle_error(e, "update_task")

    # =========================================================================
    # NOTES
    # =========================================================================

    async def create_note(self, data: NoteCreate) -> Note:
        """Create a new note."""
        self._ensure_configured()
        await self._rate_limit()

        try:
            properties = data.properties.model_dump(exclude_none=True)
            create_input = {"properties": properties}

            if data.associations:
                create_input["associations"] = data.associations

            response = await asyncio.to_thread(
                self._client.crm.objects.notes.basic_api.create,
                simple_public_object_input_for_create=create_input,
            )

            logger.info(f"Created note: {response.id}")

            return Note(
                id=response.id,
                properties=response.properties,
                created_at=response.created_at,
                updated_at=response.updated_at,
            )

        except Exception as e:
            self._handle_error(e, "create_note")

    # =========================================================================
    # ASSOCIATIONS
    # =========================================================================

    async def create_association(
        self,
        from_object_type: HubSpotObjectType,
        from_object_id: str,
        to_object_type: HubSpotObjectType,
        to_object_id: str,
        association_type: str | None = None,
    ) -> bool:
        """
        Create an association between two objects.

        Args:
            from_object_type: Source object type
            from_object_id: Source object ID
            to_object_type: Target object type
            to_object_id: Target object ID
            association_type: Optional association type ID

        Returns:
            True if successful
        """
        self._ensure_configured()
        await self._rate_limit()

        try:
            await asyncio.to_thread(
                self._client.crm.associations.v4.basic_api.create,
                object_type=from_object_type.value,
                object_id=from_object_id,
                to_object_type=to_object_type.value,
                to_object_id=to_object_id,
                association_spec=[{
                    "associationCategory": "HUBSPOT_DEFINED",
                    "associationTypeId": association_type or 1
                }]
            )

            logger.info(
                f"Created association: {from_object_type.value}/{from_object_id} "
                f"-> {to_object_type.value}/{to_object_id}"
            )

            return True

        except Exception as e:
            self._handle_error(e, "create_association")

    # =========================================================================
    # OWNERS
    # =========================================================================

    async def get_owners(self) -> list[Owner]:
        """Get all owners (users) in the portal."""
        self._ensure_configured()
        await self._rate_limit()

        try:
            response = await asyncio.to_thread(
                self._client.crm.owners.owners_api.get_page,
                limit=100,
            )

            return [
                Owner(
                    id=str(o.id),
                    email=o.email,
                    first_name=o.first_name,
                    last_name=o.last_name,
                    user_id=o.user_id,
                )
                for o in response.results
            ]

        except Exception as e:
            self._handle_error(e, "get_owners")

    async def get_owner_by_email(self, email: str) -> Owner | None:
        """Get an owner by email."""
        owners = await self.get_owners()
        for owner in owners:
            if owner.email and owner.email.lower() == email.lower():
                return owner
        return None

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    async def health_check(self) -> dict[str, Any]:
        """
        Check HubSpot connection health.

        Returns:
            Health status dict
        """
        if not self.is_configured:
            return {
                "status": "not_configured",
                "message": "HubSpot access token not set",
            }

        try:
            # Try to get owners as a simple health check
            owners = await self.get_owners()

            return {
                "status": "healthy",
                "portal_id": self._settings.hubspot_portal_id,
                "owners_count": len(owners),
            }

        except HubSpotAuthError:
            return {
                "status": "auth_error",
                "message": "Invalid access token",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
            }


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

def get_hubspot_service() -> HubSpotService:
    """
    Dependency injection for HubSpot Service.

    Usage:
        @router.get("/contacts/{id}")
        async def get_contact(
            id: str,
            hubspot: HubSpotService = Depends(get_hubspot_service)
        ):
            return await hubspot.get_contact(id)
    """
    return HubSpotService.get_instance()


@lru_cache
def get_hubspot_client() -> HubSpotService:
    """Get cached HubSpot client instance."""
    return HubSpotService.get_instance()


# =============================================================================
# MULTI-TENANT DEPENDENCY
# =============================================================================

async def get_hubspot_for_client(client_id: str) -> HubSpotService:
    """
    Get HubSpotService for a specific client (multi-tenant).

    Usage in FastAPI:
        @router.get("/contacts/{id}")
        async def get_contact(
            id: str,
            client: ClientContextDep,
        ):
            hubspot = await get_hubspot_for_client(client.client_id)
            return await hubspot.get_contact(id)
    """
    factory = HubSpotClientFactory.get_instance()
    return await factory.get_service(client_id)


def get_hubspot_factory() -> HubSpotClientFactory:
    """Get the HubSpot client factory singleton."""
    return HubSpotClientFactory.get_instance()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Exceptions
    "HubSpotError",
    "HubSpotRateLimitError",
    "HubSpotNotFoundError",
    "HubSpotValidationError",
    "HubSpotAuthError",
    # Multi-Tenant Factory
    "HubSpotClientFactory",
    "get_hubspot_factory",
    "get_hubspot_for_client",
    # Service
    "HubSpotService",
    # DI (Legacy)
    "get_hubspot_service",
    "get_hubspot_client",
]

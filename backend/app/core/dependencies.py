"""
Dependency Injection Setup
==========================
FastAPI dependency injection fonksiyonları.
Tüm endpoint'ler bu bağımlılıkları kullanarak servislerelere erişir.
"""

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated, Any, AsyncGenerator, Optional

from fastapi import Depends, Header, HTTPException, Request, status
from supabase import AsyncClient, create_async_client
import httpx

from backend.app.core.config import Settings, get_settings

# Lazy import to avoid circular dependency
# SupabaseService is imported inside functions that need it


# =============================================================================
# SETTINGS DEPENDENCY
# =============================================================================

def get_settings_dep() -> Settings:
    """
    Settings dependency for FastAPI endpoints.

    Usage:
        @router.get("/")
        async def endpoint(settings: Annotated[Settings, Depends(get_settings_dep)]):
            return {"debug": settings.debug}
    """
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


# =============================================================================
# SUPABASE CLIENT
# =============================================================================

class SupabaseClientManager:
    """
    Supabase async client manager.
    Manages connection lifecycle for the application.
    """

    _client: Optional[AsyncClient] = None
    _admin_client: Optional[AsyncClient] = None

    @classmethod
    async def get_client(cls, settings: Settings) -> AsyncClient:
        """Get or create Supabase client with anon key."""
        if cls._client is None:
            cls._client = await create_async_client(
                settings.supabase_url,
                settings.supabase_key
            )
        return cls._client

    @classmethod
    async def get_admin_client(cls, settings: Settings) -> AsyncClient:
        """Get or create Supabase client with service role key."""
        if cls._admin_client is None:
            if not settings.supabase_service_key:
                raise ValueError("SUPABASE_SERVICE_KEY is required for admin operations")
            cls._admin_client = await create_async_client(
                settings.supabase_url,
                settings.supabase_service_key
            )
        return cls._admin_client

    @classmethod
    async def close(cls) -> None:
        """Close all Supabase connections."""
        if cls._client:
            await cls._client.aclose()
            cls._client = None
        if cls._admin_client:
            await cls._admin_client.aclose()
            cls._admin_client = None


async def get_supabase(
    settings: SettingsDep
) -> AsyncGenerator[AsyncClient, None]:
    """
    Supabase client dependency.

    Usage:
        @router.get("/")
        async def endpoint(db: Annotated[AsyncClient, Depends(get_supabase)]):
            result = await db.table("users").select("*").execute()
    """
    client = await SupabaseClientManager.get_client(settings)
    yield client


async def get_supabase_admin(
    settings: SettingsDep
) -> AsyncGenerator[AsyncClient, None]:
    """
    Supabase admin client dependency (service role).
    Use for operations that bypass RLS.
    """
    client = await SupabaseClientManager.get_admin_client(settings)
    yield client


SupabaseDep = Annotated[AsyncClient, Depends(get_supabase)]
SupabaseAdminDep = Annotated[AsyncClient, Depends(get_supabase_admin)]


# NEW: SupabaseService dependency (preferred)
async def get_db():
    """
    Get SupabaseService instance (preferred over raw client).

    Usage:
        @router.get("/items")
        async def get_items(db: Annotated[SupabaseService, Depends(get_db)]):
            return await db.fetch_many("items")
    """
    # Lazy import to avoid circular dependency
    from backend.services.supabase_client import get_supabase as get_supabase_service
    return await get_supabase_service()


# Type alias - import SupabaseService only for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backend.services.supabase_client import SupabaseService
    SupabaseServiceDep = Annotated["SupabaseService", Depends(get_db)]
else:
    SupabaseServiceDep = Annotated[Any, Depends(get_db)]


# =============================================================================
# HTTP CLIENT
# =============================================================================

class HTTPClientManager:
    """HTTP client manager for external API calls."""

    _client: Optional[httpx.AsyncClient] = None

    @classmethod
    def get_client(cls) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if cls._client is None:
            cls._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                follow_redirects=True
            )
        return cls._client

    @classmethod
    async def close(cls) -> None:
        """Close HTTP client."""
        if cls._client:
            await cls._client.aclose()
            cls._client = None


async def get_http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """
    HTTP client dependency for external API calls.

    Usage:
        @router.get("/")
        async def endpoint(http: Annotated[httpx.AsyncClient, Depends(get_http_client)]):
            response = await http.get("https://api.example.com")
    """
    yield HTTPClientManager.get_client()


HTTPClientDep = Annotated[httpx.AsyncClient, Depends(get_http_client)]


# =============================================================================
# CLIENT CONTEXT (Authentication)
# =============================================================================

class ClientContext:
    """
    Authenticated client context.
    Contains client information extracted from request headers/token.
    """

    def __init__(
        self,
        client_id: str,
        client_name: str,
        hubspot_token: Optional[str] = None,
        salesforce_token: Optional[str] = None
    ):
        self.client_id = client_id
        self.client_name = client_name
        self.hubspot_token = hubspot_token
        self.salesforce_token = salesforce_token

    def __repr__(self) -> str:
        return f"ClientContext(client_id={self.client_id}, client_name={self.client_name})"


async def get_client_context(
    request: Request,
    settings: SettingsDep,
    x_client_id: Annotated[Optional[str], Header()] = None,
) -> ClientContext:
    """
    Extract and validate client context from request.

    For MVP, uses header-based client identification.
    Production should use JWT tokens with proper validation.

    Raises:
        HTTPException: If client identification fails
    """
    # MVP: Simple header-based identification
    # TODO: Replace with JWT token validation in production
    client_id = x_client_id

    if not client_id:
        # Development mode: allow default client
        if settings.is_development:
            return ClientContext(
                client_id="dev-client-001",
                client_name="Development Client",
                hubspot_token=settings.hubspot_access_token
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Client-ID header is required"
        )

    # TODO: Fetch client details from database
    # For now, return basic context
    return ClientContext(
        client_id=client_id,
        client_name=f"Client-{client_id}",
        hubspot_token=settings.hubspot_access_token
    )


ClientContextDep = Annotated[ClientContext, Depends(get_client_context)]


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """
    Simple in-memory rate limiter.
    Production should use Redis for distributed rate limiting.
    """

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    async def is_allowed(self, key: str) -> bool:
        """Check if request is allowed under rate limit."""
        import time

        now = time.time()
        window_start = now - self.window_seconds

        # Get existing requests for this key
        if key not in self._requests:
            self._requests[key] = []

        # Filter to only requests within window
        self._requests[key] = [
            ts for ts in self._requests[key] if ts > window_start
        ]

        # Check if under limit
        if len(self._requests[key]) >= self.max_requests:
            return False

        # Record this request
        self._requests[key].append(now)
        return True


# Global rate limiters
_api_rate_limiter: Optional[RateLimiter] = None
_hubspot_rate_limiter: Optional[RateLimiter] = None


def get_api_rate_limiter(settings: Settings) -> RateLimiter:
    """Get API rate limiter instance."""
    global _api_rate_limiter
    if _api_rate_limiter is None:
        _api_rate_limiter = RateLimiter(
            max_requests=settings.llm_rate_limit_per_minute,
            window_seconds=60
        )
    return _api_rate_limiter


def get_hubspot_rate_limiter(settings: Settings) -> RateLimiter:
    """Get HubSpot rate limiter instance."""
    global _hubspot_rate_limiter
    if _hubspot_rate_limiter is None:
        _hubspot_rate_limiter = RateLimiter(
            max_requests=settings.hubspot_rate_limit_per_second,
            window_seconds=1
        )
    return _hubspot_rate_limiter


async def check_rate_limit(
    request: Request,
    settings: SettingsDep,
    client: ClientContextDep
) -> None:
    """
    Rate limit check dependency.

    Usage:
        @router.post("/", dependencies=[Depends(check_rate_limit)])
        async def endpoint():
            pass
    """
    limiter = get_api_rate_limiter(settings)
    if not await limiter.is_allowed(client.client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )


RateLimitDep = Depends(check_rate_limit)


# =============================================================================
# LIFESPAN CONTEXT MANAGER
# =============================================================================

@asynccontextmanager
async def lifespan_manager(app):
    """
    Application lifespan manager.
    Handles startup and shutdown events.

    Usage in main.py:
        app = FastAPI(lifespan=lifespan_manager)
    """
    # Startup
    settings = get_settings()
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"Environment: {settings.environment}")
    print(f"Debug: {settings.debug}")

    # Initialize LangSmith tracing if configured
    if settings.langsmith_tracing and settings.langsmith_api_key:
        import os
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        print(f"LangSmith tracing enabled: {settings.langsmith_project}")

    yield

    # Shutdown
    print("Shutting down...")
    await SupabaseClientManager.close()
    await HTTPClientManager.close()
    print("Cleanup complete")


# =============================================================================
# DEPENDENCY EXPORTS
# =============================================================================

__all__ = [
    # Settings
    "get_settings_dep",
    "SettingsDep",
    # Supabase
    "get_supabase",
    "get_supabase_admin",
    "SupabaseDep",
    "SupabaseAdminDep",
    "SupabaseClientManager",
    # SupabaseService (preferred)
    "get_db",
    "SupabaseServiceDep",
    # HTTP
    "get_http_client",
    "HTTPClientDep",
    "HTTPClientManager",
    # Client Context
    "ClientContext",
    "get_client_context",
    "ClientContextDep",
    # Rate Limiting
    "RateLimiter",
    "check_rate_limit",
    "RateLimitDep",
    # Lifespan
    "lifespan_manager",
]

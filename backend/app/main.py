"""
CRM AI Orchestrator - FastAPI Application
==========================================
Main entry point for the API server.
Handles middleware setup, router registration, and application lifecycle.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from backend.app.core.config import get_settings
from backend.app.core.dependencies import (
    SupabaseClientManager,
    HTTPClientManager,
)
from backend.services.supabase_client import SupabaseService
from backend.app.api.v1 import api_router


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    version: str
    environment: str


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str
    detail: str | None = None
    path: str | None = None
    timestamp: str


# =============================================================================
# LIFESPAN MANAGER
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    settings = get_settings()

    # === STARTUP ===
    print("=" * 60)
    print(f"[START] Starting {settings.app_name} v{settings.app_version}")
    print(f"[ENV] Environment: {settings.environment}")
    print(f"[DEBUG] Debug Mode: {settings.debug}")
    print(f"[LOG] Log Level: {settings.log_level}")
    print("=" * 60)

    # Initialize LangSmith tracing if configured
    if settings.langsmith_tracing and settings.langsmith_api_key:
        import os
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        print(f"[OK] LangSmith tracing enabled: {settings.langsmith_project}")

    # Log active integrations
    if settings.hubspot_access_token:
        print("[OK] HubSpot integration configured")
    if settings.anthropic_api_key:
        print("[OK] Anthropic (Claude) configured")
    if settings.openai_api_key:
        print("[OK] OpenAI configured")
    if settings.tavily_api_key:
        print("[OK] Tavily search configured")

    print("=" * 60)
    print("[READY] Application started successfully")
    print("=" * 60)

    yield

    # === SHUTDOWN ===
    print("=" * 60)
    print("[STOP] Shutting down application...")

    # Cleanup connections
    await SupabaseClientManager.close()
    await HTTPClientManager.close()

    print("[OK] Cleanup complete")
    print("=" * 60)


# =============================================================================
# APPLICATION FACTORY
# =============================================================================

def create_application() -> FastAPI:
    """
    Application factory function.
    Creates and configures the FastAPI application.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Autonomous Growth & Sales Engine for HubSpot and Salesforce. "
            "Multi-agent system for market research, content generation, "
            "SEO analysis, and CRM automation."
        ),
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
    )

    # =========================================================================
    # CORS MIDDLEWARE
    # =========================================================================
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time"],
    )

    # =========================================================================
    # EXCEPTION HANDLERS
    # =========================================================================

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation errors."""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="Validation Error",
                detail=str(exc.errors()),
                path=str(request.url.path),
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """Handle all unhandled exceptions."""
        # Log the error (in production, send to monitoring service)
        print(f"[ERROR] Unhandled exception: {exc}")

        # Don't expose internal errors in production
        detail = str(exc) if settings.debug else "Internal server error"

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Internal Server Error",
                detail=detail,
                path=str(request.url.path),
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump(),
        )

    # =========================================================================
    # MIDDLEWARE - Request Timing
    # =========================================================================

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Add processing time to response headers."""
        import time
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        return response

    # =========================================================================
    # ROOT ENDPOINTS
    # =========================================================================

    @app.get(
        "/",
        tags=["Root"],
        summary="API Root",
        response_model=dict[str, Any]
    )
    async def root() -> dict[str, Any]:
        """
        API root endpoint.
        Returns basic API information and available endpoints.
        """
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "description": "CRM AI Orchestrator API",
            "docs": "/docs" if settings.debug else None,
            "health": "/health",
            "api": f"{settings.api_v1_prefix}",
        }

    @app.get(
        "/health",
        tags=["Health"],
        summary="Health Check",
        response_model=HealthResponse
    )
    async def health_check() -> HealthResponse:
        """
        Health check endpoint.
        Used by load balancers and container orchestrators.
        """
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(timezone.utc).isoformat(),
            version=settings.app_version,
            environment=settings.environment,
        )

    @app.get(
        "/health/ready",
        tags=["Health"],
        summary="Readiness Check",
        response_model=dict[str, Any]
    )
    async def readiness_check() -> dict[str, Any]:
        """
        Readiness check endpoint.
        Verifies all dependencies are available.
        """
        checks: dict[str, Any] = {}

        # Check Supabase connection using SupabaseService
        try:
            db = await SupabaseService.get_instance()
            health_result = await db.health_check()
            checks["supabase"] = health_result.get("connected", False)
            checks["supabase_status"] = health_result.get("status", "unknown")
        except Exception as e:
            checks["supabase"] = False
            checks["supabase_error"] = str(e)

        # Check LLM provider
        checks["llm_configured"] = bool(
            settings.anthropic_api_key or settings.openai_api_key
        )
        checks["llm_provider"] = settings.active_llm_provider

        # Check HubSpot
        checks["hubspot_configured"] = bool(settings.hubspot_access_token)

        # Overall readiness
        required_checks = [checks.get("supabase", False), checks.get("llm_configured", False)]
        all_ready = all(required_checks)

        return {
            "ready": all_ready,
            "checks": checks,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # =========================================================================
    # ROUTER REGISTRATION
    # =========================================================================

    # Register v1 API routes
    app.include_router(
        api_router,
        prefix=settings.api_v1_prefix,
    )

    return app


# =============================================================================
# APPLICATION INSTANCE
# =============================================================================

app = create_application()


# =============================================================================
# DEVELOPMENT SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )

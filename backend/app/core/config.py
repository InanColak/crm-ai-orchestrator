"""
Application Configuration
=========================
Pydantic Settings V2 ile merkezi konfigürasyon yönetimi.
Tüm environment variables bu modül üzerinden erişilir.
"""

from functools import lru_cache
from typing import Optional, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings - Environment Variables

    Tüm konfigürasyon değerleri .env dosyasından veya
    environment variables'dan okunur.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # APP CONFIG
    # =========================================================================
    app_name: str = Field(default="CRM AI Orchestrator", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode flag")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment"
    )

    # =========================================================================
    # API CONFIG
    # =========================================================================
    api_v1_prefix: str = Field(default="/api/v1", description="API version prefix")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins"
    )

    # =========================================================================
    # SUPABASE (Database + Vector DB + Storage)
    # =========================================================================
    supabase_url: str = Field(
        ...,
        description="Supabase project URL",
        examples=["https://xxx.supabase.co"]
    )
    supabase_key: str = Field(
        ...,
        description="Supabase anon/public key for client operations"
    )
    supabase_service_key: Optional[str] = Field(
        default=None,
        description="Supabase service role key for admin operations"
    )

    # =========================================================================
    # LLM PROVIDERS
    # =========================================================================
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key for Claude models"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (fallback provider)"
    )
    default_llm_provider: Literal["anthropic", "openai"] = Field(
        default="anthropic",
        description="Default LLM provider to use"
    )
    default_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default model identifier"
    )

    # =========================================================================
    # SECURITY (Encryption)
    # =========================================================================
    crm_encryption_key: Optional[str] = Field(
        default=None,
        description="AES-256 encryption key for CRM tokens (32 bytes, base64 encoded)"
    )
    encryption_key_version: int = Field(
        default=1,
        description="Current encryption key version for key rotation",
        ge=1
    )

    # =========================================================================
    # CRM INTEGRATIONS - Legacy (for backward compatibility)
    # =========================================================================
    hubspot_access_token: Optional[str] = Field(
        default=None,
        description="DEPRECATED: HubSpot private app access token (use OAuth instead)"
    )
    hubspot_portal_id: Optional[str] = Field(
        default=None,
        description="HubSpot portal ID"
    )

    # =========================================================================
    # HUBSPOT OAUTH (Multi-tenant)
    # =========================================================================
    hubspot_client_id: Optional[str] = Field(
        default=None,
        description="HubSpot OAuth app client ID"
    )
    hubspot_client_secret: Optional[str] = Field(
        default=None,
        description="HubSpot OAuth app client secret"
    )
    hubspot_oauth_redirect_uri: str = Field(
        default="http://localhost:8000/api/v1/oauth/hubspot/callback",
        description="HubSpot OAuth callback URL"
    )
    hubspot_oauth_scopes: list[str] = Field(
        default=[
            "crm.objects.contacts.read",
            "crm.objects.contacts.write",
            "crm.objects.deals.read",
            "crm.objects.deals.write",
            "crm.objects.companies.read",
            "crm.objects.owners.read",
            "sales-email-read",
            "tickets"
        ],
        description="HubSpot OAuth scopes to request"
    )

    # =========================================================================
    # SALESFORCE OAUTH (Multi-tenant)
    # =========================================================================
    salesforce_client_id: Optional[str] = Field(
        default=None,
        description="Salesforce connected app client ID"
    )
    salesforce_client_secret: Optional[str] = Field(
        default=None,
        description="Salesforce connected app client secret"
    )
    salesforce_instance_url: Optional[str] = Field(
        default=None,
        description="Salesforce instance URL"
    )
    salesforce_oauth_redirect_uri: str = Field(
        default="http://localhost:8000/api/v1/oauth/salesforce/callback",
        description="Salesforce OAuth callback URL"
    )

    # =========================================================================
    # EXTERNAL SERVICES
    # =========================================================================
    tavily_api_key: Optional[str] = Field(
        default=None,
        description="Tavily API key for web search"
    )
    tavily_rate_limit_per_minute: int = Field(
        default=60,
        description="Tavily API calls per minute limit",
        ge=1,
        le=1000
    )
    tavily_cache_ttl_seconds: int = Field(
        default=300,
        description="Tavily search cache TTL in seconds",
        ge=60,
        le=3600
    )

    # =========================================================================
    # OBSERVABILITY (LangSmith)
    # =========================================================================
    langsmith_api_key: Optional[str] = Field(
        default=None,
        description="LangSmith API key for tracing"
    )
    langsmith_project: str = Field(
        default="crm-ai-orchestrator",
        description="LangSmith project name"
    )
    langsmith_tracing: bool = Field(
        default=False,
        description="Enable LangSmith tracing"
    )

    # =========================================================================
    # WORKFLOW CONFIG
    # =========================================================================
    workflow_timeout_seconds: int = Field(
        default=300,
        description="Maximum workflow execution time in seconds",
        ge=30,
        le=3600
    )
    max_agent_iterations: int = Field(
        default=10,
        description="Maximum iterations per agent",
        ge=1,
        le=50
    )
    approval_timeout_hours: int = Field(
        default=48,
        description="Hours before approval requests expire",
        ge=1,
        le=168
    )

    # =========================================================================
    # RATE LIMITING
    # =========================================================================
    hubspot_rate_limit_per_second: int = Field(
        default=10,
        description="HubSpot API calls per second limit",
        ge=1,
        le=100
    )
    llm_rate_limit_per_minute: int = Field(
        default=60,
        description="LLM API calls per minute limit",
        ge=1,
        le=1000
    )

    # =========================================================================
    # RAG / DOCUMENT PROCESSING (Phase 4.4)
    # =========================================================================
    # Document Upload Limits
    document_max_file_size_mb: int = Field(
        default=10,
        description="Maximum file size in MB for document upload",
        ge=1,
        le=50
    )
    document_max_files_per_client: int = Field(
        default=50,
        description="Maximum number of documents per client",
        ge=1,
        le=500
    )
    document_allowed_types: list[str] = Field(
        default=["pdf", "docx", "txt", "md"],
        description="Allowed document file types"
    )

    # Embedding Configuration
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model to use"
    )
    embedding_dimensions: int = Field(
        default=1536,
        description="Embedding vector dimensions",
        ge=256,
        le=3072
    )
    embedding_max_tokens_per_document: int = Field(
        default=100000,
        description="Maximum tokens per document for embedding",
        ge=1000,
        le=1000000
    )
    embedding_daily_token_limit: int = Field(
        default=500000,
        description="Daily embedding token limit per client",
        ge=10000,
        le=10000000
    )

    # Chunking Configuration
    chunk_size: int = Field(
        default=1000,
        description="Target chunk size in tokens",
        ge=100,
        le=4000
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between chunks in tokens",
        ge=0,
        le=1000
    )

    # Vector Search Configuration
    vector_search_daily_limit: int = Field(
        default=1000,
        description="Daily vector search queries per client",
        ge=100,
        le=100000
    )
    vector_search_top_k_default: int = Field(
        default=5,
        description="Default number of results for vector search",
        ge=1,
        le=20
    )
    vector_search_top_k_max: int = Field(
        default=10,
        description="Maximum results for vector search",
        ge=1,
        le=50
    )
    vector_similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity score for vector search",
        ge=0.0,
        le=1.0
    )

    # =========================================================================
    # VALIDATORS
    # =========================================================================

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("hubspot_oauth_scopes", mode="before")
    @classmethod
    def parse_hubspot_scopes(cls, v: str | list[str]) -> list[str]:
        """Parse HubSpot OAuth scopes from comma-separated string or list."""
        if isinstance(v, str):
            return [scope.strip() for scope in v.split(",") if scope.strip()]
        return v

    @model_validator(mode="after")
    def validate_encryption_key(self) -> "Settings":
        """Validate encryption key format and presence in production."""
        if self.crm_encryption_key:
            import base64
            try:
                key_bytes = base64.b64decode(self.crm_encryption_key)
                if len(key_bytes) != 32:
                    raise ValueError(
                        f"CRM_ENCRYPTION_KEY must be 32 bytes (256 bits), got {len(key_bytes)} bytes"
                    )
            except Exception as e:
                raise ValueError(f"Invalid CRM_ENCRYPTION_KEY format: {e}")
        elif self.environment == "production":
            raise ValueError("CRM_ENCRYPTION_KEY is required in production")
        return self

    @model_validator(mode="after")
    def validate_llm_config(self) -> "Settings":
        """Ensure at least one LLM provider is configured."""
        if not self.anthropic_api_key and not self.openai_api_key:
            if self.environment == "production":
                raise ValueError(
                    "At least one LLM provider (ANTHROPIC_API_KEY or OPENAI_API_KEY) "
                    "must be configured in production"
                )
        return self

    @model_validator(mode="after")
    def configure_langsmith(self) -> "Settings":
        """Auto-enable LangSmith tracing if API key is provided."""
        if self.langsmith_api_key and not self.langsmith_tracing:
            self.langsmith_tracing = True
        return self

    # =========================================================================
    # COMPUTED PROPERTIES
    # =========================================================================

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    @property
    def database_url(self) -> str:
        """Construct Supabase PostgreSQL connection URL."""
        # Supabase URL format: https://xxx.supabase.co
        # PostgreSQL URL: postgresql://postgres:[password]@db.xxx.supabase.co:5432/postgres
        project_ref = self.supabase_url.replace("https://", "").replace(".supabase.co", "")
        return f"postgresql://postgres.{project_ref}:5432/postgres"

    @property
    def active_llm_provider(self) -> str:
        """Get the active LLM provider based on available keys."""
        if self.default_llm_provider == "anthropic" and self.anthropic_api_key:
            return "anthropic"
        if self.default_llm_provider == "openai" and self.openai_api_key:
            return "openai"
        # Fallback logic
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_api_key:
            return "openai"
        return "none"


@lru_cache
def get_settings() -> Settings:
    """
    Singleton Settings Instance

    LRU cache ensures settings are loaded once and reused.
    Call get_settings.cache_clear() to reload if needed.

    Returns:
        Settings: Application settings instance

    Example:
        >>> from backend.app.core.config import get_settings
        >>> settings = get_settings()
        >>> print(settings.supabase_url)
    """
    return Settings()


# Convenience alias for direct import
settings = get_settings()

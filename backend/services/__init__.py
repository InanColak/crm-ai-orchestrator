"""
Services Module
===============
Business logic and external service integrations.
"""

from backend.services.supabase_client import (
    SupabaseService,
    get_supabase,
    supabase_session,
    DatabaseError,
    ConnectionError,
    QueryError,
    NotFoundError,
)

from backend.services.llm_service import (
    # Enums
    LLMProvider,
    LLMModel,
    # Models
    TokenUsage,
    LLMResponse,
    # Errors
    LLMError,
    RateLimitError,
    TokenLimitError,
    # Service
    LLMService,
    # DI
    get_llm_service,
    get_default_client,
)

from backend.services.hubspot_service import (
    # Errors
    HubSpotError,
    HubSpotRateLimitError,
    HubSpotNotFoundError,
    HubSpotValidationError,
    HubSpotAuthError,
    # Multi-Tenant Factory
    HubSpotClientFactory,
    get_hubspot_factory,
    get_hubspot_for_client,
    # Service
    HubSpotService,
    # DI (Legacy)
    get_hubspot_service,
    get_hubspot_client,
)

from backend.services.encryption_service import (
    # Errors
    EncryptionError,
    DecryptionError,
    KeyNotConfiguredError,
    # Classes
    EncryptedData,
    EncryptionService,
    # DI
    get_encryption_service,
    # Utilities
    generate_encryption_key,
    encrypt_token,
    decrypt_token,
)

from backend.services.oauth_service import (
    # Enums
    CRMProvider,
    ConnectionStatus,
    # Exceptions
    OAuthError,
    OAuthStateError,
    OAuthTokenError,
    OAuthNotConnectedError,
    # Models
    OAuthTokens,
    CRMConnection,
    # Service
    OAuthService,
    # DI
    get_oauth_service,
)

from backend.services.approval_service import (
    # Exceptions
    ApprovalError,
    ApprovalNotFoundError,
    ApprovalAlreadyProcessedError,
    ApprovalExpiredError,
    ApprovalExecutionError,
    # Service
    ApprovalService,
    # DI
    get_approval_service,
)

from backend.services.workflow_service import (
    # Exceptions
    WorkflowError,
    WorkflowNotFoundError,
    WorkflowAlreadyExistsError,
    WorkflowInvalidStateError,
    WorkflowExecutionError,
    # Service
    WorkflowService,
    # DI
    get_workflow_service,
)

from backend.services.tavily_service import (
    # Enums
    SearchDepth,
    SearchTopic,
    # Models
    SearchResult,
    ImageResult,
    SearchResponse,
    CompanyResearchResult,
    NewsSearchResult,
    # Errors
    TavilyError,
    RateLimitError as TavilyRateLimitError,
    AuthenticationError as TavilyAuthError,
    QuotaExceededError as TavilyQuotaError,
    # Service
    TavilyService,
    # DI
    get_tavily_service,
)

__all__ = [
    # Supabase
    "SupabaseService",
    "get_supabase",
    "supabase_session",
    "DatabaseError",
    "ConnectionError",
    "QueryError",
    "NotFoundError",
    # LLM Service
    "LLMProvider",
    "LLMModel",
    "TokenUsage",
    "LLMResponse",
    "LLMError",
    "RateLimitError",
    "TokenLimitError",
    "LLMService",
    "get_llm_service",
    "get_default_client",
    # HubSpot Service
    "HubSpotError",
    "HubSpotRateLimitError",
    "HubSpotNotFoundError",
    "HubSpotValidationError",
    "HubSpotAuthError",
    "HubSpotClientFactory",
    "get_hubspot_factory",
    "get_hubspot_for_client",
    "HubSpotService",
    "get_hubspot_service",
    "get_hubspot_client",
    # Encryption Service
    "EncryptionError",
    "DecryptionError",
    "KeyNotConfiguredError",
    "EncryptedData",
    "EncryptionService",
    "get_encryption_service",
    "generate_encryption_key",
    "encrypt_token",
    "decrypt_token",
    # OAuth Service
    "CRMProvider",
    "ConnectionStatus",
    "OAuthError",
    "OAuthStateError",
    "OAuthTokenError",
    "OAuthNotConnectedError",
    "OAuthTokens",
    "CRMConnection",
    "OAuthService",
    "get_oauth_service",
    # Approval Service
    "ApprovalError",
    "ApprovalNotFoundError",
    "ApprovalAlreadyProcessedError",
    "ApprovalExpiredError",
    "ApprovalExecutionError",
    "ApprovalService",
    "get_approval_service",
    # Workflow Service
    "WorkflowError",
    "WorkflowNotFoundError",
    "WorkflowAlreadyExistsError",
    "WorkflowInvalidStateError",
    "WorkflowExecutionError",
    "WorkflowService",
    "get_workflow_service",
    # Tavily Service (Phase 4.1)
    "SearchDepth",
    "SearchTopic",
    "SearchResult",
    "ImageResult",
    "SearchResponse",
    "CompanyResearchResult",
    "NewsSearchResult",
    "TavilyError",
    "TavilyRateLimitError",
    "TavilyAuthError",
    "TavilyQuotaError",
    "TavilyService",
    "get_tavily_service",
]

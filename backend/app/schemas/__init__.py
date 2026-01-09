"""
Pydantic Schemas Module
=======================
Data validation schemas for API requests/responses and database models.
"""

from backend.app.schemas.workflow import (
    # Workflow Enums (from new module)
    WorkflowType as WorkflowTypeV2,
    WorkflowStatus as WorkflowStatusV2,
    WorkflowPriority,
    WorkflowEventType,
    # Input Payloads
    MeetingAnalysisInput,
    LeadResearchInput,
    ContentGenerationInput,
    IntelligenceInput,
    # Requests
    WorkflowTriggerRequest,
    WorkflowResumeRequest,
    WorkflowCancelRequest,
    # Responses
    WorkflowStepSummary,
    WorkflowResponse as WorkflowResponseV2,
    WorkflowDetail as WorkflowDetailV2,
    WorkflowSummary,
    WorkflowListResponse,
    WorkflowStats,
    # Events
    WorkflowEvent,
)

from backend.app.schemas.meeting_notes import (
    # Enums
    MeetingInputSource,
    MeetingSentiment,
    ActionItemPriority,
    # Input Schemas
    NormalizedMeetingInput,
    MeetingAnalysisRequest,
    # Output Schemas
    ActionItem,
    KeyDecision,
    DealStageRecommendation,
    MeetingAnalysis,
    # Response Schemas
    MeetingAnalysisResponse,
    MeetingNotesSummary,
    MeetingInputAdapterResult,
)

from backend.app.schemas.tasks import (
    # Enums
    TaskPriority,
    TaskStatus,
    TaskType,
    AssociationType,
    # Input Schemas
    ActionItemInput,
    TaskExtractionInput,
    # Output Schemas
    TaskAssociation,
    ExtractedTask,
    TaskExtractionResult,
    # CRM Payload Schemas
    HubSpotTaskPayload,
    TaskApprovalPayload,
    # API Schemas
    TaskExtractionRequest,
    TaskExtractionResponse,
)

from backend.app.schemas.crm_updates import (
    # Enums
    CRMOperationType,
    OperationRiskLevel,
    OperationStatus,
    # Payload Schemas
    TaskOperationPayload,
    DealUpdatePayload,
    NotePayload,
    ActivityPayload,
    # Operation Schemas
    CRMUpdateOperation,
    CRMUpdateOperationResult,
    # API Schemas
    CRMUpdateRequest,
    CRMUpdateResponse,
    # Execution Schemas
    OperationExecutionResult,
    BatchExecutionResult,
)

from backend.app.schemas.lead_research import (
    # Enums
    LeadSource,
    ResearchDepth,
    EnrichmentConfidence,
    LeadQualificationScore,
    # Input
    LeadResearchInput,
    LeadResearchRequest,
    # Company Data
    CompanyOverview,
    FundingInfo,
    KeyPerson,
    TechnologyStack,
    NewsItem,
    CompetitorInfo,
    SocialPresence,
    # Output
    LeadResearchResult,
    LeadEnrichmentPayload,
    LeadResearchResponse,
)

from backend.app.schemas.email import (
    # Enums
    EmailType,
    EmailTone,
    EmailPriority,
    EmailStatus,
    DeliveryProvider,
    # Input
    EmailRecipient,
    LeadContext,
    MeetingContext,
    PreviousEmailContext,
    EmailCopilotInput,
    EmailCopilotRequest,
    # Output - Draft
    EmailDraft,
    EmailGenerationResult,
    # Output - Delivery
    EmailDeliveryPayload,
    EmailDeliveryResult,
    # Approval
    EmailApprovalPayload,
    EmailApprovalDecision,
    # Context (RAG-ready)
    BrandvoiceContext,
    EmailContext,
    # API
    EmailCopilotResponse,
)

from backend.app.schemas.documents import (
    # Upload
    DocumentUploadResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentDeleteResponse,
    # Search
    SearchRequest,
    SearchResultItem,
    SearchResponse,
    # Usage
    UsageInfo,
    UsageSummaryResponse,
    # Error
    DocumentErrorResponse,
)

from backend.app.schemas.base import (
    # Enums
    WorkflowStatus,
    WorkflowType,
    ApprovalStatus,
    ApprovalType,
    ApprovalPriority,
    DocumentType,
    ProcessingStatus,
    CRMType,
    CRMOperation,
    AgentLogStatus,
    AgentSquad,
    # Base
    BaseSchema,
    TimestampMixin,
    IDMixin,
    # Client
    ClientBase,
    ClientCreate,
    ClientUpdate,
    ClientResponse,
    ClientInDB,
    # Workflow
    WorkflowBase,
    WorkflowCreate,
    WorkflowUpdate,
    WorkflowResponse,
    WorkflowDetail,
    # Approval
    ApprovalBase,
    ApprovalCreate,
    ApprovalDecision,
    ApprovalResponse,
    # Document
    DocumentBase,
    DocumentCreate,
    DocumentUpdate,
    DocumentResponse,
    # Document Chunk
    DocumentChunkBase,
    DocumentChunkCreate,
    DocumentChunkResponse,
    VectorSearchResult,
    # Agent Log
    AgentLogBase,
    AgentLogCreate,
    AgentLogResponse,
    # CRM Sync Log
    CRMSyncLogCreate,
    CRMSyncLogResponse,
    # Pagination & Response
    PaginationParams,
    PaginatedResponse,
    APIResponse,
    ErrorResponse,
)

__all__ = [
    # Enums
    "WorkflowStatus",
    "WorkflowType",
    "ApprovalStatus",
    "ApprovalType",
    "ApprovalPriority",
    "DocumentType",
    "ProcessingStatus",
    "CRMType",
    "CRMOperation",
    "AgentLogStatus",
    "AgentSquad",
    # Base
    "BaseSchema",
    "TimestampMixin",
    "IDMixin",
    # Client
    "ClientBase",
    "ClientCreate",
    "ClientUpdate",
    "ClientResponse",
    "ClientInDB",
    # Workflow
    "WorkflowBase",
    "WorkflowCreate",
    "WorkflowUpdate",
    "WorkflowResponse",
    "WorkflowDetail",
    # Approval
    "ApprovalBase",
    "ApprovalCreate",
    "ApprovalDecision",
    "ApprovalResponse",
    # Document
    "DocumentBase",
    "DocumentCreate",
    "DocumentUpdate",
    "DocumentResponse",
    # Document Chunk
    "DocumentChunkBase",
    "DocumentChunkCreate",
    "DocumentChunkResponse",
    "VectorSearchResult",
    # Agent Log
    "AgentLogBase",
    "AgentLogCreate",
    "AgentLogResponse",
    # CRM Sync Log
    "CRMSyncLogCreate",
    "CRMSyncLogResponse",
    # Pagination & Response
    "PaginationParams",
    "PaginatedResponse",
    "APIResponse",
    "ErrorResponse",
    # Workflow V2 (new module)
    "WorkflowTypeV2",
    "WorkflowStatusV2",
    "WorkflowPriority",
    "WorkflowEventType",
    "MeetingAnalysisInput",
    "LeadResearchInput",
    "ContentGenerationInput",
    "IntelligenceInput",
    "WorkflowTriggerRequest",
    "WorkflowResumeRequest",
    "WorkflowCancelRequest",
    "WorkflowStepSummary",
    "WorkflowResponseV2",
    "WorkflowDetailV2",
    "WorkflowSummary",
    "WorkflowListResponse",
    "WorkflowStats",
    "WorkflowEvent",
    # Meeting Notes Schemas
    "MeetingInputSource",
    "MeetingSentiment",
    "ActionItemPriority",
    "NormalizedMeetingInput",
    "MeetingAnalysisRequest",
    "ActionItem",
    "KeyDecision",
    "DealStageRecommendation",
    "MeetingAnalysis",
    "MeetingAnalysisResponse",
    "MeetingNotesSummary",
    "MeetingInputAdapterResult",
    # Task Extraction Schemas (Phase 3.2)
    "TaskPriority",
    "TaskStatus",
    "TaskType",
    "AssociationType",
    "ActionItemInput",
    "TaskExtractionInput",
    "TaskAssociation",
    "ExtractedTask",
    "TaskExtractionResult",
    "HubSpotTaskPayload",
    "TaskApprovalPayload",
    "TaskExtractionRequest",
    "TaskExtractionResponse",
    # CRM Update Schemas (Phase 3.3)
    "CRMOperationType",
    "OperationRiskLevel",
    "OperationStatus",
    "TaskOperationPayload",
    "DealUpdatePayload",
    "NotePayload",
    "ActivityPayload",
    "CRMUpdateOperation",
    "CRMUpdateOperationResult",
    "CRMUpdateRequest",
    "CRMUpdateResponse",
    "OperationExecutionResult",
    "BatchExecutionResult",
    # Lead Research Schemas (Phase 4.2)
    "LeadSource",
    "ResearchDepth",
    "EnrichmentConfidence",
    "LeadQualificationScore",
    "LeadResearchInput",
    "LeadResearchRequest",
    "CompanyOverview",
    "FundingInfo",
    "KeyPerson",
    "TechnologyStack",
    "NewsItem",
    "CompetitorInfo",
    "SocialPresence",
    "LeadResearchResult",
    "LeadEnrichmentPayload",
    "LeadResearchResponse",
    # Email Copilot Schemas (Phase 4.3)
    "EmailType",
    "EmailTone",
    "EmailPriority",
    "EmailStatus",
    "DeliveryProvider",
    "EmailRecipient",
    "LeadContext",
    "MeetingContext",
    "PreviousEmailContext",
    "EmailCopilotInput",
    "EmailCopilotRequest",
    "EmailDraft",
    "EmailGenerationResult",
    "EmailDeliveryPayload",
    "EmailDeliveryResult",
    "EmailApprovalPayload",
    "EmailApprovalDecision",
    "BrandvoiceContext",
    "EmailContext",
    "EmailCopilotResponse",
    # Document / RAG Schemas (Phase 4.4)
    "DocumentUploadResponse",
    "DocumentInfo",
    "DocumentListResponse",
    "DocumentDeleteResponse",
    "SearchRequest",
    "SearchResultItem",
    "SearchResponse",
    "UsageInfo",
    "UsageSummaryResponse",
    "DocumentErrorResponse",
]

"""
Base Pydantic Schemas
=====================
Core schema definitions for database models and common patterns.
All schemas use Pydantic V2 with strict validation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Import canonical enums from workflow.py (single source of truth)
from backend.app.schemas.workflow import WorkflowStatus, WorkflowType


# =============================================================================
# ENUMS
# =============================================================================


class ApprovalStatus(str, Enum):
    """Approval decision status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ApprovalType(str, Enum):
    """Types of actions requiring approval."""
    CRM_UPDATE = "crm_update"
    CRM_CREATE = "crm_create"
    CONTENT_PUBLISH = "content_publish"
    EMAIL_SEND = "email_send"
    TASK_CREATE = "task_create"
    MEETING_SCHEDULE = "meeting_schedule"


class ApprovalPriority(str, Enum):
    """Priority levels for approvals."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class DocumentType(str, Enum):
    """Types of customer documents."""
    BRANDVOICE = "brandvoice"
    PRODUCT_CATALOG = "product_catalog"
    COMPANY_INFO = "company_info"
    COMPETITOR_INFO = "competitor_info"
    TEMPLATE = "template"
    OTHER = "other"


class ProcessingStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class CRMType(str, Enum):
    """Supported CRM platforms."""
    HUBSPOT = "hubspot"
    SALESFORCE = "salesforce"


class CRMOperation(str, Enum):
    """CRM operation types."""
    READ = "read"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class AgentLogStatus(str, Enum):
    """Agent execution status."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


class AgentSquad(str, Enum):
    """Agent squad classification."""
    SALES_OPS = "sales_ops"
    INTELLIGENCE = "intelligence"
    CONTENT = "content"


# =============================================================================
# BASE SCHEMAS
# =============================================================================

class BaseSchema(BaseModel):
    """
    Base schema with common configuration.

    All schemas inherit from this base for consistent behavior.
    """
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None


class IDMixin(BaseModel):
    """Mixin for ID field."""
    id: UUID


# =============================================================================
# CLIENT SCHEMAS
# =============================================================================

class ClientBase(BaseSchema):
    """Base client fields."""
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-z0-9-]+$")
    industry: str | None = Field(None, max_length=100)
    hubspot_portal_id: str | None = Field(None, max_length=50)
    salesforce_org_id: str | None = Field(None, max_length=50)
    settings: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class ClientCreate(ClientBase):
    """Schema for creating a client."""
    hubspot_access_token: str | None = Field(None, description="HubSpot access token")
    salesforce_credentials: dict[str, Any] | None = None


class ClientUpdate(BaseSchema):
    """Schema for updating a client."""
    name: str | None = Field(None, min_length=1, max_length=255)
    industry: str | None = Field(None, max_length=100)
    settings: dict[str, Any] | None = None
    is_active: bool | None = None


class ClientResponse(ClientBase, IDMixin, TimestampMixin):
    """Schema for client response (excludes sensitive data)."""
    pass


class ClientInDB(ClientBase, IDMixin, TimestampMixin):
    """Schema for client in database (internal use)."""
    hubspot_access_token: str | None = None
    salesforce_credentials: dict[str, Any] | None = None


# =============================================================================
# WORKFLOW SCHEMAS
# =============================================================================

class WorkflowBase(BaseSchema):
    """Base workflow fields."""
    workflow_type: WorkflowType
    name: str | None = Field(None, max_length=255)
    description: str | None = None


class WorkflowCreate(WorkflowBase):
    """Schema for creating a workflow."""
    client_id: UUID
    state: dict[str, Any] = Field(default_factory=dict)
    input_summary: str | None = None


class WorkflowUpdate(BaseSchema):
    """Schema for updating a workflow."""
    status: WorkflowStatus | None = None
    state: dict[str, Any] | None = None
    output_summary: str | None = None
    error_message: str | None = None
    total_tokens_used: int | None = None
    total_cost_usd: float | None = None


class WorkflowResponse(WorkflowBase, IDMixin, TimestampMixin):
    """Schema for workflow response."""
    client_id: UUID
    status: WorkflowStatus = WorkflowStatus.PENDING
    input_summary: str | None = None
    output_summary: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    total_tokens_used: int = 0


class WorkflowDetail(WorkflowResponse):
    """Detailed workflow response including state."""
    state: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    checkpointer_thread_id: str | None = None


# =============================================================================
# APPROVAL SCHEMAS
# =============================================================================

class ApprovalBase(BaseSchema):
    """Base approval fields."""
    approval_type: ApprovalType
    title: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    priority: ApprovalPriority = ApprovalPriority.NORMAL


class ApprovalCreate(ApprovalBase):
    """Schema for creating an approval."""
    workflow_id: UUID
    client_id: UUID
    payload: dict[str, Any]
    diff_before: dict[str, Any] | None = None
    diff_after: dict[str, Any] | None = None
    expires_at: datetime | None = None


class ApprovalDecision(BaseSchema):
    """Schema for approval decision."""
    status: ApprovalStatus = Field(..., description="Must be 'approved' or 'rejected'")
    reviewed_by: str = Field(..., min_length=1, max_length=255)
    rejection_reason: str | None = Field(None, max_length=1000)

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: ApprovalStatus) -> ApprovalStatus:
        if v not in (ApprovalStatus.APPROVED, ApprovalStatus.REJECTED):
            raise ValueError("Status must be 'approved' or 'rejected'")
        return v


class ApprovalResponse(ApprovalBase, IDMixin, TimestampMixin):
    """Schema for approval response."""
    workflow_id: UUID
    client_id: UUID
    status: ApprovalStatus = ApprovalStatus.PENDING
    payload: dict[str, Any]
    diff_before: dict[str, Any] | None = None
    diff_after: dict[str, Any] | None = None
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    rejection_reason: str | None = None
    expires_at: datetime | None = None


# =============================================================================
# DOCUMENT SCHEMAS
# =============================================================================

class DocumentBase(BaseSchema):
    """Base document fields."""
    doc_type: DocumentType
    title: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentCreate(DocumentBase):
    """Schema for creating a document."""
    client_id: UUID
    content: str | None = None
    storage_path: str | None = None
    file_name: str | None = None
    file_size: int | None = None
    mime_type: str | None = None


class DocumentUpdate(BaseSchema):
    """Schema for updating a document."""
    title: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    metadata: dict[str, Any] | None = None
    processing_status: ProcessingStatus | None = None
    processing_error: str | None = None
    chunk_count: int | None = None


class DocumentResponse(DocumentBase, IDMixin, TimestampMixin):
    """Schema for document response."""
    client_id: UUID
    content: str | None = None
    storage_path: str | None = None
    file_name: str | None = None
    file_size: int | None = None
    mime_type: str | None = None
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    chunk_count: int = 0


# =============================================================================
# DOCUMENT CHUNK SCHEMAS
# =============================================================================

class DocumentChunkBase(BaseSchema):
    """Base document chunk fields."""
    chunk_index: int = Field(..., ge=0)
    content: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_count: int | None = None


class DocumentChunkCreate(DocumentChunkBase):
    """Schema for creating a document chunk."""
    document_id: UUID
    client_id: UUID
    embedding: list[float] | None = Field(None, description="1536-dimensional vector")


class DocumentChunkResponse(DocumentChunkBase, IDMixin):
    """Schema for document chunk response."""
    document_id: UUID
    created_at: datetime


class VectorSearchResult(BaseSchema):
    """Schema for vector search results."""
    id: UUID
    document_id: UUID
    content: str
    metadata: dict[str, Any]
    similarity: float = Field(..., ge=0, le=1)


# =============================================================================
# AGENT LOG SCHEMAS
# =============================================================================

class AgentLogBase(BaseSchema):
    """Base agent log fields."""
    agent_name: str = Field(..., max_length=100)
    agent_type: str | None = Field(None, max_length=50)
    squad: AgentSquad | None = None


class AgentLogCreate(AgentLogBase):
    """Schema for creating an agent log."""
    workflow_id: UUID
    client_id: UUID
    input_summary: str | None = None
    output_summary: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_ms: int | None = None
    status: AgentLogStatus = AgentLogStatus.SUCCESS
    error_message: str | None = None
    error_traceback: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class AgentLogResponse(AgentLogBase, IDMixin):
    """Schema for agent log response."""
    workflow_id: UUID
    input_summary: str | None = None
    output_summary: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    total_tokens: int = 0
    duration_ms: int | None = None
    status: AgentLogStatus
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime


# =============================================================================
# CRM SYNC LOG SCHEMAS
# =============================================================================

class CRMSyncLogCreate(BaseSchema):
    """Schema for creating a CRM sync log."""
    workflow_id: UUID | None = None
    client_id: UUID
    crm_type: CRMType
    operation: CRMOperation
    object_type: str = Field(..., max_length=50)
    object_id: str | None = Field(None, max_length=100)
    request_payload: dict[str, Any] | None = None
    response_payload: dict[str, Any] | None = None
    status: str = Field(..., pattern=r"^(success|error|rate_limited)$")
    error_message: str | None = None
    retry_count: int = 0


class CRMSyncLogResponse(BaseSchema, IDMixin):
    """Schema for CRM sync log response."""
    workflow_id: UUID | None = None
    client_id: UUID
    crm_type: CRMType
    operation: CRMOperation
    object_type: str
    object_id: str | None = None
    status: str
    error_message: str | None = None
    created_at: datetime


# =============================================================================
# PAGINATION & RESPONSE WRAPPERS
# =============================================================================

T = TypeVar("T", bound=BaseModel)


class PaginationParams(BaseSchema):
    """Pagination parameters."""
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")

    @property
    def offset(self) -> int:
        """Calculate offset for database query."""
        return (self.page - 1) * self.page_size


class PaginatedResponse(BaseSchema, Generic[T]):
    """Generic paginated response wrapper."""
    items: list[T]
    total: int
    page: int
    page_size: int
    total_pages: int

    @classmethod
    def create(
        cls,
        items: list[T],
        total: int,
        page: int,
        page_size: int,
    ) -> "PaginatedResponse[T]":
        """Create a paginated response."""
        total_pages = (total + page_size - 1) // page_size
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )


class APIResponse(BaseSchema, Generic[T]):
    """Standard API response wrapper."""
    success: bool = True
    data: T | None = None
    message: str | None = None


class ErrorResponse(BaseSchema):
    """Standard error response."""
    success: bool = False
    error: str
    detail: str | None = None
    code: str | None = None

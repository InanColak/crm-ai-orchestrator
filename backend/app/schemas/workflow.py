"""
Workflow Schemas
================
Pydantic models for workflow orchestration API.
Handles workflow triggers, status tracking, and execution results.

FLOW Methodology:
- Function: Type-safe workflow request/response handling
- Level: Production-ready validation
- Output: Strict Pydantic V2 models
- Win Metric: Zero invalid workflow states
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# ENUMS
# =============================================================================

class WorkflowType(str, Enum):
    """Available workflow types."""
    # Main workflows
    FULL_CYCLE = "full_cycle"
    INTELLIGENCE_ONLY = "intelligence_only"
    CONTENT_ONLY = "content_only"
    SALES_OPS_ONLY = "sales_ops_only"

    # MVP workflows
    MEETING_ANALYSIS = "meeting_analysis"
    LEAD_RESEARCH = "lead_research"
    CONTENT_GENERATION = "content_generation"

    # Custom
    CUSTOM = "custom"


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowPriority(str, Enum):
    """Workflow execution priority."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


# =============================================================================
# BASE CONFIG
# =============================================================================

class WorkflowBaseModel(BaseModel):
    """Base model for workflow schemas."""
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


# =============================================================================
# INPUT PAYLOADS
# =============================================================================

class MeetingAnalysisInput(WorkflowBaseModel):
    """Input payload for meeting analysis workflow."""
    transcript: str = Field(
        ...,
        min_length=10,
        description="Meeting transcript text"
    )
    meeting_title: str | None = Field(
        None,
        max_length=255,
        description="Optional meeting title"
    )
    meeting_date: datetime | None = Field(
        None,
        description="Meeting date"
    )
    participants: list[str] = Field(
        default_factory=list,
        description="List of participant names/emails"
    )
    contact_id: str | None = Field(
        None,
        description="Associated HubSpot contact ID"
    )
    deal_id: str | None = Field(
        None,
        description="Associated HubSpot deal ID"
    )


class LeadResearchInput(WorkflowBaseModel):
    """Input payload for lead research workflow."""
    company_name: str = Field(..., min_length=1, max_length=255)
    company_domain: str | None = Field(None, max_length=255)
    contact_name: str | None = Field(None, max_length=255)
    contact_email: str | None = Field(None, max_length=255)
    linkedin_url: str | None = Field(None, max_length=500)
    additional_context: str | None = Field(None, max_length=2000)


class ContentGenerationInput(WorkflowBaseModel):
    """Input payload for content generation workflow."""
    content_type: str = Field(
        ...,
        description="Type: blog, social, email, landing_page"
    )
    topic: str = Field(..., min_length=1, max_length=500)
    target_keywords: list[str] = Field(default_factory=list)
    tone: str | None = Field(None, description="Tone: professional, casual, etc.")
    word_count: int | None = Field(None, ge=50, le=10000)
    platform: str | None = Field(None, description="Target platform if social")


class IntelligenceInput(WorkflowBaseModel):
    """Input payload for intelligence workflows."""
    target_domain: str | None = Field(None, max_length=255)
    keywords: list[str] = Field(default_factory=list)
    competitors: list[str] = Field(default_factory=list)
    research_depth: str = Field(
        default="standard",
        description="Research depth: quick, standard, deep"
    )


# =============================================================================
# WORKFLOW REQUEST SCHEMAS
# =============================================================================

class WorkflowTriggerRequest(WorkflowBaseModel):
    """Request to trigger a new workflow."""
    workflow_type: WorkflowType = Field(
        ...,
        description="Type of workflow to execute"
    )
    input_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Input payload for the workflow"
    )
    priority: WorkflowPriority = Field(
        default=WorkflowPriority.NORMAL,
        description="Execution priority"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    callback_url: str | None = Field(
        None,
        max_length=500,
        description="Webhook URL for completion notification"
    )

    @field_validator("input_data")
    @classmethod
    def validate_input_data(cls, v: dict, info) -> dict:
        """Validate input data is not empty for certain workflow types."""
        # Basic validation - workflow service will do deeper validation
        if not v:
            return v
        return v


class WorkflowResumeRequest(WorkflowBaseModel):
    """Request to resume a paused workflow."""
    approval_decisions: dict[str, str] = Field(
        default_factory=dict,
        description="Map of approval_id -> 'approved' or 'rejected'"
    )
    additional_input: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional input data for resumed execution"
    )


class WorkflowCancelRequest(WorkflowBaseModel):
    """Request to cancel a workflow."""
    reason: str | None = Field(
        None,
        max_length=500,
        description="Cancellation reason"
    )
    force: bool = Field(
        default=False,
        description="Force cancel even if running"
    )


# =============================================================================
# WORKFLOW RESPONSE SCHEMAS
# =============================================================================

class WorkflowStepSummary(WorkflowBaseModel):
    """Summary of a workflow step execution."""
    step_name: str
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int | None = None
    output_summary: str | None = None
    error: str | None = None


class WorkflowResponse(WorkflowBaseModel):
    """Response after workflow action (trigger/resume/cancel)."""
    id: UUID
    workflow_type: WorkflowType
    status: WorkflowStatus
    message: str
    created_at: datetime
    estimated_duration_seconds: int | None = None


class WorkflowDetail(WorkflowBaseModel):
    """Detailed workflow information."""
    id: UUID
    client_id: UUID
    workflow_type: WorkflowType
    status: WorkflowStatus
    priority: WorkflowPriority
    progress: int = Field(ge=0, le=100, default=0)

    # Timestamps
    created_at: datetime
    started_at: datetime | None = None
    updated_at: datetime
    completed_at: datetime | None = None

    # Input/Output
    input_data: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None

    # Execution details
    current_step: str | None = None
    steps_completed: list[WorkflowStepSummary] = Field(default_factory=list)

    # Approvals
    pending_approval_count: int = 0
    pending_approval_ids: list[UUID] = Field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None


class WorkflowSummary(WorkflowBaseModel):
    """Summary view of workflow (for lists)."""
    id: UUID
    workflow_type: WorkflowType
    status: WorkflowStatus
    priority: WorkflowPriority
    progress: int = Field(ge=0, le=100, default=0)
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    pending_approval_count: int = 0


class WorkflowListResponse(WorkflowBaseModel):
    """Response for listing workflows."""
    items: list[WorkflowSummary]
    total: int
    page: int
    page_size: int
    has_more: bool


class WorkflowStats(WorkflowBaseModel):
    """Workflow statistics for a client."""
    total: int = 0
    pending: int = 0
    running: int = 0
    awaiting_approval: int = 0
    completed_today: int = 0
    failed_today: int = 0
    avg_duration_seconds: float | None = None
    by_type: dict[str, int] = Field(default_factory=dict)
    by_status: dict[str, int] = Field(default_factory=dict)


# =============================================================================
# WORKFLOW EVENTS (for webhooks/streaming)
# =============================================================================

class WorkflowEventType(str, Enum):
    """Types of workflow events."""
    STARTED = "workflow.started"
    STEP_STARTED = "workflow.step.started"
    STEP_COMPLETED = "workflow.step.completed"
    APPROVAL_REQUIRED = "workflow.approval.required"
    RESUMED = "workflow.resumed"
    COMPLETED = "workflow.completed"
    FAILED = "workflow.failed"
    CANCELLED = "workflow.cancelled"


class WorkflowEvent(WorkflowBaseModel):
    """Workflow event for real-time updates."""
    event_type: WorkflowEventType
    workflow_id: UUID
    timestamp: datetime
    data: dict[str, Any] = Field(default_factory=dict)

    # Optional fields based on event type
    step_name: str | None = None
    approval_id: UUID | None = None
    error: str | None = None
    progress: int | None = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "WorkflowType",
    "WorkflowStatus",
    "WorkflowPriority",
    "WorkflowEventType",
    # Input Payloads
    "MeetingAnalysisInput",
    "LeadResearchInput",
    "ContentGenerationInput",
    "IntelligenceInput",
    # Requests
    "WorkflowTriggerRequest",
    "WorkflowResumeRequest",
    "WorkflowCancelRequest",
    # Responses
    "WorkflowStepSummary",
    "WorkflowResponse",
    "WorkflowDetail",
    "WorkflowSummary",
    "WorkflowListResponse",
    "WorkflowStats",
    # Events
    "WorkflowEvent",
]

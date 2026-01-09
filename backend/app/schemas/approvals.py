"""
Approval Schemas
================
Pydantic models for Human-in-the-Loop (HITL) approval system.
Implements ADR-014 risk-based tiered approval.

FLOW Methodology:
- Function: Type-safe approval request/response handling
- Level: Production-ready validation
- Output: Strict Pydantic V2 models
- Win Metric: Zero invalid approval states
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

class ApprovalType(str, Enum):
    """Types of actions requiring approval (ADR-014)."""
    # CRM Operations
    CRM_CREATE_CONTACT = "crm_create_contact"
    CRM_CREATE_DEAL = "crm_create_deal"
    CRM_CREATE_TASK = "crm_create_task"
    CRM_CREATE_NOTE = "crm_create_note"
    CRM_UPDATE_CONTACT = "crm_update_contact"
    CRM_UPDATE_DEAL = "crm_update_deal"

    # Legacy types (backward compatibility)
    CRM_UPDATE = "crm_update"
    CRM_CREATE = "crm_create"

    # Content & Communication
    CONTENT_PUBLISH = "content_publish"
    EMAIL_SEND = "email_send"
    TASK_CREATE = "task_create"
    MEETING_SCHEDULE = "meeting_schedule"


class ApprovalStatus(str, Enum):
    """Approval decision status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    EXECUTING = "executing"  # Being executed after approval
    EXECUTED = "executed"  # Successfully executed
    FAILED = "failed"  # Execution failed


class ApprovalPriority(str, Enum):
    """Priority levels for approvals."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class RiskLevel(str, Enum):
    """Risk level for approval (ADR-014)."""
    LOW = "low"  # Auto-approve capable
    MEDIUM = "medium"  # Requires single approval
    HIGH = "high"  # Requires review + confirmation


# =============================================================================
# BASE CONFIG
# =============================================================================

class ApprovalBaseModel(BaseModel):
    """Base model for approval schemas."""
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


# =============================================================================
# RISK ASSESSMENT (ADR-014)
# =============================================================================

class RiskFactor(ApprovalBaseModel):
    """Individual risk factor for an operation."""
    name: str = Field(..., description="Risk factor name")
    description: str = Field(..., description="Risk factor description")
    level: RiskLevel = Field(..., description="Risk level")


class RiskAssessment(ApprovalBaseModel):
    """Risk assessment for an approval request."""
    overall_level: RiskLevel = Field(..., description="Overall risk level")
    factors: list[RiskFactor] = Field(default_factory=list)
    auto_approve_eligible: bool = Field(
        default=False,
        description="Can be auto-approved based on risk"
    )
    requires_confirmation: bool = Field(
        default=False,
        description="Requires additional confirmation step"
    )


# =============================================================================
# APPROVAL PAYLOAD
# =============================================================================

class CRMPayload(ApprovalBaseModel):
    """Payload for CRM operations."""
    operation: str = Field(..., description="create, update, delete")
    object_type: str = Field(..., description="contact, deal, task, note")
    object_id: str | None = Field(None, description="ID for update/delete")
    properties: dict[str, Any] = Field(default_factory=dict)
    associations: list[dict[str, Any]] | None = None


class EmailPayload(ApprovalBaseModel):
    """Payload for email operations."""
    to: list[str]
    cc: list[str] | None = None
    bcc: list[str] | None = None
    subject: str
    body: str
    template_id: str | None = None


class ContentPayload(ApprovalBaseModel):
    """Payload for content publish operations."""
    content_type: str = Field(..., description="blog, social, email, etc.")
    title: str
    content: str
    platform: str | None = None
    scheduled_at: datetime | None = None


# =============================================================================
# APPROVAL REQUEST SCHEMAS
# =============================================================================

class ApprovalCreateRequest(ApprovalBaseModel):
    """Request to create a new approval."""
    workflow_id: UUID
    client_id: UUID
    approval_type: ApprovalType
    title: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    payload: dict[str, Any]
    diff_before: dict[str, Any] | None = Field(
        None,
        description="State before the change (for updates)"
    )
    diff_after: dict[str, Any] | None = Field(
        None,
        description="State after the change (for updates)"
    )
    priority: ApprovalPriority = ApprovalPriority.NORMAL
    risk_assessment: RiskAssessment | None = None
    expires_in_hours: int | None = Field(
        None,
        ge=1,
        le=168,
        description="Hours until expiration (1-168)"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecisionRequest(ApprovalBaseModel):
    """Request to approve or reject."""
    reviewed_by: str = Field(..., min_length=1, max_length=255)
    rejection_reason: str | None = Field(
        None,
        max_length=1000,
        description="Required when rejecting"
    )

    @field_validator("rejection_reason")
    @classmethod
    def validate_rejection_reason(cls, v: str | None) -> str | None:
        if v is not None and len(v.strip()) == 0:
            return None
        return v


class BulkApprovalRequest(ApprovalBaseModel):
    """Request for bulk approval/rejection."""
    approval_ids: list[UUID] = Field(..., min_length=1, max_length=100)
    reviewed_by: str = Field(..., min_length=1, max_length=255)
    rejection_reason: str | None = None


# =============================================================================
# APPROVAL RESPONSE SCHEMAS
# =============================================================================

class ApprovalSummary(ApprovalBaseModel):
    """Summary view of an approval (for lists)."""
    id: UUID
    workflow_id: UUID
    approval_type: ApprovalType
    title: str
    description: str | None = None
    status: ApprovalStatus
    priority: ApprovalPriority
    risk_level: RiskLevel | None = None
    created_at: datetime
    expires_at: datetime | None = None


class ApprovalDetail(ApprovalSummary):
    """Detailed view of an approval."""
    client_id: UUID
    payload: dict[str, Any]
    diff_before: dict[str, Any] | None = None
    diff_after: dict[str, Any] | None = None
    risk_assessment: RiskAssessment | None = None
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    rejection_reason: str | None = None
    execution_result: dict[str, Any] | None = None
    execution_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime | None = None


class ApprovalActionResponse(ApprovalBaseModel):
    """Response after approval action."""
    id: UUID
    status: ApprovalStatus
    message: str
    workflow_resumed: bool = False
    execution_result: dict[str, Any] | None = None


class BulkApprovalResponse(ApprovalBaseModel):
    """Response for bulk operations."""
    total: int
    succeeded: int
    failed: int
    results: list[ApprovalActionResponse]


class ApprovalListResponse(ApprovalBaseModel):
    """Response for listing approvals."""
    items: list[ApprovalSummary]
    total: int
    pending_count: int
    page: int
    page_size: int


class ApprovalStats(ApprovalBaseModel):
    """Approval statistics for a client."""
    pending: int = 0
    approved_today: int = 0
    rejected_today: int = 0
    expired_today: int = 0
    avg_response_time_hours: float | None = None
    by_type: dict[str, int] = Field(default_factory=dict)
    by_priority: dict[str, int] = Field(default_factory=dict)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_risk_level(approval_type: ApprovalType, payload: dict) -> RiskAssessment:
    """
    Calculate risk level based on operation type and payload.

    ADR-014 Risk Classification:
    - LOW: Read operations, minor updates
    - MEDIUM: Single record create/update
    - HIGH: Bulk operations, external communications
    """
    factors: list[RiskFactor] = []

    # Base risk by type
    if approval_type in (ApprovalType.CRM_CREATE_NOTE, ApprovalType.TASK_CREATE):
        base_level = RiskLevel.LOW
        factors.append(RiskFactor(
            name="operation_type",
            description="Low-risk CRM operation",
            level=RiskLevel.LOW
        ))
    elif approval_type in (
        ApprovalType.CRM_CREATE_CONTACT,
        ApprovalType.CRM_CREATE_DEAL,
        ApprovalType.CRM_UPDATE_CONTACT,
        ApprovalType.CRM_UPDATE_DEAL,
        ApprovalType.CRM_CREATE,
        ApprovalType.CRM_UPDATE
    ):
        base_level = RiskLevel.MEDIUM
        factors.append(RiskFactor(
            name="operation_type",
            description="CRM record modification",
            level=RiskLevel.MEDIUM
        ))
    elif approval_type in (ApprovalType.EMAIL_SEND, ApprovalType.CONTENT_PUBLISH):
        base_level = RiskLevel.HIGH
        factors.append(RiskFactor(
            name="operation_type",
            description="External communication",
            level=RiskLevel.HIGH
        ))
    else:
        base_level = RiskLevel.MEDIUM

    # Check for bulk operations
    if isinstance(payload.get("items"), list) and len(payload["items"]) > 1:
        factors.append(RiskFactor(
            name="bulk_operation",
            description=f"Bulk operation with {len(payload['items'])} items",
            level=RiskLevel.HIGH
        ))
        base_level = RiskLevel.HIGH

    # Check for sensitive fields
    sensitive_fields = {"email", "phone", "amount", "dealstage", "lifecyclestage"}
    if payload.get("properties"):
        modified_fields = set(payload["properties"].keys())
        sensitive_modified = modified_fields & sensitive_fields
        if sensitive_modified:
            factors.append(RiskFactor(
                name="sensitive_fields",
                description=f"Modifies sensitive fields: {', '.join(sensitive_modified)}",
                level=RiskLevel.MEDIUM
            ))
            if base_level == RiskLevel.LOW:
                base_level = RiskLevel.MEDIUM

    return RiskAssessment(
        overall_level=base_level,
        factors=factors,
        auto_approve_eligible=base_level == RiskLevel.LOW,
        requires_confirmation=base_level == RiskLevel.HIGH
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ApprovalType",
    "ApprovalStatus",
    "ApprovalPriority",
    "RiskLevel",
    # Risk Assessment
    "RiskFactor",
    "RiskAssessment",
    # Payloads
    "CRMPayload",
    "EmailPayload",
    "ContentPayload",
    # Requests
    "ApprovalCreateRequest",
    "ApprovalDecisionRequest",
    "BulkApprovalRequest",
    # Responses
    "ApprovalSummary",
    "ApprovalDetail",
    "ApprovalActionResponse",
    "BulkApprovalResponse",
    "ApprovalListResponse",
    "ApprovalStats",
    # Helpers
    "calculate_risk_level",
]

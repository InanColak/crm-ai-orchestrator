"""
CRM Update Schemas (Phase 3.3)
==============================
Pydantic schemas for CRM Updater agent.
Prepares CRM operations for human approval and execution.

Data Flow:
  CRMTask (pending) -> CRMUpdater -> CRMUpdateOperation -> Approval -> HubSpot SDK
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# ENUMS
# =============================================================================

class CRMOperationType(str, Enum):
    """Types of CRM operations."""
    CREATE_TASK = "create_task"
    UPDATE_DEAL = "update_deal"
    ADD_NOTE = "add_note"
    CREATE_ACTIVITY = "create_activity"
    UPDATE_CONTACT = "update_contact"
    UPDATE_DEAL_STAGE = "update_deal_stage"


class OperationRiskLevel(str, Enum):
    """Risk level for CRM operations (ADR-014)."""
    LOW = "low"      # Notes, tasks - auto-approve eligible
    MEDIUM = "medium"  # Single record updates
    HIGH = "high"    # Bulk operations, sensitive field updates


class OperationStatus(str, Enum):
    """Status of a CRM operation."""
    PENDING = "pending"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


# =============================================================================
# OPERATION PAYLOADS
# =============================================================================

class TaskOperationPayload(BaseModel):
    """Payload for creating a task in CRM."""
    model_config = ConfigDict(extra="forbid")

    # HubSpot task properties
    hs_task_subject: str = Field(..., min_length=1, max_length=255)
    hs_task_body: str | None = Field(None, max_length=5000)
    hs_task_status: str = Field("NOT_STARTED")
    hs_task_priority: str = Field("MEDIUM")
    hs_task_type: str = Field("TODO")
    hs_timestamp: str | None = Field(None, description="Due date as Unix timestamp (ms)")
    hubspot_owner_id: str | None = Field(None, description="Assigned owner ID")

    # Associations
    contact_ids: list[str] = Field(default_factory=list)
    deal_ids: list[str] = Field(default_factory=list)
    company_ids: list[str] = Field(default_factory=list)

    # Metadata
    source_meeting_id: str | None = None


class DealUpdatePayload(BaseModel):
    """Payload for updating a deal."""
    model_config = ConfigDict(extra="forbid")

    deal_id: str = Field(..., description="HubSpot deal ID")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Properties to update"
    )

    # Common deal updates
    dealstage: str | None = Field(None, description="New deal stage")
    amount: float | None = Field(None, description="Deal amount")
    closedate: str | None = Field(None, description="Expected close date")
    notes_last_updated: str | None = Field(None)


class NotePayload(BaseModel):
    """Payload for adding a note to CRM entity."""
    model_config = ConfigDict(extra="forbid")

    body: str = Field(..., min_length=1, max_length=10000)
    timestamp: str | None = Field(None, description="Note timestamp")

    # Entity to attach note to
    contact_id: str | None = None
    deal_id: str | None = None
    company_id: str | None = None

    # Metadata
    source: str = Field("meeting_notes", description="Source of the note")


class ActivityPayload(BaseModel):
    """Payload for logging an activity."""
    model_config = ConfigDict(extra="forbid")

    activity_type: Literal["meeting", "call", "email", "task_completed"] = "meeting"
    subject: str = Field(..., min_length=1, max_length=255)
    body: str | None = None
    timestamp: str | None = None
    duration_minutes: int | None = None

    # Associations
    contact_ids: list[str] = Field(default_factory=list)
    deal_ids: list[str] = Field(default_factory=list)


# =============================================================================
# CRM UPDATE OPERATION
# =============================================================================

class CRMUpdateOperation(BaseModel):
    """
    Single CRM update operation ready for approval.
    Created by LLM from CRMTask data.
    """
    model_config = ConfigDict(extra="forbid")

    # Operation identity
    operation_id: str = Field(..., description="Unique operation ID")
    operation_type: CRMOperationType = Field(..., description="Type of CRM operation")

    # Human-readable description
    summary: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Human-readable summary of this operation"
    )
    details: str | None = Field(
        None,
        description="Additional details for the reviewer"
    )

    # API payload (exact data to send to CRM)
    payload: dict[str, Any] = Field(
        ...,
        description="Exact API payload for CRM"
    )

    # Risk assessment
    risk_level: OperationRiskLevel = Field(
        OperationRiskLevel.LOW,
        description="Risk level of this operation"
    )
    risk_factors: list[str] = Field(
        default_factory=list,
        description="Factors contributing to risk level"
    )

    # Approval
    approval_required: bool = Field(True, description="Whether human approval is needed")
    auto_approve_eligible: bool = Field(
        False,
        description="Can be auto-approved based on low risk"
    )

    # Rollback info
    rollback_info: str | None = Field(
        None,
        description="How to undo this operation if needed"
    )

    # Source tracking
    source_task_id: str | None = Field(
        None,
        description="ID of source CRMTask"
    )


# =============================================================================
# LLM OUTPUT SCHEMA
# =============================================================================

class CRMUpdateOperationResult(BaseModel):
    """
    LLM output for CRM update preparation.
    Contains all operations to be approved.
    """
    model_config = ConfigDict(extra="forbid")

    # Operations list
    operations: list[CRMUpdateOperation] = Field(
        default_factory=list,
        description="List of CRM operations to execute"
    )

    # Summary
    batch_summary: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Overall summary of all operations"
    )
    total_operations: int = Field(0, description="Total number of operations")
    high_risk_count: int = Field(0, description="Count of high-risk operations")

    # Additional notes/context
    deal_stage_changes: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Recommended deal stage changes based on meeting"
    )
    notes_to_add: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Meeting notes to add to CRM entities"
    )

    # Processing info
    processing_notes: str | None = Field(
        None,
        description="Notes about the preparation process"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings for the reviewer"
    )


# =============================================================================
# API REQUEST/RESPONSE SCHEMAS
# =============================================================================

class CRMUpdateRequest(BaseModel):
    """API request to prepare CRM updates."""
    model_config = ConfigDict(extra="forbid")

    client_id: UUID = Field(..., description="Client tenant ID")
    workflow_id: str = Field(..., description="Source workflow ID")

    # Options
    include_deal_updates: bool = Field(
        True,
        description="Include deal stage recommendations"
    )
    include_meeting_notes: bool = Field(
        True,
        description="Add meeting notes to CRM entities"
    )
    auto_approve_low_risk: bool = Field(
        False,
        description="Auto-approve low-risk operations"
    )


class CRMUpdateResponse(BaseModel):
    """API response for CRM update preparation."""
    model_config = ConfigDict(extra="forbid")

    success: bool = True
    workflow_id: str
    operations_prepared: int
    approval_ids: list[str] = Field(
        default_factory=list,
        description="IDs of created approval requests"
    )
    auto_approved: int = Field(0, description="Operations auto-approved")
    execution_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Results of auto-approved operations"
    )

    # Metadata
    processing_time_ms: int | None = None
    llm_model: str | None = None


# =============================================================================
# EXECUTION RESULT SCHEMAS
# =============================================================================

class OperationExecutionResult(BaseModel):
    """Result of executing a single CRM operation."""
    model_config = ConfigDict(extra="forbid")

    operation_id: str
    success: bool
    crm_entity_id: str | None = Field(
        None,
        description="ID of created/updated entity in CRM"
    )
    error_message: str | None = None
    executed_at: datetime | None = None

    # Rollback info
    rollback_available: bool = False
    rollback_data: dict[str, Any] | None = None


class BatchExecutionResult(BaseModel):
    """Result of executing a batch of CRM operations."""
    model_config = ConfigDict(extra="forbid")

    total: int
    succeeded: int
    failed: int
    results: list[OperationExecutionResult]

    # Summary
    summary: str


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "CRMOperationType",
    "OperationRiskLevel",
    "OperationStatus",
    # Payloads
    "TaskOperationPayload",
    "DealUpdatePayload",
    "NotePayload",
    "ActivityPayload",
    # Operation schemas
    "CRMUpdateOperation",
    "CRMUpdateOperationResult",
    # API schemas
    "CRMUpdateRequest",
    "CRMUpdateResponse",
    # Execution schemas
    "OperationExecutionResult",
    "BatchExecutionResult",
]

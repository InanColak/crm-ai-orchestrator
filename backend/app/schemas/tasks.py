"""
Task Extraction Schemas (Phase 3.2)
===================================
Pydantic schemas for Task Extractor agent.
Converts meeting action items into CRM-ready HubSpot/Salesforce tasks.

Data Flow:
  MeetingAnalysis.action_items -> TaskExtractor -> CRMTask (pending approval)
"""

from __future__ import annotations

from datetime import datetime, date
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# ENUMS
# =============================================================================

class TaskPriority(str, Enum):
    """HubSpot task priority levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TaskStatus(str, Enum):
    """HubSpot task status values."""
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    WAITING = "WAITING"
    COMPLETED = "COMPLETED"


class TaskType(str, Enum):
    """HubSpot task types."""
    TODO = "TODO"
    CALL = "CALL"
    EMAIL = "EMAIL"


class AssociationType(str, Enum):
    """CRM entity types for task associations."""
    CONTACT = "contact"
    COMPANY = "company"
    DEAL = "deal"


# =============================================================================
# INPUT SCHEMAS
# =============================================================================

class ActionItemInput(BaseModel):
    """
    Action item from meeting notes analysis.
    This is the input from MeetingAnalysis.action_items.
    """
    model_config = ConfigDict(extra="forbid")

    task: str = Field(..., description="Description of the task")
    assignee: str | None = Field(None, description="Person responsible")
    due_date: str | None = Field(None, description="Due date (relative or absolute)")
    priority: str = Field("medium", description="Priority level")
    context: str | None = Field(None, description="Additional context")


class TaskExtractionInput(BaseModel):
    """
    Input for task extraction from a meeting.
    Contains meeting context and action items to process.
    """
    model_config = ConfigDict(extra="forbid")

    # Meeting context
    meeting_id: str = Field(..., description="Source meeting ID")
    meeting_title: str | None = Field(None, description="Meeting title")
    meeting_date: datetime | None = Field(None, description="When the meeting occurred")
    meeting_summary: str | None = Field(None, description="Meeting summary for context")

    # Action items to process
    action_items: list[ActionItemInput] = Field(
        ...,
        min_length=1,
        description="Action items extracted from meeting"
    )

    # CRM context
    deal_id: str | None = Field(None, description="Associated HubSpot deal ID")
    contact_id: str | None = Field(None, description="Associated HubSpot contact ID")
    company_id: str | None = Field(None, description="Associated HubSpot company ID")

    # Team context for assignee resolution
    team_members: list[dict] | None = Field(
        None,
        description="Team member list for assignee matching [{'name': str, 'email': str, 'hubspot_owner_id': str}]"
    )


# =============================================================================
# OUTPUT SCHEMAS (LLM Structured Output)
# =============================================================================

class TaskAssociation(BaseModel):
    """Association between task and CRM entity."""
    model_config = ConfigDict(extra="forbid")

    association_type: AssociationType = Field(
        ...,
        description="Type of CRM entity to associate with"
    )
    entity_id: str | None = Field(
        None,
        description="CRM entity ID (if known)"
    )
    entity_name: str | None = Field(
        None,
        description="Entity name for reference"
    )


class ExtractedTask(BaseModel):
    """
    Single task ready for CRM creation.
    LLM structured output schema.
    """
    model_config = ConfigDict(extra="forbid")

    # Task content
    subject: str = Field(
        ...,
        min_length=5,
        max_length=255,
        description="Task subject/title (action-oriented, starts with verb)"
    )
    body: str | None = Field(
        None,
        description="Task description with context from meeting"
    )

    # Task metadata
    task_type: TaskType = Field(
        TaskType.TODO,
        description="Type of task (TODO, CALL, EMAIL)"
    )
    priority: TaskPriority = Field(
        TaskPriority.MEDIUM,
        description="Task priority level"
    )
    status: TaskStatus = Field(
        TaskStatus.NOT_STARTED,
        description="Initial task status"
    )

    # Timing
    due_date: str | None = Field(
        None,
        description="Due date in YYYY-MM-DD format"
    )
    due_date_reasoning: str | None = Field(
        None,
        description="Explanation for the chosen due date"
    )

    # Assignment
    assignee_name: str | None = Field(
        None,
        description="Name of person responsible"
    )
    assignee_email: str | None = Field(
        None,
        description="Email of person responsible"
    )
    hubspot_owner_id: str | None = Field(
        None,
        description="HubSpot owner ID for assignment"
    )

    # Associations
    associations: list[TaskAssociation] = Field(
        default_factory=list,
        description="CRM entities to associate task with"
    )

    # Source tracking
    source_action_item: str = Field(
        ...,
        description="Original action item text from meeting"
    )
    meeting_context: str | None = Field(
        None,
        description="Relevant meeting context for this task"
    )

    # Confidence
    extraction_confidence: Literal["low", "medium", "high"] = Field(
        "medium",
        description="Confidence in the extraction quality"
    )
    needs_review: bool = Field(
        False,
        description="Whether this task needs human review before creation"
    )
    review_reason: str | None = Field(
        None,
        description="Reason for needing review"
    )


class TaskExtractionResult(BaseModel):
    """
    Complete task extraction output.
    Contains all extracted tasks and metadata.
    """
    model_config = ConfigDict(extra="forbid")

    # Extracted tasks
    tasks: list[ExtractedTask] = Field(
        default_factory=list,
        description="Tasks extracted and enriched from action items"
    )

    # Skipped items
    skipped_items: list[dict] = Field(
        default_factory=list,
        description="Action items that couldn't be converted to tasks [{'item': str, 'reason': str}]"
    )

    # Summary
    total_action_items: int = Field(
        0,
        description="Total action items processed"
    )
    tasks_created: int = Field(
        0,
        description="Number of tasks successfully extracted"
    )
    tasks_needing_review: int = Field(
        0,
        description="Number of tasks flagged for review"
    )

    # Processing notes
    processing_notes: str | None = Field(
        None,
        description="Notes about the extraction process"
    )


# =============================================================================
# CRM PAYLOAD SCHEMAS
# =============================================================================

class HubSpotTaskPayload(BaseModel):
    """
    HubSpot Task API payload.
    Ready to be sent to HubSpot API after approval.
    """
    model_config = ConfigDict(extra="forbid")

    # Required properties
    hs_task_subject: str = Field(..., description="Task subject")
    hs_task_body: str | None = Field(None, description="Task description")
    hs_task_status: str = Field("NOT_STARTED", description="Task status")
    hs_task_priority: str = Field("MEDIUM", description="Task priority")
    hs_task_type: str = Field("TODO", description="Task type")

    # Optional properties
    hs_timestamp: str | None = Field(None, description="Due date timestamp")
    hubspot_owner_id: str | None = Field(None, description="Assigned owner ID")

    # Custom properties (if configured in HubSpot)
    source_meeting_id: str | None = Field(None, description="Source meeting ID")


class TaskApprovalPayload(BaseModel):
    """
    Payload for task creation approval request.
    Includes task details and HubSpot-ready payload.
    """
    model_config = ConfigDict(extra="forbid")

    # Task details for display
    extracted_task: ExtractedTask

    # HubSpot-ready payload
    hubspot_payload: HubSpotTaskPayload

    # Associations to create
    associations: list[dict] = Field(
        default_factory=list,
        description="HubSpot association objects to create"
    )


# =============================================================================
# API SCHEMAS
# =============================================================================

class TaskExtractionRequest(BaseModel):
    """API request for task extraction."""
    model_config = ConfigDict(extra="forbid")

    client_id: UUID = Field(..., description="Client tenant ID")
    meeting_id: str = Field(..., description="Meeting ID to extract tasks from")

    # Options
    auto_assign: bool = Field(
        True,
        description="Attempt to auto-assign tasks based on meeting participants"
    )
    default_due_days: int = Field(
        7,
        ge=1,
        le=90,
        description="Default days until due if not specified"
    )
    create_approval_requests: bool = Field(
        True,
        description="Create approval requests for extracted tasks"
    )


class TaskExtractionResponse(BaseModel):
    """API response for task extraction."""
    model_config = ConfigDict(extra="forbid")

    success: bool = True
    meeting_id: str
    extraction_result: TaskExtractionResult

    # Approval tracking
    approval_ids: list[str] = Field(
        default_factory=list,
        description="IDs of created approval requests"
    )

    # Metadata
    processing_time_ms: int | None = None
    llm_model: str | None = None

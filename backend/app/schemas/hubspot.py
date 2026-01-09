"""
HubSpot Schemas
===============
Pydantic models for HubSpot CRM entities.
Validates all data before sending to HubSpot API.

FLOW Methodology:
- Function: Type-safe CRM data exchange
- Level: Production-ready with strict validation
- Output: Validated Pydantic models
- Win Metric: Zero invalid data sent to HubSpot
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, ConfigDict


# =============================================================================
# ENUMS
# =============================================================================

class HubSpotObjectType(str, Enum):
    """Supported HubSpot CRM object types."""
    CONTACT = "contacts"
    COMPANY = "companies"
    DEAL = "deals"
    TASK = "tasks"
    NOTE = "notes"
    ENGAGEMENT = "engagements"


class DealStage(str, Enum):
    """Standard HubSpot deal stages."""
    APPOINTMENT_SCHEDULED = "appointmentscheduled"
    QUALIFIED_TO_BUY = "qualifiedtobuy"
    PRESENTATION_SCHEDULED = "presentationscheduled"
    DECISION_MAKER_BOUGHT_IN = "decisionmakerboughtin"
    CONTRACT_SENT = "contractsent"
    CLOSED_WON = "closedwon"
    CLOSED_LOST = "closedlost"


class TaskStatus(str, Enum):
    """HubSpot task statuses."""
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    WAITING = "WAITING"
    COMPLETED = "COMPLETED"


class TaskPriority(str, Enum):
    """HubSpot task priorities."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TaskType(str, Enum):
    """HubSpot task types."""
    CALL = "CALL"
    EMAIL = "EMAIL"
    TODO = "TODO"


# =============================================================================
# BASE MODELS
# =============================================================================

class HubSpotBaseModel(BaseModel):
    """Base model for all HubSpot entities."""

    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )


class HubSpotProperties(HubSpotBaseModel):
    """Base for HubSpot properties dict."""
    pass


# =============================================================================
# CONTACT
# =============================================================================

class ContactProperties(HubSpotProperties):
    """HubSpot Contact properties."""
    email: str | None = Field(None, description="Contact email address")
    firstname: str | None = Field(None, description="First name")
    lastname: str | None = Field(None, description="Last name")
    phone: str | None = Field(None, description="Phone number")
    company: str | None = Field(None, description="Company name")
    jobtitle: str | None = Field(None, description="Job title")
    lifecyclestage: str | None = Field(None, description="Lifecycle stage")
    hs_lead_status: str | None = Field(None, description="Lead status")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str | None) -> str | None:
        if v and "@" not in v:
            raise ValueError("Invalid email format")
        return v.lower() if v else None


class ContactCreate(HubSpotBaseModel):
    """Request model for creating a contact."""
    properties: ContactProperties


class ContactUpdate(HubSpotBaseModel):
    """Request model for updating a contact."""
    properties: ContactProperties


class Contact(HubSpotBaseModel):
    """HubSpot Contact response model."""
    id: str = Field(..., description="HubSpot contact ID")
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = Field(None, alias="createdAt")
    updated_at: datetime | None = Field(None, alias="updatedAt")
    archived: bool = False


# =============================================================================
# COMPANY
# =============================================================================

class CompanyProperties(HubSpotProperties):
    """HubSpot Company properties."""
    name: str | None = Field(None, description="Company name")
    domain: str | None = Field(None, description="Company website domain")
    industry: str | None = Field(None, description="Industry")
    phone: str | None = Field(None, description="Company phone")
    city: str | None = Field(None, description="City")
    state: str | None = Field(None, description="State/Region")
    country: str | None = Field(None, description="Country")
    numberofemployees: str | None = Field(None, description="Number of employees")
    annualrevenue: str | None = Field(None, description="Annual revenue")


class CompanyCreate(HubSpotBaseModel):
    """Request model for creating a company."""
    properties: CompanyProperties


class Company(HubSpotBaseModel):
    """HubSpot Company response model."""
    id: str = Field(..., description="HubSpot company ID")
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = Field(None, alias="createdAt")
    updated_at: datetime | None = Field(None, alias="updatedAt")
    archived: bool = False


# =============================================================================
# DEAL
# =============================================================================

class DealProperties(HubSpotProperties):
    """HubSpot Deal properties."""
    dealname: str | None = Field(None, description="Deal name")
    amount: str | None = Field(None, description="Deal amount")
    dealstage: str | None = Field(None, description="Deal stage")
    pipeline: str | None = Field(None, description="Pipeline ID")
    closedate: str | None = Field(None, description="Expected close date (Unix ms)")
    hubspot_owner_id: str | None = Field(None, description="Deal owner ID")
    description: str | None = Field(None, description="Deal description")


class DealCreate(HubSpotBaseModel):
    """Request model for creating a deal."""
    properties: DealProperties
    associations: list[dict[str, Any]] | None = None


class DealUpdate(HubSpotBaseModel):
    """Request model for updating a deal."""
    properties: DealProperties


class Deal(HubSpotBaseModel):
    """HubSpot Deal response model."""
    id: str = Field(..., description="HubSpot deal ID")
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = Field(None, alias="createdAt")
    updated_at: datetime | None = Field(None, alias="updatedAt")
    archived: bool = False


# =============================================================================
# TASK
# =============================================================================

class TaskProperties(HubSpotProperties):
    """HubSpot Task properties."""
    hs_task_subject: str = Field(..., description="Task subject/title")
    hs_task_body: str | None = Field(None, description="Task description")
    hs_task_status: TaskStatus = Field(
        default=TaskStatus.NOT_STARTED,
        description="Task status"
    )
    hs_task_priority: TaskPriority = Field(
        default=TaskPriority.MEDIUM,
        description="Task priority"
    )
    hs_task_type: TaskType = Field(
        default=TaskType.TODO,
        description="Task type"
    )
    hs_timestamp: str | None = Field(None, description="Due date (Unix ms)")
    hubspot_owner_id: str | None = Field(None, description="Task owner ID")


class TaskCreate(HubSpotBaseModel):
    """Request model for creating a task."""
    properties: TaskProperties
    associations: list[dict[str, Any]] | None = None


class TaskUpdate(HubSpotBaseModel):
    """Request model for updating a task."""
    properties: TaskProperties


class Task(HubSpotBaseModel):
    """HubSpot Task response model."""
    id: str = Field(..., description="HubSpot task ID")
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = Field(None, alias="createdAt")
    updated_at: datetime | None = Field(None, alias="updatedAt")
    archived: bool = False


# =============================================================================
# NOTE
# =============================================================================

class NoteProperties(HubSpotProperties):
    """HubSpot Note properties."""
    hs_note_body: str = Field(..., description="Note content (HTML allowed)")
    hs_timestamp: str | None = Field(None, description="Note timestamp (Unix ms)")
    hubspot_owner_id: str | None = Field(None, description="Note owner ID")


class NoteCreate(HubSpotBaseModel):
    """Request model for creating a note."""
    properties: NoteProperties
    associations: list[dict[str, Any]] | None = None


class Note(HubSpotBaseModel):
    """HubSpot Note response model."""
    id: str = Field(..., description="HubSpot note ID")
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = Field(None, alias="createdAt")
    updated_at: datetime | None = Field(None, alias="updatedAt")


# =============================================================================
# SEARCH & PAGINATION
# =============================================================================

class SearchFilter(HubSpotBaseModel):
    """Single search filter."""
    propertyName: str
    operator: str  # EQ, NEQ, LT, LTE, GT, GTE, CONTAINS_TOKEN, etc.
    value: str


class SearchFilterGroup(HubSpotBaseModel):
    """Group of filters (AND logic within group)."""
    filters: list[SearchFilter]


class SearchRequest(HubSpotBaseModel):
    """HubSpot CRM search request."""
    filterGroups: list[SearchFilterGroup] = Field(default_factory=list)
    sorts: list[dict[str, str]] = Field(default_factory=list)
    properties: list[str] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=100)
    after: str | None = None


class SearchResult(HubSpotBaseModel):
    """HubSpot search result."""
    total: int
    results: list[dict[str, Any]]
    paging: dict[str, Any] | None = None


# =============================================================================
# ASSOCIATION
# =============================================================================

class AssociationType(str, Enum):
    """Common HubSpot association types."""
    CONTACT_TO_COMPANY = "contact_to_company"
    CONTACT_TO_DEAL = "contact_to_deal"
    DEAL_TO_COMPANY = "deal_to_company"
    TASK_TO_CONTACT = "task_to_contact"
    TASK_TO_DEAL = "task_to_deal"
    NOTE_TO_CONTACT = "note_to_contact"
    NOTE_TO_DEAL = "note_to_deal"


class Association(HubSpotBaseModel):
    """HubSpot association between objects."""
    to_object_type: HubSpotObjectType
    to_object_id: str
    association_type: str | None = None


class AssociationCreate(HubSpotBaseModel):
    """Request to create an association."""
    to: dict[str, str]  # {"id": "123"}
    types: list[dict[str, Any]]  # [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 1}]


# =============================================================================
# API RESPONSE WRAPPERS
# =============================================================================

class HubSpotResponse(HubSpotBaseModel):
    """Generic HubSpot API response wrapper."""
    success: bool = True
    data: Any = None
    error: str | None = None


class HubSpotBatchResponse(HubSpotBaseModel):
    """Batch operation response."""
    status: str  # COMPLETE, PENDING, etc.
    results: list[dict[str, Any]]
    num_errors: int = 0
    errors: list[dict[str, Any]] = Field(default_factory=list)


# =============================================================================
# OWNER
# =============================================================================

class Owner(HubSpotBaseModel):
    """HubSpot Owner (user)."""
    id: str
    email: str | None = None
    first_name: str | None = Field(None, alias="firstName")
    last_name: str | None = Field(None, alias="lastName")
    user_id: int | None = Field(None, alias="userId")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "HubSpotObjectType",
    "DealStage",
    "TaskStatus",
    "TaskPriority",
    "TaskType",
    "AssociationType",
    # Contact
    "ContactProperties",
    "ContactCreate",
    "ContactUpdate",
    "Contact",
    # Company
    "CompanyProperties",
    "CompanyCreate",
    "Company",
    # Deal
    "DealProperties",
    "DealCreate",
    "DealUpdate",
    "Deal",
    # Task
    "TaskProperties",
    "TaskCreate",
    "TaskUpdate",
    "Task",
    # Note
    "NoteProperties",
    "NoteCreate",
    "Note",
    # Search
    "SearchFilter",
    "SearchFilterGroup",
    "SearchRequest",
    "SearchResult",
    # Association
    "Association",
    "AssociationCreate",
    # Response
    "HubSpotResponse",
    "HubSpotBatchResponse",
    # Owner
    "Owner",
]

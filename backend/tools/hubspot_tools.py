"""
HubSpot LangChain Tools
=======================
LangChain tools that wrap HubSpot Service for agent use.

These tools are designed to be used by LangGraph agents.
All write operations return data for HITL approval (ADR-014).

FLOW Methodology:
- Function: Agent-callable CRM operations
- Level: Tool-spec compliant with descriptions
- Output: Structured data for agent consumption
- Win Metric: Clear tool descriptions, type-safe I/O
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Type
from uuid import uuid4

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from backend.services.hubspot_service import (
    HubSpotService,
    HubSpotError,
    HubSpotNotFoundError,
    get_hubspot_service,
)
from backend.app.schemas.hubspot import (
    ContactProperties,
    ContactCreate,
    ContactUpdate,
    DealProperties,
    DealCreate,
    DealUpdate,
    TaskProperties,
    TaskCreate,
    TaskStatus,
    TaskPriority,
    TaskType,
    NoteProperties,
    NoteCreate,
    HubSpotObjectType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL INPUT SCHEMAS
# =============================================================================

class GetContactInput(BaseModel):
    """Input for getting a contact."""
    contact_id: str = Field(..., description="HubSpot contact ID")


class SearchContactsInput(BaseModel):
    """Input for searching contacts."""
    email: str | None = Field(None, description="Filter by exact email address")
    firstname: str | None = Field(None, description="Filter by first name (contains)")
    lastname: str | None = Field(None, description="Filter by last name (contains)")
    company: str | None = Field(None, description="Filter by company name (contains)")
    limit: int = Field(default=10, description="Maximum results (1-100)", ge=1, le=100)


class CreateContactInput(BaseModel):
    """Input for creating a contact."""
    email: str = Field(..., description="Contact email address (required)")
    firstname: str | None = Field(None, description="First name")
    lastname: str | None = Field(None, description="Last name")
    phone: str | None = Field(None, description="Phone number")
    company: str | None = Field(None, description="Company name")
    jobtitle: str | None = Field(None, description="Job title")


class UpdateContactInput(BaseModel):
    """Input for updating a contact."""
    contact_id: str = Field(..., description="HubSpot contact ID to update")
    email: str | None = Field(None, description="New email address")
    firstname: str | None = Field(None, description="New first name")
    lastname: str | None = Field(None, description="New last name")
    phone: str | None = Field(None, description="New phone number")
    company: str | None = Field(None, description="New company name")
    jobtitle: str | None = Field(None, description="New job title")


class GetDealInput(BaseModel):
    """Input for getting a deal."""
    deal_id: str = Field(..., description="HubSpot deal ID")


class SearchDealsInput(BaseModel):
    """Input for searching deals."""
    dealname: str | None = Field(None, description="Filter by deal name (contains)")
    dealstage: str | None = Field(None, description="Filter by deal stage")
    pipeline: str | None = Field(None, description="Filter by pipeline ID")
    limit: int = Field(default=10, description="Maximum results (1-100)", ge=1, le=100)


class CreateDealInput(BaseModel):
    """Input for creating a deal."""
    dealname: str = Field(..., description="Deal name (required)")
    amount: str | None = Field(None, description="Deal amount")
    dealstage: str | None = Field(None, description="Deal stage")
    pipeline: str | None = Field(None, description="Pipeline ID")
    closedate: str | None = Field(None, description="Expected close date (Unix ms or ISO format)")
    description: str | None = Field(None, description="Deal description")
    contact_id: str | None = Field(None, description="Associate with contact ID")
    company_id: str | None = Field(None, description="Associate with company ID")


class UpdateDealInput(BaseModel):
    """Input for updating a deal."""
    deal_id: str = Field(..., description="HubSpot deal ID to update")
    dealname: str | None = Field(None, description="New deal name")
    amount: str | None = Field(None, description="New deal amount")
    dealstage: str | None = Field(None, description="New deal stage")
    closedate: str | None = Field(None, description="New close date")
    description: str | None = Field(None, description="New description")


class CreateTaskInput(BaseModel):
    """Input for creating a task."""
    subject: str = Field(..., description="Task subject/title (required)")
    body: str | None = Field(None, description="Task description/body")
    due_date: str | None = Field(None, description="Due date (Unix ms or ISO format)")
    priority: str = Field(default="MEDIUM", description="Priority: LOW, MEDIUM, or HIGH")
    task_type: str = Field(default="TODO", description="Type: CALL, EMAIL, or TODO")
    contact_id: str | None = Field(None, description="Associate with contact ID")
    deal_id: str | None = Field(None, description="Associate with deal ID")


class CreateNoteInput(BaseModel):
    """Input for creating a note."""
    body: str = Field(..., description="Note content (HTML allowed)")
    contact_id: str | None = Field(None, description="Associate with contact ID")
    deal_id: str | None = Field(None, description="Associate with deal ID")


class GetOwnersInput(BaseModel):
    """Input for getting owners (no params needed)."""
    pass


# =============================================================================
# READ TOOLS (No approval needed)
# =============================================================================

class GetContactTool(BaseTool):
    """Tool to get a HubSpot contact by ID."""

    name: str = "hubspot_get_contact"
    description: str = """
    Get a HubSpot contact by ID.
    Returns contact details including email, name, phone, company, and job title.
    Use this when you have a specific contact ID and need their information.
    """
    args_schema: Type[BaseModel] = GetContactInput
    return_direct: bool = False

    async def _arun(self, contact_id: str) -> str:
        """Get contact asynchronously."""
        try:
            service = get_hubspot_service()
            contact = await service.get_contact(contact_id)

            return json.dumps({
                "success": True,
                "contact": {
                    "id": contact.id,
                    "properties": contact.properties,
                    "created_at": contact.created_at.isoformat() if contact.created_at else None,
                }
            }, indent=2)

        except HubSpotNotFoundError:
            return json.dumps({
                "success": False,
                "error": f"Contact {contact_id} not found"
            })
        except HubSpotError as e:
            raise ToolException(f"HubSpot error: {e.message}")

    def _run(self, contact_id: str) -> str:
        """Sync version - raises error, use async."""
        raise NotImplementedError("Use async version (_arun)")


class SearchContactsTool(BaseTool):
    """Tool to search HubSpot contacts."""

    name: str = "hubspot_search_contacts"
    description: str = """
    Search HubSpot contacts by various criteria.
    You can filter by email (exact match), firstname, lastname, or company (contains match).
    Returns a list of matching contacts with their details.
    Use this when you need to find contacts but don't have their exact ID.
    """
    args_schema: Type[BaseModel] = SearchContactsInput
    return_direct: bool = False

    async def _arun(
        self,
        email: str | None = None,
        firstname: str | None = None,
        lastname: str | None = None,
        company: str | None = None,
        limit: int = 10,
    ) -> str:
        """Search contacts asynchronously."""
        try:
            service = get_hubspot_service()
            contacts = await service.search_contacts(
                email=email,
                firstname=firstname,
                lastname=lastname,
                company=company,
                limit=limit,
            )

            return json.dumps({
                "success": True,
                "count": len(contacts),
                "contacts": [
                    {
                        "id": c.id,
                        "properties": c.properties,
                    }
                    for c in contacts
                ]
            }, indent=2)

        except HubSpotError as e:
            raise ToolException(f"HubSpot error: {e.message}")

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async version (_arun)")


class GetDealTool(BaseTool):
    """Tool to get a HubSpot deal by ID."""

    name: str = "hubspot_get_deal"
    description: str = """
    Get a HubSpot deal by ID.
    Returns deal details including name, amount, stage, pipeline, and close date.
    Use this when you have a specific deal ID and need its information.
    """
    args_schema: Type[BaseModel] = GetDealInput
    return_direct: bool = False

    async def _arun(self, deal_id: str) -> str:
        """Get deal asynchronously."""
        try:
            service = get_hubspot_service()
            deal = await service.get_deal(deal_id)

            return json.dumps({
                "success": True,
                "deal": {
                    "id": deal.id,
                    "properties": deal.properties,
                    "created_at": deal.created_at.isoformat() if deal.created_at else None,
                }
            }, indent=2)

        except HubSpotNotFoundError:
            return json.dumps({
                "success": False,
                "error": f"Deal {deal_id} not found"
            })
        except HubSpotError as e:
            raise ToolException(f"HubSpot error: {e.message}")

    def _run(self, deal_id: str) -> str:
        raise NotImplementedError("Use async version (_arun)")


class SearchDealsTool(BaseTool):
    """Tool to search HubSpot deals."""

    name: str = "hubspot_search_deals"
    description: str = """
    Search HubSpot deals by various criteria.
    You can filter by dealname (contains), dealstage (exact), or pipeline (exact).
    Returns a list of matching deals with their details.
    """
    args_schema: Type[BaseModel] = SearchDealsInput
    return_direct: bool = False

    async def _arun(
        self,
        dealname: str | None = None,
        dealstage: str | None = None,
        pipeline: str | None = None,
        limit: int = 10,
    ) -> str:
        """Search deals asynchronously."""
        try:
            service = get_hubspot_service()
            deals = await service.search_deals(
                dealname=dealname,
                dealstage=dealstage,
                pipeline=pipeline,
                limit=limit,
            )

            return json.dumps({
                "success": True,
                "count": len(deals),
                "deals": [
                    {
                        "id": d.id,
                        "properties": d.properties,
                    }
                    for d in deals
                ]
            }, indent=2)

        except HubSpotError as e:
            raise ToolException(f"HubSpot error: {e.message}")

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async version (_arun)")


class GetOwnersTool(BaseTool):
    """Tool to get HubSpot owners (users)."""

    name: str = "hubspot_get_owners"
    description: str = """
    Get all HubSpot owners (users) in the portal.
    Returns a list of users with their IDs, emails, and names.
    Use this to find owner IDs for assigning tasks or deals.
    """
    args_schema: Type[BaseModel] = GetOwnersInput
    return_direct: bool = False

    async def _arun(self) -> str:
        """Get owners asynchronously."""
        try:
            service = get_hubspot_service()
            owners = await service.get_owners()

            return json.dumps({
                "success": True,
                "count": len(owners),
                "owners": [
                    {
                        "id": o.id,
                        "email": o.email,
                        "first_name": o.first_name,
                        "last_name": o.last_name,
                    }
                    for o in owners
                ]
            }, indent=2)

        except HubSpotError as e:
            raise ToolException(f"HubSpot error: {e.message}")

    def _run(self) -> str:
        raise NotImplementedError("Use async version (_arun)")


# =============================================================================
# WRITE TOOLS (Return data for HITL approval - ADR-014)
# =============================================================================

class PrepareCreateContactTool(BaseTool):
    """
    Tool to prepare contact creation for HITL approval.

    NOTE: This tool does NOT directly create the contact.
    It returns a payload that must be approved before execution.
    """

    name: str = "hubspot_prepare_create_contact"
    description: str = """
    Prepare a new HubSpot contact for creation.
    IMPORTANT: This does NOT create the contact immediately.
    It returns a payload that requires human approval before execution.
    Use this when you need to create a new contact in CRM.
    """
    args_schema: Type[BaseModel] = CreateContactInput
    return_direct: bool = False

    async def _arun(
        self,
        email: str,
        firstname: str | None = None,
        lastname: str | None = None,
        phone: str | None = None,
        company: str | None = None,
        jobtitle: str | None = None,
    ) -> str:
        """Prepare contact creation payload."""
        properties = ContactProperties(
            email=email,
            firstname=firstname,
            lastname=lastname,
            phone=phone,
            company=company,
            jobtitle=jobtitle,
        )

        payload = ContactCreate(properties=properties)

        return json.dumps({
            "action": "CREATE_CONTACT",
            "requires_approval": True,
            "operation_id": str(uuid4()),
            "payload": payload.model_dump(exclude_none=True),
            "summary": f"Create contact: {email}" + (f" ({firstname} {lastname})" if firstname else ""),
            "risk_factors": ["new_record"],
        }, indent=2)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async version (_arun)")


class PrepareUpdateContactTool(BaseTool):
    """Tool to prepare contact update for HITL approval."""

    name: str = "hubspot_prepare_update_contact"
    description: str = """
    Prepare a HubSpot contact update.
    IMPORTANT: This does NOT update the contact immediately.
    It returns a payload that requires human approval before execution.
    Only include fields you want to change.
    """
    args_schema: Type[BaseModel] = UpdateContactInput
    return_direct: bool = False

    async def _arun(
        self,
        contact_id: str,
        email: str | None = None,
        firstname: str | None = None,
        lastname: str | None = None,
        phone: str | None = None,
        company: str | None = None,
        jobtitle: str | None = None,
    ) -> str:
        """Prepare contact update payload."""
        # Get current state for diff
        try:
            service = get_hubspot_service()
            current = await service.get_contact(contact_id)
            current_state = current.properties
        except HubSpotNotFoundError:
            return json.dumps({
                "success": False,
                "error": f"Contact {contact_id} not found"
            })

        properties = ContactProperties(
            email=email,
            firstname=firstname,
            lastname=lastname,
            phone=phone,
            company=company,
            jobtitle=jobtitle,
        )

        payload = ContactUpdate(properties=properties)
        proposed_changes = payload.model_dump(exclude_none=True)

        # Calculate diff
        diff = []
        for key, new_value in proposed_changes.get("properties", {}).items():
            old_value = current_state.get(key)
            if old_value != new_value:
                diff.append({
                    "field": key,
                    "old": old_value,
                    "new": new_value,
                })

        return json.dumps({
            "action": "UPDATE_CONTACT",
            "requires_approval": True,
            "operation_id": str(uuid4()),
            "contact_id": contact_id,
            "current_state": current_state,
            "payload": proposed_changes,
            "diff": diff,
            "summary": f"Update contact {contact_id}: {len(diff)} field(s) changed",
            "risk_factors": ["update_existing"] + (["sensitive_field:email"] if email else []),
        }, indent=2)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async version (_arun)")


class PrepareCreateDealTool(BaseTool):
    """Tool to prepare deal creation for HITL approval."""

    name: str = "hubspot_prepare_create_deal"
    description: str = """
    Prepare a new HubSpot deal for creation.
    IMPORTANT: This does NOT create the deal immediately.
    It returns a payload that requires human approval before execution.
    """
    args_schema: Type[BaseModel] = CreateDealInput
    return_direct: bool = False

    async def _arun(
        self,
        dealname: str,
        amount: str | None = None,
        dealstage: str | None = None,
        pipeline: str | None = None,
        closedate: str | None = None,
        description: str | None = None,
        contact_id: str | None = None,
        company_id: str | None = None,
    ) -> str:
        """Prepare deal creation payload."""
        properties = DealProperties(
            dealname=dealname,
            amount=amount,
            dealstage=dealstage,
            pipeline=pipeline,
            closedate=closedate,
            description=description,
        )

        associations = []
        if contact_id:
            associations.append({
                "to": {"id": contact_id},
                "types": [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 3}]
            })
        if company_id:
            associations.append({
                "to": {"id": company_id},
                "types": [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 5}]
            })

        payload = DealCreate(
            properties=properties,
            associations=associations if associations else None,
        )

        risk_factors = ["new_record"]
        if amount:
            try:
                amt = float(amount)
                if amt > 50000:
                    risk_factors.append("high_value_deal")
                elif amt > 5000:
                    risk_factors.append("medium_value_deal")
            except ValueError:
                pass

        return json.dumps({
            "action": "CREATE_DEAL",
            "requires_approval": True,
            "operation_id": str(uuid4()),
            "payload": payload.model_dump(exclude_none=True),
            "summary": f"Create deal: {dealname}" + (f" (${amount})" if amount else ""),
            "risk_factors": risk_factors,
        }, indent=2)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async version (_arun)")


class PrepareUpdateDealTool(BaseTool):
    """Tool to prepare deal update for HITL approval."""

    name: str = "hubspot_prepare_update_deal"
    description: str = """
    Prepare a HubSpot deal update.
    IMPORTANT: This does NOT update the deal immediately.
    It returns a payload that requires human approval before execution.
    """
    args_schema: Type[BaseModel] = UpdateDealInput
    return_direct: bool = False

    async def _arun(
        self,
        deal_id: str,
        dealname: str | None = None,
        amount: str | None = None,
        dealstage: str | None = None,
        closedate: str | None = None,
        description: str | None = None,
    ) -> str:
        """Prepare deal update payload."""
        try:
            service = get_hubspot_service()
            current = await service.get_deal(deal_id)
            current_state = current.properties
        except HubSpotNotFoundError:
            return json.dumps({
                "success": False,
                "error": f"Deal {deal_id} not found"
            })

        properties = DealProperties(
            dealname=dealname,
            amount=amount,
            dealstage=dealstage,
            closedate=closedate,
            description=description,
        )

        payload = DealUpdate(properties=properties)
        proposed_changes = payload.model_dump(exclude_none=True)

        # Calculate diff
        diff = []
        risk_factors = ["update_existing"]

        for key, new_value in proposed_changes.get("properties", {}).items():
            old_value = current_state.get(key)
            if old_value != new_value:
                diff.append({
                    "field": key,
                    "old": old_value,
                    "new": new_value,
                })

                if key == "dealstage":
                    risk_factors.append("sensitive_field:dealstage")
                if key == "amount":
                    risk_factors.append("sensitive_field:amount")

        return json.dumps({
            "action": "UPDATE_DEAL",
            "requires_approval": True,
            "operation_id": str(uuid4()),
            "deal_id": deal_id,
            "current_state": current_state,
            "payload": proposed_changes,
            "diff": diff,
            "summary": f"Update deal {deal_id}: {len(diff)} field(s) changed",
            "risk_factors": risk_factors,
        }, indent=2)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async version (_arun)")


class PrepareCreateTaskTool(BaseTool):
    """Tool to prepare task creation for HITL approval."""

    name: str = "hubspot_prepare_create_task"
    description: str = """
    Prepare a new HubSpot task for creation.
    IMPORTANT: This does NOT create the task immediately.
    It returns a payload that requires human approval before execution.
    Use this for creating follow-up tasks from meetings.
    """
    args_schema: Type[BaseModel] = CreateTaskInput
    return_direct: bool = False

    async def _arun(
        self,
        subject: str,
        body: str | None = None,
        due_date: str | None = None,
        priority: str = "MEDIUM",
        task_type: str = "TODO",
        contact_id: str | None = None,
        deal_id: str | None = None,
    ) -> str:
        """Prepare task creation payload."""
        # Validate enums
        try:
            priority_enum = TaskPriority(priority.upper())
        except ValueError:
            priority_enum = TaskPriority.MEDIUM

        try:
            type_enum = TaskType(task_type.upper())
        except ValueError:
            type_enum = TaskType.TODO

        properties = TaskProperties(
            hs_task_subject=subject,
            hs_task_body=body,
            hs_task_status=TaskStatus.NOT_STARTED,
            hs_task_priority=priority_enum,
            hs_task_type=type_enum,
            hs_timestamp=due_date,
        )

        associations = []
        if contact_id:
            associations.append({
                "to": {"id": contact_id},
                "types": [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 204}]
            })
        if deal_id:
            associations.append({
                "to": {"id": deal_id},
                "types": [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 216}]
            })

        payload = TaskCreate(
            properties=properties,
            associations=associations if associations else None,
        )

        return json.dumps({
            "action": "CREATE_TASK",
            "requires_approval": True,
            "operation_id": str(uuid4()),
            "payload": payload.model_dump(exclude_none=True, mode="json"),
            "summary": f"Create task: {subject} (Priority: {priority_enum.value})",
            "risk_factors": ["new_record"],
        }, indent=2)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async version (_arun)")


class PrepareCreateNoteTool(BaseTool):
    """Tool to prepare note creation for HITL approval."""

    name: str = "hubspot_prepare_create_note"
    description: str = """
    Prepare a new HubSpot note for creation.
    IMPORTANT: This does NOT create the note immediately.
    It returns a payload that requires human approval before execution.
    Use this for adding meeting notes or summaries to contacts/deals.
    """
    args_schema: Type[BaseModel] = CreateNoteInput
    return_direct: bool = False

    async def _arun(
        self,
        body: str,
        contact_id: str | None = None,
        deal_id: str | None = None,
    ) -> str:
        """Prepare note creation payload."""
        timestamp = str(int(datetime.now(timezone.utc).timestamp() * 1000))

        properties = NoteProperties(
            hs_note_body=body,
            hs_timestamp=timestamp,
        )

        associations = []
        if contact_id:
            associations.append({
                "to": {"id": contact_id},
                "types": [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 202}]
            })
        if deal_id:
            associations.append({
                "to": {"id": deal_id},
                "types": [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 214}]
            })

        payload = NoteCreate(
            properties=properties,
            associations=associations if associations else None,
        )

        # Truncate body for summary
        summary_body = body[:100] + "..." if len(body) > 100 else body

        return json.dumps({
            "action": "CREATE_NOTE",
            "requires_approval": True,
            "operation_id": str(uuid4()),
            "payload": payload.model_dump(exclude_none=True),
            "summary": f"Create note: {summary_body}",
            "risk_factors": ["new_record"],
        }, indent=2)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async version (_arun)")


# =============================================================================
# TOOL COLLECTIONS
# =============================================================================

def get_hubspot_read_tools() -> list[BaseTool]:
    """Get all HubSpot read-only tools (no approval needed)."""
    return [
        GetContactTool(),
        SearchContactsTool(),
        GetDealTool(),
        SearchDealsTool(),
        GetOwnersTool(),
    ]


def get_hubspot_write_tools() -> list[BaseTool]:
    """Get all HubSpot write tools (require approval)."""
    return [
        PrepareCreateContactTool(),
        PrepareUpdateContactTool(),
        PrepareCreateDealTool(),
        PrepareUpdateDealTool(),
        PrepareCreateTaskTool(),
        PrepareCreateNoteTool(),
    ]


def get_all_hubspot_tools() -> list[BaseTool]:
    """Get all HubSpot tools."""
    return get_hubspot_read_tools() + get_hubspot_write_tools()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Read tools
    "GetContactTool",
    "SearchContactsTool",
    "GetDealTool",
    "SearchDealsTool",
    "GetOwnersTool",
    # Write tools (HITL)
    "PrepareCreateContactTool",
    "PrepareUpdateContactTool",
    "PrepareCreateDealTool",
    "PrepareUpdateDealTool",
    "PrepareCreateTaskTool",
    "PrepareCreateNoteTool",
    # Collections
    "get_hubspot_read_tools",
    "get_hubspot_write_tools",
    "get_all_hubspot_tools",
]

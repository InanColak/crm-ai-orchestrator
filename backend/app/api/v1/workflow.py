"""
Workflow API Endpoints
======================
Endpoints for triggering, querying, and managing AI workflows.
Implements LangGraph workflow orchestration with approval integration.

FLOW Methodology:
- Function: REST API for workflow lifecycle management
- Level: Production-ready async endpoints
- Output: Consistent JSON responses with proper error handling
- Win Metric: < 200ms response time, zero unhandled exceptions
"""

import hashlib
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from backend.app.core.dependencies import (
    ClientContextDep,
    SettingsDep,
    RateLimitDep,
)
from backend.app.schemas.workflow import (
    WorkflowType,
    WorkflowStatus,
    WorkflowTriggerRequest,
    WorkflowResumeRequest,
    WorkflowResponse,
    WorkflowDetail,
    WorkflowListResponse,
    WorkflowStats,
)
from backend.services.workflow_service import (
    get_workflow_service,
    WorkflowNotFoundError,
    WorkflowInvalidStateError,
)


def _to_uuid(value: str) -> UUID:
    """Convert string to UUID, generating deterministic UUID for non-UUID strings."""
    try:
        return UUID(value)
    except ValueError:
        # For non-UUID strings (like "dev-client-001"), generate a deterministic UUID
        hash_bytes = hashlib.md5(value.encode()).digest()
        return UUID(bytes=hash_bytes)


router = APIRouter()


# =============================================================================
# WORKFLOW TRIGGER
# =============================================================================

@router.post(
    "",
    response_model=WorkflowResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger a new workflow",
    dependencies=[RateLimitDep],
)
async def trigger_workflow(
    request: WorkflowTriggerRequest,
    client: ClientContextDep,
    settings: SettingsDep,
) -> WorkflowResponse:
    """
    Trigger a new AI workflow.

    The workflow runs asynchronously. Use the returned workflow_id
    to check status and retrieve results.

    **Available Workflow Types:**

    - `meeting_analysis`: Analyze meeting transcript, extract action items, update CRM
    - `lead_research`: Research leads and enrich CRM data
    - `content_generation`: Generate content based on intelligence data
    - `intelligence_only`: Market research, SEO analysis, audience building
    - `content_only`: Content creation and publishing pipeline
    - `sales_ops_only`: Full sales operations workflow
    - `full_cycle`: Complete end-to-end workflow (all squads)

    **Example Input (meeting_analysis):**
    ```json
    {
        "workflow_type": "meeting_analysis",
        "input_data": {
            "transcript": "Meeting transcript text...",
            "meeting_title": "Sales Call with Acme Corp",
            "participants": ["John", "Jane"],
            "deal_id": "12345"
        }
    }
    ```

    **Response:**
    Returns immediately with workflow_id. Poll `/workflows/{id}` for status.
    """
    try:
        service = await get_workflow_service()

        return await service.trigger_workflow(
            client_id=_to_uuid(client.client_id),
            client_name=client.client_name,
            request=request,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# =============================================================================
# WORKFLOW LIST
# =============================================================================

@router.get(
    "",
    response_model=WorkflowListResponse,
    summary="List workflows",
)
async def list_workflows(
    client: ClientContextDep,
    status_filter: str | None = Query(
        None,
        description="Filter by status: pending, running, completed, failed, cancelled, awaiting_approval"
    ),
    workflow_type: WorkflowType | None = Query(
        None,
        description="Filter by workflow type"
    ),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
) -> WorkflowListResponse:
    """
    List all workflows for the current client.

    Supports pagination and filtering by status and workflow type.

    **Status Values:**
    - `pending`: Queued, not started
    - `running`: Currently executing
    - `awaiting_approval`: Paused for human approval
    - `completed`: Successfully finished
    - `failed`: Execution failed
    - `cancelled`: Manually cancelled
    """
    try:
        service = await get_workflow_service()

        return await service.list_workflows(
            client_id=_to_uuid(client.client_id),
            status_filter=status_filter,
            workflow_type=workflow_type,
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# =============================================================================
# WORKFLOW TYPES (Reference) - Must be before /{workflow_id} route
# =============================================================================

@router.get(
    "/types",
    summary="List available workflow types",
)
async def list_workflow_types() -> dict:
    """
    List all available workflow types with descriptions.

    Useful for building UI dropdowns or documentation.
    """
    return {
        "types": [
            {
                "value": WorkflowType.MEETING_ANALYSIS.value,
                "label": "Meeting Analysis",
                "description": "Analyze meeting transcript, extract action items, update CRM",
                "input_fields": ["transcript", "meeting_title", "participants", "deal_id"],
            },
            {
                "value": WorkflowType.LEAD_RESEARCH.value,
                "label": "Lead Research",
                "description": "Research leads and enrich CRM data with intelligence",
                "input_fields": ["company_name", "company_domain", "contact_name", "linkedin_url"],
            },
            {
                "value": WorkflowType.CONTENT_GENERATION.value,
                "label": "Content Generation",
                "description": "Generate content based on intelligence and brand voice",
                "input_fields": ["content_type", "topic", "target_keywords", "tone"],
            },
            {
                "value": WorkflowType.INTELLIGENCE_ONLY.value,
                "label": "Intelligence Only",
                "description": "Market research, SEO analysis, audience building",
                "input_fields": ["target_domain", "keywords", "competitors"],
            },
            {
                "value": WorkflowType.CONTENT_ONLY.value,
                "label": "Content Only",
                "description": "Content creation and publishing pipeline",
                "input_fields": ["content_type", "topic", "platform"],
            },
            {
                "value": WorkflowType.SALES_OPS_ONLY.value,
                "label": "Sales Ops Only",
                "description": "Full sales operations workflow",
                "input_fields": ["transcript", "leads"],
            },
            {
                "value": WorkflowType.FULL_CYCLE.value,
                "label": "Full Cycle",
                "description": "Complete end-to-end workflow with all agent squads",
                "input_fields": ["varies"],
            },
        ]
    }


# =============================================================================
# WORKFLOW STATS
# =============================================================================

@router.get(
    "/stats",
    response_model=WorkflowStats,
    summary="Get workflow statistics",
)
async def get_workflow_stats(
    client: ClientContextDep,
) -> WorkflowStats:
    """
    Get workflow statistics for the current client.

    Returns counts by status, type, and daily metrics.
    Useful for dashboard displays.
    """
    try:
        service = await get_workflow_service()
        return await service.get_stats(_to_uuid(client.client_id))

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# =============================================================================
# WORKFLOW DETAIL
# =============================================================================

@router.get(
    "/{workflow_id}",
    response_model=WorkflowDetail,
    summary="Get workflow details",
)
async def get_workflow(
    workflow_id: str,
    client: ClientContextDep,
) -> WorkflowDetail:
    """
    Get detailed information about a specific workflow.

    **Includes:**
    - Current status and progress percentage
    - Input data and execution result
    - Steps completed with timing
    - Pending approvals (if awaiting_approval)
    - Error details (if failed)
    - LangSmith trace ID (if tracing enabled)
    """
    try:
        service = await get_workflow_service()

        return await service.get_workflow(
            workflow_id=UUID(workflow_id),
            client_id=_to_uuid(client.client_id),
        )

    except WorkflowNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# =============================================================================
# WORKFLOW RESUME
# =============================================================================

@router.post(
    "/{workflow_id}/resume",
    response_model=WorkflowResponse,
    summary="Resume a paused workflow",
)
async def resume_workflow(
    workflow_id: str,
    client: ClientContextDep,
    request: WorkflowResumeRequest | None = None,
) -> WorkflowResponse:
    """
    Resume a workflow that is paused or awaiting approval.

    **Prerequisites:**
    - Workflow must be in `awaiting_approval` or `paused` status
    - All pending approvals should be processed via `/approvals` endpoints

    **Optional Input:**
    - `approval_decisions`: Map of approval_id to decision (approved/rejected)
    - `additional_input`: Extra data to inject into resumed workflow

    The workflow continues from where it left off using LangGraph checkpointing.
    """
    try:
        service = await get_workflow_service()

        resume_request = request or WorkflowResumeRequest()

        return await service.resume_workflow(
            workflow_id=UUID(workflow_id),
            client_id=_to_uuid(client.client_id),
            request=resume_request,
        )

    except WorkflowNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found"
        )
    except WorkflowInvalidStateError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# =============================================================================
# WORKFLOW CANCEL
# =============================================================================

@router.delete(
    "/{workflow_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel a workflow",
)
async def cancel_workflow(
    workflow_id: str,
    client: ClientContextDep,
    reason: str | None = Query(None, description="Cancellation reason"),
    force: bool = Query(False, description="Force cancel running workflow"),
) -> None:
    """
    Cancel a pending or running workflow.

    **Behavior:**
    - `pending` workflows are cancelled immediately
    - `running` workflows require `force=true` to cancel
    - `completed`, `failed`, and `cancelled` workflows cannot be cancelled

    **Note:** Cancelled workflows cannot be resumed. The workflow state is preserved
    in the database for audit purposes.
    """
    try:
        service = await get_workflow_service()

        await service.cancel_workflow(
            workflow_id=UUID(workflow_id),
            client_id=_to_uuid(client.client_id),
            reason=reason,
            force=force,
        )

    except WorkflowNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found"
        )
    except WorkflowInvalidStateError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

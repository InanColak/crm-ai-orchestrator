"""
Approvals API Endpoints
=======================
Human-in-the-Loop approval system for critical actions.
CRM updates, content publishing, and email sending require approval.

Implements ADR-014: CRM HITL Control Mechanism.
"""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.app.core.dependencies import ClientContextDep, SettingsDep
from backend.app.schemas.approvals import (
    ApprovalActionResponse,
    ApprovalDecisionRequest,
    ApprovalDetail,
    ApprovalListResponse,
    ApprovalStats,
    ApprovalStatus,
    ApprovalType,
    BulkApprovalRequest,
    BulkApprovalResponse,
)
from backend.services.approval_service import (
    ApprovalAlreadyProcessedError,
    ApprovalExpiredError,
    ApprovalNotFoundError,
    get_approval_service,
)


router = APIRouter()


# =============================================================================
# LEGACY MODELS (backward compatibility)
# =============================================================================

class LegacyApprovalItem(BaseModel):
    """Legacy model for backward compatibility."""
    approval_id: str
    approval_type: str
    title: str
    description: str
    payload: dict[str, Any]
    workflow_id: str
    requested_at: str
    requested_by: str = "system"
    status: str
    reviewed_at: str | None = None
    reviewed_by: str | None = None
    rejection_reason: str | None = None


class LegacyApprovalListResponse(BaseModel):
    """Legacy response format."""
    approvals: list[LegacyApprovalItem]
    total: int
    pending_count: int


class LegacyActionRequest(BaseModel):
    """Legacy action request."""
    rejection_reason: str | None = Field(
        default=None,
        description="Required when rejecting an approval"
    )


class LegacyActionResponse(BaseModel):
    """Legacy action response."""
    approval_id: str
    status: str
    message: str
    workflow_resumed: bool = False


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get(
    "",
    response_model=ApprovalListResponse,
    summary="List approvals",
)
async def list_approvals(
    client: ClientContextDep,
    settings: SettingsDep,
    status_filter: str = Query(
        "pending",
        description="Filter by status: pending, approved, rejected, expired, all"
    ),
    approval_type: str | None = Query(
        None,
        description="Filter by approval type"
    ),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
) -> ApprovalListResponse:
    """
    List approval requests for the current client.

    By default, returns only pending approvals.
    Use status_filter to see approved/rejected items.

    **Filters:**
    - `status_filter`: pending, approved, rejected, expired, all
    - `approval_type`: crm_create_contact, crm_update_deal, email_send, etc.
    """
    try:
        service = await get_approval_service()

        # Parse approval type if provided
        parsed_type = None
        if approval_type:
            try:
                parsed_type = ApprovalType(approval_type)
            except ValueError:
                pass  # Ignore invalid types

        return await service.list_approvals(
            client_id=UUID(client.client_id),
            status_filter=status_filter if status_filter != "all" else None,
            approval_type=parsed_type,
            page=page,
            page_size=page_size
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    "/stats",
    response_model=ApprovalStats,
    summary="Get approval statistics",
)
async def get_approval_stats(
    client: ClientContextDep,
) -> ApprovalStats:
    """
    Get approval statistics for the current client.

    Returns counts by status, type, and priority.
    """
    try:
        service = await get_approval_service()
        return await service.get_stats(UUID(client.client_id))

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    "/{approval_id}",
    response_model=ApprovalDetail,
    summary="Get approval details",
)
async def get_approval(
    approval_id: str,
    client: ClientContextDep,
) -> ApprovalDetail:
    """
    Get detailed information about a specific approval request.

    Includes:
    - Full payload that will be executed
    - Diff before/after for updates
    - Risk assessment
    - Execution results (if approved)
    """
    try:
        service = await get_approval_service()
        return await service.get_approval(
            approval_id=UUID(approval_id),
            client_id=UUID(client.client_id)
        )

    except ApprovalNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Approval '{approval_id}' not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post(
    "/{approval_id}/approve",
    response_model=ApprovalActionResponse,
    summary="Approve an action",
)
async def approve_action(
    approval_id: str,
    client: ClientContextDep,
    reviewed_by: str = Query(
        None,
        description="Name/email of reviewer (optional, defaults to client name)"
    ),
) -> ApprovalActionResponse:
    """
    Approve a pending action.

    This will:
    1. Mark the approval as approved
    2. Execute the pending action (CRM update, email send, etc.)
    3. Resume the associated workflow if applicable

    **ADR-014 Compliance:**
    - All CRM write operations are executed only after explicit approval
    - Execution results are logged for audit
    """
    try:
        service = await get_approval_service()

        decision = ApprovalDecisionRequest(
            reviewed_by=reviewed_by or client.client_name
        )

        return await service.approve(
            approval_id=UUID(approval_id),
            client_id=UUID(client.client_id),
            decision=decision
        )

    except ApprovalNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Approval '{approval_id}' not found"
        )
    except ApprovalAlreadyProcessedError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ApprovalExpiredError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post(
    "/{approval_id}/reject",
    response_model=ApprovalActionResponse,
    summary="Reject an action",
)
async def reject_action(
    approval_id: str,
    client: ClientContextDep,
    rejection_reason: str | None = Query(
        None,
        description="Reason for rejection"
    ),
    reviewed_by: str = Query(
        None,
        description="Name/email of reviewer"
    ),
) -> ApprovalActionResponse:
    """
    Reject a pending action.

    The rejection reason is optional but recommended for audit purposes.
    The associated workflow will be notified of the rejection.
    """
    try:
        service = await get_approval_service()

        decision = ApprovalDecisionRequest(
            reviewed_by=reviewed_by or client.client_name,
            rejection_reason=rejection_reason
        )

        return await service.reject(
            approval_id=UUID(approval_id),
            client_id=UUID(client.client_id),
            decision=decision
        )

    except ApprovalNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Approval '{approval_id}' not found"
        )
    except ApprovalAlreadyProcessedError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ApprovalExpiredError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post(
    "/bulk/approve",
    response_model=BulkApprovalResponse,
    summary="Bulk approve actions",
)
async def bulk_approve(
    approval_ids: list[str],
    client: ClientContextDep,
    reviewed_by: str = Query(
        None,
        description="Name/email of reviewer"
    ),
) -> BulkApprovalResponse:
    """
    Approve multiple pending actions at once.

    Returns a summary of successful and failed approvals.
    Each approval is processed independently.
    """
    try:
        service = await get_approval_service()

        request = BulkApprovalRequest(
            approval_ids=[UUID(aid) for aid in approval_ids],
            reviewed_by=reviewed_by or client.client_name
        )

        return await service.bulk_approve(
            request=request,
            client_id=UUID(client.client_id)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post(
    "/bulk/reject",
    response_model=BulkApprovalResponse,
    summary="Bulk reject actions",
)
async def bulk_reject(
    approval_ids: list[str],
    request: LegacyActionRequest,
    client: ClientContextDep,
    reviewed_by: str = Query(
        None,
        description="Name/email of reviewer"
    ),
) -> BulkApprovalResponse:
    """
    Reject multiple pending actions at once.

    All rejections will use the same rejection reason.
    """
    try:
        service = await get_approval_service()

        bulk_request = BulkApprovalRequest(
            approval_ids=[UUID(aid) for aid in approval_ids],
            reviewed_by=reviewed_by or client.client_name,
            rejection_reason=request.rejection_reason
        )

        return await service.bulk_reject(
            request=bulk_request,
            client_id=UUID(client.client_id)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# =============================================================================
# WORKFLOW INTEGRATION
# =============================================================================

@router.get(
    "/workflow/{workflow_id}",
    response_model=list[ApprovalDetail],
    summary="Get approvals for workflow",
)
async def get_workflow_approvals(
    workflow_id: str,
    client: ClientContextDep,
    status_filter: str = Query(
        "all",
        description="Filter by status"
    ),
) -> list[ApprovalDetail]:
    """
    Get all approvals associated with a workflow.

    Useful for checking if a workflow is blocked waiting for approvals.
    """
    try:
        service = await get_approval_service()

        # Get all approvals for this workflow
        result = await service.list_approvals(
            client_id=UUID(client.client_id),
            status_filter=status_filter if status_filter != "all" else None,
            page=1,
            page_size=100
        )

        # Filter by workflow_id
        workflow_uuid = UUID(workflow_id)
        return [
            await service.get_approval(item.id, UUID(client.client_id))
            for item in result.items
            if item.workflow_id == workflow_uuid
        ]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

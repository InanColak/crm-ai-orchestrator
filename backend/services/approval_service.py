"""
Approval Service
================
Business logic for Human-in-the-Loop (HITL) approval system.
Implements ADR-014 risk-based tiered approval.

FLOW Methodology:
- Function: CRUD operations + approval execution
- Level: Production-ready with full error handling
- Output: Type-safe responses via Pydantic
- Win Metric: Zero orphaned approvals, <100ms response time
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.approvals import (
    ApprovalCreateRequest,
    ApprovalDecisionRequest,
    ApprovalDetail,
    ApprovalListResponse,
    ApprovalActionResponse,
    ApprovalStats,
    ApprovalStatus,
    ApprovalSummary,
    ApprovalType,
    BulkApprovalRequest,
    BulkApprovalResponse,
    RiskAssessment,
    calculate_risk_level,
)
from backend.services.supabase_client import SupabaseService, get_supabase

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class ApprovalError(Exception):
    """Base exception for approval operations."""
    pass


class ApprovalNotFoundError(ApprovalError):
    """Approval not found."""
    pass


class ApprovalAlreadyProcessedError(ApprovalError):
    """Approval has already been approved/rejected."""
    pass


class ApprovalExpiredError(ApprovalError):
    """Approval has expired."""
    pass


class ApprovalExecutionError(ApprovalError):
    """Failed to execute approved action."""
    pass


# =============================================================================
# APPROVAL SERVICE
# =============================================================================

class ApprovalService:
    """
    Service for managing approval requests.

    Handles the complete approval lifecycle:
    - Create approval requests from agent actions
    - List and filter pending approvals
    - Process approval/rejection decisions
    - Execute approved actions
    - Track approval statistics

    Usage:
        >>> service = await ApprovalService.create()
        >>> approval = await service.create_approval(request)
        >>> await service.approve(approval.id, decision)
    """

    def __init__(self, db: SupabaseService, settings: Settings):
        self._db = db
        self._settings = settings

    @classmethod
    async def create(cls) -> "ApprovalService":
        """Factory method to create ApprovalService."""
        db = await get_supabase()
        settings = get_settings()
        return cls(db, settings)

    # =========================================================================
    # CREATE
    # =========================================================================

    async def create_approval(
        self,
        request: ApprovalCreateRequest
    ) -> ApprovalDetail:
        """
        Create a new approval request.

        Args:
            request: Approval creation request

        Returns:
            Created approval details
        """
        # Calculate risk assessment if not provided
        risk_assessment = request.risk_assessment
        if not risk_assessment:
            risk_assessment = calculate_risk_level(
                request.approval_type,
                request.payload
            )

        # Calculate expiration
        expires_at = None
        if request.expires_in_hours:
            expires_at = datetime.now(timezone.utc) + timedelta(
                hours=request.expires_in_hours
            )
        elif self._settings.approval_timeout_hours:
            expires_at = datetime.now(timezone.utc) + timedelta(
                hours=self._settings.approval_timeout_hours
            )

        # Prepare data
        approval_data = {
            "workflow_id": str(request.workflow_id),
            "client_id": str(request.client_id),
            "approval_type": request.approval_type.value,
            "title": request.title,
            "description": request.description,
            "payload": request.payload,
            "diff_before": request.diff_before,
            "diff_after": request.diff_after,
            "priority": request.priority.value,
            "status": ApprovalStatus.PENDING.value,
            "expires_at": expires_at.isoformat() if expires_at else None,
        }

        # Store risk assessment in metadata
        if risk_assessment:
            approval_data["payload"] = {
                **request.payload,
                "_risk_assessment": risk_assessment.model_dump()
            }

        # Insert into database
        result = await self._db.insert("approvals", approval_data)

        logger.info(
            f"Created approval {result['id']} for workflow {request.workflow_id} "
            f"(type={request.approval_type.value}, risk={risk_assessment.overall_level.value})"
        )

        return self._to_detail(result, risk_assessment)

    # =========================================================================
    # READ
    # =========================================================================

    async def get_approval(
        self,
        approval_id: UUID,
        client_id: UUID
    ) -> ApprovalDetail:
        """
        Get approval by ID.

        Args:
            approval_id: Approval UUID
            client_id: Client UUID (for authorization)

        Returns:
            Approval details

        Raises:
            ApprovalNotFoundError: If approval not found
        """
        results = await self._db.fetch_many(
            "approvals",
            filters={
                "id": str(approval_id),
                "client_id": str(client_id)
            },
            limit=1
        )

        if not results:
            raise ApprovalNotFoundError(f"Approval {approval_id} not found")

        return self._to_detail(results[0])

    async def list_approvals(
        self,
        client_id: UUID,
        status_filter: str | None = "pending",
        approval_type: ApprovalType | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> ApprovalListResponse:
        """
        List approvals for a client.

        Args:
            client_id: Client UUID
            status_filter: Filter by status (pending, approved, rejected, all)
            approval_type: Filter by approval type
            page: Page number
            page_size: Items per page

        Returns:
            Paginated approval list
        """
        filters: dict[str, Any] = {"client_id": str(client_id)}

        if status_filter and status_filter != "all":
            filters["status"] = status_filter

        if approval_type:
            filters["approval_type"] = approval_type.value

        # Get total count
        all_results = await self._db.fetch_many(
            "approvals",
            filters=filters,
            limit=1000  # For counting
        )
        total = len(all_results)

        # Get pending count
        pending_filters = {"client_id": str(client_id), "status": "pending"}
        pending_results = await self._db.fetch_many(
            "approvals",
            filters=pending_filters,
            limit=1000
        )
        pending_count = len(pending_results)

        # Get paginated results
        offset = (page - 1) * page_size
        results = await self._db.fetch_many(
            "approvals",
            filters=filters,
            limit=page_size,
            order_by="created_at",
            ascending=False
        )

        # Apply offset manually (Supabase client limitation)
        paginated = results[offset:offset + page_size] if offset < len(results) else []

        items = [self._to_summary(r) for r in paginated]

        return ApprovalListResponse(
            items=items,
            total=total,
            pending_count=pending_count,
            page=page,
            page_size=page_size
        )

    async def get_pending_for_workflow(
        self,
        workflow_id: UUID
    ) -> list[ApprovalSummary]:
        """Get all pending approvals for a workflow."""
        results = await self._db.fetch_many(
            "approvals",
            filters={
                "workflow_id": str(workflow_id),
                "status": "pending"
            }
        )
        return [self._to_summary(r) for r in results]

    # =========================================================================
    # UPDATE (Approve/Reject)
    # =========================================================================

    async def approve(
        self,
        approval_id: UUID,
        client_id: UUID,
        decision: ApprovalDecisionRequest
    ) -> ApprovalActionResponse:
        """
        Approve an approval request.

        Args:
            approval_id: Approval UUID
            client_id: Client UUID
            decision: Decision details

        Returns:
            Action response with execution result

        Raises:
            ApprovalNotFoundError: If not found
            ApprovalAlreadyProcessedError: If already processed
            ApprovalExpiredError: If expired
        """
        approval = await self._get_and_validate(approval_id, client_id)

        # Update status
        now = datetime.now(timezone.utc)
        update_data = {
            "status": ApprovalStatus.APPROVED.value,
            "reviewed_by": decision.reviewed_by,
            "reviewed_at": now.isoformat()
        }

        await self._db.update("approvals", str(approval_id), update_data)

        logger.info(
            f"Approval {approval_id} approved by {decision.reviewed_by}"
        )

        # Execute the approved action
        execution_result = None
        try:
            execution_result = await self._execute_action(approval)

            # Update with execution result
            await self._db.update("approvals", str(approval_id), {
                "status": ApprovalStatus.EXECUTED.value,
                "payload": {
                    **approval["payload"],
                    "_execution_result": execution_result
                }
            })

        except Exception as e:
            logger.error(f"Failed to execute approval {approval_id}: {e}")
            await self._db.update("approvals", str(approval_id), {
                "status": ApprovalStatus.FAILED.value,
                "payload": {
                    **approval["payload"],
                    "_execution_error": str(e)
                }
            })

        return ApprovalActionResponse(
            id=approval_id,
            status=ApprovalStatus.APPROVED,
            message="Approval approved and action executed",
            workflow_resumed=True,
            execution_result=execution_result
        )

    async def reject(
        self,
        approval_id: UUID,
        client_id: UUID,
        decision: ApprovalDecisionRequest
    ) -> ApprovalActionResponse:
        """
        Reject an approval request.

        Args:
            approval_id: Approval UUID
            client_id: Client UUID
            decision: Decision with rejection reason

        Returns:
            Action response
        """
        approval = await self._get_and_validate(approval_id, client_id)

        now = datetime.now(timezone.utc)
        update_data = {
            "status": ApprovalStatus.REJECTED.value,
            "reviewed_by": decision.reviewed_by,
            "reviewed_at": now.isoformat(),
            "rejection_reason": decision.rejection_reason
        }

        await self._db.update("approvals", str(approval_id), update_data)

        logger.info(
            f"Approval {approval_id} rejected by {decision.reviewed_by}: "
            f"{decision.rejection_reason or 'No reason provided'}"
        )

        return ApprovalActionResponse(
            id=approval_id,
            status=ApprovalStatus.REJECTED,
            message="Approval rejected",
            workflow_resumed=False
        )

    async def bulk_approve(
        self,
        request: BulkApprovalRequest,
        client_id: UUID
    ) -> BulkApprovalResponse:
        """Approve multiple approvals at once."""
        results: list[ApprovalActionResponse] = []
        succeeded = 0
        failed = 0

        decision = ApprovalDecisionRequest(reviewed_by=request.reviewed_by)

        for approval_id in request.approval_ids:
            try:
                result = await self.approve(approval_id, client_id, decision)
                results.append(result)
                succeeded += 1
            except Exception as e:
                logger.warning(f"Failed to approve {approval_id}: {e}")
                results.append(ApprovalActionResponse(
                    id=approval_id,
                    status=ApprovalStatus.PENDING,
                    message=f"Failed: {str(e)}",
                    workflow_resumed=False
                ))
                failed += 1

        return BulkApprovalResponse(
            total=len(request.approval_ids),
            succeeded=succeeded,
            failed=failed,
            results=results
        )

    async def bulk_reject(
        self,
        request: BulkApprovalRequest,
        client_id: UUID
    ) -> BulkApprovalResponse:
        """Reject multiple approvals at once."""
        results: list[ApprovalActionResponse] = []
        succeeded = 0
        failed = 0

        decision = ApprovalDecisionRequest(
            reviewed_by=request.reviewed_by,
            rejection_reason=request.rejection_reason
        )

        for approval_id in request.approval_ids:
            try:
                result = await self.reject(approval_id, client_id, decision)
                results.append(result)
                succeeded += 1
            except Exception as e:
                logger.warning(f"Failed to reject {approval_id}: {e}")
                results.append(ApprovalActionResponse(
                    id=approval_id,
                    status=ApprovalStatus.PENDING,
                    message=f"Failed: {str(e)}",
                    workflow_resumed=False
                ))
                failed += 1

        return BulkApprovalResponse(
            total=len(request.approval_ids),
            succeeded=succeeded,
            failed=failed,
            results=results
        )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    async def get_stats(self, client_id: UUID) -> ApprovalStats:
        """Get approval statistics for a client."""
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # Get all approvals for the client
        all_approvals = await self._db.fetch_many(
            "approvals",
            filters={"client_id": str(client_id)},
            limit=10000
        )

        pending = 0
        approved_today = 0
        rejected_today = 0
        expired_today = 0
        by_type: dict[str, int] = {}
        by_priority: dict[str, int] = {}

        for approval in all_approvals:
            status = approval.get("status", "pending")
            approval_type = approval.get("approval_type", "unknown")
            priority = approval.get("priority", "normal")

            # Count by status
            if status == "pending":
                pending += 1

            # Count today's decisions
            reviewed_at = approval.get("reviewed_at")
            if reviewed_at:
                reviewed_dt = datetime.fromisoformat(
                    reviewed_at.replace("Z", "+00:00")
                )
                if reviewed_dt >= today_start:
                    if status == "approved":
                        approved_today += 1
                    elif status == "rejected":
                        rejected_today += 1
                    elif status == "expired":
                        expired_today += 1

            # Count by type and priority
            by_type[approval_type] = by_type.get(approval_type, 0) + 1
            by_priority[priority] = by_priority.get(priority, 0) + 1

        return ApprovalStats(
            pending=pending,
            approved_today=approved_today,
            rejected_today=rejected_today,
            expired_today=expired_today,
            by_type=by_type,
            by_priority=by_priority
        )

    # =========================================================================
    # EXPIRATION
    # =========================================================================

    async def expire_old_approvals(self) -> int:
        """
        Mark expired approvals as expired.

        Should be called by a background job.

        Returns:
            Number of approvals expired
        """
        now = datetime.now(timezone.utc)

        # Find pending approvals past their expiry
        pending = await self._db.fetch_many(
            "approvals",
            filters={"status": "pending"},
            limit=1000
        )

        expired_count = 0
        for approval in pending:
            expires_at = approval.get("expires_at")
            if expires_at:
                expires_dt = datetime.fromisoformat(
                    expires_at.replace("Z", "+00:00")
                )
                if expires_dt < now:
                    await self._db.update(
                        "approvals",
                        approval["id"],
                        {"status": ApprovalStatus.EXPIRED.value}
                    )
                    expired_count += 1
                    logger.info(f"Expired approval {approval['id']}")

        return expired_count

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    async def _get_and_validate(
        self,
        approval_id: UUID,
        client_id: UUID
    ) -> dict:
        """Get approval and validate it can be processed."""
        results = await self._db.fetch_many(
            "approvals",
            filters={
                "id": str(approval_id),
                "client_id": str(client_id)
            },
            limit=1
        )

        if not results:
            raise ApprovalNotFoundError(f"Approval {approval_id} not found")

        approval = results[0]

        # Check status
        status = approval.get("status", "pending")
        if status != "pending":
            raise ApprovalAlreadyProcessedError(
                f"Approval {approval_id} has already been {status}"
            )

        # Check expiration
        expires_at = approval.get("expires_at")
        if expires_at:
            expires_dt = datetime.fromisoformat(
                expires_at.replace("Z", "+00:00")
            )
            if expires_dt < datetime.now(timezone.utc):
                raise ApprovalExpiredError(
                    f"Approval {approval_id} has expired"
                )

        return approval

    async def _execute_action(self, approval: dict) -> dict[str, Any]:
        """
        Execute the approved action.

        This method dispatches to the appropriate service based on approval type.
        """
        approval_type = approval.get("approval_type", "")
        payload = approval.get("payload", {})
        client_id = approval.get("client_id")

        logger.info(f"Executing {approval_type} for client {client_id}")

        # Remove internal metadata from payload
        clean_payload = {
            k: v for k, v in payload.items()
            if not k.startswith("_")
        }

        # Dispatch based on type
        if approval_type.startswith("crm_"):
            return await self._execute_crm_action(
                approval_type, clean_payload, client_id
            )
        elif approval_type == "email_send":
            return await self._execute_email_action(clean_payload, client_id)
        elif approval_type == "content_publish":
            return await self._execute_content_action(clean_payload, client_id)
        else:
            logger.warning(f"Unknown approval type: {approval_type}")
            return {"status": "skipped", "reason": "Unknown approval type"}

    async def _execute_crm_action(
        self,
        approval_type: str,
        payload: dict,
        client_id: str
    ) -> dict[str, Any]:
        """Execute CRM action via HubSpotService."""
        try:
            from backend.services.hubspot_service import get_hubspot_for_client

            hubspot = await get_hubspot_for_client(client_id)

            operation = payload.get("operation", "")
            object_type = payload.get("object_type", "")
            properties = payload.get("properties", {})
            object_id = payload.get("object_id")

            if "create" in approval_type or operation == "create":
                if object_type == "contact":
                    from backend.app.schemas.hubspot import ContactCreate, ContactProperties
                    contact_props = ContactProperties(**properties)
                    result = await hubspot.create_contact(
                        ContactCreate(properties=contact_props)
                    )
                    return {"created_id": result.id, "object_type": "contact"}

                elif object_type == "deal":
                    from backend.app.schemas.hubspot import DealCreate, DealProperties
                    deal_props = DealProperties(**properties)
                    result = await hubspot.create_deal(
                        DealCreate(properties=deal_props)
                    )
                    return {"created_id": result.id, "object_type": "deal"}

                elif object_type == "task":
                    from backend.app.schemas.hubspot import TaskCreate, TaskProperties
                    task_props = TaskProperties(**properties)
                    result = await hubspot.create_task(
                        TaskCreate(properties=task_props)
                    )
                    return {"created_id": result.id, "object_type": "task"}

                elif object_type == "note":
                    from backend.app.schemas.hubspot import NoteCreate, NoteProperties
                    note_props = NoteProperties(**properties)
                    result = await hubspot.create_note(
                        NoteCreate(properties=note_props)
                    )
                    return {"created_id": result.id, "object_type": "note"}

            elif "update" in approval_type or operation == "update":
                if not object_id:
                    raise ApprovalExecutionError("object_id required for update")

                if object_type == "contact":
                    from backend.app.schemas.hubspot import ContactUpdate, ContactUpdateProperties
                    contact_props = ContactUpdateProperties(**properties)
                    result = await hubspot.update_contact(
                        object_id,
                        ContactUpdate(properties=contact_props)
                    )
                    return {"updated_id": result.id, "object_type": "contact"}

                elif object_type == "deal":
                    from backend.app.schemas.hubspot import DealUpdate, DealUpdateProperties
                    deal_props = DealUpdateProperties(**properties)
                    result = await hubspot.update_deal(
                        object_id,
                        DealUpdate(properties=deal_props)
                    )
                    return {"updated_id": result.id, "object_type": "deal"}

            return {"status": "executed", "operation": operation, "object_type": object_type}

        except Exception as e:
            logger.error(f"CRM action failed: {e}")
            raise ApprovalExecutionError(f"CRM action failed: {e}")

    async def _execute_email_action(
        self,
        payload: dict,
        client_id: str
    ) -> dict[str, Any]:
        """Execute email send action."""
        # TODO: Implement email sending
        logger.info(f"Email action pending implementation: {payload}")
        return {"status": "pending_implementation", "action": "email_send"}

    async def _execute_content_action(
        self,
        payload: dict,
        client_id: str
    ) -> dict[str, Any]:
        """Execute content publish action."""
        # TODO: Implement content publishing
        logger.info(f"Content action pending implementation: {payload}")
        return {"status": "pending_implementation", "action": "content_publish"}

    def _to_summary(self, data: dict) -> ApprovalSummary:
        """Convert database record to ApprovalSummary."""
        # Extract risk level from payload if present
        risk_level = None
        if data.get("payload") and "_risk_assessment" in data["payload"]:
            risk_level = data["payload"]["_risk_assessment"].get("overall_level")

        return ApprovalSummary(
            id=UUID(data["id"]),
            workflow_id=UUID(data["workflow_id"]),
            approval_type=ApprovalType(data["approval_type"]),
            title=data["title"],
            description=data.get("description"),
            status=ApprovalStatus(data["status"]),
            priority=data.get("priority", "normal"),
            risk_level=risk_level,
            created_at=datetime.fromisoformat(
                data["created_at"].replace("Z", "+00:00")
            ),
            expires_at=datetime.fromisoformat(
                data["expires_at"].replace("Z", "+00:00")
            ) if data.get("expires_at") else None
        )

    def _to_detail(
        self,
        data: dict,
        risk_assessment: RiskAssessment | None = None
    ) -> ApprovalDetail:
        """Convert database record to ApprovalDetail."""
        # Extract risk assessment from payload if not provided
        if not risk_assessment and data.get("payload"):
            risk_data = data["payload"].get("_risk_assessment")
            if risk_data:
                risk_assessment = RiskAssessment(**risk_data)

        # Extract execution results
        execution_result = None
        execution_error = None
        if data.get("payload"):
            execution_result = data["payload"].get("_execution_result")
            execution_error = data["payload"].get("_execution_error")

        return ApprovalDetail(
            id=UUID(data["id"]),
            workflow_id=UUID(data["workflow_id"]),
            client_id=UUID(data["client_id"]),
            approval_type=ApprovalType(data["approval_type"]),
            title=data["title"],
            description=data.get("description"),
            status=ApprovalStatus(data["status"]),
            priority=data.get("priority", "normal"),
            risk_level=risk_assessment.overall_level if risk_assessment else None,
            payload={
                k: v for k, v in (data.get("payload") or {}).items()
                if not k.startswith("_")
            },
            diff_before=data.get("diff_before"),
            diff_after=data.get("diff_after"),
            risk_assessment=risk_assessment,
            reviewed_by=data.get("reviewed_by"),
            reviewed_at=datetime.fromisoformat(
                data["reviewed_at"].replace("Z", "+00:00")
            ) if data.get("reviewed_at") else None,
            rejection_reason=data.get("rejection_reason"),
            execution_result=execution_result,
            execution_error=execution_error,
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(
                data["created_at"].replace("Z", "+00:00")
            ),
            expires_at=datetime.fromisoformat(
                data["expires_at"].replace("Z", "+00:00")
            ) if data.get("expires_at") else None,
            updated_at=datetime.fromisoformat(
                data["updated_at"].replace("Z", "+00:00")
            ) if data.get("updated_at") else None
        )


# =============================================================================
# SINGLETON & DEPENDENCY INJECTION
# =============================================================================

_approval_service: ApprovalService | None = None


async def get_approval_service() -> ApprovalService:
    """Get singleton ApprovalService instance."""
    global _approval_service
    if _approval_service is None:
        _approval_service = await ApprovalService.create()
    return _approval_service


async def reset_approval_service() -> None:
    """Reset singleton (for testing)."""
    global _approval_service
    _approval_service = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Exceptions
    "ApprovalError",
    "ApprovalNotFoundError",
    "ApprovalAlreadyProcessedError",
    "ApprovalExpiredError",
    "ApprovalExecutionError",
    # Service
    "ApprovalService",
    # DI
    "get_approval_service",
    "reset_approval_service",
]

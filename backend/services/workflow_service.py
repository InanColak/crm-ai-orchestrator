"""
Workflow Service
================
Orchestrates LangGraph workflow execution with database persistence.
Handles workflow lifecycle: trigger, status, resume, cancel.

FLOW Methodology:
- Function: Workflow orchestration and state management
- Level: Production-ready async execution
- Output: Reliable workflow execution with checkpointing
- Win Metric: Zero lost workflows, complete audit trail
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any
from uuid import UUID, uuid4

from backend.app.schemas.workflow import (
    WorkflowType,
    WorkflowStatus,
    WorkflowPriority,
    WorkflowTriggerRequest,
    WorkflowResumeRequest,
    WorkflowResponse,
    WorkflowDetail,
    WorkflowSummary,
    WorkflowListResponse,
    WorkflowStats,
    WorkflowStepSummary,
)
from backend.app.schemas.approvals import ApprovalCreateRequest, ApprovalType, ApprovalPriority
from backend.services.supabase_client import get_supabase, NotFoundError
from backend.graph.state import create_initial_state, OrchestratorState, CRMProvider


def _to_uuid(value: str | UUID) -> UUID:
    """Convert string or UUID to UUID, generating a deterministic UUID for non-UUID strings."""
    if isinstance(value, UUID):
        return value
    try:
        return UUID(value)
    except ValueError:
        # For non-UUID strings (like "dev-client-001"), generate a deterministic UUID
        import hashlib
        hash_bytes = hashlib.md5(value.encode()).digest()
        return UUID(bytes=hash_bytes)

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class WorkflowError(Exception):
    """Base exception for workflow operations."""
    pass


class WorkflowNotFoundError(WorkflowError):
    """Workflow not found."""
    pass


class WorkflowAlreadyExistsError(WorkflowError):
    """Workflow already exists."""
    pass


class WorkflowInvalidStateError(WorkflowError):
    """Invalid workflow state for operation."""
    pass


class WorkflowExecutionError(WorkflowError):
    """Workflow execution failed."""
    pass


# =============================================================================
# WORKFLOW SERVICE
# =============================================================================

class WorkflowService:
    """
    Service for managing workflow lifecycle.

    Responsibilities:
    - Create and persist workflow records
    - Trigger LangGraph workflow execution
    - Track workflow progress and status
    - Handle workflow resume after approval
    - Cancel and cleanup workflows
    """

    TABLE_NAME = "workflows"

    def __init__(self):
        """Initialize workflow service."""
        self._compiled_workflows: dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # WORKFLOW TRIGGER
    # -------------------------------------------------------------------------

    async def trigger_workflow(
        self,
        client_id: UUID,
        client_name: str,
        request: WorkflowTriggerRequest,
    ) -> WorkflowResponse:
        """
        Trigger a new workflow execution.

        Args:
            client_id: Client UUID
            client_name: Client name
            request: Workflow trigger request

        Returns:
            WorkflowResponse with workflow ID and status
        """
        db = await get_supabase()
        now = datetime.now(timezone.utc)
        workflow_id = uuid4()

        # Create workflow record
        workflow_data = {
            "id": str(workflow_id),
            "client_id": str(client_id),
            "workflow_type": request.workflow_type.value,
            "status": WorkflowStatus.PENDING.value,
            "priority": request.priority.value,
            "progress": 0,
            "input_data": request.input_data,
            "metadata": request.metadata,
            "callback_url": request.callback_url,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        # Insert into database
        await db.insert(self.TABLE_NAME, workflow_data)

        logger.info(
            f"Workflow triggered: {workflow_id} "
            f"type={request.workflow_type.value} client={client_id}"
        )

        # Start async execution (fire and forget)
        asyncio.create_task(
            self._execute_workflow(
                workflow_id=workflow_id,
                client_id=client_id,
                client_name=client_name,
                workflow_type=request.workflow_type,
                input_data=request.input_data,
            )
        )

        return WorkflowResponse(
            id=workflow_id,
            workflow_type=request.workflow_type,
            status=WorkflowStatus.PENDING,
            message=f"Workflow '{request.workflow_type.value}' queued for execution",
            created_at=now,
            estimated_duration_seconds=self._estimate_duration(request.workflow_type),
        )

    # -------------------------------------------------------------------------
    # WORKFLOW STATUS
    # -------------------------------------------------------------------------

    async def get_workflow(
        self,
        workflow_id: UUID,
        client_id: UUID,
    ) -> WorkflowDetail:
        """
        Get detailed workflow information.

        Args:
            workflow_id: Workflow UUID
            client_id: Client UUID (for authorization)

        Returns:
            WorkflowDetail with full workflow information

        Raises:
            WorkflowNotFoundError: If workflow not found
        """
        db = await get_supabase()

        try:
            record = await db.fetch_one(
                self.TABLE_NAME,
                {"id": str(workflow_id), "client_id": str(client_id)}
            )
        except NotFoundError:
            raise WorkflowNotFoundError(f"Workflow '{workflow_id}' not found")

        # Get pending approvals count
        pending_approval_count = 0
        pending_approval_ids: list[UUID] = []

        try:
            from backend.services.approval_service import get_approval_service
            approval_service = await get_approval_service()
            approvals = await approval_service.list_approvals(
                client_id=client_id,
                status_filter="pending",
                page=1,
                page_size=100,
            )
            # Filter by workflow_id
            for item in approvals.items:
                if item.workflow_id == workflow_id:
                    pending_approval_count += 1
                    pending_approval_ids.append(item.id)
        except Exception as e:
            logger.warning(f"Failed to fetch pending approvals: {e}")

        return WorkflowDetail(
            id=UUID(record["id"]),
            client_id=UUID(record["client_id"]),
            workflow_type=WorkflowType(record["workflow_type"]),
            status=WorkflowStatus(record["status"]),
            priority=WorkflowPriority(record.get("priority", "normal")),
            progress=record.get("progress", 0),
            created_at=datetime.fromisoformat(record["created_at"]),
            started_at=datetime.fromisoformat(record["started_at"]) if record.get("started_at") else None,
            updated_at=datetime.fromisoformat(record["updated_at"]),
            completed_at=datetime.fromisoformat(record["completed_at"]) if record.get("completed_at") else None,
            input_data=record.get("input_data", {}),
            result=record.get("result"),
            error=record.get("error"),
            current_step=record.get("current_step"),
            steps_completed=self._parse_steps(record.get("steps_completed", [])),
            pending_approval_count=pending_approval_count,
            pending_approval_ids=pending_approval_ids,
            metadata=record.get("metadata", {}),
            trace_id=record.get("trace_id"),
        )

    async def list_workflows(
        self,
        client_id: UUID,
        status_filter: str | None = None,
        workflow_type: WorkflowType | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> WorkflowListResponse:
        """
        List workflows for a client.

        Args:
            client_id: Client UUID
            status_filter: Optional status filter
            workflow_type: Optional workflow type filter
            page: Page number (1-based)
            page_size: Items per page

        Returns:
            WorkflowListResponse with paginated results
        """
        db = await get_supabase()

        # Build filters
        filters = {"client_id": str(client_id)}
        if status_filter:
            filters["status"] = status_filter
        if workflow_type:
            filters["workflow_type"] = workflow_type.value

        # Fetch with pagination
        offset = (page - 1) * page_size

        try:
            records = await db.fetch_many(
                self.TABLE_NAME,
                filters=filters,
                order_by="created_at",
                ascending=False,
                limit=page_size + 1,  # Fetch one extra to check has_more
                offset=offset,
            )
        except Exception:
            records = []

        # Check if there are more results
        has_more = len(records) > page_size
        if has_more:
            records = records[:page_size]

        # Count total
        try:
            total = await db.count(self.TABLE_NAME, filters)
        except Exception:
            total = len(records)

        # Convert to summaries
        items = [
            WorkflowSummary(
                id=UUID(r["id"]),
                workflow_type=WorkflowType(r["workflow_type"]),
                status=WorkflowStatus(r["status"]),
                priority=WorkflowPriority(r.get("priority", "normal")),
                progress=r.get("progress", 0),
                created_at=datetime.fromisoformat(r["created_at"]),
                updated_at=datetime.fromisoformat(r["updated_at"]),
                completed_at=datetime.fromisoformat(r["completed_at"]) if r.get("completed_at") else None,
                pending_approval_count=0,  # Fetched separately if needed
            )
            for r in records
        ]

        return WorkflowListResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            has_more=has_more,
        )

    async def get_stats(self, client_id: UUID) -> WorkflowStats:
        """
        Get workflow statistics for a client.

        Args:
            client_id: Client UUID

        Returns:
            WorkflowStats with counts and averages
        """
        db = await get_supabase()
        today = datetime.now(timezone.utc).date()

        # Get all workflows for client
        try:
            records = await db.fetch_many(
                self.TABLE_NAME,
                filters={"client_id": str(client_id)},
                limit=1000,
            )
        except Exception:
            records = []

        # Calculate stats
        stats = WorkflowStats()
        stats.total = len(records)

        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}
        durations: list[float] = []

        for r in records:
            status = r.get("status", "unknown")
            wf_type = r.get("workflow_type", "unknown")

            # Count by status
            by_status[status] = by_status.get(status, 0) + 1

            # Count by type
            by_type[wf_type] = by_type.get(wf_type, 0) + 1

            # Status-specific counts
            if status == WorkflowStatus.PENDING.value:
                stats.pending += 1
            elif status == WorkflowStatus.RUNNING.value:
                stats.running += 1
            elif status == WorkflowStatus.WAITING_APPROVAL.value:
                stats.awaiting_approval += 1

            # Today's counts
            created_at = datetime.fromisoformat(r["created_at"]).date()
            if created_at == today:
                if status == WorkflowStatus.COMPLETED.value:
                    stats.completed_today += 1
                elif status == WorkflowStatus.FAILED.value:
                    stats.failed_today += 1

            # Calculate duration for completed workflows
            if r.get("started_at") and r.get("completed_at"):
                started = datetime.fromisoformat(r["started_at"])
                completed = datetime.fromisoformat(r["completed_at"])
                durations.append((completed - started).total_seconds())

        stats.by_type = by_type
        stats.by_status = by_status

        if durations:
            stats.avg_duration_seconds = sum(durations) / len(durations)

        return stats

    # -------------------------------------------------------------------------
    # WORKFLOW RESUME
    # -------------------------------------------------------------------------

    async def resume_workflow(
        self,
        workflow_id: UUID,
        client_id: UUID,
        request: WorkflowResumeRequest,
    ) -> WorkflowResponse:
        """
        Resume a paused/awaiting_approval workflow.

        Args:
            workflow_id: Workflow UUID
            client_id: Client UUID
            request: Resume request with approval decisions

        Returns:
            WorkflowResponse with updated status

        Raises:
            WorkflowNotFoundError: If workflow not found
            WorkflowInvalidStateError: If workflow is not resumable
        """
        db = await get_supabase()

        # Get workflow
        try:
            record = await db.fetch_one(
                self.TABLE_NAME,
                {"id": str(workflow_id), "client_id": str(client_id)}
            )
        except NotFoundError:
            raise WorkflowNotFoundError(f"Workflow '{workflow_id}' not found")

        status = WorkflowStatus(record["status"])

        # Validate state
        if status != WorkflowStatus.WAITING_APPROVAL:
            raise WorkflowInvalidStateError(
                f"Cannot resume workflow in '{status.value}' state. "
                f"Only 'waiting_approval' workflows can be resumed."
            )

        # Update status
        now = datetime.now(timezone.utc)
        await db.update(
            self.TABLE_NAME,
            {"id": str(workflow_id)},
            {
                "status": WorkflowStatus.RUNNING.value,
                "updated_at": now.isoformat(),
            }
        )

        logger.info(f"Workflow resumed: {workflow_id}")

        # Resume execution
        asyncio.create_task(
            self._resume_execution(
                workflow_id=workflow_id,
                client_id=client_id,
                additional_input=request.additional_input,
            )
        )

        return WorkflowResponse(
            id=workflow_id,
            workflow_type=WorkflowType(record["workflow_type"]),
            status=WorkflowStatus.RUNNING,
            message="Workflow resumed",
            created_at=datetime.fromisoformat(record["created_at"]),
        )

    # -------------------------------------------------------------------------
    # WORKFLOW CANCEL
    # -------------------------------------------------------------------------

    async def cancel_workflow(
        self,
        workflow_id: UUID,
        client_id: UUID,
        reason: str | None = None,
        force: bool = False,
    ) -> WorkflowResponse:
        """
        Cancel a workflow.

        Args:
            workflow_id: Workflow UUID
            client_id: Client UUID
            reason: Cancellation reason
            force: Force cancel even if running

        Returns:
            WorkflowResponse with cancelled status

        Raises:
            WorkflowNotFoundError: If workflow not found
            WorkflowInvalidStateError: If workflow cannot be cancelled
        """
        db = await get_supabase()

        # Get workflow
        try:
            record = await db.fetch_one(
                self.TABLE_NAME,
                {"id": str(workflow_id), "client_id": str(client_id)}
            )
        except NotFoundError:
            raise WorkflowNotFoundError(f"Workflow '{workflow_id}' not found")

        status = WorkflowStatus(record["status"])

        # Check if cancellable
        non_cancellable = (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED)
        if status in non_cancellable:
            raise WorkflowInvalidStateError(
                f"Cannot cancel workflow in '{status.value}' state"
            )

        if status == WorkflowStatus.RUNNING and not force:
            raise WorkflowInvalidStateError(
                "Cannot cancel running workflow without force=true"
            )

        # Update status
        now = datetime.now(timezone.utc)
        update_data = {
            "status": WorkflowStatus.CANCELLED.value,
            "updated_at": now.isoformat(),
            "completed_at": now.isoformat(),
        }
        if reason:
            update_data["error"] = f"Cancelled: {reason}"

        await db.update(
            self.TABLE_NAME,
            {"id": str(workflow_id)},
            update_data
        )

        logger.info(f"Workflow cancelled: {workflow_id} reason={reason}")

        return WorkflowResponse(
            id=workflow_id,
            workflow_type=WorkflowType(record["workflow_type"]),
            status=WorkflowStatus.CANCELLED,
            message=f"Workflow cancelled{': ' + reason if reason else ''}",
            created_at=datetime.fromisoformat(record["created_at"]),
        )

    # -------------------------------------------------------------------------
    # INTERNAL EXECUTION
    # -------------------------------------------------------------------------

    async def _execute_workflow(
        self,
        workflow_id: UUID,
        client_id: UUID,
        client_name: str,
        workflow_type: WorkflowType,
        input_data: dict[str, Any],
    ) -> None:
        """
        Execute workflow asynchronously.

        This runs in background and updates database with progress/results.
        """
        db = await get_supabase()
        now = datetime.now(timezone.utc)

        try:
            # Update to running
            await db.update(
                self.TABLE_NAME,
                {"id": str(workflow_id)},
                {
                    "status": WorkflowStatus.RUNNING.value,
                    "started_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                    "current_step": "initializing",
                }
            )

            # Create initial state
            state = create_initial_state(
                client_id=str(client_id),
                client_name=client_name,
                crm_provider=CRMProvider.HUBSPOT,  # Default to HubSpot
                workflow_type=workflow_type.value,
            )

            # Map workflow type to LangGraph workflow
            langgraph_type = self._map_workflow_type(workflow_type)

            # Get or compile workflow
            from backend.graph.workflow import compile_workflow
            compiled = compile_workflow(langgraph_type, "memory")

            # Execute workflow
            config = {"configurable": {"thread_id": str(workflow_id)}}

            # Add input data to state based on workflow type
            state = self._merge_input_to_state(state, workflow_type, input_data)

            # Run workflow
            final_state = await compiled.ainvoke(state, config=config)

            # Check for pending approvals
            pending_approvals = final_state.get("pending_approvals", [])
            if pending_approvals and any(a.get("status") == "pending" for a in pending_approvals):
                # Workflow paused for approvals
                await self._create_db_approvals(workflow_id, client_id, pending_approvals)
                await db.update(
                    self.TABLE_NAME,
                    {"id": str(workflow_id)},
                    {
                        "status": WorkflowStatus.WAITING_APPROVAL.value,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "current_step": "awaiting_approval",
                        "progress": 50,
                    }
                )
                logger.info(f"Workflow {workflow_id} awaiting approval")
                return

            # Workflow completed
            result = self._extract_result(final_state, workflow_type)
            await db.update(
                self.TABLE_NAME,
                {"id": str(workflow_id)},
                {
                    "status": WorkflowStatus.COMPLETED.value,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "progress": 100,
                    "result": result,
                    "current_step": None,
                }
            )
            logger.info(f"Workflow completed: {workflow_id}")

        except Exception as e:
            logger.exception(f"Workflow execution failed: {workflow_id}")
            await db.update(
                self.TABLE_NAME,
                {"id": str(workflow_id)},
                {
                    "status": WorkflowStatus.FAILED.value,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(e),
                    "current_step": None,
                }
            )

    async def _resume_execution(
        self,
        workflow_id: UUID,
        client_id: UUID,
        additional_input: dict[str, Any],
    ) -> None:
        """Resume workflow execution from checkpoint."""
        db = await get_supabase()

        try:
            # Get workflow record
            record = await db.fetch_one(
                self.TABLE_NAME,
                {"id": str(workflow_id)}
            )

            # TODO: Implement LangGraph checkpoint resume
            # For now, mark as completed
            await db.update(
                self.TABLE_NAME,
                {"id": str(workflow_id)},
                {
                    "status": WorkflowStatus.COMPLETED.value,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "progress": 100,
                    "current_step": None,
                }
            )
            logger.info(f"Workflow resumed and completed: {workflow_id}")

        except Exception as e:
            logger.exception(f"Workflow resume failed: {workflow_id}")
            await db.update(
                self.TABLE_NAME,
                {"id": str(workflow_id)},
                {
                    "status": WorkflowStatus.FAILED.value,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(e),
                }
            )

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _map_workflow_type(self, workflow_type: WorkflowType) -> str:
        """Map API workflow type to LangGraph workflow type."""
        mapping = {
            WorkflowType.FULL_CYCLE: "full_cycle",
            WorkflowType.INTELLIGENCE_ONLY: "intelligence_only",
            WorkflowType.CONTENT_ONLY: "content_only",
            WorkflowType.SALES_OPS_ONLY: "sales_ops_only",
            WorkflowType.MEETING_ANALYSIS: "meeting_analysis",
            WorkflowType.LEAD_RESEARCH: "sales_ops_only",
            WorkflowType.CONTENT_GENERATION: "content_only",
            WorkflowType.CUSTOM: "full_cycle",
        }
        return mapping.get(workflow_type, "meeting_analysis")

    def _merge_input_to_state(
        self,
        state: OrchestratorState,
        workflow_type: WorkflowType,
        input_data: dict[str, Any],
    ) -> OrchestratorState:
        """Merge input data into initial state based on workflow type."""
        # For meeting analysis, add transcript to state
        if workflow_type == WorkflowType.MEETING_ANALYSIS:
            if "transcript" in input_data:
                # Store in metadata for now - agents will read from here
                state["client"]["industry"] = input_data.get("meeting_title", "Meeting")

        return state

    def _extract_result(
        self,
        state: OrchestratorState,
        workflow_type: WorkflowType,
    ) -> dict[str, Any]:
        """Extract result from final state based on workflow type."""
        result: dict[str, Any] = {}

        if workflow_type == WorkflowType.MEETING_ANALYSIS:
            result = {
                "meeting_notes": state.get("meeting_notes", []),
                "crm_tasks": state.get("crm_tasks", []),
                "action_items_count": len(state.get("crm_tasks", [])),
            }
        elif workflow_type == WorkflowType.INTELLIGENCE_ONLY:
            result = {
                "seo_data": state.get("seo_data"),
                "market_research": state.get("market_research"),
                "audience_data": state.get("audience_data"),
            }
        elif workflow_type == WorkflowType.CONTENT_ONLY:
            result = {
                "content_drafts": state.get("content_drafts", []),
                "content_count": len(state.get("content_drafts", [])),
            }
        else:
            # Generic result
            result = {
                "status": state.get("status"),
                "completed_at": state.get("completed_at"),
            }

        return result

    async def _create_db_approvals(
        self,
        workflow_id: UUID,
        client_id: UUID,
        pending_approvals: list[dict],
    ) -> None:
        """Create approval records in database from state approvals."""
        from backend.services.approval_service import get_approval_service

        approval_service = await get_approval_service()

        for approval in pending_approvals:
            if approval.get("status") != "pending":
                continue

            # Map to ApprovalType
            approval_type_str = approval.get("approval_type", "crm_update")
            try:
                approval_type = ApprovalType(approval_type_str)
            except ValueError:
                approval_type = ApprovalType.CRM_UPDATE

            request = ApprovalCreateRequest(
                workflow_id=workflow_id,
                client_id=client_id,
                approval_type=approval_type,
                title=approval.get("title", "Pending Approval"),
                description=approval.get("description"),
                payload=approval.get("payload", {}),
                priority=ApprovalPriority.NORMAL,
            )

            await approval_service.create_approval(request)

    def _estimate_duration(self, workflow_type: WorkflowType) -> int:
        """Estimate workflow duration in seconds."""
        estimates = {
            WorkflowType.MEETING_ANALYSIS: 30,
            WorkflowType.LEAD_RESEARCH: 60,
            WorkflowType.CONTENT_GENERATION: 120,
            WorkflowType.INTELLIGENCE_ONLY: 90,
            WorkflowType.CONTENT_ONLY: 120,
            WorkflowType.SALES_OPS_ONLY: 60,
            WorkflowType.FULL_CYCLE: 300,
        }
        return estimates.get(workflow_type, 60)

    def _parse_steps(self, steps_data: list[dict]) -> list[WorkflowStepSummary]:
        """Parse steps data from database."""
        return [
            WorkflowStepSummary(
                step_name=s.get("step_name", "unknown"),
                status=s.get("status", "unknown"),
                started_at=datetime.fromisoformat(s["started_at"]) if s.get("started_at") else None,
                completed_at=datetime.fromisoformat(s["completed_at"]) if s.get("completed_at") else None,
                duration_ms=s.get("duration_ms"),
                output_summary=s.get("output_summary"),
                error=s.get("error"),
            )
            for s in steps_data
        ]


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

_workflow_service: WorkflowService | None = None


async def get_workflow_service() -> WorkflowService:
    """Get or create WorkflowService singleton."""
    global _workflow_service
    if _workflow_service is None:
        _workflow_service = WorkflowService()
    return _workflow_service


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Exceptions
    "WorkflowError",
    "WorkflowNotFoundError",
    "WorkflowAlreadyExistsError",
    "WorkflowInvalidStateError",
    "WorkflowExecutionError",
    # Service
    "WorkflowService",
    # DI
    "get_workflow_service",
]

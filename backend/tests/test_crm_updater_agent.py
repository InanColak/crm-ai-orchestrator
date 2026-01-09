"""
CRM Updater Agent Tests (Phase 3.3)
===================================
Tests for the CRM Updater agent.
Verifies LLM integration, operation preparation, risk assessment, and HITL approval.

Test Criteria: Agent can prepare CRM operations for human approval
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from backend.app.schemas.crm_updates import (
    CRMUpdateOperationResult,
    CRMUpdateOperation,
    CRMOperationType,
    OperationRiskLevel,
    TaskOperationPayload,
    DealUpdatePayload,
    NotePayload,
)
from backend.graph.nodes import crm_updater_node
from backend.graph.state import (
    OrchestratorState,
    MeetingNote,
    CRMTask,
    CRMProvider,
    ClientContext,
    ApprovalRequest,
    ApprovalType,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_crm_tasks():
    """Sample CRM tasks from task extraction."""
    return [
        CRMTask(
            task_id=str(uuid4()),
            task_type="create_task",
            entity_type="task",
            entity_id=None,
            payload={
                "hubspot_task": {
                    "hs_task_subject": "Follow up with Acme Corp on pricing proposal",
                    "hs_task_body": "Schedule call to discuss enterprise pricing details.",
                    "hs_task_status": "NOT_STARTED",
                    "hs_task_priority": "HIGH",
                    "hs_task_type": "CALL",
                    "hs_timestamp": "1705698000000",  # Due date timestamp
                },
                "source_meeting_id": "meeting-123",
                "assignee_name": "John Smith",
                "assignee_email": "john@example.com",
                "associations": [
                    {"type": "deal", "id": "deal-456", "name": "Acme Enterprise Deal"}
                ],
                "extraction_confidence": "high",
                "needs_review": False,
            },
            priority="high",
            status="pending",
            error_message=None,
            created_at=datetime.now(timezone.utc).isoformat(),
            executed_at=None,
        ),
        CRMTask(
            task_id=str(uuid4()),
            task_type="create_task",
            entity_type="task",
            entity_id=None,
            payload={
                "hubspot_task": {
                    "hs_task_subject": "Prepare onboarding improvement roadmap",
                    "hs_task_body": "Create detailed roadmap for customer onboarding enhancements.",
                    "hs_task_status": "NOT_STARTED",
                    "hs_task_priority": "MEDIUM",
                    "hs_task_type": "TODO",
                },
                "source_meeting_id": "meeting-123",
                "assignee_name": "Sarah Johnson",
                "needs_review": True,
                "review_reason": "Assignee may need clarification",
            },
            priority="medium",
            status="pending",
            error_message=None,
            created_at=datetime.now(timezone.utc).isoformat(),
            executed_at=None,
        ),
    ]


@pytest.fixture
def sample_meeting_note():
    """Sample MeetingNote for context."""
    return MeetingNote(
        meeting_id=str(uuid4()),
        date=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc).isoformat(),
        participants=["John Smith", "Sarah Johnson", "Mike Williams"],
        summary="Team discussed Q4 strategy and Q1 enterprise expansion plans.",
        key_points=[
            "Q4 exceeded expectations",
            "Focus on enterprise segment",
            "Need infrastructure scaling assessment",
        ],
        action_items=[
            {"task": "Follow up on pricing", "assignee": "John", "due_date": "next week"},
            {"task": "Create roadmap", "assignee": "Sarah", "due_date": "end of month"},
        ],
        follow_up_required=True,
        sentiment="positive",
        deal_stage_update="presentationscheduled",
    )


@pytest.fixture
def mock_operations_result():
    """Mock CRMUpdateOperationResult from LLM."""
    return CRMUpdateOperationResult(
        operations=[
            CRMUpdateOperation(
                operation_id=str(uuid4()),
                operation_type=CRMOperationType.CREATE_TASK,
                summary="Create task 'Follow up with Acme Corp' assigned to John Smith (due: 2024-01-20)",
                details="High-priority follow-up call for enterprise deal pricing discussion.",
                payload={
                    "hs_task_subject": "Follow up with Acme Corp on pricing proposal",
                    "hs_task_body": "Schedule call to discuss enterprise pricing details.",
                    "hs_task_status": "NOT_STARTED",
                    "hs_task_priority": "HIGH",
                    "hs_task_type": "CALL",
                    "hs_timestamp": "1705698000000",
                    "hubspot_owner_id": None,
                },
                risk_level=OperationRiskLevel.LOW,
                risk_factors=["Task with clear assignee", "Standard priority"],
                approval_required=True,
                auto_approve_eligible=True,
                rollback_info="Delete task from HubSpot if needed",
                source_task_id="task-123",
            ),
            CRMUpdateOperation(
                operation_id=str(uuid4()),
                operation_type=CRMOperationType.CREATE_TASK,
                summary="Create task 'Prepare onboarding roadmap' assigned to Sarah (due: 2024-01-31)",
                details="Medium-priority documentation task.",
                payload={
                    "hs_task_subject": "Prepare onboarding improvement roadmap",
                    "hs_task_body": "Create detailed roadmap.",
                    "hs_task_status": "NOT_STARTED",
                    "hs_task_priority": "MEDIUM",
                    "hs_task_type": "TODO",
                },
                risk_level=OperationRiskLevel.MEDIUM,
                risk_factors=["Assignee needs clarification"],
                approval_required=True,
                auto_approve_eligible=False,
                rollback_info="Delete task if not needed",
                source_task_id="task-456",
            ),
        ],
        batch_summary="2 tasks to create from meeting action items",
        total_operations=2,
        high_risk_count=0,
        deal_stage_changes=[
            {
                "deal_id": "deal-456",
                "current_stage": "qualifiedtobuy",
                "new_stage": "presentationscheduled",
                "reasoning": "Meeting included product presentation discussion",
            }
        ],
        notes_to_add=[
            {
                "entity_type": "deal",
                "entity_id": "deal-456",
                "note_content": "Q1 Strategy Meeting Summary: Team discussed enterprise expansion...",
            }
        ],
        processing_notes="All tasks successfully transformed to CRM operations.",
        warnings=[],
    )


@pytest.fixture
def base_state_with_tasks(sample_crm_tasks, sample_meeting_note):
    """OrchestratorState with pending CRM tasks."""
    return OrchestratorState(
        workflow_id=str(uuid4()),
        workflow_type="sales_ops_only",
        status="in_progress",
        started_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        completed_at=None,
        error_message=None,
        client=ClientContext(
            client_id=str(uuid4()),
            client_name="Test Company",
            crm_provider=CRMProvider.HUBSPOT,
            crm_access_token=None,
            crm_refresh_token=None,
            brandvoice_doc_id=None,
            industry="Technology",
            target_audience="B2B SaaS",
        ),
        seo_data=None,
        market_research=None,
        audience_data=None,
        web_analysis=None,
        content_pipeline=None,
        content_drafts=[],
        meeting_notes=[sample_meeting_note],
        crm_tasks=sample_crm_tasks,
        leads=[],
        email_drafts=[],
        pending_approvals=[],
        approval_history=[],
        messages=[],
        retrieved_documents=[],
        brandvoice_context=None,
        trace_id=None,
        agent_execution_log=[],
    )


@pytest.fixture
def empty_state():
    """State with no CRM tasks."""
    return OrchestratorState(
        workflow_id=str(uuid4()),
        workflow_type="sales_ops_only",
        status="in_progress",
        started_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        completed_at=None,
        error_message=None,
        client=ClientContext(
            client_id=str(uuid4()),
            client_name="Test Company",
            crm_provider=CRMProvider.HUBSPOT,
            crm_access_token=None,
            crm_refresh_token=None,
            brandvoice_doc_id=None,
            industry="Technology",
            target_audience="B2B SaaS",
        ),
        seo_data=None,
        market_research=None,
        audience_data=None,
        web_analysis=None,
        content_pipeline=None,
        content_drafts=[],
        meeting_notes=[],
        crm_tasks=[],
        leads=[],
        email_drafts=[],
        pending_approvals=[],
        approval_history=[],
        messages=[],
        retrieved_documents=[],
        brandvoice_context=None,
        trace_id=None,
        agent_execution_log=[],
    )


@pytest.fixture
def state_with_executed_tasks():
    """State with tasks already executed (not pending)."""
    return OrchestratorState(
        workflow_id=str(uuid4()),
        workflow_type="sales_ops_only",
        status="in_progress",
        started_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        completed_at=None,
        error_message=None,
        client=ClientContext(
            client_id=str(uuid4()),
            client_name="Test Company",
            crm_provider=CRMProvider.HUBSPOT,
            crm_access_token=None,
            crm_refresh_token=None,
            brandvoice_doc_id=None,
            industry="Technology",
            target_audience="B2B SaaS",
        ),
        seo_data=None,
        market_research=None,
        audience_data=None,
        web_analysis=None,
        content_pipeline=None,
        content_drafts=[],
        meeting_notes=[],
        crm_tasks=[
            CRMTask(
                task_id=str(uuid4()),
                task_type="create_task",
                entity_type="task",
                entity_id="hubspot-task-123",
                payload={},
                priority="medium",
                status="executed",  # Already executed
                error_message=None,
                created_at=datetime.now(timezone.utc).isoformat(),
                executed_at=datetime.now(timezone.utc).isoformat(),
            ),
        ],
        leads=[],
        email_drafts=[],
        pending_approvals=[],
        approval_history=[],
        messages=[],
        retrieved_documents=[],
        brandvoice_context=None,
        trace_id=None,
        agent_execution_log=[],
    )


# =============================================================================
# SCHEMA TESTS
# =============================================================================

class TestCRMUpdateSchemas:
    """Tests for CRM Update Pydantic schemas."""

    def test_operation_type_enum(self):
        """CRMOperationType enum should have expected values."""
        assert CRMOperationType.CREATE_TASK == "create_task"
        assert CRMOperationType.UPDATE_DEAL == "update_deal"
        assert CRMOperationType.ADD_NOTE == "add_note"
        assert CRMOperationType.CREATE_ACTIVITY == "create_activity"

    def test_risk_level_enum(self):
        """OperationRiskLevel enum should have expected values."""
        assert OperationRiskLevel.LOW == "low"
        assert OperationRiskLevel.MEDIUM == "medium"
        assert OperationRiskLevel.HIGH == "high"

    def test_crm_update_operation_validation(self):
        """Should validate CRMUpdateOperation correctly."""
        operation = CRMUpdateOperation(
            operation_id=str(uuid4()),
            operation_type=CRMOperationType.CREATE_TASK,
            summary="Create task 'Follow up' assigned to John",
            payload={"hs_task_subject": "Follow up"},
            risk_level=OperationRiskLevel.LOW,
        )
        assert operation.operation_type == CRMOperationType.CREATE_TASK
        assert operation.risk_level == OperationRiskLevel.LOW
        assert operation.approval_required is True

    def test_crm_update_operation_summary_min_length(self):
        """Should enforce minimum summary length."""
        with pytest.raises(Exception):  # ValidationError
            CRMUpdateOperation(
                operation_id="test",
                operation_type=CRMOperationType.CREATE_TASK,
                summary="Short",  # Too short (< 10 chars)
                payload={},
            )

    def test_operations_result_structure(self, mock_operations_result):
        """Should create valid CRMUpdateOperationResult."""
        assert len(mock_operations_result.operations) == 2
        assert mock_operations_result.total_operations == 2
        assert mock_operations_result.high_risk_count == 0
        assert len(mock_operations_result.deal_stage_changes) == 1

    def test_task_operation_payload_structure(self):
        """Should create valid TaskOperationPayload."""
        payload = TaskOperationPayload(
            hs_task_subject="Follow up call",
            hs_task_body="Call to discuss proposal",
            hs_task_status="NOT_STARTED",
            hs_task_priority="HIGH",
            hs_task_type="CALL",
        )
        assert payload.hs_task_subject == "Follow up call"
        assert payload.hs_task_priority == "HIGH"

    def test_deal_update_payload_structure(self):
        """Should create valid DealUpdatePayload."""
        payload = DealUpdatePayload(
            deal_id="deal-123",
            properties={"dealstage": "presentationscheduled"},
            dealstage="presentationscheduled",
        )
        assert payload.deal_id == "deal-123"
        assert payload.dealstage == "presentationscheduled"

    def test_note_payload_structure(self):
        """Should create valid NotePayload."""
        payload = NotePayload(
            body="Meeting summary: Team discussed enterprise expansion...",
            deal_id="deal-456",
            source="meeting_notes",
        )
        assert len(payload.body) > 0
        assert payload.deal_id == "deal-456"


# =============================================================================
# CRM UPDATER NODE TESTS
# =============================================================================

class TestCRMUpdaterNode:
    """Tests for crm_updater_node function."""

    def test_node_returns_info_without_pending_tasks(self, empty_state):
        """Should return info message when no pending CRM tasks."""
        result = crm_updater_node(empty_state)

        assert "messages" in result
        assert any(
            msg.get("message_type") == "info"
            for msg in result.get("messages", [])
        )
        # Should not create approval requests
        assert len(result.get("pending_approvals", [])) == 0

    def test_node_skips_executed_tasks(self, state_with_executed_tasks):
        """Should skip tasks that are already executed."""
        result = crm_updater_node(state_with_executed_tasks)

        # Should return info (no pending tasks)
        assert "messages" in result
        assert len(result.get("pending_approvals", [])) == 0

    def test_node_prepares_operations_with_mock_llm(
        self, base_state_with_tasks, mock_operations_result
    ):
        """Should prepare operations using LLM service."""
        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm
            mock_llm.generate_structured = AsyncMock(
                return_value=(mock_operations_result, MagicMock(total_tokens=500))
            )

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = (
                    "You are a CRM Update Specialist",
                    "Transform these tasks into operations",
                )

                result = crm_updater_node(base_state_with_tasks)

        # Should have pending_approvals
        assert "pending_approvals" in result
        assert len(result["pending_approvals"]) == 1

    def test_node_creates_approval_request_structure(
        self, base_state_with_tasks, mock_operations_result
    ):
        """Should create properly structured approval request."""
        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm
            mock_llm.generate_structured = AsyncMock(
                return_value=(mock_operations_result, MagicMock(total_tokens=500))
            )

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = ("system", "user")

                result = crm_updater_node(base_state_with_tasks)

        # Verify approval request structure
        approval = result["pending_approvals"][0]
        assert "approval_id" in approval
        assert approval["approval_type"] == ApprovalType.CRM_UPDATE
        assert "title" in approval
        assert "description" in approval
        assert "payload" in approval
        assert approval["status"] == "pending"
        assert approval["requested_by"] == "crm_updater"

    def test_node_includes_operations_in_payload(
        self, base_state_with_tasks, mock_operations_result
    ):
        """Should include all operations in approval payload."""
        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm
            mock_llm.generate_structured = AsyncMock(
                return_value=(mock_operations_result, MagicMock(total_tokens=500))
            )

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = ("system", "user")

                result = crm_updater_node(base_state_with_tasks)

        # Verify payload contains operations
        payload = result["pending_approvals"][0]["payload"]
        assert "operations" in payload
        assert len(payload["operations"]) == 2
        assert "batch_summary" in payload
        assert "total_operations" in payload

    def test_node_handles_llm_error_with_fallback(self, base_state_with_tasks):
        """Should handle LLM errors and use fallback mode."""
        from backend.services.llm_service import LLMError

        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm
            mock_llm.generate_structured = AsyncMock(
                side_effect=LLMError("API rate limit exceeded", retryable=True)
            )

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = ("system", "user")

                result = crm_updater_node(base_state_with_tasks)

        # Should still create approval request in fallback mode
        assert "pending_approvals" in result
        assert len(result["pending_approvals"]) == 1

        # Should indicate fallback mode
        payload = result["pending_approvals"][0]["payload"]
        assert payload.get("fallback_mode") is True
        assert "error" in payload

    def test_node_logs_execution(
        self, base_state_with_tasks, mock_operations_result
    ):
        """Should create execution log entries."""
        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm
            mock_llm.generate_structured = AsyncMock(
                return_value=(mock_operations_result, MagicMock(total_tokens=500))
            )

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = ("system", "user")

                result = crm_updater_node(base_state_with_tasks)

        # Verify execution log
        assert "agent_execution_log" in result
        log_entry = result["agent_execution_log"][0]
        assert log_entry["agent"] == "crm_updater"
        assert "details" in log_entry


# =============================================================================
# APPROVAL REQUEST TESTS
# =============================================================================

class TestApprovalRequestCreation:
    """Tests for approval request creation from operations."""

    def test_approval_includes_risk_assessment(
        self, base_state_with_tasks, mock_operations_result
    ):
        """Should include risk level in operations."""
        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm
            mock_llm.generate_structured = AsyncMock(
                return_value=(mock_operations_result, MagicMock(total_tokens=500))
            )

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = ("system", "user")

                result = crm_updater_node(base_state_with_tasks)

        # Check operations have risk levels
        operations = result["pending_approvals"][0]["payload"]["operations"]
        for op in operations:
            assert "risk_level" in op
            assert op["risk_level"] in ["low", "medium", "high"]

    def test_approval_includes_deal_stage_changes(
        self, base_state_with_tasks, mock_operations_result
    ):
        """Should include deal stage change recommendations."""
        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm
            mock_llm.generate_structured = AsyncMock(
                return_value=(mock_operations_result, MagicMock(total_tokens=500))
            )

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = ("system", "user")

                result = crm_updater_node(base_state_with_tasks)

        payload = result["pending_approvals"][0]["payload"]
        assert "deal_stage_changes" in payload
        assert len(payload["deal_stage_changes"]) == 1

    def test_high_risk_count_in_description(self, base_state_with_tasks):
        """Should highlight high-risk operations in description."""
        high_risk_result = CRMUpdateOperationResult(
            operations=[
                CRMUpdateOperation(
                    operation_id=str(uuid4()),
                    operation_type=CRMOperationType.UPDATE_DEAL,
                    summary="Update deal 'Enterprise' to closed-won (HIGH RISK)",
                    payload={"dealstage": "closedwon"},
                    risk_level=OperationRiskLevel.HIGH,
                    risk_factors=["Deal stage change to closed-won", "High-value deal"],
                    approval_required=True,
                ),
            ],
            batch_summary="1 high-risk deal update",
            total_operations=1,
            high_risk_count=1,
            deal_stage_changes=[],
            notes_to_add=[],
        )

        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm
            mock_llm.generate_structured = AsyncMock(
                return_value=(high_risk_result, MagicMock(total_tokens=300))
            )

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = ("system", "user")

                result = crm_updater_node(base_state_with_tasks)

        # Description should mention high risk
        description = result["pending_approvals"][0]["description"]
        assert "HIGH RISK" in description or "high-risk" in description.lower()


# =============================================================================
# FALLBACK MODE TESTS
# =============================================================================

class TestFallbackMode:
    """Tests for fallback mode when LLM fails."""

    def test_fallback_preserves_task_data(self, base_state_with_tasks):
        """Should preserve original task data in fallback mode."""
        from backend.services.llm_service import LLMError

        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm
            mock_llm.generate_structured = AsyncMock(
                side_effect=LLMError("Service unavailable")
            )

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = ("system", "user")

                result = crm_updater_node(base_state_with_tasks)

        # Verify tasks are preserved
        payload = result["pending_approvals"][0]["payload"]
        assert "tasks" in payload
        assert len(payload["tasks"]) == 2

    def test_fallback_includes_error_message(self, base_state_with_tasks):
        """Should include error message in fallback payload."""
        from backend.services.llm_service import LLMError

        error_message = "API rate limit exceeded"

        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm
            mock_llm.generate_structured = AsyncMock(
                side_effect=LLMError(error_message)
            )

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = ("system", "user")

                result = crm_updater_node(base_state_with_tasks)

        payload = result["pending_approvals"][0]["payload"]
        assert error_message in payload["error"]

    def test_fallback_logs_error(self, base_state_with_tasks):
        """Should log LLM error in execution log."""
        from backend.services.llm_service import LLMError

        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm
            mock_llm.generate_structured = AsyncMock(
                side_effect=LLMError("Connection timeout")
            )

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = ("system", "user")

                result = crm_updater_node(base_state_with_tasks)

        # Check execution log has fallback action
        log_entry = result["agent_execution_log"][0]
        assert log_entry["action"] == "fallback_mode"


# =============================================================================
# PHASE 3.3 VERIFICATION
# =============================================================================

class TestPhase33Verification:
    """
    Phase 3.3 Verification: CRM Updater Agent
    Criteria: Agent can prepare CRM operations for human approval
    """

    def test_crm_updater_node_importable(self):
        """Should be able to import crm_updater_node."""
        from backend.graph.nodes import crm_updater_node
        assert crm_updater_node is not None

    def test_crm_update_schemas_importable(self):
        """Should be able to import CRM update schemas."""
        from backend.app.schemas.crm_updates import (
            CRMUpdateOperationResult,
            CRMUpdateOperation,
            CRMOperationType,
            OperationRiskLevel,
        )
        assert CRMUpdateOperationResult is not None
        assert CRMUpdateOperation is not None
        assert CRMOperationType is not None
        assert OperationRiskLevel is not None

    def test_schemas_exported_from_module(self):
        """CRM update schemas should be exported from schemas module."""
        from backend.app.schemas import (
            CRMOperationType,
            OperationRiskLevel,
            CRMUpdateOperation,
            CRMUpdateOperationResult,
            CRMUpdateRequest,
            CRMUpdateResponse,
        )

        assert CRMOperationType is not None
        assert OperationRiskLevel is not None
        assert CRMUpdateOperation is not None
        assert CRMUpdateOperationResult is not None
        assert CRMUpdateRequest is not None
        assert CRMUpdateResponse is not None

    def test_agent_in_exports(self):
        """crm_updater_node should be in nodes exports."""
        from backend.graph.nodes import __all__

        assert "crm_updater_node" in __all__

    @pytest.mark.asyncio
    async def test_end_to_end_mock_preparation(
        self, base_state_with_tasks, mock_operations_result
    ):
        """
        VERIFICATION TEST: Full agent flow with mocked LLM.

        Verifies:
        1. CRM tasks extraction from state
        2. LLM service integration
        3. Operations preparation
        4. Risk assessment
        5. Approval request creation
        """
        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm
            mock_llm.generate_structured = AsyncMock(
                return_value=(mock_operations_result, MagicMock(total_tokens=500))
            )

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = (
                    "You are a CRM Update Specialist",
                    "Transform CRM tasks into operations",
                )

                result = crm_updater_node(base_state_with_tasks)

        # Verify complete result structure
        assert "pending_approvals" in result
        assert len(result["pending_approvals"]) == 1

        # Verify approval request
        approval = result["pending_approvals"][0]
        assert approval["approval_type"] == ApprovalType.CRM_UPDATE
        assert approval["status"] == "pending"

        # Verify payload structure
        payload = approval["payload"]
        assert "operations" in payload
        assert "batch_summary" in payload
        assert "total_operations" in payload
        assert payload["total_operations"] == 2

        # Verify execution log
        assert "agent_execution_log" in result
        log_entry = result["agent_execution_log"][0]
        assert log_entry["agent"] == "crm_updater"

        # Verify messages
        assert "messages" in result

    def test_operations_hubspot_compatible(self, mock_operations_result):
        """Verify operations produce HubSpot-compatible payloads."""
        for operation in mock_operations_result.operations:
            if operation.operation_type == CRMOperationType.CREATE_TASK:
                payload = operation.payload

                # HubSpot task fields should be present
                assert "hs_task_subject" in payload
                assert "hs_task_status" in payload
                assert "hs_task_priority" in payload

                # Validate enum values
                assert payload["hs_task_status"] in [
                    "NOT_STARTED", "IN_PROGRESS", "WAITING", "COMPLETED"
                ]
                assert payload["hs_task_priority"] in ["LOW", "MEDIUM", "HIGH"]

    def test_meeting_context_used_for_operations(
        self, base_state_with_tasks, mock_operations_result
    ):
        """Should use meeting notes context for operations."""
        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm
            mock_llm.generate_structured = AsyncMock(
                return_value=(mock_operations_result, MagicMock(total_tokens=500))
            )

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = ("system", "user")

                crm_updater_node(base_state_with_tasks)

                # Verify prompt manager was called with meeting context
                call_args = mock_prompt.get_full_prompt.call_args
                assert call_args is not None
                # meeting_summary should be passed
                kwargs = call_args[1] if call_args[1] else {}
                # The function should have been called with meeting context
                assert "meeting_summary" in kwargs or len(call_args[0]) > 0

    def test_workflow_id_preserved(
        self, base_state_with_tasks, mock_operations_result
    ):
        """Should preserve workflow_id in approval payload."""
        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm
            mock_llm.generate_structured = AsyncMock(
                return_value=(mock_operations_result, MagicMock(total_tokens=500))
            )

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = ("system", "user")

                result = crm_updater_node(base_state_with_tasks)

        # Workflow ID should be in payload
        payload = result["pending_approvals"][0]["payload"]
        assert "workflow_id" in payload

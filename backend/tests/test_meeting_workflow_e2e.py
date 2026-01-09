"""
Meeting-to-CRM Workflow End-to-End Tests (Phase 3.4)
====================================================
E2E tests for the complete MVP workflow:
Meeting Transcript -> Meeting Notes -> Task Extractor -> CRM Updater -> Approval

Test Criteria:
- Full pipeline execution from meeting input to CRM approval request
- Correct state transitions through all nodes
- Proper data flow between agents
- HITL approval integration
- Error handling across the pipeline
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from backend.graph.workflow import (
    compile_workflow,
    build_meeting_analysis_workflow,
    WORKFLOW_BUILDERS,
)
from backend.graph.state import (
    OrchestratorState,
    WorkflowStatus,
    CRMProvider,
    ClientContext,
    MeetingNote,
    CRMTask,
    ApprovalRequest,
    ApprovalType,
    AgentMessage,
    create_initial_state,
)
from backend.graph.routers import (
    route_from_sales_ops,
    should_request_approval,
    get_workflow_progress,
)
from backend.graph.nodes import (
    meeting_notes_node,
    task_extractor_node,
    crm_updater_node,
)
from backend.app.schemas.meeting_notes import (
    MeetingAnalysis,
    ActionItem,
    KeyDecision,
    DealStageRecommendation,
    MeetingSentiment,
    ActionItemPriority,
)
from backend.app.schemas.tasks import (
    TaskExtractionResult,
    ExtractedTask,
    TaskPriority,
    TaskStatus,
    TaskType,
)
from backend.app.schemas.crm_updates import (
    CRMUpdateOperationResult,
    CRMUpdateOperation,
    CRMOperationType,
    OperationRiskLevel,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_meeting_transcript():
    """Realistic meeting transcript for E2E testing."""
    return """
    Meeting: Q1 Strategy Planning Session
    Date: January 15, 2024
    Participants: John Smith (Sales Director), Sarah Johnson (Product Manager), Mike Chen (Engineering Lead)

    John: Welcome everyone to our Q1 planning session. Let's discuss our enterprise expansion strategy.

    Sarah: I've been analyzing customer feedback from Q4. The main request is for better onboarding documentation.

    Mike: We can prioritize that. I'll need about two weeks to get the technical docs updated.

    John: Great. We also need to follow up with Acme Corp about the enterprise deal. They're waiting for our pricing proposal.

    Sarah: I can prepare the pricing deck by Friday if Mike can provide the technical specifications.

    Mike: Sure, I'll send those over by Wednesday.

    John: Perfect. One more thing - we should schedule a demo with the Acme team next week.

    Sarah: I'll coordinate with their team and find a suitable time.

    John: Let's reconvene on Friday to review progress. Meeting adjourned.
    """


@pytest.fixture
def sample_meeting_input(sample_meeting_transcript):
    """NormalizedMeetingInput for testing."""
    return {
        "source": "manual_text",
        "source_id": None,
        "title": "Q1 Strategy Planning Session",
        "meeting_date": "2024-01-15T10:00:00Z",
        "duration_minutes": 45,
        "participants": ["John Smith", "Sarah Johnson", "Mike Chen"],
        "organizer": "John Smith",
        "transcript": sample_meeting_transcript,
        "deal_id": "deal-12345",
        "contact_id": None,
        "company_id": "company-acme",
        "additional_context": "Enterprise expansion initiative",
    }


@pytest.fixture
def mock_meeting_analysis():
    """Mock MeetingAnalysis response from LLM."""
    return MeetingAnalysis(
        summary="Q1 planning session focused on enterprise expansion and Acme Corp deal. Team agreed on action items for pricing proposal, technical specifications, and demo scheduling.",
        key_points=[
            "Customer feedback requests better onboarding documentation",
            "Enterprise expansion is Q1 priority",
            "Acme Corp deal needs pricing proposal",
            "Technical documentation updates planned",
        ],
        key_decisions=[
            KeyDecision(
                decision="Prioritize onboarding documentation improvements",
                rationale="Based on Q4 customer feedback",
                stakeholders=["Sarah Johnson", "Mike Chen"],
            ),
            KeyDecision(
                decision="Prepare enterprise pricing proposal for Acme Corp",
                rationale="Deal waiting for pricing information",
                stakeholders=["John Smith", "Sarah Johnson"],
            ),
        ],
        action_items=[
            ActionItem(
                task="Prepare pricing proposal deck for Acme Corp",
                assignee="Sarah Johnson",
                due_date="Friday",
                priority=ActionItemPriority.HIGH,
                context="Needed for enterprise deal progression",
            ),
            ActionItem(
                task="Send technical specifications to Sarah",
                assignee="Mike Chen",
                due_date="Wednesday",
                priority=ActionItemPriority.MEDIUM,
                context="Required for pricing deck",
            ),
            ActionItem(
                task="Schedule demo with Acme Corp team",
                assignee="Sarah Johnson",
                due_date="next week",
                priority=ActionItemPriority.HIGH,
                context="Part of enterprise sales process",
            ),
            ActionItem(
                task="Update technical documentation for onboarding",
                assignee="Mike Chen",
                due_date="two weeks",
                priority=ActionItemPriority.MEDIUM,
                context="Customer feedback driven improvement",
            ),
        ],
        overall_sentiment=MeetingSentiment.POSITIVE,
        sentiment_explanation="Productive meeting with clear action items and positive team collaboration",
        follow_up_required=True,
        follow_up_reason="Friday review meeting scheduled",
        deal_stage_recommendation=DealStageRecommendation(
            current_signals=["Pricing discussion initiated", "Demo scheduling planned"],
            recommended_stage="presentationscheduled",
            confidence="medium",
            reasoning="Deal progressing from qualification to presentation phase",
        ),
        identified_participants=["John Smith", "Sarah Johnson", "Mike Chen"],
        risks_concerns=["Tight timeline for pricing proposal"],
        opportunities=["Enterprise expansion potential", "Acme Corp deal progression"],
        next_steps=["Prepare pricing deck", "Send tech specs", "Schedule demo", "Friday review meeting"],
    )


@pytest.fixture
def mock_task_extraction_result():
    """Mock TaskExtractionResult response from LLM."""
    return TaskExtractionResult(
        tasks=[
            ExtractedTask(
                subject="Prepare pricing proposal deck for Acme Corp",
                body="Create comprehensive pricing deck for enterprise tier. Include all feature tiers and volume discounts. Context: Needed for Acme Corp deal progression.",
                task_type=TaskType.TODO,
                priority=TaskPriority.HIGH,
                status=TaskStatus.NOT_STARTED,
                due_date="2024-01-19",
                due_date_reasoning="'Friday' from meeting date 2024-01-15",
                assignee_name="Sarah Johnson",
                assignee_email="sarah@example.com",
                hubspot_owner_id=None,
                associations=[],
                source_action_item="Prepare pricing proposal deck for Acme Corp",
                meeting_context="Enterprise expansion initiative",
                extraction_confidence="high",
                needs_review=False,
                review_reason=None,
            ),
            ExtractedTask(
                subject="Send technical specifications to Sarah",
                body="Compile and send technical specifications document for pricing deck preparation.",
                task_type=TaskType.EMAIL,
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.NOT_STARTED,
                due_date="2024-01-17",
                due_date_reasoning="'Wednesday' from meeting date 2024-01-15",
                assignee_name="Mike Chen",
                assignee_email="mike@example.com",
                hubspot_owner_id=None,
                associations=[],
                source_action_item="Send technical specifications to Sarah",
                meeting_context="Required for pricing deck",
                extraction_confidence="high",
                needs_review=False,
                review_reason=None,
            ),
            ExtractedTask(
                subject="Schedule demo with Acme Corp team",
                body="Coordinate with Acme Corp team to schedule product demonstration for next week.",
                task_type=TaskType.CALL,
                priority=TaskPriority.HIGH,
                status=TaskStatus.NOT_STARTED,
                due_date="2024-01-22",
                due_date_reasoning="'next week' from meeting date 2024-01-15",
                assignee_name="Sarah Johnson",
                assignee_email="sarah@example.com",
                hubspot_owner_id=None,
                associations=[],
                source_action_item="Schedule demo with Acme Corp team",
                meeting_context="Part of enterprise sales process",
                extraction_confidence="high",
                needs_review=False,
                review_reason=None,
            ),
        ],
        skipped_items=[],
        total_action_items=4,
        tasks_created=3,
        tasks_needing_review=0,
        processing_notes="One task (documentation update) skipped due to 2-week timeline",
    )


@pytest.fixture
def mock_crm_operations_result():
    """Mock CRMUpdateOperationResult response from LLM."""
    return CRMUpdateOperationResult(
        operations=[
            CRMUpdateOperation(
                operation_id=str(uuid4()),
                operation_type=CRMOperationType.CREATE_TASK,
                summary="Create task 'Prepare pricing proposal' assigned to Sarah Johnson (due: 2024-01-19)",
                details="High-priority task for Acme Corp deal",
                payload={
                    "hs_task_subject": "Prepare pricing proposal deck for Acme Corp",
                    "hs_task_body": "Create comprehensive pricing deck for enterprise tier.",
                    "hs_task_status": "NOT_STARTED",
                    "hs_task_priority": "HIGH",
                    "hs_task_type": "TODO",
                    "hs_timestamp": "1705622400000",
                },
                risk_level=OperationRiskLevel.LOW,
                risk_factors=["Clear assignee and due date"],
                approval_required=True,
                auto_approve_eligible=True,
                rollback_info="Delete task from HubSpot",
                source_task_id="task-1",
            ),
            CRMUpdateOperation(
                operation_id=str(uuid4()),
                operation_type=CRMOperationType.CREATE_TASK,
                summary="Create task 'Send tech specs' assigned to Mike Chen (due: 2024-01-17)",
                details="Medium-priority email task",
                payload={
                    "hs_task_subject": "Send technical specifications to Sarah",
                    "hs_task_body": "Compile and send technical specifications.",
                    "hs_task_status": "NOT_STARTED",
                    "hs_task_priority": "MEDIUM",
                    "hs_task_type": "EMAIL",
                },
                risk_level=OperationRiskLevel.LOW,
                risk_factors=[],
                approval_required=True,
                auto_approve_eligible=True,
                rollback_info="Delete task from HubSpot",
                source_task_id="task-2",
            ),
            CRMUpdateOperation(
                operation_id=str(uuid4()),
                operation_type=CRMOperationType.CREATE_TASK,
                summary="Create task 'Schedule demo' assigned to Sarah Johnson (due: 2024-01-22)",
                details="High-priority call task for Acme Corp",
                payload={
                    "hs_task_subject": "Schedule demo with Acme Corp team",
                    "hs_task_body": "Coordinate demo scheduling.",
                    "hs_task_status": "NOT_STARTED",
                    "hs_task_priority": "HIGH",
                    "hs_task_type": "CALL",
                },
                risk_level=OperationRiskLevel.LOW,
                risk_factors=[],
                approval_required=True,
                auto_approve_eligible=True,
                rollback_info="Delete task from HubSpot",
                source_task_id="task-3",
            ),
        ],
        batch_summary="3 tasks to create for Acme Corp deal follow-up",
        total_operations=3,
        high_risk_count=0,
        deal_stage_changes=[
            {
                "deal_id": "deal-12345",
                "current_stage": "qualifiedtobuy",
                "new_stage": "presentationscheduled",
                "reasoning": "Demo scheduling indicates progression",
            }
        ],
        notes_to_add=[
            {
                "entity_type": "deal",
                "entity_id": "deal-12345",
                "note_content": "Q1 Strategy Meeting Summary: Discussed enterprise expansion and Acme Corp deal...",
            }
        ],
        processing_notes="All tasks successfully transformed",
        warnings=[],
    )


@pytest.fixture
def initial_meeting_state(sample_meeting_input):
    """Initial OrchestratorState for meeting analysis workflow."""
    state = create_initial_state(
        client_id="client-12345",
        client_name="Test Enterprise",
        crm_provider=CRMProvider.HUBSPOT,
        workflow_type="meeting_analysis",
    )
    # Add meeting input as a message (must match nodes.py expectations)
    # meeting_notes_node looks for message_type="meeting_input" and data in metadata
    state["messages"] = [
        AgentMessage(
            message_id=str(uuid4()),
            from_agent="user",
            to_agent="meeting_notes",
            message_type="meeting_input",  # Must be "meeting_input" for node to recognize
            content="Analyze this meeting transcript",
            metadata={  # Node reads from metadata, not payload
                "transcript": sample_meeting_input["transcript"],
                "title": sample_meeting_input.get("title"),
                "meeting_date": sample_meeting_input.get("meeting_date"),
                "participants": sample_meeting_input.get("participants", []),
                "deal_id": sample_meeting_input.get("deal_id"),
                "contact_id": sample_meeting_input.get("contact_id"),
                "additional_context": sample_meeting_input.get("additional_context"),
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    ]
    state["status"] = WorkflowStatus.IN_PROGRESS
    return state


# =============================================================================
# WORKFLOW STRUCTURE TESTS
# =============================================================================

class TestWorkflowStructure:
    """Tests for workflow structure and configuration."""

    def test_meeting_analysis_workflow_exists(self):
        """Meeting analysis workflow should be registered."""
        assert "meeting_analysis" in WORKFLOW_BUILDERS

    def test_meeting_analysis_workflow_builds(self):
        """Should build meeting analysis workflow without errors."""
        workflow = build_meeting_analysis_workflow()
        assert workflow is not None

    def test_workflow_compiles(self):
        """Should compile workflow with memory checkpointer."""
        compiled = compile_workflow("meeting_analysis", "memory")
        assert compiled is not None

    def test_workflow_has_required_nodes(self):
        """Workflow should have all required MVP nodes."""
        workflow = build_meeting_analysis_workflow()
        compiled = workflow.compile()

        # Get node names from the compiled graph
        graph = compiled.get_graph()
        node_names = set(graph.nodes.keys())

        required_nodes = {
            "meeting_notes",
            "task_extractor",
            "crm_updater",
            "human_approval",
            "finalize",
        }

        for node in required_nodes:
            assert node in node_names, f"Missing required node: {node}"


# =============================================================================
# ROUTER TESTS
# =============================================================================

class TestRouterLogic:
    """Tests for routing logic in Sales Ops workflow."""

    def test_route_to_meeting_notes_first(self, initial_meeting_state):
        """Should route to meeting_notes when no notes exist."""
        route = route_from_sales_ops(initial_meeting_state)
        assert route == "meeting_notes"

    def test_route_to_task_extractor_after_notes(self, initial_meeting_state):
        """Should route to task_extractor after meeting notes created."""
        # Add meeting notes to state
        initial_meeting_state["meeting_notes"] = [
            MeetingNote(
                meeting_id=str(uuid4()),
                transcript_id="test",
                meeting_date="2024-01-15",
                title="Test Meeting",
                participants=["John"],
                summary="Test summary",
                key_points=["Point 1"],
                action_items=[{"task": "Test task"}],
                sentiment="positive",
                follow_up_required=True,
            )
        ]

        route = route_from_sales_ops(initial_meeting_state)
        assert route == "task_extractor"

    def test_route_to_crm_updater_after_tasks(self, initial_meeting_state):
        """Should route to crm_updater when pending tasks exist."""
        initial_meeting_state["meeting_notes"] = [MeetingNote(
            meeting_id=str(uuid4()),
            transcript_id="test",
            meeting_date="2024-01-15",
            title="Test",
            participants=[],
            summary="Test",
            key_points=[],
            action_items=[],
            sentiment="neutral",
            follow_up_required=False,
        )]
        initial_meeting_state["crm_tasks"] = [
            CRMTask(
                task_id=str(uuid4()),
                task_type="create_task",
                entity_type="task",
                entity_id=None,
                payload={},
                priority="medium",
                status="pending",
                error_message=None,
                created_at=datetime.now(timezone.utc).isoformat(),
                executed_at=None,
            )
        ]

        route = route_from_sales_ops(initial_meeting_state)
        assert route == "crm_updater"

    def test_route_to_done_when_complete(self, initial_meeting_state):
        """Should route to done when all tasks are processed."""
        initial_meeting_state["meeting_notes"] = [MeetingNote(
            meeting_id=str(uuid4()),
            transcript_id="test",
            meeting_date="2024-01-15",
            title="Test",
            participants=[],
            summary="Test",
            key_points=[],
            action_items=[],
            sentiment="neutral",
            follow_up_required=False,
        )]
        initial_meeting_state["crm_tasks"] = [
            CRMTask(
                task_id=str(uuid4()),
                task_type="create_task",
                entity_type="task",
                entity_id="hubspot-123",
                payload={},
                priority="medium",
                status="executed",  # Already executed
                error_message=None,
                created_at=datetime.now(timezone.utc).isoformat(),
                executed_at=datetime.now(timezone.utc).isoformat(),
            )
        ]

        route = route_from_sales_ops(initial_meeting_state)
        assert route == "done"

    def test_approval_check_with_pending_approvals(self, initial_meeting_state):
        """Should return needs_approval when approvals are pending."""
        initial_meeting_state["pending_approvals"] = [
            ApprovalRequest(
                approval_id=str(uuid4()),
                approval_type=ApprovalType.CRM_UPDATE,
                title="Test Approval",
                description="Test",
                payload={},
                requested_at=datetime.now(timezone.utc).isoformat(),
                requested_by="crm_updater",
                status="pending",
            )
        ]

        result = should_request_approval(initial_meeting_state)
        assert result == "needs_approval"

    def test_no_approval_when_empty(self, initial_meeting_state):
        """Should return no_approval when no pending approvals."""
        result = should_request_approval(initial_meeting_state)
        assert result == "no_approval"


# =============================================================================
# NODE INTEGRATION TESTS
# =============================================================================

class TestNodeIntegration:
    """Tests for individual node execution with mocks."""

    def test_meeting_notes_node_extracts_input(self, initial_meeting_state, mock_meeting_analysis):
        """Meeting notes node should extract input from messages."""
        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm
            mock_llm.generate_structured = AsyncMock(
                return_value=(mock_meeting_analysis, MagicMock(total_tokens=500))
            )

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = ("system", "user")

                result = meeting_notes_node(initial_meeting_state)

        assert "meeting_notes" in result
        assert len(result["meeting_notes"]) == 1

    def test_task_extractor_processes_action_items(
        self, initial_meeting_state, mock_meeting_analysis, mock_task_extraction_result
    ):
        """Task extractor should process action items from meeting notes."""
        # Add meeting note to state
        initial_meeting_state["meeting_notes"] = [
            MeetingNote(
                meeting_id=str(uuid4()),
                transcript_id="test",
                meeting_date="2024-01-15",
                title="Q1 Strategy",
                participants=["John", "Sarah", "Mike"],
                summary=mock_meeting_analysis.summary,
                key_points=mock_meeting_analysis.key_points,
                action_items=[
                    {
                        "task": item.task,
                        "assignee": item.assignee,
                        "due_date": item.due_date,
                        "priority": item.priority,
                        "context": item.context,
                    }
                    for item in mock_meeting_analysis.action_items
                ],
                sentiment="positive",
                follow_up_required=True,
            )
        ]

        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm
            mock_llm.generate_structured = AsyncMock(
                return_value=(mock_task_extraction_result, MagicMock(total_tokens=400))
            )

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = ("system", "user")

                result = task_extractor_node(initial_meeting_state)

        assert "crm_tasks" in result
        assert len(result["crm_tasks"]) == 3

    def test_crm_updater_creates_approval(
        self, initial_meeting_state, mock_crm_operations_result
    ):
        """CRM updater should create approval request."""
        # Add pending CRM tasks
        initial_meeting_state["crm_tasks"] = [
            CRMTask(
                task_id=str(uuid4()),
                task_type="create_task",
                entity_type="task",
                entity_id=None,
                payload={
                    "hubspot_task": {
                        "hs_task_subject": "Test Task",
                        "hs_task_body": "Test body",
                        "hs_task_status": "NOT_STARTED",
                        "hs_task_priority": "MEDIUM",
                        "hs_task_type": "TODO",
                    },
                    "needs_review": False,
                },
                priority="medium",
                status="pending",
                error_message=None,
                created_at=datetime.now(timezone.utc).isoformat(),
                executed_at=None,
            )
        ]

        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm
            mock_llm.generate_structured = AsyncMock(
                return_value=(mock_crm_operations_result, MagicMock(total_tokens=600))
            )

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = ("system", "user")

                result = crm_updater_node(initial_meeting_state)

        assert "pending_approvals" in result
        assert len(result["pending_approvals"]) == 1
        assert result["pending_approvals"][0]["approval_type"] == ApprovalType.CRM_UPDATE


# =============================================================================
# E2E WORKFLOW TESTS
# =============================================================================

class TestE2EWorkflow:
    """End-to-end workflow tests with full pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_mocks(
        self,
        initial_meeting_state,
        mock_meeting_analysis,
        mock_task_extraction_result,
        mock_crm_operations_result,
    ):
        """
        E2E Test: Full pipeline from meeting transcript to CRM approval.

        Verifies:
        1. Meeting notes extraction
        2. Task extraction from action items
        3. CRM update preparation
        4. Approval request creation
        """
        # Patch all LLM calls
        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm

            # Configure LLM responses for each node
            async def mock_generate(output_schema, prompt, system_prompt):
                if "MeetingAnalysis" in str(output_schema):
                    return (mock_meeting_analysis, MagicMock(total_tokens=500))
                elif "TaskExtractionResult" in str(output_schema):
                    return (mock_task_extraction_result, MagicMock(total_tokens=400))
                elif "CRMUpdateOperationResult" in str(output_schema):
                    return (mock_crm_operations_result, MagicMock(total_tokens=600))
                else:
                    raise ValueError(f"Unknown schema: {output_schema}")

            mock_llm.generate_structured = AsyncMock(side_effect=mock_generate)

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = ("system", "user")

                # Execute nodes sequentially (simulating workflow)
                state = initial_meeting_state

                # Step 1: Meeting Notes
                result1 = meeting_notes_node(state)
                state = {**state, **result1}

                assert "meeting_notes" in state
                assert len(state["meeting_notes"]) == 1

                # Step 2: Task Extractor
                result2 = task_extractor_node(state)
                state = {**state, **result2}

                assert "crm_tasks" in state
                assert len(state["crm_tasks"]) == 3

                # Step 3: CRM Updater
                result3 = crm_updater_node(state)
                state = {**state, **result3}

                assert "pending_approvals" in state
                assert len(state["pending_approvals"]) == 1

                # Verify final state
                approval = state["pending_approvals"][0]
                assert approval["approval_type"] == ApprovalType.CRM_UPDATE
                assert approval["status"] == "pending"
                assert "operations" in approval["payload"]

    def test_workflow_progress_tracking(self, initial_meeting_state):
        """Should track workflow progress correctly."""
        # Initial progress (no data)
        progress = get_workflow_progress(initial_meeting_state)
        assert progress["sales_ops"]["meeting_notes"] is False

        # After meeting notes
        initial_meeting_state["meeting_notes"] = [MeetingNote(
            meeting_id=str(uuid4()),
            transcript_id="test",
            meeting_date="2024-01-15",
            title="Test",
            participants=[],
            summary="Test",
            key_points=[],
            action_items=[],
            sentiment="neutral",
            follow_up_required=False,
        )]
        progress = get_workflow_progress(initial_meeting_state)
        assert progress["sales_ops"]["meeting_notes"] is True
        assert progress["sales_ops"]["task_extractor"] is False

        # After task extraction
        initial_meeting_state["crm_tasks"] = [CRMTask(
            task_id=str(uuid4()),
            task_type="create_task",
            entity_type="task",
            entity_id=None,
            payload={},
            priority="medium",
            status="pending",
            error_message=None,
            created_at=datetime.now(timezone.utc).isoformat(),
            executed_at=None,
        )]
        progress = get_workflow_progress(initial_meeting_state)
        assert progress["sales_ops"]["task_extractor"] is True


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Tests for error handling across the pipeline."""

    def test_meeting_notes_handles_no_input(self):
        """Should handle state with no meeting input gracefully."""
        empty_state = create_initial_state(
            client_id="test",
            client_name="Test",
            crm_provider=CRMProvider.HUBSPOT,
            workflow_type="meeting_analysis",
        )
        empty_state["status"] = WorkflowStatus.IN_PROGRESS

        result = meeting_notes_node(empty_state)

        # Should return error message without crashing
        assert "messages" in result

    def test_task_extractor_handles_no_action_items(self):
        """Should handle meeting notes with no action items."""
        state = create_initial_state(
            client_id="test",
            client_name="Test",
            crm_provider=CRMProvider.HUBSPOT,
            workflow_type="meeting_analysis",
        )
        state["meeting_notes"] = [MeetingNote(
            meeting_id=str(uuid4()),
            transcript_id="test",
            meeting_date="2024-01-15",
            title="Test",
            participants=[],
            summary="Test",
            key_points=[],
            action_items=[],  # No action items
            sentiment="neutral",
            follow_up_required=False,
        )]

        result = task_extractor_node(state)

        # Should not crash, may produce empty tasks
        assert "messages" in result or "crm_tasks" in result

    def test_crm_updater_handles_no_pending_tasks(self):
        """Should handle state with no pending CRM tasks."""
        state = create_initial_state(
            client_id="test",
            client_name="Test",
            crm_provider=CRMProvider.HUBSPOT,
            workflow_type="meeting_analysis",
        )
        state["crm_tasks"] = []  # No tasks

        result = crm_updater_node(state)

        # Should return info message, not crash
        assert "messages" in result
        assert len(result.get("pending_approvals", [])) == 0

    def test_llm_error_fallback(self, initial_meeting_state):
        """Should fall back gracefully when LLM fails."""
        from backend.services.llm_service import LLMError

        # Add pending tasks
        initial_meeting_state["crm_tasks"] = [CRMTask(
            task_id=str(uuid4()),
            task_type="create_task",
            entity_type="task",
            entity_id=None,
            payload={"hubspot_task": {"hs_task_subject": "Test"}},
            priority="medium",
            status="pending",
            error_message=None,
            created_at=datetime.now(timezone.utc).isoformat(),
            executed_at=None,
        )]

        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm
            mock_llm.generate_structured = AsyncMock(
                side_effect=LLMError("API Error", retryable=True)
            )

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = ("system", "user")

                result = crm_updater_node(initial_meeting_state)

        # Should still create approval in fallback mode
        assert "pending_approvals" in result
        assert len(result["pending_approvals"]) == 1
        assert result["pending_approvals"][0]["payload"].get("fallback_mode") is True


# =============================================================================
# PHASE 3.4 VERIFICATION
# =============================================================================

class TestPhase34Verification:
    """
    Phase 3.4 Verification: Integration & E2E Testing
    Criteria: Full pipeline works from meeting input to CRM approval
    """

    def test_all_mvp_nodes_importable(self):
        """Should be able to import all MVP nodes."""
        from backend.graph.nodes import (
            meeting_notes_node,
            task_extractor_node,
            crm_updater_node,
        )
        assert meeting_notes_node is not None
        assert task_extractor_node is not None
        assert crm_updater_node is not None

    def test_workflow_compiles_without_error(self):
        """Should compile meeting_analysis workflow."""
        compiled = compile_workflow("meeting_analysis", "memory")
        assert compiled is not None

    def test_routers_importable(self):
        """Should be able to import all routers."""
        from backend.graph.routers import (
            route_from_sales_ops,
            should_request_approval,
            get_workflow_progress,
        )
        assert route_from_sales_ops is not None
        assert should_request_approval is not None
        assert get_workflow_progress is not None

    def test_state_factory_works(self):
        """Should create initial state correctly."""
        state = create_initial_state(
            client_id="test-123",
            client_name="Test Company",
            crm_provider=CRMProvider.HUBSPOT,
            workflow_type="meeting_analysis",
        )

        assert state["workflow_type"] == "meeting_analysis"
        assert state["client"]["client_name"] == "Test Company"
        assert state["meeting_notes"] == []
        assert state["crm_tasks"] == []
        assert state["pending_approvals"] == []

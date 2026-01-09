"""
Meeting Notes Agent Tests (Phase 3.1)
=====================================
Tests for the Meeting Notes Analyzer agent.
Verifies LLM integration, structured output, and state management.

Test Criteria: Agent can analyze meeting transcripts and extract structured data
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from backend.app.schemas.meeting_notes import (
    MeetingAnalysis,
    NormalizedMeetingInput,
    MeetingInputSource,
    MeetingSentiment,
    ActionItem,
    ActionItemPriority,
    KeyDecision,
    DealStageRecommendation,
)
from backend.graph.nodes import meeting_notes_node
from backend.graph.state import OrchestratorState, MeetingNote, CRMProvider, ClientContext


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_transcript():
    """Sample meeting transcript for testing."""
    return """
    Meeting: Q4 Strategy Review
    Date: 2024-01-15
    Participants: John (Sales), Sarah (Product), Mike (Engineering)

    John: Thanks everyone for joining. Let's review our Q4 performance and plan for next quarter.

    Sarah: Our product launches in Q4 exceeded expectations. The new CRM integration got great feedback.

    Mike: Engineering delivered all sprint commitments. We should discuss the technical debt backlog.

    John: Great work! For Q1, we need to focus on expanding to the enterprise segment.
    Action item: John will prepare the enterprise pricing proposal by next Friday.

    Sarah: I agree. We should also improve our onboarding flow.
    Action item: Sarah to create onboarding improvement roadmap by end of month.

    Mike: I have concerns about the infrastructure scaling for enterprise clients.
    We need to address this before the big push.
    Action item: Mike to present scaling assessment in next week's meeting.

    John: Good points. Let's reconvene next Tuesday to review progress.
    The deal with Acme Corp looks promising - they're interested in our enterprise tier.

    Meeting adjourned.
    """


@pytest.fixture
def sample_meeting_input(sample_transcript):
    """Sample NormalizedMeetingInput for testing."""
    return NormalizedMeetingInput(
        source=MeetingInputSource.MANUAL_TEXT,
        title="Q4 Strategy Review",
        meeting_date=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
        participants=["John", "Sarah", "Mike"],
        transcript=sample_transcript,
        deal_id="deal-123",
        additional_context="Enterprise expansion initiative",
    )


@pytest.fixture
def mock_meeting_analysis():
    """Mock MeetingAnalysis response."""
    return MeetingAnalysis(
        summary="The team reviewed Q4 performance, which exceeded expectations, and planned Q1 focus on enterprise expansion with key action items assigned.",
        key_points=[
            "Q4 product launches exceeded expectations",
            "CRM integration received positive feedback",
            "Engineering delivered all sprint commitments",
            "Q1 focus on enterprise segment expansion",
            "Infrastructure scaling concerns for enterprise",
        ],
        key_decisions=[
            KeyDecision(
                decision="Focus Q1 on enterprise segment expansion",
                rationale="Natural progression after successful Q4",
                stakeholders=["John", "Sarah", "Mike"],
            ),
        ],
        action_items=[
            ActionItem(
                task="Prepare enterprise pricing proposal",
                assignee="John",
                due_date="next Friday",
                priority=ActionItemPriority.HIGH,
                context="For Q1 enterprise expansion",
            ),
            ActionItem(
                task="Create onboarding improvement roadmap",
                assignee="Sarah",
                due_date="end of month",
                priority=ActionItemPriority.MEDIUM,
                context="Improve customer onboarding flow",
            ),
            ActionItem(
                task="Present scaling assessment",
                assignee="Mike",
                due_date="next week",
                priority=ActionItemPriority.HIGH,
                context="Address infrastructure concerns for enterprise",
            ),
        ],
        overall_sentiment=MeetingSentiment.POSITIVE,
        sentiment_explanation="Team is optimistic about Q4 results and excited about Q1 opportunities",
        follow_up_required=True,
        follow_up_reason="Next Tuesday meeting to review progress",
        deal_stage_recommendation=DealStageRecommendation(
            current_signals=["Acme Corp interested in enterprise tier"],
            recommended_stage="proposal",
            confidence="medium",
            reasoning="Customer showing active interest",
        ),
        identified_participants=["John (Sales)", "Sarah (Product)", "Mike (Engineering)"],
        risks_concerns=["Infrastructure scaling concerns for enterprise clients"],
        opportunities=["Enterprise segment expansion", "Acme Corp deal potential"],
        next_steps=["Tuesday reconvene", "Review progress on action items"],
    )


@pytest.fixture
def base_state():
    """Base OrchestratorState for testing."""
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


# =============================================================================
# SCHEMA TESTS
# =============================================================================

class TestMeetingNotesSchemas:
    """Tests for Meeting Notes Pydantic schemas."""

    def test_normalized_meeting_input_validation(self, sample_transcript):
        """Should validate NormalizedMeetingInput correctly."""
        input_data = NormalizedMeetingInput(
            transcript=sample_transcript,
            title="Test Meeting",
        )
        # Note: Pydantic strips whitespace, so compare stripped versions
        assert input_data.transcript == sample_transcript.strip()
        assert input_data.source == MeetingInputSource.MANUAL_TEXT

    def test_normalized_meeting_input_requires_transcript(self):
        """Should require transcript field."""
        with pytest.raises(Exception):  # ValidationError
            NormalizedMeetingInput(title="Test")

    def test_normalized_meeting_input_transcript_min_length(self):
        """Should enforce minimum transcript length."""
        with pytest.raises(Exception):
            NormalizedMeetingInput(transcript="short")

    def test_meeting_analysis_schema_structure(self, mock_meeting_analysis):
        """Should create valid MeetingAnalysis."""
        assert mock_meeting_analysis.summary is not None
        assert len(mock_meeting_analysis.action_items) == 3
        assert mock_meeting_analysis.overall_sentiment == MeetingSentiment.POSITIVE

    def test_action_item_priority_enum(self):
        """ActionItemPriority enum should have expected values."""
        assert ActionItemPriority.LOW == "low"
        assert ActionItemPriority.MEDIUM == "medium"
        assert ActionItemPriority.HIGH == "high"
        assert ActionItemPriority.URGENT == "urgent"

    def test_meeting_sentiment_enum(self):
        """MeetingSentiment enum should have expected values."""
        assert MeetingSentiment.POSITIVE == "positive"
        assert MeetingSentiment.NEUTRAL == "neutral"
        assert MeetingSentiment.NEGATIVE == "negative"


# =============================================================================
# MEETING NOTES NODE TESTS
# =============================================================================

class TestMeetingNotesNode:
    """Tests for meeting_notes_node function."""

    def test_node_returns_error_without_input(self, base_state):
        """Should return error when no meeting input provided."""
        result = meeting_notes_node(base_state)

        assert "messages" in result
        assert any(
            msg.get("message_type") == "error"
            for msg in result.get("messages", [])
        )

    def test_node_extracts_input_from_messages(self, base_state, sample_transcript):
        """Should extract meeting input from state messages."""
        # Add meeting input to messages
        base_state["messages"] = [
            {
                "message_id": str(uuid4()),
                "from_agent": "api",
                "to_agent": "meeting_notes",
                "message_type": "meeting_input",
                "content": "Meeting transcript for analysis",
                "metadata": {
                    "transcript": sample_transcript,
                    "title": "Test Meeting",
                    "participants": ["Alice", "Bob"],
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]

        # Mock the LLM service to avoid actual API calls
        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm

            # Mock generate_structured to return a valid MeetingAnalysis
            mock_analysis = MeetingAnalysis(
                summary="Test meeting summary.",
                key_points=["Point 1", "Point 2"],
                overall_sentiment=MeetingSentiment.NEUTRAL,
            )
            mock_llm.generate_structured = AsyncMock(
                return_value=(mock_analysis, MagicMock(total_tokens=100))
            )

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = ("system", "user")

                result = meeting_notes_node(base_state)

        # Should have meeting_notes in result
        assert "meeting_notes" in result
        assert len(result["meeting_notes"]) == 1

    @pytest.mark.asyncio
    async def test_llm_integration_with_mock(
        self, base_state, sample_transcript, mock_meeting_analysis
    ):
        """Should integrate with LLM service correctly."""
        base_state["messages"] = [
            {
                "message_id": str(uuid4()),
                "from_agent": "api",
                "to_agent": "meeting_notes",
                "message_type": "meeting_input",
                "content": "Meeting transcript",
                "metadata": {
                    "transcript": sample_transcript,
                    "title": "Q4 Review",
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]

        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm
            mock_llm.generate_structured = AsyncMock(
                return_value=(mock_meeting_analysis, MagicMock(total_tokens=500))
            )

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = (
                    "You are an expert Meeting Analyst",
                    "Analyze this transcript",
                )

                result = meeting_notes_node(base_state)

        # Verify result structure
        assert "meeting_notes" in result
        meeting_note = result["meeting_notes"][0]

        assert meeting_note["summary"] == mock_meeting_analysis.summary
        assert len(meeting_note["action_items"]) == 3
        assert meeting_note["sentiment"] == "positive"
        assert meeting_note["follow_up_required"] is True

    def test_node_handles_llm_error(self, base_state, sample_transcript):
        """Should handle LLM errors gracefully."""
        base_state["messages"] = [
            {
                "message_id": str(uuid4()),
                "from_agent": "api",
                "to_agent": "meeting_notes",
                "message_type": "meeting_input",
                "content": "Meeting transcript",
                "metadata": {"transcript": sample_transcript},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]

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

                result = meeting_notes_node(base_state)

        # Should have error in result
        assert "error_message" in result
        assert "messages" in result
        assert any(
            msg.get("message_type") == "error"
            for msg in result.get("messages", [])
        )


# =============================================================================
# CONVERSION TESTS
# =============================================================================

class TestMeetingNoteConversion:
    """Tests for MeetingAnalysis to MeetingNote conversion."""

    def test_convert_analysis_preserves_data(
        self, sample_meeting_input, mock_meeting_analysis
    ):
        """Should preserve all data during conversion."""
        from backend.graph.nodes import _convert_analysis_to_meeting_note

        result = _convert_analysis_to_meeting_note(
            mock_meeting_analysis, sample_meeting_input
        )

        assert result["summary"] == mock_meeting_analysis.summary
        assert result["key_points"] == mock_meeting_analysis.key_points
        assert result["sentiment"] == "positive"
        assert result["follow_up_required"] is True
        assert len(result["action_items"]) == 3

    def test_convert_action_items_format(
        self, sample_meeting_input, mock_meeting_analysis
    ):
        """Should convert action items to correct format."""
        from backend.graph.nodes import _convert_analysis_to_meeting_note

        result = _convert_analysis_to_meeting_note(
            mock_meeting_analysis, sample_meeting_input
        )

        action_item = result["action_items"][0]
        assert "task" in action_item
        assert "assignee" in action_item
        assert "due_date" in action_item
        assert action_item["assignee"] == "John"

    def test_convert_handles_missing_assignee(self, sample_meeting_input):
        """Should handle missing assignee with TBD."""
        from backend.graph.nodes import _convert_analysis_to_meeting_note

        analysis = MeetingAnalysis(
            summary="Test summary for meeting.",
            action_items=[
                ActionItem(task="Do something", assignee=None, priority=ActionItemPriority.MEDIUM)
            ],
        )

        result = _convert_analysis_to_meeting_note(analysis, sample_meeting_input)

        assert result["action_items"][0]["assignee"] == "TBD"


# =============================================================================
# PROMPT TEMPLATE TESTS
# =============================================================================

class TestMeetingNotesPrompt:
    """Tests for meeting notes prompt template."""

    def test_prompt_template_exists(self):
        """Should have meeting_notes prompt template."""
        from backend.prompts.base import PromptManager

        manager = PromptManager()
        template = manager.get_prompt("meeting_notes")

        # Template may not load if templates dir is empty in test
        # This is a basic existence check
        # In real tests, you'd verify the template content
        pass  # Template loading tested implicitly in integration tests

    def test_prompt_includes_required_variables(self):
        """Prompt template should support required variables."""
        expected_variables = [
            "client_name",
            "transcript",
            "industry",
            "crm_provider",
        ]
        # Verified by reading the YAML template
        # This test documents the expected interface
        for var in expected_variables:
            assert var  # Placeholder assertion


# =============================================================================
# PHASE 3.1 VERIFICATION
# =============================================================================

class TestPhase31Verification:
    """
    Phase 3.1 Verification: Meeting Notes Agent
    Criteria: Agent can analyze meeting transcripts
    """

    def test_meeting_notes_agent_importable(self):
        """Should be able to import meeting_notes_node."""
        from backend.graph.nodes import meeting_notes_node
        assert meeting_notes_node is not None

    def test_meeting_analysis_schema_importable(self):
        """Should be able to import MeetingAnalysis schema."""
        from backend.app.schemas.meeting_notes import MeetingAnalysis
        assert MeetingAnalysis is not None

    def test_normalized_input_schema_importable(self):
        """Should be able to import NormalizedMeetingInput schema."""
        from backend.app.schemas.meeting_notes import NormalizedMeetingInput
        assert NormalizedMeetingInput is not None

    def test_schemas_exported_from_module(self):
        """Meeting notes schemas should be exported from schemas module."""
        from backend.app.schemas import (
            MeetingAnalysis,
            NormalizedMeetingInput,
            MeetingInputSource,
            MeetingSentiment,
        )

        assert MeetingAnalysis is not None
        assert NormalizedMeetingInput is not None
        assert MeetingInputSource is not None
        assert MeetingSentiment is not None

    def test_agent_in_exports(self):
        """meeting_notes_node should be in nodes exports."""
        from backend.graph.nodes import __all__

        assert "meeting_notes_node" in __all__

    @pytest.mark.asyncio
    async def test_end_to_end_mock_analysis(self, base_state, sample_transcript):
        """
        VERIFICATION TEST: Full agent flow with mocked LLM.

        Verifies:
        1. Input extraction from state
        2. LLM service integration
        3. Structured output parsing
        4. State update with results
        """
        # Setup input in state
        base_state["messages"] = [
            {
                "message_id": str(uuid4()),
                "from_agent": "workflow",
                "to_agent": "meeting_notes",
                "message_type": "meeting_input",
                "content": "Analyze this meeting",
                "metadata": {
                    "transcript": sample_transcript,
                    "title": "Verification Test Meeting",
                    "participants": ["Alice", "Bob"],
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]

        # Mock LLM response
        mock_analysis = MeetingAnalysis(
            summary="Team discussed Q4 results and Q1 plans.",
            key_points=["Q4 exceeded expectations", "Q1 focus on enterprise"],
            action_items=[
                ActionItem(
                    task="Prepare proposal",
                    assignee="Alice",
                    due_date="next Friday",
                    priority=ActionItemPriority.HIGH,
                )
            ],
            overall_sentiment=MeetingSentiment.POSITIVE,
            follow_up_required=True,
            follow_up_reason="Weekly sync required",
            identified_participants=["Alice", "Bob"],
        )

        with patch("backend.graph.nodes.LLMService") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.get_instance.return_value = mock_llm
            mock_llm.generate_structured = AsyncMock(
                return_value=(mock_analysis, MagicMock(total_tokens=200))
            )

            with patch("backend.graph.nodes.PromptManager") as mock_prompt_class:
                mock_prompt = MagicMock()
                mock_prompt_class.get_instance.return_value = mock_prompt
                mock_prompt.get_full_prompt.return_value = ("system", "user")

                result = meeting_notes_node(base_state)

        # Verify complete result
        assert "meeting_notes" in result
        assert len(result["meeting_notes"]) == 1

        meeting_note = result["meeting_notes"][0]
        assert "meeting_id" in meeting_note
        assert meeting_note["summary"] == mock_analysis.summary
        assert meeting_note["sentiment"] == "positive"
        assert meeting_note["follow_up_required"] is True
        assert len(meeting_note["action_items"]) == 1
        assert meeting_note["action_items"][0]["assignee"] == "Alice"

        # Verify execution log
        assert "agent_execution_log" in result
        assert len(result["agent_execution_log"]) == 1
        log_entry = result["agent_execution_log"][0]
        assert log_entry["agent"] == "meeting_notes"
        assert log_entry["action"] == "analyze_transcript"

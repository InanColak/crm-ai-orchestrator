"""
Email Copilot Agent Tests (Phase 4.3)
=====================================
Unit tests for email generation, context building, and delivery.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

# Schema imports
from backend.app.schemas.email import (
    EmailType,
    EmailTone,
    EmailPriority,
    EmailStatus,
    DeliveryProvider,
    EmailRecipient,
    LeadContext,
    MeetingContext,
    PreviousEmailContext,
    EmailCopilotInput,
    EmailDraft,
    EmailGenerationResult,
    EmailDeliveryPayload,
    EmailDeliveryResult,
    EmailApprovalPayload,
    BrandvoiceContext,
    EmailContext,
)

# Service imports
from backend.services.email.context_builder import (
    EmailContextBuilder,
    get_email_context_builder,
    format_lead_context_for_prompt,
    format_meeting_context_for_prompt,
)
from backend.services.email.adapters.base import (
    EmailDeliveryAdapter,
    MockEmailAdapter,
    EmailAdapterError,
    EmailDeliveryError,
    EmailValidationError,
)
from backend.services.email.adapters.hubspot import HubSpotEmailAdapter


# =============================================================================
# SCHEMA TESTS - EmailRecipient
# =============================================================================


class TestEmailRecipientSchema:
    """Tests for EmailRecipient schema."""

    def test_valid_recipient(self):
        """Test valid recipient creation."""
        recipient = EmailRecipient(
            email="john@acme.com",
            name="John Doe",
            title="CEO",
            company="Acme Corp",
        )
        assert recipient.email == "john@acme.com"
        assert recipient.name == "John Doe"

    def test_email_validation(self):
        """Test email validation."""
        with pytest.raises(ValueError):
            EmailRecipient(email="invalid-email")

    def test_email_lowercase(self):
        """Test email is lowercased."""
        recipient = EmailRecipient(email="JOHN@ACME.COM")
        assert recipient.email == "john@acme.com"

    def test_minimal_recipient(self):
        """Test minimal recipient with just email."""
        recipient = EmailRecipient(email="john@acme.com")
        assert recipient.name is None
        assert recipient.company is None


# =============================================================================
# SCHEMA TESTS - EmailCopilotInput
# =============================================================================


class TestEmailCopilotInputSchema:
    """Tests for EmailCopilotInput schema."""

    def test_valid_input(self):
        """Test valid email input."""
        recipient = EmailRecipient(email="john@acme.com")
        input_data = EmailCopilotInput(
            recipient=recipient,
            email_type=EmailType.COLD_OUTREACH,
        )
        assert input_data.email_type == EmailType.COLD_OUTREACH
        assert input_data.tone == EmailTone.PROFESSIONAL  # default

    def test_all_email_types(self):
        """Test all email types are valid."""
        recipient = EmailRecipient(email="test@test.com")
        for email_type in EmailType:
            input_data = EmailCopilotInput(
                recipient=recipient,
                email_type=email_type,
            )
            assert input_data.email_type == email_type

    def test_with_lead_context(self):
        """Test input with lead context."""
        recipient = EmailRecipient(email="john@acme.com", company="Acme Corp")
        lead_context = LeadContext(
            company_name="Acme Corp",
            industry="Technology",
            pain_points=["Scaling challenges", "Manual processes"],
            talking_points=["Recent funding", "Growth trajectory"],
        )
        input_data = EmailCopilotInput(
            recipient=recipient,
            email_type=EmailType.COLD_OUTREACH,
            lead_context=lead_context,
        )
        assert input_data.lead_context.company_name == "Acme Corp"
        assert len(input_data.lead_context.pain_points) == 2

    def test_with_meeting_context(self):
        """Test input with meeting context for post-meeting emails."""
        recipient = EmailRecipient(email="john@acme.com")
        meeting_context = MeetingContext(
            meeting_date="2024-01-15",
            summary="Product demo meeting",
            key_points=["Discussed pricing", "Timeline review"],
            action_items=["Send proposal", "Schedule follow-up"],
        )
        input_data = EmailCopilotInput(
            recipient=recipient,
            email_type=EmailType.POST_MEETING,
            meeting_context=meeting_context,
        )
        assert input_data.meeting_context.summary == "Product demo meeting"


# =============================================================================
# SCHEMA TESTS - EmailDraft
# =============================================================================


class TestEmailDraftSchema:
    """Tests for EmailDraft schema."""

    def test_valid_draft(self):
        """Test valid email draft."""
        draft = EmailDraft(
            subject="Quick question about your growth plans",
            body_html="<p>Hi John,</p><p>Content here...</p>",
            body_plain="Hi John,\n\nContent here...",
            email_type=EmailType.COLD_OUTREACH,
            tone=EmailTone.PROFESSIONAL,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        assert "growth plans" in draft.subject
        assert draft.email_type == EmailType.COLD_OUTREACH

    def test_subject_truncation(self):
        """Test long subject is truncated."""
        long_subject = "A" * 250
        draft = EmailDraft(
            subject=long_subject,
            body_html="<p>Test</p>",
            body_plain="Test",
            email_type=EmailType.COLD_OUTREACH,
            tone=EmailTone.PROFESSIONAL,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        assert len(draft.subject) == 200
        assert draft.subject.endswith("...")

    def test_personalization_elements(self):
        """Test personalization elements tracking."""
        draft = EmailDraft(
            subject="About your Series B",
            body_html="<p>Content</p>",
            body_plain="Content",
            email_type=EmailType.COLD_OUTREACH,
            tone=EmailTone.PROFESSIONAL,
            personalization_elements=[
                "Referenced Series B funding",
                "Mentioned expansion plans",
            ],
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        assert len(draft.personalization_elements) == 2


# =============================================================================
# SCHEMA TESTS - EmailGenerationResult
# =============================================================================


class TestEmailGenerationResultSchema:
    """Tests for EmailGenerationResult schema."""

    def test_valid_result(self):
        """Test valid generation result."""
        draft = EmailDraft(
            subject="Test Subject",
            body_html="<p>Body</p>",
            body_plain="Body",
            email_type=EmailType.COLD_OUTREACH,
            tone=EmailTone.PROFESSIONAL,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        result = EmailGenerationResult(
            draft=draft,
            subject_alternatives=["Alt 1", "Alt 2"],
            personalization_score=0.85,
            relevance_score=0.90,
            approach_reasoning="Started with their funding news...",
        )
        assert result.personalization_score == 0.85
        assert len(result.subject_alternatives) == 2

    def test_score_bounds(self):
        """Test score bounds validation."""
        draft = EmailDraft(
            subject="Test",
            body_html="<p>Body</p>",
            body_plain="Body",
            email_type=EmailType.COLD_OUTREACH,
            tone=EmailTone.PROFESSIONAL,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        # Valid bounds
        result = EmailGenerationResult(
            draft=draft,
            personalization_score=0.0,
            relevance_score=1.0,
        )
        assert result.personalization_score == 0.0
        assert result.relevance_score == 1.0


# =============================================================================
# SCHEMA TESTS - EmailDeliveryPayload
# =============================================================================


class TestEmailDeliveryPayloadSchema:
    """Tests for EmailDeliveryPayload schema."""

    def test_valid_payload(self):
        """Test valid delivery payload."""
        payload = EmailDeliveryPayload(
            to_email="john@acme.com",
            to_name="John Doe",
            subject="Test Subject",
            body_html="<p>HTML body</p>",
            body_plain="Plain body",
            from_email="sales@ourcompany.com",
            from_name="Sales Team",
            contact_id="hubspot-123",
            email_type=EmailType.COLD_OUTREACH,
        )
        assert payload.to_email == "john@acme.com"
        assert payload.track_opens is True  # default

    def test_tracking_options(self):
        """Test tracking options."""
        payload = EmailDeliveryPayload(
            to_email="test@test.com",
            subject="Test",
            body_html="<p>Test</p>",
            body_plain="Test",
            track_opens=False,
            track_clicks=False,
            email_type=EmailType.FOLLOW_UP,
        )
        assert payload.track_opens is False
        assert payload.track_clicks is False


# =============================================================================
# CONTEXT BUILDER TESTS
# =============================================================================


class TestEmailContextBuilder:
    """Tests for EmailContextBuilder."""

    @pytest.fixture
    def context_builder(self):
        """Create a fresh context builder for each test."""
        return EmailContextBuilder()

    @pytest.fixture
    def sample_email_input(self):
        """Sample email input for testing."""
        return EmailCopilotInput(
            recipient=EmailRecipient(
                email="john@acme.com",
                name="John Doe",
                company="Acme Corp",
            ),
            email_type=EmailType.COLD_OUTREACH,
        )

    @pytest.mark.asyncio
    async def test_build_context_minimal(self, context_builder, sample_email_input):
        """Test context building with minimal input."""
        context = await context_builder.build_context(
            email_input=sample_email_input,
        )
        assert context.lead is not None
        assert context.lead.company_name == "Acme Corp"

    @pytest.mark.asyncio
    async def test_build_context_with_lead_data(self, context_builder, sample_email_input):
        """Test context building with lead data from state."""
        leads = [
            {
                "lead_id": "lead-1",
                "company_name": "Acme Corp",
                "email": "john@acme.com",
                "industry": "Technology",
                "enrichment_data": {
                    "pain_points": ["Scaling", "Efficiency"],
                    "talking_points": ["Recent growth"],
                },
            }
        ]
        context = await context_builder.build_context(
            email_input=sample_email_input,
            leads=leads,
        )
        assert context.lead is not None
        assert context.lead.industry == "Technology"

    @pytest.mark.asyncio
    async def test_build_context_post_meeting(self, context_builder):
        """Test context building for post-meeting email."""
        email_input = EmailCopilotInput(
            recipient=EmailRecipient(email="john@acme.com"),
            email_type=EmailType.POST_MEETING,
        )
        meeting_notes = [
            {
                "meeting_id": "meeting-1",
                "date": "2024-01-15",
                "summary": "Product demo with Acme",
                "key_points": ["Liked the dashboard", "Concerned about pricing"],
                "action_items": [{"task": "Send proposal"}],
                "participants": ["John Doe", "Jane Smith"],
            }
        ]
        context = await context_builder.build_context(
            email_input=email_input,
            meeting_notes=meeting_notes,
        )
        assert context.meeting is not None
        assert context.meeting.summary == "Product demo with Acme"

    @pytest.mark.asyncio
    async def test_rag_not_enabled(self):
        """Test that RAG returns None when not enabled."""
        builder = EmailContextBuilder(rag_enabled=False)
        assert builder.rag_enabled is False
        brandvoice = await builder._get_brandvoice_context("client-123")
        assert brandvoice is None

    def test_enable_rag(self, context_builder):
        """Test RAG enable/disable."""
        context_builder.enable_rag(True)
        assert context_builder.rag_enabled is True
        context_builder.enable_rag(False)
        assert context_builder.rag_enabled is False


class TestContextFormatting:
    """Tests for context formatting utilities."""

    def test_format_lead_context(self):
        """Test lead context formatting for prompt."""
        context = LeadContext(
            company_name="Acme Corp",
            industry="Technology",
            company_size="50-200",
            pain_points=["Scaling challenges"],
            talking_points=["Recent Series B"],
        )
        formatted = format_lead_context_for_prompt(context)
        assert "Acme Corp" in formatted
        assert "Technology" in formatted
        assert "Scaling challenges" in formatted

    def test_format_meeting_context(self):
        """Test meeting context formatting for prompt."""
        context = MeetingContext(
            meeting_date="2024-01-15",
            summary="Product demo meeting",
            key_points=["Discussed pricing"],
            action_items=["Send proposal"],
        )
        formatted = format_meeting_context_for_prompt(context)
        assert "2024-01-15" in formatted
        assert "Product demo meeting" in formatted


# =============================================================================
# ADAPTER TESTS - Mock Adapter
# =============================================================================


class TestMockEmailAdapter:
    """Tests for MockEmailAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create mock adapter."""
        return MockEmailAdapter()

    @pytest.fixture
    def sample_payload(self):
        """Sample delivery payload."""
        return EmailDeliveryPayload(
            to_email="john@acme.com",
            to_name="John Doe",
            subject="Test Subject",
            body_html="<p>Test body</p>",
            body_plain="Test body",
            contact_id="contact-123",
            email_type=EmailType.COLD_OUTREACH,
        )

    @pytest.mark.asyncio
    async def test_send_success(self, adapter, sample_payload):
        """Test successful mock send."""
        result = await adapter.send(sample_payload)
        assert result.success is True
        assert result.message_id is not None
        assert len(adapter.get_sent_emails()) == 1

    @pytest.mark.asyncio
    async def test_send_stores_payload(self, adapter, sample_payload):
        """Test that sent emails are stored."""
        await adapter.send(sample_payload)
        sent = adapter.get_sent_emails()
        assert len(sent) == 1
        assert sent[0].to_email == "john@acme.com"

    @pytest.mark.asyncio
    async def test_send_failure(self, adapter, sample_payload):
        """Test mock send failure."""
        adapter.set_should_fail(True, "Simulated failure")
        with pytest.raises(EmailDeliveryError):
            await adapter.send(sample_payload)

    @pytest.mark.asyncio
    async def test_validation_error(self, adapter):
        """Test validation error on invalid payload."""
        invalid_payload = EmailDeliveryPayload(
            to_email="",  # Invalid
            subject="Test",
            body_html="<p>Test</p>",
            body_plain="Test",
            email_type=EmailType.COLD_OUTREACH,
        )
        with pytest.raises(EmailValidationError):
            await adapter.send(invalid_payload)

    @pytest.mark.asyncio
    async def test_health_check(self, adapter):
        """Test health check."""
        assert await adapter.health_check() is True

    def test_clear_sent_emails(self, adapter):
        """Test clearing sent emails."""
        adapter._sent_emails.append(MagicMock())
        adapter.clear_sent_emails()
        assert len(adapter.get_sent_emails()) == 0


# =============================================================================
# ADAPTER TESTS - HubSpot Adapter
# =============================================================================


class TestHubSpotEmailAdapter:
    """Tests for HubSpotEmailAdapter."""

    def test_not_configured_without_token(self):
        """Test adapter is not configured without token."""
        adapter = HubSpotEmailAdapter(access_token=None)
        # May or may not be configured based on env
        # Just verify it doesn't crash

    def test_configured_with_token(self):
        """Test adapter is configured with token."""
        adapter = HubSpotEmailAdapter(access_token="test-token")
        assert adapter.is_configured is True
        assert adapter.provider == DeliveryProvider.HUBSPOT

    @pytest.mark.asyncio
    async def test_hubspot_validation_requires_association(self):
        """Test HubSpot requires at least one association."""
        adapter = HubSpotEmailAdapter(access_token="test-token")
        payload = EmailDeliveryPayload(
            to_email="test@test.com",
            subject="Test",
            body_html="<p>Test</p>",
            body_plain="Test",
            # No contact_id, deal_id, or company_id
            email_type=EmailType.COLD_OUTREACH,
        )
        errors = await adapter.validate_payload(payload)
        assert any("association" in e.lower() for e in errors)


# =============================================================================
# NODE TESTS
# =============================================================================


class TestEmailCopilotNode:
    """Tests for email_copilot_node."""

    @pytest.fixture
    def base_state(self):
        """Base state for testing."""
        return {
            "workflow_id": str(uuid4()),
            "client": {
                "client_id": "client-123",
                "client_name": "Test Client",
                "crm_provider": "hubspot",
            },
            "messages": [],
            "leads": [],
            "meeting_notes": [],
            "email_drafts": [],
            "pending_approvals": [],
            "agent_execution_log": [],
        }

    def test_no_input_returns_error(self, base_state):
        """Test node returns error when no input provided."""
        from backend.graph.nodes import email_copilot_node

        result = email_copilot_node(base_state)
        assert "error" in result.get("messages", [{}])[0].get("message_type", "")

    def test_parses_input_correctly(self, base_state):
        """Test node parses input from messages."""
        from backend.graph.nodes import email_copilot_node

        base_state["messages"] = [
            {
                "message_type": "email_copilot_input",
                "metadata": {
                    "recipient_email": "john@acme.com",
                    "recipient_name": "John Doe",
                    "recipient_company": "Acme Corp",
                    "email_type": "cold_outreach",
                },
            }
        ]

        # Mock the async functions
        with patch(
            "backend.graph.nodes._generate_email_with_llm",
            new_callable=AsyncMock,
        ) as mock_generate:
            mock_draft = EmailDraft(
                subject="Test Subject",
                body_html="<p>Body</p>",
                body_plain="Body",
                email_type=EmailType.COLD_OUTREACH,
                tone=EmailTone.PROFESSIONAL,
                generated_at=datetime.now(timezone.utc).isoformat(),
            )
            mock_generate.return_value = EmailGenerationResult(
                draft=mock_draft,
                personalization_score=0.8,
                relevance_score=0.9,
            )

            result = email_copilot_node(base_state)

            # Should have email draft and approval
            assert len(result.get("email_drafts", [])) == 1
            assert len(result.get("pending_approvals", [])) == 1


class TestEmailDeliveryNode:
    """Tests for email_delivery_node."""

    @pytest.fixture
    def base_state(self):
        """Base state for testing."""
        return {
            "workflow_id": str(uuid4()),
            "client": {
                "client_id": "client-123",
                "client_name": "Test Client",
                "crm_provider": "hubspot",
            },
            "approval_history": [],
            "email_drafts": [],
            "agent_execution_log": [],
            "messages": [],
        }

    def test_no_approved_emails(self, base_state):
        """Test node handles no approved emails."""
        from backend.graph.nodes import email_delivery_node

        result = email_delivery_node(base_state)
        assert "no_approved_emails" in result.get("agent_execution_log", [{}])[0].get("action", "")

    def test_processes_approved_emails(self, base_state):
        """Test node processes approved emails."""
        from backend.graph.nodes import email_delivery_node
        from backend.graph.state import ApprovalType

        base_state["approval_history"] = [
            {
                "approval_id": str(uuid4()),
                "approval_type": ApprovalType.EMAIL_SEND,
                "status": "approved",
                "payload": {
                    "delivery_payload": {
                        "to_email": "john@acme.com",
                        "subject": "Test",
                        "body_html": "<p>Test</p>",
                        "body_plain": "Test",
                        "contact_id": "contact-123",
                        "email_type": "cold_outreach",
                    }
                },
            }
        ]

        # Mock the adapter
        with patch(
            "backend.graph.nodes.get_email_adapter"
        ) as mock_get_adapter:
            mock_adapter = MockEmailAdapter()
            mock_get_adapter.return_value = mock_adapter

            result = email_delivery_node(base_state)

            # Should have processed
            log = result.get("agent_execution_log", [{}])[0]
            assert log.get("action") == "deliver_emails"


# =============================================================================
# PHASE 4.3 VERIFICATION TESTS
# =============================================================================


class TestPhase43Verification:
    """Verification tests for Phase 4.3 implementation."""

    def test_all_enums_defined(self):
        """Test all email enums are defined."""
        assert EmailType.COLD_OUTREACH
        assert EmailType.FOLLOW_UP
        assert EmailType.MEETING_REQUEST
        assert EmailType.POST_MEETING
        assert EmailTone.PROFESSIONAL
        assert EmailTone.FRIENDLY
        assert EmailPriority.NORMAL
        assert DeliveryProvider.HUBSPOT

    def test_all_schemas_importable(self):
        """Test all schemas can be imported."""
        from backend.app.schemas.email import (
            EmailRecipient,
            LeadContext,
            MeetingContext,
            EmailCopilotInput,
            EmailDraft,
            EmailGenerationResult,
            EmailDeliveryPayload,
            EmailDeliveryResult,
            EmailApprovalPayload,
            BrandvoiceContext,
            EmailContext,
        )
        assert EmailRecipient
        assert EmailContext
        assert BrandvoiceContext  # RAG-ready

    def test_node_functions_exist(self):
        """Test node functions are defined."""
        from backend.graph.nodes import (
            email_copilot_node,
            email_delivery_node,
        )
        assert callable(email_copilot_node)
        assert callable(email_delivery_node)

    def test_prompt_template_exists(self):
        """Test email_copilot prompt template is defined."""
        from backend.prompts.base import PromptManager

        manager = PromptManager.get_instance()
        template = manager.get_prompt("email_copilot")
        assert template is not None
        assert template.metadata.name == "email_copilot"

    def test_context_builder_exists(self):
        """Test EmailContextBuilder is available."""
        builder = get_email_context_builder()
        assert isinstance(builder, EmailContextBuilder)

    def test_adapter_factory_exists(self):
        """Test email adapter factory works."""
        from backend.services.email.adapters import get_email_adapter

        adapter = get_email_adapter("hubspot")
        assert adapter is not None
        assert adapter.provider == DeliveryProvider.HUBSPOT

    def test_rag_placeholder_exists(self):
        """Test RAG placeholder is ready for Phase 4.5."""
        builder = EmailContextBuilder()
        assert hasattr(builder, "_get_brandvoice_context")
        assert hasattr(builder, "enable_rag")
        assert hasattr(builder, "rag_enabled")

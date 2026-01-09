"""
Email Copilot Schemas (Phase 4.3)
=================================
Pydantic schemas for email generation and delivery operations.

This module defines structured schemas for:
- Email copilot input (lead data, email type, context)
- Email draft output (personalized email content)
- Email delivery adapters (HubSpot, Salesforce, etc.)
- Email approval workflow
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ENUMS
# =============================================================================


class EmailType(str, Enum):
    """Type of email to generate."""
    COLD_OUTREACH = "cold_outreach"      # Initial contact with a lead
    FOLLOW_UP = "follow_up"              # Follow-up on previous contact
    MEETING_REQUEST = "meeting_request"  # Request for a meeting/demo
    POST_MEETING = "post_meeting"        # Summary after a meeting


class EmailTone(str, Enum):
    """Tone of the email."""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    CASUAL = "casual"


class EmailPriority(str, Enum):
    """Priority level for email sending."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class EmailStatus(str, Enum):
    """Status of the email in the workflow."""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    SENT = "sent"
    DELIVERED = "delivered"
    BOUNCED = "bounced"
    FAILED = "failed"


class DeliveryProvider(str, Enum):
    """Email delivery provider."""
    HUBSPOT = "hubspot"
    SALESFORCE = "salesforce"
    SMTP = "smtp"  # Future: direct SMTP/SendGrid


# =============================================================================
# INPUT SCHEMAS
# =============================================================================


class EmailRecipient(BaseModel):
    """Email recipient information."""
    email: str = Field(description="Recipient email address")
    name: str | None = Field(default=None, description="Recipient's full name")
    title: str | None = Field(default=None, description="Recipient's job title")
    company: str | None = Field(default=None, description="Recipient's company")

    # CRM references
    contact_id: str | None = Field(default=None, description="CRM contact ID")
    company_id: str | None = Field(default=None, description="CRM company ID")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Basic email validation."""
        if not v or "@" not in v:
            raise ValueError("Invalid email address")
        return v.strip().lower()


class LeadContext(BaseModel):
    """Context from lead research for personalization."""
    company_name: str = Field(description="Lead's company name")
    company_description: str | None = Field(default=None, description="Company description")
    industry: str | None = Field(default=None, description="Company industry")
    company_size: str | None = Field(default=None, description="Company size range")

    # Research insights
    pain_points: list[str] = Field(default_factory=list, description="Identified pain points")
    talking_points: list[str] = Field(default_factory=list, description="Suggested talking points")
    recent_news: list[str] = Field(default_factory=list, description="Recent news headlines")
    business_signals: list[str] = Field(default_factory=list, description="Business signals")

    # Qualification
    qualification_score: str | None = Field(default=None, description="Lead qualification (hot/warm/cold)")

    # Additional context
    technologies: list[str] = Field(default_factory=list, description="Technologies they use")
    key_people: list[str] = Field(default_factory=list, description="Key decision makers")


class MeetingContext(BaseModel):
    """Context from a meeting for post-meeting emails."""
    meeting_id: str | None = Field(default=None, description="Meeting ID reference")
    meeting_date: str | None = Field(default=None, description="When the meeting occurred")
    summary: str | None = Field(default=None, description="Meeting summary")
    key_points: list[str] = Field(default_factory=list, description="Key discussion points")
    action_items: list[str] = Field(default_factory=list, description="Action items from meeting")
    next_steps: list[str] = Field(default_factory=list, description="Agreed next steps")
    attendees: list[str] = Field(default_factory=list, description="Meeting attendees")


class PreviousEmailContext(BaseModel):
    """Context from previous email thread for follow-ups."""
    thread_id: str | None = Field(default=None, description="Email thread ID")
    last_email_date: str | None = Field(default=None, description="Date of last email")
    last_email_subject: str | None = Field(default=None, description="Subject of last email")
    last_email_snippet: str | None = Field(default=None, description="Snippet from last email")
    response_received: bool = Field(default=False, description="Whether they responded")
    follow_up_count: int = Field(default=0, ge=0, description="Number of follow-ups sent")


class EmailCopilotInput(BaseModel):
    """Input for email generation."""

    # Recipient
    recipient: EmailRecipient = Field(description="Email recipient")

    # Email type and settings
    email_type: EmailType = Field(description="Type of email to generate")
    tone: EmailTone = Field(default=EmailTone.PROFESSIONAL, description="Email tone")
    priority: EmailPriority = Field(default=EmailPriority.NORMAL, description="Email priority")

    # Context for personalization
    lead_context: LeadContext | None = Field(default=None, description="Lead research context")
    meeting_context: MeetingContext | None = Field(default=None, description="Meeting context for post-meeting")
    previous_email: PreviousEmailContext | None = Field(default=None, description="Previous email for follow-ups")

    # Sender info
    sender_name: str | None = Field(default=None, description="Sender's name")
    sender_title: str | None = Field(default=None, description="Sender's job title")
    sender_company: str | None = Field(default=None, description="Sender's company")
    sender_email: str | None = Field(default=None, description="Sender's email")

    # Custom instructions
    custom_instructions: str | None = Field(
        default=None,
        description="Custom instructions for email generation"
    )
    include_calendar_link: bool = Field(
        default=False,
        description="Include a calendar/meeting link"
    )
    calendar_link: str | None = Field(
        default=None,
        description="Calendar booking link to include"
    )

    # CRM associations
    deal_id: str | None = Field(default=None, description="Associated deal ID")

    @field_validator("email_type")
    @classmethod
    def validate_context_for_type(cls, v: EmailType) -> EmailType:
        """Email type validation (context validation done in node)."""
        return v


class EmailCopilotRequest(BaseModel):
    """API request to generate an email."""
    input: EmailCopilotInput
    client_id: str = Field(description="Client ID for multi-tenant context")
    workflow_id: str | None = Field(default=None, description="Associated workflow ID")


# =============================================================================
# OUTPUT SCHEMAS - Email Draft
# =============================================================================


class EmailDraft(BaseModel):
    """Generated email draft."""

    # Core content
    subject: str = Field(description="Email subject line")
    body_html: str = Field(description="HTML formatted email body")
    body_plain: str = Field(description="Plain text email body")

    # Preview
    preview_text: str | None = Field(
        default=None,
        description="Email preview text (first line visible in inbox)"
    )

    # Metadata
    email_type: EmailType = Field(description="Type of email")
    tone: EmailTone = Field(description="Tone used")
    word_count: int = Field(default=0, ge=0, description="Word count of body")

    # Personalization tracking
    personalization_elements: list[str] = Field(
        default_factory=list,
        description="Personalization elements used"
    )

    # Generation metadata
    generated_at: str = Field(description="When the draft was generated")
    generation_model: str | None = Field(default=None, description="LLM model used")

    @field_validator("subject")
    @classmethod
    def validate_subject(cls, v: str) -> str:
        """Ensure subject is not too long."""
        if len(v) > 200:
            return v[:197] + "..."
        return v.strip()

    @field_validator("body_plain")
    @classmethod
    def calculate_word_count(cls, v: str, info) -> str:
        """Auto-calculate word count."""
        # Note: word_count is set separately, this just returns the value
        return v


class EmailGenerationResult(BaseModel):
    """Complete result from email generation LLM."""

    # Generated draft
    draft: EmailDraft = Field(description="Generated email draft")

    # Alternative options
    subject_alternatives: list[str] = Field(
        default_factory=list,
        description="Alternative subject lines"
    )

    # Quality assessment
    personalization_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How personalized the email is (0-1)"
    )
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How relevant to recipient's context (0-1)"
    )

    # Reasoning
    approach_reasoning: str | None = Field(
        default=None,
        description="Reasoning behind the approach taken"
    )

    # Warnings
    warnings: list[str] = Field(
        default_factory=list,
        description="Any warnings about the generated content"
    )


# =============================================================================
# OUTPUT SCHEMAS - Delivery
# =============================================================================


class EmailDeliveryPayload(BaseModel):
    """Payload for email delivery adapter."""

    # Core email
    to_email: str = Field(description="Recipient email address")
    to_name: str | None = Field(default=None, description="Recipient name")
    subject: str = Field(description="Email subject")
    body_html: str = Field(description="HTML body")
    body_plain: str = Field(description="Plain text body")

    # Sender
    from_email: str | None = Field(default=None, description="Sender email")
    from_name: str | None = Field(default=None, description="Sender name")
    reply_to: str | None = Field(default=None, description="Reply-to address")

    # CRM associations
    contact_id: str | None = Field(default=None, description="CRM contact ID")
    deal_id: str | None = Field(default=None, description="CRM deal ID")
    company_id: str | None = Field(default=None, description="CRM company ID")

    # Tracking
    track_opens: bool = Field(default=True, description="Track email opens")
    track_clicks: bool = Field(default=True, description="Track link clicks")

    # Metadata
    email_type: EmailType = Field(description="Type of email")
    workflow_id: str | None = Field(default=None, description="Associated workflow ID")


class EmailDeliveryResult(BaseModel):
    """Result from email delivery."""

    success: bool = Field(description="Whether delivery was successful")

    # Provider response
    provider: DeliveryProvider = Field(description="Provider used")
    message_id: str | None = Field(default=None, description="Provider message ID")
    thread_id: str | None = Field(default=None, description="Email thread ID")

    # Timestamps
    sent_at: str | None = Field(default=None, description="When email was sent")

    # CRM tracking
    crm_activity_id: str | None = Field(
        default=None,
        description="ID of activity logged in CRM"
    )

    # Error handling
    error_code: str | None = Field(default=None, description="Error code if failed")
    error_message: str | None = Field(default=None, description="Error message if failed")
    retry_possible: bool = Field(default=False, description="Whether retry is possible")


# =============================================================================
# APPROVAL SCHEMAS
# =============================================================================


class EmailApprovalPayload(BaseModel):
    """Payload for email approval request."""

    # Draft
    draft: EmailDraft = Field(description="Email draft to approve")
    delivery_payload: EmailDeliveryPayload = Field(description="Delivery payload")

    # Context summary for reviewer
    recipient_summary: str = Field(description="Brief summary of recipient")
    context_summary: str = Field(description="Context used for personalization")

    # Metadata
    email_type: EmailType = Field(description="Type of email")
    priority: EmailPriority = Field(description="Email priority")

    # Quality metrics
    personalization_score: float = Field(default=0.0, description="Personalization score")

    # Warnings for reviewer
    warnings: list[str] = Field(default_factory=list, description="Warnings to review")


class EmailApprovalDecision(BaseModel):
    """Decision on email approval."""

    approved: bool = Field(description="Whether email is approved")

    # Modifications (if approved with changes)
    modified_subject: str | None = Field(default=None, description="Modified subject")
    modified_body: str | None = Field(default=None, description="Modified body")

    # Rejection
    rejection_reason: str | None = Field(default=None, description="Reason for rejection")

    # Metadata
    reviewed_by: str | None = Field(default=None, description="Who reviewed")
    reviewed_at: str | None = Field(default=None, description="When reviewed")


# =============================================================================
# CONTEXT BUILDER SCHEMAS (RAG-ready)
# =============================================================================


class BrandvoiceContext(BaseModel):
    """Brandvoice context from RAG (Phase 4.5)."""

    tone_guidelines: str | None = Field(default=None, description="Tone guidelines")
    writing_style: str | None = Field(default=None, description="Writing style notes")
    key_phrases: list[str] = Field(default_factory=list, description="Key phrases to use")
    phrases_to_avoid: list[str] = Field(default_factory=list, description="Phrases to avoid")
    example_snippets: list[str] = Field(default_factory=list, description="Example content snippets")

    # RAG metadata
    source_documents: list[str] = Field(default_factory=list, description="Source document IDs")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="RAG confidence")


class EmailContext(BaseModel):
    """Complete context for email generation."""

    # Required context
    lead: LeadContext | None = Field(default=None, description="Lead research context")

    # Optional context based on email type
    meeting: MeetingContext | None = Field(default=None, description="Meeting context")
    previous_email: PreviousEmailContext | None = Field(default=None, description="Previous email context")

    # RAG context (Phase 4.5)
    brandvoice: BrandvoiceContext | None = Field(default=None, description="Brandvoice from RAG")

    # Aggregated context quality
    context_completeness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How complete the context is (0-1)"
    )

    def calculate_completeness(self) -> float:
        """Calculate context completeness score."""
        score = 0.0

        if self.lead:
            score += 0.4
            if self.lead.pain_points:
                score += 0.1
            if self.lead.talking_points:
                score += 0.1

        if self.brandvoice:
            score += 0.2

        if self.meeting:
            score += 0.1

        if self.previous_email:
            score += 0.1

        return min(score, 1.0)


# =============================================================================
# API RESPONSE SCHEMAS
# =============================================================================


class EmailCopilotResponse(BaseModel):
    """API response for email copilot."""

    success: bool = Field(description="Whether generation was successful")
    email_id: str = Field(description="Generated email ID")

    # Draft
    draft: EmailDraft | None = Field(default=None, description="Generated draft")
    generation_result: EmailGenerationResult | None = Field(
        default=None,
        description="Full generation result"
    )

    # Approval
    needs_approval: bool = Field(default=True, description="Whether approval is needed")
    approval_id: str | None = Field(default=None, description="Approval request ID")

    # Error
    error_message: str | None = Field(default=None, description="Error if failed")

    # Performance
    processing_time_ms: int | None = Field(default=None, description="Processing time")


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "EmailType",
    "EmailTone",
    "EmailPriority",
    "EmailStatus",
    "DeliveryProvider",
    # Input
    "EmailRecipient",
    "LeadContext",
    "MeetingContext",
    "PreviousEmailContext",
    "EmailCopilotInput",
    "EmailCopilotRequest",
    # Output - Draft
    "EmailDraft",
    "EmailGenerationResult",
    # Output - Delivery
    "EmailDeliveryPayload",
    "EmailDeliveryResult",
    # Approval
    "EmailApprovalPayload",
    "EmailApprovalDecision",
    # Context (RAG-ready)
    "BrandvoiceContext",
    "EmailContext",
    # API
    "EmailCopilotResponse",
]

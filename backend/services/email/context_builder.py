"""
Email Context Builder (Phase 4.3 + 4.5)
=======================================
Aggregates context from multiple sources for email personalization.

Sources:
- Lead Research data (from state)
- Meeting Notes (from state)
- Previous Email threads (from state)
- Brandvoice documents (RAG - Phase 4.5)

Phase 4.5 Update: Integrated with BrandvoiceService for RAG-based
brandvoice context retrieval.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.app.schemas.email import (
    EmailType,
    LeadContext,
    MeetingContext,
    PreviousEmailContext,
    BrandvoiceContext,
    EmailContext,
    EmailCopilotInput,
)
from backend.graph.state import LeadData, MeetingNote, EmailDraft as StateEmailDraft

logger = logging.getLogger(__name__)


# Singleton instance
_context_builder_instance: EmailContextBuilder | None = None


def get_email_context_builder() -> EmailContextBuilder:
    """Get or create singleton EmailContextBuilder instance."""
    global _context_builder_instance
    if _context_builder_instance is None:
        _context_builder_instance = EmailContextBuilder()
    return _context_builder_instance


class EmailContextBuilder:
    """
    Builds comprehensive context for email generation.

    Aggregates data from:
    1. Lead research (company info, pain points, talking points)
    2. Meeting notes (for post-meeting emails)
    3. Previous emails (for follow-ups)
    4. Brandvoice RAG (via BrandvoiceService)
    """

    def __init__(self, rag_enabled: bool = True):
        """
        Initialize context builder.

        Args:
            rag_enabled: Whether to use RAG for brandvoice retrieval
        """
        self._rag_enabled = rag_enabled
        self._brandvoice_service = None

    @property
    def rag_enabled(self) -> bool:
        """Check if RAG is enabled."""
        return self._rag_enabled

    def _get_brandvoice_service(self):
        """Lazy-load brandvoice service."""
        if self._brandvoice_service is None:
            from backend.services.documents.brandvoice_service import get_brandvoice_service
            self._brandvoice_service = get_brandvoice_service()
        return self._brandvoice_service

    async def build_context(
        self,
        email_input: EmailCopilotInput,
        leads: list[LeadData] | None = None,
        meeting_notes: list[MeetingNote] | None = None,
        email_drafts: list[StateEmailDraft] | None = None,
        client_id: str | None = None,
    ) -> EmailContext:
        """
        Build complete email context from available sources.

        Args:
            email_input: Email generation input
            leads: Lead research data from state
            meeting_notes: Meeting notes from state
            email_drafts: Previous email drafts from state
            client_id: Client ID for brandvoice retrieval

        Returns:
            EmailContext: Aggregated context for email generation
        """
        # Build lead context
        lead_context = self._build_lead_context(email_input, leads)

        # Build meeting context (for post_meeting emails)
        meeting_context = None
        if email_input.email_type == EmailType.POST_MEETING:
            meeting_context = self._build_meeting_context(email_input, meeting_notes)

        # Build previous email context (for follow_up emails)
        previous_email_context = None
        if email_input.email_type == EmailType.FOLLOW_UP:
            previous_email_context = self._build_previous_email_context(
                email_input, email_drafts
            )

        # Get brandvoice context via RAG (Phase 4.5)
        brandvoice_context = await self._get_brandvoice_context(
            client_id=client_id,
            email_type=email_input.email_type.value if email_input.email_type else None,
        )

        # Create final context
        context = EmailContext(
            lead=lead_context,
            meeting=meeting_context,
            previous_email=previous_email_context,
            brandvoice=brandvoice_context,
        )

        # Calculate completeness
        context.context_completeness = context.calculate_completeness()

        logger.info(
            f"[EmailContextBuilder] Built context with completeness: "
            f"{context.context_completeness:.2f}"
        )

        return context

    def _build_lead_context(
        self,
        email_input: EmailCopilotInput,
        leads: list[LeadData] | None,
    ) -> LeadContext | None:
        """
        Build lead context from input and state.

        Args:
            email_input: Email input with recipient info
            leads: Lead data from state

        Returns:
            LeadContext: Lead context for personalization
        """
        # Start with input lead context if provided
        if email_input.lead_context:
            return email_input.lead_context

        # Try to find matching lead from state
        if leads:
            matching_lead = self._find_matching_lead(
                email_input.recipient.email,
                email_input.recipient.company,
                leads,
            )
            if matching_lead:
                return self._convert_lead_data_to_context(matching_lead)

        # Create minimal context from recipient info
        if email_input.recipient.company:
            return LeadContext(
                company_name=email_input.recipient.company,
                industry=None,
            )

        return None

    def _find_matching_lead(
        self,
        email: str,
        company: str | None,
        leads: list[LeadData],
    ) -> LeadData | None:
        """
        Find a lead matching the recipient.

        Args:
            email: Recipient email
            company: Recipient company name
            leads: Available leads

        Returns:
            LeadData: Matching lead or None
        """
        # Try to match by email first
        email_lower = email.lower()
        for lead in leads:
            if lead.get("email") and lead["email"].lower() == email_lower:
                return lead

        # Try to match by company name
        if company:
            company_lower = company.lower()
            for lead in leads:
                if lead.get("company_name") and lead["company_name"].lower() == company_lower:
                    return lead

        return None

    def _convert_lead_data_to_context(self, lead: LeadData) -> LeadContext:
        """
        Convert LeadData TypedDict to LeadContext Pydantic model.

        Args:
            lead: Lead data from state

        Returns:
            LeadContext: Pydantic context model
        """
        enrichment = lead.get("enrichment_data", {})

        return LeadContext(
            company_name=lead.get("company_name", "Unknown"),
            company_description=enrichment.get("company_description"),
            industry=lead.get("industry"),
            company_size=lead.get("company_size"),
            pain_points=enrichment.get("pain_points", []),
            talking_points=enrichment.get("talking_points", []),
            recent_news=enrichment.get("recent_news", []),
            business_signals=enrichment.get("business_signals", []),
            qualification_score=enrichment.get("qualification_score"),
            technologies=enrichment.get("technologies", []),
            key_people=enrichment.get("key_people", []),
        )

    def _build_meeting_context(
        self,
        email_input: EmailCopilotInput,
        meeting_notes: list[MeetingNote] | None,
    ) -> MeetingContext | None:
        """
        Build meeting context for post-meeting emails.

        Args:
            email_input: Email input with meeting context
            meeting_notes: Meeting notes from state

        Returns:
            MeetingContext: Meeting context or None
        """
        # Use input meeting context if provided
        if email_input.meeting_context:
            return email_input.meeting_context

        # Find most recent meeting from state
        if meeting_notes:
            # Get most recent meeting
            latest = meeting_notes[-1]

            return MeetingContext(
                meeting_id=latest.get("meeting_id"),
                meeting_date=latest.get("date"),
                summary=latest.get("summary"),
                key_points=latest.get("key_points", []),
                action_items=[
                    item.get("task", "")
                    for item in latest.get("action_items", [])
                ],
                next_steps=[],  # Not directly stored in MeetingNote
                attendees=latest.get("participants", []),
            )

        return None

    def _build_previous_email_context(
        self,
        email_input: EmailCopilotInput,
        email_drafts: list[StateEmailDraft] | None,
    ) -> PreviousEmailContext | None:
        """
        Build previous email context for follow-ups.

        Args:
            email_input: Email input with previous email context
            email_drafts: Email drafts from state

        Returns:
            PreviousEmailContext: Previous email context or None
        """
        # Use input previous email context if provided
        if email_input.previous_email:
            return email_input.previous_email

        # Find previous emails to this recipient from state
        if email_drafts:
            recipient_email = email_input.recipient.email.lower()
            previous = [
                draft for draft in email_drafts
                if draft.get("recipient_email", "").lower() == recipient_email
                and draft.get("status") == "sent"
            ]

            if previous:
                latest = previous[-1]
                return PreviousEmailContext(
                    thread_id=latest.get("email_id"),
                    last_email_date=latest.get("sent_at"),
                    last_email_subject=latest.get("subject"),
                    last_email_snippet=latest.get("body", "")[:200],
                    response_received=False,  # Would need tracking to know
                    follow_up_count=len(previous),
                )

        return None

    async def _get_brandvoice_context(
        self,
        client_id: str | None,
        email_type: str | None = None,
    ) -> BrandvoiceContext | None:
        """
        Get brandvoice context from RAG via BrandvoiceService.

        Phase 4.5 Implementation: Uses BrandvoiceService to retrieve
        brandvoice documents and build context.

        Args:
            client_id: Client ID for brandvoice retrieval
            email_type: Optional email type for context tuning

        Returns:
            BrandvoiceContext: Brandvoice context or None
        """
        if not self._rag_enabled or not client_id:
            logger.debug(
                f"[EmailContextBuilder] RAG disabled or no client_id. "
                f"Skipping brandvoice retrieval."
            )
            return None

        try:
            brandvoice_service = self._get_brandvoice_service()

            # Check if client has any documents
            has_docs = await brandvoice_service.has_brandvoice_documents(client_id)
            if not has_docs:
                logger.debug(
                    f"[EmailContextBuilder] No brandvoice documents for client {client_id}"
                )
                return None

            # Retrieve brandvoice context
            context = await brandvoice_service.get_brandvoice_context(
                client_id=client_id,
                email_type=email_type,
            )

            if context:
                logger.info(
                    f"[EmailContextBuilder] Retrieved brandvoice context for client {client_id} "
                    f"with confidence {context.confidence:.2f}"
                )
            else:
                logger.debug(
                    f"[EmailContextBuilder] No matching brandvoice content found for client {client_id}"
                )

            return context

        except Exception as e:
            logger.exception(
                f"[EmailContextBuilder] Error retrieving brandvoice context: {e}"
            )
            return None

    def enable_rag(self, enabled: bool = True):
        """
        Enable or disable RAG functionality.

        Args:
            enabled: Whether RAG should be enabled
        """
        self._rag_enabled = enabled
        logger.info(f"[EmailContextBuilder] RAG {'enabled' if enabled else 'disabled'}")


# =============================================================================
# CONTEXT FORMATTING UTILITIES
# =============================================================================


def format_lead_context_for_prompt(context: LeadContext) -> str:
    """
    Format lead context as a string for LLM prompt.

    Args:
        context: Lead context

    Returns:
        str: Formatted context string
    """
    parts = [f"**Company:** {context.company_name}"]

    if context.industry:
        parts.append(f"**Industry:** {context.industry}")
    if context.company_size:
        parts.append(f"**Size:** {context.company_size}")
    if context.company_description:
        parts.append(f"**About:** {context.company_description}")

    if context.pain_points:
        parts.append(f"**Pain Points:**\n- " + "\n- ".join(context.pain_points[:3]))

    if context.talking_points:
        parts.append(f"**Talking Points:**\n- " + "\n- ".join(context.talking_points[:3]))

    if context.recent_news:
        parts.append(f"**Recent News:**\n- " + "\n- ".join(context.recent_news[:2]))

    if context.business_signals:
        parts.append(f"**Business Signals:**\n- " + "\n- ".join(context.business_signals[:2]))

    if context.qualification_score:
        parts.append(f"**Lead Score:** {context.qualification_score}")

    return "\n".join(parts)


def format_meeting_context_for_prompt(context: MeetingContext) -> str:
    """
    Format meeting context as a string for LLM prompt.

    Args:
        context: Meeting context

    Returns:
        str: Formatted context string
    """
    parts = []

    if context.meeting_date:
        parts.append(f"**Meeting Date:** {context.meeting_date}")

    if context.summary:
        parts.append(f"**Summary:** {context.summary}")

    if context.key_points:
        parts.append(f"**Key Discussion Points:**\n- " + "\n- ".join(context.key_points[:5]))

    if context.action_items:
        parts.append(f"**Action Items:**\n- " + "\n- ".join(context.action_items[:5]))

    if context.next_steps:
        parts.append(f"**Next Steps:**\n- " + "\n- ".join(context.next_steps[:3]))

    if context.attendees:
        parts.append(f"**Attendees:** " + ", ".join(context.attendees))

    return "\n".join(parts)


def format_brandvoice_context_for_prompt(context: BrandvoiceContext) -> str:
    """
    Format brandvoice context as a string for LLM prompt.

    Args:
        context: Brandvoice context

    Returns:
        str: Formatted context string
    """
    parts = []

    if context.tone_guidelines:
        parts.append(f"**Tone Guidelines:** {context.tone_guidelines}")

    if context.writing_style:
        parts.append(f"**Writing Style:** {context.writing_style}")

    if context.key_phrases:
        parts.append(f"**Key Phrases to Use:**\n- " + "\n- ".join(context.key_phrases[:5]))

    if context.phrases_to_avoid:
        parts.append(f"**Phrases to Avoid:**\n- " + "\n- ".join(context.phrases_to_avoid[:5]))

    if context.example_snippets:
        parts.append(f"**Example Content:**")
        for snippet in context.example_snippets[:2]:
            parts.append(f"> {snippet[:200]}...")

    return "\n".join(parts)

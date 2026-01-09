"""
Meeting Notes Schemas (Phase 3.1)
=================================
Pydantic schemas for Meeting Notes Analyzer agent.
Supports extensible input sources (manual text, file upload, calendar, etc.)

ADR Decision: MVP uses manual text input with extensible adapter pattern.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# ENUMS
# =============================================================================

class MeetingInputSource(str, Enum):
    """Source of meeting notes input."""
    MANUAL_TEXT = "manual_text"         # Direct text input (MVP)
    FILE_UPLOAD = "file_upload"         # Uploaded file (future)
    GOOGLE_CALENDAR = "google_calendar" # Google Calendar event (future)
    OUTLOOK_CALENDAR = "outlook_calendar"  # Outlook event (future)
    OTTER_AI = "otter_ai"               # Otter.ai transcript (future)
    FIREFLIES = "fireflies"             # Fireflies.ai (future)
    ZOOM_TRANSCRIPT = "zoom_transcript" # Zoom native transcript (future)


class MeetingSentiment(str, Enum):
    """Meeting sentiment classification."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class ActionItemPriority(str, Enum):
    """Action item priority level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


# =============================================================================
# INPUT SCHEMAS
# =============================================================================

class NormalizedMeetingInput(BaseModel):
    """
    Normalized meeting input - Standard format regardless of source.

    All input adapters convert their source data to this schema.
    This enables the Meeting Notes agent to work with any input source.
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    # Source tracking
    source: MeetingInputSource = MeetingInputSource.MANUAL_TEXT
    source_id: str | None = Field(
        None,
        description="External ID from source system (calendar event ID, etc.)"
    )

    # Meeting metadata
    title: str | None = Field(
        None,
        max_length=500,
        description="Meeting title/subject"
    )
    meeting_date: datetime | None = Field(
        None,
        description="When the meeting occurred"
    )
    duration_minutes: int | None = Field(
        None,
        ge=1,
        description="Meeting duration in minutes"
    )

    # Participants
    participants: list[str] = Field(
        default_factory=list,
        description="List of participant names/emails"
    )
    organizer: str | None = Field(
        None,
        description="Meeting organizer"
    )

    # Content
    transcript: str = Field(
        ...,
        min_length=10,
        description="Meeting transcript or notes text"
    )

    # Context
    deal_id: str | None = Field(
        None,
        description="Associated CRM deal ID"
    )
    contact_id: str | None = Field(
        None,
        description="Associated CRM contact ID"
    )
    company_id: str | None = Field(
        None,
        description="Associated CRM company ID"
    )
    additional_context: str | None = Field(
        None,
        description="Additional context for analysis"
    )


class MeetingAnalysisRequest(BaseModel):
    """Request schema for meeting notes analysis API endpoint."""
    model_config = ConfigDict(extra="forbid")

    client_id: UUID = Field(..., description="Client tenant ID")
    meeting_input: NormalizedMeetingInput

    # Analysis options
    extract_action_items: bool = Field(
        True,
        description="Extract action items from meeting"
    )
    analyze_sentiment: bool = Field(
        True,
        description="Analyze overall meeting sentiment"
    )
    suggest_deal_stage: bool = Field(
        True,
        description="Suggest CRM deal stage update"
    )
    generate_follow_up_email: bool = Field(
        False,
        description="Generate follow-up email draft"
    )


# =============================================================================
# OUTPUT SCHEMAS (LLM Structured Output)
# =============================================================================

class ActionItem(BaseModel):
    """Single action item extracted from meeting."""
    model_config = ConfigDict(extra="forbid")

    task: str = Field(..., description="Description of the task")
    assignee: str | None = Field(
        None,
        description="Person responsible for the task"
    )
    due_date: str | None = Field(
        None,
        description="Due date in ISO format or relative (e.g., 'next week')"
    )
    priority: ActionItemPriority = Field(
        ActionItemPriority.MEDIUM,
        description="Task priority level"
    )
    context: str | None = Field(
        None,
        description="Additional context from the meeting"
    )


class KeyDecision(BaseModel):
    """Key decision made during the meeting."""
    model_config = ConfigDict(extra="forbid")

    decision: str = Field(..., description="The decision that was made")
    rationale: str | None = Field(
        None,
        description="Reasoning behind the decision"
    )
    stakeholders: list[str] = Field(
        default_factory=list,
        description="People involved in the decision"
    )


class DealStageRecommendation(BaseModel):
    """CRM deal stage update recommendation."""
    model_config = ConfigDict(extra="forbid")

    current_signals: list[str] = Field(
        default_factory=list,
        description="Signals indicating deal progress"
    )
    recommended_stage: str | None = Field(
        None,
        description="Recommended deal stage to update to"
    )
    confidence: Literal["low", "medium", "high"] = Field(
        "medium",
        description="Confidence level of the recommendation"
    )
    reasoning: str | None = Field(
        None,
        description="Explanation for the recommendation"
    )


class MeetingAnalysis(BaseModel):
    """
    Meeting analysis output - LLM structured output schema.

    This schema is used with LangChain's with_structured_output() method
    to ensure reliable, validated LLM responses.
    """
    model_config = ConfigDict(extra="forbid")

    # Summary
    summary: str = Field(
        ...,
        min_length=10,
        description="Concise meeting summary (2-3 sentences)"
    )

    # Key points
    key_points: list[str] = Field(
        default_factory=list,
        description="Main discussion points from the meeting"
    )

    # Decisions
    key_decisions: list[KeyDecision] = Field(
        default_factory=list,
        description="Key decisions made during the meeting"
    )

    # Action items
    action_items: list[ActionItem] = Field(
        default_factory=list,
        description="Tasks and follow-ups extracted from the meeting"
    )

    # Sentiment analysis
    overall_sentiment: MeetingSentiment = Field(
        MeetingSentiment.NEUTRAL,
        description="Overall meeting sentiment"
    )
    sentiment_explanation: str | None = Field(
        None,
        description="Brief explanation of sentiment assessment"
    )

    # Follow-up
    follow_up_required: bool = Field(
        False,
        description="Whether follow-up actions are needed"
    )
    follow_up_reason: str | None = Field(
        None,
        description="Reason for required follow-up"
    )

    # Deal stage
    deal_stage_recommendation: DealStageRecommendation | None = Field(
        None,
        description="Recommendation for CRM deal stage update"
    )

    # Participants analysis
    identified_participants: list[str] = Field(
        default_factory=list,
        description="Participants identified from the transcript"
    )

    # Additional insights
    risks_concerns: list[str] = Field(
        default_factory=list,
        description="Risks or concerns raised in the meeting"
    )
    opportunities: list[str] = Field(
        default_factory=list,
        description="Opportunities identified in the meeting"
    )
    next_steps: list[str] = Field(
        default_factory=list,
        description="Agreed next steps"
    )


# =============================================================================
# API RESPONSE SCHEMAS
# =============================================================================

class MeetingAnalysisResponse(BaseModel):
    """Response schema for meeting notes analysis API endpoint."""
    model_config = ConfigDict(extra="forbid")

    success: bool = True
    meeting_id: str = Field(..., description="Generated meeting ID")
    analysis: MeetingAnalysis

    # CRM integration
    crm_tasks_created: int = Field(
        0,
        description="Number of CRM tasks created (pending approval)"
    )
    approval_ids: list[str] = Field(
        default_factory=list,
        description="IDs of pending approval requests"
    )

    # Metadata
    processing_time_ms: int | None = None
    llm_model: str | None = None
    tokens_used: int | None = None


class MeetingNotesSummary(BaseModel):
    """Summary schema for listing meeting notes."""
    model_config = ConfigDict(extra="forbid")

    meeting_id: str
    title: str | None
    date: datetime | None
    summary: str
    action_items_count: int
    sentiment: MeetingSentiment
    follow_up_required: bool
    created_at: datetime


# =============================================================================
# INPUT ADAPTER INTERFACE (for future extensibility)
# =============================================================================

class MeetingInputAdapterResult(BaseModel):
    """
    Result from input adapters.

    All input source adapters (Google Calendar, Otter.ai, file upload, etc.)
    should return this structure containing the normalized input.
    """
    model_config = ConfigDict(extra="forbid")

    success: bool
    normalized_input: NormalizedMeetingInput | None = None
    error_message: str | None = None
    raw_data: dict[str, Any] | None = Field(
        None,
        description="Original raw data from source (for debugging)"
    )

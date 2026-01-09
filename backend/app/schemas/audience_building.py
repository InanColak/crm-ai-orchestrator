"""
Audience Building Schemas (Phase 5.4)
=====================================
Pydantic schemas for audience building and persona creation operations.

This module defines structured schemas for:
- Ideal Customer Profile (ICP) definition
- Buyer Persona creation
- Pain point analysis by segment
- Buying journey mapping
- Messaging matrix for personas
- Channel strategy recommendations

Part of Intelligence Squad - synthesizes all intelligence data into actionable audience profiles.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ENUMS
# =============================================================================


class PersonaType(str, Enum):
    """Type of buyer persona."""
    DECISION_MAKER = "decision_maker"          # Final decision authority
    INFLUENCER = "influencer"                  # Influences the decision
    USER = "user"                              # End user of product
    GATEKEEPER = "gatekeeper"                  # Controls access to decision makers
    CHAMPION = "champion"                      # Internal advocate


class CompanySize(str, Enum):
    """Company size categories."""
    STARTUP = "startup"                        # 1-10 employees
    SMALL = "small"                            # 11-50 employees
    MEDIUM = "medium"                          # 51-200 employees
    MID_MARKET = "mid_market"                  # 201-1000 employees
    ENTERPRISE = "enterprise"                  # 1000+ employees


class BuyingStage(str, Enum):
    """Stages in the buying journey."""
    AWARENESS = "awareness"                    # Problem recognition
    CONSIDERATION = "consideration"            # Solution research
    DECISION = "decision"                      # Vendor selection
    PURCHASE = "purchase"                      # Transaction
    ONBOARDING = "onboarding"                  # Implementation
    ADVOCACY = "advocacy"                      # Post-purchase loyalty


class ChannelType(str, Enum):
    """Marketing/sales channel types."""
    ORGANIC_SEARCH = "organic_search"
    PAID_SEARCH = "paid_search"
    SOCIAL_LINKEDIN = "social_linkedin"
    SOCIAL_TWITTER = "social_twitter"
    SOCIAL_FACEBOOK = "social_facebook"
    EMAIL = "email"
    CONTENT_MARKETING = "content_marketing"
    WEBINARS = "webinars"
    EVENTS = "events"
    REFERRAL = "referral"
    PARTNERSHIPS = "partnerships"
    DIRECT_SALES = "direct_sales"
    PRODUCT_LED = "product_led"


class PriorityLevel(str, Enum):
    """Priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConfidenceLevel(str, Enum):
    """Confidence level in analysis findings."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# =============================================================================
# INPUT SCHEMAS
# =============================================================================


class AudienceBuildingInput(BaseModel):
    """Input for audience building operation."""

    # Product/Service context
    product_category: str = Field(
        description="Product or service category (e.g., 'CRM Software', 'Marketing Automation')"
    )
    value_proposition: str | None = Field(
        default=None,
        description="Core value proposition of the product"
    )

    # Target market
    target_market: str | None = Field(
        default=None,
        description="Target market description (e.g., 'B2B SaaS', 'E-commerce')"
    )
    geographic_focus: str | None = Field(
        default=None,
        description="Geographic focus (e.g., 'North America', 'Global')"
    )

    # Company characteristics
    target_company_sizes: list[CompanySize] = Field(
        default_factory=list,
        description="Target company sizes"
    )
    target_industries: list[str] = Field(
        default_factory=list,
        description="Target industries"
    )

    # Existing knowledge
    existing_customer_traits: list[str] = Field(
        default_factory=list,
        description="Known traits of existing successful customers"
    )
    known_pain_points: list[str] = Field(
        default_factory=list,
        description="Known pain points from customer feedback"
    )

    # Intelligence integration
    use_intelligence_data: bool = Field(
        default=True,
        description="Whether to use existing intelligence data from state (market_research, seo_data, web_analysis)"
    )

    # Analysis preferences
    num_personas: int = Field(
        default=3,
        ge=1,
        le=7,
        description="Number of personas to create (1-7)"
    )

    # Additional context
    additional_context: str | None = Field(
        default=None,
        description="Any additional context for audience building"
    )

    @field_validator("product_category")
    @classmethod
    def validate_product_category(cls, v):
        """Validate product category is not empty."""
        if not v or not v.strip():
            raise ValueError("product_category cannot be empty")
        return v.strip()


class AudienceBuildingRequest(BaseModel):
    """API request for audience building."""
    input: AudienceBuildingInput
    client_id: str = Field(description="Client ID for multi-tenant context")
    workflow_id: str | None = Field(default=None, description="Associated workflow ID")


# =============================================================================
# OUTPUT SCHEMAS - ICP (Ideal Customer Profile)
# =============================================================================


class ICPFirmographics(BaseModel):
    """Firmographic characteristics of ideal customer."""
    company_size: list[CompanySize] = Field(
        default_factory=list,
        description="Target company sizes"
    )
    employee_count_range: str | None = Field(
        default=None,
        description="Employee count range (e.g., '50-500')"
    )
    revenue_range: str | None = Field(
        default=None,
        description="Annual revenue range (e.g., '$5M-$50M')"
    )
    industries: list[str] = Field(
        default_factory=list,
        description="Target industries"
    )
    geographic_regions: list[str] = Field(
        default_factory=list,
        description="Target geographic regions"
    )
    business_model: str | None = Field(
        default=None,
        description="Business model type (B2B, B2C, SaaS, etc.)"
    )


class ICPTechnographics(BaseModel):
    """Technographic characteristics of ideal customer."""
    current_tools: list[str] = Field(
        default_factory=list,
        description="Tools they likely use currently"
    )
    tech_maturity: str | None = Field(
        default=None,
        description="Technology maturity level"
    )
    integration_needs: list[str] = Field(
        default_factory=list,
        description="Key integration requirements"
    )


class ICPBehavioral(BaseModel):
    """Behavioral characteristics of ideal customer."""
    buying_triggers: list[str] = Field(
        default_factory=list,
        description="Events that trigger buying"
    )
    decision_timeline: str | None = Field(
        default=None,
        description="Typical decision timeline"
    )
    budget_characteristics: str | None = Field(
        default=None,
        description="Budget considerations"
    )
    evaluation_criteria: list[str] = Field(
        default_factory=list,
        description="Key evaluation criteria"
    )


class ICPProfile(BaseModel):
    """Complete Ideal Customer Profile."""
    summary: str = Field(description="One-paragraph ICP summary")
    firmographics: ICPFirmographics = Field(
        default_factory=ICPFirmographics,
        description="Firmographic characteristics"
    )
    technographics: ICPTechnographics = Field(
        default_factory=ICPTechnographics,
        description="Technographic characteristics"
    )
    behavioral: ICPBehavioral = Field(
        default_factory=ICPBehavioral,
        description="Behavioral characteristics"
    )
    key_pain_points: list[str] = Field(
        default_factory=list,
        description="Primary pain points"
    )
    desired_outcomes: list[str] = Field(
        default_factory=list,
        description="Desired outcomes/goals"
    )
    disqualification_criteria: list[str] = Field(
        default_factory=list,
        description="Criteria that disqualify a prospect"
    )


# =============================================================================
# OUTPUT SCHEMAS - Buyer Personas
# =============================================================================


class PersonaDemographics(BaseModel):
    """Demographic information for a persona."""
    job_titles: list[str] = Field(
        default_factory=list,
        description="Common job titles"
    )
    seniority_level: str | None = Field(
        default=None,
        description="Seniority level (e.g., 'Director', 'VP', 'C-Level')"
    )
    department: str | None = Field(
        default=None,
        description="Department (e.g., 'Sales', 'Marketing', 'IT')"
    )
    age_range: str | None = Field(
        default=None,
        description="Typical age range"
    )
    education: str | None = Field(
        default=None,
        description="Education background"
    )
    experience_years: str | None = Field(
        default=None,
        description="Years of experience"
    )


class PersonaPsychographics(BaseModel):
    """Psychographic information for a persona."""
    goals: list[str] = Field(
        default_factory=list,
        description="Professional goals"
    )
    challenges: list[str] = Field(
        default_factory=list,
        description="Day-to-day challenges"
    )
    motivations: list[str] = Field(
        default_factory=list,
        description="What motivates them"
    )
    fears: list[str] = Field(
        default_factory=list,
        description="Professional fears/concerns"
    )
    values: list[str] = Field(
        default_factory=list,
        description="Values they prioritize"
    )


class PersonaBehavior(BaseModel):
    """Behavioral patterns for a persona."""
    information_sources: list[str] = Field(
        default_factory=list,
        description="Where they get information"
    )
    preferred_content_types: list[str] = Field(
        default_factory=list,
        description="Content types they consume"
    )
    social_platforms: list[str] = Field(
        default_factory=list,
        description="Social platforms they use"
    )
    buying_role: PersonaType = Field(
        default=PersonaType.INFLUENCER,
        description="Role in buying process"
    )
    decision_influence: str | None = Field(
        default=None,
        description="Level of influence in decisions"
    )


class BuyerPersona(BaseModel):
    """Complete buyer persona profile."""
    persona_name: str = Field(description="Memorable persona name (e.g., 'Sales Director Sarah')")
    persona_type: PersonaType = Field(
        default=PersonaType.DECISION_MAKER,
        description="Type of persona"
    )
    one_liner: str = Field(description="One-sentence persona description")
    demographics: PersonaDemographics = Field(
        default_factory=PersonaDemographics,
        description="Demographic information"
    )
    psychographics: PersonaPsychographics = Field(
        default_factory=PersonaPsychographics,
        description="Psychographic information"
    )
    behavior: PersonaBehavior = Field(
        default_factory=PersonaBehavior,
        description="Behavioral patterns"
    )
    pain_points: list[str] = Field(
        default_factory=list,
        description="Specific pain points"
    )
    objections: list[str] = Field(
        default_factory=list,
        description="Common objections"
    )
    key_messages: list[str] = Field(
        default_factory=list,
        description="Key messages that resonate"
    )
    quotes: list[str] = Field(
        default_factory=list,
        description="Representative quotes (what they might say)"
    )


# =============================================================================
# OUTPUT SCHEMAS - Pain Points
# =============================================================================


class PainPointAnalysis(BaseModel):
    """Detailed pain point analysis."""
    pain_point: str = Field(description="The pain point")
    description: str = Field(description="Detailed description")
    affected_personas: list[str] = Field(
        default_factory=list,
        description="Personas most affected"
    )
    severity: PriorityLevel = Field(
        default=PriorityLevel.MEDIUM,
        description="Severity level"
    )
    frequency: str | None = Field(
        default=None,
        description="How often this occurs"
    )
    current_solutions: list[str] = Field(
        default_factory=list,
        description="How they currently address this"
    )
    our_solution: str | None = Field(
        default=None,
        description="How our product addresses this"
    )


# =============================================================================
# OUTPUT SCHEMAS - Buying Journey
# =============================================================================


class JourneyTouchpoint(BaseModel):
    """Touchpoint in the buying journey."""
    channel: ChannelType = Field(description="Channel type")
    content_type: str = Field(description="Type of content")
    purpose: str = Field(description="Purpose of this touchpoint")
    key_message: str | None = Field(
        default=None,
        description="Key message to deliver"
    )


class JourneyStage(BaseModel):
    """Stage in the buying journey."""
    stage: BuyingStage = Field(description="Journey stage")
    description: str = Field(description="What happens at this stage")
    buyer_goals: list[str] = Field(
        default_factory=list,
        description="What buyer wants to achieve"
    )
    buyer_questions: list[str] = Field(
        default_factory=list,
        description="Questions buyer has"
    )
    buyer_emotions: list[str] = Field(
        default_factory=list,
        description="Emotional state of buyer"
    )
    touchpoints: list[JourneyTouchpoint] = Field(
        default_factory=list,
        description="Recommended touchpoints"
    )
    content_needs: list[str] = Field(
        default_factory=list,
        description="Content needed at this stage"
    )
    success_metrics: list[str] = Field(
        default_factory=list,
        description="How to measure success"
    )


# =============================================================================
# OUTPUT SCHEMAS - Messaging Matrix
# =============================================================================


class PersonaMessage(BaseModel):
    """Messaging for a specific persona."""
    persona_name: str = Field(description="Target persona name")
    value_proposition: str = Field(description="Tailored value proposition")
    key_benefits: list[str] = Field(
        default_factory=list,
        description="Key benefits to emphasize"
    )
    proof_points: list[str] = Field(
        default_factory=list,
        description="Evidence/proof points"
    )
    call_to_action: str | None = Field(
        default=None,
        description="Primary CTA"
    )
    tone: str | None = Field(
        default=None,
        description="Recommended tone"
    )
    words_to_use: list[str] = Field(
        default_factory=list,
        description="Keywords/phrases to use"
    )
    words_to_avoid: list[str] = Field(
        default_factory=list,
        description="Words/phrases to avoid"
    )


# =============================================================================
# OUTPUT SCHEMAS - Channel Strategy
# =============================================================================


class ChannelStrategy(BaseModel):
    """Strategy for a marketing/sales channel."""
    channel: ChannelType = Field(description="Channel type")
    priority: PriorityLevel = Field(
        default=PriorityLevel.MEDIUM,
        description="Priority level"
    )
    target_personas: list[str] = Field(
        default_factory=list,
        description="Personas to target on this channel"
    )
    journey_stages: list[BuyingStage] = Field(
        default_factory=list,
        description="Journey stages this channel serves"
    )
    content_types: list[str] = Field(
        default_factory=list,
        description="Content types for this channel"
    )
    estimated_effectiveness: str | None = Field(
        default=None,
        description="Estimated effectiveness"
    )
    key_tactics: list[str] = Field(
        default_factory=list,
        description="Key tactics to employ"
    )
    success_metrics: list[str] = Field(
        default_factory=list,
        description="Metrics to track"
    )


# =============================================================================
# OUTPUT SCHEMAS - Complete Result
# =============================================================================


class AudienceBuildingResult(BaseModel):
    """Complete audience building result."""

    # Summary
    analysis_summary: str = Field(
        description="Executive summary of audience analysis"
    )

    # ICP
    ideal_customer_profile: ICPProfile = Field(
        description="Ideal Customer Profile"
    )

    # Personas
    buyer_personas: list[BuyerPersona] = Field(
        default_factory=list,
        description="Buyer personas"
    )

    # Pain Points
    pain_point_analysis: list[PainPointAnalysis] = Field(
        default_factory=list,
        description="Detailed pain point analysis"
    )

    # Buying Journey
    buying_journey: list[JourneyStage] = Field(
        default_factory=list,
        description="Buying journey stages"
    )

    # Messaging
    messaging_matrix: list[PersonaMessage] = Field(
        default_factory=list,
        description="Messaging for each persona"
    )

    # Channels
    channel_strategy: list[ChannelStrategy] = Field(
        default_factory=list,
        description="Channel strategies"
    )

    # Recommendations
    quick_wins: list[str] = Field(
        default_factory=list,
        description="Quick win opportunities"
    )
    strategic_recommendations: list[str] = Field(
        default_factory=list,
        description="Long-term strategic recommendations"
    )

    # Metadata
    data_sources_used: list[str] = Field(
        default_factory=list,
        description="Data sources used (market_research, seo_data, etc.)"
    )
    confidence_level: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Overall confidence in analysis"
    )
    data_limitations: list[str] = Field(
        default_factory=list,
        description="Known limitations"
    )
    analysis_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When analysis was conducted"
    )


class AudienceBuildingResponse(BaseModel):
    """API response for audience building."""
    success: bool = Field(description="Whether analysis was successful")
    analysis_id: str = Field(description="Generated analysis ID")
    analysis_result: AudienceBuildingResult | None = Field(
        default=None,
        description="Analysis results"
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if failed"
    )
    processing_time_ms: int | None = Field(
        default=None,
        description="Processing time in milliseconds"
    )


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "PersonaType",
    "CompanySize",
    "BuyingStage",
    "ChannelType",
    "PriorityLevel",
    "ConfidenceLevel",
    # Input
    "AudienceBuildingInput",
    "AudienceBuildingRequest",
    # ICP
    "ICPFirmographics",
    "ICPTechnographics",
    "ICPBehavioral",
    "ICPProfile",
    # Personas
    "PersonaDemographics",
    "PersonaPsychographics",
    "PersonaBehavior",
    "BuyerPersona",
    # Pain Points
    "PainPointAnalysis",
    # Journey
    "JourneyTouchpoint",
    "JourneyStage",
    # Messaging
    "PersonaMessage",
    # Channels
    "ChannelStrategy",
    # Output
    "AudienceBuildingResult",
    "AudienceBuildingResponse",
]

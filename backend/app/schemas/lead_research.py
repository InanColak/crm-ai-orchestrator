"""
Lead Research Schemas (Phase 4.2)
==================================
Pydantic schemas for lead/company research operations.

This module defines structured schemas for:
- Lead research input (company/contact to research)
- Research results (enriched data from web search)
- Lead enrichment output for CRM integration
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ENUMS
# =============================================================================


class LeadSource(str, Enum):
    """Source of the lead."""
    MANUAL = "manual"          # User-provided lead
    CRM_IMPORT = "crm_import"  # Imported from CRM
    MEETING = "meeting"        # Derived from meeting notes
    REFERRAL = "referral"      # Referral from existing contact
    INBOUND = "inbound"        # Inbound inquiry


class ResearchDepth(str, Enum):
    """Depth of research to perform."""
    QUICK = "quick"        # Basic info only (company overview)
    STANDARD = "standard"  # Company + recent news
    DEEP = "deep"          # Comprehensive (company, news, competitors, key people)


class EnrichmentConfidence(str, Enum):
    """Confidence level of enriched data."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LeadQualificationScore(str, Enum):
    """Lead qualification tier."""
    HOT = "hot"            # High intent, immediate opportunity
    WARM = "warm"          # Interested, needs nurturing
    COLD = "cold"          # Low engagement, exploratory
    UNQUALIFIED = "unqualified"  # Does not fit ICP


# =============================================================================
# INPUT SCHEMAS
# =============================================================================


class LeadResearchInput(BaseModel):
    """Input for lead research operation."""

    # Company info (at least one required)
    company_name: str = Field(
        description="Company name to research"
    )
    company_domain: str | None = Field(
        default=None,
        description="Company website domain (e.g., 'anthropic.com')"
    )

    # Contact info (optional)
    contact_name: str | None = Field(
        default=None,
        description="Contact person's name"
    )
    contact_title: str | None = Field(
        default=None,
        description="Contact's job title"
    )
    contact_email: str | None = Field(
        default=None,
        description="Contact's email address"
    )
    contact_linkedin: str | None = Field(
        default=None,
        description="Contact's LinkedIn URL"
    )

    # Research context
    research_depth: ResearchDepth = Field(
        default=ResearchDepth.STANDARD,
        description="How thorough the research should be"
    )
    focus_areas: list[str] = Field(
        default_factory=list,
        description="Specific areas to focus research on (e.g., 'funding', 'technology')"
    )
    source: LeadSource = Field(
        default=LeadSource.MANUAL,
        description="Source of this lead"
    )

    # CRM context
    deal_id: str | None = Field(
        default=None,
        description="Associated deal ID in CRM"
    )
    existing_crm_data: dict[str, Any] | None = Field(
        default=None,
        description="Existing data from CRM to enrich"
    )

    # Additional context
    additional_context: str | None = Field(
        default=None,
        description="Any additional context for research"
    )

    @field_validator("company_name")
    @classmethod
    def validate_company_name(cls, v: str) -> str:
        """Ensure company name is not empty."""
        if not v or not v.strip():
            raise ValueError("company_name cannot be empty")
        return v.strip()


class LeadResearchRequest(BaseModel):
    """API request to research a lead."""
    input: LeadResearchInput
    client_id: str = Field(description="Client ID for multi-tenant context")
    workflow_id: str | None = Field(default=None, description="Associated workflow ID")


# =============================================================================
# OUTPUT SCHEMAS - Company Research
# =============================================================================


class CompanyOverview(BaseModel):
    """Basic company overview data."""
    name: str = Field(description="Official company name")
    description: str | None = Field(
        default=None,
        description="Company description/about"
    )
    website: str | None = Field(
        default=None,
        description="Official website URL"
    )
    industry: str | None = Field(
        default=None,
        description="Industry/sector"
    )
    founded_year: int | None = Field(
        default=None,
        description="Year company was founded"
    )
    headquarters: str | None = Field(
        default=None,
        description="HQ location (city, country)"
    )
    employee_count: str | None = Field(
        default=None,
        description="Employee count range (e.g., '50-200', '1000+')"
    )


class FundingInfo(BaseModel):
    """Company funding information."""
    total_raised: str | None = Field(
        default=None,
        description="Total funding raised (e.g., '$50M')"
    )
    last_round: str | None = Field(
        default=None,
        description="Last funding round type (e.g., 'Series B')"
    )
    last_round_amount: str | None = Field(
        default=None,
        description="Last round amount"
    )
    last_round_date: str | None = Field(
        default=None,
        description="Date of last funding round"
    )
    investors: list[str] = Field(
        default_factory=list,
        description="Known investors"
    )


class KeyPerson(BaseModel):
    """Key person at the company."""
    name: str = Field(description="Person's name")
    title: str | None = Field(
        default=None,
        description="Job title"
    )
    linkedin_url: str | None = Field(
        default=None,
        description="LinkedIn profile URL"
    )
    bio_snippet: str | None = Field(
        default=None,
        description="Brief bio if available"
    )


class TechnologyStack(BaseModel):
    """Technology information about the company."""
    technologies: list[str] = Field(
        default_factory=list,
        description="Known technologies used"
    )
    tech_categories: list[str] = Field(
        default_factory=list,
        description="Technology categories (e.g., 'Cloud', 'AI/ML')"
    )
    source: str | None = Field(
        default=None,
        description="Source of tech stack info"
    )


class NewsItem(BaseModel):
    """News article about the company."""
    title: str = Field(description="Article title")
    url: str = Field(description="Article URL")
    snippet: str | None = Field(
        default=None,
        description="Brief snippet from article"
    )
    published_date: str | None = Field(
        default=None,
        description="Publication date"
    )
    sentiment: str | None = Field(
        default=None,
        description="Sentiment (positive/neutral/negative)"
    )


class CompetitorInfo(BaseModel):
    """Competitor information."""
    name: str = Field(description="Competitor name")
    description: str | None = Field(
        default=None,
        description="Brief description"
    )
    website: str | None = Field(
        default=None,
        description="Competitor website"
    )
    competitive_advantage: str | None = Field(
        default=None,
        description="Their competitive advantage"
    )


class SocialPresence(BaseModel):
    """Company social media presence."""
    linkedin_url: str | None = Field(default=None)
    twitter_url: str | None = Field(default=None)
    facebook_url: str | None = Field(default=None)
    youtube_url: str | None = Field(default=None)
    blog_url: str | None = Field(default=None)


# =============================================================================
# OUTPUT SCHEMAS - Research Result
# =============================================================================


class LeadResearchResult(BaseModel):
    """Complete lead research result from LLM analysis."""

    # Company data
    company: CompanyOverview = Field(description="Company overview")
    funding: FundingInfo | None = Field(
        default=None,
        description="Funding information"
    )
    key_people: list[KeyPerson] = Field(
        default_factory=list,
        description="Key people identified"
    )
    technology: TechnologyStack | None = Field(
        default=None,
        description="Technology stack info"
    )
    recent_news: list[NewsItem] = Field(
        default_factory=list,
        description="Recent news articles"
    )
    competitors: list[CompetitorInfo] = Field(
        default_factory=list,
        description="Known competitors"
    )
    social_presence: SocialPresence | None = Field(
        default=None,
        description="Social media presence"
    )

    # Sales intelligence
    business_signals: list[str] = Field(
        default_factory=list,
        description="Buying signals or business indicators"
    )
    pain_points: list[str] = Field(
        default_factory=list,
        description="Potential pain points based on research"
    )
    opportunities: list[str] = Field(
        default_factory=list,
        description="Potential sales opportunities"
    )
    talking_points: list[str] = Field(
        default_factory=list,
        description="Suggested talking points for outreach"
    )

    # Qualification
    qualification_score: LeadQualificationScore = Field(
        default=LeadQualificationScore.WARM,
        description="Lead qualification tier"
    )
    qualification_reasoning: str | None = Field(
        default=None,
        description="Reasoning for qualification score"
    )

    # Metadata
    research_summary: str = Field(
        description="Executive summary of research findings"
    )
    confidence_level: EnrichmentConfidence = Field(
        default=EnrichmentConfidence.MEDIUM,
        description="Overall confidence in research accuracy"
    )
    sources_used: list[str] = Field(
        default_factory=list,
        description="URLs of sources used"
    )
    research_notes: str | None = Field(
        default=None,
        description="Additional notes about the research"
    )


# =============================================================================
# OUTPUT SCHEMAS - CRM Integration
# =============================================================================


class LeadEnrichmentPayload(BaseModel):
    """Data to update/create in CRM."""

    # Company fields (HubSpot properties)
    company_name: str
    company_description: str | None = None
    company_website: str | None = None
    company_industry: str | None = None
    company_size: str | None = None
    company_headquarters: str | None = None
    company_linkedin_url: str | None = None
    company_founded_year: int | None = None
    company_annual_revenue: str | None = None

    # Contact fields (if contact research was done)
    contact_name: str | None = None
    contact_title: str | None = None
    contact_email: str | None = None
    contact_phone: str | None = None
    contact_linkedin_url: str | None = None

    # Custom fields
    lead_score: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Lead score 0-100"
    )
    enrichment_date: str = Field(
        description="When enrichment was performed"
    )
    enrichment_source: str = Field(
        default="ai_research",
        description="Source of enrichment"
    )
    enrichment_notes: str | None = Field(
        default=None,
        description="Notes for CRM record"
    )

    # Research data as JSON
    research_data: dict[str, Any] | None = Field(
        default=None,
        description="Full research data as JSON for custom fields"
    )


class LeadResearchResponse(BaseModel):
    """API response for lead research."""
    success: bool = Field(description="Whether research was successful")
    lead_id: str = Field(description="Generated lead ID")
    research_result: LeadResearchResult | None = Field(
        default=None,
        description="Research results"
    )
    enrichment_payload: LeadEnrichmentPayload | None = Field(
        default=None,
        description="CRM enrichment payload"
    )
    needs_approval: bool = Field(
        default=True,
        description="Whether CRM update needs approval"
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
    "LeadSource",
    "ResearchDepth",
    "EnrichmentConfidence",
    "LeadQualificationScore",
    # Input
    "LeadResearchInput",
    "LeadResearchRequest",
    # Company Data
    "CompanyOverview",
    "FundingInfo",
    "KeyPerson",
    "TechnologyStack",
    "NewsItem",
    "CompetitorInfo",
    "SocialPresence",
    # Output
    "LeadResearchResult",
    "LeadEnrichmentPayload",
    "LeadResearchResponse",
]

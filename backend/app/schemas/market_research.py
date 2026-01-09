"""
Market Research Schemas (Phase 5.1)
====================================
Pydantic schemas for market research operations.

This module defines structured schemas for:
- Market research input (industry/market to research)
- Research results (trends, competitors, opportunities)
- Intelligence output for strategic planning

Part of Intelligence Squad - enables autonomous market intelligence gathering.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ENUMS
# =============================================================================


class ResearchScope(str, Enum):
    """Scope of market research."""
    NARROW = "narrow"        # Single market/segment
    STANDARD = "standard"    # Market + adjacent segments
    COMPREHENSIVE = "comprehensive"  # Full market ecosystem analysis


class MarketMaturity(str, Enum):
    """Market maturity stage."""
    EMERGING = "emerging"      # New market, high growth potential
    GROWING = "growing"        # Established growth trajectory
    MATURE = "mature"          # Stable, competitive market
    DECLINING = "declining"    # Shrinking market


class TrendSentiment(str, Enum):
    """Sentiment of a market trend."""
    BULLISH = "bullish"      # Positive outlook
    NEUTRAL = "neutral"      # Mixed signals
    BEARISH = "bearish"      # Negative outlook


class OpportunityPriority(str, Enum):
    """Priority level of identified opportunity."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConfidenceLevel(str, Enum):
    """Confidence level in research findings."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# =============================================================================
# INPUT SCHEMAS
# =============================================================================


class MarketResearchInput(BaseModel):
    """Input for market research operation."""

    # Primary research target
    industry: str = Field(
        description="Industry or market to research (e.g., 'B2B SaaS', 'FinTech', 'Healthcare AI')"
    )

    # Optional refinements
    sub_segment: str | None = Field(
        default=None,
        description="Specific sub-segment within the industry (e.g., 'Sales Automation')"
    )
    geographic_focus: str | None = Field(
        default=None,
        description="Geographic focus (e.g., 'North America', 'DACH region', 'Global')"
    )

    # Research parameters
    research_scope: ResearchScope = Field(
        default=ResearchScope.STANDARD,
        description="How comprehensive the research should be"
    )
    focus_areas: list[str] = Field(
        default_factory=list,
        description="Specific areas to focus on (e.g., 'emerging technologies', 'regulatory changes')"
    )

    # Competitor context
    known_competitors: list[str] = Field(
        default_factory=list,
        description="Known competitors to include in analysis"
    )
    exclude_competitors: list[str] = Field(
        default_factory=list,
        description="Competitors to exclude from analysis"
    )

    # Time context
    time_horizon: str = Field(
        default="12 months",
        description="Time horizon for trend analysis (e.g., '6 months', '2 years')"
    )

    # Additional context
    client_context: str | None = Field(
        default=None,
        description="Context about the client's position in this market"
    )
    additional_questions: list[str] = Field(
        default_factory=list,
        description="Specific questions to answer in the research"
    )

    @field_validator("industry")
    @classmethod
    def validate_industry(cls, v: str) -> str:
        """Ensure industry is not empty."""
        if not v or not v.strip():
            raise ValueError("industry cannot be empty")
        return v.strip()


class MarketResearchRequest(BaseModel):
    """API request to conduct market research."""
    input: MarketResearchInput
    client_id: str = Field(description="Client ID for multi-tenant context")
    workflow_id: str | None = Field(default=None, description="Associated workflow ID")


# =============================================================================
# OUTPUT SCHEMAS - Market Analysis Components
# =============================================================================


class MarketOverview(BaseModel):
    """High-level market overview."""
    market_name: str = Field(description="Name of the market/industry")
    description: str = Field(description="Brief market description")
    market_size: str | None = Field(
        default=None,
        description="Estimated market size (e.g., '$50B globally')"
    )
    growth_rate: str | None = Field(
        default=None,
        description="Annual growth rate (e.g., '15% CAGR')"
    )
    maturity: MarketMaturity = Field(
        default=MarketMaturity.GROWING,
        description="Market maturity stage"
    )
    key_drivers: list[str] = Field(
        default_factory=list,
        description="Key market growth drivers"
    )
    key_challenges: list[str] = Field(
        default_factory=list,
        description="Key market challenges"
    )


class MarketTrend(BaseModel):
    """Individual market trend."""
    title: str = Field(description="Trend title")
    description: str = Field(description="Trend description and implications")
    category: str = Field(
        description="Trend category (e.g., 'Technology', 'Regulatory', 'Consumer Behavior')"
    )
    sentiment: TrendSentiment = Field(
        default=TrendSentiment.NEUTRAL,
        description="Outlook sentiment for this trend"
    )
    impact_level: str = Field(
        default="medium",
        description="Impact level (low/medium/high)"
    )
    time_horizon: str | None = Field(
        default=None,
        description="When this trend is expected to peak/mature"
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence/sources supporting this trend"
    )


class CompetitorAnalysis(BaseModel):
    """Competitor analysis data."""
    name: str = Field(description="Competitor name")
    description: str | None = Field(
        default=None,
        description="Brief company description"
    )
    website: str | None = Field(
        default=None,
        description="Company website"
    )
    market_position: str | None = Field(
        default=None,
        description="Market position (leader/challenger/niche)"
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Key strengths"
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Known weaknesses"
    )
    recent_developments: list[str] = Field(
        default_factory=list,
        description="Recent news/developments"
    )
    estimated_market_share: str | None = Field(
        default=None,
        description="Estimated market share if known"
    )


class TargetSegment(BaseModel):
    """Target market segment profile."""
    segment_name: str = Field(description="Segment name")
    description: str = Field(description="Segment description")
    size_estimate: str | None = Field(
        default=None,
        description="Segment size estimate"
    )
    growth_potential: str = Field(
        default="medium",
        description="Growth potential (low/medium/high)"
    )
    key_characteristics: list[str] = Field(
        default_factory=list,
        description="Key segment characteristics"
    )
    pain_points: list[str] = Field(
        default_factory=list,
        description="Common pain points in this segment"
    )
    buying_criteria: list[str] = Field(
        default_factory=list,
        description="Key buying criteria"
    )
    ideal_customer_profile: str | None = Field(
        default=None,
        description="ICP description for this segment"
    )


class MarketOpportunity(BaseModel):
    """Identified market opportunity."""
    title: str = Field(description="Opportunity title")
    description: str = Field(description="Detailed opportunity description")
    opportunity_type: str = Field(
        description="Type (e.g., 'market gap', 'emerging segment', 'technology adoption')"
    )
    priority: OpportunityPriority = Field(
        default=OpportunityPriority.MEDIUM,
        description="Priority level"
    )
    target_segment: str | None = Field(
        default=None,
        description="Target segment for this opportunity"
    )
    estimated_value: str | None = Field(
        default=None,
        description="Estimated value/potential"
    )
    time_to_capture: str | None = Field(
        default=None,
        description="Estimated time to capture opportunity"
    )
    requirements: list[str] = Field(
        default_factory=list,
        description="Requirements to capture this opportunity"
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Associated risks"
    )


class MarketThreat(BaseModel):
    """Identified market threat."""
    title: str = Field(description="Threat title")
    description: str = Field(description="Threat description")
    threat_type: str = Field(
        description="Type (e.g., 'competitive', 'regulatory', 'technological', 'economic')"
    )
    severity: str = Field(
        default="medium",
        description="Severity level (low/medium/high/critical)"
    )
    likelihood: str = Field(
        default="medium",
        description="Likelihood (low/medium/high)"
    )
    mitigation_strategies: list[str] = Field(
        default_factory=list,
        description="Potential mitigation strategies"
    )


class NewsInsight(BaseModel):
    """News article with market insight."""
    title: str = Field(description="Article title")
    url: str = Field(description="Article URL")
    snippet: str | None = Field(
        default=None,
        description="Relevant snippet"
    )
    published_date: str | None = Field(
        default=None,
        description="Publication date"
    )
    relevance: str = Field(
        default="medium",
        description="Relevance to research (low/medium/high)"
    )
    key_insight: str | None = Field(
        default=None,
        description="Key insight from this article"
    )


# =============================================================================
# OUTPUT SCHEMAS - Complete Research Result
# =============================================================================


class MarketResearchResult(BaseModel):
    """Complete market research result from LLM analysis."""

    # Market Overview
    market_overview: MarketOverview = Field(
        description="High-level market overview"
    )

    # Trends Analysis
    trends: list[MarketTrend] = Field(
        default_factory=list,
        description="Identified market trends"
    )

    # Competitive Landscape
    competitors: list[CompetitorAnalysis] = Field(
        default_factory=list,
        description="Competitor analysis"
    )
    competitive_dynamics: str | None = Field(
        default=None,
        description="Summary of competitive dynamics"
    )

    # Target Segments
    target_segments: list[TargetSegment] = Field(
        default_factory=list,
        description="Target market segments"
    )

    # Opportunities & Threats
    opportunities: list[MarketOpportunity] = Field(
        default_factory=list,
        description="Identified opportunities"
    )
    threats: list[MarketThreat] = Field(
        default_factory=list,
        description="Identified threats"
    )

    # News & Insights
    recent_news: list[NewsInsight] = Field(
        default_factory=list,
        description="Recent relevant news"
    )

    # Strategic Recommendations
    strategic_recommendations: list[str] = Field(
        default_factory=list,
        description="Strategic recommendations based on research"
    )
    key_questions_answered: list[dict[str, str]] = Field(
        default_factory=list,
        description="Answers to specific questions if provided"
    )

    # Metadata
    research_summary: str = Field(
        description="Executive summary of research findings"
    )
    confidence_level: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Overall confidence in research accuracy"
    )
    data_limitations: list[str] = Field(
        default_factory=list,
        description="Known limitations of the research"
    )
    sources_used: list[str] = Field(
        default_factory=list,
        description="URLs of sources used"
    )
    research_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When research was conducted"
    )


class MarketResearchResponse(BaseModel):
    """API response for market research."""
    success: bool = Field(description="Whether research was successful")
    research_id: str = Field(description="Generated research ID")
    research_result: MarketResearchResult | None = Field(
        default=None,
        description="Research results"
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
    "ResearchScope",
    "MarketMaturity",
    "TrendSentiment",
    "OpportunityPriority",
    "ConfidenceLevel",
    # Input
    "MarketResearchInput",
    "MarketResearchRequest",
    # Market Analysis Components
    "MarketOverview",
    "MarketTrend",
    "CompetitorAnalysis",
    "TargetSegment",
    "MarketOpportunity",
    "MarketThreat",
    "NewsInsight",
    # Output
    "MarketResearchResult",
    "MarketResearchResponse",
]

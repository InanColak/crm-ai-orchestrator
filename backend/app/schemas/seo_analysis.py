"""
SEO Analysis Schemas (Phase 5.2)
=================================
Pydantic schemas for SEO analysis operations.

This module defines structured schemas for:
- SEO analysis input (target URL/domain, keywords)
- Analysis results (keywords, SERP, competitors, content gaps)
- SEO intelligence output for content strategy

Part of Intelligence Squad - enables autonomous SEO intelligence gathering.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ENUMS
# =============================================================================


class AnalysisDepth(str, Enum):
    """Depth of SEO analysis."""
    QUICK = "quick"            # Basic keyword and SERP check
    STANDARD = "standard"      # Keywords + competitors + content gaps
    COMPREHENSIVE = "comprehensive"  # Full SEO audit with technical insights


class KeywordDifficulty(str, Enum):
    """Keyword ranking difficulty level."""
    EASY = "easy"              # Low competition, easy to rank
    MEDIUM = "medium"          # Moderate competition
    HARD = "hard"              # High competition
    VERY_HARD = "very_hard"    # Extremely competitive


class KeywordIntent(str, Enum):
    """Search intent behind a keyword."""
    INFORMATIONAL = "informational"    # Seeking information
    NAVIGATIONAL = "navigational"      # Looking for specific site/page
    TRANSACTIONAL = "transactional"    # Ready to buy/convert
    COMMERCIAL = "commercial"          # Researching before purchase


class ContentType(str, Enum):
    """Type of content for SEO."""
    BLOG_POST = "blog_post"
    LANDING_PAGE = "landing_page"
    PRODUCT_PAGE = "product_page"
    GUIDE = "guide"
    COMPARISON = "comparison"
    LIST = "list"
    HOW_TO = "how_to"
    FAQ = "faq"


class SERPFeature(str, Enum):
    """SERP feature types."""
    FEATURED_SNIPPET = "featured_snippet"
    PEOPLE_ALSO_ASK = "people_also_ask"
    LOCAL_PACK = "local_pack"
    IMAGE_PACK = "image_pack"
    VIDEO_CAROUSEL = "video_carousel"
    KNOWLEDGE_PANEL = "knowledge_panel"
    TOP_STORIES = "top_stories"
    SHOPPING_RESULTS = "shopping_results"
    SITE_LINKS = "site_links"


class ContentGapPriority(str, Enum):
    """Priority for content gap opportunities."""
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


class SEOAnalysisInput(BaseModel):
    """Input for SEO analysis operation."""

    # Primary target (at least one required)
    target_url: str | None = Field(
        default=None,
        description="Target URL or domain to analyze (e.g., 'example.com', 'https://example.com/page')"
    )
    target_keywords: list[str] = Field(
        default_factory=list,
        description="Primary keywords to analyze (e.g., ['crm software', 'sales automation'])"
    )

    # Analysis context
    industry: str | None = Field(
        default=None,
        description="Industry context for relevance (e.g., 'B2B SaaS', 'E-commerce')"
    )
    geographic_target: str | None = Field(
        default=None,
        description="Geographic target market (e.g., 'United States', 'Germany')"
    )
    language: str = Field(
        default="en",
        description="Target language for SEO (e.g., 'en', 'de', 'es')"
    )

    # Analysis parameters
    analysis_depth: AnalysisDepth = Field(
        default=AnalysisDepth.STANDARD,
        description="How comprehensive the analysis should be"
    )
    include_competitors: bool = Field(
        default=True,
        description="Whether to include competitor analysis"
    )
    known_competitors: list[str] = Field(
        default_factory=list,
        description="Known competitor domains to analyze"
    )

    # Content focus
    content_types: list[ContentType] = Field(
        default_factory=list,
        description="Content types to focus on for recommendations"
    )

    # Additional context
    business_goals: list[str] = Field(
        default_factory=list,
        description="Business goals for SEO (e.g., 'increase organic traffic', 'improve conversions')"
    )
    additional_context: str | None = Field(
        default=None,
        description="Any additional context for the analysis"
    )

    @field_validator("target_keywords", "target_url")
    @classmethod
    def validate_has_target(cls, v, info):
        """Ensure at least one target is provided."""
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate that at least one target is provided."""
        if not self.target_url and not self.target_keywords:
            raise ValueError("At least one of target_url or target_keywords must be provided")


class SEOAnalysisRequest(BaseModel):
    """API request for SEO analysis."""
    input: SEOAnalysisInput
    client_id: str = Field(description="Client ID for multi-tenant context")
    workflow_id: str | None = Field(default=None, description="Associated workflow ID")


# =============================================================================
# OUTPUT SCHEMAS - Keyword Analysis
# =============================================================================


class KeywordData(BaseModel):
    """Individual keyword analysis data."""
    keyword: str = Field(description="The keyword phrase")
    search_volume: str | None = Field(
        default=None,
        description="Estimated monthly search volume (e.g., '1K-10K', '10K-100K')"
    )
    difficulty: KeywordDifficulty = Field(
        default=KeywordDifficulty.MEDIUM,
        description="Ranking difficulty"
    )
    intent: KeywordIntent = Field(
        default=KeywordIntent.INFORMATIONAL,
        description="Search intent"
    )
    cpc_estimate: str | None = Field(
        default=None,
        description="Estimated CPC if available (e.g., '$2.50')"
    )
    trend: str | None = Field(
        default=None,
        description="Trend direction (rising/stable/declining)"
    )
    relevance_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relevance to target business (0-1)"
    )


class KeywordCluster(BaseModel):
    """Group of related keywords."""
    cluster_name: str = Field(description="Name/theme of the cluster")
    primary_keyword: str = Field(description="Main keyword in cluster")
    related_keywords: list[str] = Field(
        default_factory=list,
        description="Related keywords in this cluster"
    )
    total_volume: str | None = Field(
        default=None,
        description="Combined search volume estimate"
    )
    average_difficulty: KeywordDifficulty = Field(
        default=KeywordDifficulty.MEDIUM,
        description="Average difficulty of cluster"
    )
    recommended_content_type: ContentType | None = Field(
        default=None,
        description="Best content type for this cluster"
    )


# =============================================================================
# OUTPUT SCHEMAS - SERP Analysis
# =============================================================================


class SERPResult(BaseModel):
    """Individual SERP result data."""
    position: int = Field(description="Ranking position (1-10)")
    url: str = Field(description="URL of the ranking page")
    title: str = Field(description="Page title")
    domain: str = Field(description="Domain name")
    snippet: str | None = Field(
        default=None,
        description="Meta description or snippet"
    )
    content_type: ContentType | None = Field(
        default=None,
        description="Detected content type"
    )
    word_count_estimate: str | None = Field(
        default=None,
        description="Estimated word count (e.g., '1500-2000')"
    )


class SERPAnalysis(BaseModel):
    """SERP analysis for a keyword."""
    keyword: str = Field(description="Analyzed keyword")
    serp_features: list[SERPFeature] = Field(
        default_factory=list,
        description="SERP features present"
    )
    top_results: list[SERPResult] = Field(
        default_factory=list,
        description="Top ranking results"
    )
    dominant_content_type: ContentType | None = Field(
        default=None,
        description="Most common content type in top results"
    )
    average_word_count: str | None = Field(
        default=None,
        description="Average word count of top results"
    )
    ranking_factors_observed: list[str] = Field(
        default_factory=list,
        description="Key ranking factors observed"
    )


# =============================================================================
# OUTPUT SCHEMAS - Competitor Analysis
# =============================================================================


class CompetitorSEOProfile(BaseModel):
    """SEO profile of a competitor."""
    domain: str = Field(description="Competitor domain")
    estimated_traffic: str | None = Field(
        default=None,
        description="Estimated organic traffic (e.g., '10K-50K/month')"
    )
    domain_authority: str | None = Field(
        default=None,
        description="Domain authority estimate (e.g., 'High', 'Medium')"
    )
    top_ranking_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords they rank well for"
    )
    content_strengths: list[str] = Field(
        default_factory=list,
        description="Content areas they excel in"
    )
    content_weaknesses: list[str] = Field(
        default_factory=list,
        description="Content gaps or weaknesses"
    )
    backlink_profile: str | None = Field(
        default=None,
        description="Backlink profile assessment"
    )


class KeywordGap(BaseModel):
    """Keyword gap opportunity vs competitors."""
    keyword: str = Field(description="Gap keyword")
    search_volume: str | None = Field(
        default=None,
        description="Estimated search volume"
    )
    difficulty: KeywordDifficulty = Field(
        default=KeywordDifficulty.MEDIUM,
        description="Ranking difficulty"
    )
    competitors_ranking: list[str] = Field(
        default_factory=list,
        description="Competitors ranking for this keyword"
    )
    opportunity_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Opportunity score (0-1)"
    )
    recommended_action: str | None = Field(
        default=None,
        description="Recommended action to capture this keyword"
    )


# =============================================================================
# OUTPUT SCHEMAS - Content Gaps & Opportunities
# =============================================================================


class ContentGap(BaseModel):
    """Content gap opportunity."""
    topic: str = Field(description="Content topic/theme")
    description: str = Field(description="Description of the gap")
    target_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords to target"
    )
    recommended_content_type: ContentType = Field(
        default=ContentType.BLOG_POST,
        description="Recommended content format"
    )
    estimated_traffic_potential: str | None = Field(
        default=None,
        description="Traffic potential if ranking achieved"
    )
    priority: ContentGapPriority = Field(
        default=ContentGapPriority.MEDIUM,
        description="Priority level"
    )
    competitor_coverage: str | None = Field(
        default=None,
        description="How well competitors cover this topic"
    )
    suggested_outline: list[str] = Field(
        default_factory=list,
        description="Suggested content outline/sections"
    )


class ContentRecommendation(BaseModel):
    """Specific content recommendation."""
    title_suggestion: str = Field(description="Suggested content title")
    target_keyword: str = Field(description="Primary target keyword")
    secondary_keywords: list[str] = Field(
        default_factory=list,
        description="Secondary keywords to include"
    )
    content_type: ContentType = Field(description="Content type")
    word_count_target: str = Field(
        default="1500-2000",
        description="Target word count range"
    )
    key_sections: list[str] = Field(
        default_factory=list,
        description="Key sections to include"
    )
    internal_linking_opportunities: list[str] = Field(
        default_factory=list,
        description="Internal linking suggestions"
    )
    priority: ContentGapPriority = Field(
        default=ContentGapPriority.MEDIUM,
        description="Content priority"
    )


# =============================================================================
# OUTPUT SCHEMAS - Technical SEO
# =============================================================================


class TechnicalSEOInsight(BaseModel):
    """Technical SEO observation or recommendation."""
    category: str = Field(
        description="Category (e.g., 'Site Speed', 'Mobile', 'Indexing')"
    )
    observation: str = Field(description="What was observed")
    impact: str = Field(
        default="medium",
        description="Impact level (low/medium/high)"
    )
    recommendation: str | None = Field(
        default=None,
        description="Recommended action"
    )


# =============================================================================
# OUTPUT SCHEMAS - Complete Analysis Result
# =============================================================================


class SEOAnalysisResult(BaseModel):
    """Complete SEO analysis result from LLM analysis."""

    # Overview
    analysis_summary: str = Field(
        description="Executive summary of SEO analysis findings"
    )
    overall_seo_score: str | None = Field(
        default=None,
        description="Overall SEO assessment (e.g., 'Good', 'Needs Improvement')"
    )

    # Keyword Analysis
    primary_keywords: list[KeywordData] = Field(
        default_factory=list,
        description="Analysis of primary target keywords"
    )
    keyword_clusters: list[KeywordCluster] = Field(
        default_factory=list,
        description="Keyword clusters identified"
    )
    long_tail_opportunities: list[str] = Field(
        default_factory=list,
        description="Long-tail keyword opportunities"
    )

    # SERP Analysis
    serp_analyses: list[SERPAnalysis] = Field(
        default_factory=list,
        description="SERP analysis for key terms"
    )
    serp_opportunities: list[str] = Field(
        default_factory=list,
        description="SERP feature opportunities"
    )

    # Competitor Analysis
    competitors: list[CompetitorSEOProfile] = Field(
        default_factory=list,
        description="Competitor SEO profiles"
    )
    keyword_gaps: list[KeywordGap] = Field(
        default_factory=list,
        description="Keyword gaps vs competitors"
    )
    competitive_advantages: list[str] = Field(
        default_factory=list,
        description="Potential competitive advantages"
    )

    # Content Opportunities
    content_gaps: list[ContentGap] = Field(
        default_factory=list,
        description="Content gap opportunities"
    )
    content_recommendations: list[ContentRecommendation] = Field(
        default_factory=list,
        description="Specific content recommendations"
    )

    # Technical SEO (if comprehensive analysis)
    technical_insights: list[TechnicalSEOInsight] = Field(
        default_factory=list,
        description="Technical SEO insights"
    )

    # Strategic Recommendations
    quick_wins: list[str] = Field(
        default_factory=list,
        description="Quick win opportunities"
    )
    strategic_recommendations: list[str] = Field(
        default_factory=list,
        description="Long-term strategic recommendations"
    )

    # Metadata
    confidence_level: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Overall confidence in analysis"
    )
    data_limitations: list[str] = Field(
        default_factory=list,
        description="Known limitations of the analysis"
    )
    sources_used: list[str] = Field(
        default_factory=list,
        description="Sources used in analysis"
    )
    analysis_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When analysis was conducted"
    )


class SEOAnalysisResponse(BaseModel):
    """API response for SEO analysis."""
    success: bool = Field(description="Whether analysis was successful")
    analysis_id: str = Field(description="Generated analysis ID")
    analysis_result: SEOAnalysisResult | None = Field(
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
    "AnalysisDepth",
    "KeywordDifficulty",
    "KeywordIntent",
    "ContentType",
    "SERPFeature",
    "ContentGapPriority",
    "ConfidenceLevel",
    # Input
    "SEOAnalysisInput",
    "SEOAnalysisRequest",
    # Keyword Analysis
    "KeywordData",
    "KeywordCluster",
    # SERP Analysis
    "SERPResult",
    "SERPAnalysis",
    # Competitor Analysis
    "CompetitorSEOProfile",
    "KeywordGap",
    # Content Opportunities
    "ContentGap",
    "ContentRecommendation",
    # Technical SEO
    "TechnicalSEOInsight",
    # Output
    "SEOAnalysisResult",
    "SEOAnalysisResponse",
]

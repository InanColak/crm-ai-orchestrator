"""
Web Analysis Schemas (Phase 5.3)
================================
Pydantic schemas for website and competitor analysis operations.

This module defines structured schemas for:
- Website structure and navigation analysis
- Technology stack detection
- Performance metrics and Core Web Vitals
- Content quality assessment
- Competitor website comparison

Part of Intelligence Squad - enables autonomous web intelligence gathering.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ENUMS
# =============================================================================


class AnalysisType(str, Enum):
    """Type of web analysis to perform."""
    QUICK = "quick"                    # Basic structure and tech check
    STANDARD = "standard"              # Full website analysis
    COMPETITIVE = "competitive"        # Focus on competitor comparison
    TECHNICAL = "technical"            # Deep technical audit


class PerformanceGrade(str, Enum):
    """Performance grade levels."""
    EXCELLENT = "excellent"            # 90-100 score
    GOOD = "good"                      # 70-89 score
    NEEDS_IMPROVEMENT = "needs_improvement"  # 50-69 score
    POOR = "poor"                      # Below 50


class ContentQuality(str, Enum):
    """Content quality assessment level."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MobileReadiness(str, Enum):
    """Mobile readiness status."""
    FULLY_RESPONSIVE = "fully_responsive"
    PARTIALLY_RESPONSIVE = "partially_responsive"
    NOT_RESPONSIVE = "not_responsive"
    MOBILE_FIRST = "mobile_first"


class SecurityStatus(str, Enum):
    """Website security status."""
    SECURE = "secure"                  # HTTPS, good practices
    PARTIALLY_SECURE = "partially_secure"
    INSECURE = "insecure"


class CMSType(str, Enum):
    """Common CMS types."""
    WORDPRESS = "wordpress"
    WEBFLOW = "webflow"
    SHOPIFY = "shopify"
    SQUARESPACE = "squarespace"
    WIX = "wix"
    DRUPAL = "drupal"
    CUSTOM = "custom"
    HEADLESS = "headless"
    UNKNOWN = "unknown"


class HostingType(str, Enum):
    """Hosting infrastructure type."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    VERCEL = "vercel"
    NETLIFY = "netlify"
    CLOUDFLARE = "cloudflare"
    TRADITIONAL = "traditional"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Confidence level in analysis findings."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# =============================================================================
# INPUT SCHEMAS
# =============================================================================


class WebAnalysisInput(BaseModel):
    """Input for web analysis operation."""

    # Primary targets
    target_urls: list[str] = Field(
        default_factory=list,
        description="URLs to analyze (e.g., ['https://example.com', 'https://competitor.com'])"
    )
    primary_domain: str | None = Field(
        default=None,
        description="Primary domain to analyze in depth"
    )

    # Analysis context
    industry: str | None = Field(
        default=None,
        description="Industry context for benchmarking (e.g., 'B2B SaaS', 'E-commerce')"
    )
    analysis_type: AnalysisType = Field(
        default=AnalysisType.STANDARD,
        description="Type of analysis to perform"
    )

    # Competitor analysis
    include_competitors: bool = Field(
        default=True,
        description="Whether to include competitor comparison"
    )
    competitor_urls: list[str] = Field(
        default_factory=list,
        description="Known competitor websites to compare"
    )

    # Focus areas
    focus_areas: list[str] = Field(
        default_factory=list,
        description="Specific areas to focus on (e.g., 'performance', 'UX', 'content')"
    )

    # Additional context
    business_goals: list[str] = Field(
        default_factory=list,
        description="Business goals for the analysis (e.g., 'improve conversion', 'reduce bounce rate')"
    )
    additional_context: str | None = Field(
        default=None,
        description="Any additional context for the analysis"
    )

    @field_validator("target_urls", "primary_domain")
    @classmethod
    def validate_has_target(cls, v, info):
        """Validate URL format."""
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate that at least one target is provided."""
        if not self.target_urls and not self.primary_domain:
            raise ValueError("At least one of target_urls or primary_domain must be provided")


class WebAnalysisRequest(BaseModel):
    """API request for web analysis."""
    input: WebAnalysisInput
    client_id: str = Field(description="Client ID for multi-tenant context")
    workflow_id: str | None = Field(default=None, description="Associated workflow ID")


# =============================================================================
# OUTPUT SCHEMAS - Site Structure
# =============================================================================


class PageInfo(BaseModel):
    """Information about a specific page."""
    url: str = Field(description="Page URL")
    title: str | None = Field(default=None, description="Page title")
    page_type: str | None = Field(
        default=None,
        description="Page type (homepage, product, blog, landing, etc.)"
    )
    estimated_word_count: str | None = Field(
        default=None,
        description="Estimated word count range"
    )
    has_cta: bool = Field(default=False, description="Has clear call-to-action")
    notes: str | None = Field(default=None, description="Additional notes")


class SiteStructure(BaseModel):
    """Website structure analysis."""
    total_pages_estimate: str | None = Field(
        default=None,
        description="Estimated total pages (e.g., '50-100')"
    )
    navigation_type: str | None = Field(
        default=None,
        description="Navigation style (mega-menu, simple, hamburger, etc.)"
    )
    url_structure: str | None = Field(
        default=None,
        description="URL structure pattern (clean, parameterized, etc.)"
    )
    key_pages: list[PageInfo] = Field(
        default_factory=list,
        description="Key pages identified"
    )
    sitemap_available: bool = Field(
        default=False,
        description="Whether sitemap.xml is available"
    )
    information_architecture: str | None = Field(
        default=None,
        description="Assessment of information architecture quality"
    )
    navigation_depth: str | None = Field(
        default=None,
        description="Average clicks to reach content (e.g., '2-3 clicks')"
    )


# =============================================================================
# OUTPUT SCHEMAS - Technology Stack
# =============================================================================


class TechnologyItem(BaseModel):
    """Individual technology detected."""
    name: str = Field(description="Technology name")
    category: str = Field(description="Category (frontend, backend, analytics, etc.)")
    version: str | None = Field(default=None, description="Version if detected")
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Detection confidence"
    )


class TechnologyStack(BaseModel):
    """Complete technology stack analysis."""
    cms: CMSType = Field(
        default=CMSType.UNKNOWN,
        description="Content Management System"
    )
    cms_details: str | None = Field(
        default=None,
        description="Additional CMS details"
    )
    hosting: HostingType = Field(
        default=HostingType.UNKNOWN,
        description="Hosting provider"
    )
    cdn: str | None = Field(
        default=None,
        description="CDN provider if detected"
    )
    frontend_framework: str | None = Field(
        default=None,
        description="Frontend framework (React, Vue, Angular, etc.)"
    )
    css_framework: str | None = Field(
        default=None,
        description="CSS framework (Tailwind, Bootstrap, etc.)"
    )
    analytics_tools: list[str] = Field(
        default_factory=list,
        description="Analytics tools detected (GA4, Mixpanel, etc.)"
    )
    marketing_tools: list[str] = Field(
        default_factory=list,
        description="Marketing tools (HubSpot, Intercom, Drift, etc.)"
    )
    other_technologies: list[TechnologyItem] = Field(
        default_factory=list,
        description="Other technologies detected"
    )
    security_features: list[str] = Field(
        default_factory=list,
        description="Security features (HTTPS, HSTS, CSP, etc.)"
    )


# =============================================================================
# OUTPUT SCHEMAS - Performance
# =============================================================================


class CoreWebVitals(BaseModel):
    """Core Web Vitals assessment."""
    lcp: str | None = Field(
        default=None,
        description="Largest Contentful Paint estimate"
    )
    fid: str | None = Field(
        default=None,
        description="First Input Delay estimate"
    )
    cls: str | None = Field(
        default=None,
        description="Cumulative Layout Shift estimate"
    )
    overall_assessment: str | None = Field(
        default=None,
        description="Overall Core Web Vitals assessment"
    )


class PerformanceMetrics(BaseModel):
    """Website performance metrics."""
    overall_grade: PerformanceGrade = Field(
        default=PerformanceGrade.NEEDS_IMPROVEMENT,
        description="Overall performance grade"
    )
    page_load_estimate: str | None = Field(
        default=None,
        description="Estimated page load time (e.g., '2-3 seconds')"
    )
    core_web_vitals: CoreWebVitals | None = Field(
        default=None,
        description="Core Web Vitals assessment"
    )
    mobile_readiness: MobileReadiness = Field(
        default=MobileReadiness.PARTIALLY_RESPONSIVE,
        description="Mobile readiness status"
    )
    mobile_score: str | None = Field(
        default=None,
        description="Mobile performance score estimate"
    )
    desktop_score: str | None = Field(
        default=None,
        description="Desktop performance score estimate"
    )
    optimization_opportunities: list[str] = Field(
        default_factory=list,
        description="Performance optimization opportunities"
    )
    performance_issues: list[str] = Field(
        default_factory=list,
        description="Identified performance issues"
    )


# =============================================================================
# OUTPUT SCHEMAS - Content Quality
# =============================================================================


class ContentMetrics(BaseModel):
    """Content quality metrics."""
    overall_quality: ContentQuality = Field(
        default=ContentQuality.MEDIUM,
        description="Overall content quality"
    )
    content_freshness: str | None = Field(
        default=None,
        description="Content freshness assessment"
    )
    content_depth: str | None = Field(
        default=None,
        description="Content depth assessment"
    )
    content_types_found: list[str] = Field(
        default_factory=list,
        description="Types of content found (blog, case studies, whitepapers, etc.)"
    )
    estimated_blog_posts: str | None = Field(
        default=None,
        description="Estimated number of blog posts"
    )
    publishing_frequency: str | None = Field(
        default=None,
        description="Estimated publishing frequency"
    )
    content_strengths: list[str] = Field(
        default_factory=list,
        description="Content strengths identified"
    )
    content_weaknesses: list[str] = Field(
        default_factory=list,
        description="Content weaknesses identified"
    )
    content_recommendations: list[str] = Field(
        default_factory=list,
        description="Content improvement recommendations"
    )


# =============================================================================
# OUTPUT SCHEMAS - UX Analysis
# =============================================================================


class UXAnalysis(BaseModel):
    """User experience analysis."""
    overall_ux_score: str | None = Field(
        default=None,
        description="Overall UX assessment (e.g., 'Good', 'Needs Work')"
    )
    design_style: str | None = Field(
        default=None,
        description="Design style (modern, traditional, minimalist, etc.)"
    )
    brand_consistency: str | None = Field(
        default=None,
        description="Brand consistency assessment"
    )
    cta_effectiveness: str | None = Field(
        default=None,
        description="Call-to-action effectiveness"
    )
    form_quality: str | None = Field(
        default=None,
        description="Form design and UX quality"
    )
    accessibility_notes: list[str] = Field(
        default_factory=list,
        description="Accessibility observations"
    )
    ux_strengths: list[str] = Field(
        default_factory=list,
        description="UX strengths"
    )
    ux_weaknesses: list[str] = Field(
        default_factory=list,
        description="UX weaknesses"
    )
    ux_recommendations: list[str] = Field(
        default_factory=list,
        description="UX improvement recommendations"
    )


# =============================================================================
# OUTPUT SCHEMAS - Competitor Comparison
# =============================================================================


class CompetitorWebProfile(BaseModel):
    """Competitor website profile."""
    domain: str = Field(description="Competitor domain")
    overall_assessment: str | None = Field(
        default=None,
        description="Overall website assessment"
    )
    tech_stack_summary: str | None = Field(
        default=None,
        description="Technology stack summary"
    )
    design_approach: str | None = Field(
        default=None,
        description="Design approach description"
    )
    content_strategy: str | None = Field(
        default=None,
        description="Content strategy observations"
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Website strengths"
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Website weaknesses"
    )
    notable_features: list[str] = Field(
        default_factory=list,
        description="Notable features or innovations"
    )
    traffic_estimate: str | None = Field(
        default=None,
        description="Estimated traffic range"
    )


class CompetitiveInsight(BaseModel):
    """Insight from competitive analysis."""
    category: str = Field(description="Insight category")
    insight: str = Field(description="The insight")
    action_item: str | None = Field(
        default=None,
        description="Recommended action"
    )
    priority: str = Field(
        default="medium",
        description="Priority level (high, medium, low)"
    )


# =============================================================================
# OUTPUT SCHEMAS - Security Assessment
# =============================================================================


class SecurityAssessment(BaseModel):
    """Basic security assessment."""
    overall_status: SecurityStatus = Field(
        default=SecurityStatus.PARTIALLY_SECURE,
        description="Overall security status"
    )
    https_enabled: bool = Field(
        default=True,
        description="HTTPS enabled"
    )
    ssl_certificate_valid: bool = Field(
        default=True,
        description="SSL certificate valid"
    )
    security_headers: list[str] = Field(
        default_factory=list,
        description="Security headers detected"
    )
    potential_issues: list[str] = Field(
        default_factory=list,
        description="Potential security issues"
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Security recommendations"
    )


# =============================================================================
# OUTPUT SCHEMAS - Complete Analysis Result
# =============================================================================


class WebAnalysisResult(BaseModel):
    """Complete web analysis result from LLM analysis."""

    # Overview
    analysis_summary: str = Field(
        description="Executive summary of web analysis findings"
    )
    analyzed_domain: str | None = Field(
        default=None,
        description="Primary domain analyzed"
    )

    # Site Structure
    site_structure: SiteStructure | None = Field(
        default=None,
        description="Site structure analysis"
    )

    # Technology
    technology_stack: TechnologyStack | None = Field(
        default=None,
        description="Technology stack analysis"
    )

    # Performance
    performance: PerformanceMetrics | None = Field(
        default=None,
        description="Performance metrics"
    )

    # Content
    content_analysis: ContentMetrics | None = Field(
        default=None,
        description="Content quality analysis"
    )

    # UX
    ux_analysis: UXAnalysis | None = Field(
        default=None,
        description="User experience analysis"
    )

    # Security
    security: SecurityAssessment | None = Field(
        default=None,
        description="Security assessment"
    )

    # Competitor Analysis
    competitor_profiles: list[CompetitorWebProfile] = Field(
        default_factory=list,
        description="Competitor website profiles"
    )
    competitive_insights: list[CompetitiveInsight] = Field(
        default_factory=list,
        description="Competitive insights"
    )

    # Recommendations
    quick_wins: list[str] = Field(
        default_factory=list,
        description="Quick win improvements"
    )
    strategic_recommendations: list[str] = Field(
        default_factory=list,
        description="Long-term strategic recommendations"
    )
    priority_actions: list[str] = Field(
        default_factory=list,
        description="Priority action items"
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


class WebAnalysisResponse(BaseModel):
    """API response for web analysis."""
    success: bool = Field(description="Whether analysis was successful")
    analysis_id: str = Field(description="Generated analysis ID")
    analysis_result: WebAnalysisResult | None = Field(
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
    "AnalysisType",
    "PerformanceGrade",
    "ContentQuality",
    "MobileReadiness",
    "SecurityStatus",
    "CMSType",
    "HostingType",
    "ConfidenceLevel",
    # Input
    "WebAnalysisInput",
    "WebAnalysisRequest",
    # Site Structure
    "PageInfo",
    "SiteStructure",
    # Technology
    "TechnologyItem",
    "TechnologyStack",
    # Performance
    "CoreWebVitals",
    "PerformanceMetrics",
    # Content
    "ContentMetrics",
    # UX
    "UXAnalysis",
    # Competitors
    "CompetitorWebProfile",
    "CompetitiveInsight",
    # Security
    "SecurityAssessment",
    # Output
    "WebAnalysisResult",
    "WebAnalysisResponse",
]

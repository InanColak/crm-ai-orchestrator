"""
Unit Tests for Web Analysis Agent (Phase 5.3)
=============================================
Comprehensive tests for web analysis operations.

Test Coverage:
- Schema validation (WebAnalysisInput, WebAnalysisResult, etc.)
- Node execution with mocked Tavily and LLM
- Fallback behavior on errors
- State conversion
- Various analysis types
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from backend.app.schemas.web_analysis import (
    # Enums
    AnalysisType,
    PerformanceGrade,
    ContentQuality,
    MobileReadiness,
    SecurityStatus,
    CMSType,
    HostingType,
    ConfidenceLevel,
    # Input
    WebAnalysisInput,
    WebAnalysisRequest,
    # Site Structure
    PageInfo,
    SiteStructure,
    # Technology
    TechnologyItem,
    TechnologyStack,
    # Performance
    CoreWebVitals,
    PerformanceMetrics,
    # Content
    ContentMetrics,
    # UX
    UXAnalysis,
    # Security
    SecurityAssessment,
    # Competitors
    CompetitorWebProfile,
    CompetitiveInsight,
    # Output
    WebAnalysisResult,
    WebAnalysisResponse,
)
from backend.graph.state import (
    OrchestratorState,
    create_initial_state,
    CRMProvider,
    WorkflowStatus,
    AgentMessage,
)
from backend.graph.nodes import (
    web_analysis_node,
    _conduct_web_research_with_tavily,
    _analyze_web_with_llm,
    _convert_web_analysis_to_state_format,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_web_input():
    """Sample web analysis input."""
    return WebAnalysisInput(
        target_urls=["https://example-crm.com"],
        primary_domain="example-crm.com",
        industry="B2B SaaS",
        analysis_type=AnalysisType.STANDARD,
        include_competitors=True,
        competitor_urls=["https://salesforce.com", "https://hubspot.com"],
        focus_areas=["performance", "UX"],
        business_goals=["improve conversion rate", "reduce bounce rate"],
    )


@pytest.fixture
def sample_site_structure():
    """Sample site structure."""
    return SiteStructure(
        total_pages_estimate="50-100",
        navigation_type="mega-menu",
        url_structure="clean",
        key_pages=[
            PageInfo(
                url="https://example-crm.com/",
                title="Home - Example CRM",
                page_type="homepage",
                estimated_word_count="800-1000",
                has_cta=True,
                notes="Strong hero section with clear value proposition",
            ),
            PageInfo(
                url="https://example-crm.com/features",
                title="Features - Example CRM",
                page_type="product",
                estimated_word_count="1500-2000",
                has_cta=True,
                notes="Comprehensive feature overview",
            ),
        ],
        sitemap_available=True,
        information_architecture="Well-organized",
        navigation_depth="2-3 clicks",
    )


@pytest.fixture
def sample_technology_stack():
    """Sample technology stack."""
    return TechnologyStack(
        cms=CMSType.WEBFLOW,
        cms_details="Webflow with custom integrations",
        hosting=HostingType.CLOUDFLARE,
        cdn="Cloudflare",
        frontend_framework="React",
        css_framework="Tailwind CSS",
        analytics_tools=["Google Analytics 4", "Mixpanel"],
        marketing_tools=["HubSpot", "Intercom"],
        other_technologies=[
            TechnologyItem(
                name="Stripe",
                category="payment",
                version=None,
                confidence=ConfidenceLevel.HIGH,
            ),
        ],
        security_features=["HTTPS", "HSTS", "CSP"],
    )


@pytest.fixture
def sample_performance():
    """Sample performance metrics."""
    return PerformanceMetrics(
        overall_grade=PerformanceGrade.GOOD,
        page_load_estimate="2-3 seconds",
        core_web_vitals=CoreWebVitals(
            lcp="Good",
            fid="Good",
            cls="Needs Improvement",
            overall_assessment="Mostly passing",
        ),
        mobile_readiness=MobileReadiness.FULLY_RESPONSIVE,
        mobile_score="Good",
        desktop_score="Excellent",
        optimization_opportunities=["Optimize images", "Enable lazy loading"],
        performance_issues=["Large JavaScript bundle", "Render-blocking CSS"],
    )


@pytest.fixture
def sample_content_metrics():
    """Sample content metrics."""
    return ContentMetrics(
        overall_quality=ContentQuality.HIGH,
        content_freshness="Current",
        content_depth="Comprehensive",
        content_types_found=["blog", "case studies", "product pages", "videos"],
        estimated_blog_posts="50-100",
        publishing_frequency="Weekly",
        content_strengths=["Strong thought leadership", "Good use of visuals"],
        content_weaknesses=["Limited localization"],
        content_recommendations=["Add more case studies", "Create comparison guides"],
    )


@pytest.fixture
def sample_ux_analysis():
    """Sample UX analysis."""
    return UXAnalysis(
        overall_ux_score="Good",
        design_style="Modern and clean",
        brand_consistency="Consistent",
        cta_effectiveness="Strong",
        form_quality="Good",
        accessibility_notes=["Alt text present", "Good color contrast"],
        ux_strengths=["Clear navigation", "Fast load times", "Mobile-friendly"],
        ux_weaknesses=["Some forms too long", "Limited search functionality"],
        ux_recommendations=["Add breadcrumbs", "Implement search suggestions"],
    )


@pytest.fixture
def sample_security():
    """Sample security assessment."""
    return SecurityAssessment(
        overall_status=SecurityStatus.SECURE,
        https_enabled=True,
        ssl_certificate_valid=True,
        security_headers=["HSTS", "X-Content-Type-Options", "X-Frame-Options"],
        potential_issues=["Missing CSP header"],
        recommendations=["Implement Content Security Policy"],
    )


@pytest.fixture
def sample_competitor_profiles():
    """Sample competitor web profiles."""
    return [
        CompetitorWebProfile(
            domain="salesforce.com",
            overall_assessment="Enterprise-grade with comprehensive features",
            tech_stack_summary="Custom platform with React frontend",
            design_approach="Corporate, feature-rich",
            content_strategy="Active blog, extensive resource library",
            strengths=["Brand recognition", "Comprehensive platform", "Strong SEO"],
            weaknesses=["Complex navigation", "Slow load times"],
            notable_features=["AI-powered recommendations", "Extensive integrations"],
            traffic_estimate="50M+/month",
        ),
        CompetitorWebProfile(
            domain="hubspot.com",
            overall_assessment="User-friendly with strong marketing focus",
            tech_stack_summary="Next.js with headless CMS",
            design_approach="Modern, conversion-focused",
            content_strategy="Inbound marketing leader, academy platform",
            strengths=["Great UX", "Fast performance", "Strong content"],
            weaknesses=["Limited enterprise features"],
            notable_features=["Free tools", "HubSpot Academy"],
            traffic_estimate="30M+/month",
        ),
    ]


@pytest.fixture
def sample_competitive_insights():
    """Sample competitive insights."""
    return [
        CompetitiveInsight(
            category="Technology",
            insight="Competitors are adopting JAMstack architecture for better performance",
            action_item="Consider migrating to Next.js or similar framework",
            priority="medium",
        ),
        CompetitiveInsight(
            category="Content",
            insight="Top competitors have extensive educational content hubs",
            action_item="Develop a learning center with courses and certifications",
            priority="high",
        ),
    ]


@pytest.fixture
def sample_web_result(
    sample_site_structure,
    sample_technology_stack,
    sample_performance,
    sample_content_metrics,
    sample_ux_analysis,
    sample_security,
    sample_competitor_profiles,
    sample_competitive_insights,
):
    """Sample complete web analysis result."""
    return WebAnalysisResult(
        analysis_summary="Example CRM has a well-designed website with good performance. "
                        "Modern tech stack and strong content, but opportunities exist to "
                        "improve mobile experience and add more educational content.",
        analyzed_domain="example-crm.com",
        site_structure=sample_site_structure,
        technology_stack=sample_technology_stack,
        performance=sample_performance,
        content_analysis=sample_content_metrics,
        ux_analysis=sample_ux_analysis,
        security=sample_security,
        competitor_profiles=sample_competitor_profiles,
        competitive_insights=sample_competitive_insights,
        quick_wins=[
            "Optimize hero image for faster LCP",
            "Add schema markup for rich snippets",
            "Implement breadcrumb navigation",
        ],
        strategic_recommendations=[
            "Develop a content hub with educational resources",
            "Implement progressive web app features",
            "Add personalization based on visitor segments",
        ],
        priority_actions=[
            "High: Fix CLS issues on mobile",
            "High: Implement CSP header",
            "Medium: Add more customer testimonials",
        ],
        confidence_level=ConfidenceLevel.MEDIUM,
        data_limitations=["Analysis based on publicly available data"],
        sources_used=["Tavily web search", "Public technology databases"],
    )


@pytest.fixture
def initial_web_analysis_state(sample_web_input):
    """Initial state with web analysis input."""
    state = create_initial_state(
        client_id="client-12345",
        client_name="Test Consulting",
        crm_provider=CRMProvider.HUBSPOT,
        workflow_type="intelligence_only",
    )
    # Add web analysis input as a message
    state["messages"] = [
        AgentMessage(
            message_id=str(uuid4()),
            from_agent="user",
            to_agent="web_analysis",
            message_type="web_analysis_input",
            content="Analyze this website",
            metadata={
                "target_urls": sample_web_input.target_urls,
                "primary_domain": sample_web_input.primary_domain,
                "industry": sample_web_input.industry,
                "analysis_type": sample_web_input.analysis_type.value,
                "include_competitors": sample_web_input.include_competitors,
                "competitor_urls": sample_web_input.competitor_urls,
                "focus_areas": sample_web_input.focus_areas,
                "business_goals": sample_web_input.business_goals,
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    ]
    return state


@pytest.fixture
def mock_tavily_search_response():
    """Mock Tavily search response."""
    mock_result = MagicMock()
    mock_result.answer = "Example CRM uses modern web technologies with good performance."
    mock_result.results = [
        MagicMock(
            title="Example CRM Website Analysis",
            url="https://example.com/analysis",
            content="The website uses Webflow CMS with React components...",
            score=0.95,
            published_date="2024-01-15",
        ),
        MagicMock(
            title="B2B SaaS Website Best Practices",
            url="https://example.com/best-practices",
            content="Modern B2B websites should focus on performance and UX...",
            score=0.90,
            published_date="2024-01-10",
        ),
    ]
    return mock_result


# =============================================================================
# SCHEMA VALIDATION TESTS
# =============================================================================


class TestWebAnalysisSchemas:
    """Tests for Pydantic schema validation."""

    def test_web_analysis_input_valid(self, sample_web_input):
        """Test valid web analysis input."""
        assert sample_web_input.primary_domain == "example-crm.com"
        assert len(sample_web_input.target_urls) == 1
        assert sample_web_input.analysis_type == AnalysisType.STANDARD
        assert len(sample_web_input.competitor_urls) == 2

    def test_web_analysis_input_with_domain_only(self):
        """Test web input with domain only."""
        input_data = WebAnalysisInput(primary_domain="example.com")
        assert input_data.primary_domain == "example.com"
        assert input_data.target_urls == []
        assert input_data.analysis_type == AnalysisType.STANDARD

    def test_web_analysis_input_with_urls_only(self):
        """Test web input with URLs only."""
        input_data = WebAnalysisInput(target_urls=["https://example.com", "https://test.com"])
        assert input_data.primary_domain is None
        assert len(input_data.target_urls) == 2

    def test_web_analysis_input_no_target_fails(self):
        """Test that web input without URLs or domain raises error."""
        with pytest.raises(Exception, match="At least one of target_urls or primary_domain"):
            WebAnalysisInput()

    def test_site_structure_valid(self, sample_site_structure):
        """Test valid site structure."""
        assert sample_site_structure.total_pages_estimate == "50-100"
        assert sample_site_structure.navigation_type == "mega-menu"
        assert len(sample_site_structure.key_pages) == 2
        assert sample_site_structure.sitemap_available is True

    def test_page_info_valid(self, sample_site_structure):
        """Test valid page info."""
        page = sample_site_structure.key_pages[0]
        assert page.url == "https://example-crm.com/"
        assert page.page_type == "homepage"
        assert page.has_cta is True

    def test_technology_stack_valid(self, sample_technology_stack):
        """Test valid technology stack."""
        assert sample_technology_stack.cms == CMSType.WEBFLOW
        assert sample_technology_stack.hosting == HostingType.CLOUDFLARE
        assert sample_technology_stack.frontend_framework == "React"
        assert len(sample_technology_stack.analytics_tools) == 2

    def test_performance_metrics_valid(self, sample_performance):
        """Test valid performance metrics."""
        assert sample_performance.overall_grade == PerformanceGrade.GOOD
        assert sample_performance.mobile_readiness == MobileReadiness.FULLY_RESPONSIVE
        assert len(sample_performance.optimization_opportunities) == 2

    def test_core_web_vitals_valid(self, sample_performance):
        """Test valid Core Web Vitals."""
        cwv = sample_performance.core_web_vitals
        assert cwv.lcp == "Good"
        assert cwv.fid == "Good"
        assert cwv.cls == "Needs Improvement"

    def test_content_metrics_valid(self, sample_content_metrics):
        """Test valid content metrics."""
        assert sample_content_metrics.overall_quality == ContentQuality.HIGH
        assert sample_content_metrics.publishing_frequency == "Weekly"
        assert len(sample_content_metrics.content_types_found) == 4

    def test_ux_analysis_valid(self, sample_ux_analysis):
        """Test valid UX analysis."""
        assert sample_ux_analysis.overall_ux_score == "Good"
        assert sample_ux_analysis.brand_consistency == "Consistent"
        assert len(sample_ux_analysis.ux_strengths) == 3

    def test_security_assessment_valid(self, sample_security):
        """Test valid security assessment."""
        assert sample_security.overall_status == SecurityStatus.SECURE
        assert sample_security.https_enabled is True
        assert len(sample_security.security_headers) == 3

    def test_competitor_web_profile_valid(self, sample_competitor_profiles):
        """Test valid competitor web profile."""
        competitor = sample_competitor_profiles[0]
        assert competitor.domain == "salesforce.com"
        assert len(competitor.strengths) == 3
        assert competitor.traffic_estimate == "50M+/month"

    def test_competitive_insight_valid(self, sample_competitive_insights):
        """Test valid competitive insight."""
        insight = sample_competitive_insights[0]
        assert insight.category == "Technology"
        assert insight.priority == "medium"

    def test_web_analysis_result_valid(self, sample_web_result):
        """Test valid web analysis result."""
        assert sample_web_result.analyzed_domain == "example-crm.com"
        assert sample_web_result.site_structure is not None
        assert sample_web_result.technology_stack is not None
        assert len(sample_web_result.competitor_profiles) == 2
        assert sample_web_result.confidence_level == ConfidenceLevel.MEDIUM


class TestAnalysisTypeEnum:
    """Tests for AnalysisType enum."""

    def test_quick_type(self):
        """Test quick analysis type."""
        assert AnalysisType.QUICK.value == "quick"

    def test_standard_type(self):
        """Test standard analysis type."""
        assert AnalysisType.STANDARD.value == "standard"

    def test_competitive_type(self):
        """Test competitive analysis type."""
        assert AnalysisType.COMPETITIVE.value == "competitive"

    def test_technical_type(self):
        """Test technical analysis type."""
        assert AnalysisType.TECHNICAL.value == "technical"


class TestPerformanceGradeEnum:
    """Tests for PerformanceGrade enum."""

    def test_all_grades(self):
        """Test all performance grades."""
        assert PerformanceGrade.EXCELLENT.value == "excellent"
        assert PerformanceGrade.GOOD.value == "good"
        assert PerformanceGrade.NEEDS_IMPROVEMENT.value == "needs_improvement"
        assert PerformanceGrade.POOR.value == "poor"


class TestCMSTypeEnum:
    """Tests for CMSType enum."""

    def test_common_cms_types(self):
        """Test common CMS types."""
        assert CMSType.WORDPRESS.value == "wordpress"
        assert CMSType.WEBFLOW.value == "webflow"
        assert CMSType.SHOPIFY.value == "shopify"
        assert CMSType.CUSTOM.value == "custom"
        assert CMSType.HEADLESS.value == "headless"


class TestMobileReadinessEnum:
    """Tests for MobileReadiness enum."""

    def test_all_mobile_states(self):
        """Test all mobile readiness states."""
        assert MobileReadiness.MOBILE_FIRST.value == "mobile_first"
        assert MobileReadiness.FULLY_RESPONSIVE.value == "fully_responsive"
        assert MobileReadiness.PARTIALLY_RESPONSIVE.value == "partially_responsive"
        assert MobileReadiness.NOT_RESPONSIVE.value == "not_responsive"


class TestSecurityStatusEnum:
    """Tests for SecurityStatus enum."""

    def test_all_security_states(self):
        """Test all security states."""
        assert SecurityStatus.SECURE.value == "secure"
        assert SecurityStatus.PARTIALLY_SECURE.value == "partially_secure"
        assert SecurityStatus.INSECURE.value == "insecure"


# =============================================================================
# NODE EXECUTION TESTS
# =============================================================================


class TestWebAnalysisNode:
    """Tests for web_analysis_node function."""

    def test_node_without_input_returns_error(self):
        """Test node returns error when no input is provided."""
        state = create_initial_state(
            client_id="client-123",
            client_name="Test Client",
            crm_provider=CRMProvider.HUBSPOT,
            workflow_type="intelligence_only",
        )
        state["messages"] = []

        result = web_analysis_node(state)

        assert "agent_execution_log" in result
        assert result["agent_execution_log"][0]["action"] == "error"
        assert "messages" in result
        assert result["messages"][0]["message_type"] == "error"
        assert "No web analysis input" in result["messages"][0]["content"]

    @patch("backend.graph.nodes._conduct_web_research_with_tavily")
    @patch("backend.graph.nodes._analyze_web_with_llm")
    def test_node_successful_execution(
        self,
        mock_llm_analyze,
        mock_tavily_research,
        initial_web_analysis_state,
        sample_web_result,
    ):
        """Test successful web analysis node execution."""
        # Setup mocks
        mock_tavily_research.return_value = {
            "structure_search_results": "Test structure data",
            "technology_search_results": "Test tech data",
            "performance_search_results": "Test performance data",
            "content_ux_search_results": "Test content data",
            "competitor_search_results": "Test competitor data",
            "security_search_results": None,
        }
        mock_llm_analyze.return_value = sample_web_result

        # Create async mock
        async def async_tavily_mock(*args, **kwargs):
            return mock_tavily_research.return_value

        async def async_llm_mock(*args, **kwargs):
            return mock_llm_analyze.return_value

        mock_tavily_research.side_effect = async_tavily_mock
        mock_llm_analyze.side_effect = async_llm_mock

        result = web_analysis_node(initial_web_analysis_state)

        assert "web_analysis" in result
        assert "agent_execution_log" in result
        assert result["agent_execution_log"][0]["action"] == "analyze_web"
        assert "messages" in result
        assert result["messages"][0]["message_type"] == "info"

    @patch("backend.graph.nodes._conduct_web_research_with_tavily")
    def test_node_handles_tavily_error(
        self,
        mock_tavily_research,
        initial_web_analysis_state,
    ):
        """Test node handles Tavily errors gracefully."""
        from backend.services.tavily_service import TavilyError

        async def raise_tavily_error(*args, **kwargs):
            raise TavilyError("API rate limit exceeded")

        mock_tavily_research.side_effect = raise_tavily_error

        result = web_analysis_node(initial_web_analysis_state)

        assert "agent_execution_log" in result
        assert result["agent_execution_log"][0]["action"] == "error"
        assert "error_message" in result
        assert "Tavily" in result["error_message"]

    @patch("backend.graph.nodes._conduct_web_research_with_tavily")
    @patch("backend.graph.nodes._analyze_web_with_llm")
    def test_node_handles_llm_error(
        self,
        mock_llm_analyze,
        mock_tavily_research,
        initial_web_analysis_state,
    ):
        """Test node handles LLM errors gracefully."""
        from backend.services.llm_service import LLMError

        async def async_tavily_mock(*args, **kwargs):
            return {"structure_search_results": "data"}

        async def raise_llm_error(*args, **kwargs):
            raise LLMError("Model overloaded")

        mock_tavily_research.side_effect = async_tavily_mock
        mock_llm_analyze.side_effect = raise_llm_error

        result = web_analysis_node(initial_web_analysis_state)

        assert "agent_execution_log" in result
        assert result["agent_execution_log"][0]["action"] == "error"
        assert "error_message" in result
        assert "LLM" in result["error_message"]


# =============================================================================
# STATE CONVERSION TESTS
# =============================================================================


class TestWebAnalysisStateConversion:
    """Tests for _convert_web_analysis_to_state_format function."""

    def test_convert_web_analysis_to_state_format(self, sample_web_result):
        """Test conversion of web analysis result to state format."""
        state_format = _convert_web_analysis_to_state_format(sample_web_result)

        # Check basic fields
        assert "analyzed_domain" in state_format
        assert state_format["analyzed_domain"] == "example-crm.com"
        assert "analysis_summary" in state_format

        # Check site structure
        assert "site_structure" in state_format
        assert state_format["site_structure"]["navigation_type"] == "mega-menu"

        # Check tech stack
        assert "tech_stack" in state_format
        assert state_format["tech_stack"]["cms"] == "webflow"
        assert state_format["tech_stack"]["frontend_framework"] == "React"

        # Check performance
        assert "performance" in state_format
        assert state_format["performance"]["overall_grade"] == "good"

        # Check content quality
        assert "content_quality" in state_format
        assert state_format["content_quality"]["overall_quality"] == "high"

        # Check UX analysis
        assert "ux_analysis" in state_format
        assert state_format["ux_analysis"]["overall_score"] == "Good"

        # Check security
        assert "security" in state_format
        assert state_format["security"]["status"] == "secure"

        # Check competitors
        assert "competitors" in state_format
        assert len(state_format["competitors"]) == 2

        # Check recommendations
        assert "quick_wins" in state_format
        assert len(state_format["quick_wins"]) == 3

        # Check timestamp
        assert "last_updated" in state_format


# =============================================================================
# TAVILY INTEGRATION TESTS
# =============================================================================


class TestWebAnalysisTavilyIntegration:
    """Tests for Tavily integration in web analysis."""

    @pytest.mark.asyncio
    @patch("backend.services.tavily_service.TavilyService.get_instance")
    async def test_conduct_web_research_parallel_queries(
        self,
        mock_tavily_instance,
        sample_web_input,
    ):
        """Test that web research conducts parallel Tavily queries."""
        mock_service = MagicMock()
        mock_tavily_instance.return_value = mock_service

        mock_response = MagicMock()
        mock_response.answer = "Test answer"
        mock_response.results = []

        mock_service.search = AsyncMock(return_value=mock_response)

        result = await _conduct_web_research_with_tavily(sample_web_input)

        # Should have called search at least 4 times for standard analysis
        # (structure, technology, performance, content/UX, competitors)
        assert mock_service.search.call_count >= 4

        assert "structure_search_results" in result
        assert "technology_search_results" in result
        assert "performance_search_results" in result
        assert "content_ux_search_results" in result

    @pytest.mark.asyncio
    @patch("backend.services.tavily_service.TavilyService.get_instance")
    async def test_technical_analysis_includes_security(
        self,
        mock_tavily_instance,
    ):
        """Test technical analysis includes security search."""
        mock_service = MagicMock()
        mock_tavily_instance.return_value = mock_service

        mock_response = MagicMock()
        mock_response.answer = "Test answer"
        mock_response.results = []

        mock_service.search = AsyncMock(return_value=mock_response)

        technical_input = WebAnalysisInput(
            primary_domain="example.com",
            analysis_type=AnalysisType.TECHNICAL,
            include_competitors=False,
        )

        result = await _conduct_web_research_with_tavily(technical_input)

        # Technical analysis should include security search
        assert mock_service.search.call_count >= 5
        assert "security_search_results" in result


# =============================================================================
# API REQUEST/RESPONSE TESTS
# =============================================================================


class TestWebAnalysisAPISchemas:
    """Tests for API request/response schemas."""

    def test_web_analysis_request_valid(self, sample_web_input):
        """Test valid web analysis request."""
        request = WebAnalysisRequest(
            input=sample_web_input,
            client_id="client-123",
            workflow_id="workflow-456",
        )
        assert request.client_id == "client-123"
        assert request.workflow_id == "workflow-456"
        assert request.input.primary_domain == sample_web_input.primary_domain

    def test_web_analysis_response_success(self, sample_web_result):
        """Test successful web analysis response."""
        response = WebAnalysisResponse(
            success=True,
            analysis_id="analysis-789",
            analysis_result=sample_web_result,
            processing_time_ms=8000,
        )
        assert response.success is True
        assert response.analysis_result is not None
        assert response.error_message is None

    def test_web_analysis_response_error(self):
        """Test error web analysis response."""
        response = WebAnalysisResponse(
            success=False,
            analysis_id="analysis-789",
            error_message="Failed to analyze website",
        )
        assert response.success is False
        assert response.analysis_result is None
        assert "Failed" in response.error_message


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestWebAnalysisEdgeCases:
    """Tests for edge cases in web analysis."""

    def test_input_with_empty_urls_list(self):
        """Test input with empty URLs but valid domain."""
        input_data = WebAnalysisInput(
            primary_domain="example.com",
            target_urls=[],
        )
        assert input_data.primary_domain == "example.com"
        assert input_data.target_urls == []

    def test_input_with_many_competitor_urls(self):
        """Test input with many competitor URLs."""
        competitors = [f"https://competitor{i}.com" for i in range(10)]
        input_data = WebAnalysisInput(
            primary_domain="example.com",
            competitor_urls=competitors,
        )
        assert len(input_data.competitor_urls) == 10

    def test_input_without_competitors(self, sample_web_input):
        """Test input with competitors disabled."""
        input_data = WebAnalysisInput(
            primary_domain=sample_web_input.primary_domain,
            include_competitors=False,
            competitor_urls=[],
        )
        assert input_data.include_competitors is False
        assert input_data.competitor_urls == []

    def test_page_info_minimal(self):
        """Test page info with minimal data."""
        page = PageInfo(url="https://example.com/page")
        assert page.url == "https://example.com/page"
        assert page.title is None
        assert page.has_cta is False

    def test_technology_stack_minimal(self):
        """Test technology stack with defaults."""
        tech = TechnologyStack()
        assert tech.cms == CMSType.UNKNOWN
        assert tech.hosting == HostingType.UNKNOWN
        assert tech.analytics_tools == []

    def test_performance_metrics_minimal(self):
        """Test performance metrics with defaults."""
        perf = PerformanceMetrics()
        assert perf.overall_grade == PerformanceGrade.NEEDS_IMPROVEMENT
        assert perf.mobile_readiness == MobileReadiness.PARTIALLY_RESPONSIVE

    def test_security_assessment_minimal(self):
        """Test security assessment with defaults."""
        security = SecurityAssessment()
        assert security.overall_status == SecurityStatus.PARTIALLY_SECURE
        assert security.https_enabled is True

    def test_web_result_with_no_competitors(self):
        """Test web result with empty competitors."""
        result = WebAnalysisResult(
            analysis_summary="Basic website analysis",
            competitor_profiles=[],
            competitive_insights=[],
        )
        assert len(result.competitor_profiles) == 0
        assert len(result.competitive_insights) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

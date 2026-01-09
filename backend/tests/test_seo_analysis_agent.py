"""
Unit Tests for SEO Analysis Agent (Phase 5.2)
=============================================
Comprehensive tests for SEO analysis operations.

Test Coverage:
- Schema validation (SEOAnalysisInput, SEOAnalysisResult, etc.)
- Node execution with mocked Tavily and LLM
- Fallback behavior on errors
- State conversion
- Various analysis depths
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from backend.app.schemas.seo_analysis import (
    # Enums
    AnalysisDepth,
    KeywordDifficulty,
    KeywordIntent,
    ContentType,
    SERPFeature,
    ContentGapPriority,
    ConfidenceLevel,
    # Input
    SEOAnalysisInput,
    SEOAnalysisRequest,
    # Keyword Analysis
    KeywordData,
    KeywordCluster,
    # SERP Analysis
    SERPResult,
    SERPAnalysis,
    # Competitor Analysis
    CompetitorSEOProfile,
    KeywordGap,
    # Content Opportunities
    ContentGap,
    ContentRecommendation,
    # Technical SEO
    TechnicalSEOInsight,
    # Output
    SEOAnalysisResult,
    SEOAnalysisResponse,
)
from backend.graph.state import (
    OrchestratorState,
    create_initial_state,
    CRMProvider,
    WorkflowStatus,
    AgentMessage,
)
from backend.graph.nodes import (
    seo_analysis_node,
    _conduct_seo_research_with_tavily,
    _analyze_seo_with_llm,
    _convert_seo_to_state_format,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_seo_input():
    """Sample SEO analysis input."""
    return SEOAnalysisInput(
        target_url="https://example-crm.com",
        target_keywords=["crm software", "sales automation", "lead management"],
        industry="B2B SaaS",
        geographic_target="United States",
        language="en",
        analysis_depth=AnalysisDepth.STANDARD,
        include_competitors=True,
        known_competitors=["salesforce.com", "hubspot.com", "pipedrive.com"],
        business_goals=["increase organic traffic", "improve keyword rankings"],
    )


@pytest.fixture
def sample_keyword_data():
    """Sample keyword data list."""
    return [
        KeywordData(
            keyword="crm software",
            search_volume="10K-100K",
            difficulty=KeywordDifficulty.HARD,
            intent=KeywordIntent.COMMERCIAL,
            cpc_estimate="$15.00",
            trend="stable",
            relevance_score=0.95,
        ),
        KeywordData(
            keyword="sales automation tools",
            search_volume="1K-10K",
            difficulty=KeywordDifficulty.MEDIUM,
            intent=KeywordIntent.COMMERCIAL,
            cpc_estimate="$12.00",
            trend="rising",
            relevance_score=0.88,
        ),
        KeywordData(
            keyword="best crm for small business",
            search_volume="1K-10K",
            difficulty=KeywordDifficulty.MEDIUM,
            intent=KeywordIntent.TRANSACTIONAL,
            cpc_estimate="$18.00",
            trend="rising",
            relevance_score=0.82,
        ),
    ]


@pytest.fixture
def sample_keyword_clusters():
    """Sample keyword clusters."""
    return [
        KeywordCluster(
            cluster_name="CRM Software",
            primary_keyword="crm software",
            related_keywords=["crm system", "crm platform", "customer relationship management"],
            total_volume="50K-100K",
            average_difficulty=KeywordDifficulty.HARD,
            recommended_content_type=ContentType.COMPARISON,
        ),
        KeywordCluster(
            cluster_name="Sales Automation",
            primary_keyword="sales automation",
            related_keywords=["sales tools", "sales software", "automated sales"],
            total_volume="10K-50K",
            average_difficulty=KeywordDifficulty.MEDIUM,
            recommended_content_type=ContentType.GUIDE,
        ),
    ]


@pytest.fixture
def sample_serp_analyses():
    """Sample SERP analyses."""
    return [
        SERPAnalysis(
            keyword="crm software",
            serp_features=[
                SERPFeature.FEATURED_SNIPPET,
                SERPFeature.PEOPLE_ALSO_ASK,
                SERPFeature.SITE_LINKS,
            ],
            top_results=[
                SERPResult(
                    position=1,
                    url="https://salesforce.com/crm",
                    title="CRM Software - Salesforce",
                    domain="salesforce.com",
                    snippet="The #1 CRM platform for growing businesses...",
                    content_type=ContentType.LANDING_PAGE,
                    word_count_estimate="800-1000",
                ),
                SERPResult(
                    position=2,
                    url="https://hubspot.com/crm",
                    title="Free CRM Software - HubSpot",
                    domain="hubspot.com",
                    snippet="HubSpot CRM has free tools for everyone...",
                    content_type=ContentType.LANDING_PAGE,
                    word_count_estimate="1000-1500",
                ),
            ],
            dominant_content_type=ContentType.LANDING_PAGE,
            average_word_count="1000-1500",
            ranking_factors_observed=["Brand authority", "Backlink profile", "Content depth"],
        ),
    ]


@pytest.fixture
def sample_competitors():
    """Sample competitor SEO profiles."""
    return [
        CompetitorSEOProfile(
            domain="salesforce.com",
            estimated_traffic="50M+/month",
            domain_authority="High",
            top_ranking_keywords=["crm", "salesforce crm", "sales software"],
            content_strengths=["Comprehensive resource center", "Strong blog", "Case studies"],
            content_weaknesses=["Content can be too enterprise-focused"],
            backlink_profile="Extremely strong, millions of referring domains",
        ),
        CompetitorSEOProfile(
            domain="hubspot.com",
            estimated_traffic="30M+/month",
            domain_authority="High",
            top_ranking_keywords=["free crm", "marketing automation", "inbound marketing"],
            content_strengths=["Educational content", "Free tools", "Academy"],
            content_weaknesses=["Some content dated"],
            backlink_profile="Very strong, diverse link sources",
        ),
    ]


@pytest.fixture
def sample_keyword_gaps():
    """Sample keyword gaps."""
    return [
        KeywordGap(
            keyword="crm for startups",
            search_volume="1K-10K",
            difficulty=KeywordDifficulty.MEDIUM,
            competitors_ranking=["hubspot.com", "pipedrive.com"],
            opportunity_score=0.85,
            recommended_action="Create dedicated startup CRM comparison page",
        ),
        KeywordGap(
            keyword="ai crm features",
            search_volume="100-1K",
            difficulty=KeywordDifficulty.EASY,
            competitors_ranking=["salesforce.com"],
            opportunity_score=0.90,
            recommended_action="Create content about AI capabilities in CRM",
        ),
    ]


@pytest.fixture
def sample_content_gaps():
    """Sample content gaps."""
    return [
        ContentGap(
            topic="CRM Implementation Guide",
            description="Comprehensive guide for CRM implementation best practices",
            target_keywords=["crm implementation", "how to implement crm"],
            recommended_content_type=ContentType.GUIDE,
            estimated_traffic_potential="5K-10K/month",
            priority=ContentGapPriority.HIGH,
            competitor_coverage="Partial - no comprehensive guides available",
            suggested_outline=[
                "Introduction to CRM Implementation",
                "Pre-Implementation Planning",
                "Data Migration Steps",
                "Team Training",
                "Post-Launch Optimization",
            ],
        ),
    ]


@pytest.fixture
def sample_content_recommendations():
    """Sample content recommendations."""
    return [
        ContentRecommendation(
            title_suggestion="The Ultimate Guide to Choosing CRM Software in 2024",
            target_keyword="best crm software",
            secondary_keywords=["crm comparison", "crm features", "crm pricing"],
            content_type=ContentType.GUIDE,
            word_count_target="3000-4000",
            key_sections=[
                "What is CRM Software?",
                "Key Features to Look For",
                "Top CRM Options Compared",
                "Pricing Guide",
                "Implementation Tips",
            ],
            internal_linking_opportunities=["crm-features", "crm-pricing", "crm-demo"],
            priority=ContentGapPriority.HIGH,
        ),
    ]


@pytest.fixture
def sample_technical_insights():
    """Sample technical SEO insights."""
    return [
        TechnicalSEOInsight(
            category="Site Speed",
            observation="Page load time is critical for CRM comparison pages",
            impact="high",
            recommendation="Optimize images and implement lazy loading",
        ),
        TechnicalSEOInsight(
            category="Mobile",
            observation="Mobile-first indexing is standard",
            impact="high",
            recommendation="Ensure all pages are mobile-responsive",
        ),
    ]


@pytest.fixture
def sample_seo_result(
    sample_keyword_data,
    sample_keyword_clusters,
    sample_serp_analyses,
    sample_competitors,
    sample_keyword_gaps,
    sample_content_gaps,
    sample_content_recommendations,
    sample_technical_insights,
):
    """Sample complete SEO analysis result."""
    return SEOAnalysisResult(
        analysis_summary="Comprehensive SEO analysis for CRM software market. "
                        "Strong competition from established players but opportunities exist "
                        "in long-tail keywords and content gaps.",
        overall_seo_score="Good - Room for Improvement",
        primary_keywords=sample_keyword_data,
        keyword_clusters=sample_keyword_clusters,
        long_tail_opportunities=[
            "crm software for real estate agents",
            "crm with email integration",
            "affordable crm for freelancers",
        ],
        serp_analyses=sample_serp_analyses,
        serp_opportunities=[
            "Featured snippet opportunity for 'what is crm' queries",
            "People Also Ask optimization potential",
        ],
        competitors=sample_competitors,
        keyword_gaps=sample_keyword_gaps,
        competitive_advantages=[
            "Can target underserved mid-market segment",
            "Opportunity in vertical-specific content",
        ],
        content_gaps=sample_content_gaps,
        content_recommendations=sample_content_recommendations,
        technical_insights=sample_technical_insights,
        quick_wins=[
            "Optimize meta titles for primary keywords",
            "Add FAQ schema to comparison pages",
            "Create 'CRM for startups' landing page",
        ],
        strategic_recommendations=[
            "Build topical authority through pillar/cluster content strategy",
            "Invest in linkable assets like CRM benchmark reports",
            "Target featured snippets with structured content",
        ],
        confidence_level=ConfidenceLevel.MEDIUM,
        data_limitations=["Search volume estimates may vary", "Competitor traffic is estimated"],
        sources_used=["Tavily web search", "SERP analysis"],
    )


@pytest.fixture
def initial_seo_analysis_state(sample_seo_input):
    """Initial state with SEO analysis input."""
    state = create_initial_state(
        client_id="client-12345",
        client_name="Test Consulting",
        crm_provider=CRMProvider.HUBSPOT,
        workflow_type="intelligence_only",
    )
    # Add SEO analysis input as a message
    state["messages"] = [
        AgentMessage(
            message_id=str(uuid4()),
            from_agent="user",
            to_agent="seo_analysis",
            message_type="seo_analysis_input",
            content="Analyze SEO for this target",
            metadata={
                "target_url": sample_seo_input.target_url,
                "target_keywords": sample_seo_input.target_keywords,
                "industry": sample_seo_input.industry,
                "geographic_target": sample_seo_input.geographic_target,
                "language": sample_seo_input.language,
                "analysis_depth": sample_seo_input.analysis_depth.value,
                "include_competitors": sample_seo_input.include_competitors,
                "known_competitors": sample_seo_input.known_competitors,
                "business_goals": sample_seo_input.business_goals,
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    ]
    return state


@pytest.fixture
def mock_tavily_search_response():
    """Mock Tavily search response."""
    mock_result = MagicMock()
    mock_result.answer = "CRM software keywords have high competition with significant search volumes."
    mock_result.results = [
        MagicMock(
            title="CRM Software Keyword Analysis 2024",
            url="https://example.com/seo-analysis",
            content="CRM keywords show strong commercial intent with average CPC of $15...",
            score=0.95,
            published_date="2024-01-15",
        ),
        MagicMock(
            title="SEO Best Practices for SaaS",
            url="https://example.com/saas-seo",
            content="SaaS companies should focus on comparison and feature-based keywords...",
            score=0.90,
            published_date="2024-01-10",
        ),
    ]
    return mock_result


# =============================================================================
# SCHEMA VALIDATION TESTS
# =============================================================================


class TestSEOAnalysisSchemas:
    """Tests for Pydantic schema validation."""

    def test_seo_analysis_input_valid(self, sample_seo_input):
        """Test valid SEO analysis input."""
        assert sample_seo_input.target_url == "https://example-crm.com"
        assert len(sample_seo_input.target_keywords) == 3
        assert sample_seo_input.analysis_depth == AnalysisDepth.STANDARD
        assert len(sample_seo_input.known_competitors) == 3

    def test_seo_analysis_input_with_url_only(self):
        """Test SEO input with URL only."""
        input_data = SEOAnalysisInput(target_url="https://example.com")
        assert input_data.target_url == "https://example.com"
        assert input_data.target_keywords == []
        assert input_data.analysis_depth == AnalysisDepth.STANDARD

    def test_seo_analysis_input_with_keywords_only(self):
        """Test SEO input with keywords only."""
        input_data = SEOAnalysisInput(target_keywords=["crm software", "sales tools"])
        assert input_data.target_url is None
        assert len(input_data.target_keywords) == 2

    def test_seo_analysis_input_no_target_fails(self):
        """Test that SEO input without URL or keywords raises error."""
        with pytest.raises(Exception, match="At least one of target_url or target_keywords"):
            SEOAnalysisInput()

    def test_keyword_data_valid(self, sample_keyword_data):
        """Test valid keyword data."""
        keyword = sample_keyword_data[0]
        assert keyword.keyword == "crm software"
        assert keyword.difficulty == KeywordDifficulty.HARD
        assert keyword.intent == KeywordIntent.COMMERCIAL
        assert keyword.relevance_score == 0.95

    def test_keyword_data_default_values(self):
        """Test keyword data with defaults."""
        keyword = KeywordData(keyword="test keyword")
        assert keyword.difficulty == KeywordDifficulty.MEDIUM
        assert keyword.intent == KeywordIntent.INFORMATIONAL
        assert keyword.relevance_score == 0.5

    def test_keyword_cluster_valid(self, sample_keyword_clusters):
        """Test valid keyword cluster."""
        cluster = sample_keyword_clusters[0]
        assert cluster.cluster_name == "CRM Software"
        assert cluster.primary_keyword == "crm software"
        assert len(cluster.related_keywords) == 3
        assert cluster.recommended_content_type == ContentType.COMPARISON

    def test_serp_result_valid(self, sample_serp_analyses):
        """Test valid SERP result."""
        serp = sample_serp_analyses[0]
        result = serp.top_results[0]
        assert result.position == 1
        assert result.domain == "salesforce.com"
        assert result.content_type == ContentType.LANDING_PAGE

    def test_serp_analysis_valid(self, sample_serp_analyses):
        """Test valid SERP analysis."""
        serp = sample_serp_analyses[0]
        assert serp.keyword == "crm software"
        assert SERPFeature.FEATURED_SNIPPET in serp.serp_features
        assert len(serp.top_results) == 2
        assert serp.dominant_content_type == ContentType.LANDING_PAGE

    def test_competitor_seo_profile_valid(self, sample_competitors):
        """Test valid competitor SEO profile."""
        competitor = sample_competitors[0]
        assert competitor.domain == "salesforce.com"
        assert competitor.domain_authority == "High"
        assert len(competitor.top_ranking_keywords) == 3

    def test_keyword_gap_valid(self, sample_keyword_gaps):
        """Test valid keyword gap."""
        gap = sample_keyword_gaps[0]
        assert gap.keyword == "crm for startups"
        assert gap.opportunity_score == 0.85
        assert len(gap.competitors_ranking) == 2

    def test_content_gap_valid(self, sample_content_gaps):
        """Test valid content gap."""
        gap = sample_content_gaps[0]
        assert gap.topic == "CRM Implementation Guide"
        assert gap.priority == ContentGapPriority.HIGH
        assert gap.recommended_content_type == ContentType.GUIDE
        assert len(gap.suggested_outline) == 5

    def test_content_recommendation_valid(self, sample_content_recommendations):
        """Test valid content recommendation."""
        rec = sample_content_recommendations[0]
        assert "Ultimate Guide" in rec.title_suggestion
        assert rec.target_keyword == "best crm software"
        assert rec.content_type == ContentType.GUIDE
        assert rec.priority == ContentGapPriority.HIGH

    def test_seo_analysis_result_valid(self, sample_seo_result):
        """Test valid SEO analysis result."""
        assert "CRM software" in sample_seo_result.analysis_summary
        assert len(sample_seo_result.primary_keywords) == 3
        assert len(sample_seo_result.keyword_clusters) == 2
        assert len(sample_seo_result.competitors) == 2
        assert sample_seo_result.confidence_level == ConfidenceLevel.MEDIUM


class TestAnalysisDepthEnum:
    """Tests for AnalysisDepth enum."""

    def test_quick_depth(self):
        """Test quick analysis depth."""
        assert AnalysisDepth.QUICK.value == "quick"

    def test_standard_depth(self):
        """Test standard analysis depth."""
        assert AnalysisDepth.STANDARD.value == "standard"

    def test_comprehensive_depth(self):
        """Test comprehensive analysis depth."""
        assert AnalysisDepth.COMPREHENSIVE.value == "comprehensive"


class TestKeywordDifficultyEnum:
    """Tests for KeywordDifficulty enum."""

    def test_all_difficulty_levels(self):
        """Test all keyword difficulty levels."""
        assert KeywordDifficulty.EASY.value == "easy"
        assert KeywordDifficulty.MEDIUM.value == "medium"
        assert KeywordDifficulty.HARD.value == "hard"
        assert KeywordDifficulty.VERY_HARD.value == "very_hard"


class TestKeywordIntentEnum:
    """Tests for KeywordIntent enum."""

    def test_all_intents(self):
        """Test all keyword intents."""
        assert KeywordIntent.INFORMATIONAL.value == "informational"
        assert KeywordIntent.NAVIGATIONAL.value == "navigational"
        assert KeywordIntent.TRANSACTIONAL.value == "transactional"
        assert KeywordIntent.COMMERCIAL.value == "commercial"


class TestContentTypeEnum:
    """Tests for ContentType enum."""

    def test_all_content_types(self):
        """Test all content types."""
        assert ContentType.BLOG_POST.value == "blog_post"
        assert ContentType.LANDING_PAGE.value == "landing_page"
        assert ContentType.GUIDE.value == "guide"
        assert ContentType.COMPARISON.value == "comparison"
        assert ContentType.FAQ.value == "faq"


class TestSERPFeatureEnum:
    """Tests for SERPFeature enum."""

    def test_key_serp_features(self):
        """Test key SERP features."""
        assert SERPFeature.FEATURED_SNIPPET.value == "featured_snippet"
        assert SERPFeature.PEOPLE_ALSO_ASK.value == "people_also_ask"
        assert SERPFeature.LOCAL_PACK.value == "local_pack"
        assert SERPFeature.KNOWLEDGE_PANEL.value == "knowledge_panel"


# =============================================================================
# NODE EXECUTION TESTS
# =============================================================================


class TestSEOAnalysisNode:
    """Tests for seo_analysis_node function."""

    def test_node_without_input_returns_error(self):
        """Test node returns error when no input is provided."""
        state = create_initial_state(
            client_id="client-123",
            client_name="Test Client",
            crm_provider=CRMProvider.HUBSPOT,
            workflow_type="intelligence_only",
        )
        state["messages"] = []

        result = seo_analysis_node(state)

        assert "agent_execution_log" in result
        assert result["agent_execution_log"][0]["action"] == "error"
        assert "messages" in result
        assert result["messages"][0]["message_type"] == "error"
        assert "No SEO analysis input" in result["messages"][0]["content"]

    @patch("backend.graph.nodes._conduct_seo_research_with_tavily")
    @patch("backend.graph.nodes._analyze_seo_with_llm")
    def test_node_successful_execution(
        self,
        mock_llm_analyze,
        mock_tavily_research,
        initial_seo_analysis_state,
        sample_seo_result,
    ):
        """Test successful SEO analysis node execution."""
        # Setup mocks
        mock_tavily_research.return_value = {
            "keyword_search_results": "Test keyword data",
            "serp_search_results": "Test SERP data",
            "competitor_search_results": "Test competitor data",
            "content_gap_results": "Test content gap data",
            "technical_seo_results": None,
        }
        mock_llm_analyze.return_value = sample_seo_result

        # Create async mock
        async def async_tavily_mock(*args, **kwargs):
            return mock_tavily_research.return_value

        async def async_llm_mock(*args, **kwargs):
            return mock_llm_analyze.return_value

        mock_tavily_research.side_effect = async_tavily_mock
        mock_llm_analyze.side_effect = async_llm_mock

        result = seo_analysis_node(initial_seo_analysis_state)

        assert "seo_data" in result
        assert "agent_execution_log" in result
        assert result["agent_execution_log"][0]["action"] == "analyze_seo"
        assert "messages" in result
        assert result["messages"][0]["message_type"] == "info"

    @patch("backend.graph.nodes._conduct_seo_research_with_tavily")
    def test_node_handles_tavily_error(
        self,
        mock_tavily_research,
        initial_seo_analysis_state,
    ):
        """Test node handles Tavily errors gracefully."""
        from backend.services.tavily_service import TavilyError

        async def raise_tavily_error(*args, **kwargs):
            raise TavilyError("API rate limit exceeded")

        mock_tavily_research.side_effect = raise_tavily_error

        result = seo_analysis_node(initial_seo_analysis_state)

        assert "agent_execution_log" in result
        assert result["agent_execution_log"][0]["action"] == "error"
        assert "error_message" in result
        assert "Tavily" in result["error_message"]

    @patch("backend.graph.nodes._conduct_seo_research_with_tavily")
    @patch("backend.graph.nodes._analyze_seo_with_llm")
    def test_node_handles_llm_error(
        self,
        mock_llm_analyze,
        mock_tavily_research,
        initial_seo_analysis_state,
    ):
        """Test node handles LLM errors gracefully."""
        from backend.services.llm_service import LLMError

        async def async_tavily_mock(*args, **kwargs):
            return {"keyword_search_results": "data"}

        async def raise_llm_error(*args, **kwargs):
            raise LLMError("Model overloaded")

        mock_tavily_research.side_effect = async_tavily_mock
        mock_llm_analyze.side_effect = raise_llm_error

        result = seo_analysis_node(initial_seo_analysis_state)

        assert "agent_execution_log" in result
        assert result["agent_execution_log"][0]["action"] == "error"
        assert "error_message" in result
        assert "LLM" in result["error_message"]


# =============================================================================
# STATE CONVERSION TESTS
# =============================================================================


class TestSEOStateConversion:
    """Tests for _convert_seo_to_state_format function."""

    def test_convert_seo_to_state_format(self, sample_seo_result):
        """Test conversion of SEO result to state format."""
        state_format = _convert_seo_to_state_format(sample_seo_result)

        # Check keywords
        assert "keywords" in state_format
        assert len(state_format["keywords"]) == 3
        assert state_format["keywords"][0]["keyword"] == "crm software"
        assert state_format["keywords"][0]["difficulty"] == "hard"

        # Check competitors
        assert "competitors" in state_format
        assert len(state_format["competitors"]) == 2
        assert state_format["competitors"][0]["domain"] == "salesforce.com"

        # Check content gaps
        assert "content_gaps" in state_format
        assert len(state_format["content_gaps"]) == 1

        # Check SERP analysis
        assert "serp_analysis" in state_format
        assert len(state_format["serp_analysis"]) == 1

        # Check keyword clusters
        assert "keyword_clusters" in state_format
        assert len(state_format["keyword_clusters"]) == 2

        # Check keyword gaps
        assert "keyword_gaps" in state_format
        assert len(state_format["keyword_gaps"]) == 2

        # Check content recommendations
        assert "content_recommendations" in state_format
        assert len(state_format["content_recommendations"]) == 1

        # Check quick wins
        assert "quick_wins" in state_format
        assert len(state_format["quick_wins"]) == 3

        # Check timestamp
        assert "last_updated" in state_format


# =============================================================================
# TAVILY INTEGRATION TESTS
# =============================================================================


class TestSEOTavilyIntegration:
    """Tests for Tavily integration in SEO analysis."""

    @pytest.mark.asyncio
    @patch("backend.services.tavily_service.TavilyService.get_instance")
    async def test_conduct_seo_research_parallel_queries(
        self,
        mock_tavily_instance,
        sample_seo_input,
    ):
        """Test that SEO research conducts parallel Tavily queries."""
        mock_service = MagicMock()
        mock_tavily_instance.return_value = mock_service

        mock_response = MagicMock()
        mock_response.answer = "Test answer"
        mock_response.results = []

        mock_service.search = AsyncMock(return_value=mock_response)

        result = await _conduct_seo_research_with_tavily(sample_seo_input)

        # Should have called search at least 4 times for standard depth
        # (keywords, SERP, competitors, content gaps)
        assert mock_service.search.call_count >= 4

        assert "keyword_search_results" in result
        assert "serp_search_results" in result
        assert "competitor_search_results" in result
        assert "content_gap_results" in result

    @pytest.mark.asyncio
    @patch("backend.services.tavily_service.TavilyService.get_instance")
    async def test_comprehensive_depth_includes_technical_seo(
        self,
        mock_tavily_instance,
    ):
        """Test comprehensive depth includes technical SEO search."""
        mock_service = MagicMock()
        mock_tavily_instance.return_value = mock_service

        mock_response = MagicMock()
        mock_response.answer = "Test answer"
        mock_response.results = []

        mock_service.search = AsyncMock(return_value=mock_response)

        comprehensive_input = SEOAnalysisInput(
            target_keywords=["test keyword"],
            analysis_depth=AnalysisDepth.COMPREHENSIVE,
        )

        result = await _conduct_seo_research_with_tavily(comprehensive_input)

        # Comprehensive should have 5 searches (including technical SEO)
        assert mock_service.search.call_count == 5
        assert "technical_seo_results" in result


# =============================================================================
# API REQUEST/RESPONSE TESTS
# =============================================================================


class TestSEOAPISchemas:
    """Tests for API request/response schemas."""

    def test_seo_analysis_request_valid(self, sample_seo_input):
        """Test valid SEO analysis request."""
        request = SEOAnalysisRequest(
            input=sample_seo_input,
            client_id="client-123",
            workflow_id="workflow-456",
        )
        assert request.client_id == "client-123"
        assert request.workflow_id == "workflow-456"
        assert request.input.target_url == sample_seo_input.target_url

    def test_seo_analysis_response_success(self, sample_seo_result):
        """Test successful SEO analysis response."""
        response = SEOAnalysisResponse(
            success=True,
            analysis_id="analysis-789",
            analysis_result=sample_seo_result,
            processing_time_ms=5000,
        )
        assert response.success is True
        assert response.analysis_result is not None
        assert response.error_message is None

    def test_seo_analysis_response_error(self):
        """Test error SEO analysis response."""
        response = SEOAnalysisResponse(
            success=False,
            analysis_id="analysis-789",
            error_message="API rate limit exceeded",
        )
        assert response.success is False
        assert response.analysis_result is None
        assert "rate limit" in response.error_message


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestSEOEdgeCases:
    """Tests for edge cases in SEO analysis."""

    def test_input_with_empty_keywords_list(self):
        """Test input with empty keywords but valid URL."""
        input_data = SEOAnalysisInput(
            target_url="https://example.com",
            target_keywords=[],
        )
        assert input_data.target_url == "https://example.com"
        assert input_data.target_keywords == []

    def test_input_with_many_keywords(self):
        """Test input with many keywords."""
        keywords = [f"keyword{i}" for i in range(20)]
        input_data = SEOAnalysisInput(target_keywords=keywords)
        assert len(input_data.target_keywords) == 20

    def test_keyword_data_relevance_score_bounds(self):
        """Test keyword data relevance score bounds."""
        # Valid range
        keyword = KeywordData(keyword="test", relevance_score=0.0)
        assert keyword.relevance_score == 0.0

        keyword = KeywordData(keyword="test", relevance_score=1.0)
        assert keyword.relevance_score == 1.0

        # Out of bounds should raise
        with pytest.raises(ValueError):
            KeywordData(keyword="test", relevance_score=1.5)

        with pytest.raises(ValueError):
            KeywordData(keyword="test", relevance_score=-0.1)

    def test_keyword_gap_opportunity_score_bounds(self):
        """Test keyword gap opportunity score bounds."""
        gap = KeywordGap(keyword="test", opportunity_score=0.5)
        assert gap.opportunity_score == 0.5

        with pytest.raises(ValueError):
            KeywordGap(keyword="test", opportunity_score=1.5)

    def test_serp_result_position_bounds(self):
        """Test SERP result position constraints."""
        result = SERPResult(
            position=1,
            url="https://example.com",
            title="Test",
            domain="example.com",
        )
        assert result.position == 1

    def test_empty_competitors_list(self, sample_seo_input):
        """Test input with no known competitors."""
        input_data = SEOAnalysisInput(
            target_keywords=sample_seo_input.target_keywords,
            known_competitors=[],
            include_competitors=True,
        )
        assert input_data.include_competitors is True
        assert input_data.known_competitors == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

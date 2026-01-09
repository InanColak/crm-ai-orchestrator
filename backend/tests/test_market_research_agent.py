"""
Unit Tests for Market Research Agent (Phase 5.1)
=================================================
Comprehensive tests for market research operations.

Test Coverage:
- Schema validation (MarketResearchInput, MarketResearchResult, etc.)
- Node execution with mocked Tavily and LLM
- Fallback behavior on errors
- State conversion
- Various research scopes
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from backend.app.schemas.market_research import (
    # Enums
    ResearchScope,
    MarketMaturity,
    TrendSentiment,
    OpportunityPriority,
    ConfidenceLevel,
    # Input
    MarketResearchInput,
    MarketResearchRequest,
    # Market Analysis Components
    MarketOverview,
    MarketTrend,
    CompetitorAnalysis,
    TargetSegment,
    MarketOpportunity,
    MarketThreat,
    NewsInsight,
    # Output
    MarketResearchResult,
    MarketResearchResponse,
)
from backend.graph.state import (
    OrchestratorState,
    create_initial_state,
    CRMProvider,
    WorkflowStatus,
    AgentMessage,
)
from backend.graph.nodes import (
    market_research_node,
    _conduct_market_research_with_tavily,
    _analyze_market_with_llm,
    _convert_research_to_state_format,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_market_input():
    """Sample market research input."""
    return MarketResearchInput(
        industry="B2B SaaS",
        sub_segment="Sales Automation",
        geographic_focus="North America",
        research_scope=ResearchScope.STANDARD,
        focus_areas=["AI adoption", "pricing models"],
        known_competitors=["Salesforce", "HubSpot", "Outreach"],
        time_horizon="12 months",
        client_context="Mid-market software company exploring new market entry",
        additional_questions=["What is the average deal size?"],
    )


@pytest.fixture
def sample_market_overview():
    """Sample market overview."""
    return MarketOverview(
        market_name="B2B Sales Automation",
        description="Software solutions that automate sales processes and workflows",
        market_size="$15B globally",
        growth_rate="12% CAGR",
        maturity=MarketMaturity.GROWING,
        key_drivers=["AI adoption", "Remote work", "Digital transformation"],
        key_challenges=["Integration complexity", "Data quality", "Change management"],
    )


@pytest.fixture
def sample_trends():
    """Sample market trends."""
    return [
        MarketTrend(
            title="AI-Powered Sales Intelligence",
            description="AI tools for lead scoring, forecasting, and personalization are becoming standard",
            category="Technology",
            sentiment=TrendSentiment.BULLISH,
            impact_level="high",
            time_horizon="6-12 months",
            evidence=["Gartner report", "Vendor announcements"],
        ),
        MarketTrend(
            title="Revenue Operations Consolidation",
            description="Companies are combining sales, marketing, and CS ops under unified RevOps",
            category="Operational",
            sentiment=TrendSentiment.BULLISH,
            impact_level="medium",
            time_horizon="12-24 months",
            evidence=["Industry surveys", "Job posting trends"],
        ),
    ]


@pytest.fixture
def sample_competitors():
    """Sample competitor analysis."""
    return [
        CompetitorAnalysis(
            name="Salesforce Sales Cloud",
            description="Leading CRM with comprehensive sales automation",
            website="https://salesforce.com",
            market_position="leader",
            strengths=["Brand recognition", "Ecosystem", "Enterprise features"],
            weaknesses=["Complexity", "Cost", "Implementation time"],
            recent_developments=["Einstein AI enhancements", "Slack integration"],
            estimated_market_share="~20%",
        ),
        CompetitorAnalysis(
            name="HubSpot Sales Hub",
            description="All-in-one sales platform for growing businesses",
            website="https://hubspot.com",
            market_position="challenger",
            strengths=["Ease of use", "Free tier", "Marketing integration"],
            weaknesses=["Enterprise scalability", "Advanced reporting"],
            recent_developments=["AI writing assistant", "Prospecting tools"],
            estimated_market_share="~10%",
        ),
    ]


@pytest.fixture
def sample_opportunities():
    """Sample market opportunities."""
    return [
        MarketOpportunity(
            title="Mid-market AI automation gap",
            description="Mid-market companies lack affordable AI-powered sales tools",
            opportunity_type="market gap",
            priority=OpportunityPriority.HIGH,
            target_segment="Mid-market (50-500 employees)",
            estimated_value="$2B addressable market",
            time_to_capture="6-12 months",
            requirements=["AI capabilities", "Competitive pricing", "Easy onboarding"],
            risks=["Enterprise vendors moving downmarket", "Economic uncertainty"],
        ),
    ]


@pytest.fixture
def sample_threats():
    """Sample market threats."""
    return [
        MarketThreat(
            title="AI commoditization",
            description="AI features becoming table stakes, reducing differentiation",
            threat_type="technological",
            severity="medium",
            likelihood="high",
            mitigation_strategies=["Focus on unique data advantages", "Vertical specialization"],
        ),
    ]


@pytest.fixture
def sample_research_result(
    sample_market_overview,
    sample_trends,
    sample_competitors,
    sample_opportunities,
    sample_threats,
):
    """Sample complete market research result."""
    return MarketResearchResult(
        market_overview=sample_market_overview,
        trends=sample_trends,
        competitors=sample_competitors,
        competitive_dynamics="Highly competitive market with consolidation trend",
        target_segments=[
            TargetSegment(
                segment_name="Mid-market Tech Companies",
                description="Software and tech companies with 50-500 employees",
                size_estimate="$5B",
                growth_potential="high",
                key_characteristics=["Growth-focused", "Tech-savvy", "Agile"],
                pain_points=["Scaling sales process", "Data silos", "Lead quality"],
                buying_criteria=["Ease of use", "Integration", "ROI"],
                ideal_customer_profile="SaaS company, 100-300 employees, $10-50M ARR",
            ),
        ],
        opportunities=sample_opportunities,
        threats=sample_threats,
        recent_news=[
            NewsInsight(
                title="AI Sales Tools Market to Reach $30B by 2030",
                url="https://example.com/news/ai-sales",
                snippet="Market research firm projects strong growth",
                published_date="2024-01-10",
                relevance="high",
                key_insight="Market growth validates opportunity",
            ),
        ],
        strategic_recommendations=[
            "Focus on mid-market segment with AI-native solution",
            "Build strong integration ecosystem",
            "Emphasize time-to-value in positioning",
        ],
        key_questions_answered=[
            {"question": "What is the average deal size?", "answer": "$15K-50K for mid-market"}
        ],
        research_summary="B2B Sales Automation is a $15B growing market with AI driving innovation. "
                        "Mid-market segment shows strong opportunity with limited competition.",
        confidence_level=ConfidenceLevel.MEDIUM,
        data_limitations=["Limited access to proprietary market data", "Rapid market changes"],
        sources_used=["https://example.com/source1", "https://example.com/source2"],
    )


@pytest.fixture
def initial_market_research_state(sample_market_input):
    """Initial state with market research input."""
    state = create_initial_state(
        client_id="client-12345",
        client_name="Test Consulting",
        crm_provider=CRMProvider.HUBSPOT,
        workflow_type="intelligence_only",
    )
    # Add market research input as a message
    state["messages"] = [
        AgentMessage(
            message_id=str(uuid4()),
            from_agent="user",
            to_agent="market_research",
            message_type="market_research_input",
            content="Research this market",
            metadata={
                "industry": sample_market_input.industry,
                "sub_segment": sample_market_input.sub_segment,
                "geographic_focus": sample_market_input.geographic_focus,
                "research_scope": sample_market_input.research_scope.value,
                "focus_areas": sample_market_input.focus_areas,
                "known_competitors": sample_market_input.known_competitors,
                "time_horizon": sample_market_input.time_horizon,
                "client_context": sample_market_input.client_context,
                "additional_questions": sample_market_input.additional_questions,
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    ]
    return state


@pytest.fixture
def mock_tavily_search_response():
    """Mock Tavily search response."""
    mock_result = MagicMock()
    mock_result.answer = "The B2B SaaS market is growing rapidly with AI adoption driving innovation."
    mock_result.results = [
        MagicMock(
            title="B2B SaaS Market Analysis 2024",
            url="https://example.com/market-analysis",
            content="The B2B SaaS market continues to grow at 12% CAGR...",
            score=0.95,
            published_date="2024-01-15",
        ),
        MagicMock(
            title="Sales Automation Trends",
            url="https://example.com/trends",
            content="AI-powered sales tools are transforming the industry...",
            score=0.90,
            published_date="2024-01-10",
        ),
    ]
    return mock_result


@pytest.fixture
def mock_tavily_news_response():
    """Mock Tavily news response."""
    mock_result = MagicMock()
    mock_result.results = [
        MagicMock(
            title="Salesforce Announces New AI Features",
            url="https://example.com/news/salesforce",
            content="Salesforce today announced new AI capabilities...",
            score=0.88,
            published_date="2024-01-12",
        ),
    ]
    return mock_result


# =============================================================================
# SCHEMA VALIDATION TESTS
# =============================================================================


class TestMarketResearchSchemas:
    """Tests for Pydantic schema validation."""

    def test_market_research_input_valid(self, sample_market_input):
        """Test valid market research input."""
        assert sample_market_input.industry == "B2B SaaS"
        assert sample_market_input.sub_segment == "Sales Automation"
        assert sample_market_input.research_scope == ResearchScope.STANDARD
        assert len(sample_market_input.known_competitors) == 3

    def test_market_research_input_minimal(self):
        """Test minimal market research input."""
        input_data = MarketResearchInput(industry="Healthcare")
        assert input_data.industry == "Healthcare"
        assert input_data.sub_segment is None
        assert input_data.research_scope == ResearchScope.STANDARD

    def test_market_research_input_empty_industry_fails(self):
        """Test that empty industry raises validation error."""
        with pytest.raises(ValueError, match="industry cannot be empty"):
            MarketResearchInput(industry="")

    def test_market_research_input_whitespace_industry_fails(self):
        """Test that whitespace-only industry raises validation error."""
        with pytest.raises(ValueError, match="industry cannot be empty"):
            MarketResearchInput(industry="   ")

    def test_market_overview_valid(self, sample_market_overview):
        """Test valid market overview."""
        assert sample_market_overview.market_name == "B2B Sales Automation"
        assert sample_market_overview.maturity == MarketMaturity.GROWING
        assert len(sample_market_overview.key_drivers) == 3

    def test_market_trend_valid(self, sample_trends):
        """Test valid market trends."""
        trend = sample_trends[0]
        assert trend.title == "AI-Powered Sales Intelligence"
        assert trend.sentiment == TrendSentiment.BULLISH
        assert trend.impact_level == "high"

    def test_competitor_analysis_valid(self, sample_competitors):
        """Test valid competitor analysis."""
        competitor = sample_competitors[0]
        assert competitor.name == "Salesforce Sales Cloud"
        assert competitor.market_position == "leader"
        assert len(competitor.strengths) == 3

    def test_market_opportunity_valid(self, sample_opportunities):
        """Test valid market opportunity."""
        opp = sample_opportunities[0]
        assert opp.title == "Mid-market AI automation gap"
        assert opp.priority == OpportunityPriority.HIGH
        assert len(opp.requirements) == 3

    def test_market_research_result_valid(self, sample_research_result):
        """Test valid market research result."""
        assert sample_research_result.market_overview.market_name == "B2B Sales Automation"
        assert len(sample_research_result.trends) == 2
        assert len(sample_research_result.competitors) == 2
        assert len(sample_research_result.opportunities) == 1
        assert sample_research_result.confidence_level == ConfidenceLevel.MEDIUM


class TestResearchScopeEnum:
    """Tests for ResearchScope enum."""

    def test_narrow_scope(self):
        """Test narrow research scope."""
        assert ResearchScope.NARROW.value == "narrow"

    def test_standard_scope(self):
        """Test standard research scope."""
        assert ResearchScope.STANDARD.value == "standard"

    def test_comprehensive_scope(self):
        """Test comprehensive research scope."""
        assert ResearchScope.COMPREHENSIVE.value == "comprehensive"


class TestMarketMaturityEnum:
    """Tests for MarketMaturity enum."""

    def test_all_maturity_stages(self):
        """Test all market maturity stages."""
        assert MarketMaturity.EMERGING.value == "emerging"
        assert MarketMaturity.GROWING.value == "growing"
        assert MarketMaturity.MATURE.value == "mature"
        assert MarketMaturity.DECLINING.value == "declining"


class TestTrendSentimentEnum:
    """Tests for TrendSentiment enum."""

    def test_all_sentiments(self):
        """Test all trend sentiments."""
        assert TrendSentiment.BULLISH.value == "bullish"
        assert TrendSentiment.NEUTRAL.value == "neutral"
        assert TrendSentiment.BEARISH.value == "bearish"


# =============================================================================
# NODE EXECUTION TESTS
# =============================================================================


class TestMarketResearchNode:
    """Tests for market_research_node function."""

    def test_node_without_input_returns_error(self):
        """Test node returns error when no input is provided."""
        state = create_initial_state(
            client_id="client-123",
            client_name="Test Client",
            crm_provider=CRMProvider.HUBSPOT,
            workflow_type="intelligence_only",
        )
        state["messages"] = []

        result = market_research_node(state)

        assert "agent_execution_log" in result
        assert result["agent_execution_log"][0]["action"] == "error"
        assert "messages" in result
        assert result["messages"][0]["message_type"] == "error"

    @patch("backend.graph.nodes._conduct_market_research_with_tavily")
    @patch("backend.graph.nodes._analyze_market_with_llm")
    def test_node_successful_execution(
        self,
        mock_llm_analyze,
        mock_tavily_research,
        initial_market_research_state,
        sample_research_result,
    ):
        """Test successful market research node execution."""
        # Setup mocks
        mock_tavily_research.return_value = {
            "industry_search_results": "Test industry data",
            "trends_search_results": "Test trends data",
            "competitor_search_results": "Test competitor data",
            "news_results": [],
            "market_size_results": None,
        }
        mock_llm_analyze.return_value = sample_research_result

        # Run with asyncio mock
        with patch("asyncio.run") as mock_run:
            mock_run.side_effect = [
                mock_tavily_research.return_value,
                mock_llm_analyze.return_value,
            ]

            result = market_research_node(initial_market_research_state)

        # Verify result structure
        assert "market_research" in result
        assert "agent_execution_log" in result
        assert "messages" in result
        assert result["agent_execution_log"][0]["action"] == "analyze_market"

    def test_node_with_invalid_input_handles_gracefully(self):
        """Test node handles invalid input gracefully."""
        state = create_initial_state(
            client_id="client-123",
            client_name="Test Client",
            crm_provider=CRMProvider.HUBSPOT,
            workflow_type="intelligence_only",
        )
        # Add message with invalid metadata
        state["messages"] = [
            AgentMessage(
                message_id=str(uuid4()),
                from_agent="user",
                to_agent="market_research",
                message_type="market_research_input",
                content="Invalid research",
                metadata={"industry": ""},  # Invalid - empty industry
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        ]

        result = market_research_node(state)

        # Should handle gracefully
        assert "agent_execution_log" in result
        assert "messages" in result


# =============================================================================
# TAVILY RESEARCH TESTS
# =============================================================================


class TestTavilyResearch:
    """Tests for Tavily web research functionality."""

    @pytest.mark.asyncio
    async def test_conduct_research_standard_scope(
        self,
        sample_market_input,
        mock_tavily_search_response,
        mock_tavily_news_response,
    ):
        """Test Tavily research with standard scope."""
        with patch("backend.graph.nodes.TavilyService") as MockTavily:
            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=mock_tavily_search_response)
            mock_service.search_news = AsyncMock(return_value=mock_tavily_news_response)
            MockTavily.get_instance.return_value = mock_service

            result = await _conduct_market_research_with_tavily(sample_market_input)

            # Verify searches were called
            assert mock_service.search.call_count >= 3  # industry, trends, competitors
            assert mock_service.search_news.call_count == 1

            # Verify result structure
            assert "industry_search_results" in result
            assert "trends_search_results" in result
            assert "competitor_search_results" in result
            assert "news_results" in result

    @pytest.mark.asyncio
    async def test_conduct_research_comprehensive_scope(
        self,
        mock_tavily_search_response,
        mock_tavily_news_response,
    ):
        """Test Tavily research with comprehensive scope."""
        comprehensive_input = MarketResearchInput(
            industry="FinTech",
            research_scope=ResearchScope.COMPREHENSIVE,
        )

        with patch("backend.graph.nodes.TavilyService") as MockTavily:
            mock_service = MagicMock()
            mock_service.search = AsyncMock(return_value=mock_tavily_search_response)
            mock_service.search_news = AsyncMock(return_value=mock_tavily_news_response)
            MockTavily.get_instance.return_value = mock_service

            result = await _conduct_market_research_with_tavily(comprehensive_input)

            # Comprehensive scope should include market size search
            assert mock_service.search.call_count >= 4  # +1 for market size

    @pytest.mark.asyncio
    async def test_conduct_research_handles_search_errors(self, sample_market_input):
        """Test that research handles search errors gracefully."""
        with patch("backend.graph.nodes.TavilyService") as MockTavily:
            mock_service = MagicMock()
            # First search succeeds, others fail
            mock_result = MagicMock()
            mock_result.answer = "Test answer"
            mock_result.results = []

            mock_service.search = AsyncMock(
                side_effect=[mock_result, Exception("Search failed"), mock_result, mock_result]
            )
            mock_service.search_news = AsyncMock(side_effect=Exception("News search failed"))
            MockTavily.get_instance.return_value = mock_service

            result = await _conduct_market_research_with_tavily(sample_market_input)

            # Should still return partial results
            assert "industry_search_results" in result


# =============================================================================
# STATE CONVERSION TESTS
# =============================================================================


class TestStateConversion:
    """Tests for state conversion functions."""

    def test_convert_research_to_state_format(self, sample_research_result):
        """Test converting research result to state format."""
        state_data = _convert_research_to_state_format(sample_research_result)

        # Verify structure matches MarketResearch TypedDict
        assert "industry_trends" in state_data
        assert "market_size" in state_data
        assert "key_players" in state_data
        assert "opportunities" in state_data
        assert "threats" in state_data
        assert "news_articles" in state_data

        # Verify content
        assert len(state_data["industry_trends"]) == 2
        assert len(state_data["key_players"]) == 2
        assert state_data["market_size"]["estimate"] == "$15B globally"

    def test_convert_empty_research_result(self, sample_market_overview):
        """Test converting minimal research result."""
        minimal_result = MarketResearchResult(
            market_overview=sample_market_overview,
            research_summary="Minimal research",
        )

        state_data = _convert_research_to_state_format(minimal_result)

        assert state_data["industry_trends"] == []
        assert state_data["key_players"] == []
        assert state_data["opportunities"] == []
        assert state_data["threats"] == []


# =============================================================================
# INPUT PARSING TESTS
# =============================================================================


class TestInputParsing:
    """Tests for input parsing from state messages."""

    def test_parse_standard_input(self, initial_market_research_state):
        """Test parsing standard market research input from state."""
        messages = initial_market_research_state.get("messages", [])
        input_message = None

        for msg in messages:
            if msg.get("message_type") == "market_research_input":
                input_message = msg
                break

        assert input_message is not None
        metadata = input_message.get("metadata", {})
        assert metadata["industry"] == "B2B SaaS"
        assert metadata["research_scope"] == "standard"

    def test_parse_minimal_input(self):
        """Test parsing minimal input."""
        state = create_initial_state(
            client_id="client-123",
            client_name="Test",
            crm_provider=CRMProvider.HUBSPOT,
        )
        state["messages"] = [
            AgentMessage(
                message_id=str(uuid4()),
                from_agent="user",
                to_agent="market_research",
                message_type="market_research_input",
                content="Research",
                metadata={"industry": "Healthcare"},
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        ]

        # Verify input can be constructed from minimal metadata
        metadata = state["messages"][0]["metadata"]
        input_data = MarketResearchInput(
            industry=metadata.get("industry", ""),
            research_scope=ResearchScope(metadata.get("research_scope", "standard")),
        )

        assert input_data.industry == "Healthcare"
        assert input_data.research_scope == ResearchScope.STANDARD


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_tavily_error_handling(self, initial_market_research_state):
        """Test handling of Tavily errors."""
        from backend.services.tavily_service import TavilyError

        with patch("backend.graph.nodes._conduct_market_research_with_tavily") as mock_tavily:
            mock_tavily.side_effect = TavilyError("API rate limit exceeded")

            with patch("asyncio.run", side_effect=TavilyError("API rate limit exceeded")):
                result = market_research_node(initial_market_research_state)

            assert "error_message" in result
            assert "Tavily" in result["error_message"]

    def test_llm_error_handling(self, initial_market_research_state):
        """Test handling of LLM errors."""
        from backend.services.llm_service import LLMError, LLMProvider

        with patch("asyncio.run") as mock_run:
            # First call (Tavily) succeeds, second (LLM) fails
            mock_run.side_effect = [
                {"industry_search_results": "test"},
                LLMError("Model overloaded", provider=LLMProvider.ANTHROPIC),
            ]

            result = market_research_node(initial_market_research_state)

            assert "error_message" in result
            assert "LLM" in result["error_message"]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestMarketResearchIntegration:
    """Integration tests for market research workflow."""

    def test_full_research_flow_mock(
        self,
        initial_market_research_state,
        sample_research_result,
    ):
        """Test full research flow with mocked services."""
        research_data = {
            "industry_search_results": "B2B SaaS market data...",
            "trends_search_results": "AI adoption trends...",
            "competitor_search_results": "Salesforce, HubSpot analysis...",
            "news_results": [{"title": "Test news", "url": "https://example.com"}],
            "market_size_results": None,
        }

        with patch("asyncio.run") as mock_run:
            mock_run.side_effect = [research_data, sample_research_result]

            result = market_research_node(initial_market_research_state)

        # Verify complete result
        assert "market_research" in result
        assert "agent_execution_log" in result
        assert result["agent_execution_log"][0]["action"] == "analyze_market"

        # Verify market research data
        market_data = result["market_research"]
        assert "industry_trends" in market_data
        assert "key_players" in market_data
        assert "opportunities" in market_data

    def test_state_update_immutability(self, initial_market_research_state):
        """Test that node returns update dict without mutating original state."""
        original_messages = initial_market_research_state["messages"].copy()

        with patch("asyncio.run") as mock_run:
            mock_run.side_effect = [
                {"industry_search_results": "test"},
                Exception("Intentional error"),
            ]

            result = market_research_node(initial_market_research_state)

        # Original state should be unchanged
        assert initial_market_research_state["messages"] == original_messages


# =============================================================================
# EXECUTION LOG TESTS
# =============================================================================


class TestExecutionLogging:
    """Tests for execution logging."""

    def test_success_log_contains_metrics(
        self,
        initial_market_research_state,
        sample_research_result,
    ):
        """Test that success log contains relevant metrics."""
        with patch("asyncio.run") as mock_run:
            mock_run.side_effect = [
                {"industry_search_results": "test"},
                sample_research_result,
            ]

            result = market_research_node(initial_market_research_state)

        log = result["agent_execution_log"][0]
        assert log["action"] == "analyze_market"
        assert "details" in log
        assert "industry" in log["details"]
        assert "trends_count" in log["details"]
        assert "competitors_count" in log["details"]

    def test_error_log_contains_error_info(self, initial_market_research_state):
        """Test that error log contains error information."""
        with patch("asyncio.run") as mock_run:
            mock_run.side_effect = Exception("Test error")

            result = market_research_node(initial_market_research_state)

        log = result["agent_execution_log"][0]
        assert log["action"] == "error"
        assert "error" in log["details"]


# =============================================================================
# MESSAGE OUTPUT TESTS
# =============================================================================


class TestMessageOutput:
    """Tests for agent message output."""

    def test_success_message_format(
        self,
        initial_market_research_state,
        sample_research_result,
    ):
        """Test success message format and content."""
        with patch("asyncio.run") as mock_run:
            mock_run.side_effect = [
                {"industry_search_results": "test"},
                sample_research_result,
            ]

            result = market_research_node(initial_market_research_state)

        message = result["messages"][0]
        assert message["from_agent"] == "market_research"
        assert message["message_type"] == "info"
        assert "B2B SaaS" in message["content"]
        assert "trends" in message["content"]
        assert "competitors" in message["content"]

    def test_error_message_format(self, initial_market_research_state):
        """Test error message format."""
        with patch("asyncio.run") as mock_run:
            mock_run.side_effect = Exception("Test error")

            result = market_research_node(initial_market_research_state)

        message = result["messages"][0]
        assert message["from_agent"] == "market_research"
        assert message["message_type"] == "error"
        assert "error" in message["content"].lower()

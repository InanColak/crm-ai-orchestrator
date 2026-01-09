"""
Unit Tests for Audience Builder Agent (Phase 5.4)
=================================================
Comprehensive tests for audience building and persona creation operations.

Test Coverage:
- Schema validation (AudienceBuildingInput, AudienceBuildingResult, etc.)
- Node execution with mocked Tavily and LLM
- Intelligence context gathering from other agents
- Fallback behavior on errors
- State conversion
- Various persona configurations
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from backend.app.schemas.audience_building import (
    # Enums
    PersonaType,
    CompanySize,
    BuyingStage,
    ChannelType,
    PriorityLevel,
    ConfidenceLevel,
    # Input
    AudienceBuildingInput,
    AudienceBuildingRequest,
    # ICP
    ICPFirmographics,
    ICPTechnographics,
    ICPBehavioral,
    ICPProfile,
    # Personas
    PersonaDemographics,
    PersonaPsychographics,
    PersonaBehavior,
    BuyerPersona,
    # Pain Points
    PainPointAnalysis,
    # Journey
    JourneyTouchpoint,
    JourneyStage,
    # Messaging
    PersonaMessage,
    # Channels
    ChannelStrategy,
    # Output
    AudienceBuildingResult,
    AudienceBuildingResponse,
)
from backend.graph.state import (
    OrchestratorState,
    create_initial_state,
    CRMProvider,
    WorkflowStatus,
    AgentMessage,
)
from backend.graph.nodes import (
    audience_builder_node,
    _gather_intelligence_context,
    _conduct_audience_research_with_tavily,
    _analyze_audience_with_llm,
    _convert_audience_to_state_format,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_audience_input():
    """Sample audience building input."""
    return AudienceBuildingInput(
        product_category="CRM Software",
        value_proposition="All-in-one CRM platform for growing B2B companies",
        target_market="B2B SaaS",
        geographic_focus="North America",
        target_company_sizes=[CompanySize.SMALL, CompanySize.MEDIUM, CompanySize.MID_MARKET],
        target_industries=["Technology", "Professional Services", "Financial Services"],
        existing_customer_traits=["Growth-focused", "Digital-first", "Sales-driven"],
        known_pain_points=["Fragmented customer data", "Manual sales processes"],
        use_intelligence_data=True,
        num_personas=3,
        additional_context="Focus on companies transitioning from spreadsheets to CRM",
    )


@pytest.fixture
def sample_icp_firmographics():
    """Sample ICP firmographics."""
    return ICPFirmographics(
        company_size=[CompanySize.SMALL, CompanySize.MEDIUM],
        employee_count_range="20-200",
        revenue_range="$2M-$50M",
        industries=["Technology", "Professional Services"],
        geographic_regions=["North America", "Europe"],
        business_model="B2B SaaS",
    )


@pytest.fixture
def sample_icp_technographics():
    """Sample ICP technographics."""
    return ICPTechnographics(
        current_tools=["Google Workspace", "Slack", "QuickBooks"],
        tech_maturity="Growth Stage",
        integration_needs=["Email", "Calendar", "Accounting"],
    )


@pytest.fixture
def sample_icp_behavioral():
    """Sample ICP behavioral."""
    return ICPBehavioral(
        buying_triggers=["Rapid team growth", "Lost deals due to poor follow-up"],
        decision_timeline="2-4 months",
        budget_characteristics="$500-$5000/month",
        evaluation_criteria=["Ease of use", "Integration capabilities", "Price"],
    )


@pytest.fixture
def sample_icp(sample_icp_firmographics, sample_icp_technographics, sample_icp_behavioral):
    """Sample ICP profile."""
    return ICPProfile(
        summary="Growing B2B companies in technology and professional services with 20-200 employees, "
                "looking to modernize their sales processes and improve customer relationships.",
        firmographics=sample_icp_firmographics,
        technographics=sample_icp_technographics,
        behavioral=sample_icp_behavioral,
        key_pain_points=["Data silos", "Manual processes", "Lack of visibility"],
        desired_outcomes=["360-degree customer view", "Automated workflows", "Better forecasting"],
        disqualification_criteria=["Enterprise with complex requirements", "Very small budget"],
    )


@pytest.fixture
def sample_persona_demographics():
    """Sample persona demographics."""
    return PersonaDemographics(
        job_titles=["VP of Sales", "Head of Sales", "Sales Director"],
        seniority_level="Director/VP",
        department="Sales",
        age_range="35-50",
        education="Bachelor's degree, often MBA",
        experience_years="10-20 years",
    )


@pytest.fixture
def sample_persona_psychographics():
    """Sample persona psychographics."""
    return PersonaPsychographics(
        goals=["Hit revenue targets", "Build high-performing team", "Improve sales efficiency"],
        challenges=["Forecasting accuracy", "Rep productivity", "Pipeline visibility"],
        motivations=["Recognition", "Team success", "Career advancement"],
        fears=["Missing targets", "Losing top performers", "Being blindsided by churn"],
        values=["Data-driven decisions", "Team development", "Results orientation"],
    )


@pytest.fixture
def sample_persona_behavior():
    """Sample persona behavior."""
    return PersonaBehavior(
        information_sources=["LinkedIn", "Sales conferences", "Peer recommendations"],
        preferred_content_types=["Case studies", "ROI calculators", "Demo videos"],
        social_platforms=["LinkedIn", "Twitter"],
        buying_role=PersonaType.DECISION_MAKER,
        decision_influence="Final decision authority for tools under $50k",
    )


@pytest.fixture
def sample_buyer_persona(
    sample_persona_demographics,
    sample_persona_psychographics,
    sample_persona_behavior,
):
    """Sample buyer persona."""
    return BuyerPersona(
        persona_name="Sales Leader Sam",
        persona_type=PersonaType.DECISION_MAKER,
        one_liner="Revenue-focused sales executive looking to scale team performance",
        demographics=sample_persona_demographics,
        psychographics=sample_persona_psychographics,
        behavior=sample_persona_behavior,
        pain_points=[
            "Can't accurately forecast pipeline",
            "Reps spend too much time on admin",
            "Deals slip through the cracks",
        ],
        objections=[
            "My team won't adopt another tool",
            "Our current process works fine",
            "Too expensive for what we need",
        ],
        key_messages=[
            "Increase forecast accuracy by 30%",
            "Give reps 5+ hours back per week",
            "Never lose a deal to poor follow-up",
        ],
        quotes=[
            "I need to see the ROI before I commit",
            "If my team won't use it, it's worthless",
        ],
    )


@pytest.fixture
def sample_pain_point_analysis():
    """Sample pain point analysis."""
    return [
        PainPointAnalysis(
            pain_point="Fragmented Customer Data",
            description="Customer information scattered across multiple systems and spreadsheets",
            affected_personas=["Sales Leader Sam", "Marketing Manager Maya"],
            severity=PriorityLevel.CRITICAL,
            frequency="Daily",
            current_solutions=["Manual data entry", "Spreadsheet syncing"],
            our_solution="Unified customer database with automatic sync",
        ),
        PainPointAnalysis(
            pain_point="Manual Sales Processes",
            description="Reps spend too much time on administrative tasks instead of selling",
            affected_personas=["Sales Leader Sam"],
            severity=PriorityLevel.HIGH,
            frequency="Daily",
            current_solutions=["Basic email templates", "Reminders"],
            our_solution="Automated workflows and task management",
        ),
    ]


@pytest.fixture
def sample_journey_stage():
    """Sample journey stage."""
    return JourneyStage(
        stage=BuyingStage.CONSIDERATION,
        description="Evaluating different CRM solutions and comparing features",
        buyer_goals=["Understand capabilities", "Compare vendors", "Build business case"],
        buyer_questions=[
            "How does this compare to competitors?",
            "What integrations are available?",
            "What's the implementation timeline?",
        ],
        buyer_emotions=["Overwhelmed", "Cautiously optimistic", "Time-pressured"],
        touchpoints=[
            JourneyTouchpoint(
                channel=ChannelType.CONTENT_MARKETING,
                content_type="Comparison guide",
                purpose="Help compare options",
                key_message="All the features you need at a fraction of the cost",
            ),
        ],
        content_needs=["Comparison guides", "Feature matrices", "Case studies"],
        success_metrics=["Time on comparison page", "Guide downloads", "Demo requests"],
    )


@pytest.fixture
def sample_persona_message():
    """Sample persona message."""
    return PersonaMessage(
        persona_name="Sales Leader Sam",
        value_proposition="Scale your sales team without scaling your overhead",
        key_benefits=["30% better forecast accuracy", "5+ hours saved per rep weekly"],
        proof_points=["500+ B2B companies trust us", "G2 Leader"],
        call_to_action="See how similar companies grew 40%",
        tone="Confident and results-oriented",
        words_to_use=["scale", "efficiency", "pipeline", "revenue"],
        words_to_avoid=["cheap", "simple", "basic"],
    )


@pytest.fixture
def sample_channel_strategy():
    """Sample channel strategy."""
    return ChannelStrategy(
        channel=ChannelType.SOCIAL_LINKEDIN,
        priority=PriorityLevel.HIGH,
        target_personas=["Sales Leader Sam", "Marketing Manager Maya"],
        journey_stages=[BuyingStage.AWARENESS, BuyingStage.CONSIDERATION],
        content_types=["Thought leadership", "Case studies", "Industry insights"],
        estimated_effectiveness="High for B2B decision makers",
        key_tactics=["Sponsored content", "Employee advocacy", "InMail campaigns"],
        success_metrics=["Engagement rate", "Lead form fills", "Website visits"],
    )


@pytest.fixture
def sample_audience_result(
    sample_icp,
    sample_buyer_persona,
    sample_pain_point_analysis,
    sample_journey_stage,
    sample_persona_message,
    sample_channel_strategy,
):
    """Sample complete audience building result."""
    return AudienceBuildingResult(
        analysis_summary="Target audience consists of growing B2B companies seeking to modernize "
                        "sales processes. Key decision makers are sales leaders and marketing managers "
                        "frustrated with fragmented data and manual processes.",
        ideal_customer_profile=sample_icp,
        buyer_personas=[sample_buyer_persona],
        pain_point_analysis=sample_pain_point_analysis,
        buying_journey=[sample_journey_stage],
        messaging_matrix=[sample_persona_message],
        channel_strategy=[sample_channel_strategy],
        quick_wins=[
            "Create comparison guide targeting sales leaders",
            "Develop LinkedIn content strategy",
            "Build ROI calculator landing page",
        ],
        strategic_recommendations=[
            "Develop customer advocacy program",
            "Create industry-specific messaging",
            "Build partner ecosystem for integrations",
        ],
        data_sources_used=["market_research", "seo_data", "tavily_search"],
        confidence_level=ConfidenceLevel.MEDIUM,
        data_limitations=["Limited access to competitor pricing", "Sample size for industry data"],
    )


@pytest.fixture
def initial_audience_state(sample_audience_input):
    """Initial state with audience building input."""
    state = create_initial_state(
        client_id="client-12345",
        client_name="Test Consulting",
        crm_provider=CRMProvider.HUBSPOT,
        workflow_type="intelligence_only",
    )
    # Add audience building input as a message
    state["messages"] = [
        AgentMessage(
            message_id=str(uuid4()),
            from_agent="user",
            to_agent="audience_builder",
            message_type="audience_building_input",
            content="Build audience profiles",
            metadata={
                "product_category": sample_audience_input.product_category,
                "value_proposition": sample_audience_input.value_proposition,
                "target_market": sample_audience_input.target_market,
                "geographic_focus": sample_audience_input.geographic_focus,
                "target_company_sizes": [cs.value for cs in sample_audience_input.target_company_sizes],
                "target_industries": sample_audience_input.target_industries,
                "existing_customer_traits": sample_audience_input.existing_customer_traits,
                "known_pain_points": sample_audience_input.known_pain_points,
                "use_intelligence_data": sample_audience_input.use_intelligence_data,
                "num_personas": sample_audience_input.num_personas,
                "additional_context": sample_audience_input.additional_context,
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    ]
    return state


@pytest.fixture
def state_with_intelligence_data(initial_audience_state):
    """State with intelligence data from other agents."""
    state = initial_audience_state.copy()
    state["market_research"] = {
        "industry": "CRM Software",
        "market_overview": "Growing B2B CRM market valued at $50B",
        "trends": [
            {"name": "AI-powered automation"},
            {"name": "Mobile-first solutions"},
        ],
        "competitors": [{"name": "Salesforce"}, {"name": "HubSpot"}],
        "target_segments": [{"name": "SMBs"}, {"name": "Mid-market"}],
        "opportunities": [{"name": "Integration marketplace"}],
        "threats": [{"name": "Enterprise consolidation"}],
    }
    state["seo_data"] = {
        "primary_domain": "example-crm.com",
        "keyword_clusters": [
            {"cluster_name": "crm software"},
            {"cluster_name": "sales automation"},
        ],
        "competitor_profiles": [],
        "content_gaps": ["comparison guides", "ROI calculators"],
        "content_recommendations": ["How-to guides", "Case studies"],
        "search_intent_insights": "High commercial intent for CRM comparisons",
    }
    state["web_analysis"] = {
        "analyzed_domain": "example-crm.com",
        "analysis_summary": "Modern website with good UX but lacking educational content",
        "tech_stack": {"cms": "webflow"},
        "content_quality": {"overall_quality": "good"},
        "ux_analysis": {"overall_score": "8/10"},
        "competitors": [{"domain": "competitor.com"}],
        "competitive_insights": [],
    }
    return state


@pytest.fixture
def mock_tavily_search_response():
    """Mock Tavily search response."""
    mock_result = MagicMock()
    mock_result.answer = "B2B CRM buyers typically include sales leaders, marketing managers, and operations executives."
    mock_result.results = [
        MagicMock(
            title="B2B Buyer Persona Guide",
            url="https://example.com/buyer-personas",
            content="Decision makers in B2B purchases typically include VP of Sales...",
            score=0.95,
            published_date="2024-01-15",
        ),
        MagicMock(
            title="CRM Buying Journey",
            url="https://example.com/crm-journey",
            content="The average B2B CRM buying journey takes 2-4 months...",
            score=0.90,
            published_date="2024-01-10",
        ),
    ]
    return mock_result


# =============================================================================
# SCHEMA VALIDATION TESTS
# =============================================================================


class TestAudienceBuildingSchemas:
    """Tests for Pydantic schema validation."""

    def test_audience_input_valid(self, sample_audience_input):
        """Test valid audience building input."""
        assert sample_audience_input.product_category == "CRM Software"
        assert sample_audience_input.target_market == "B2B SaaS"
        assert len(sample_audience_input.target_company_sizes) == 3
        assert len(sample_audience_input.target_industries) == 3

    def test_audience_input_minimal(self):
        """Test minimal audience building input."""
        input_data = AudienceBuildingInput(product_category="Marketing Automation")
        assert input_data.product_category == "Marketing Automation"
        assert input_data.target_market is None
        assert input_data.num_personas == 3  # default

    def test_audience_input_empty_category_fails(self):
        """Test that empty product category raises error."""
        with pytest.raises(Exception, match="product_category cannot be empty"):
            AudienceBuildingInput(product_category="")

    def test_audience_input_whitespace_category_fails(self):
        """Test that whitespace-only product category raises error."""
        with pytest.raises(Exception, match="product_category cannot be empty"):
            AudienceBuildingInput(product_category="   ")

    def test_audience_input_persona_count_validation(self):
        """Test persona count validation."""
        # Valid range
        input_1 = AudienceBuildingInput(product_category="CRM", num_personas=1)
        assert input_1.num_personas == 1

        input_7 = AudienceBuildingInput(product_category="CRM", num_personas=7)
        assert input_7.num_personas == 7

        # Invalid: too many
        with pytest.raises(Exception):
            AudienceBuildingInput(product_category="CRM", num_personas=10)

        # Invalid: zero
        with pytest.raises(Exception):
            AudienceBuildingInput(product_category="CRM", num_personas=0)

    def test_icp_profile_valid(self, sample_icp):
        """Test valid ICP profile."""
        assert "Growing B2B companies" in sample_icp.summary
        assert len(sample_icp.firmographics.industries) == 2
        assert len(sample_icp.technographics.current_tools) == 3
        assert len(sample_icp.behavioral.buying_triggers) == 2

    def test_buyer_persona_valid(self, sample_buyer_persona):
        """Test valid buyer persona."""
        assert sample_buyer_persona.persona_name == "Sales Leader Sam"
        assert sample_buyer_persona.persona_type == PersonaType.DECISION_MAKER
        assert len(sample_buyer_persona.pain_points) == 3
        assert len(sample_buyer_persona.key_messages) == 3

    def test_pain_point_analysis_valid(self, sample_pain_point_analysis):
        """Test valid pain point analysis."""
        pain_point = sample_pain_point_analysis[0]
        assert pain_point.pain_point == "Fragmented Customer Data"
        assert pain_point.severity == PriorityLevel.CRITICAL
        assert len(pain_point.affected_personas) == 2

    def test_journey_stage_valid(self, sample_journey_stage):
        """Test valid journey stage."""
        assert sample_journey_stage.stage == BuyingStage.CONSIDERATION
        assert len(sample_journey_stage.buyer_goals) == 3
        assert len(sample_journey_stage.touchpoints) == 1

    def test_journey_touchpoint_valid(self, sample_journey_stage):
        """Test valid journey touchpoint."""
        touchpoint = sample_journey_stage.touchpoints[0]
        assert touchpoint.channel == ChannelType.CONTENT_MARKETING
        assert touchpoint.content_type == "Comparison guide"

    def test_persona_message_valid(self, sample_persona_message):
        """Test valid persona message."""
        assert sample_persona_message.persona_name == "Sales Leader Sam"
        assert len(sample_persona_message.key_benefits) == 2
        assert len(sample_persona_message.words_to_use) == 4
        assert len(sample_persona_message.words_to_avoid) == 3

    def test_channel_strategy_valid(self, sample_channel_strategy):
        """Test valid channel strategy."""
        assert sample_channel_strategy.channel == ChannelType.SOCIAL_LINKEDIN
        assert sample_channel_strategy.priority == PriorityLevel.HIGH
        assert len(sample_channel_strategy.journey_stages) == 2

    def test_audience_result_valid(self, sample_audience_result):
        """Test valid audience building result."""
        assert len(sample_audience_result.buyer_personas) == 1
        assert len(sample_audience_result.pain_point_analysis) == 2
        assert len(sample_audience_result.buying_journey) == 1
        assert sample_audience_result.confidence_level == ConfidenceLevel.MEDIUM


class TestPersonaTypeEnum:
    """Tests for PersonaType enum."""

    def test_decision_maker_type(self):
        """Test decision maker type."""
        assert PersonaType.DECISION_MAKER.value == "decision_maker"

    def test_influencer_type(self):
        """Test influencer type."""
        assert PersonaType.INFLUENCER.value == "influencer"

    def test_user_type(self):
        """Test user type."""
        assert PersonaType.USER.value == "user"

    def test_gatekeeper_type(self):
        """Test gatekeeper type."""
        assert PersonaType.GATEKEEPER.value == "gatekeeper"

    def test_champion_type(self):
        """Test champion type."""
        assert PersonaType.CHAMPION.value == "champion"


class TestCompanySizeEnum:
    """Tests for CompanySize enum."""

    def test_all_company_sizes(self):
        """Test all company sizes."""
        assert CompanySize.STARTUP.value == "startup"
        assert CompanySize.SMALL.value == "small"
        assert CompanySize.MEDIUM.value == "medium"
        assert CompanySize.MID_MARKET.value == "mid_market"
        assert CompanySize.ENTERPRISE.value == "enterprise"


class TestBuyingStageEnum:
    """Tests for BuyingStage enum."""

    def test_all_buying_stages(self):
        """Test all buying stages."""
        assert BuyingStage.AWARENESS.value == "awareness"
        assert BuyingStage.CONSIDERATION.value == "consideration"
        assert BuyingStage.DECISION.value == "decision"
        assert BuyingStage.PURCHASE.value == "purchase"
        assert BuyingStage.ONBOARDING.value == "onboarding"
        assert BuyingStage.ADVOCACY.value == "advocacy"


class TestChannelTypeEnum:
    """Tests for ChannelType enum."""

    def test_common_channel_types(self):
        """Test common channel types."""
        assert ChannelType.ORGANIC_SEARCH.value == "organic_search"
        assert ChannelType.SOCIAL_LINKEDIN.value == "social_linkedin"
        assert ChannelType.EMAIL.value == "email"
        assert ChannelType.CONTENT_MARKETING.value == "content_marketing"
        assert ChannelType.WEBINARS.value == "webinars"
        assert ChannelType.DIRECT_SALES.value == "direct_sales"


class TestPriorityLevelEnum:
    """Tests for PriorityLevel enum."""

    def test_all_priority_levels(self):
        """Test all priority levels."""
        assert PriorityLevel.CRITICAL.value == "critical"
        assert PriorityLevel.HIGH.value == "high"
        assert PriorityLevel.MEDIUM.value == "medium"
        assert PriorityLevel.LOW.value == "low"


# =============================================================================
# NODE EXECUTION TESTS
# =============================================================================


class TestAudienceBuilderNode:
    """Tests for audience_builder_node function."""

    def test_node_without_input_returns_error(self):
        """Test node returns error when no input is provided."""
        state = create_initial_state(
            client_id="client-123",
            client_name="Test Client",
            crm_provider=CRMProvider.HUBSPOT,
            workflow_type="intelligence_only",
        )
        state["messages"] = []

        result = audience_builder_node(state)

        assert "agent_execution_log" in result
        assert result["agent_execution_log"][0]["action"] == "error"
        assert "messages" in result
        assert result["messages"][0]["message_type"] == "error"
        assert "No audience building input" in result["messages"][0]["content"]

    @patch("backend.graph.nodes._conduct_audience_research_with_tavily")
    @patch("backend.graph.nodes._analyze_audience_with_llm")
    def test_node_successful_execution(
        self,
        mock_llm_analyze,
        mock_tavily_research,
        initial_audience_state,
        sample_audience_result,
    ):
        """Test successful audience builder node execution."""
        # Setup mocks
        mock_tavily_research.return_value = {
            "persona_search_results": "Test persona data",
            "pain_points_search_results": "Test pain points data",
            "journey_search_results": "Test journey data",
            "icp_search_results": "Test ICP data",
            "channel_search_results": "Test channel data",
            "industry_search_results": None,
        }
        mock_llm_analyze.return_value = sample_audience_result

        # Create async mock
        async def async_tavily_mock(*args, **kwargs):
            return mock_tavily_research.return_value

        async def async_llm_mock(*args, **kwargs):
            return mock_llm_analyze.return_value

        mock_tavily_research.side_effect = async_tavily_mock
        mock_llm_analyze.side_effect = async_llm_mock

        result = audience_builder_node(initial_audience_state)

        assert "audience_data" in result
        assert "agent_execution_log" in result
        assert result["agent_execution_log"][0]["action"] == "build_audience"
        assert "messages" in result
        assert result["messages"][0]["message_type"] == "info"

    @patch("backend.graph.nodes._conduct_audience_research_with_tavily")
    @patch("backend.graph.nodes._analyze_audience_with_llm")
    def test_node_with_intelligence_data(
        self,
        mock_llm_analyze,
        mock_tavily_research,
        state_with_intelligence_data,
        sample_audience_result,
    ):
        """Test node uses intelligence data from other agents."""
        mock_tavily_research.return_value = {
            "persona_search_results": "Test data",
            "pain_points_search_results": "Test data",
            "journey_search_results": "Test data",
            "icp_search_results": "Test data",
            "channel_search_results": "Test data",
            "industry_search_results": None,
        }
        mock_llm_analyze.return_value = sample_audience_result

        async def async_tavily_mock(*args, **kwargs):
            return mock_tavily_research.return_value

        async def async_llm_mock(*args, **kwargs):
            return mock_llm_analyze.return_value

        mock_tavily_research.side_effect = async_tavily_mock
        mock_llm_analyze.side_effect = async_llm_mock

        result = audience_builder_node(state_with_intelligence_data)

        assert "audience_data" in result
        # Verify intelligence sources are logged
        log_details = result["agent_execution_log"][0]["details"]
        assert "intelligence_sources" in log_details
        assert "market_research" in log_details["intelligence_sources"]

    @patch("backend.graph.nodes._conduct_audience_research_with_tavily")
    def test_node_handles_tavily_error(
        self,
        mock_tavily_research,
        initial_audience_state,
    ):
        """Test node handles Tavily errors gracefully."""
        from backend.services.tavily_service import TavilyError

        async def raise_tavily_error(*args, **kwargs):
            raise TavilyError("API rate limit exceeded")

        mock_tavily_research.side_effect = raise_tavily_error

        result = audience_builder_node(initial_audience_state)

        assert "agent_execution_log" in result
        assert result["agent_execution_log"][0]["action"] == "error"
        assert "error_message" in result
        assert "Tavily" in result["error_message"]

    @patch("backend.graph.nodes._conduct_audience_research_with_tavily")
    @patch("backend.graph.nodes._analyze_audience_with_llm")
    def test_node_handles_llm_error(
        self,
        mock_llm_analyze,
        mock_tavily_research,
        initial_audience_state,
    ):
        """Test node handles LLM errors gracefully."""
        from backend.services.llm_service import LLMError

        async def async_tavily_mock(*args, **kwargs):
            return {"persona_search_results": "data"}

        async def raise_llm_error(*args, **kwargs):
            raise LLMError("Model overloaded")

        mock_tavily_research.side_effect = async_tavily_mock
        mock_llm_analyze.side_effect = raise_llm_error

        result = audience_builder_node(initial_audience_state)

        assert "agent_execution_log" in result
        assert result["agent_execution_log"][0]["action"] == "error"
        assert "error_message" in result
        assert "LLM" in result["error_message"]


# =============================================================================
# INTELLIGENCE CONTEXT TESTS
# =============================================================================


class TestIntelligenceContextGathering:
    """Tests for _gather_intelligence_context function."""

    def test_gather_with_all_sources(self, state_with_intelligence_data):
        """Test gathering with all intelligence sources available."""
        context = _gather_intelligence_context(state_with_intelligence_data)

        assert "market_research" in context
        assert "seo_data" in context
        assert "web_analysis" in context
        assert len(context["sources_available"]) == 3
        assert "market_research" in context["sources_available"]
        assert "seo_data" in context["sources_available"]
        assert "web_analysis" in context["sources_available"]

    def test_gather_with_no_sources(self):
        """Test gathering with no intelligence sources available."""
        state = create_initial_state(
            client_id="client-123",
            client_name="Test Client",
            crm_provider=CRMProvider.HUBSPOT,
            workflow_type="intelligence_only",
        )

        context = _gather_intelligence_context(state)

        assert context["market_research"] is None
        assert context["seo_data"] is None
        assert context["web_analysis"] is None
        assert len(context["sources_available"]) == 0

    def test_gather_with_partial_sources(self, initial_audience_state):
        """Test gathering with partial intelligence sources."""
        state = initial_audience_state.copy()
        state["market_research"] = {
            "industry": "CRM Software",
            "trends": [{"name": "AI automation"}],
        }
        # No seo_data or web_analysis

        context = _gather_intelligence_context(state)

        assert context["market_research"] is not None
        assert context["market_research"]["industry"] == "CRM Software"
        assert context["seo_data"] is None
        assert context["web_analysis"] is None
        assert len(context["sources_available"]) == 1


# =============================================================================
# STATE CONVERSION TESTS
# =============================================================================


class TestAudienceStateConversion:
    """Tests for _convert_audience_to_state_format function."""

    def test_convert_audience_to_state_format(self, sample_audience_result):
        """Test conversion of audience result to state format."""
        state_format = _convert_audience_to_state_format(sample_audience_result)

        # Check basic fields
        assert "analysis_summary" in state_format
        assert "Target audience" in state_format["analysis_summary"]

        # Check ICP
        assert "ideal_customer_profile" in state_format
        icp = state_format["ideal_customer_profile"]
        assert "summary" in icp
        assert "firmographics" in icp
        assert "technographics" in icp
        assert "behavioral" in icp

        # Check personas
        assert "personas" in state_format
        assert len(state_format["personas"]) == 1
        persona = state_format["personas"][0]
        assert persona["persona_name"] == "Sales Leader Sam"
        assert persona["persona_type"] == "decision_maker"
        assert "demographics" in persona
        assert "psychographics" in persona

        # Check pain points
        assert "pain_point_analysis" in state_format
        assert len(state_format["pain_point_analysis"]) == 2

        # Check buying journey
        assert "buying_journey" in state_format
        assert len(state_format["buying_journey"]) == 1
        stage = state_format["buying_journey"][0]
        assert stage["stage"] == "consideration"

        # Check messaging matrix
        assert "messaging_matrix" in state_format
        assert len(state_format["messaging_matrix"]) == 1

        # Check channel strategy
        assert "channel_strategy" in state_format
        assert len(state_format["channel_strategy"]) == 1
        channel = state_format["channel_strategy"][0]
        assert channel["channel"] == "social_linkedin"

        # Check recommendations
        assert "quick_wins" in state_format
        assert len(state_format["quick_wins"]) == 3

        # Check metadata
        assert "confidence_level" in state_format
        assert state_format["confidence_level"] == "medium"
        assert "last_updated" in state_format


# =============================================================================
# TAVILY INTEGRATION TESTS
# =============================================================================


class TestAudienceTavilyIntegration:
    """Tests for Tavily integration in audience building."""

    @pytest.mark.asyncio
    @patch("backend.services.tavily_service.TavilyService.get_instance")
    async def test_conduct_audience_research_parallel_queries(
        self,
        mock_tavily_instance,
        sample_audience_input,
    ):
        """Test that audience research conducts parallel Tavily queries."""
        mock_service = MagicMock()
        mock_tavily_instance.return_value = mock_service

        mock_response = MagicMock()
        mock_response.answer = "Test answer"
        mock_response.results = []

        mock_service.search = AsyncMock(return_value=mock_response)

        intelligence_context = {
            "market_research": None,
            "seo_data": None,
            "web_analysis": None,
            "sources_available": [],
        }

        result = await _conduct_audience_research_with_tavily(
            sample_audience_input, intelligence_context
        )

        # Should have called search at least 5 times
        # (persona, pain points, journey, ICP, channels + potentially industry)
        assert mock_service.search.call_count >= 5

        assert "persona_search_results" in result
        assert "pain_points_search_results" in result
        assert "journey_search_results" in result
        assert "icp_search_results" in result
        assert "channel_search_results" in result

    @pytest.mark.asyncio
    @patch("backend.services.tavily_service.TavilyService.get_instance")
    async def test_industry_specific_search_when_industries_provided(
        self,
        mock_tavily_instance,
        sample_audience_input,
    ):
        """Test that industry-specific search is conducted when industries are provided."""
        mock_service = MagicMock()
        mock_tavily_instance.return_value = mock_service

        mock_response = MagicMock()
        mock_response.answer = "Test answer"
        mock_response.results = []

        mock_service.search = AsyncMock(return_value=mock_response)

        intelligence_context = {"sources_available": []}

        result = await _conduct_audience_research_with_tavily(
            sample_audience_input, intelligence_context
        )

        # Should include industry-specific search since target_industries is provided
        # Total: 6 searches (5 standard + 1 industry-specific)
        assert mock_service.search.call_count == 6
        assert "industry_search_results" in result

    @pytest.mark.asyncio
    @patch("backend.services.tavily_service.TavilyService.get_instance")
    async def test_no_industry_search_when_no_industries(
        self,
        mock_tavily_instance,
    ):
        """Test that industry-specific search is skipped when no industries provided."""
        mock_service = MagicMock()
        mock_tavily_instance.return_value = mock_service

        mock_response = MagicMock()
        mock_response.answer = "Test answer"
        mock_response.results = []

        mock_service.search = AsyncMock(return_value=mock_response)

        input_data = AudienceBuildingInput(
            product_category="CRM Software",
            target_industries=[],  # Empty industries
        )
        intelligence_context = {"sources_available": []}

        result = await _conduct_audience_research_with_tavily(
            input_data, intelligence_context
        )

        # Should only have 5 searches (no industry-specific)
        assert mock_service.search.call_count == 5
        assert result.get("industry_search_results") is None


# =============================================================================
# API SCHEMA TESTS
# =============================================================================


class TestAudienceBuildingAPISchemas:
    """Tests for API request/response schemas."""

    def test_audience_building_request_valid(self, sample_audience_input):
        """Test valid audience building request."""
        request = AudienceBuildingRequest(
            input=sample_audience_input,
            client_id="client-123",
            workflow_id="workflow-456",
        )
        assert request.client_id == "client-123"
        assert request.workflow_id == "workflow-456"
        assert request.input.product_category == "CRM Software"

    def test_audience_building_response_success(self, sample_audience_result):
        """Test successful audience building response."""
        response = AudienceBuildingResponse(
            success=True,
            analysis_id="analysis-789",
            analysis_result=sample_audience_result,
            processing_time_ms=5000,
        )
        assert response.success is True
        assert response.analysis_id == "analysis-789"
        assert response.analysis_result is not None
        assert response.error_message is None

    def test_audience_building_response_error(self):
        """Test error audience building response."""
        response = AudienceBuildingResponse(
            success=False,
            analysis_id="analysis-789",
            error_message="Failed to analyze audience",
        )
        assert response.success is False
        assert response.analysis_result is None
        assert response.error_message == "Failed to analyze audience"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestAudienceBuildingEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_persona_creation(self):
        """Test creating just one persona."""
        input_data = AudienceBuildingInput(
            product_category="Project Management",
            num_personas=1,
        )
        assert input_data.num_personas == 1

    def test_maximum_personas_creation(self):
        """Test creating maximum number of personas."""
        input_data = AudienceBuildingInput(
            product_category="Project Management",
            num_personas=7,
        )
        assert input_data.num_personas == 7

    def test_empty_optional_fields(self):
        """Test with all optional fields empty."""
        input_data = AudienceBuildingInput(
            product_category="CRM",
        )
        assert input_data.target_market is None
        assert input_data.geographic_focus is None
        assert len(input_data.target_company_sizes) == 0
        assert len(input_data.target_industries) == 0

    def test_persona_with_minimal_data(self):
        """Test buyer persona with minimal data."""
        persona = BuyerPersona(
            persona_name="Minimal Mike",
            one_liner="A minimal persona",
        )
        assert persona.persona_name == "Minimal Mike"
        assert persona.persona_type == PersonaType.DECISION_MAKER  # default
        assert len(persona.pain_points) == 0
        assert len(persona.key_messages) == 0

    def test_icp_with_defaults(self):
        """Test ICP profile with default values."""
        icp = ICPProfile(
            summary="Test summary",
        )
        assert icp.summary == "Test summary"
        assert len(icp.key_pain_points) == 0
        assert len(icp.desired_outcomes) == 0

    def test_journey_stage_with_no_touchpoints(self):
        """Test journey stage without touchpoints."""
        stage = JourneyStage(
            stage=BuyingStage.AWARENESS,
            description="Early awareness stage",
        )
        assert stage.stage == BuyingStage.AWARENESS
        assert len(stage.touchpoints) == 0
        assert len(stage.content_needs) == 0

    def test_channel_strategy_minimal(self):
        """Test channel strategy with minimal data."""
        strategy = ChannelStrategy(
            channel=ChannelType.EMAIL,
        )
        assert strategy.channel == ChannelType.EMAIL
        assert strategy.priority == PriorityLevel.MEDIUM  # default
        assert len(strategy.target_personas) == 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestAudienceBuildingIntegration:
    """Integration tests for the complete audience building flow."""

    @patch("backend.graph.nodes._conduct_audience_research_with_tavily")
    @patch("backend.graph.nodes._analyze_audience_with_llm")
    def test_full_workflow_execution(
        self,
        mock_llm_analyze,
        mock_tavily_research,
        state_with_intelligence_data,
        sample_audience_result,
    ):
        """Test complete workflow from input to output."""
        mock_tavily_research.return_value = {
            "persona_search_results": "Research data",
            "pain_points_search_results": "Research data",
            "journey_search_results": "Research data",
            "icp_search_results": "Research data",
            "channel_search_results": "Research data",
            "industry_search_results": "Research data",
        }
        mock_llm_analyze.return_value = sample_audience_result

        async def async_tavily_mock(*args, **kwargs):
            return mock_tavily_research.return_value

        async def async_llm_mock(*args, **kwargs):
            return mock_llm_analyze.return_value

        mock_tavily_research.side_effect = async_tavily_mock
        mock_llm_analyze.side_effect = async_llm_mock

        result = audience_builder_node(state_with_intelligence_data)

        # Verify complete output structure
        assert "audience_data" in result
        audience_data = result["audience_data"]

        # Verify all major sections are present
        assert "analysis_summary" in audience_data
        assert "ideal_customer_profile" in audience_data
        assert "personas" in audience_data
        assert "pain_point_analysis" in audience_data
        assert "buying_journey" in audience_data
        assert "messaging_matrix" in audience_data
        assert "channel_strategy" in audience_data
        assert "quick_wins" in audience_data
        assert "strategic_recommendations" in audience_data

        # Verify execution log
        assert result["agent_execution_log"][0]["agent"] == "audience_builder"
        assert result["agent_execution_log"][0]["action"] == "build_audience"

        # Verify message contains expected information
        message_content = result["messages"][0]["content"]
        assert "personas created" in message_content
        assert "pain points identified" in message_content

    @patch("backend.graph.nodes._conduct_audience_research_with_tavily")
    @patch("backend.graph.nodes._analyze_audience_with_llm")
    def test_workflow_without_intelligence_data(
        self,
        mock_llm_analyze,
        mock_tavily_research,
        initial_audience_state,
        sample_audience_result,
    ):
        """Test workflow works without pre-existing intelligence data."""
        # Disable intelligence data usage
        for msg in initial_audience_state["messages"]:
            if msg.get("message_type") == "audience_building_input":
                msg["metadata"]["use_intelligence_data"] = False

        mock_tavily_research.return_value = {
            "persona_search_results": "Research data",
            "pain_points_search_results": "Research data",
            "journey_search_results": "Research data",
            "icp_search_results": "Research data",
            "channel_search_results": "Research data",
            "industry_search_results": None,
        }
        mock_llm_analyze.return_value = sample_audience_result

        async def async_tavily_mock(*args, **kwargs):
            return mock_tavily_research.return_value

        async def async_llm_mock(*args, **kwargs):
            return mock_llm_analyze.return_value

        mock_tavily_research.side_effect = async_tavily_mock
        mock_llm_analyze.side_effect = async_llm_mock

        result = audience_builder_node(initial_audience_state)

        assert "audience_data" in result
        # Should still work, just without intelligence context
        assert result["agent_execution_log"][0]["action"] == "build_audience"

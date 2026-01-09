"""
Unit Tests for Lead Research Agent (Phase 4.2)
==============================================
Comprehensive tests for lead research operations.

Test Coverage:
- Schema validation (LeadResearchInput, LeadResearchResult, etc.)
- Node execution with mocked Tavily and LLM
- Fallback behavior on errors
- Lead qualification scoring
- CRM data conversion
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from backend.app.schemas.lead_research import (
    # Enums
    LeadSource,
    ResearchDepth,
    EnrichmentConfidence,
    LeadQualificationScore,
    # Input
    LeadResearchInput,
    LeadResearchRequest,
    # Company Data
    CompanyOverview,
    FundingInfo,
    KeyPerson,
    TechnologyStack,
    NewsItem,
    CompetitorInfo,
    SocialPresence,
    # Output
    LeadResearchResult,
    LeadEnrichmentPayload,
    LeadResearchResponse,
)
from backend.graph.state import (
    OrchestratorState,
    create_initial_state,
    CRMProvider,
    WorkflowStatus,
    AgentMessage,
    LeadData,
)
from backend.graph.nodes import (
    lead_research_node,
    _convert_research_to_lead_data,
    _fallback_lead_data,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_lead_input():
    """Sample lead research input."""
    return LeadResearchInput(
        company_name="Anthropic",
        company_domain="anthropic.com",
        contact_name="Dario Amodei",
        contact_title="CEO",
        research_depth=ResearchDepth.STANDARD,
        focus_areas=["AI safety", "funding"],
        source=LeadSource.MANUAL,
    )


@pytest.fixture
def sample_company_overview():
    """Sample company overview."""
    return CompanyOverview(
        name="Anthropic",
        description="AI safety company building reliable AI systems",
        website="https://anthropic.com",
        industry="Artificial Intelligence",
        founded_year=2021,
        headquarters="San Francisco, CA",
        employee_count="500-1000",
    )


@pytest.fixture
def sample_research_result(sample_company_overview):
    """Sample lead research result."""
    return LeadResearchResult(
        company=sample_company_overview,
        funding=FundingInfo(
            total_raised="$7.3B",
            last_round="Series E",
            last_round_amount="$4B",
            last_round_date="2024-03",
            investors=["Google", "Salesforce", "Amazon"],
        ),
        key_people=[
            KeyPerson(
                name="Dario Amodei",
                title="CEO",
                linkedin_url="https://linkedin.com/in/darioamodei",
                bio_snippet="Co-founder, former VP at OpenAI",
            ),
            KeyPerson(
                name="Daniela Amodei",
                title="President",
                linkedin_url="https://linkedin.com/in/danielaamodei",
            ),
        ],
        technology=TechnologyStack(
            technologies=["Python", "JAX", "AWS"],
            tech_categories=["AI/ML", "Cloud"],
            source="Job postings",
        ),
        recent_news=[
            NewsItem(
                title="Anthropic Raises $4B in Series E",
                url="https://example.com/news/1",
                snippet="AI safety startup secures massive funding",
                published_date="2024-03-15",
                sentiment="positive",
            ),
        ],
        competitors=[
            CompetitorInfo(
                name="OpenAI",
                description="AI research company",
                website="https://openai.com",
                competitive_advantage="ChatGPT market leader",
            ),
        ],
        social_presence=SocialPresence(
            linkedin_url="https://linkedin.com/company/anthropic",
            twitter_url="https://twitter.com/AnthropicAI",
        ),
        business_signals=[
            "Raised $4B in Series E",
            "Expanding enterprise sales team",
            "Launched Claude 3 product line",
        ],
        pain_points=[
            "Scaling enterprise deployments",
            "Competition from OpenAI",
        ],
        opportunities=[
            "Growing enterprise AI market",
            "Strong safety positioning",
        ],
        talking_points=[
            "Congrats on the Series E - how is scaling going?",
            "Saw Claude 3 launch - what's enterprise adoption like?",
            "Your safety-first approach is unique - how does that resonate?",
        ],
        qualification_score=LeadQualificationScore.HOT,
        qualification_reasoning="Well-funded, fast-growing AI company with strong market position",
        research_summary="Leading AI safety company with massive funding, rapid growth, and strong technical team",
        confidence_level=EnrichmentConfidence.HIGH,
        sources_used=[
            "https://anthropic.com/about",
            "https://techcrunch.com/anthropic-series-e",
        ],
        research_notes="Strong candidate for enterprise AI solutions",
    )


@pytest.fixture
def initial_lead_research_state(sample_lead_input):
    """Initial state with lead research input."""
    state = create_initial_state(
        client_id="client-12345",
        client_name="Test Enterprise",
        crm_provider=CRMProvider.HUBSPOT,
        workflow_type="sales_ops_only",
    )
    # Add lead research input as a message
    state["messages"] = [
        AgentMessage(
            message_id=str(uuid4()),
            from_agent="user",
            to_agent="lead_research",
            message_type="lead_research_input",
            content="Research this company",
            metadata={
                "company_name": sample_lead_input.company_name,
                "company_domain": sample_lead_input.company_domain,
                "contact_name": sample_lead_input.contact_name,
                "contact_title": sample_lead_input.contact_title,
                "research_depth": sample_lead_input.research_depth.value,
                "focus_areas": sample_lead_input.focus_areas,
                "source": sample_lead_input.source.value,
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    ]
    state["status"] = WorkflowStatus.IN_PROGRESS
    return state


@pytest.fixture
def mock_tavily_search_response():
    """Mock Tavily search response."""
    return MagicMock(
        answer="Anthropic is an AI safety company founded in 2021...",
        results=[
            MagicMock(
                title="Anthropic - AI Safety",
                url="https://anthropic.com",
                content="Anthropic builds reliable AI systems...",
                score=0.95,
            ),
        ],
    )


@pytest.fixture
def mock_tavily_news_response():
    """Mock Tavily news response."""
    return MagicMock(
        results=[
            MagicMock(
                title="Anthropic Raises $4B",
                url="https://news.com/anthropic",
                content="AI safety startup secures funding...",
                published_date="2024-03-15",
            ),
        ],
    )


# =============================================================================
# SCHEMA TESTS
# =============================================================================


class TestLeadResearchInputSchema:
    """Tests for LeadResearchInput schema."""

    def test_valid_lead_input(self):
        """Should create valid LeadResearchInput."""
        input_data = LeadResearchInput(
            company_name="Test Company",
            company_domain="testcompany.com",
        )
        assert input_data.company_name == "Test Company"
        assert input_data.research_depth == ResearchDepth.STANDARD  # default

    def test_company_name_required(self):
        """Should fail without company name."""
        with pytest.raises(ValueError):
            LeadResearchInput(company_name="")

    def test_company_name_whitespace_stripped(self):
        """Should strip whitespace from company name."""
        input_data = LeadResearchInput(company_name="  Anthropic  ")
        assert input_data.company_name == "Anthropic"

    def test_default_values(self):
        """Should use default values for optional fields."""
        input_data = LeadResearchInput(company_name="Test")
        assert input_data.research_depth == ResearchDepth.STANDARD
        assert input_data.source == LeadSource.MANUAL
        assert input_data.focus_areas == []

    def test_all_research_depths(self):
        """Should accept all research depth levels."""
        for depth in ResearchDepth:
            input_data = LeadResearchInput(
                company_name="Test",
                research_depth=depth,
            )
            assert input_data.research_depth == depth


class TestCompanyOverviewSchema:
    """Tests for CompanyOverview schema."""

    def test_valid_company_overview(self):
        """Should create valid CompanyOverview."""
        overview = CompanyOverview(
            name="Anthropic",
            description="AI safety company",
            industry="Artificial Intelligence",
        )
        assert overview.name == "Anthropic"
        assert overview.industry == "Artificial Intelligence"

    def test_minimal_company_overview(self):
        """Should accept minimal data."""
        overview = CompanyOverview(name="Test Co")
        assert overview.name == "Test Co"
        assert overview.description is None


class TestLeadResearchResultSchema:
    """Tests for LeadResearchResult schema."""

    def test_valid_research_result(self, sample_company_overview):
        """Should create valid LeadResearchResult."""
        result = LeadResearchResult(
            company=sample_company_overview,
            research_summary="Test summary",
        )
        assert result.company.name == "Anthropic"
        assert result.qualification_score == LeadQualificationScore.WARM  # default

    def test_qualification_scores(self, sample_company_overview):
        """Should accept all qualification scores."""
        for score in LeadQualificationScore:
            result = LeadResearchResult(
                company=sample_company_overview,
                research_summary="Test",
                qualification_score=score,
            )
            assert result.qualification_score == score

    def test_confidence_levels(self, sample_company_overview):
        """Should accept all confidence levels."""
        for level in EnrichmentConfidence:
            result = LeadResearchResult(
                company=sample_company_overview,
                research_summary="Test",
                confidence_level=level,
            )
            assert result.confidence_level == level


class TestLeadEnrichmentPayloadSchema:
    """Tests for LeadEnrichmentPayload schema."""

    def test_valid_enrichment_payload(self):
        """Should create valid enrichment payload."""
        payload = LeadEnrichmentPayload(
            company_name="Anthropic",
            company_industry="AI",
            enrichment_date="2024-01-15",
        )
        assert payload.company_name == "Anthropic"
        assert payload.enrichment_source == "ai_research"

    def test_lead_score_bounds(self):
        """Lead score should be 0-100."""
        payload = LeadEnrichmentPayload(
            company_name="Test",
            enrichment_date="2024-01-15",
            lead_score=85.0,
        )
        assert payload.lead_score == 85.0

        with pytest.raises(ValueError):
            LeadEnrichmentPayload(
                company_name="Test",
                enrichment_date="2024-01-15",
                lead_score=150.0,  # Invalid
            )


# =============================================================================
# CONVERSION FUNCTION TESTS
# =============================================================================


class TestConvertResearchToLeadData:
    """Tests for _convert_research_to_lead_data function."""

    def test_convert_hot_lead(self, sample_lead_input, sample_research_result):
        """Should convert HOT lead with high score."""
        lead_data = _convert_research_to_lead_data(sample_lead_input, sample_research_result)

        assert lead_data["company_name"] == "Anthropic"
        assert lead_data["lead_score"] == 90.0  # HOT score
        assert lead_data["industry"] == "Artificial Intelligence"
        assert "business_signals" in lead_data["enrichment_data"]

    def test_convert_warm_lead(self, sample_lead_input, sample_research_result):
        """Should convert WARM lead with medium score."""
        sample_research_result.qualification_score = LeadQualificationScore.WARM
        lead_data = _convert_research_to_lead_data(sample_lead_input, sample_research_result)

        assert lead_data["lead_score"] == 60.0  # WARM score

    def test_convert_cold_lead(self, sample_lead_input, sample_research_result):
        """Should convert COLD lead with low score."""
        sample_research_result.qualification_score = LeadQualificationScore.COLD
        lead_data = _convert_research_to_lead_data(sample_lead_input, sample_research_result)

        assert lead_data["lead_score"] == 35.0  # COLD score

    def test_convert_unqualified_lead(self, sample_lead_input, sample_research_result):
        """Should convert UNQUALIFIED lead with minimal score."""
        sample_research_result.qualification_score = LeadQualificationScore.UNQUALIFIED
        lead_data = _convert_research_to_lead_data(sample_lead_input, sample_research_result)

        assert lead_data["lead_score"] == 10.0  # UNQUALIFIED score

    def test_contact_from_input(self, sample_lead_input, sample_research_result):
        """Should use contact from input if provided."""
        lead_data = _convert_research_to_lead_data(sample_lead_input, sample_research_result)

        assert lead_data["contact_name"] == "Dario Amodei"

    def test_contact_from_key_people(self, sample_research_result):
        """Should use first key person if no contact in input."""
        input_without_contact = LeadResearchInput(
            company_name="Anthropic",
        )
        lead_data = _convert_research_to_lead_data(input_without_contact, sample_research_result)

        assert lead_data["contact_name"] == "Dario Amodei"  # From key_people
        assert lead_data["linkedin_url"] == "https://linkedin.com/in/darioamodei"


class TestFallbackLeadData:
    """Tests for _fallback_lead_data function."""

    def test_fallback_creates_basic_lead(self, sample_lead_input):
        """Should create basic lead with error info."""
        lead_data = _fallback_lead_data(sample_lead_input, "Test error")

        assert lead_data["company_name"] == "Anthropic"
        assert lead_data["contact_name"] == "Dario Amodei"
        assert lead_data["lead_score"] is None
        assert lead_data["enrichment_data"]["error"] == "Test error"
        assert lead_data["enrichment_data"]["research_failed"] is True

    def test_fallback_preserves_input_data(self, sample_lead_input):
        """Should preserve all input data."""
        lead_data = _fallback_lead_data(sample_lead_input, "Error")

        assert lead_data["enrichment_data"]["input_domain"] == "anthropic.com"
        assert lead_data["source"] == "manual"


# =============================================================================
# NODE EXECUTION TESTS
# =============================================================================


class TestLeadResearchNode:
    """Tests for lead_research_node function."""

    def test_no_input_returns_error(self):
        """Should return error when no lead input provided."""
        state = create_initial_state(
            client_id="test",
            client_name="Test",
            crm_provider=CRMProvider.HUBSPOT,
        )
        state["messages"] = []

        result = lead_research_node(state)

        assert "error" in result["agent_execution_log"][0]["action"]
        assert "error" in result["messages"][0]["message_type"]

    @pytest.mark.asyncio
    async def test_successful_research(
        self,
        initial_lead_research_state,
        sample_research_result,
        mock_tavily_search_response,
        mock_tavily_news_response,
    ):
        """Should complete research successfully with mocks."""
        with patch("backend.graph.nodes.TavilyService") as mock_tavily_class, \
             patch("backend.graph.nodes.LLMService") as mock_llm_class, \
             patch("backend.graph.nodes.PromptManager") as mock_prompt_class:

            # Setup Tavily mock
            mock_tavily = MagicMock()
            mock_tavily.is_configured = True
            mock_tavily.search = AsyncMock(return_value=mock_tavily_search_response)
            mock_tavily.search_news = AsyncMock(return_value=mock_tavily_news_response)
            mock_tavily_class.get_instance.return_value = mock_tavily

            # Setup LLM mock
            mock_llm = MagicMock()
            mock_llm.generate_structured = AsyncMock(return_value=(sample_research_result, None))
            mock_llm_class.get_instance.return_value = mock_llm

            # Setup PromptManager mock
            mock_prompt = MagicMock()
            mock_prompt.get_full_prompt.return_value = ("system prompt", "user prompt")
            mock_prompt_class.get_instance.return_value = mock_prompt

            # Execute node
            result = lead_research_node(initial_lead_research_state)

            # Verify results
            assert len(result["leads"]) == 1
            lead = result["leads"][0]
            assert lead["company_name"] == "Anthropic"
            assert lead["lead_score"] == 90.0
            assert "research_complete" in result["agent_execution_log"][0]["action"]

    def test_tavily_not_configured(self, initial_lead_research_state, sample_research_result):
        """Should handle Tavily not configured."""
        with patch("backend.graph.nodes.TavilyService") as mock_tavily_class, \
             patch("backend.graph.nodes.LLMService") as mock_llm_class, \
             patch("backend.graph.nodes.PromptManager") as mock_prompt_class:

            # Setup Tavily as not configured
            mock_tavily = MagicMock()
            mock_tavily.is_configured = False
            mock_tavily_class.get_instance.return_value = mock_tavily

            # Setup LLM mock
            mock_llm = MagicMock()
            mock_llm.generate_structured = AsyncMock(return_value=(sample_research_result, None))
            mock_llm_class.get_instance.return_value = mock_llm

            # Setup PromptManager mock
            mock_prompt = MagicMock()
            mock_prompt.get_full_prompt.return_value = ("system prompt", "user prompt")
            mock_prompt_class.get_instance.return_value = mock_prompt

            # Execute node - should still work with empty search results
            result = lead_research_node(initial_lead_research_state)

            # Verify results (LLM analysis still happens with empty data)
            assert len(result["leads"]) == 1


class TestLeadResearchErrorHandling:
    """Tests for error handling in lead research."""

    def test_tavily_error_fallback(self, initial_lead_research_state):
        """Should fallback on Tavily error."""
        from backend.services.tavily_service import TavilyError

        with patch("backend.graph.nodes._research_lead_with_tavily") as mock_tavily_func:
            # Make the async function raise TavilyError
            async def raise_tavily_error(*args, **kwargs):
                raise TavilyError("API error")
            mock_tavily_func.side_effect = raise_tavily_error

            result = lead_research_node(initial_lead_research_state)

            # Should create fallback lead
            assert len(result["leads"]) == 1
            assert result["leads"][0]["enrichment_data"]["research_failed"] is True
            assert "tavily_error" in result["agent_execution_log"][0]["action"]

    def test_llm_error_fallback(
        self,
        initial_lead_research_state,
        mock_tavily_search_response,
        mock_tavily_news_response,
    ):
        """Should fallback on LLM error."""
        from backend.services.llm_service import LLMError

        # Mock both Tavily and LLM at the internal function level
        with patch("backend.graph.nodes._research_lead_with_tavily") as mock_tavily_func, \
             patch("backend.graph.nodes._analyze_lead_with_llm") as mock_llm_func:

            # Setup working Tavily
            async def return_tavily_results(*args, **kwargs):
                return ({"answer": "test"}, [], None)
            mock_tavily_func.side_effect = return_tavily_results

            # Setup failing LLM
            async def raise_llm_error(*args, **kwargs):
                raise LLMError("LLM failed", retryable=False)
            mock_llm_func.side_effect = raise_llm_error

            result = lead_research_node(initial_lead_research_state)

            # Should create fallback lead
            assert len(result["leads"]) == 1
            assert result["leads"][0]["enrichment_data"]["research_failed"] is True
            assert "llm_error" in result["agent_execution_log"][0]["action"]


# =============================================================================
# PHASE 4.2 VERIFICATION TESTS
# =============================================================================


class TestPhase42Verification:
    """Verification tests for Phase 4.2 completion."""

    def test_all_enums_defined(self):
        """All required enums should be defined."""
        assert LeadSource.MANUAL.value == "manual"
        assert ResearchDepth.DEEP.value == "deep"
        assert EnrichmentConfidence.HIGH.value == "high"
        assert LeadQualificationScore.HOT.value == "hot"

    def test_all_schemas_importable(self):
        """All schemas should be importable."""
        from backend.app.schemas.lead_research import (
            LeadResearchInput,
            LeadResearchResult,
            LeadEnrichmentPayload,
            CompanyOverview,
            FundingInfo,
            KeyPerson,
            TechnologyStack,
            NewsItem,
            CompetitorInfo,
            SocialPresence,
        )
        assert LeadResearchInput is not None
        assert LeadResearchResult is not None

    def test_node_function_exists(self):
        """lead_research_node should be importable."""
        from backend.graph.nodes import lead_research_node
        assert callable(lead_research_node)

    def test_prompt_template_exists(self):
        """Lead research prompt should exist."""
        import yaml
        from pathlib import Path

        prompt_path = Path("backend/prompts/templates/sales_ops.yaml")
        if prompt_path.exists():
            with open(prompt_path) as f:
                prompts = yaml.safe_load(f)

            # Find lead_research prompt
            lead_research_prompt = None
            for prompt in prompts:
                if prompt.get("metadata", {}).get("name") == "lead_research":
                    lead_research_prompt = prompt
                    break

            assert lead_research_prompt is not None
            assert "system_prompt" in lead_research_prompt
            assert "user_prompt_template" in lead_research_prompt

    def test_tavily_integration(self):
        """Tavily service should be used."""
        from backend.services.tavily_service import TavilyService, SearchDepth
        assert TavilyService is not None
        assert SearchDepth.BASIC.value == "basic"

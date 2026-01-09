"""
LangGraph Node Functions
========================
Her node, OrchestratorState'i alır ve state güncellemelerini döndürür.
Bu dosya tüm squad'lar için node fonksiyonlarını tanımlar.

Node Types:
- Router Nodes: Squad içi yönlendirme kararları verir
- Agent Nodes: Gerçek işleri yapan ajanlar
- Utility Nodes: Approval, finalization gibi yardımcı işlemler
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from backend.graph.state import (
    OrchestratorState,
    WorkflowStatus,
    ApprovalRequest,
    ApprovalType,
    AgentMessage,
    MeetingNote,
    CRMTask,
    LeadData,
    EmailDraft as StateEmailDraft,
)
from backend.app.schemas.meeting_notes import (
    MeetingAnalysis,
    NormalizedMeetingInput,
    MeetingSentiment,
    ActionItem,
)
from backend.app.schemas.tasks import (
    TaskExtractionResult,
    ExtractedTask,
    TaskPriority,
    TaskStatus,
    TaskType,
    ActionItemInput,
)
from backend.app.schemas.crm_updates import (
    CRMUpdateOperationResult,
    CRMUpdateOperation,
    CRMOperationType,
    OperationRiskLevel,
)
from backend.app.schemas.lead_research import (
    LeadResearchInput,
    LeadResearchResult,
    LeadEnrichmentPayload,
    ResearchDepth,
    EnrichmentConfidence,
    LeadQualificationScore,
    CompanyOverview,
    NewsItem,
)
from backend.app.schemas.market_research import (
    MarketResearchInput,
    MarketResearchResult,
    ResearchScope,
    MarketOverview,
    MarketTrend,
    CompetitorAnalysis,
    TargetSegment,
    MarketOpportunity,
    MarketThreat,
    NewsInsight,
    ConfidenceLevel,
)
from backend.app.schemas.seo_analysis import (
    SEOAnalysisInput,
    SEOAnalysisResult,
    AnalysisDepth,
    KeywordData,
    KeywordCluster,
    SERPAnalysis,
    SERPResult,
    CompetitorSEOProfile,
    KeywordGap,
    ContentGap,
    ContentRecommendation,
    ConfidenceLevel as SEOConfidenceLevel,
)
from backend.app.schemas.web_analysis import (
    WebAnalysisInput,
    WebAnalysisResult,
    AnalysisType,
    PerformanceGrade,
    ContentQuality,
    MobileReadiness,
    SecurityStatus,
    CMSType,
    HostingType,
    SiteStructure,
    TechnologyStack,
    PerformanceMetrics,
    ContentMetrics,
    UXAnalysis,
    SecurityAssessment,
    CompetitorWebProfile,
    CompetitiveInsight,
    ConfidenceLevel as WebConfidenceLevel,
)
from backend.app.schemas.audience_building import (
    AudienceBuildingInput,
    AudienceBuildingResult,
    CompanySize,
    PersonaType,
    BuyingStage,
    ChannelType,
    PriorityLevel,
    ICPProfile,
    ICPFirmographics,
    ICPTechnographics,
    ICPBehavioral,
    BuyerPersona,
    PersonaDemographics,
    PersonaPsychographics,
    PersonaBehavior,
    PainPointAnalysis,
    JourneyStage,
    JourneyTouchpoint,
    PersonaMessage,
    ChannelStrategy,
    ConfidenceLevel as AudienceConfidenceLevel,
)
from backend.app.schemas.email import (
    EmailType,
    EmailTone,
    EmailPriority,
    EmailCopilotInput,
    EmailRecipient,
    LeadContext,
    MeetingContext,
    EmailDraft as PydanticEmailDraft,
    EmailGenerationResult,
    EmailDeliveryPayload,
    EmailApprovalPayload,
    EmailContext,
)
from backend.services.llm_service import LLMService, LLMError
from backend.services.tavily_service import TavilyService, TavilyError, SearchDepth
from backend.services.email.context_builder import (
    EmailContextBuilder,
    get_email_context_builder,
    format_lead_context_for_prompt,
    format_meeting_context_for_prompt,
    format_brandvoice_context_for_prompt,
)
from backend.services.email.adapters import (
    get_email_adapter,
    EmailAdapterError,
    EmailDeliveryError,
)
from backend.prompts.base import PromptManager

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _create_execution_log(
    agent_name: str,
    action: str,
    details: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create a standardized execution log entry."""
    return {
        "agent": agent_name,
        "action": action,
        "details": details or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _create_agent_message(
    from_agent: str,
    content: str,
    message_type: str = "info",
    to_agent: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentMessage:
    """Create a standardized agent message."""
    return AgentMessage(
        message_id=str(uuid4()),
        from_agent=from_agent,
        to_agent=to_agent,
        message_type=message_type,
        content=content,
        metadata=metadata or {},
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# SUPERVISOR NODE
# =============================================================================

def supervisor_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Supervisor Node - Main orchestration decision maker.

    Determines which squad should run next based on:
    - Workflow type
    - Current progress
    - Pending tasks

    Returns state updates with routing decision in messages.
    """
    logger.info(f"[Supervisor] Processing workflow: {state['workflow_id']}")

    workflow_type = state["workflow_type"]
    current_status = state["status"]

    # Update status if starting
    updates: dict[str, Any] = {}

    if current_status == WorkflowStatus.PENDING:
        updates["status"] = WorkflowStatus.IN_PROGRESS

    # Log execution
    updates["agent_execution_log"] = [
        _create_execution_log(
            "supervisor",
            "route_decision",
            {"workflow_type": workflow_type}
        )
    ]

    # Create routing message (actual routing logic is in routers.py)
    updates["messages"] = [
        _create_agent_message(
            from_agent="supervisor",
            content=f"Routing workflow type: {workflow_type}",
            message_type="info",
        )
    ]

    updates["updated_at"] = datetime.now(timezone.utc).isoformat()

    return updates


# =============================================================================
# INTELLIGENCE SQUAD NODES
# =============================================================================

def intelligence_router_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Intelligence Squad Router - Coordinates intelligence agents.

    Decides which intelligence agent should run next based on:
    - Which analyses are complete
    - Which are still needed
    """
    logger.info("[Intelligence Router] Evaluating next agent")

    return {
        "agent_execution_log": [
            _create_execution_log("intelligence_router", "evaluate_progress")
        ],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


async def _conduct_market_research_with_tavily(
    research_input: MarketResearchInput,
) -> dict[str, Any]:
    """
    Internal helper to conduct web research using Tavily API.

    Args:
        research_input: Market research input parameters

    Returns:
        Dict containing raw search results for LLM analysis
    """
    tavily = TavilyService.get_instance()

    # Build search queries based on input
    industry = research_input.industry
    sub_segment = research_input.sub_segment or ""
    geo_focus = research_input.geographic_focus or ""

    # Parallel search tasks for efficiency
    search_tasks = []

    # 1. Industry overview search
    industry_query = f"{industry} {sub_segment} market overview industry analysis {geo_focus}".strip()
    search_tasks.append(
        tavily.search(
            query=industry_query,
            search_depth=SearchDepth.ADVANCED,
            max_results=8,
            include_answer=True,
        )
    )

    # 2. Market trends search
    trends_query = f"{industry} {sub_segment} market trends {research_input.time_horizon} forecast {geo_focus}".strip()
    search_tasks.append(
        tavily.search(
            query=trends_query,
            search_depth=SearchDepth.ADVANCED,
            max_results=6,
            include_answer=True,
        )
    )

    # 3. Competitor search
    if research_input.known_competitors:
        competitor_names = ", ".join(research_input.known_competitors[:5])
        competitor_query = f"{industry} competitors {competitor_names} market share analysis"
    else:
        competitor_query = f"{industry} {sub_segment} top companies market leaders competitors {geo_focus}".strip()

    search_tasks.append(
        tavily.search(
            query=competitor_query,
            search_depth=SearchDepth.ADVANCED,
            max_results=8,
            include_answer=True,
        )
    )

    # 4. News search
    search_tasks.append(
        tavily.search_news(
            query=f"{industry} {sub_segment} news developments",
            days=30,
            max_results=8,
        )
    )

    # 5. Market size search (if comprehensive scope)
    if research_input.research_scope == ResearchScope.COMPREHENSIVE:
        market_size_query = f"{industry} {sub_segment} market size value TAM SAM {geo_focus} {research_input.time_horizon}".strip()
        search_tasks.append(
            tavily.search(
                query=market_size_query,
                search_depth=SearchDepth.ADVANCED,
                max_results=5,
                include_answer=True,
            )
        )

    # Execute all searches in parallel
    results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Process results
    research_data = {
        "industry_search_results": None,
        "trends_search_results": None,
        "competitor_search_results": None,
        "news_results": [],
        "market_size_results": None,
    }

    # Industry overview
    if not isinstance(results[0], Exception):
        industry_result = results[0]
        research_data["industry_search_results"] = (
            f"AI Answer: {industry_result.answer}\n\n"
            + "\n\n".join([
                f"**{r.title}**\n{r.content}\nSource: {r.url}"
                for r in industry_result.results
            ])
        )

    # Trends
    if not isinstance(results[1], Exception):
        trends_result = results[1]
        research_data["trends_search_results"] = (
            f"AI Answer: {trends_result.answer}\n\n"
            + "\n\n".join([
                f"**{r.title}**\n{r.content}\nSource: {r.url}"
                for r in trends_result.results
            ])
        )

    # Competitors
    if not isinstance(results[2], Exception):
        competitor_result = results[2]
        research_data["competitor_search_results"] = (
            f"AI Answer: {competitor_result.answer}\n\n"
            + "\n\n".join([
                f"**{r.title}**\n{r.content}\nSource: {r.url}"
                for r in competitor_result.results
            ])
        )

    # News
    if not isinstance(results[3], Exception):
        news_result = results[3]
        research_data["news_results"] = [
            {
                "title": r.title,
                "url": r.url,
                "content": r.content,
                "published_date": r.published_date,
            }
            for r in news_result.results
        ]

    # Market size (if applicable)
    if len(results) > 4 and not isinstance(results[4], Exception):
        market_size_result = results[4]
        research_data["market_size_results"] = (
            f"AI Answer: {market_size_result.answer}\n\n"
            + "\n\n".join([
                f"**{r.title}**\n{r.content}\nSource: {r.url}"
                for r in market_size_result.results
            ])
        )

    return research_data


async def _analyze_market_with_llm(
    research_input: MarketResearchInput,
    research_data: dict[str, Any],
    client_context: dict[str, Any],
) -> MarketResearchResult:
    """
    Internal helper to call LLM for market research analysis.

    Args:
        research_input: Original research input
        research_data: Raw research data from Tavily
        client_context: Client context from state

    Returns:
        MarketResearchResult: Structured market analysis
    """
    llm_service = LLMService.get_instance()
    prompt_manager = PromptManager.get_instance()

    # Get today's date for prompt
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Get and render the prompt template
    system_prompt, user_prompt = prompt_manager.get_full_prompt(
        "market_research",
        client_name=client_context.get("client_name", "Unknown"),
        industry=research_input.industry,
        sub_segment=research_input.sub_segment,
        geographic_focus=research_input.geographic_focus,
        research_scope=research_input.research_scope.value if research_input.research_scope else "standard",
        time_horizon=research_input.time_horizon,
        today_date=today,
        client_context=research_input.client_context,
        known_competitors=research_input.known_competitors,
        focus_areas=research_input.focus_areas,
        additional_questions=research_input.additional_questions,
        industry_search_results=research_data.get("industry_search_results"),
        trends_search_results=research_data.get("trends_search_results"),
        competitor_search_results=research_data.get("competitor_search_results"),
        news_results=research_data.get("news_results", []),
        market_size_results=research_data.get("market_size_results"),
    )

    # Call LLM with structured output
    analysis, usage = await llm_service.generate_structured(
        output_schema=MarketResearchResult,
        prompt=user_prompt,
        system_prompt=system_prompt,
    )

    logger.info(
        f"[Market Research] LLM analysis complete. "
        f"Trends: {len(analysis.trends)}, "
        f"Competitors: {len(analysis.competitors)}, "
        f"Opportunities: {len(analysis.opportunities)}, "
        f"Tokens: {usage.total_tokens}"
    )

    return analysis


def _convert_research_to_state_format(
    analysis: MarketResearchResult,
) -> dict[str, Any]:
    """
    Convert MarketResearchResult to the MarketResearch TypedDict format for state.

    Args:
        analysis: LLM analysis result

    Returns:
        Dict matching MarketResearch TypedDict structure
    """
    return {
        "industry_trends": [
            {
                "title": t.title,
                "description": t.description,
                "category": t.category,
                "sentiment": t.sentiment.value if hasattr(t.sentiment, 'value') else t.sentiment,
                "impact_level": t.impact_level,
            }
            for t in analysis.trends
        ],
        "market_size": {
            "estimate": analysis.market_overview.market_size,
            "growth_rate": analysis.market_overview.growth_rate,
            "maturity": analysis.market_overview.maturity.value if hasattr(analysis.market_overview.maturity, 'value') else analysis.market_overview.maturity,
        } if analysis.market_overview else None,
        "key_players": [
            {
                "name": c.name,
                "description": c.description,
                "website": c.website,
                "market_position": c.market_position,
                "strengths": c.strengths,
                "weaknesses": c.weaknesses,
            }
            for c in analysis.competitors
        ],
        "opportunities": [opp.title for opp in analysis.opportunities],
        "threats": [threat.title for threat in analysis.threats],
        "news_articles": [
            {
                "title": n.title,
                "url": n.url,
                "summary": n.snippet,
                "date": n.published_date,
            }
            for n in analysis.recent_news
        ],
    }


def market_research_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Market Research Agent - Conducts comprehensive market analysis.

    Uses Tavily API and LLM to research:
    - Industry overview and market size
    - Market trends and forecasts
    - Competitive landscape
    - Target segments
    - Opportunities and threats
    - Strategic recommendations

    This is a core Intelligence Squad agent (Phase 5.1).

    Input: Expects market research parameters in state["messages"] with type="market_research_input"
           or via workflow trigger payload.

    Output: Populates state["market_research"] with structured analysis.
    """
    logger.info("[Market Research] Conducting market analysis")

    # Extract research input from state
    research_input: MarketResearchInput | None = None

    # Look for market research input in messages
    for message in state.get("messages", []):
        if message.get("message_type") == "market_research_input":
            try:
                input_data = message.get("metadata", {})
                research_input = MarketResearchInput(
                    industry=input_data.get("industry", ""),
                    sub_segment=input_data.get("sub_segment"),
                    geographic_focus=input_data.get("geographic_focus"),
                    research_scope=ResearchScope(input_data.get("research_scope", "standard")),
                    focus_areas=input_data.get("focus_areas", []),
                    known_competitors=input_data.get("known_competitors", []),
                    exclude_competitors=input_data.get("exclude_competitors", []),
                    time_horizon=input_data.get("time_horizon", "12 months"),
                    client_context=input_data.get("client_context"),
                    additional_questions=input_data.get("additional_questions", []),
                )
                break
            except Exception as e:
                logger.warning(f"[Market Research] Failed to parse research input: {e}")

    if not research_input:
        # No research input found - return error state
        logger.warning("[Market Research] No research input provided")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "market_research",
                    "error",
                    {"error": "No research input provided"}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="market_research",
                    content="Error: No market research input provided. Please provide industry to analyze.",
                    message_type="error",
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Get client context
    client_context = state.get("client", {})

    try:
        # Run async research in sync context
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()

            # First conduct web research
            research_data = loop.run_until_complete(
                _conduct_market_research_with_tavily(research_input)
            )

            # Then analyze with LLM
            analysis = loop.run_until_complete(
                _analyze_market_with_llm(research_input, research_data, client_context)
            )
        except RuntimeError:
            # No event loop running, create a new one
            research_data = asyncio.run(
                _conduct_market_research_with_tavily(research_input)
            )
            analysis = asyncio.run(
                _analyze_market_with_llm(research_input, research_data, client_context)
            )

        # Convert to state format
        market_research_state = _convert_research_to_state_format(analysis)

        return {
            "market_research": market_research_state,
            "agent_execution_log": [
                _create_execution_log(
                    "market_research",
                    "analyze_market",
                    {
                        "industry": research_input.industry,
                        "scope": research_input.research_scope.value if research_input.research_scope else "standard",
                        "trends_count": len(analysis.trends),
                        "competitors_count": len(analysis.competitors),
                        "opportunities_count": len(analysis.opportunities),
                        "threats_count": len(analysis.threats),
                        "confidence": analysis.confidence_level.value if hasattr(analysis.confidence_level, 'value') else analysis.confidence_level,
                    }
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="market_research",
                    content=f"Market research completed for {research_input.industry}: "
                            f"{len(analysis.trends)} trends, {len(analysis.competitors)} competitors, "
                            f"{len(analysis.opportunities)} opportunities identified",
                    message_type="info",
                    metadata={
                        "industry": research_input.industry,
                        "research_summary": analysis.research_summary,
                        "confidence_level": analysis.confidence_level.value if hasattr(analysis.confidence_level, 'value') else analysis.confidence_level,
                    },
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except TavilyError as e:
        logger.error(f"[Market Research] Tavily error: {e}")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "market_research",
                    "error",
                    {"error": str(e), "type": "tavily_error"}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="market_research",
                    content=f"Error conducting web research: {e.message}",
                    message_type="error",
                )
            ],
            "error_message": f"Market Research Agent Error (Tavily): {e.message}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except LLMError as e:
        logger.error(f"[Market Research] LLM error: {e}")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "market_research",
                    "error",
                    {"error": str(e), "provider": e.provider.value if e.provider else None}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="market_research",
                    content=f"Error analyzing market data: {e.message}",
                    message_type="error",
                )
            ],
            "error_message": f"Market Research Agent Error (LLM): {e.message}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.exception(f"[Market Research] Unexpected error: {e}")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "market_research",
                    "error",
                    {"error": str(e)}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="market_research",
                    content=f"Unexpected error in market research: {str(e)}",
                    message_type="error",
                )
            ],
            "error_message": f"Market Research Agent Error: {str(e)}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


async def _conduct_seo_research_with_tavily(
    seo_input: SEOAnalysisInput,
) -> dict[str, Any]:
    """
    Internal helper to conduct SEO research using Tavily API.

    Args:
        seo_input: SEO analysis input parameters

    Returns:
        Dict containing raw search results for LLM analysis
    """
    tavily = TavilyService.get_instance()

    # Build search queries based on input
    target_url = seo_input.target_url or ""
    target_keywords = seo_input.target_keywords or []
    industry = seo_input.industry or ""
    geo_focus = seo_input.geographic_target or ""
    language = seo_input.language or "en"

    # Parallel search tasks for efficiency
    search_tasks = []

    # 1. Keyword research search
    if target_keywords:
        keyword_list = ", ".join(target_keywords[:5])
        keyword_query = f"{keyword_list} keyword research SEO search volume difficulty {industry} {geo_focus}".strip()
    else:
        keyword_query = f"{target_url} main keywords SEO focus topics {industry}".strip()

    search_tasks.append(
        tavily.search(
            query=keyword_query,
            search_depth=SearchDepth.ADVANCED,
            max_results=8,
            include_answer=True,
        )
    )

    # 2. SERP analysis search
    serp_query = f"{target_keywords[0] if target_keywords else target_url} SERP analysis top ranking pages featured snippets {geo_focus}".strip()
    search_tasks.append(
        tavily.search(
            query=serp_query,
            search_depth=SearchDepth.ADVANCED,
            max_results=10,
            include_answer=True,
        )
    )

    # 3. Competitor SEO search
    if seo_input.known_competitors:
        competitor_domains = ", ".join(seo_input.known_competitors[:5])
        competitor_query = f"{competitor_domains} SEO strategy organic traffic keywords backlinks {industry}"
    else:
        competitor_query = f"{industry} top SEO competitors organic leaders {geo_focus} keyword rankings".strip()

    search_tasks.append(
        tavily.search(
            query=competitor_query,
            search_depth=SearchDepth.ADVANCED,
            max_results=8,
            include_answer=True,
        )
    )

    # 4. Content gap analysis search
    content_gap_query = f"{industry} {target_keywords[0] if target_keywords else ''} content gaps opportunities untapped keywords {geo_focus}".strip()
    search_tasks.append(
        tavily.search(
            query=content_gap_query,
            search_depth=SearchDepth.ADVANCED,
            max_results=6,
            include_answer=True,
        )
    )

    # 5. Technical SEO search (if comprehensive analysis)
    if seo_input.analysis_depth == AnalysisDepth.COMPREHENSIVE:
        tech_seo_query = f"SEO best practices technical site speed mobile optimization {industry}".strip()
        search_tasks.append(
            tavily.search(
                query=tech_seo_query,
                search_depth=SearchDepth.ADVANCED,
                max_results=5,
                include_answer=True,
            )
        )

    # Execute all searches in parallel
    results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Process results
    research_data = {
        "keyword_search_results": None,
        "serp_search_results": None,
        "competitor_search_results": None,
        "content_gap_results": None,
        "technical_seo_results": None,
    }

    # Keyword research results
    if not isinstance(results[0], Exception):
        keyword_result = results[0]
        research_data["keyword_search_results"] = (
            f"AI Answer: {keyword_result.answer}\n\n"
            + "\n\n".join([
                f"**{r.title}**\n{r.content}\nSource: {r.url}"
                for r in keyword_result.results
            ])
        )

    # SERP analysis results
    if not isinstance(results[1], Exception):
        serp_result = results[1]
        research_data["serp_search_results"] = (
            f"AI Answer: {serp_result.answer}\n\n"
            + "\n\n".join([
                f"**{r.title}**\n{r.content}\nSource: {r.url}"
                for r in serp_result.results
            ])
        )

    # Competitor SEO results
    if not isinstance(results[2], Exception):
        competitor_result = results[2]
        research_data["competitor_search_results"] = (
            f"AI Answer: {competitor_result.answer}\n\n"
            + "\n\n".join([
                f"**{r.title}**\n{r.content}\nSource: {r.url}"
                for r in competitor_result.results
            ])
        )

    # Content gap results
    if not isinstance(results[3], Exception):
        gap_result = results[3]
        research_data["content_gap_results"] = (
            f"AI Answer: {gap_result.answer}\n\n"
            + "\n\n".join([
                f"**{r.title}**\n{r.content}\nSource: {r.url}"
                for r in gap_result.results
            ])
        )

    # Technical SEO results (if applicable)
    if len(results) > 4 and not isinstance(results[4], Exception):
        tech_result = results[4]
        research_data["technical_seo_results"] = (
            f"AI Answer: {tech_result.answer}\n\n"
            + "\n\n".join([
                f"**{r.title}**\n{r.content}\nSource: {r.url}"
                for r in tech_result.results
            ])
        )

    return research_data


async def _analyze_seo_with_llm(
    seo_input: SEOAnalysisInput,
    research_data: dict[str, Any],
    client_context: dict[str, Any],
) -> SEOAnalysisResult:
    """
    Internal helper to call LLM for SEO analysis.

    Args:
        seo_input: Original SEO input
        research_data: Raw research data from Tavily
        client_context: Client context from state

    Returns:
        SEOAnalysisResult: Structured SEO analysis
    """
    llm_service = LLMService.get_instance()
    prompt_manager = PromptManager.get_instance()

    # Get today's date for prompt
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Get and render the prompt template
    system_prompt, user_prompt = prompt_manager.get_full_prompt(
        "seo_analysis",
        client_name=client_context.get("client_name", "Unknown"),
        target_url=seo_input.target_url,
        target_keywords=seo_input.target_keywords,
        industry=seo_input.industry,
        geographic_target=seo_input.geographic_target,
        language=seo_input.language,
        analysis_depth=seo_input.analysis_depth.value if seo_input.analysis_depth else "standard",
        today_date=today,
        keyword_search_results=research_data.get("keyword_search_results"),
        serp_search_results=research_data.get("serp_search_results"),
        competitor_search_results=research_data.get("competitor_search_results"),
        content_gap_results=research_data.get("content_gap_results"),
        technical_seo_results=research_data.get("technical_seo_results"),
    )

    # Call LLM with structured output
    analysis, usage = await llm_service.generate_structured(
        output_schema=SEOAnalysisResult,
        prompt=user_prompt,
        system_prompt=system_prompt,
    )

    logger.info(
        f"[SEO Analysis] LLM analysis complete. "
        f"Keywords: {len(analysis.primary_keywords)}, "
        f"Clusters: {len(analysis.keyword_clusters)}, "
        f"Competitors: {len(analysis.competitors)}, "
        f"Content gaps: {len(analysis.content_gaps)}, "
        f"Tokens: {usage.total_tokens}"
    )

    return analysis


def _convert_seo_to_state_format(
    analysis: SEOAnalysisResult,
) -> dict[str, Any]:
    """
    Convert SEOAnalysisResult to the SEOData TypedDict format for state.

    Args:
        analysis: LLM analysis result

    Returns:
        Dict matching SEOData TypedDict structure
    """
    return {
        "keywords": [
            {
                "keyword": kw.keyword,
                "volume": kw.search_volume,
                "difficulty": kw.difficulty.value if hasattr(kw.difficulty, 'value') else kw.difficulty,
                "intent": kw.intent.value if hasattr(kw.intent, 'value') else kw.intent,
                "relevance_score": kw.relevance_score,
            }
            for kw in analysis.primary_keywords
        ],
        "competitors": [
            {
                "domain": c.domain,
                "traffic": c.estimated_traffic,
                "authority": c.domain_authority,
                "top_keywords": c.top_ranking_keywords,
                "strengths": c.content_strengths,
                "weaknesses": c.content_weaknesses,
            }
            for c in analysis.competitors
        ],
        "content_gaps": [
            gap.topic for gap in analysis.content_gaps
        ],
        "serp_analysis": [
            {
                "query": serp.keyword,
                "features": [f.value if hasattr(f, 'value') else f for f in serp.serp_features],
                "dominant_content_type": serp.dominant_content_type.value if serp.dominant_content_type and hasattr(serp.dominant_content_type, 'value') else serp.dominant_content_type,
                "top_results": [
                    {
                        "position": r.position,
                        "url": r.url,
                        "title": r.title,
                        "domain": r.domain,
                    }
                    for r in serp.top_results[:5]  # Limit to top 5
                ],
            }
            for serp in analysis.serp_analyses
        ],
        "keyword_clusters": [
            {
                "name": cluster.cluster_name,
                "primary": cluster.primary_keyword,
                "related": cluster.related_keywords,
                "recommended_content": cluster.recommended_content_type.value if cluster.recommended_content_type and hasattr(cluster.recommended_content_type, 'value') else cluster.recommended_content_type,
            }
            for cluster in analysis.keyword_clusters
        ],
        "keyword_gaps": [
            {
                "keyword": gap.keyword,
                "volume": gap.search_volume,
                "difficulty": gap.difficulty.value if hasattr(gap.difficulty, 'value') else gap.difficulty,
                "opportunity_score": gap.opportunity_score,
                "competitors_ranking": gap.competitors_ranking,
            }
            for gap in analysis.keyword_gaps
        ],
        "content_recommendations": [
            {
                "title": rec.title_suggestion,
                "target_keyword": rec.target_keyword,
                "content_type": rec.content_type.value if hasattr(rec.content_type, 'value') else rec.content_type,
                "priority": rec.priority.value if hasattr(rec.priority, 'value') else rec.priority,
            }
            for rec in analysis.content_recommendations
        ],
        "quick_wins": analysis.quick_wins,
        "strategic_recommendations": analysis.strategic_recommendations,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


def seo_analysis_node(state: OrchestratorState) -> dict[str, Any]:
    """
    SEO Analysis Agent - Conducts comprehensive SEO analysis.

    Uses Tavily API and LLM to research:
    - Keyword opportunities and search volumes
    - SERP analysis and ranking factors
    - Competitor SEO profiles
    - Content gaps and opportunities
    - Technical SEO insights (if comprehensive)

    This is a core Intelligence Squad agent (Phase 5.2).

    Input: Expects SEO analysis parameters in state["messages"] with type="seo_analysis_input"
           or via workflow trigger payload.

    Output: Populates state["seo_data"] with structured analysis.
    """
    logger.info("[SEO Analysis] Conducting SEO analysis")

    # Extract SEO input from state
    seo_input: SEOAnalysisInput | None = None

    # Look for SEO analysis input in messages
    for message in state.get("messages", []):
        if message.get("message_type") == "seo_analysis_input":
            try:
                input_data = message.get("metadata", {})
                seo_input = SEOAnalysisInput(
                    target_url=input_data.get("target_url"),
                    target_keywords=input_data.get("target_keywords", []),
                    industry=input_data.get("industry"),
                    geographic_target=input_data.get("geographic_target"),
                    language=input_data.get("language", "en"),
                    analysis_depth=AnalysisDepth(input_data.get("analysis_depth", "standard")),
                    include_competitors=input_data.get("include_competitors", True),
                    known_competitors=input_data.get("known_competitors", []),
                    content_types=input_data.get("content_types", []),
                    business_goals=input_data.get("business_goals", []),
                    additional_context=input_data.get("additional_context"),
                )
                break
            except Exception as e:
                logger.warning(f"[SEO Analysis] Failed to parse SEO input: {e}")

    if not seo_input:
        # No SEO input found - return error state
        logger.warning("[SEO Analysis] No SEO input provided")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "seo_analysis",
                    "error",
                    {"error": "No SEO input provided"}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="seo_analysis",
                    content="Error: No SEO analysis input provided. Please provide target URL or keywords to analyze.",
                    message_type="error",
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Get client context
    client_context = state.get("client", {})

    try:
        # Run async research in sync context
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()

            # First conduct web research
            research_data = loop.run_until_complete(
                _conduct_seo_research_with_tavily(seo_input)
            )

            # Then analyze with LLM
            analysis = loop.run_until_complete(
                _analyze_seo_with_llm(seo_input, research_data, client_context)
            )
        except RuntimeError:
            # No event loop running, create a new one
            research_data = asyncio.run(
                _conduct_seo_research_with_tavily(seo_input)
            )
            analysis = asyncio.run(
                _analyze_seo_with_llm(seo_input, research_data, client_context)
            )

        # Convert to state format
        seo_data_state = _convert_seo_to_state_format(analysis)

        return {
            "seo_data": seo_data_state,
            "agent_execution_log": [
                _create_execution_log(
                    "seo_analysis",
                    "analyze_seo",
                    {
                        "target_url": seo_input.target_url,
                        "target_keywords": seo_input.target_keywords[:3] if seo_input.target_keywords else [],
                        "depth": seo_input.analysis_depth.value if seo_input.analysis_depth else "standard",
                        "keywords_count": len(analysis.primary_keywords),
                        "clusters_count": len(analysis.keyword_clusters),
                        "competitors_count": len(analysis.competitors),
                        "content_gaps_count": len(analysis.content_gaps),
                        "confidence": analysis.confidence_level.value if hasattr(analysis.confidence_level, 'value') else analysis.confidence_level,
                    }
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="seo_analysis",
                    content=f"SEO analysis completed: "
                            f"{len(analysis.primary_keywords)} keywords, {len(analysis.keyword_clusters)} clusters, "
                            f"{len(analysis.competitors)} competitors, {len(analysis.content_gaps)} content gaps identified",
                    message_type="info",
                    metadata={
                        "target": seo_input.target_url or seo_input.target_keywords[0] if seo_input.target_keywords else "N/A",
                        "analysis_summary": analysis.analysis_summary,
                        "confidence_level": analysis.confidence_level.value if hasattr(analysis.confidence_level, 'value') else analysis.confidence_level,
                        "quick_wins": analysis.quick_wins[:3] if analysis.quick_wins else [],
                    },
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except TavilyError as e:
        logger.error(f"[SEO Analysis] Tavily error: {e}")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "seo_analysis",
                    "error",
                    {"error": str(e), "type": "tavily_error"}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="seo_analysis",
                    content=f"Error conducting SEO research: {e.message}",
                    message_type="error",
                )
            ],
            "error_message": f"SEO Analysis Agent Error (Tavily): {e.message}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except LLMError as e:
        logger.error(f"[SEO Analysis] LLM error: {e}")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "seo_analysis",
                    "error",
                    {"error": str(e), "provider": e.provider.value if e.provider else None}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="seo_analysis",
                    content=f"Error analyzing SEO data: {e.message}",
                    message_type="error",
                )
            ],
            "error_message": f"SEO Analysis Agent Error (LLM): {e.message}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.exception(f"[SEO Analysis] Unexpected error: {e}")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "seo_analysis",
                    "error",
                    {"error": str(e)}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="seo_analysis",
                    content=f"Unexpected error in SEO analysis: {str(e)}",
                    message_type="error",
                )
            ],
            "error_message": f"SEO Analysis Agent Error: {str(e)}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


async def _conduct_web_research_with_tavily(
    web_input: WebAnalysisInput,
) -> dict[str, Any]:
    """
    Internal helper to conduct web analysis research using Tavily API.

    Args:
        web_input: Web analysis input parameters

    Returns:
        Dict containing raw search results for LLM analysis
    """
    tavily = TavilyService.get_instance()

    # Build search queries based on input
    primary_domain = web_input.primary_domain or ""
    target_urls = web_input.target_urls or []
    industry = web_input.industry or ""

    # Use primary domain or first target URL for queries
    target = primary_domain or (target_urls[0] if target_urls else "")

    # Parallel search tasks for efficiency
    search_tasks = []

    # 1. Website structure and navigation search
    structure_query = f"{target} website structure navigation site map information architecture".strip()
    search_tasks.append(
        tavily.search(
            query=structure_query,
            search_depth=SearchDepth.ADVANCED,
            max_results=6,
            include_answer=True,
        )
    )

    # 2. Technology stack search
    tech_query = f"{target} technology stack CMS platform framework hosting built with".strip()
    search_tasks.append(
        tavily.search(
            query=tech_query,
            search_depth=SearchDepth.ADVANCED,
            max_results=8,
            include_answer=True,
        )
    )

    # 3. Performance analysis search
    performance_query = f"{target} website performance page speed Core Web Vitals mobile optimization".strip()
    search_tasks.append(
        tavily.search(
            query=performance_query,
            search_depth=SearchDepth.ADVANCED,
            max_results=6,
            include_answer=True,
        )
    )

    # 4. Content and UX analysis search
    content_ux_query = f"{target} website content quality user experience design UX review".strip()
    search_tasks.append(
        tavily.search(
            query=content_ux_query,
            search_depth=SearchDepth.ADVANCED,
            max_results=6,
            include_answer=True,
        )
    )

    # 5. Competitor website analysis (if requested)
    if web_input.include_competitors:
        if web_input.competitor_urls:
            competitor_domains = ", ".join(web_input.competitor_urls[:3])
            competitor_query = f"{competitor_domains} website technology design comparison {industry}".strip()
        else:
            competitor_query = f"{industry} top company websites design technology trends best practices".strip()

        search_tasks.append(
            tavily.search(
                query=competitor_query,
                search_depth=SearchDepth.ADVANCED,
                max_results=8,
                include_answer=True,
            )
        )

    # 6. Security analysis (if technical or competitive analysis)
    if web_input.analysis_type in [AnalysisType.TECHNICAL, AnalysisType.COMPETITIVE]:
        security_query = f"{target} website security HTTPS SSL headers best practices".strip()
        search_tasks.append(
            tavily.search(
                query=security_query,
                search_depth=SearchDepth.ADVANCED,
                max_results=5,
                include_answer=True,
            )
        )

    # Execute all searches in parallel
    results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Process results
    research_data = {
        "structure_search_results": None,
        "technology_search_results": None,
        "performance_search_results": None,
        "content_ux_search_results": None,
        "competitor_search_results": None,
        "security_search_results": None,
    }

    # Structure results
    if not isinstance(results[0], Exception):
        structure_result = results[0]
        research_data["structure_search_results"] = (
            f"AI Answer: {structure_result.answer}\n\n"
            + "\n\n".join([
                f"**{r.title}**\n{r.content}\nSource: {r.url}"
                for r in structure_result.results
            ])
        )

    # Technology stack results
    if not isinstance(results[1], Exception):
        tech_result = results[1]
        research_data["technology_search_results"] = (
            f"AI Answer: {tech_result.answer}\n\n"
            + "\n\n".join([
                f"**{r.title}**\n{r.content}\nSource: {r.url}"
                for r in tech_result.results
            ])
        )

    # Performance results
    if not isinstance(results[2], Exception):
        perf_result = results[2]
        research_data["performance_search_results"] = (
            f"AI Answer: {perf_result.answer}\n\n"
            + "\n\n".join([
                f"**{r.title}**\n{r.content}\nSource: {r.url}"
                for r in perf_result.results
            ])
        )

    # Content/UX results
    if not isinstance(results[3], Exception):
        content_result = results[3]
        research_data["content_ux_search_results"] = (
            f"AI Answer: {content_result.answer}\n\n"
            + "\n\n".join([
                f"**{r.title}**\n{r.content}\nSource: {r.url}"
                for r in content_result.results
            ])
        )

    # Competitor results (if applicable)
    result_idx = 4
    if web_input.include_competitors and len(results) > result_idx and not isinstance(results[result_idx], Exception):
        competitor_result = results[result_idx]
        research_data["competitor_search_results"] = (
            f"AI Answer: {competitor_result.answer}\n\n"
            + "\n\n".join([
                f"**{r.title}**\n{r.content}\nSource: {r.url}"
                for r in competitor_result.results
            ])
        )
        result_idx += 1

    # Security results (if applicable)
    if len(results) > result_idx and not isinstance(results[result_idx], Exception):
        security_result = results[result_idx]
        research_data["security_search_results"] = (
            f"AI Answer: {security_result.answer}\n\n"
            + "\n\n".join([
                f"**{r.title}**\n{r.content}\nSource: {r.url}"
                for r in security_result.results
            ])
        )

    return research_data


async def _analyze_web_with_llm(
    web_input: WebAnalysisInput,
    research_data: dict[str, Any],
    client_context: dict[str, Any],
) -> WebAnalysisResult:
    """
    Internal helper to call LLM for web analysis.

    Args:
        web_input: Original web analysis input
        research_data: Raw research data from Tavily
        client_context: Client context from state

    Returns:
        WebAnalysisResult: Structured web analysis
    """
    llm_service = LLMService.get_instance()
    prompt_manager = PromptManager.get_instance()

    # Get today's date for prompt
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Get and render the prompt template
    system_prompt, user_prompt = prompt_manager.get_full_prompt(
        "web_analysis",
        client_name=client_context.get("client_name", "Unknown"),
        primary_domain=web_input.primary_domain,
        target_urls=web_input.target_urls,
        industry=web_input.industry,
        analysis_type=web_input.analysis_type.value if web_input.analysis_type else "standard",
        today_date=today,
        business_goals=web_input.business_goals,
        competitor_urls=web_input.competitor_urls,
        focus_areas=web_input.focus_areas,
        additional_context=web_input.additional_context,
        structure_search_results=research_data.get("structure_search_results"),
        technology_search_results=research_data.get("technology_search_results"),
        performance_search_results=research_data.get("performance_search_results"),
        content_ux_search_results=research_data.get("content_ux_search_results"),
        competitor_search_results=research_data.get("competitor_search_results"),
        security_search_results=research_data.get("security_search_results"),
    )

    # Call LLM with structured output
    analysis, usage = await llm_service.generate_structured(
        output_schema=WebAnalysisResult,
        prompt=user_prompt,
        system_prompt=system_prompt,
    )

    logger.info(
        f"[Web Analysis] LLM analysis complete. "
        f"Domain: {analysis.analyzed_domain}, "
        f"Competitors: {len(analysis.competitor_profiles)}, "
        f"Quick wins: {len(analysis.quick_wins)}, "
        f"Tokens: {usage.total_tokens}"
    )

    return analysis


def _convert_web_analysis_to_state_format(
    analysis: WebAnalysisResult,
) -> dict[str, Any]:
    """
    Convert WebAnalysisResult to the web_analysis dict format for state.

    Args:
        analysis: LLM analysis result

    Returns:
        Dict matching web_analysis state structure
    """
    return {
        "analyzed_domain": analysis.analyzed_domain,
        "analysis_summary": analysis.analysis_summary,
        "site_structure": {
            "total_pages": analysis.site_structure.total_pages_estimate if analysis.site_structure else None,
            "navigation_type": analysis.site_structure.navigation_type if analysis.site_structure else None,
            "url_structure": analysis.site_structure.url_structure if analysis.site_structure else None,
            "sitemap_available": analysis.site_structure.sitemap_available if analysis.site_structure else False,
            "key_pages": [
                {
                    "url": p.url,
                    "title": p.title,
                    "page_type": p.page_type,
                }
                for p in (analysis.site_structure.key_pages if analysis.site_structure else [])
            ],
        } if analysis.site_structure else None,
        "tech_stack": {
            "cms": analysis.technology_stack.cms.value if analysis.technology_stack and analysis.technology_stack.cms else None,
            "hosting": analysis.technology_stack.hosting.value if analysis.technology_stack and analysis.technology_stack.hosting else None,
            "cdn": analysis.technology_stack.cdn if analysis.technology_stack else None,
            "frontend_framework": analysis.technology_stack.frontend_framework if analysis.technology_stack else None,
            "css_framework": analysis.technology_stack.css_framework if analysis.technology_stack else None,
            "analytics_tools": analysis.technology_stack.analytics_tools if analysis.technology_stack else [],
            "marketing_tools": analysis.technology_stack.marketing_tools if analysis.technology_stack else [],
        } if analysis.technology_stack else None,
        "performance": {
            "overall_grade": analysis.performance.overall_grade.value if analysis.performance and analysis.performance.overall_grade else None,
            "page_load_estimate": analysis.performance.page_load_estimate if analysis.performance else None,
            "mobile_readiness": analysis.performance.mobile_readiness.value if analysis.performance and analysis.performance.mobile_readiness else None,
            "optimization_opportunities": analysis.performance.optimization_opportunities if analysis.performance else [],
            "performance_issues": analysis.performance.performance_issues if analysis.performance else [],
        } if analysis.performance else None,
        "content_quality": {
            "overall_quality": analysis.content_analysis.overall_quality.value if analysis.content_analysis and analysis.content_analysis.overall_quality else None,
            "content_freshness": analysis.content_analysis.content_freshness if analysis.content_analysis else None,
            "content_types": analysis.content_analysis.content_types_found if analysis.content_analysis else [],
            "strengths": analysis.content_analysis.content_strengths if analysis.content_analysis else [],
            "weaknesses": analysis.content_analysis.content_weaknesses if analysis.content_analysis else [],
        } if analysis.content_analysis else None,
        "ux_analysis": {
            "overall_score": analysis.ux_analysis.overall_ux_score if analysis.ux_analysis else None,
            "design_style": analysis.ux_analysis.design_style if analysis.ux_analysis else None,
            "strengths": analysis.ux_analysis.ux_strengths if analysis.ux_analysis else [],
            "weaknesses": analysis.ux_analysis.ux_weaknesses if analysis.ux_analysis else [],
        } if analysis.ux_analysis else None,
        "security": {
            "status": analysis.security.overall_status.value if analysis.security and analysis.security.overall_status else None,
            "https_enabled": analysis.security.https_enabled if analysis.security else True,
            "issues": analysis.security.potential_issues if analysis.security else [],
        } if analysis.security else None,
        "competitors": [
            {
                "domain": c.domain,
                "assessment": c.overall_assessment,
                "strengths": c.strengths,
                "weaknesses": c.weaknesses,
                "notable_features": c.notable_features,
            }
            for c in analysis.competitor_profiles
        ],
        "competitive_insights": [
            {
                "category": ci.category,
                "insight": ci.insight,
                "action": ci.action_item,
                "priority": ci.priority,
            }
            for ci in analysis.competitive_insights
        ],
        "quick_wins": analysis.quick_wins,
        "strategic_recommendations": analysis.strategic_recommendations,
        "priority_actions": analysis.priority_actions,
        "confidence_level": analysis.confidence_level.value if hasattr(analysis.confidence_level, 'value') else analysis.confidence_level,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# AUDIENCE BUILDING HELPER FUNCTIONS (Phase 5.4)
# =============================================================================


def _gather_intelligence_context(state: OrchestratorState) -> dict[str, Any]:
    """
    Gather context from other intelligence agents in state.

    This function reads data from market_research, seo_data, and web_analysis
    to provide comprehensive context for audience building.

    Args:
        state: Current orchestrator state

    Returns:
        Dict containing synthesized intelligence data
    """
    intelligence_context = {
        "market_research": None,
        "seo_data": None,
        "web_analysis": None,
        "sources_available": [],
    }

    # Extract market research data
    market_research = state.get("market_research")
    if market_research:
        intelligence_context["market_research"] = {
            "industry": market_research.get("industry"),
            "market_overview": market_research.get("market_overview"),
            "trends": market_research.get("trends", []),
            "competitors": market_research.get("competitors", []),
            "target_segments": market_research.get("target_segments", []),
            "opportunities": market_research.get("opportunities", []),
            "threats": market_research.get("threats", []),
        }
        intelligence_context["sources_available"].append("market_research")

    # Extract SEO data
    seo_data = state.get("seo_data")
    if seo_data:
        intelligence_context["seo_data"] = {
            "primary_domain": seo_data.get("primary_domain"),
            "keyword_clusters": seo_data.get("keyword_clusters", []),
            "competitor_profiles": seo_data.get("competitor_profiles", []),
            "content_gaps": seo_data.get("content_gaps", []),
            "content_recommendations": seo_data.get("content_recommendations", []),
            "search_intent_insights": seo_data.get("search_intent_insights"),
        }
        intelligence_context["sources_available"].append("seo_data")

    # Extract web analysis data
    web_analysis = state.get("web_analysis")
    if web_analysis:
        intelligence_context["web_analysis"] = {
            "analyzed_domain": web_analysis.get("analyzed_domain"),
            "analysis_summary": web_analysis.get("analysis_summary"),
            "tech_stack": web_analysis.get("tech_stack"),
            "content_quality": web_analysis.get("content_quality"),
            "ux_analysis": web_analysis.get("ux_analysis"),
            "competitors": web_analysis.get("competitors", []),
            "competitive_insights": web_analysis.get("competitive_insights", []),
        }
        intelligence_context["sources_available"].append("web_analysis")

    return intelligence_context


async def _conduct_audience_research_with_tavily(
    audience_input: AudienceBuildingInput,
    intelligence_context: dict[str, Any],
) -> dict[str, Any]:
    """
    Conduct audience-focused web research using Tavily API.

    Args:
        audience_input: Audience building input parameters
        intelligence_context: Data from other intelligence agents

    Returns:
        Dict containing raw research data for audience analysis
    """
    tavily = TavilyService.get_instance()

    # Build search context
    product_category = audience_input.product_category
    target_market = audience_input.target_market or ""
    industries = ", ".join(audience_input.target_industries[:3]) if audience_input.target_industries else ""

    # Prepare search tasks
    search_tasks = []

    # 1. Buyer persona research
    persona_query = f"{product_category} buyer persona decision maker {target_market} {industries}".strip()
    search_tasks.append(
        tavily.search(
            query=persona_query,
            search_depth=SearchDepth.ADVANCED,
            max_results=8,
            include_answer=True,
        )
    )

    # 2. Pain points and challenges research
    pain_points_query = f"{product_category} customer pain points challenges problems {target_market}".strip()
    search_tasks.append(
        tavily.search(
            query=pain_points_query,
            search_depth=SearchDepth.ADVANCED,
            max_results=8,
            include_answer=True,
        )
    )

    # 3. Buying journey and decision process research
    journey_query = f"{product_category} B2B buying journey decision process evaluation criteria {industries}".strip()
    search_tasks.append(
        tavily.search(
            query=journey_query,
            search_depth=SearchDepth.ADVANCED,
            max_results=6,
            include_answer=True,
        )
    )

    # 4. ICP (Ideal Customer Profile) research
    icp_query = f"{product_category} ideal customer profile ICP characteristics {target_market} success factors".strip()
    search_tasks.append(
        tavily.search(
            query=icp_query,
            search_depth=SearchDepth.ADVANCED,
            max_results=6,
            include_answer=True,
        )
    )

    # 5. Channel preferences and content consumption
    channel_query = f"{product_category} {target_market} marketing channels content preferences B2B engagement".strip()
    search_tasks.append(
        tavily.search(
            query=channel_query,
            search_depth=SearchDepth.ADVANCED,
            max_results=6,
            include_answer=True,
        )
    )

    # 6. Industry-specific audience insights (if industries specified)
    if industries:
        industry_query = f"{industries} technology adoption trends buyer behavior {product_category}".strip()
        search_tasks.append(
            tavily.search(
                query=industry_query,
                search_depth=SearchDepth.ADVANCED,
                max_results=6,
                include_answer=True,
            )
        )

    # Execute all searches in parallel
    results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Process results
    research_data = {
        "persona_search_results": None,
        "pain_points_search_results": None,
        "journey_search_results": None,
        "icp_search_results": None,
        "channel_search_results": None,
        "industry_search_results": None,
    }

    # Persona research results
    if not isinstance(results[0], Exception):
        persona_result = results[0]
        research_data["persona_search_results"] = (
            f"AI Answer: {persona_result.answer}\n\n"
            + "\n\n".join([
                f"**{r.title}**\n{r.content}\nSource: {r.url}"
                for r in persona_result.results
            ])
        )

    # Pain points results
    if not isinstance(results[1], Exception):
        pain_result = results[1]
        research_data["pain_points_search_results"] = (
            f"AI Answer: {pain_result.answer}\n\n"
            + "\n\n".join([
                f"**{r.title}**\n{r.content}\nSource: {r.url}"
                for r in pain_result.results
            ])
        )

    # Journey results
    if not isinstance(results[2], Exception):
        journey_result = results[2]
        research_data["journey_search_results"] = (
            f"AI Answer: {journey_result.answer}\n\n"
            + "\n\n".join([
                f"**{r.title}**\n{r.content}\nSource: {r.url}"
                for r in journey_result.results
            ])
        )

    # ICP results
    if not isinstance(results[3], Exception):
        icp_result = results[3]
        research_data["icp_search_results"] = (
            f"AI Answer: {icp_result.answer}\n\n"
            + "\n\n".join([
                f"**{r.title}**\n{r.content}\nSource: {r.url}"
                for r in icp_result.results
            ])
        )

    # Channel results
    if not isinstance(results[4], Exception):
        channel_result = results[4]
        research_data["channel_search_results"] = (
            f"AI Answer: {channel_result.answer}\n\n"
            + "\n\n".join([
                f"**{r.title}**\n{r.content}\nSource: {r.url}"
                for r in channel_result.results
            ])
        )

    # Industry-specific results (if applicable)
    if len(results) > 5 and not isinstance(results[5], Exception):
        industry_result = results[5]
        research_data["industry_search_results"] = (
            f"AI Answer: {industry_result.answer}\n\n"
            + "\n\n".join([
                f"**{r.title}**\n{r.content}\nSource: {r.url}"
                for r in industry_result.results
            ])
        )

    return research_data


async def _analyze_audience_with_llm(
    audience_input: AudienceBuildingInput,
    research_data: dict[str, Any],
    intelligence_context: dict[str, Any],
    client_context: dict[str, Any],
) -> AudienceBuildingResult:
    """
    Call LLM to analyze audience data and create personas.

    Args:
        audience_input: Original audience building input
        research_data: Raw research data from Tavily
        intelligence_context: Data from other intelligence agents
        client_context: Client context from state

    Returns:
        AudienceBuildingResult: Structured audience analysis
    """
    llm_service = LLMService.get_instance()
    prompt_manager = PromptManager.get_instance()

    # Get today's date for prompt
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Format intelligence context for the prompt
    market_context_str = ""
    if intelligence_context.get("market_research"):
        mr = intelligence_context["market_research"]
        market_context_str = f"""
Market Research Data:
- Industry: {mr.get('industry', 'N/A')}
- Market Overview: {mr.get('market_overview', 'N/A')}
- Key Trends: {', '.join([t.get('name', str(t)) if isinstance(t, dict) else str(t) for t in mr.get('trends', [])[:5]])}
- Target Segments: {', '.join([s.get('name', str(s)) if isinstance(s, dict) else str(s) for s in mr.get('target_segments', [])[:3]])}
- Opportunities: {', '.join([o.get('name', str(o)) if isinstance(o, dict) else str(o) for o in mr.get('opportunities', [])[:3]])}
"""

    seo_context_str = ""
    if intelligence_context.get("seo_data"):
        seo = intelligence_context["seo_data"]
        clusters = seo.get("keyword_clusters", [])
        cluster_names = [c.get("cluster_name", str(c)) if isinstance(c, dict) else str(c) for c in clusters[:5]]
        seo_context_str = f"""
SEO Intelligence Data:
- Primary Domain: {seo.get('primary_domain', 'N/A')}
- Top Keyword Clusters: {', '.join(cluster_names)}
- Search Intent Insights: {seo.get('search_intent_insights', 'N/A')}
"""

    web_context_str = ""
    if intelligence_context.get("web_analysis"):
        web = intelligence_context["web_analysis"]
        web_context_str = f"""
Web Analysis Data:
- Analyzed Domain: {web.get('analyzed_domain', 'N/A')}
- Analysis Summary: {web.get('analysis_summary', 'N/A')}
"""

    # Combine intelligence context
    combined_intelligence = f"{market_context_str}\n{seo_context_str}\n{web_context_str}".strip()

    # Format company sizes
    company_sizes_str = ", ".join([
        cs.value if hasattr(cs, 'value') else str(cs)
        for cs in audience_input.target_company_sizes
    ]) if audience_input.target_company_sizes else "Not specified"

    # Get and render the prompt template
    system_prompt, user_prompt = prompt_manager.get_full_prompt(
        "audience_builder",
        client_name=client_context.get("client_name", "Unknown"),
        product_category=audience_input.product_category,
        value_proposition=audience_input.value_proposition,
        target_market=audience_input.target_market,
        geographic_focus=audience_input.geographic_focus,
        target_company_sizes=company_sizes_str,
        target_industries=audience_input.target_industries,
        existing_customer_traits=audience_input.existing_customer_traits,
        known_pain_points=audience_input.known_pain_points,
        num_personas=audience_input.num_personas,
        additional_context=audience_input.additional_context,
        today_date=today,
        intelligence_context=combined_intelligence,
        sources_available=intelligence_context.get("sources_available", []),
        persona_search_results=research_data.get("persona_search_results"),
        pain_points_search_results=research_data.get("pain_points_search_results"),
        journey_search_results=research_data.get("journey_search_results"),
        icp_search_results=research_data.get("icp_search_results"),
        channel_search_results=research_data.get("channel_search_results"),
        industry_search_results=research_data.get("industry_search_results"),
    )

    # Call LLM with structured output
    analysis, usage = await llm_service.generate_structured(
        output_schema=AudienceBuildingResult,
        prompt=user_prompt,
        system_prompt=system_prompt,
    )

    logger.info(
        f"[Audience Builder] LLM analysis complete. "
        f"Personas: {len(analysis.buyer_personas)}, "
        f"Pain Points: {len(analysis.pain_point_analysis)}, "
        f"Journey Stages: {len(analysis.buying_journey)}, "
        f"Channels: {len(analysis.channel_strategy)}, "
        f"Tokens: {usage.total_tokens}"
    )

    return analysis


def _convert_audience_to_state_format(
    analysis: AudienceBuildingResult,
) -> dict[str, Any]:
    """
    Convert AudienceBuildingResult to the audience_data dict format for state.

    Args:
        analysis: LLM analysis result

    Returns:
        Dict matching audience_data state structure
    """
    return {
        "analysis_summary": analysis.analysis_summary,
        "ideal_customer_profile": {
            "summary": analysis.ideal_customer_profile.summary,
            "firmographics": {
                "company_sizes": [
                    cs.value if hasattr(cs, 'value') else str(cs)
                    for cs in analysis.ideal_customer_profile.firmographics.company_size
                ],
                "employee_count_range": analysis.ideal_customer_profile.firmographics.employee_count_range,
                "revenue_range": analysis.ideal_customer_profile.firmographics.revenue_range,
                "industries": analysis.ideal_customer_profile.firmographics.industries,
                "geographic_regions": analysis.ideal_customer_profile.firmographics.geographic_regions,
                "business_model": analysis.ideal_customer_profile.firmographics.business_model,
            },
            "technographics": {
                "current_tools": analysis.ideal_customer_profile.technographics.current_tools,
                "tech_maturity": analysis.ideal_customer_profile.technographics.tech_maturity,
                "integration_needs": analysis.ideal_customer_profile.technographics.integration_needs,
            },
            "behavioral": {
                "buying_triggers": analysis.ideal_customer_profile.behavioral.buying_triggers,
                "decision_timeline": analysis.ideal_customer_profile.behavioral.decision_timeline,
                "budget_characteristics": analysis.ideal_customer_profile.behavioral.budget_characteristics,
                "evaluation_criteria": analysis.ideal_customer_profile.behavioral.evaluation_criteria,
            },
            "key_pain_points": analysis.ideal_customer_profile.key_pain_points,
            "desired_outcomes": analysis.ideal_customer_profile.desired_outcomes,
            "disqualification_criteria": analysis.ideal_customer_profile.disqualification_criteria,
        },
        "personas": [
            {
                "persona_name": persona.persona_name,
                "persona_type": persona.persona_type.value if hasattr(persona.persona_type, 'value') else str(persona.persona_type),
                "one_liner": persona.one_liner,
                "demographics": {
                    "job_titles": persona.demographics.job_titles,
                    "seniority_level": persona.demographics.seniority_level,
                    "department": persona.demographics.department,
                    "age_range": persona.demographics.age_range,
                    "education": persona.demographics.education,
                    "experience_years": persona.demographics.experience_years,
                },
                "psychographics": {
                    "goals": persona.psychographics.goals,
                    "challenges": persona.psychographics.challenges,
                    "motivations": persona.psychographics.motivations,
                    "fears": persona.psychographics.fears,
                    "values": persona.psychographics.values,
                },
                "behavior": {
                    "information_sources": persona.behavior.information_sources,
                    "preferred_content_types": persona.behavior.preferred_content_types,
                    "social_platforms": persona.behavior.social_platforms,
                    "buying_role": persona.behavior.buying_role.value if hasattr(persona.behavior.buying_role, 'value') else str(persona.behavior.buying_role),
                    "decision_influence": persona.behavior.decision_influence,
                },
                "pain_points": persona.pain_points,
                "objections": persona.objections,
                "key_messages": persona.key_messages,
                "quotes": persona.quotes,
            }
            for persona in analysis.buyer_personas
        ],
        "pain_point_analysis": [
            {
                "pain_point": pp.pain_point,
                "description": pp.description,
                "affected_personas": pp.affected_personas,
                "severity": pp.severity.value if hasattr(pp.severity, 'value') else str(pp.severity),
                "frequency": pp.frequency,
                "current_solutions": pp.current_solutions,
                "our_solution": pp.our_solution,
            }
            for pp in analysis.pain_point_analysis
        ],
        "buying_journey": [
            {
                "stage": stage.stage.value if hasattr(stage.stage, 'value') else str(stage.stage),
                "description": stage.description,
                "buyer_goals": stage.buyer_goals,
                "buyer_questions": stage.buyer_questions,
                "buyer_emotions": stage.buyer_emotions,
                "touchpoints": [
                    {
                        "channel": tp.channel.value if hasattr(tp.channel, 'value') else str(tp.channel),
                        "content_type": tp.content_type,
                        "purpose": tp.purpose,
                        "key_message": tp.key_message,
                    }
                    for tp in stage.touchpoints
                ],
                "content_needs": stage.content_needs,
                "success_metrics": stage.success_metrics,
            }
            for stage in analysis.buying_journey
        ],
        "messaging_matrix": [
            {
                "persona_name": msg.persona_name,
                "value_proposition": msg.value_proposition,
                "key_benefits": msg.key_benefits,
                "proof_points": msg.proof_points,
                "call_to_action": msg.call_to_action,
                "tone": msg.tone,
                "words_to_use": msg.words_to_use,
                "words_to_avoid": msg.words_to_avoid,
            }
            for msg in analysis.messaging_matrix
        ],
        "channel_strategy": [
            {
                "channel": ch.channel.value if hasattr(ch.channel, 'value') else str(ch.channel),
                "priority": ch.priority.value if hasattr(ch.priority, 'value') else str(ch.priority),
                "target_personas": ch.target_personas,
                "journey_stages": [
                    js.value if hasattr(js, 'value') else str(js)
                    for js in ch.journey_stages
                ],
                "content_types": ch.content_types,
                "estimated_effectiveness": ch.estimated_effectiveness,
                "key_tactics": ch.key_tactics,
                "success_metrics": ch.success_metrics,
            }
            for ch in analysis.channel_strategy
        ],
        "quick_wins": analysis.quick_wins,
        "strategic_recommendations": analysis.strategic_recommendations,
        "data_sources_used": analysis.data_sources_used,
        "confidence_level": analysis.confidence_level.value if hasattr(analysis.confidence_level, 'value') else str(analysis.confidence_level),
        "data_limitations": analysis.data_limitations,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


def web_analysis_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Web Analysis Agent - Conducts comprehensive website analysis.

    Uses Tavily API and LLM to analyze:
    - Website structure and navigation
    - Technology stack (CMS, frameworks, hosting)
    - Performance metrics and Core Web Vitals
    - Content quality and UX design
    - Competitor website comparison
    - Security assessment

    This is a core Intelligence Squad agent (Phase 5.3).

    Input: Expects web analysis parameters in state["messages"] with type="web_analysis_input"
           or via workflow trigger payload.

    Output: Populates state["web_analysis"] with structured analysis.
    """
    logger.info("[Web Analysis] Conducting website analysis")

    # Extract web analysis input from state
    web_input: WebAnalysisInput | None = None

    # Look for web analysis input in messages
    for message in state.get("messages", []):
        if message.get("message_type") == "web_analysis_input":
            try:
                input_data = message.get("metadata", {})
                web_input = WebAnalysisInput(
                    target_urls=input_data.get("target_urls", []),
                    primary_domain=input_data.get("primary_domain"),
                    industry=input_data.get("industry"),
                    analysis_type=AnalysisType(input_data.get("analysis_type", "standard")),
                    include_competitors=input_data.get("include_competitors", True),
                    competitor_urls=input_data.get("competitor_urls", []),
                    focus_areas=input_data.get("focus_areas", []),
                    business_goals=input_data.get("business_goals", []),
                    additional_context=input_data.get("additional_context"),
                )
                break
            except Exception as e:
                logger.warning(f"[Web Analysis] Failed to parse web input: {e}")

    if not web_input:
        # No web input found - return error state
        logger.warning("[Web Analysis] No web input provided")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "web_analysis",
                    "error",
                    {"error": "No web input provided"}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="web_analysis",
                    content="Error: No web analysis input provided. Please provide target URLs or domain to analyze.",
                    message_type="error",
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Get client context
    client_context = state.get("client", {})

    try:
        # Run async research in sync context
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()

            # First conduct web research
            research_data = loop.run_until_complete(
                _conduct_web_research_with_tavily(web_input)
            )

            # Then analyze with LLM
            analysis = loop.run_until_complete(
                _analyze_web_with_llm(web_input, research_data, client_context)
            )
        except RuntimeError:
            # No event loop running, create a new one
            research_data = asyncio.run(
                _conduct_web_research_with_tavily(web_input)
            )
            analysis = asyncio.run(
                _analyze_web_with_llm(web_input, research_data, client_context)
            )

        # Convert to state format
        web_analysis_state = _convert_web_analysis_to_state_format(analysis)

        return {
            "web_analysis": web_analysis_state,
            "agent_execution_log": [
                _create_execution_log(
                    "web_analysis",
                    "analyze_web",
                    {
                        "primary_domain": web_input.primary_domain,
                        "target_urls": web_input.target_urls[:2] if web_input.target_urls else [],
                        "analysis_type": web_input.analysis_type.value if web_input.analysis_type else "standard",
                        "competitors_analyzed": len(analysis.competitor_profiles),
                        "quick_wins_count": len(analysis.quick_wins),
                        "confidence": analysis.confidence_level.value if hasattr(analysis.confidence_level, 'value') else analysis.confidence_level,
                    }
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="web_analysis",
                    content=f"Web analysis completed for {analysis.analyzed_domain or 'target'}: "
                            f"{len(analysis.competitor_profiles)} competitors analyzed, "
                            f"{len(analysis.quick_wins)} quick wins, "
                            f"{len(analysis.strategic_recommendations)} strategic recommendations",
                    message_type="info",
                    metadata={
                        "domain": analysis.analyzed_domain,
                        "analysis_summary": analysis.analysis_summary,
                        "confidence_level": analysis.confidence_level.value if hasattr(analysis.confidence_level, 'value') else analysis.confidence_level,
                        "quick_wins": analysis.quick_wins[:3] if analysis.quick_wins else [],
                    },
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except TavilyError as e:
        logger.error(f"[Web Analysis] Tavily error: {e}")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "web_analysis",
                    "error",
                    {"error": str(e), "type": "tavily_error"}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="web_analysis",
                    content=f"Error conducting web research: {e.message}",
                    message_type="error",
                )
            ],
            "error_message": f"Web Analysis Agent Error (Tavily): {e.message}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except LLMError as e:
        logger.error(f"[Web Analysis] LLM error: {e}")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "web_analysis",
                    "error",
                    {"error": str(e), "provider": e.provider.value if e.provider else None}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="web_analysis",
                    content=f"Error analyzing web data: {e.message}",
                    message_type="error",
                )
            ],
            "error_message": f"Web Analysis Agent Error (LLM): {e.message}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.exception(f"[Web Analysis] Unexpected error: {e}")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "web_analysis",
                    "error",
                    {"error": str(e)}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="web_analysis",
                    content=f"Unexpected error in web analysis: {str(e)}",
                    message_type="error",
                )
            ],
            "error_message": f"Web Analysis Agent Error: {str(e)}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


def audience_builder_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Audience Builder Agent - Creates comprehensive audience profiles and personas.

    Uses Tavily API and LLM to build:
    - Ideal Customer Profile (ICP)
    - Buyer Personas with detailed characteristics
    - Pain Point Analysis
    - Buying Journey Mapping
    - Messaging Matrix per persona
    - Channel Strategy recommendations

    This is a core Intelligence Squad agent (Phase 5.4).

    Key Feature: Synthesizes data from other intelligence agents:
    - market_research: Market trends, competitors, opportunities
    - seo_data: Keyword clusters, search intent, content gaps
    - web_analysis: Website insights, UX analysis, competitor web profiles

    Input: Expects audience building parameters in state["messages"] with type="audience_building_input"
           or via workflow trigger payload.

    Output: Populates state["audience_data"] with structured audience profiles.
    """
    logger.info("[Audience Builder] Building audience profiles and personas")

    # Extract audience building input from state
    audience_input: AudienceBuildingInput | None = None

    # Look for audience building input in messages
    for message in state.get("messages", []):
        if message.get("message_type") == "audience_building_input":
            try:
                input_data = message.get("metadata", {})

                # Parse company sizes
                target_company_sizes = []
                for size in input_data.get("target_company_sizes", []):
                    try:
                        target_company_sizes.append(CompanySize(size))
                    except ValueError:
                        pass

                audience_input = AudienceBuildingInput(
                    product_category=input_data.get("product_category", ""),
                    value_proposition=input_data.get("value_proposition"),
                    target_market=input_data.get("target_market"),
                    geographic_focus=input_data.get("geographic_focus"),
                    target_company_sizes=target_company_sizes,
                    target_industries=input_data.get("target_industries", []),
                    existing_customer_traits=input_data.get("existing_customer_traits", []),
                    known_pain_points=input_data.get("known_pain_points", []),
                    use_intelligence_data=input_data.get("use_intelligence_data", True),
                    num_personas=input_data.get("num_personas", 3),
                    additional_context=input_data.get("additional_context"),
                )
                break
            except Exception as e:
                logger.warning(f"[Audience Builder] Failed to parse audience input: {e}")

    if not audience_input:
        # No audience input found - return error state
        logger.warning("[Audience Builder] No audience input provided")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "audience_builder",
                    "error",
                    {"error": "No audience input provided"}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="audience_builder",
                    content="Error: No audience building input provided. Please provide product category and target market information.",
                    message_type="error",
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Get client context
    client_context = state.get("client", {})

    # Gather intelligence context from other agents
    intelligence_context = _gather_intelligence_context(state) if audience_input.use_intelligence_data else {
        "market_research": None,
        "seo_data": None,
        "web_analysis": None,
        "sources_available": [],
    }

    logger.info(
        f"[Audience Builder] Intelligence sources available: {intelligence_context['sources_available']}"
    )

    try:
        # Run async research in sync context
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()

            # First conduct audience research
            research_data = loop.run_until_complete(
                _conduct_audience_research_with_tavily(audience_input, intelligence_context)
            )

            # Then analyze with LLM
            analysis = loop.run_until_complete(
                _analyze_audience_with_llm(
                    audience_input, research_data, intelligence_context, client_context
                )
            )
        except RuntimeError:
            # No event loop running, create a new one
            research_data = asyncio.run(
                _conduct_audience_research_with_tavily(audience_input, intelligence_context)
            )
            analysis = asyncio.run(
                _analyze_audience_with_llm(
                    audience_input, research_data, intelligence_context, client_context
                )
            )

        # Convert to state format
        audience_data_state = _convert_audience_to_state_format(analysis)

        return {
            "audience_data": audience_data_state,
            "agent_execution_log": [
                _create_execution_log(
                    "audience_builder",
                    "build_audience",
                    {
                        "product_category": audience_input.product_category,
                        "target_market": audience_input.target_market,
                        "personas_created": len(analysis.buyer_personas),
                        "pain_points_identified": len(analysis.pain_point_analysis),
                        "journey_stages": len(analysis.buying_journey),
                        "channel_strategies": len(analysis.channel_strategy),
                        "intelligence_sources": intelligence_context.get("sources_available", []),
                        "confidence": analysis.confidence_level.value if hasattr(analysis.confidence_level, 'value') else str(analysis.confidence_level),
                    }
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="audience_builder",
                    content=f"Audience analysis completed for {audience_input.product_category}: "
                            f"{len(analysis.buyer_personas)} personas created, "
                            f"{len(analysis.pain_point_analysis)} pain points identified, "
                            f"{len(analysis.buying_journey)} journey stages mapped, "
                            f"{len(analysis.channel_strategy)} channel strategies recommended",
                    message_type="info",
                    metadata={
                        "product_category": audience_input.product_category,
                        "analysis_summary": analysis.analysis_summary,
                        "persona_names": [p.persona_name for p in analysis.buyer_personas],
                        "confidence_level": analysis.confidence_level.value if hasattr(analysis.confidence_level, 'value') else str(analysis.confidence_level),
                        "quick_wins": analysis.quick_wins[:3] if analysis.quick_wins else [],
                        "intelligence_sources": intelligence_context.get("sources_available", []),
                    },
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except TavilyError as e:
        logger.error(f"[Audience Builder] Tavily error: {e}")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "audience_builder",
                    "error",
                    {"error": str(e), "type": "tavily_error"}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="audience_builder",
                    content=f"Error conducting audience research: {e.message}",
                    message_type="error",
                )
            ],
            "error_message": f"Audience Builder Agent Error (Tavily): {e.message}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except LLMError as e:
        logger.error(f"[Audience Builder] LLM error: {e}")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "audience_builder",
                    "error",
                    {"error": str(e), "provider": e.provider.value if e.provider else None}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="audience_builder",
                    content=f"Error analyzing audience data: {e.message}",
                    message_type="error",
                )
            ],
            "error_message": f"Audience Builder Agent Error (LLM): {e.message}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.exception(f"[Audience Builder] Unexpected error: {e}")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "audience_builder",
                    "error",
                    {"error": str(e)}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="audience_builder",
                    content=f"Unexpected error in audience building: {str(e)}",
                    message_type="error",
                )
            ],
            "error_message": f"Audience Builder Agent Error: {str(e)}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


# =============================================================================
# CONTENT SQUAD NODES
# =============================================================================

def content_router_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Content Squad Router - Coordinates content agents.

    Manages content pipeline flow:
    Pipeline Manager -> Writer -> SEO -> Distribution -> Publishing
    """
    logger.info("[Content Router] Evaluating content pipeline")

    return {
        "agent_execution_log": [
            _create_execution_log("content_router", "evaluate_pipeline")
        ],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def pipeline_manager_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Pipeline Manager Agent - Manages editorial calendar.

    Responsibilities:
    - Content scheduling
    - Topic prioritization
    - Workflow coordination
    """
    logger.info("[Pipeline Manager] Managing content pipeline")

    # TODO: Implement actual pipeline management logic

    return {
        "content_pipeline": {
            "calendar": [],
            "active_drafts": [],
            "published_content": [],
            "performance_metrics": {},
        },
        "agent_execution_log": [
            _create_execution_log("pipeline_manager", "update_pipeline")
        ],
        "messages": [
            _create_agent_message(
                from_agent="pipeline_manager",
                content="Pipeline updated (stub)",
                message_type="info",
            )
        ],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def brandvoice_writer_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Brandvoice Writer Agent - Creates content in brand voice.

    Uses RAG to:
    - Retrieve brand voice guidelines
    - Generate content matching brand tone
    - Maintain consistency across content
    """
    logger.info("[Brandvoice Writer] Creating content")

    # TODO: Implement actual content writing logic with RAG

    return {
        "content_drafts": [],  # Will append new drafts
        "agent_execution_log": [
            _create_execution_log("brandvoice_writer", "create_content")
        ],
        "messages": [
            _create_agent_message(
                from_agent="brandvoice_writer",
                content="Content draft created (stub)",
                message_type="info",
            )
        ],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def seo_optimizer_node(state: OrchestratorState) -> dict[str, Any]:
    """
    SEO/GEO Optimizer Agent - Optimizes content for search.

    Optimizes:
    - Keywords integration
    - Meta descriptions
    - Header structure
    - Internal linking
    """
    logger.info("[SEO Optimizer] Optimizing content")

    # TODO: Implement actual SEO optimization logic

    return {
        "agent_execution_log": [
            _create_execution_log("seo_optimizer", "optimize_content")
        ],
        "messages": [
            _create_agent_message(
                from_agent="seo_optimizer",
                content="Content optimized for SEO (stub)",
                message_type="info",
            )
        ],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def social_distributor_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Social Media Distributor Agent - Adapts content for social platforms.

    Creates:
    - Platform-specific versions
    - Hashtag strategies
    - Posting schedules
    """
    logger.info("[Social Distributor] Preparing social content")

    # TODO: Implement actual social distribution logic

    return {
        "agent_execution_log": [
            _create_execution_log("social_distributor", "prepare_social")
        ],
        "messages": [
            _create_agent_message(
                from_agent="social_distributor",
                content="Social content prepared (stub)",
                message_type="info",
            )
        ],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def web_publisher_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Web Publisher Agent - Publishes content to web platforms.

    Handles:
    - CMS integration
    - Publishing workflow
    - Post-publish verification
    """
    logger.info("[Web Publisher] Publishing content")

    # TODO: Implement actual publishing logic
    # This requires human approval before execution

    return {
        "pending_approvals": [
            ApprovalRequest(
                approval_id=str(uuid4()),
                approval_type=ApprovalType.CONTENT_PUBLISH,
                title="Content Ready for Publishing",
                description="Content has been prepared and optimized. Awaiting approval to publish.",
                payload={"content_ids": []},  # Will contain actual content IDs
                requested_at=datetime.now(timezone.utc).isoformat(),
                requested_by="web_publisher",
                status="pending",
                reviewed_at=None,
                reviewed_by=None,
                rejection_reason=None,
            )
        ],
        "agent_execution_log": [
            _create_execution_log("web_publisher", "request_publish_approval")
        ],
        "messages": [
            _create_agent_message(
                from_agent="web_publisher",
                content="Publishing approval requested",
                message_type="request",
            )
        ],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# SALES OPS SQUAD NODES
# =============================================================================

def sales_ops_router_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Sales Ops Squad Router - Coordinates sales operations agents.

    Routes based on:
    - Available input data (meeting notes, leads, etc.)
    - Pending CRM tasks
    - Approval requirements
    """
    logger.info("[Sales Ops Router] Evaluating sales operations")

    return {
        "agent_execution_log": [
            _create_execution_log("sales_ops_router", "evaluate_operations")
        ],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


async def _analyze_meeting_with_llm(
    meeting_input: NormalizedMeetingInput,
    client_context: dict[str, Any],
) -> MeetingAnalysis:
    """
    Internal helper to call LLM for meeting analysis.

    Args:
        meeting_input: Normalized meeting input data
        client_context: Client context from state

    Returns:
        MeetingAnalysis: Structured analysis from LLM
    """
    llm_service = LLMService.get_instance()
    prompt_manager = PromptManager.get_instance()

    # Get and render the prompt template
    system_prompt, user_prompt = prompt_manager.get_full_prompt(
        "meeting_notes",
        client_name=client_context.get("client_name", "Unknown"),
        industry=client_context.get("industry"),
        crm_provider=client_context.get("crm_provider", "hubspot"),
        meeting_title=meeting_input.title,
        meeting_date=meeting_input.meeting_date.isoformat() if meeting_input.meeting_date else None,
        known_participants=meeting_input.participants if meeting_input.participants else None,
        deal_context=f"Deal ID: {meeting_input.deal_id}" if meeting_input.deal_id else None,
        transcript=meeting_input.transcript,
        additional_context=meeting_input.additional_context,
    )

    # Call LLM with structured output
    analysis, usage = await llm_service.generate_structured(
        output_schema=MeetingAnalysis,
        prompt=user_prompt,
        system_prompt=system_prompt,
    )

    logger.info(
        f"[Meeting Notes] LLM analysis complete. "
        f"Summary length: {len(analysis.summary)}, "
        f"Action items: {len(analysis.action_items)}, "
        f"Tokens: {usage.total_tokens}"
    )

    return analysis


def _convert_analysis_to_meeting_note(
    analysis: MeetingAnalysis,
    meeting_input: NormalizedMeetingInput,
) -> MeetingNote:
    """
    Convert MeetingAnalysis Pydantic model to MeetingNote TypedDict.

    Args:
        analysis: LLM analysis result
        meeting_input: Original meeting input

    Returns:
        MeetingNote: TypedDict for state storage
    """
    # Convert action items to the expected format
    action_items = [
        {
            "task": item.task,
            "assignee": item.assignee or "TBD",
            "due_date": item.due_date or "TBD",
        }
        for item in analysis.action_items
    ]

    # Map sentiment enum to string
    sentiment_map = {
        MeetingSentiment.POSITIVE: "positive",
        MeetingSentiment.NEUTRAL: "neutral",
        MeetingSentiment.NEGATIVE: "negative",
    }

    return MeetingNote(
        meeting_id=str(uuid4()),
        date=meeting_input.meeting_date.isoformat() if meeting_input.meeting_date else datetime.now(timezone.utc).isoformat(),
        participants=analysis.identified_participants or meeting_input.participants,
        summary=analysis.summary,
        key_points=analysis.key_points,
        action_items=action_items,
        follow_up_required=analysis.follow_up_required,
        sentiment=sentiment_map.get(analysis.overall_sentiment, "neutral"),
        deal_stage_update=analysis.deal_stage_recommendation.recommended_stage if analysis.deal_stage_recommendation else None,
    )


def meeting_notes_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Meeting Notes Analyzer Agent - Analyzes meeting transcripts.

    Extracts:
    - Meeting summary
    - Key discussion points
    - Action items with assignees
    - Follow-up requirements
    - Sentiment analysis
    - Deal stage recommendations

    This is a core MVP agent (Phase 3.1).

    Input: Expects meeting transcript in state["messages"] with type="meeting_input"
           or via workflow trigger payload.

    Output: Populates state["meeting_notes"] with structured analysis.
    """
    logger.info("[Meeting Notes] Analyzing meeting transcript")

    # Extract meeting input from state
    # The meeting input can come from:
    # 1. Workflow trigger payload (via messages with type="meeting_input")
    # 2. Direct state input (for future integrations)
    meeting_input: NormalizedMeetingInput | None = None

    # Look for meeting input in messages
    for message in state.get("messages", []):
        if message.get("message_type") == "meeting_input":
            try:
                input_data = message.get("metadata", {})
                meeting_input = NormalizedMeetingInput(
                    transcript=input_data.get("transcript", ""),
                    title=input_data.get("title"),
                    meeting_date=datetime.fromisoformat(input_data["meeting_date"]) if input_data.get("meeting_date") else None,
                    participants=input_data.get("participants", []),
                    deal_id=input_data.get("deal_id"),
                    contact_id=input_data.get("contact_id"),
                    additional_context=input_data.get("additional_context"),
                )
                break
            except Exception as e:
                logger.warning(f"[Meeting Notes] Failed to parse meeting input: {e}")

    if not meeting_input:
        # No meeting input found - return error state
        logger.warning("[Meeting Notes] No meeting transcript provided")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "meeting_notes",
                    "error",
                    {"error": "No meeting transcript provided"}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="meeting_notes",
                    content="Error: No meeting transcript provided. Please provide a transcript to analyze.",
                    message_type="error",
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Get client context
    client_context = state.get("client", {})

    try:
        # Run async LLM analysis in sync context
        # Handle both cases: already running event loop (test/LangGraph) or not
        try:
            loop = asyncio.get_running_loop()
            # If we get here, a loop is running - use nest_asyncio or schedule
            import nest_asyncio
            nest_asyncio.apply()
            analysis = loop.run_until_complete(
                _analyze_meeting_with_llm(meeting_input, client_context)
            )
        except RuntimeError:
            # No event loop running, create a new one
            analysis = asyncio.run(
                _analyze_meeting_with_llm(meeting_input, client_context)
            )

        # Convert to MeetingNote TypedDict
        meeting_note = _convert_analysis_to_meeting_note(analysis, meeting_input)

        return {
            "meeting_notes": [meeting_note],
            "agent_execution_log": [
                _create_execution_log(
                    "meeting_notes",
                    "analyze_transcript",
                    {
                        "meeting_id": meeting_note["meeting_id"],
                        "summary_length": len(meeting_note["summary"]),
                        "action_items_count": len(meeting_note["action_items"]),
                        "sentiment": meeting_note["sentiment"],
                        "follow_up_required": meeting_note["follow_up_required"],
                    }
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="meeting_notes",
                    content=f"Meeting analyzed: {len(meeting_note['action_items'])} action items extracted",
                    message_type="info",
                    metadata={
                        "meeting_id": meeting_note["meeting_id"],
                        "sentiment": meeting_note["sentiment"],
                        "follow_up_required": meeting_note["follow_up_required"],
                    },
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except LLMError as e:
        logger.error(f"[Meeting Notes] LLM error: {e}")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "meeting_notes",
                    "error",
                    {"error": str(e), "provider": e.provider.value if e.provider else None}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="meeting_notes",
                    content=f"Error analyzing meeting: {e.message}",
                    message_type="error",
                )
            ],
            "error_message": f"Meeting Notes Agent Error: {e.message}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.exception(f"[Meeting Notes] Unexpected error: {e}")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "meeting_notes",
                    "error",
                    {"error": str(e)}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="meeting_notes",
                    content=f"Unexpected error analyzing meeting: {str(e)}",
                    message_type="error",
                )
            ],
            "error_message": f"Meeting Notes Agent Error: {str(e)}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


async def _extract_tasks_with_llm(
    meeting_note: MeetingNote,
    client_context: dict[str, Any],
    default_due_days: int = 7,
) -> TaskExtractionResult:
    """
    Internal helper to call LLM for task extraction.

    Args:
        meeting_note: Meeting note with action items
        client_context: Client context from state
        default_due_days: Default days until due if not specified

    Returns:
        TaskExtractionResult: Structured task extraction result
    """
    llm_service = LLMService.get_instance()
    prompt_manager = PromptManager.get_instance()

    # Convert action items to ActionItemInput format
    action_items = [
        {
            "task": item.get("task", ""),
            "assignee": item.get("assignee"),
            "due_date": item.get("due_date"),
            "priority": "medium",  # Default from meeting notes
            "context": None,
        }
        for item in meeting_note.get("action_items", [])
    ]

    if not action_items:
        return TaskExtractionResult(
            tasks=[],
            skipped_items=[],
            total_action_items=0,
            tasks_created=0,
            tasks_needing_review=0,
            processing_notes="No action items found in meeting notes",
        )

    # Get today's date for due date calculation
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Get and render the prompt template
    system_prompt, user_prompt = prompt_manager.get_full_prompt(
        "task_extractor",
        client_name=client_context.get("client_name", "Unknown"),
        crm_provider=client_context.get("crm_provider", "hubspot"),
        today_date=today,
        meeting_title=None,  # Could be added to MeetingNote in future
        meeting_date=meeting_note.get("date"),
        meeting_summary=meeting_note.get("summary"),
        action_items=action_items,
        deal_id=None,  # Could be extracted from meeting context
        contact_id=None,
        company_id=None,
        team_members=None,  # Could be fetched from CRM
        deal_context=meeting_note.get("deal_stage_update"),
        default_due_days=default_due_days,
    )

    # Call LLM with structured output
    result, usage = await llm_service.generate_structured(
        output_schema=TaskExtractionResult,
        prompt=user_prompt,
        system_prompt=system_prompt,
    )

    logger.info(
        f"[Task Extractor] LLM extraction complete. "
        f"Tasks created: {result.tasks_created}, "
        f"Skipped: {len(result.skipped_items)}, "
        f"Tokens: {usage.total_tokens}"
    )

    return result


def _convert_extracted_task_to_crm_task(
    extracted_task: ExtractedTask,
    meeting_id: str,
) -> CRMTask:
    """
    Convert ExtractedTask Pydantic model to CRMTask TypedDict.

    Args:
        extracted_task: Extracted task from LLM
        meeting_id: Source meeting ID

    Returns:
        CRMTask: TypedDict for state storage and CRM creation
    """
    # Map TaskPriority enum to CRMTask priority
    priority_map = {
        TaskPriority.LOW: "low",
        TaskPriority.MEDIUM: "medium",
        TaskPriority.HIGH: "high",
    }

    # Build HubSpot task payload
    hubspot_payload = {
        "hs_task_subject": extracted_task.subject,
        "hs_task_body": extracted_task.body or "",
        "hs_task_status": extracted_task.status.value if extracted_task.status else "NOT_STARTED",
        "hs_task_priority": extracted_task.priority.value if extracted_task.priority else "MEDIUM",
        "hs_task_type": extracted_task.task_type.value if extracted_task.task_type else "TODO",
    }

    # Add due date if present (convert to Unix timestamp milliseconds)
    if extracted_task.due_date:
        try:
            due_datetime = datetime.strptime(extracted_task.due_date, "%Y-%m-%d")
            due_datetime = due_datetime.replace(hour=17, minute=0)  # 5 PM
            hubspot_payload["hs_timestamp"] = str(int(due_datetime.timestamp() * 1000))
        except ValueError:
            logger.warning(f"[Task Extractor] Invalid due date format: {extracted_task.due_date}")

    # Add owner if present
    if extracted_task.hubspot_owner_id:
        hubspot_payload["hubspot_owner_id"] = extracted_task.hubspot_owner_id

    return CRMTask(
        task_id=str(uuid4()),
        task_type="create_task",
        entity_type="task",
        entity_id=None,  # Will be set after CRM creation
        payload={
            "hubspot_task": hubspot_payload,
            "source_meeting_id": meeting_id,
            "assignee_name": extracted_task.assignee_name,
            "assignee_email": extracted_task.assignee_email,
            "associations": [
                {
                    "type": assoc.association_type.value,
                    "id": assoc.entity_id,
                    "name": assoc.entity_name,
                }
                for assoc in extracted_task.associations
            ] if extracted_task.associations else [],
            "source_action_item": extracted_task.source_action_item,
            "extraction_confidence": extracted_task.extraction_confidence,
            "needs_review": extracted_task.needs_review,
            "review_reason": extracted_task.review_reason,
        },
        priority=priority_map.get(extracted_task.priority, "medium"),
        status="pending",
        error_message=None,
        created_at=datetime.now(timezone.utc).isoformat(),
        executed_at=None,
    )


def task_extractor_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Task Extractor Agent - Transforms meeting action items into CRM-ready tasks.

    Creates CRM tasks from:
    - Action items in meeting notes
    - Follow-up requirements
    - Commitments made during meeting

    Uses LLM to:
    - Enrich task descriptions with context
    - Calculate realistic due dates
    - Set appropriate priorities
    - Flag tasks needing review

    This is a core MVP agent (Phase 3.2).

    Input: Reads from state["meeting_notes"]
    Output: Populates state["crm_tasks"] with structured tasks
    """
    logger.info("[Task Extractor] Processing meeting notes for task extraction")

    # Get meeting notes from state
    meeting_notes = state.get("meeting_notes", [])

    if not meeting_notes:
        logger.warning("[Task Extractor] No meeting notes found in state")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "task_extractor",
                    "no_input",
                    {"error": "No meeting notes to process"}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="task_extractor",
                    content="No meeting notes found to extract tasks from",
                    message_type="info",
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Get client context
    client_context = state.get("client", {})

    all_crm_tasks: list[CRMTask] = []
    total_action_items = 0
    total_tasks_created = 0
    total_needing_review = 0
    all_skipped_items: list[dict] = []

    for meeting_note in meeting_notes:
        action_items = meeting_note.get("action_items", [])
        if not action_items:
            logger.info(f"[Task Extractor] Skipping meeting {meeting_note.get('meeting_id')} - no action items")
            continue

        total_action_items += len(action_items)

        try:
            # Run async LLM extraction
            try:
                loop = asyncio.get_running_loop()
                import nest_asyncio
                nest_asyncio.apply()
                extraction_result = loop.run_until_complete(
                    _extract_tasks_with_llm(meeting_note, client_context)
                )
            except RuntimeError:
                extraction_result = asyncio.run(
                    _extract_tasks_with_llm(meeting_note, client_context)
                )

            # Convert extracted tasks to CRMTask format
            meeting_id = meeting_note.get("meeting_id", str(uuid4()))
            for extracted_task in extraction_result.tasks:
                crm_task = _convert_extracted_task_to_crm_task(extracted_task, meeting_id)
                all_crm_tasks.append(crm_task)

            total_tasks_created += extraction_result.tasks_created
            total_needing_review += extraction_result.tasks_needing_review
            all_skipped_items.extend(extraction_result.skipped_items)

        except LLMError as e:
            logger.error(f"[Task Extractor] LLM error for meeting {meeting_note.get('meeting_id')}: {e}")
            # Fallback to basic extraction without LLM enrichment
            for action_item in action_items:
                fallback_task = CRMTask(
                    task_id=str(uuid4()),
                    task_type="create_task",
                    entity_type="task",
                    entity_id=None,
                    payload={
                        "hubspot_task": {
                            "hs_task_subject": action_item.get("task", ""),
                            "hs_task_body": f"From meeting: {meeting_note.get('summary', '')}",
                            "hs_task_status": "NOT_STARTED",
                            "hs_task_priority": "MEDIUM",
                            "hs_task_type": "TODO",
                        },
                        "source_meeting_id": meeting_note.get("meeting_id"),
                        "assignee_name": action_item.get("assignee"),
                        "needs_review": True,
                        "review_reason": "LLM extraction failed - basic extraction used",
                    },
                    priority="medium",
                    status="pending",
                    error_message=None,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    executed_at=None,
                )
                all_crm_tasks.append(fallback_task)
                total_tasks_created += 1
                total_needing_review += 1

        except Exception as e:
            logger.exception(f"[Task Extractor] Unexpected error: {e}")
            # Continue processing other meetings

    # Prepare result message
    result_message = (
        f"Extracted {total_tasks_created} tasks from {len(meeting_notes)} meeting(s). "
        f"Action items processed: {total_action_items}. "
        f"Tasks needing review: {total_needing_review}. "
        f"Skipped items: {len(all_skipped_items)}."
    )

    return {
        "crm_tasks": all_crm_tasks,
        "agent_execution_log": [
            _create_execution_log(
                "task_extractor",
                "extract_tasks",
                {
                    "meetings_processed": len(meeting_notes),
                    "total_action_items": total_action_items,
                    "tasks_created": total_tasks_created,
                    "tasks_needing_review": total_needing_review,
                    "skipped_items": len(all_skipped_items),
                }
            )
        ],
        "messages": [
            _create_agent_message(
                from_agent="task_extractor",
                content=result_message,
                message_type="info",
                metadata={
                    "tasks_created": total_tasks_created,
                    "tasks_needing_review": total_needing_review,
                    "skipped_items": all_skipped_items,
                },
            )
        ],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


async def _prepare_crm_updates_with_llm(
    crm_tasks: list[CRMTask],
    meeting_notes: list[MeetingNote],
    client_context: dict[str, Any],
    workflow_id: str,
) -> CRMUpdateOperationResult:
    """
    Internal helper to call LLM for CRM update preparation.

    Args:
        crm_tasks: Pending CRM tasks to process
        meeting_notes: Related meeting notes for context
        client_context: Client context from state
        workflow_id: Current workflow ID

    Returns:
        CRMUpdateOperationResult: Structured operations from LLM
    """
    llm_service = LLMService.get_instance()
    prompt_manager = PromptManager.get_instance()

    # Get today's date
    today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Extract meeting summary if available
    meeting_summary = None
    deal_stage_recommendation = None
    if meeting_notes:
        latest_meeting = meeting_notes[-1]
        meeting_summary = latest_meeting.get("summary")
        if latest_meeting.get("deal_stage_update"):
            deal_stage_recommendation = {
                "recommended_stage": latest_meeting.get("deal_stage_update"),
                "current_signals": latest_meeting.get("key_points", [])[:3],
                "confidence": "medium",
                "reasoning": f"Based on meeting sentiment: {latest_meeting.get('sentiment', 'neutral')}",
            }

    # Format tasks for prompt
    tasks_for_prompt = [
        {
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "entity_type": task["entity_type"],
            "priority": task["priority"],
            "status": task["status"],
            "payload": task["payload"],
        }
        for task in crm_tasks
    ]

    # Get and render the prompt template
    system_prompt, user_prompt = prompt_manager.get_full_prompt(
        "crm_updater",
        client_name=client_context.get("client_name", "Unknown"),
        crm_provider=client_context.get("crm_provider", "hubspot"),
        workflow_id=workflow_id,
        today_date=today_date,
        tasks=tasks_for_prompt,
        meeting_summary=meeting_summary,
        deal_stage_recommendation=deal_stage_recommendation,
        contact_ids=None,  # Could be extracted from task associations
        deal_ids=None,
    )

    # Call LLM with structured output
    result, usage = await llm_service.generate_structured(
        output_schema=CRMUpdateOperationResult,
        prompt=user_prompt,
        system_prompt=system_prompt,
    )

    logger.info(
        f"[CRM Updater] LLM preparation complete. "
        f"Operations: {result.total_operations}, "
        f"High-risk: {result.high_risk_count}, "
        f"Tokens: {usage.total_tokens}"
    )

    return result


def _create_approval_from_operations(
    operations_result: CRMUpdateOperationResult,
    workflow_id: str,
) -> ApprovalRequest:
    """
    Create an approval request from CRM operations.

    Args:
        operations_result: LLM-generated operations
        workflow_id: Current workflow ID

    Returns:
        ApprovalRequest: Structured approval request for HITL
    """
    # Determine priority based on high-risk count
    if operations_result.high_risk_count > 0:
        description_prefix = f"⚠️ HIGH RISK: {operations_result.high_risk_count} high-risk operation(s). "
    else:
        description_prefix = ""

    # Build detailed description
    operation_summaries = [op.summary for op in operations_result.operations[:5]]
    if len(operations_result.operations) > 5:
        operation_summaries.append(f"... and {len(operations_result.operations) - 5} more")

    return ApprovalRequest(
        approval_id=str(uuid4()),
        approval_type=ApprovalType.CRM_UPDATE,
        title=f"CRM Update: {operations_result.total_operations} operations pending",
        description=f"{description_prefix}{operations_result.batch_summary}",
        payload={
            "operations": [
                {
                    "operation_id": op.operation_id,
                    "operation_type": op.operation_type.value if isinstance(op.operation_type, CRMOperationType) else op.operation_type,
                    "summary": op.summary,
                    "details": op.details,
                    "payload": op.payload,
                    "risk_level": op.risk_level.value if isinstance(op.risk_level, OperationRiskLevel) else op.risk_level,
                    "risk_factors": op.risk_factors,
                    "approval_required": op.approval_required,
                    "auto_approve_eligible": op.auto_approve_eligible,
                    "rollback_info": op.rollback_info,
                    "source_task_id": op.source_task_id,
                }
                for op in operations_result.operations
            ],
            "batch_summary": operations_result.batch_summary,
            "total_operations": operations_result.total_operations,
            "high_risk_count": operations_result.high_risk_count,
            "deal_stage_changes": operations_result.deal_stage_changes,
            "notes_to_add": operations_result.notes_to_add,
            "warnings": operations_result.warnings,
            "workflow_id": workflow_id,
        },
        requested_at=datetime.now(timezone.utc).isoformat(),
        requested_by="crm_updater",
        status="pending",
        reviewed_at=None,
        reviewed_by=None,
        rejection_reason=None,
    )


def _fallback_approval_from_tasks(
    crm_tasks: list[CRMTask],
    workflow_id: str,
    error_message: str,
) -> ApprovalRequest:
    """
    Create a fallback approval request when LLM fails.

    Args:
        crm_tasks: Original CRM tasks
        workflow_id: Current workflow ID
        error_message: Error that caused fallback

    Returns:
        ApprovalRequest: Basic approval request without LLM enhancement
    """
    return ApprovalRequest(
        approval_id=str(uuid4()),
        approval_type=ApprovalType.CRM_UPDATE,
        title=f"CRM Update: {len(crm_tasks)} tasks (LLM fallback)",
        description=f"⚠️ LLM processing failed. Please review raw task data. Error: {error_message}",
        payload={
            "tasks": [
                {
                    "task_id": task["task_id"],
                    "task_type": task["task_type"],
                    "entity_type": task["entity_type"],
                    "payload": task["payload"],
                    "priority": task["priority"],
                }
                for task in crm_tasks
            ],
            "fallback_mode": True,
            "error": error_message,
            "workflow_id": workflow_id,
        },
        requested_at=datetime.now(timezone.utc).isoformat(),
        requested_by="crm_updater",
        status="pending",
        reviewed_at=None,
        reviewed_by=None,
        rejection_reason=None,
    )


def crm_updater_node(state: OrchestratorState) -> dict[str, Any]:
    """
    CRM Updater Agent - Prepares CRM updates for human approval.

    This is a core MVP agent (Phase 3.3).

    Workflow:
    1. Extract pending CRM tasks from state
    2. Call LLM to transform tasks into structured CRM operations
    3. Assess risk levels for each operation
    4. Create approval request with detailed payloads
    5. Return approval request for HITL review

    Input: Reads from state["crm_tasks"] (pending status)
    Output: Populates state["pending_approvals"] with structured approval request

    Features:
    - LLM-powered operation formatting
    - Risk assessment per ADR-014
    - Deal stage change recommendations
    - Meeting notes integration as CRM notes
    - Fallback to basic approval if LLM fails
    """
    logger.info("[CRM Updater] Preparing CRM updates for approval")

    workflow_id = state.get("workflow_id", str(uuid4()))

    # Get pending CRM tasks
    crm_tasks = [
        task for task in state.get("crm_tasks", [])
        if task.get("status") == "pending"
    ]

    if not crm_tasks:
        logger.info("[CRM Updater] No pending CRM tasks to process")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "crm_updater",
                    "no_pending_tasks",
                    {"message": "No pending CRM tasks found"}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="crm_updater",
                    content="No pending CRM tasks to process. Skipping CRM update preparation.",
                    message_type="info",
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Get meeting notes for context
    meeting_notes = state.get("meeting_notes", [])
    client_context = state.get("client", {})

    # Prepare approval requests
    approval_requests: list[ApprovalRequest] = []

    try:
        # Run async LLM preparation
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
            operations_result = loop.run_until_complete(
                _prepare_crm_updates_with_llm(
                    crm_tasks, meeting_notes, client_context, workflow_id
                )
            )
        except RuntimeError:
            operations_result = asyncio.run(
                _prepare_crm_updates_with_llm(
                    crm_tasks, meeting_notes, client_context, workflow_id
                )
            )

        # Create approval request from LLM result
        if operations_result.operations:
            approval_request = _create_approval_from_operations(
                operations_result, workflow_id
            )
            approval_requests.append(approval_request)

            logger.info(
                f"[CRM Updater] Created approval request with {operations_result.total_operations} operations "
                f"({operations_result.high_risk_count} high-risk)"
            )

        return {
            "pending_approvals": approval_requests,
            "agent_execution_log": [
                _create_execution_log(
                    "crm_updater",
                    "prepare_updates",
                    {
                        "pending_tasks": len(crm_tasks),
                        "operations_prepared": operations_result.total_operations,
                        "high_risk_count": operations_result.high_risk_count,
                        "deal_stage_changes": len(operations_result.deal_stage_changes),
                        "notes_to_add": len(operations_result.notes_to_add),
                        "warnings": operations_result.warnings,
                    }
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="crm_updater",
                    content=(
                        f"Prepared {operations_result.total_operations} CRM operations for approval. "
                        f"{operations_result.batch_summary}"
                    ),
                    message_type="request",
                    metadata={
                        "total_operations": operations_result.total_operations,
                        "high_risk_count": operations_result.high_risk_count,
                        "warnings": operations_result.warnings,
                    },
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except LLMError as e:
        logger.error(f"[CRM Updater] LLM error: {e}")
        # Fallback to basic approval without LLM enhancement
        approval_request = _fallback_approval_from_tasks(
            crm_tasks, workflow_id, str(e.message)
        )
        approval_requests.append(approval_request)

        return {
            "pending_approvals": approval_requests,
            "agent_execution_log": [
                _create_execution_log(
                    "crm_updater",
                    "fallback_mode",
                    {
                        "error": str(e),
                        "pending_tasks": len(crm_tasks),
                        "provider": e.provider.value if e.provider else None,
                    }
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="crm_updater",
                    content=f"LLM error - using fallback mode. {len(crm_tasks)} tasks prepared for manual review.",
                    message_type="error",
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.exception(f"[CRM Updater] Unexpected error: {e}")
        # Fallback to basic approval
        approval_request = _fallback_approval_from_tasks(
            crm_tasks, workflow_id, str(e)
        )
        approval_requests.append(approval_request)

        return {
            "pending_approvals": approval_requests,
            "agent_execution_log": [
                _create_execution_log(
                    "crm_updater",
                    "error",
                    {"error": str(e)}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="crm_updater",
                    content=f"Unexpected error - using fallback mode. {len(crm_tasks)} tasks prepared for manual review.",
                    message_type="error",
                )
            ],
            "error_message": f"CRM Updater Error: {str(e)}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


async def _research_lead_with_tavily(
    lead_input: LeadResearchInput,
) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    """
    Internal helper to fetch research data from Tavily.

    Args:
        lead_input: Lead research input with company/contact info

    Returns:
        Tuple of (company_search_results, news_results, competitor_results)
    """
    tavily_service = TavilyService.get_instance()

    if not tavily_service.is_configured:
        logger.warning("[Lead Research] Tavily not configured, skipping web search")
        return {}, [], None

    company_name = lead_input.company_name
    research_depth = lead_input.research_depth

    # Determine search depth based on research_depth setting
    tavily_depth = SearchDepth.ADVANCED if research_depth == ResearchDepth.DEEP else SearchDepth.BASIC

    tasks = []

    # 1. Company overview search
    tasks.append(
        tavily_service.search(
            f"{company_name} company overview about",
            search_depth=tavily_depth,
            max_results=5,
            include_answer=True,
        )
    )

    # 2. Recent news search
    tasks.append(
        tavily_service.search_news(
            company_name,
            days=90,
            max_results=5,
        )
    )

    # 3. Competitor search (only for standard+ depth)
    competitor_task = None
    if research_depth in [ResearchDepth.STANDARD, ResearchDepth.DEEP]:
        competitor_task = tavily_service.search(
            f"{company_name} competitors alternatives",
            search_depth=SearchDepth.BASIC,
            max_results=3,
            include_answer=True,
        )
        tasks.append(competitor_task)

    # Execute all searches in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    company_search = results[0] if not isinstance(results[0], Exception) else None
    news_search = results[1] if not isinstance(results[1], Exception) else None
    competitor_search = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else None

    # Format company search results
    company_search_results = {}
    if company_search:
        company_search_results = {
            "answer": company_search.answer,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "content": r.content,
                    "score": r.score,
                }
                for r in company_search.results
            ],
        }

    # Format news results
    news_results = []
    if news_search and news_search.results:
        news_results = [
            {
                "title": r.title,
                "url": r.url,
                "content": r.content,
                "published_date": r.published_date,
            }
            for r in news_search.results
        ]

    # Format competitor results
    competitor_results = None
    if competitor_search:
        competitor_results = competitor_search.answer

    return company_search_results, news_results, competitor_results


async def _analyze_lead_with_llm(
    lead_input: LeadResearchInput,
    company_search_results: dict[str, Any],
    news_results: list[dict[str, Any]],
    competitor_results: str | None,
    client_context: dict[str, Any],
) -> LeadResearchResult:
    """
    Internal helper to analyze research data with LLM.

    Args:
        lead_input: Original lead research input
        company_search_results: Results from company overview search
        news_results: Recent news articles
        competitor_results: Competitor analysis text
        client_context: Client context from state

    Returns:
        LeadResearchResult: Structured lead research result
    """
    llm_service = LLMService.get_instance()
    prompt_manager = PromptManager.get_instance()

    today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Format company search results for prompt
    company_search_text = ""
    if company_search_results:
        if company_search_results.get("answer"):
            company_search_text += f"**AI Summary:** {company_search_results['answer']}\n\n"
        for result in company_search_results.get("results", []):
            company_search_text += f"- **{result['title']}**\n  {result['content'][:300]}...\n  Source: {result['url']}\n\n"

    # Get and render the prompt template
    system_prompt, user_prompt = prompt_manager.get_full_prompt(
        "lead_research",
        client_name=client_context.get("client_name", "Unknown"),
        crm_provider=client_context.get("crm_provider", "hubspot"),
        today_date=today_date,
        research_depth=lead_input.research_depth.value,
        focus_areas=lead_input.focus_areas,
        additional_context=lead_input.additional_context,
        company_name=lead_input.company_name,
        company_domain=lead_input.company_domain,
        contact_name=lead_input.contact_name,
        contact_title=lead_input.contact_title,
        company_search_results=company_search_text,
        news_results=news_results,
        competitor_results=competitor_results,
        existing_crm_data=lead_input.existing_crm_data,
    )

    # Call LLM with structured output
    result, _ = await llm_service.generate_structured(
        output_schema=LeadResearchResult,
        prompt=user_prompt,
        system_prompt=system_prompt,
    )

    return result


def _convert_research_to_lead_data(
    lead_input: LeadResearchInput,
    research_result: LeadResearchResult,
) -> LeadData:
    """
    Convert LeadResearchResult to LeadData TypedDict for state.

    Args:
        lead_input: Original input
        research_result: LLM analysis result

    Returns:
        LeadData: State-compatible lead data
    """
    # Calculate lead score from qualification
    score_map = {
        LeadQualificationScore.HOT: 90.0,
        LeadQualificationScore.WARM: 60.0,
        LeadQualificationScore.COLD: 35.0,
        LeadQualificationScore.UNQUALIFIED: 10.0,
    }
    lead_score = score_map.get(research_result.qualification_score, 50.0)

    # Build enrichment data
    enrichment_data = {
        "description": research_result.company.description,
        "industry": research_result.company.industry,
        "employee_count": research_result.company.employee_count,
        "headquarters": research_result.company.headquarters,
        "founded_year": research_result.company.founded_year,
        "funding": research_result.funding.model_dump() if research_result.funding else None,
        "key_people": [p.model_dump() for p in research_result.key_people],
        "technology": research_result.technology.model_dump() if research_result.technology else None,
        "business_signals": research_result.business_signals,
        "pain_points": research_result.pain_points,
        "opportunities": research_result.opportunities,
        "talking_points": research_result.talking_points,
        "qualification_score": research_result.qualification_score.value,
        "qualification_reasoning": research_result.qualification_reasoning,
        "sources": research_result.sources_used,
    }

    # Find contact from key people if not provided
    contact_name = lead_input.contact_name
    contact_email = lead_input.contact_email
    linkedin_url = lead_input.contact_linkedin

    if not contact_name and research_result.key_people:
        # Use first key person as primary contact
        first_person = research_result.key_people[0]
        contact_name = first_person.name
        linkedin_url = first_person.linkedin_url

    return LeadData(
        lead_id=str(uuid4()),
        company_name=research_result.company.name,
        contact_name=contact_name,
        email=contact_email,
        phone=None,  # Usually not available from web research
        linkedin_url=linkedin_url,
        company_size=research_result.company.employee_count,
        industry=research_result.company.industry,
        enrichment_data=enrichment_data,
        lead_score=lead_score,
        source=lead_input.source.value,
    )


def _fallback_lead_data(
    lead_input: LeadResearchInput,
    error_message: str,
) -> LeadData:
    """
    Create minimal lead data when research fails.

    Args:
        lead_input: Original input
        error_message: Error that occurred

    Returns:
        LeadData: Basic lead data without enrichment
    """
    return LeadData(
        lead_id=str(uuid4()),
        company_name=lead_input.company_name,
        contact_name=lead_input.contact_name,
        email=lead_input.contact_email,
        phone=None,
        linkedin_url=lead_input.contact_linkedin,
        company_size=None,
        industry=None,
        enrichment_data={
            "error": error_message,
            "research_failed": True,
            "input_domain": lead_input.company_domain,
        },
        lead_score=None,
        source=lead_input.source.value,
    )


def lead_research_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Lead Research Agent - Researches and enriches leads using Tavily.

    This agent:
    1. Fetches web research data from Tavily (company overview, news, competitors)
    2. Analyzes data with LLM to extract structured intelligence
    3. Generates sales insights (business signals, pain points, talking points)
    4. Qualifies leads based on available information
    5. Prepares enriched lead data for CRM integration

    Input: Expects lead research request in state["messages"] with type="lead_research_input"
           containing company_name and optional contact info.

    Output: Populates state["leads"] with enriched lead data.

    This is a Phase 4.2 agent.
    """
    logger.info("[Lead Research] Starting lead research")

    # Extract lead research input from messages
    lead_input: LeadResearchInput | None = None

    for message in state.get("messages", []):
        if message.get("message_type") == "lead_research_input":
            try:
                input_data = message.get("metadata", {})
                lead_input = LeadResearchInput(
                    company_name=input_data.get("company_name", ""),
                    company_domain=input_data.get("company_domain"),
                    contact_name=input_data.get("contact_name"),
                    contact_title=input_data.get("contact_title"),
                    contact_email=input_data.get("contact_email"),
                    contact_linkedin=input_data.get("contact_linkedin"),
                    research_depth=ResearchDepth(input_data.get("research_depth", "standard")),
                    focus_areas=input_data.get("focus_areas", []),
                    source=input_data.get("source", "manual"),
                    deal_id=input_data.get("deal_id"),
                    existing_crm_data=input_data.get("existing_crm_data"),
                    additional_context=input_data.get("additional_context"),
                )
                break
            except Exception as e:
                logger.warning(f"[Lead Research] Failed to parse lead input: {e}")

    if not lead_input:
        logger.warning("[Lead Research] No lead research input provided")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "lead_research",
                    "error",
                    {"error": "No lead research input provided"}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="lead_research",
                    content="Error: No lead research input provided. Please provide company name to research.",
                    message_type="error",
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Get client context
    client_context = state.get("client", {})

    try:
        # Step 1: Fetch research data from Tavily
        logger.info(f"[Lead Research] Researching company: {lead_input.company_name}")

        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
            company_search, news_results, competitor_results = loop.run_until_complete(
                _research_lead_with_tavily(lead_input)
            )
        except RuntimeError:
            company_search, news_results, competitor_results = asyncio.run(
                _research_lead_with_tavily(lead_input)
            )

        # Step 2: Analyze with LLM
        logger.info("[Lead Research] Analyzing research data with LLM")

        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
            research_result = loop.run_until_complete(
                _analyze_lead_with_llm(
                    lead_input,
                    company_search,
                    news_results,
                    competitor_results,
                    client_context,
                )
            )
        except RuntimeError:
            research_result = asyncio.run(
                _analyze_lead_with_llm(
                    lead_input,
                    company_search,
                    news_results,
                    competitor_results,
                    client_context,
                )
            )

        # Step 3: Convert to LeadData
        lead_data = _convert_research_to_lead_data(lead_input, research_result)

        logger.info(
            f"[Lead Research] Completed research for {lead_input.company_name}. "
            f"Qualification: {research_result.qualification_score.value}, "
            f"Score: {lead_data['lead_score']}"
        )

        return {
            "leads": [lead_data],
            "agent_execution_log": [
                _create_execution_log(
                    "lead_research",
                    "research_complete",
                    {
                        "company_name": lead_input.company_name,
                        "qualification": research_result.qualification_score.value,
                        "lead_score": lead_data["lead_score"],
                        "key_people_found": len(research_result.key_people),
                        "news_items": len(research_result.recent_news),
                        "business_signals": len(research_result.business_signals),
                        "confidence": research_result.confidence_level.value,
                        "sources_count": len(research_result.sources_used),
                    }
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="lead_research",
                    content=(
                        f"Research completed for {lead_input.company_name}. "
                        f"Qualification: {research_result.qualification_score.value}. "
                        f"{research_result.research_summary}"
                    ),
                    message_type="info",
                    metadata={
                        "lead_id": lead_data["lead_id"],
                        "qualification": research_result.qualification_score.value,
                        "lead_score": lead_data["lead_score"],
                        "talking_points": research_result.talking_points[:3],
                    },
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except TavilyError as e:
        logger.error(f"[Lead Research] Tavily error: {e}")
        # Fallback: Create basic lead without enrichment
        lead_data = _fallback_lead_data(lead_input, f"Tavily error: {e.message}")

        return {
            "leads": [lead_data],
            "agent_execution_log": [
                _create_execution_log(
                    "lead_research",
                    "tavily_error",
                    {"error": str(e), "company": lead_input.company_name}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="lead_research",
                    content=f"Web search failed for {lead_input.company_name}. Basic lead created without enrichment.",
                    message_type="error",
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except LLMError as e:
        logger.error(f"[Lead Research] LLM error: {e}")
        # Fallback: Create basic lead without enrichment
        lead_data = _fallback_lead_data(lead_input, f"LLM error: {e.message}")

        return {
            "leads": [lead_data],
            "agent_execution_log": [
                _create_execution_log(
                    "lead_research",
                    "llm_error",
                    {"error": str(e), "company": lead_input.company_name}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="lead_research",
                    content=f"Analysis failed for {lead_input.company_name}. Basic lead created without enrichment.",
                    message_type="error",
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.exception(f"[Lead Research] Unexpected error: {e}")
        # Fallback: Create basic lead without enrichment
        lead_data = _fallback_lead_data(lead_input, str(e))

        return {
            "leads": [lead_data],
            "agent_execution_log": [
                _create_execution_log(
                    "lead_research",
                    "error",
                    {"error": str(e), "company": lead_input.company_name}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="lead_research",
                    content=f"Unexpected error researching {lead_input.company_name}. Basic lead created.",
                    message_type="error",
                )
            ],
            "error_message": f"Lead Research Error: {str(e)}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


async def _generate_email_with_llm(
    email_input: EmailCopilotInput,
    context: EmailContext,
    client_context: dict[str, Any],
) -> EmailGenerationResult:
    """
    Internal helper to generate email with LLM.

    Args:
        email_input: Email generation input
        context: Aggregated context for personalization
        client_context: Client context from state

    Returns:
        EmailGenerationResult: Generated email draft with metadata
    """
    llm_service = LLMService.get_instance()
    prompt_manager = PromptManager.get_instance()

    # Format context for prompt
    lead_context_str = None
    if context.lead:
        lead_context_str = format_lead_context_for_prompt(context.lead)

    meeting_context_str = None
    if context.meeting:
        meeting_context_str = format_meeting_context_for_prompt(context.meeting)

    brandvoice_context_str = None
    if context.brandvoice:
        brandvoice_context_str = format_brandvoice_context_for_prompt(context.brandvoice)

    # Get and render the prompt template
    system_prompt, user_prompt = prompt_manager.get_full_prompt(
        "email_copilot",
        # Sender info
        sender_name=email_input.sender_name or client_context.get("client_name", "Sales Team"),
        sender_title=email_input.sender_title,
        sender_company=email_input.sender_company or client_context.get("client_name"),
        # Email settings
        email_type=email_input.email_type.value,
        tone=email_input.tone.value,
        # Recipient info
        recipient_email=email_input.recipient.email,
        recipient_name=email_input.recipient.name,
        recipient_title=email_input.recipient.title,
        recipient_company=email_input.recipient.company,
        # Context
        lead_context=lead_context_str,
        meeting_context=meeting_context_str,
        previous_email_context=email_input.previous_email.model_dump() if email_input.previous_email else None,
        brandvoice_context=brandvoice_context_str,
        # Custom
        custom_instructions=email_input.custom_instructions,
        include_calendar_link=email_input.include_calendar_link,
        calendar_link=email_input.calendar_link,
    )

    # Call LLM with structured output
    result, usage = await llm_service.generate_structured(
        output_schema=EmailGenerationResult,
        prompt=user_prompt,
        system_prompt=system_prompt,
    )

    # Set generated_at timestamp
    result.draft.generated_at = datetime.now(timezone.utc).isoformat()

    logger.info(
        f"[Email Copilot] Generated {email_input.email_type.value} email. "
        f"Subject: '{result.draft.subject[:40]}...', "
        f"Personalization: {result.personalization_score:.2f}, "
        f"Tokens: {usage.total_tokens}"
    )

    return result


def _convert_to_state_email_draft(
    email_input: EmailCopilotInput,
    generation_result: EmailGenerationResult,
) -> StateEmailDraft:
    """
    Convert EmailGenerationResult to StateEmailDraft TypedDict.

    Args:
        email_input: Original input
        generation_result: LLM generation result

    Returns:
        StateEmailDraft: TypedDict for state storage
    """
    draft = generation_result.draft

    return StateEmailDraft(
        email_id=str(uuid4()),
        recipient_id=email_input.recipient.contact_id or "",
        recipient_email=email_input.recipient.email,
        subject=draft.subject,
        body=draft.body_html,
        email_type=email_input.email_type.value,
        personalization_data={
            "personalization_elements": draft.personalization_elements,
            "personalization_score": generation_result.personalization_score,
            "relevance_score": generation_result.relevance_score,
            "subject_alternatives": generation_result.subject_alternatives,
            "approach_reasoning": generation_result.approach_reasoning,
        },
        status="draft",
        scheduled_at=None,
        sent_at=None,
    )


def _create_email_approval(
    email_input: EmailCopilotInput,
    generation_result: EmailGenerationResult,
    state_email_draft: StateEmailDraft,
    context: EmailContext,
    workflow_id: str,
) -> ApprovalRequest:
    """
    Create approval request for email.

    Args:
        email_input: Original input
        generation_result: LLM generation result
        state_email_draft: State-compatible email draft
        context: Context used for generation
        workflow_id: Current workflow ID

    Returns:
        ApprovalRequest: Structured approval request for HITL
    """
    draft = generation_result.draft

    # Build recipient summary
    recipient_parts = [email_input.recipient.email]
    if email_input.recipient.name:
        recipient_parts.insert(0, email_input.recipient.name)
    if email_input.recipient.company:
        recipient_parts.append(f"at {email_input.recipient.company}")
    recipient_summary = " ".join(recipient_parts)

    # Build context summary
    context_parts = []
    if context.lead:
        context_parts.append(f"Lead: {context.lead.company_name}")
        if context.lead.qualification_score:
            context_parts.append(f"({context.lead.qualification_score})")
    if context.meeting:
        context_parts.append(f"Meeting: {context.meeting.meeting_date or 'Recent'}")
    if context.brandvoice:
        context_parts.append("Brandvoice: Applied")
    context_summary = " | ".join(context_parts) if context_parts else "Minimal context"

    # Build delivery payload
    delivery_payload = EmailDeliveryPayload(
        to_email=email_input.recipient.email,
        to_name=email_input.recipient.name,
        subject=draft.subject,
        body_html=draft.body_html,
        body_plain=draft.body_plain,
        from_email=email_input.sender_email,
        from_name=email_input.sender_name,
        reply_to=email_input.sender_email,
        contact_id=email_input.recipient.contact_id,
        deal_id=email_input.deal_id,
        company_id=email_input.recipient.company_id,
        track_opens=True,
        track_clicks=True,
        email_type=email_input.email_type,
        workflow_id=workflow_id,
    )

    # Build approval payload
    approval_payload = EmailApprovalPayload(
        draft=draft,
        delivery_payload=delivery_payload,
        recipient_summary=recipient_summary,
        context_summary=context_summary,
        email_type=email_input.email_type,
        priority=email_input.priority,
        personalization_score=generation_result.personalization_score,
        warnings=generation_result.warnings,
    )

    return ApprovalRequest(
        approval_id=str(uuid4()),
        approval_type=ApprovalType.EMAIL_SEND,
        title=f"Email: {draft.subject[:50]}{'...' if len(draft.subject) > 50 else ''}",
        description=f"{email_input.email_type.value.replace('_', ' ').title()} to {recipient_summary}",
        payload=approval_payload.model_dump(),
        requested_at=datetime.now(timezone.utc).isoformat(),
        requested_by="email_copilot",
        status="pending",
        reviewed_at=None,
        reviewed_by=None,
        rejection_reason=None,
    )


def _fallback_email_draft(
    email_input: EmailCopilotInput,
    error_message: str,
) -> StateEmailDraft:
    """
    Create minimal email draft when LLM fails.

    Args:
        email_input: Original input
        error_message: Error that occurred

    Returns:
        StateEmailDraft: Basic draft with error info
    """
    return StateEmailDraft(
        email_id=str(uuid4()),
        recipient_id=email_input.recipient.contact_id or "",
        recipient_email=email_input.recipient.email,
        subject=f"[Draft] Email to {email_input.recipient.name or email_input.recipient.email}",
        body=f"<p>Email generation failed: {error_message}</p><p>Please draft this email manually.</p>",
        email_type=email_input.email_type.value,
        personalization_data={
            "error": error_message,
            "generation_failed": True,
        },
        status="draft",
        scheduled_at=None,
        sent_at=None,
    )


def email_copilot_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Email Copilot Agent - Generates personalized sales emails.

    This is a Phase 4.3 agent.

    Workflow:
    1. Extract email generation input from state["messages"]
    2. Build context from lead research, meeting notes, and brandvoice (RAG)
    3. Generate personalized email with LLM
    4. Create approval request for HITL review
    5. Store draft in state for delivery after approval

    Email Types Supported:
    - cold_outreach: Initial contact with a lead
    - follow_up: Follow-up on previous contact
    - meeting_request: Request for a meeting/demo
    - post_meeting: Summary after a meeting

    Input: Expects email input in state["messages"] with type="email_copilot_input"
           containing recipient info and email type.

    Output:
    - Populates state["email_drafts"] with generated draft
    - Populates state["pending_approvals"] with approval request

    Every email requires human approval before sending (HITL).
    """
    logger.info("[Email Copilot] Starting email generation")

    # Extract email input from messages
    email_input: EmailCopilotInput | None = None

    for message in state.get("messages", []):
        if message.get("message_type") == "email_copilot_input":
            try:
                input_data = message.get("metadata", {})

                # Build recipient
                recipient = EmailRecipient(
                    email=input_data.get("recipient_email", ""),
                    name=input_data.get("recipient_name"),
                    title=input_data.get("recipient_title"),
                    company=input_data.get("recipient_company"),
                    contact_id=input_data.get("contact_id"),
                    company_id=input_data.get("company_id"),
                )

                # Build lead context if provided
                lead_context = None
                if input_data.get("lead_context"):
                    lead_context = LeadContext(**input_data["lead_context"])

                # Build meeting context if provided
                meeting_context = None
                if input_data.get("meeting_context"):
                    meeting_context = MeetingContext(**input_data["meeting_context"])

                email_input = EmailCopilotInput(
                    recipient=recipient,
                    email_type=EmailType(input_data.get("email_type", "cold_outreach")),
                    tone=EmailTone(input_data.get("tone", "professional")),
                    priority=EmailPriority(input_data.get("priority", "normal")),
                    lead_context=lead_context,
                    meeting_context=meeting_context,
                    sender_name=input_data.get("sender_name"),
                    sender_title=input_data.get("sender_title"),
                    sender_company=input_data.get("sender_company"),
                    sender_email=input_data.get("sender_email"),
                    custom_instructions=input_data.get("custom_instructions"),
                    include_calendar_link=input_data.get("include_calendar_link", False),
                    calendar_link=input_data.get("calendar_link"),
                    deal_id=input_data.get("deal_id"),
                )
                break
            except Exception as e:
                logger.warning(f"[Email Copilot] Failed to parse email input: {e}")

    if not email_input:
        logger.warning("[Email Copilot] No email input provided")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "email_copilot",
                    "error",
                    {"error": "No email input provided"}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="email_copilot",
                    content="Error: No email input provided. Please provide recipient and email type.",
                    message_type="error",
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Get context
    client_context = state.get("client", {})
    workflow_id = state.get("workflow_id", str(uuid4()))

    try:
        # Step 1: Build context
        logger.info(f"[Email Copilot] Building context for {email_input.email_type.value} email")

        context_builder = get_email_context_builder()

        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
            context = loop.run_until_complete(
                context_builder.build_context(
                    email_input=email_input,
                    leads=state.get("leads", []),
                    meeting_notes=state.get("meeting_notes", []),
                    email_drafts=state.get("email_drafts", []),
                    client_id=client_context.get("client_id"),
                )
            )
        except RuntimeError:
            context = asyncio.run(
                context_builder.build_context(
                    email_input=email_input,
                    leads=state.get("leads", []),
                    meeting_notes=state.get("meeting_notes", []),
                    email_drafts=state.get("email_drafts", []),
                    client_id=client_context.get("client_id"),
                )
            )

        # Step 2: Generate email with LLM
        logger.info("[Email Copilot] Generating email with LLM")

        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
            generation_result = loop.run_until_complete(
                _generate_email_with_llm(email_input, context, client_context)
            )
        except RuntimeError:
            generation_result = asyncio.run(
                _generate_email_with_llm(email_input, context, client_context)
            )

        # Step 3: Convert to state format
        state_email_draft = _convert_to_state_email_draft(email_input, generation_result)

        # Step 4: Create approval request (HITL - every email requires approval)
        approval_request = _create_email_approval(
            email_input, generation_result, state_email_draft, context, workflow_id
        )

        logger.info(
            f"[Email Copilot] Generated {email_input.email_type.value} email for "
            f"{email_input.recipient.email}. Awaiting approval."
        )

        return {
            "email_drafts": [state_email_draft],
            "pending_approvals": [approval_request],
            "agent_execution_log": [
                _create_execution_log(
                    "email_copilot",
                    "generate_email",
                    {
                        "email_id": state_email_draft["email_id"],
                        "email_type": email_input.email_type.value,
                        "recipient": email_input.recipient.email,
                        "subject_length": len(generation_result.draft.subject),
                        "body_word_count": generation_result.draft.word_count,
                        "personalization_score": generation_result.personalization_score,
                        "relevance_score": generation_result.relevance_score,
                        "context_completeness": context.context_completeness,
                        "warnings": generation_result.warnings,
                    }
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="email_copilot",
                    content=(
                        f"Generated {email_input.email_type.value.replace('_', ' ')} email for "
                        f"{email_input.recipient.name or email_input.recipient.email}. "
                        f"Subject: '{generation_result.draft.subject[:40]}...'. "
                        f"Awaiting approval."
                    ),
                    message_type="request",
                    metadata={
                        "email_id": state_email_draft["email_id"],
                        "approval_id": approval_request["approval_id"],
                        "personalization_score": generation_result.personalization_score,
                    },
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except LLMError as e:
        logger.error(f"[Email Copilot] LLM error: {e}")
        # Create fallback draft
        fallback_draft = _fallback_email_draft(email_input, f"LLM error: {e.message}")

        return {
            "email_drafts": [fallback_draft],
            "agent_execution_log": [
                _create_execution_log(
                    "email_copilot",
                    "llm_error",
                    {
                        "error": str(e),
                        "recipient": email_input.recipient.email,
                        "provider": e.provider.value if e.provider else None,
                    }
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="email_copilot",
                    content=f"Email generation failed: {e.message}. Basic draft created for manual editing.",
                    message_type="error",
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.exception(f"[Email Copilot] Unexpected error: {e}")
        # Create fallback draft
        fallback_draft = _fallback_email_draft(email_input, str(e))

        return {
            "email_drafts": [fallback_draft],
            "agent_execution_log": [
                _create_execution_log(
                    "email_copilot",
                    "error",
                    {"error": str(e), "recipient": email_input.recipient.email}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="email_copilot",
                    content=f"Unexpected error generating email: {str(e)}. Basic draft created.",
                    message_type="error",
                )
            ],
            "error_message": f"Email Copilot Error: {str(e)}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


def email_delivery_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Email Delivery Agent - Sends approved emails via configured provider.

    This is a Phase 4.3 agent.

    Workflow:
    1. Find approved emails from approval_history
    2. Get the corresponding email drafts
    3. Send via configured adapter (HubSpot MVP)
    4. Update email status and log activity

    Input: Reads from state["approval_history"] for approved EMAIL_SEND requests
           and state["email_drafts"] for the email content.

    Output: Updates state["email_drafts"] with sent status and logs delivery.

    Prerequisites: Email must be approved via HITL before delivery.
    """
    logger.info("[Email Delivery] Processing approved emails")

    # Find approved email requests
    approved_emails = [
        approval for approval in state.get("approval_history", [])
        if (
            approval.get("approval_type") == ApprovalType.EMAIL_SEND
            and approval.get("status") == "approved"
        )
    ]

    if not approved_emails:
        logger.info("[Email Delivery] No approved emails to send")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "email_delivery",
                    "no_approved_emails",
                    {"message": "No approved emails found"}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="email_delivery",
                    content="No approved emails to send.",
                    message_type="info",
                )
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Get email adapter
    client_context = state.get("client", {})
    crm_provider = client_context.get("crm_provider", "hubspot")

    try:
        adapter = get_email_adapter(crm_provider)
    except ValueError as e:
        logger.error(f"[Email Delivery] Adapter error: {e}")
        return {
            "agent_execution_log": [
                _create_execution_log(
                    "email_delivery",
                    "adapter_error",
                    {"error": str(e), "provider": crm_provider}
                )
            ],
            "messages": [
                _create_agent_message(
                    from_agent="email_delivery",
                    content=f"Email adapter not available: {e}",
                    message_type="error",
                )
            ],
            "error_message": f"Email Delivery Error: {str(e)}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Process each approved email
    sent_count = 0
    failed_count = 0
    delivery_log = []

    for approval in approved_emails:
        payload = approval.get("payload", {})

        # Get delivery payload from approval
        delivery_data = payload.get("delivery_payload", {})
        if not delivery_data:
            logger.warning(f"[Email Delivery] No delivery payload for approval {approval['approval_id']}")
            continue

        try:
            # Build EmailDeliveryPayload
            delivery_payload = EmailDeliveryPayload(**delivery_data)

            # Send email
            try:
                loop = asyncio.get_running_loop()
                import nest_asyncio
                nest_asyncio.apply()
                result = loop.run_until_complete(adapter.send(delivery_payload))
            except RuntimeError:
                result = asyncio.run(adapter.send(delivery_payload))

            if result.success:
                sent_count += 1
                delivery_log.append({
                    "email": delivery_payload.to_email,
                    "status": "sent",
                    "message_id": result.message_id,
                })
                logger.info(f"[Email Delivery] Sent email to {delivery_payload.to_email}")
            else:
                failed_count += 1
                delivery_log.append({
                    "email": delivery_payload.to_email,
                    "status": "failed",
                    "error": result.error_message,
                })

        except EmailAdapterError as e:
            failed_count += 1
            delivery_log.append({
                "email": delivery_data.get("to_email", "unknown"),
                "status": "failed",
                "error": str(e.message),
            })
            logger.error(f"[Email Delivery] Adapter error: {e}")

        except Exception as e:
            failed_count += 1
            delivery_log.append({
                "email": delivery_data.get("to_email", "unknown"),
                "status": "failed",
                "error": str(e),
            })
            logger.exception(f"[Email Delivery] Error: {e}")

    # Build result message
    if sent_count > 0 and failed_count == 0:
        result_message = f"Successfully sent {sent_count} email(s)."
        message_type = "info"
    elif sent_count > 0 and failed_count > 0:
        result_message = f"Sent {sent_count} email(s), {failed_count} failed."
        message_type = "info"
    else:
        result_message = f"All {failed_count} email(s) failed to send."
        message_type = "error"

    return {
        "agent_execution_log": [
            _create_execution_log(
                "email_delivery",
                "deliver_emails",
                {
                    "sent_count": sent_count,
                    "failed_count": failed_count,
                    "delivery_log": delivery_log,
                }
            )
        ],
        "messages": [
            _create_agent_message(
                from_agent="email_delivery",
                content=result_message,
                message_type=message_type,
                metadata={
                    "sent_count": sent_count,
                    "failed_count": failed_count,
                },
            )
        ],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# UTILITY NODES
# =============================================================================

def human_approval_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Human Approval Node - Pauses workflow for human review.

    This node:
    - Sets workflow status to AWAITING_APPROVAL
    - Stores checkpoint for resumption
    - Notifies frontend of pending approvals

    The workflow will be resumed via API call after human decision.
    """
    logger.info("[Human Approval] Pausing for human review")

    pending_count = len([
        a for a in state.get("pending_approvals", [])
        if a.get("status") == "pending"
    ])

    return {
        "status": WorkflowStatus.AWAITING_APPROVAL,
        "agent_execution_log": [
            _create_execution_log(
                "human_approval",
                "await_approval",
                {"pending_count": pending_count}
            )
        ],
        "messages": [
            _create_agent_message(
                from_agent="human_approval",
                content=f"Workflow paused. {pending_count} approval(s) pending.",
                message_type="info",
            )
        ],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def finalize_node(state: OrchestratorState) -> dict[str, Any]:
    """
    Finalize Node - Completes the workflow.

    This node:
    - Sets final workflow status
    - Generates execution summary
    - Prepares output for storage
    """
    logger.info("[Finalize] Completing workflow")

    # Check if there were any errors
    has_errors = bool(state.get("error_message"))

    # Check if all approvals were handled
    pending_approvals = [
        a for a in state.get("pending_approvals", [])
        if a.get("status") == "pending"
    ]

    # Determine final status
    if has_errors:
        final_status = WorkflowStatus.FAILED
    elif pending_approvals:
        final_status = WorkflowStatus.AWAITING_APPROVAL
    else:
        final_status = WorkflowStatus.COMPLETED

    # Generate summary
    execution_log = state.get("agent_execution_log", [])
    summary = {
        "total_agents_executed": len(set(log.get("agent") for log in execution_log)),
        "total_actions": len(execution_log),
        "crm_tasks_created": len(state.get("crm_tasks", [])),
        "approvals_processed": len(state.get("approval_history", [])),
        "content_drafts": len(state.get("content_drafts", [])),
    }

    return {
        "status": final_status,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "agent_execution_log": [
            _create_execution_log(
                "finalize",
                "complete_workflow",
                {"summary": summary, "final_status": final_status.value}
            )
        ],
        "messages": [
            _create_agent_message(
                from_agent="finalize",
                content=f"Workflow completed with status: {final_status.value}",
                message_type="info",
                metadata=summary,
            )
        ],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Supervisor
    "supervisor_node",
    # Intelligence Squad
    "intelligence_router_node",
    "market_research_node",
    "seo_analysis_node",
    "web_analysis_node",
    "audience_builder_node",
    # Content Squad
    "content_router_node",
    "pipeline_manager_node",
    "brandvoice_writer_node",
    "seo_optimizer_node",
    "social_distributor_node",
    "web_publisher_node",
    # Sales Ops Squad
    "sales_ops_router_node",
    "meeting_notes_node",
    "task_extractor_node",
    "crm_updater_node",
    "lead_research_node",
    "email_copilot_node",
    "email_delivery_node",
    # Utility
    "human_approval_node",
    "finalize_node",
]

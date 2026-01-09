"""
LangGraph Workflow Definition
=============================
StateGraph yapısı, checkpointer konfigürasyonu ve workflow compile işlemleri.
Bu dosya tüm workflow varyantlarını (full_cycle, sales_ops_only, vb.) yönetir.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from backend.graph.state import OrchestratorState, WorkflowStatus
from backend.graph.nodes import (
    # Supervisor
    supervisor_node,
    # Intelligence Squad
    intelligence_router_node,
    market_research_node,
    seo_analysis_node,
    web_analysis_node,
    audience_builder_node,
    # Content Squad
    content_router_node,
    pipeline_manager_node,
    brandvoice_writer_node,
    seo_optimizer_node,
    social_distributor_node,
    web_publisher_node,
    # Sales Ops Squad
    sales_ops_router_node,
    meeting_notes_node,
    task_extractor_node,
    crm_updater_node,
    lead_research_node,
    email_copilot_node,
    # Human-in-the-Loop
    human_approval_node,
    # Finalization
    finalize_node,
)
from backend.graph.routers import (
    route_from_supervisor,
    route_from_intelligence,
    route_from_content,
    route_from_sales_ops,
    should_request_approval,
    route_after_approval,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CHECKPOINTER FACTORY
# =============================================================================

class CheckpointerFactory:
    """Factory for creating different checkpointer backends."""

    @staticmethod
    def create_memory() -> MemorySaver:
        """
        In-memory checkpointer for development/testing.
        State is lost on restart.
        """
        return MemorySaver()

    @staticmethod
    def create_sqlite(db_path: str = ":memory:") -> Any:
        """
        SQLite-based checkpointer for persistent state.

        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory

        Returns:
            SqliteSaver instance (requires langgraph-checkpoint-sqlite)
        """
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            return SqliteSaver.from_conn_string(db_path)
        except ImportError:
            logger.warning(
                "SqliteSaver not available. Install langgraph-checkpoint-sqlite. "
                "Falling back to MemorySaver."
            )
            return CheckpointerFactory.create_memory()

    @staticmethod
    def create_postgres(connection_string: str) -> Any:
        """
        PostgreSQL-based checkpointer for production.

        Args:
            connection_string: PostgreSQL connection URL

        Returns:
            PostgresSaver instance (requires langgraph-checkpoint-postgres)
        """
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            return PostgresSaver.from_conn_string(connection_string)
        except ImportError:
            logger.warning(
                "PostgresSaver not available. Install langgraph-checkpoint-postgres. "
                "Falling back to MemorySaver."
            )
            return CheckpointerFactory.create_memory()


# =============================================================================
# WORKFLOW BUILDERS
# =============================================================================

def build_full_cycle_workflow() -> StateGraph:
    """
    Full cycle workflow - All squads in sequence.

    Flow:
    START -> Supervisor -> Intelligence Squad -> Content Squad -> Sales Ops Squad
          -> Human Approval (if needed) -> Finalize -> END
    """
    workflow = StateGraph(OrchestratorState)

    # --- Add Nodes ---
    # Supervisor
    workflow.add_node("supervisor", supervisor_node)

    # Intelligence Squad
    workflow.add_node("intelligence_router", intelligence_router_node)
    workflow.add_node("market_research", market_research_node)
    workflow.add_node("seo_analysis", seo_analysis_node)
    workflow.add_node("web_analysis", web_analysis_node)
    workflow.add_node("audience_builder", audience_builder_node)

    # Content Squad
    workflow.add_node("content_router", content_router_node)
    workflow.add_node("pipeline_manager", pipeline_manager_node)
    workflow.add_node("brandvoice_writer", brandvoice_writer_node)
    workflow.add_node("seo_optimizer", seo_optimizer_node)
    workflow.add_node("social_distributor", social_distributor_node)
    workflow.add_node("web_publisher", web_publisher_node)

    # Sales Ops Squad
    workflow.add_node("sales_ops_router", sales_ops_router_node)
    workflow.add_node("meeting_notes", meeting_notes_node)
    workflow.add_node("task_extractor", task_extractor_node)
    workflow.add_node("crm_updater", crm_updater_node)
    workflow.add_node("lead_research", lead_research_node)
    workflow.add_node("email_copilot", email_copilot_node)

    # Human-in-the-Loop & Finalization
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("finalize", finalize_node)

    # --- Add Edges ---
    # Entry point
    workflow.add_edge(START, "supervisor")

    # Supervisor routes to appropriate squad
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "intelligence": "intelligence_router",
            "content": "content_router",
            "sales_ops": "sales_ops_router",
            "finalize": "finalize",
        }
    )

    # Intelligence Squad internal routing
    workflow.add_conditional_edges(
        "intelligence_router",
        route_from_intelligence,
        {
            "market_research": "market_research",
            "seo_analysis": "seo_analysis",
            "web_analysis": "web_analysis",
            "audience_builder": "audience_builder",
            "done": "supervisor",
        }
    )

    # Intelligence agents return to router
    workflow.add_edge("market_research", "intelligence_router")
    workflow.add_edge("seo_analysis", "intelligence_router")
    workflow.add_edge("web_analysis", "intelligence_router")
    workflow.add_edge("audience_builder", "intelligence_router")

    # Content Squad internal routing
    workflow.add_conditional_edges(
        "content_router",
        route_from_content,
        {
            "pipeline_manager": "pipeline_manager",
            "brandvoice_writer": "brandvoice_writer",
            "seo_optimizer": "seo_optimizer",
            "social_distributor": "social_distributor",
            "web_publisher": "web_publisher",
            "done": "supervisor",
        }
    )

    # Content agents return to router
    workflow.add_edge("pipeline_manager", "content_router")
    workflow.add_edge("brandvoice_writer", "content_router")
    workflow.add_edge("seo_optimizer", "content_router")
    workflow.add_edge("social_distributor", "content_router")
    workflow.add_edge("web_publisher", "content_router")

    # Sales Ops Squad internal routing
    workflow.add_conditional_edges(
        "sales_ops_router",
        route_from_sales_ops,
        {
            "meeting_notes": "meeting_notes",
            "task_extractor": "task_extractor",
            "crm_updater": "crm_updater",
            "lead_research": "lead_research",
            "email_copilot": "email_copilot",
            "approval_check": "human_approval",
            "done": "supervisor",
        }
    )

    # Sales Ops agents return to router
    workflow.add_edge("meeting_notes", "sales_ops_router")
    workflow.add_edge("task_extractor", "sales_ops_router")
    workflow.add_edge("crm_updater", "sales_ops_router")
    workflow.add_edge("lead_research", "sales_ops_router")
    workflow.add_edge("email_copilot", "sales_ops_router")

    # Human Approval routing
    workflow.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "continue": "supervisor",
            "finalize": "finalize",
        }
    )

    # Finalize to END
    workflow.add_edge("finalize", END)

    return workflow


def build_sales_ops_workflow() -> StateGraph:
    """
    Sales Ops only workflow - MVP workflow for meeting analysis.

    Flow:
    START -> Sales Ops Router -> [Meeting Notes -> Task Extractor -> CRM Updater]
          -> Human Approval -> Finalize -> END
    """
    workflow = StateGraph(OrchestratorState)

    # --- Add Nodes ---
    workflow.add_node("sales_ops_router", sales_ops_router_node)
    workflow.add_node("meeting_notes", meeting_notes_node)
    workflow.add_node("task_extractor", task_extractor_node)
    workflow.add_node("crm_updater", crm_updater_node)
    workflow.add_node("lead_research", lead_research_node)
    workflow.add_node("email_copilot", email_copilot_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("finalize", finalize_node)

    # --- Add Edges ---
    workflow.add_edge(START, "sales_ops_router")

    # Sales Ops internal routing
    workflow.add_conditional_edges(
        "sales_ops_router",
        route_from_sales_ops,
        {
            "meeting_notes": "meeting_notes",
            "task_extractor": "task_extractor",
            "crm_updater": "crm_updater",
            "lead_research": "lead_research",
            "email_copilot": "email_copilot",
            "approval_check": "human_approval",
            "done": "finalize",
        }
    )

    # Agents return to router
    workflow.add_edge("meeting_notes", "sales_ops_router")
    workflow.add_edge("task_extractor", "sales_ops_router")
    workflow.add_edge("crm_updater", "sales_ops_router")
    workflow.add_edge("lead_research", "sales_ops_router")
    workflow.add_edge("email_copilot", "sales_ops_router")

    # Human Approval
    workflow.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "continue": "sales_ops_router",
            "finalize": "finalize",
        }
    )

    workflow.add_edge("finalize", END)

    return workflow


def build_meeting_analysis_workflow() -> StateGraph:
    """
    MVP Meeting Analysis Workflow - Minimal agents for meeting-to-CRM flow.

    Flow:
    START -> Meeting Notes -> Task Extractor -> CRM Updater
          -> Human Approval -> Finalize -> END

    This is the simplest workflow for MVP testing.
    """
    workflow = StateGraph(OrchestratorState)

    # --- Add Nodes ---
    workflow.add_node("meeting_notes", meeting_notes_node)
    workflow.add_node("task_extractor", task_extractor_node)
    workflow.add_node("crm_updater", crm_updater_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("finalize", finalize_node)

    # --- Add Edges ---
    # Linear flow: Meeting Notes -> Task Extractor -> CRM Updater
    workflow.add_edge(START, "meeting_notes")
    workflow.add_edge("meeting_notes", "task_extractor")
    workflow.add_edge("task_extractor", "crm_updater")

    # After CRM Updater, check if approval is needed
    workflow.add_conditional_edges(
        "crm_updater",
        should_request_approval,
        {
            "needs_approval": "human_approval",
            "no_approval": "finalize",
        }
    )

    # After approval, finalize
    workflow.add_edge("human_approval", "finalize")
    workflow.add_edge("finalize", END)

    return workflow


def build_intelligence_workflow() -> StateGraph:
    """
    Intelligence only workflow - Research and analysis.

    Flow:
    START -> Intelligence Router -> [Market Research, SEO, Web Analysis, Audience]
          -> Finalize -> END
    """
    workflow = StateGraph(OrchestratorState)

    # --- Add Nodes ---
    workflow.add_node("intelligence_router", intelligence_router_node)
    workflow.add_node("market_research", market_research_node)
    workflow.add_node("seo_analysis", seo_analysis_node)
    workflow.add_node("web_analysis", web_analysis_node)
    workflow.add_node("audience_builder", audience_builder_node)
    workflow.add_node("finalize", finalize_node)

    # --- Add Edges ---
    workflow.add_edge(START, "intelligence_router")

    workflow.add_conditional_edges(
        "intelligence_router",
        route_from_intelligence,
        {
            "market_research": "market_research",
            "seo_analysis": "seo_analysis",
            "web_analysis": "web_analysis",
            "audience_builder": "audience_builder",
            "done": "finalize",
        }
    )

    workflow.add_edge("market_research", "intelligence_router")
    workflow.add_edge("seo_analysis", "intelligence_router")
    workflow.add_edge("web_analysis", "intelligence_router")
    workflow.add_edge("audience_builder", "intelligence_router")
    workflow.add_edge("finalize", END)

    return workflow


def build_content_workflow() -> StateGraph:
    """
    Content only workflow - Content generation and publishing.

    Flow:
    START -> Content Router -> [Pipeline, Writer, SEO, Social, Publisher]
          -> Human Approval -> Finalize -> END
    """
    workflow = StateGraph(OrchestratorState)

    # --- Add Nodes ---
    workflow.add_node("content_router", content_router_node)
    workflow.add_node("pipeline_manager", pipeline_manager_node)
    workflow.add_node("brandvoice_writer", brandvoice_writer_node)
    workflow.add_node("seo_optimizer", seo_optimizer_node)
    workflow.add_node("social_distributor", social_distributor_node)
    workflow.add_node("web_publisher", web_publisher_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("finalize", finalize_node)

    # --- Add Edges ---
    workflow.add_edge(START, "content_router")

    workflow.add_conditional_edges(
        "content_router",
        route_from_content,
        {
            "pipeline_manager": "pipeline_manager",
            "brandvoice_writer": "brandvoice_writer",
            "seo_optimizer": "seo_optimizer",
            "social_distributor": "social_distributor",
            "web_publisher": "web_publisher",
            "approval_check": "human_approval",
            "done": "finalize",
        }
    )

    workflow.add_edge("pipeline_manager", "content_router")
    workflow.add_edge("brandvoice_writer", "content_router")
    workflow.add_edge("seo_optimizer", "content_router")
    workflow.add_edge("social_distributor", "content_router")
    workflow.add_edge("web_publisher", "content_router")

    workflow.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "continue": "content_router",
            "finalize": "finalize",
        }
    )

    workflow.add_edge("finalize", END)

    return workflow


# =============================================================================
# WORKFLOW REGISTRY & COMPILER
# =============================================================================

# Available workflow builders
WORKFLOW_BUILDERS: dict[str, Callable[[], StateGraph]] = {
    "full_cycle": build_full_cycle_workflow,
    "sales_ops_only": build_sales_ops_workflow,
    "meeting_analysis": build_meeting_analysis_workflow,
    "intelligence_only": build_intelligence_workflow,
    "content_only": build_content_workflow,
}


def compile_workflow(
    workflow_type: str = "meeting_analysis",
    checkpointer_type: Literal["memory", "sqlite", "postgres"] = "memory",
    checkpointer_config: dict[str, Any] | None = None,
) -> CompiledStateGraph:
    """
    Compile a workflow with the specified checkpointer.

    Args:
        workflow_type: Type of workflow to build (see WORKFLOW_BUILDERS)
        checkpointer_type: Type of checkpointer backend
        checkpointer_config: Optional configuration for checkpointer

    Returns:
        Compiled StateGraph ready for execution

    Example:
        >>> workflow = compile_workflow("meeting_analysis", "memory")
        >>> result = await workflow.ainvoke(initial_state, config={"configurable": {"thread_id": "123"}})
    """
    config = checkpointer_config or {}

    # Get workflow builder
    if workflow_type not in WORKFLOW_BUILDERS:
        raise ValueError(
            f"Unknown workflow type: {workflow_type}. "
            f"Available: {list(WORKFLOW_BUILDERS.keys())}"
        )

    builder = WORKFLOW_BUILDERS[workflow_type]
    workflow = builder()

    # Create checkpointer
    if checkpointer_type == "memory":
        checkpointer = CheckpointerFactory.create_memory()
    elif checkpointer_type == "sqlite":
        db_path = config.get("db_path", ":memory:")
        checkpointer = CheckpointerFactory.create_sqlite(db_path)
    elif checkpointer_type == "postgres":
        conn_string = config.get("connection_string")
        if not conn_string:
            raise ValueError("PostgreSQL checkpointer requires 'connection_string' in config")
        checkpointer = CheckpointerFactory.create_postgres(conn_string)
    else:
        raise ValueError(f"Unknown checkpointer type: {checkpointer_type}")

    # Compile with checkpointer
    compiled = workflow.compile(checkpointer=checkpointer)

    logger.info(f"Compiled workflow '{workflow_type}' with {checkpointer_type} checkpointer")

    return compiled


def get_workflow(workflow_type: str = "meeting_analysis") -> CompiledStateGraph:
    """
    Get a compiled workflow with default memory checkpointer.

    This is a convenience function for quick workflow access.

    Args:
        workflow_type: Type of workflow to get

    Returns:
        Compiled workflow with memory checkpointer
    """
    return compile_workflow(workflow_type, "memory")


# =============================================================================
# WORKFLOW VISUALIZATION (Development Only)
# =============================================================================

def get_workflow_graph_image(workflow_type: str = "meeting_analysis") -> bytes | None:
    """
    Generate a PNG image of the workflow graph.

    Requires graphviz to be installed on the system.

    Args:
        workflow_type: Type of workflow to visualize

    Returns:
        PNG image bytes or None if graphviz is not available
    """
    try:
        builder = WORKFLOW_BUILDERS.get(workflow_type)
        if not builder:
            return None

        workflow = builder()
        compiled = workflow.compile()

        # Generate graph image
        return compiled.get_graph().draw_mermaid_png()
    except Exception as e:
        logger.warning(f"Could not generate workflow image: {e}")
        return None


def get_workflow_mermaid(workflow_type: str = "meeting_analysis") -> str | None:
    """
    Get Mermaid diagram code for the workflow.

    Args:
        workflow_type: Type of workflow to visualize

    Returns:
        Mermaid diagram code string
    """
    try:
        builder = WORKFLOW_BUILDERS.get(workflow_type)
        if not builder:
            return None

        workflow = builder()
        compiled = workflow.compile()

        return compiled.get_graph().draw_mermaid()
    except Exception as e:
        logger.warning(f"Could not generate Mermaid diagram: {e}")
        return None

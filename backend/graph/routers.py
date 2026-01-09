"""
LangGraph Routing Functions
===========================
Conditional edge fonksiyonları. Her router, state'i analiz ederek
bir sonraki node'u belirler.

Router Pattern:
- State'i incele
- Karar ver
- String olarak hedef node adını döndür
"""

from __future__ import annotations

import logging
from typing import Literal

from backend.graph.state import OrchestratorState, WorkflowStatus

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

# Supervisor routing options
SupervisorRoute = Literal["intelligence", "content", "sales_ops", "finalize"]

# Intelligence squad routing options
IntelligenceRoute = Literal[
    "market_research",
    "seo_analysis",
    "web_analysis",
    "audience_builder",
    "done",
]

# Content squad routing options
ContentRoute = Literal[
    "pipeline_manager",
    "brandvoice_writer",
    "seo_optimizer",
    "social_distributor",
    "web_publisher",
    "approval_check",
    "done",
]

# Sales ops squad routing options
SalesOpsRoute = Literal[
    "meeting_notes",
    "task_extractor",
    "crm_updater",
    "lead_research",
    "email_copilot",
    "approval_check",
    "done",
]

# Approval routing options
ApprovalRoute = Literal["continue", "finalize"]

# Simple approval check
ApprovalCheck = Literal["needs_approval", "no_approval"]


# =============================================================================
# SUPERVISOR ROUTER
# =============================================================================

def route_from_supervisor(state: OrchestratorState) -> SupervisorRoute:
    """
    Main routing decision from supervisor.

    Determines which squad should run based on workflow type and progress.

    Logic:
    - full_cycle: intelligence -> content -> sales_ops -> finalize
    - intelligence_only: intelligence -> finalize
    - content_only: content -> finalize
    - sales_ops_only: sales_ops -> finalize
    """
    workflow_type = state["workflow_type"]
    status = state["status"]

    logger.info(f"[Router] Supervisor routing for workflow type: {workflow_type}")

    # Check if workflow should be finalized
    if status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED):
        return "finalize"

    # Check for pending approvals that block progress
    pending_approvals = [
        a for a in state.get("pending_approvals", [])
        if a.get("status") == "pending"
    ]
    if pending_approvals:
        logger.info("[Router] Pending approvals detected, routing to finalize (will pause)")
        return "finalize"

    # Route based on workflow type
    if workflow_type == "intelligence_only":
        return _route_intelligence_only(state)
    elif workflow_type == "content_only":
        return _route_content_only(state)
    elif workflow_type == "sales_ops_only":
        return _route_sales_ops_only(state)
    elif workflow_type == "full_cycle":
        return _route_full_cycle(state)
    else:
        # Default: sales_ops for custom/unknown types
        return "sales_ops"


def _route_full_cycle(state: OrchestratorState) -> SupervisorRoute:
    """Route full cycle workflow through all squads."""
    # Check what's completed
    has_intelligence = bool(
        state.get("seo_data") or
        state.get("market_research") or
        state.get("audience_data")
    )
    has_content = bool(state.get("content_pipeline") or state.get("content_drafts"))
    has_sales_ops = bool(state.get("meeting_notes") or state.get("crm_tasks"))

    # Sequential: Intelligence -> Content -> Sales Ops
    if not has_intelligence:
        return "intelligence"
    elif not has_content:
        return "content"
    elif not has_sales_ops:
        return "sales_ops"
    else:
        return "finalize"


def _route_intelligence_only(state: OrchestratorState) -> SupervisorRoute:
    """Route intelligence-only workflow."""
    has_intelligence = bool(
        state.get("seo_data") and
        state.get("market_research") and
        state.get("audience_data")
    )
    return "finalize" if has_intelligence else "intelligence"


def _route_content_only(state: OrchestratorState) -> SupervisorRoute:
    """Route content-only workflow."""
    content_drafts = state.get("content_drafts", [])
    # Check if we have published content
    published = [d for d in content_drafts if d.get("status") == "published"]
    return "finalize" if published else "content"


def _route_sales_ops_only(state: OrchestratorState) -> SupervisorRoute:
    """Route sales-ops-only workflow."""
    # Check if CRM tasks are processed
    crm_tasks = state.get("crm_tasks", [])
    all_processed = all(
        task.get("status") in ("executed", "failed")
        for task in crm_tasks
    ) if crm_tasks else False

    return "finalize" if all_processed else "sales_ops"


# =============================================================================
# INTELLIGENCE SQUAD ROUTER
# =============================================================================

def route_from_intelligence(state: OrchestratorState) -> IntelligenceRoute:
    """
    Route within Intelligence Squad.

    Executes agents in order:
    1. Market Research
    2. SEO Analysis
    3. Web Analysis
    4. Audience Builder
    """
    logger.info("[Router] Intelligence squad routing")

    # Check completion status of each agent
    has_market = bool(state.get("market_research"))
    has_seo = bool(state.get("seo_data"))
    has_web = bool(state.get("web_analysis"))
    has_audience = bool(state.get("audience_data"))

    # Route to first incomplete agent
    if not has_market:
        return "market_research"
    elif not has_seo:
        return "seo_analysis"
    elif not has_web:
        return "web_analysis"
    elif not has_audience:
        return "audience_builder"
    else:
        return "done"


# =============================================================================
# CONTENT SQUAD ROUTER
# =============================================================================

def route_from_content(state: OrchestratorState) -> ContentRoute:
    """
    Route within Content Squad.

    Executes agents in order:
    1. Pipeline Manager (plan content)
    2. Brandvoice Writer (create drafts)
    3. SEO Optimizer (optimize for search)
    4. Social Distributor (prepare social versions)
    5. Web Publisher (publish - requires approval)
    """
    logger.info("[Router] Content squad routing")

    content_pipeline = state.get("content_pipeline")
    content_drafts = state.get("content_drafts", [])

    # Check for pending publish approvals
    pending_publish = [
        a for a in state.get("pending_approvals", [])
        if a.get("approval_type") == "content_publish" and a.get("status") == "pending"
    ]
    if pending_publish:
        return "approval_check"

    # Check pipeline status
    if not content_pipeline:
        return "pipeline_manager"

    # Check if we have drafts
    draft_content = [d for d in content_drafts if d.get("status") == "draft"]
    review_content = [d for d in content_drafts if d.get("status") == "review"]
    approved_content = [d for d in content_drafts if d.get("status") == "approved"]

    # Route based on content status
    if not content_drafts:
        return "brandvoice_writer"
    elif draft_content and not review_content:
        return "seo_optimizer"
    elif review_content:
        return "social_distributor"
    elif approved_content:
        return "web_publisher"
    else:
        return "done"


# =============================================================================
# SALES OPS SQUAD ROUTER
# =============================================================================

def route_from_sales_ops(state: OrchestratorState) -> SalesOpsRoute:
    """
    Route within Sales Ops Squad.

    For MVP (meeting_analysis workflow):
    1. Meeting Notes (analyze transcript)
    2. Task Extractor (extract action items)
    3. CRM Updater (prepare CRM updates - requires approval)

    For full sales ops:
    - Also includes Lead Research and Email Copilot
    """
    logger.info("[Router] Sales ops squad routing")

    # Check for pending CRM approvals
    pending_crm = [
        a for a in state.get("pending_approvals", [])
        if a.get("approval_type") in ("crm_update", "task_create", "email_send")
        and a.get("status") == "pending"
    ]
    if pending_crm:
        return "approval_check"

    # Get current data
    meeting_notes = state.get("meeting_notes", [])
    crm_tasks = state.get("crm_tasks", [])
    leads = state.get("leads", [])
    email_drafts = state.get("email_drafts", [])

    # Check what needs to be done
    workflow_type = state["workflow_type"]

    # MVP Flow: Meeting Notes -> Task Extractor -> CRM Updater
    if workflow_type in ("meeting_analysis", "sales_ops_only"):
        if not meeting_notes:
            return "meeting_notes"
        elif not crm_tasks:
            return "task_extractor"
        else:
            # Check if tasks need CRM update
            pending_tasks = [t for t in crm_tasks if t.get("status") == "pending"]
            if pending_tasks:
                return "crm_updater"
            else:
                return "done"

    # Full Sales Ops flow
    if not meeting_notes:
        return "meeting_notes"
    elif not crm_tasks:
        return "task_extractor"
    elif not leads:
        return "lead_research"
    elif not email_drafts:
        return "email_copilot"
    else:
        # Check for any pending operations
        pending_tasks = [t for t in crm_tasks if t.get("status") == "pending"]
        if pending_tasks:
            return "crm_updater"
        return "done"


# =============================================================================
# APPROVAL ROUTERS
# =============================================================================

def should_request_approval(state: OrchestratorState) -> ApprovalCheck:
    """
    Check if workflow should pause for human approval.

    Used after CRM Updater to determine if approval is needed.
    """
    pending_approvals = [
        a for a in state.get("pending_approvals", [])
        if a.get("status") == "pending"
    ]

    if pending_approvals:
        logger.info(f"[Router] {len(pending_approvals)} approval(s) pending")
        return "needs_approval"
    else:
        return "no_approval"


def route_after_approval(state: OrchestratorState) -> ApprovalRoute:
    """
    Route after human approval decision.

    Checks approval status and determines next step:
    - If all approved: continue workflow
    - If rejected or still pending: finalize (end workflow)
    """
    pending_approvals = [
        a for a in state.get("pending_approvals", [])
        if a.get("status") == "pending"
    ]

    # Check if any approvals were rejected
    rejected_approvals = [
        a for a in state.get("approval_history", [])
        if a.get("status") == "rejected"
    ]

    if rejected_approvals:
        logger.info("[Router] Approvals rejected, finalizing workflow")
        return "finalize"
    elif pending_approvals:
        logger.info("[Router] Still have pending approvals, finalizing (will pause)")
        return "finalize"
    else:
        logger.info("[Router] All approvals handled, continuing workflow")
        return "continue"


# =============================================================================
# UTILITY ROUTERS
# =============================================================================

def get_next_pending_agent(state: OrchestratorState) -> str | None:
    """
    Utility function to find the next agent that needs to run.

    Useful for debugging and status reporting.
    """
    workflow_type = state["workflow_type"]

    # Get supervisor route
    supervisor_route = route_from_supervisor(state)

    if supervisor_route == "finalize":
        return None
    elif supervisor_route == "intelligence":
        intel_route = route_from_intelligence(state)
        return None if intel_route == "done" else intel_route
    elif supervisor_route == "content":
        content_route = route_from_content(state)
        return None if content_route == "done" else content_route
    elif supervisor_route == "sales_ops":
        sales_route = route_from_sales_ops(state)
        return None if sales_route == "done" else sales_route

    return None


def get_workflow_progress(state: OrchestratorState) -> dict:
    """
    Calculate workflow progress percentage.

    Returns progress info for each squad and overall.
    """
    progress = {
        "intelligence": {
            "market_research": bool(state.get("market_research")),
            "seo_analysis": bool(state.get("seo_data")),
            "web_analysis": bool(state.get("web_analysis")),
            "audience_builder": bool(state.get("audience_data")),
        },
        "content": {
            "pipeline_manager": bool(state.get("content_pipeline")),
            "has_drafts": len(state.get("content_drafts", [])) > 0,
        },
        "sales_ops": {
            "meeting_notes": len(state.get("meeting_notes", [])) > 0,
            "task_extractor": len(state.get("crm_tasks", [])) > 0,
            "crm_updater": any(
                t.get("status") == "executed"
                for t in state.get("crm_tasks", [])
            ),
        },
        "approvals": {
            "pending": len([
                a for a in state.get("pending_approvals", [])
                if a.get("status") == "pending"
            ]),
            "approved": len([
                a for a in state.get("approval_history", [])
                if a.get("status") == "approved"
            ]),
            "rejected": len([
                a for a in state.get("approval_history", [])
                if a.get("status") == "rejected"
            ]),
        },
    }

    # Calculate percentages
    workflow_type = state["workflow_type"]

    if workflow_type == "meeting_analysis":
        # MVP: 3 steps
        steps = [
            bool(state.get("meeting_notes")),
            bool(state.get("crm_tasks")),
            any(t.get("status") == "executed" for t in state.get("crm_tasks", [])),
        ]
        progress["overall_percent"] = sum(steps) / len(steps) * 100

    elif workflow_type == "sales_ops_only":
        steps = list(progress["sales_ops"].values())
        progress["overall_percent"] = sum(steps) / len(steps) * 100

    elif workflow_type == "intelligence_only":
        steps = list(progress["intelligence"].values())
        progress["overall_percent"] = sum(steps) / len(steps) * 100

    elif workflow_type == "full_cycle":
        all_steps = (
            list(progress["intelligence"].values()) +
            list(progress["content"].values()) +
            list(progress["sales_ops"].values())
        )
        progress["overall_percent"] = sum(all_steps) / len(all_steps) * 100

    else:
        progress["overall_percent"] = 0

    return progress


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main routers
    "route_from_supervisor",
    "route_from_intelligence",
    "route_from_content",
    "route_from_sales_ops",
    # Approval routers
    "should_request_approval",
    "route_after_approval",
    # Utilities
    "get_next_pending_agent",
    "get_workflow_progress",
]

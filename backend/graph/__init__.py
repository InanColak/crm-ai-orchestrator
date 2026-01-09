"""
LangGraph Workflow Module
=========================
Multi-agent orchestration with LangGraph.

This module provides:
- OrchestratorState: Central state management
- Workflow builders: Different workflow configurations
- Node functions: Agent implementations
- Routing logic: Conditional edge functions
"""

from backend.graph.state import (
    # Enums
    WorkflowStatus,
    ApprovalType,
    CRMProvider,
    # State Types
    ClientContext,
    SEOData,
    MarketResearch,
    AudienceData,
    ContentDraft,
    ContentPipeline,
    MeetingNote,
    CRMTask,
    LeadData,
    EmailDraft,
    ApprovalRequest,
    AgentMessage,
    # Main State
    OrchestratorState,
    # Helpers
    create_initial_state,
    get_pending_approvals_by_type,
    get_tasks_by_status,
    count_pending_items,
)

from backend.graph.workflow import (
    # Checkpointer
    CheckpointerFactory,
    # Workflow Builders
    build_full_cycle_workflow,
    build_sales_ops_workflow,
    build_meeting_analysis_workflow,
    build_intelligence_workflow,
    build_content_workflow,
    # Compiler
    WORKFLOW_BUILDERS,
    compile_workflow,
    get_workflow,
    # Visualization
    get_workflow_graph_image,
    get_workflow_mermaid,
)

from backend.graph.routers import (
    route_from_supervisor,
    route_from_intelligence,
    route_from_content,
    route_from_sales_ops,
    should_request_approval,
    route_after_approval,
    get_next_pending_agent,
    get_workflow_progress,
)

__all__ = [
    # State Enums
    "WorkflowStatus",
    "ApprovalType",
    "CRMProvider",
    # State Types
    "ClientContext",
    "SEOData",
    "MarketResearch",
    "AudienceData",
    "ContentDraft",
    "ContentPipeline",
    "MeetingNote",
    "CRMTask",
    "LeadData",
    "EmailDraft",
    "ApprovalRequest",
    "AgentMessage",
    "OrchestratorState",
    # State Helpers
    "create_initial_state",
    "get_pending_approvals_by_type",
    "get_tasks_by_status",
    "count_pending_items",
    # Checkpointer
    "CheckpointerFactory",
    # Workflow Builders
    "build_full_cycle_workflow",
    "build_sales_ops_workflow",
    "build_meeting_analysis_workflow",
    "build_intelligence_workflow",
    "build_content_workflow",
    "WORKFLOW_BUILDERS",
    # Compiler
    "compile_workflow",
    "get_workflow",
    # Visualization
    "get_workflow_graph_image",
    "get_workflow_mermaid",
    # Routers
    "route_from_supervisor",
    "route_from_intelligence",
    "route_from_content",
    "route_from_sales_ops",
    "should_request_approval",
    "route_after_approval",
    "get_next_pending_agent",
    "get_workflow_progress",
]

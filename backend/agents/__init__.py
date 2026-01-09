"""
Agents Module
=============
LangGraph agent implementations for the CRM AI Orchestrator.

Squads:
- Intelligence: Market research, SEO analysis, web analysis, audience building
- Content: Pipeline management, content writing, SEO optimization, publishing
- Sales Ops: Meeting notes, task extraction, CRM updates, lead research, email
"""

from backend.agents.base import (
    # Context and Result
    AgentContext,
    AgentExecutionResult,
    # Base Classes
    BaseAgent,
    ResearchAgent,
    ContentAgent,
    SalesOpsAgent,
)

__all__ = [
    # Context and Result
    "AgentContext",
    "AgentExecutionResult",
    # Base Classes
    "BaseAgent",
    "ResearchAgent",
    "ContentAgent",
    "SalesOpsAgent",
]

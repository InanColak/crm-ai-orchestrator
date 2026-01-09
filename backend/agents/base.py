"""
Base Agent Architecture
=======================
Abstract base class for all LangGraph agents in the orchestration system.

Design Principles (PIVO):
- Problem: Prevent hallucination, ensure state consistency, handle errors gracefully
- Insights: Stateful logic through OrchestratorState, structured outputs via Pydantic
- Voice: Data-driven decisions from RAG/CRM, not general LLM knowledge
- Outcome: Reliable, validated outputs with automatic HITL routing

All agents must:
1. Inherit from BaseAgent
2. Define an output schema (Pydantic model)
3. Implement process() method
4. Return state updates (never mutate state directly)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar, Type
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from backend.graph.state import OrchestratorState, AgentMessage, ApprovalRequest, ApprovalType

logger = logging.getLogger(__name__)

# Type variable for agent output schemas
T = TypeVar("T", bound=BaseModel)


# =============================================================================
# AGENT EXECUTION CONTEXT
# =============================================================================

class AgentContext(BaseModel):
    """
    Runtime context passed to agent during execution.
    Contains metadata and utilities for agent operations.
    """
    workflow_id: str
    client_id: str
    client_name: str
    trace_id: str | None = None
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # RAG context (retrieved documents)
    retrieved_context: str | None = None

    # Additional metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentExecutionResult(BaseModel, Generic[T]):
    """
    Standardized result from agent execution.
    Includes output, metrics, and any errors.
    """
    success: bool = True
    output: T | None = None
    error_message: str | None = None

    # Execution metrics
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: int = 0

    # State updates to apply
    state_updates: dict[str, Any] = Field(default_factory=dict)

    # Messages for agent communication
    messages: list[AgentMessage] = Field(default_factory=list)

    # Approval requests (if HITL required)
    approval_requests: list[ApprovalRequest] = Field(default_factory=list)


# =============================================================================
# BASE AGENT ABSTRACT CLASS
# =============================================================================

class BaseAgent(ABC, Generic[T]):
    """
    Abstract Base Agent for LangGraph workflow nodes.

    All agents in the system inherit from this class to ensure:
    - Consistent interface for workflow integration
    - Structured output validation with Pydantic
    - Proper error handling and logging
    - State immutability (returns updates, never mutates)
    - HITL gateway support

    Type Parameters:
        T: The Pydantic model type for agent output

    Example:
        >>> class MeetingAnalysis(BaseModel):
        ...     summary: str
        ...     action_items: list[dict]
        ...
        >>> class MeetingNotesAgent(BaseAgent[MeetingAnalysis]):
        ...     name = "meeting_notes"
        ...     description = "Analyzes meeting transcripts"
        ...     output_schema = MeetingAnalysis
        ...
        ...     def get_system_prompt(self) -> str:
        ...         return "You are a meeting analyst..."
        ...
        ...     async def process(self, state, context):
        ...         # Process logic here
        ...         return AgentExecutionResult(output=analysis)
    """

    # Agent identification (override in subclass)
    name: str = "base_agent"
    description: str = "Base agent - override in subclass"
    squad: str = "unknown"  # "intelligence", "content", "sales_ops"

    # Output schema (must be overridden)
    output_schema: Type[T] = None

    # Configuration
    max_retries: int = 3
    timeout_seconds: int = 60
    requires_approval: bool = False
    approval_type: ApprovalType | None = None

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool] | None = None,
        prompt_template: ChatPromptTemplate | None = None,
    ):
        """
        Initialize the agent.

        Args:
            llm: LangChain chat model (Claude, GPT, etc.)
            tools: Optional list of tools the agent can use
            prompt_template: Optional custom prompt template
        """
        self.llm = llm
        self.tools = tools or []
        self._prompt_template = prompt_template
        self._output_parser: PydanticOutputParser | None = None

        # Validate output schema is defined
        if self.output_schema is None:
            raise ValueError(
                f"Agent '{self.name}' must define output_schema class attribute"
            )

        # Initialize output parser
        self._output_parser = PydanticOutputParser(pydantic_object=self.output_schema)

        logger.info(f"Initialized agent: {self.name} ({self.squad})")

    # =========================================================================
    # ABSTRACT METHODS (Must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Return the system prompt for this agent.

        The prompt should:
        - Define the agent's role and capabilities
        - Include output format instructions
        - Reference any RAG context or tools available

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    async def process(
        self,
        state: OrchestratorState,
        context: AgentContext,
    ) -> AgentExecutionResult[T]:
        """
        Main processing logic for the agent.

        This method should:
        1. Extract relevant data from state
        2. Prepare the prompt with context
        3. Call the LLM with structured output
        4. Validate and return results

        Args:
            state: Current orchestrator state (read-only)
            context: Execution context with metadata

        Returns:
            AgentExecutionResult containing output and state updates

        Note:
            Never mutate state directly. Return updates in state_updates dict.
        """
        pass

    # =========================================================================
    # PROMPT CONSTRUCTION
    # =========================================================================

    def get_prompt_template(self) -> ChatPromptTemplate:
        """
        Get or build the prompt template for this agent.

        Can be overridden for custom prompt structures.
        """
        if self._prompt_template:
            return self._prompt_template

        # Build default template
        return ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("system", "Output Format Instructions:\n{format_instructions}"),
            MessagesPlaceholder(variable_name="context_messages", optional=True),
            ("human", "{input}"),
        ])

    def get_format_instructions(self) -> str:
        """Get Pydantic output format instructions."""
        return self._output_parser.get_format_instructions()

    # =========================================================================
    # LLM INTERACTION
    # =========================================================================

    async def invoke_llm(
        self,
        input_text: str,
        context_messages: list[Any] | None = None,
        use_structured_output: bool = True,
    ) -> tuple[T | str, dict[str, int]]:
        """
        Invoke the LLM with the given input.

        Args:
            input_text: The main input/query for the agent
            context_messages: Optional additional context messages
            use_structured_output: Whether to parse output to Pydantic model

        Returns:
            Tuple of (parsed output or raw string, token usage dict)

        Raises:
            ValueError: If structured output parsing fails
        """
        prompt = self.get_prompt_template()

        # Build chain with structured output if schema defined
        if use_structured_output and self.output_schema:
            chain = prompt | self.llm.with_structured_output(self.output_schema)
        else:
            chain = prompt | self.llm

        # Invoke chain
        result = await chain.ainvoke({
            "input": input_text,
            "format_instructions": self.get_format_instructions(),
            "context_messages": context_messages or [],
        })

        # Extract token usage if available
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # Return result with token usage
        if use_structured_output:
            return result, token_usage
        else:
            return result.content if hasattr(result, 'content') else str(result), token_usage

    async def invoke_with_tools(
        self,
        input_text: str,
        context_messages: list[Any] | None = None,
    ) -> tuple[Any, dict[str, int]]:
        """
        Invoke LLM with tool calling capabilities.

        Args:
            input_text: The main input/query
            context_messages: Optional additional context

        Returns:
            Tuple of (agent response, token usage dict)
        """
        if not self.tools:
            return await self.invoke_llm(input_text, context_messages, use_structured_output=False)

        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(self.tools)
        prompt = self.get_prompt_template()

        chain = prompt | llm_with_tools

        result = await chain.ainvoke({
            "input": input_text,
            "format_instructions": self.get_format_instructions(),
            "context_messages": context_messages or [],
        })

        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        return result, token_usage

    # =========================================================================
    # STATE UPDATE HELPERS
    # =========================================================================

    def create_execution_log(
        self,
        action: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a standardized execution log entry."""
        return {
            "agent": self.name,
            "squad": self.squad,
            "action": action,
            "details": details or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def create_message(
        self,
        content: str,
        message_type: str = "info",
        to_agent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentMessage:
        """Create a standardized agent message."""
        return AgentMessage(
            message_id=str(uuid4()),
            from_agent=self.name,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            metadata=metadata or {},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def create_approval_request(
        self,
        title: str,
        description: str,
        payload: dict[str, Any],
        approval_type: ApprovalType | None = None,
    ) -> ApprovalRequest:
        """Create a HITL approval request."""
        return ApprovalRequest(
            approval_id=str(uuid4()),
            approval_type=approval_type or self.approval_type or ApprovalType.CRM_UPDATE,
            title=title,
            description=description,
            payload=payload,
            requested_at=datetime.now(timezone.utc).isoformat(),
            requested_by=self.name,
            status="pending",
            reviewed_at=None,
            reviewed_by=None,
            rejection_reason=None,
        )

    # =========================================================================
    # NODE WRAPPER (For LangGraph integration)
    # =========================================================================

    async def __call__(self, state: OrchestratorState) -> dict[str, Any]:
        """
        LangGraph node entry point.

        This makes the agent callable as a node function.
        Wraps process() with error handling and logging.

        Args:
            state: Current orchestrator state

        Returns:
            Dictionary of state updates
        """
        # Create execution context
        context = AgentContext(
            workflow_id=state["workflow_id"],
            client_id=state["client"]["client_id"],
            client_name=state["client"]["client_name"],
            trace_id=state.get("trace_id"),
        )

        # Add RAG context if available
        if state.get("brandvoice_context"):
            context.retrieved_context = state["brandvoice_context"]

        logger.info(f"[{self.name}] Starting execution for workflow {context.workflow_id}")

        start_time = datetime.now(timezone.utc)

        try:
            # Execute agent logic
            result = await self.process(state, context)

            # Calculate duration
            duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

            # Build state updates
            updates: dict[str, Any] = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "agent_execution_log": [
                    self.create_execution_log(
                        action="process_complete",
                        details={
                            "success": result.success,
                            "duration_ms": duration_ms,
                            "tokens_used": result.tokens_used,
                        }
                    )
                ],
            }

            # Add messages if any
            if result.messages:
                updates["messages"] = result.messages

            # Add approval requests if any
            if result.approval_requests:
                updates["pending_approvals"] = result.approval_requests

            # Merge agent-specific state updates
            updates.update(result.state_updates)

            logger.info(f"[{self.name}] Completed in {duration_ms}ms")

            return updates

        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}", exc_info=True)

            # Return error state
            return {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "error_message": f"Agent {self.name} failed: {str(e)}",
                "agent_execution_log": [
                    self.create_execution_log(
                        action="process_error",
                        details={"error": str(e)}
                    )
                ],
                "messages": [
                    self.create_message(
                        content=f"Error in {self.name}: {str(e)}",
                        message_type="error",
                    )
                ],
            }


# =============================================================================
# SPECIALIZED BASE CLASSES
# =============================================================================

class ResearchAgent(BaseAgent[T]):
    """
    Base class for research/analysis agents (Intelligence Squad).

    These agents:
    - Read from external sources (web, APIs)
    - Analyze and synthesize information
    - Don't require HITL approval (read-only)
    """
    squad: str = "intelligence"
    requires_approval: bool = False


class ContentAgent(BaseAgent[T]):
    """
    Base class for content generation agents (Content Squad).

    These agents:
    - Generate content using RAG context
    - May require approval before publishing
    - Use brandvoice context for consistency
    """
    squad: str = "content"
    requires_approval: bool = True
    approval_type: ApprovalType = ApprovalType.CONTENT_PUBLISH


class SalesOpsAgent(BaseAgent[T]):
    """
    Base class for sales operations agents (Sales Ops Squad).

    These agents:
    - Interact with CRM systems
    - Process meeting notes, leads, tasks
    - Require HITL approval for write operations
    """
    squad: str = "sales_ops"
    requires_approval: bool = True
    approval_type: ApprovalType = ApprovalType.CRM_UPDATE


# =============================================================================
# EXPORTS
# =============================================================================

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

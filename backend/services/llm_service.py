"""
LLM Service - AI Model Integration
==================================
Centralized service for LLM (Claude/GPT) interactions with:
- Multi-provider support (Anthropic, OpenAI)
- Automatic fallback between providers
- Rate limiting and retry logic
- Token usage tracking
- Structured output support

FLOW Methodology:
- Function: Manage all LLM API interactions
- Level: Production-ready with async/await
- Output: Type-safe responses with Pydantic validation
- Win Metric: < 1% error rate, automatic fallback
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, Type, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# ENUMS AND MODELS
# =============================================================================

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class LLMModel(str, Enum):
    """Available models by provider."""
    # Anthropic Claude
    CLAUDE_SONNET = "claude-sonnet-4-20250514"
    CLAUDE_OPUS = "claude-opus-4-20250514"
    CLAUDE_HAIKU = "claude-3-5-haiku-20241022"

    # OpenAI GPT
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    GPT4_TURBO = "gpt-4-turbo"


class TokenUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class LLMResponse(BaseModel):
    """Standardized LLM response."""
    content: str
    provider: LLMProvider
    model: str
    usage: TokenUsage
    latency_ms: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LLMError(Exception):
    """Base exception for LLM operations."""

    def __init__(
        self,
        message: str,
        provider: LLMProvider | None = None,
        model: str | None = None,
        retryable: bool = True,
    ):
        self.message = message
        self.provider = provider
        self.model = model
        self.retryable = retryable
        super().__init__(self.message)


class RateLimitError(LLMError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: int | None = None, **kwargs):
        super().__init__(message, retryable=True, **kwargs)
        self.retry_after = retry_after


class TokenLimitError(LLMError):
    """Raised when token limit is exceeded."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, retryable=False, **kwargs)


# =============================================================================
# PRICING (USD per 1M tokens)
# =============================================================================

PRICING = {
    # Anthropic Claude (per 1M tokens)
    LLMModel.CLAUDE_SONNET: {"input": 3.0, "output": 15.0},
    LLMModel.CLAUDE_OPUS: {"input": 15.0, "output": 75.0},
    LLMModel.CLAUDE_HAIKU: {"input": 0.25, "output": 1.25},
    # OpenAI GPT (per 1M tokens)
    LLMModel.GPT4O: {"input": 2.5, "output": 10.0},
    LLMModel.GPT4O_MINI: {"input": 0.15, "output": 0.6},
    LLMModel.GPT4_TURBO: {"input": 10.0, "output": 30.0},
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate estimated cost in USD."""
    try:
        model_enum = LLMModel(model)
        pricing = PRICING.get(model_enum, {"input": 0, "output": 0})
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 6)
    except (ValueError, KeyError):
        return 0.0


# =============================================================================
# RATE LIMITER
# =============================================================================

class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for LLM API calls.

    Ensures we don't exceed provider rate limits.
    """

    def __init__(self, tokens_per_minute: int, max_burst: int | None = None):
        self.tokens_per_minute = tokens_per_minute
        self.max_burst = max_burst or tokens_per_minute
        self._tokens = float(self.max_burst)
        self._last_update = datetime.now(timezone.utc)
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False if rate limited
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            elapsed = (now - self._last_update).total_seconds()

            # Refill tokens based on time elapsed
            refill = elapsed * (self.tokens_per_minute / 60.0)
            self._tokens = min(self.max_burst, self._tokens + refill)
            self._last_update = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            return False

    async def wait_for_tokens(self, tokens: int = 1, max_wait: float = 30.0) -> bool:
        """
        Wait until tokens are available.

        Args:
            tokens: Number of tokens needed
            max_wait: Maximum seconds to wait

        Returns:
            True if tokens acquired, False if timeout
        """
        start = datetime.now(timezone.utc)

        while True:
            if await self.acquire(tokens):
                return True

            elapsed = (datetime.now(timezone.utc) - start).total_seconds()
            if elapsed >= max_wait:
                return False

            # Wait a bit before retrying
            await asyncio.sleep(0.5)


# =============================================================================
# LLM SERVICE
# =============================================================================

class LLMService:
    """
    Centralized LLM Service for AI model interactions.

    Features:
    - Multi-provider support (Anthropic, OpenAI)
    - Automatic fallback on errors
    - Rate limiting per provider
    - Retry logic with exponential backoff
    - Token usage tracking
    - Structured output with Pydantic

    Usage:
        >>> service = LLMService()
        >>> response = await service.generate("What is 2+2?")
        >>> structured = await service.generate_structured(MyModel, "Extract info from: ...")
    """

    _instance: LLMService | None = None

    def __init__(self):
        self._settings = get_settings()
        self._clients: dict[LLMProvider, BaseChatModel] = {}
        self._rate_limiters: dict[LLMProvider, TokenBucketRateLimiter] = {}
        self._initialized = False

    @classmethod
    def get_instance(cls) -> LLMService:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize LLM clients and rate limiters."""
        if self._initialized:
            return

        settings = self._settings

        # Initialize Anthropic client
        if settings.anthropic_api_key:
            try:
                from langchain_anthropic import ChatAnthropic

                self._clients[LLMProvider.ANTHROPIC] = ChatAnthropic(
                    model=settings.default_model,
                    anthropic_api_key=settings.anthropic_api_key,
                    max_tokens=4096,
                    temperature=0.7,
                )
                self._rate_limiters[LLMProvider.ANTHROPIC] = TokenBucketRateLimiter(
                    tokens_per_minute=settings.llm_rate_limit_per_minute
                )
                logger.info("Anthropic (Claude) client initialized")
            except ImportError:
                logger.warning("langchain-anthropic not installed")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")

        # Initialize OpenAI client
        if settings.openai_api_key:
            try:
                from langchain_openai import ChatOpenAI

                self._clients[LLMProvider.OPENAI] = ChatOpenAI(
                    model="gpt-4o",
                    openai_api_key=settings.openai_api_key,
                    max_tokens=4096,
                    temperature=0.7,
                )
                self._rate_limiters[LLMProvider.OPENAI] = TokenBucketRateLimiter(
                    tokens_per_minute=settings.llm_rate_limit_per_minute
                )
                logger.info("OpenAI (GPT) client initialized")
            except ImportError:
                logger.warning("langchain-openai not installed")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")

        if not self._clients:
            logger.warning("No LLM providers configured! Set ANTHROPIC_API_KEY or OPENAI_API_KEY")

        self._initialized = True

    @property
    def primary_provider(self) -> LLMProvider:
        """Get the primary (preferred) provider."""
        provider_str = self._settings.default_llm_provider
        if provider_str == "anthropic" and LLMProvider.ANTHROPIC in self._clients:
            return LLMProvider.ANTHROPIC
        if provider_str == "openai" and LLMProvider.OPENAI in self._clients:
            return LLMProvider.OPENAI
        # Return first available
        if self._clients:
            return next(iter(self._clients.keys()))
        raise LLMError("No LLM providers available", retryable=False)

    @property
    def fallback_provider(self) -> LLMProvider | None:
        """Get the fallback provider."""
        primary = self.primary_provider
        for provider in self._clients:
            if provider != primary:
                return provider
        return None

    def get_client(self, provider: LLMProvider | None = None) -> BaseChatModel:
        """
        Get LLM client for specified provider.

        Args:
            provider: Provider to use, or None for primary

        Returns:
            LangChain chat model
        """
        provider = provider or self.primary_provider
        if provider not in self._clients:
            raise LLMError(f"Provider {provider} not configured", provider=provider, retryable=False)
        return self._clients[provider]

    def get_client_with_model(
        self,
        model: str | LLMModel,
        provider: LLMProvider | None = None,
    ) -> BaseChatModel:
        """
        Get LLM client configured with specific model.

        Args:
            model: Model to use
            provider: Provider (inferred from model if not specified)

        Returns:
            Configured LangChain chat model
        """
        model_str = model.value if isinstance(model, LLMModel) else model

        # Infer provider from model name
        if provider is None:
            if "claude" in model_str.lower():
                provider = LLMProvider.ANTHROPIC
            elif "gpt" in model_str.lower():
                provider = LLMProvider.OPENAI
            else:
                provider = self.primary_provider

        base_client = self.get_client(provider)

        # Create new client with specified model
        if provider == LLMProvider.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model_str,
                anthropic_api_key=self._settings.anthropic_api_key,
                max_tokens=4096,
                temperature=0.7,
            )
        elif provider == LLMProvider.OPENAI:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_str,
                openai_api_key=self._settings.openai_api_key,
                max_tokens=4096,
                temperature=0.7,
            )

        return base_client

    # =========================================================================
    # CORE GENERATION METHODS
    # =========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((RateLimitError,)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        provider: LLMProvider | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Generate text completion from LLM.

        Args:
            prompt: User prompt/query
            system_prompt: Optional system prompt
            provider: LLM provider to use
            model: Specific model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with content and metadata

        Raises:
            LLMError: On generation failure
        """
        provider = provider or self.primary_provider

        # Check rate limit
        rate_limiter = self._rate_limiters.get(provider)
        if rate_limiter and not await rate_limiter.wait_for_tokens():
            raise RateLimitError(
                f"Rate limit exceeded for {provider}",
                provider=provider,
            )

        # Get or create client
        if model:
            client = self.get_client_with_model(model, provider)
        else:
            client = self.get_client(provider)

        # Apply optional parameters
        if temperature is not None:
            client = client.with_config({"temperature": temperature})

        # Build messages
        messages: list[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        # Generate
        start_time = datetime.now(timezone.utc)

        try:
            response = await client.ainvoke(messages)

            latency_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

            # Extract token usage
            usage_data = getattr(response, "usage_metadata", {}) or {}
            prompt_tokens = usage_data.get("input_tokens", 0)
            completion_tokens = usage_data.get("output_tokens", 0)

            usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                estimated_cost_usd=calculate_cost(
                    model or self._settings.default_model,
                    prompt_tokens,
                    completion_tokens,
                ),
            )

            return LLMResponse(
                content=response.content,
                provider=provider,
                model=model or self._settings.default_model,
                usage=usage,
                latency_ms=latency_ms,
            )

        except Exception as e:
            error_str = str(e).lower()

            # Check for rate limit errors
            if "rate" in error_str or "429" in error_str:
                raise RateLimitError(str(e), provider=provider, model=model)

            # Check for token limit errors
            if "token" in error_str and ("limit" in error_str or "exceed" in error_str):
                raise TokenLimitError(str(e), provider=provider, model=model)

            # Try fallback provider
            fallback = self.fallback_provider
            if fallback and fallback != provider:
                logger.warning(f"Primary provider {provider} failed, trying {fallback}")
                return await self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    provider=fallback,
                    model=None,  # Use default model for fallback
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            raise LLMError(str(e), provider=provider, model=model)

    async def generate_structured(
        self,
        output_schema: Type[T],
        prompt: str,
        system_prompt: str | None = None,
        provider: LLMProvider | None = None,
        model: str | None = None,
    ) -> tuple[T, TokenUsage]:
        """
        Generate structured output validated with Pydantic.

        Args:
            output_schema: Pydantic model class for output
            prompt: User prompt/query
            system_prompt: Optional system prompt
            provider: LLM provider to use
            model: Specific model to use

        Returns:
            Tuple of (parsed output, token usage)

        Raises:
            LLMError: On generation or parsing failure
        """
        provider = provider or self.primary_provider

        # Get client with structured output
        if model:
            client = self.get_client_with_model(model, provider)
        else:
            client = self.get_client(provider)

        # Use with_structured_output for reliable parsing
        structured_client = client.with_structured_output(output_schema)

        # Build messages
        messages: list[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        # Check rate limit
        rate_limiter = self._rate_limiters.get(provider)
        if rate_limiter and not await rate_limiter.wait_for_tokens():
            raise RateLimitError(f"Rate limit exceeded for {provider}", provider=provider)

        try:
            result = await structured_client.ainvoke(messages)

            # Token usage may not be available with structured output
            usage = TokenUsage()

            return result, usage

        except Exception as e:
            raise LLMError(f"Structured generation failed: {e}", provider=provider, model=model)

    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[Any],
        system_prompt: str | None = None,
        provider: LLMProvider | None = None,
        model: str | None = None,
    ) -> tuple[Any, TokenUsage]:
        """
        Generate with tool calling capabilities.

        Args:
            prompt: User prompt
            tools: List of LangChain tools
            system_prompt: Optional system prompt
            provider: LLM provider
            model: Specific model

        Returns:
            Tuple of (response with tool calls, token usage)
        """
        provider = provider or self.primary_provider

        if model:
            client = self.get_client_with_model(model, provider)
        else:
            client = self.get_client(provider)

        # Bind tools to client
        client_with_tools = client.bind_tools(tools)

        messages: list[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        # Check rate limit
        rate_limiter = self._rate_limiters.get(provider)
        if rate_limiter and not await rate_limiter.wait_for_tokens():
            raise RateLimitError(f"Rate limit exceeded for {provider}", provider=provider)

        try:
            result = await client_with_tools.ainvoke(messages)
            usage = TokenUsage()
            return result, usage

        except Exception as e:
            raise LLMError(f"Tool generation failed: {e}", provider=provider, model=model)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to count tokens for
            model: Model for tokenization (affects count)

        Returns:
            Estimated token count
        """
        try:
            import tiktoken

            # Use cl100k_base for Claude and GPT-4
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            # Rough estimation: ~4 characters per token
            return len(text) // 4

    def get_available_providers(self) -> list[LLMProvider]:
        """Get list of configured providers."""
        return list(self._clients.keys())

    def get_provider_status(self) -> dict[str, Any]:
        """Get status of all providers."""
        return {
            provider.value: {
                "available": True,
                "model": self._settings.default_model if provider == self.primary_provider else "default",
            }
            for provider in self._clients
        }

    async def health_check(self) -> dict[str, Any]:
        """
        Check health of LLM providers.

        Returns:
            Health status dict
        """
        results = {}

        for provider in self._clients:
            try:
                response = await self.generate(
                    prompt="Say 'OK' if you're working.",
                    provider=provider,
                    max_tokens=10,
                )
                results[provider.value] = {
                    "status": "healthy",
                    "latency_ms": response.latency_ms,
                }
            except Exception as e:
                results[provider.value] = {
                    "status": "unhealthy",
                    "error": str(e),
                }

        return {
            "providers": results,
            "primary": self.primary_provider.value if self._clients else None,
            "fallback": self.fallback_provider.value if self.fallback_provider else None,
        }


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

def get_llm_service() -> LLMService:
    """
    Dependency injection for LLM Service.

    Usage:
        @router.post("/generate")
        async def generate(
            request: GenerateRequest,
            llm: LLMService = Depends(get_llm_service)
        ):
            return await llm.generate(request.prompt)
    """
    return LLMService.get_instance()


@lru_cache
def get_default_client() -> BaseChatModel:
    """
    Get the default LLM client (cached).

    For simple use cases where you just need a client.
    """
    service = LLMService.get_instance()
    return service.get_client()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "LLMProvider",
    "LLMModel",
    # Models
    "TokenUsage",
    "LLMResponse",
    # Errors
    "LLMError",
    "RateLimitError",
    "TokenLimitError",
    # Service
    "LLMService",
    # DI
    "get_llm_service",
    "get_default_client",
]

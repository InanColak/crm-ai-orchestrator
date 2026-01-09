"""
Tavily Search Service - Web Research Integration
=================================================
Production-ready service for web search, news, and company research using Tavily API.

Features:
- Multiple search types: web, news, company research
- Async operations with rate limiting
- Caching for repeated queries
- Structured output with Pydantic validation
- Comprehensive error handling

FLOW Methodology:
- Function: Manage all web research operations via Tavily API
- Level: Production-ready with async/await
- Output: Type-safe SearchResult with Pydantic validation
- Win Metric: < 2s response time, 95% success rate
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND MODELS
# =============================================================================


class SearchDepth(str, Enum):
    """Search depth levels."""
    BASIC = "basic"        # Fast, less comprehensive
    ADVANCED = "advanced"  # Slower, more thorough


class SearchTopic(str, Enum):
    """Topic categories for search optimization."""
    GENERAL = "general"
    NEWS = "news"
    FINANCE = "finance"


class TavilyError(Exception):
    """Base exception for Tavily operations."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        retryable: bool = True,
    ):
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        super().__init__(self.message)


class RateLimitError(TavilyError):
    """Raised when Tavily rate limit is exceeded."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message, status_code=429, retryable=True)
        self.retry_after = retry_after


class AuthenticationError(TavilyError):
    """Raised when Tavily API key is invalid."""

    def __init__(self, message: str = "Invalid Tavily API key"):
        super().__init__(message, status_code=401, retryable=False)


class QuotaExceededError(TavilyError):
    """Raised when monthly quota is exceeded."""

    def __init__(self, message: str = "Tavily monthly quota exceeded"):
        super().__init__(message, status_code=402, retryable=False)


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================


class SearchResult(BaseModel):
    """Individual search result from Tavily."""
    title: str = Field(description="Result title")
    url: str = Field(description="Source URL")
    content: str = Field(description="Snippet/excerpt from the page")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Relevance score")
    published_date: str | None = Field(default=None, description="Publication date if available")

    @field_validator("content", mode="before")
    @classmethod
    def truncate_content(cls, v: str) -> str:
        """Truncate very long content."""
        max_length = 2000
        if len(v) > max_length:
            return v[:max_length] + "..."
        return v


class ImageResult(BaseModel):
    """Image search result."""
    url: str = Field(description="Image URL")
    description: str | None = Field(default=None, description="Image description")


class SearchResponse(BaseModel):
    """Complete search response from Tavily."""
    query: str = Field(description="Original search query")
    results: list[SearchResult] = Field(default_factory=list, description="Search results")
    images: list[ImageResult] = Field(default_factory=list, description="Image results if requested")
    answer: str | None = Field(default=None, description="AI-generated answer if requested")
    response_time_ms: int = Field(default=0, description="API response time in milliseconds")
    follow_up_questions: list[str] = Field(default_factory=list, description="Suggested follow-up questions")

    @property
    def result_count(self) -> int:
        """Number of results returned."""
        return len(self.results)

    @property
    def has_answer(self) -> bool:
        """Whether an AI answer was generated."""
        return self.answer is not None and len(self.answer) > 0


class CompanyResearchResult(BaseModel):
    """Structured company research result."""
    company_name: str = Field(description="Company name")
    query: str = Field(description="Research query")

    # Company overview
    description: str | None = Field(default=None, description="Company description")
    industry: str | None = Field(default=None, description="Industry/sector")
    website: str | None = Field(default=None, description="Official website")

    # Key information
    key_facts: list[str] = Field(default_factory=list, description="Key facts about the company")
    recent_news: list[SearchResult] = Field(default_factory=list, description="Recent news articles")
    competitors: list[str] = Field(default_factory=list, description="Known competitors")

    # Leadership (if found)
    key_people: list[dict[str, str]] = Field(default_factory=list, description="Key people (name, role)")

    # Social presence
    social_links: dict[str, str] = Field(default_factory=dict, description="Social media links")

    # Sources
    sources: list[str] = Field(default_factory=list, description="Source URLs used")
    research_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When research was conducted"
    )


class NewsSearchResult(BaseModel):
    """News-specific search result."""
    query: str = Field(description="Search query")
    results: list[SearchResult] = Field(default_factory=list, description="News articles")
    total_results: int = Field(default=0, description="Total results found")
    time_range: str | None = Field(default=None, description="Time range for news")

    @property
    def latest_article(self) -> SearchResult | None:
        """Get the most recent article."""
        if self.results:
            return self.results[0]
        return None


# =============================================================================
# RATE LIMITER (Reused pattern from llm_service)
# =============================================================================


class TokenBucketRateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, requests_per_minute: int, max_burst: int | None = None):
        self.requests_per_minute = requests_per_minute
        self.max_burst = max_burst or min(requests_per_minute, 10)
        self._tokens = float(self.max_burst)
        self._last_update = datetime.now(timezone.utc)
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire a token from the bucket."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            elapsed = (now - self._last_update).total_seconds()

            # Refill tokens
            refill = elapsed * (self.requests_per_minute / 60.0)
            self._tokens = min(self.max_burst, self._tokens + refill)
            self._last_update = now

            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False

    async def wait_for_token(self, max_wait: float = 30.0) -> bool:
        """Wait until a token is available."""
        start = datetime.now(timezone.utc)

        while True:
            if await self.acquire():
                return True

            elapsed = (datetime.now(timezone.utc) - start).total_seconds()
            if elapsed >= max_wait:
                return False

            await asyncio.sleep(0.5)


# =============================================================================
# SIMPLE CACHE
# =============================================================================


class SimpleCache:
    """Simple in-memory cache with TTL."""

    def __init__(self, ttl_seconds: int = 300):
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()

    @staticmethod
    def _hash_key(key: str) -> str:
        """Create hash for cache key."""
        return hashlib.md5(key.encode()).hexdigest()

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        async with self._lock:
            hashed = self._hash_key(key)
            if hashed in self._cache:
                value, timestamp = self._cache[hashed]
                if (datetime.now(timezone.utc) - timestamp).total_seconds() < self._ttl:
                    return value
                # Expired, remove it
                del self._cache[hashed]
            return None

    async def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        async with self._lock:
            hashed = self._hash_key(key)
            self._cache[hashed] = (value, datetime.now(timezone.utc))

    async def clear(self) -> None:
        """Clear all cached values."""
        async with self._lock:
            self._cache.clear()


# =============================================================================
# TAVILY SERVICE
# =============================================================================


class TavilyService:
    """
    Tavily Search Service for web research operations.

    Features:
    - Web search with customizable depth
    - News search with time filtering
    - Company research (aggregated search)
    - Rate limiting and caching
    - Retry logic with exponential backoff

    Usage:
        >>> service = TavilyService.get_instance()
        >>> results = await service.search("AI trends 2024")
        >>> news = await service.search_news("OpenAI", days=7)
        >>> company = await service.research_company("Anthropic")
    """

    _instance: TavilyService | None = None

    # Tavily API base URL
    BASE_URL = "https://api.tavily.com"

    def __init__(self):
        self._settings = get_settings()
        self._api_key = self._settings.tavily_api_key
        self._client: httpx.AsyncClient | None = None
        self._rate_limiter = TokenBucketRateLimiter(
            requests_per_minute=self._settings.tavily_rate_limit_per_minute,
            max_burst=10
        )
        self._cache = SimpleCache(ttl_seconds=self._settings.tavily_cache_ttl_seconds)
        self._initialized = False

    @classmethod
    def get_instance(cls) -> TavilyService:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the service."""
        if self._initialized:
            return

        if not self._api_key:
            logger.warning("Tavily API key not configured. Search functionality will be unavailable.")
        else:
            logger.info("Tavily service initialized")

        self._initialized = True

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                headers={
                    "Content-Type": "application/json",
                }
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _check_api_key(self) -> None:
        """Verify API key is configured."""
        if not self._api_key:
            raise TavilyError(
                "Tavily API key not configured. Set TAVILY_API_KEY environment variable.",
                retryable=False
            )

    async def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle API response and errors."""
        if response.status_code == 200:
            return response.json()

        # Handle specific error codes
        if response.status_code == 401:
            raise AuthenticationError()

        if response.status_code == 402:
            raise QuotaExceededError()

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                "Tavily rate limit exceeded",
                retry_after=int(retry_after) if retry_after else None
            )

        # Generic error
        try:
            error_data = response.json()
            message = error_data.get("message", response.text)
        except Exception:
            message = response.text

        raise TavilyError(
            f"Tavily API error ({response.status_code}): {message}",
            status_code=response.status_code,
            retryable=response.status_code >= 500
        )

    # =========================================================================
    # CORE SEARCH METHODS
    # =========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((RateLimitError, httpx.TimeoutException)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def search(
        self,
        query: str,
        search_depth: SearchDepth = SearchDepth.BASIC,
        topic: SearchTopic = SearchTopic.GENERAL,
        max_results: int = 10,
        include_answer: bool = True,
        include_images: bool = False,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        use_cache: bool = True,
    ) -> SearchResponse:
        """
        Perform a web search using Tavily API.

        Args:
            query: Search query string
            search_depth: Basic (fast) or Advanced (thorough)
            topic: Topic category for optimization
            max_results: Maximum results to return (1-20)
            include_answer: Generate AI summary answer
            include_images: Include image results
            include_domains: Only search these domains
            exclude_domains: Exclude these domains
            use_cache: Use cached results if available

        Returns:
            SearchResponse with results and optional AI answer

        Raises:
            TavilyError: On API errors

        Example:
            >>> results = await service.search(
            ...     "best CRM software 2024",
            ...     search_depth=SearchDepth.ADVANCED,
            ...     max_results=5
            ... )
        """
        self._check_api_key()

        # Check cache
        cache_key = f"search:{query}:{search_depth}:{topic}:{max_results}"
        if use_cache:
            cached = await self._cache.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached

        # Rate limit check
        if not await self._rate_limiter.wait_for_token():
            raise RateLimitError("Local rate limit exceeded")

        # Prepare request
        payload: dict[str, Any] = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": search_depth.value,
            "topic": topic.value,
            "max_results": min(max_results, 20),
            "include_answer": include_answer,
            "include_images": include_images,
        }

        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        # Make request
        start_time = datetime.now(timezone.utc)
        client = await self._get_client()

        response = await client.post(
            f"{self.BASE_URL}/search",
            json=payload
        )

        data = await self._handle_response(response)
        response_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

        # Parse results
        results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                published_date=r.get("published_date"),
            )
            for r in data.get("results", [])
        ]

        images = [
            ImageResult(
                url=img.get("url", ""),
                description=img.get("description"),
            )
            for img in data.get("images", [])
        ]

        search_response = SearchResponse(
            query=query,
            results=results,
            images=images,
            answer=data.get("answer"),
            response_time_ms=response_time,
            follow_up_questions=data.get("follow_up_questions", []),
        )

        # Cache result
        if use_cache:
            await self._cache.set(cache_key, search_response)

        logger.info(f"Search completed: '{query[:50]}...' - {len(results)} results in {response_time}ms")

        return search_response

    async def search_news(
        self,
        query: str,
        days: int = 7,
        max_results: int = 10,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> NewsSearchResult:
        """
        Search for recent news articles.

        Args:
            query: News search query
            days: Look back period in days (1-30)
            max_results: Maximum results
            include_domains: Only these news sources
            exclude_domains: Exclude these sources

        Returns:
            NewsSearchResult with recent articles

        Example:
            >>> news = await service.search_news("AI regulation", days=7)
        """
        # Use news topic for optimization
        response = await self.search(
            query=query,
            topic=SearchTopic.NEWS,
            search_depth=SearchDepth.ADVANCED,
            max_results=max_results,
            include_answer=False,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )

        return NewsSearchResult(
            query=query,
            results=response.results,
            total_results=len(response.results),
            time_range=f"Last {days} days",
        )

    async def research_company(
        self,
        company_name: str,
        include_news: bool = True,
        include_competitors: bool = True,
    ) -> CompanyResearchResult:
        """
        Conduct comprehensive company research.

        Performs multiple searches to gather:
        - Company overview and description
        - Recent news
        - Key people (if available)
        - Competitors (if requested)

        Args:
            company_name: Company to research
            include_news: Include recent news search
            include_competitors: Search for competitors

        Returns:
            CompanyResearchResult with aggregated information

        Example:
            >>> company = await service.research_company("Anthropic")
        """
        self._check_api_key()

        # Parallel searches for efficiency
        tasks = [
            # Main company search
            self.search(
                f"{company_name} company overview",
                search_depth=SearchDepth.ADVANCED,
                max_results=5,
                include_answer=True,
            ),
        ]

        if include_news:
            tasks.append(
                self.search_news(f"{company_name}", days=30, max_results=5)
            )

        if include_competitors:
            tasks.append(
                self.search(
                    f"{company_name} competitors alternatives",
                    search_depth=SearchDepth.BASIC,
                    max_results=5,
                    include_answer=True,
                )
            )

        # Execute searches
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process main search
        main_result = results[0] if not isinstance(results[0], Exception) else None

        # Process news
        news_results: list[SearchResult] = []
        if include_news and len(results) > 1 and not isinstance(results[1], Exception):
            news_results = results[1].results

        # Process competitors
        competitors: list[str] = []
        if include_competitors:
            idx = 2 if include_news else 1
            if len(results) > idx and not isinstance(results[idx], Exception):
                comp_result = results[idx]
                if comp_result.answer:
                    # Extract competitor names from answer (simplified)
                    competitors = [
                        line.strip("- •")
                        for line in comp_result.answer.split("\n")
                        if line.strip() and company_name.lower() not in line.lower()
                    ][:5]

        # Build result
        sources = []
        if main_result:
            sources.extend([r.url for r in main_result.results])

        return CompanyResearchResult(
            company_name=company_name,
            query=f"{company_name} company research",
            description=main_result.answer if main_result else None,
            key_facts=[r.content[:200] for r in (main_result.results if main_result else [])[:3]],
            recent_news=news_results,
            competitors=competitors,
            sources=sources[:10],
        )

    async def quick_answer(
        self,
        question: str,
    ) -> str | None:
        """
        Get a quick AI-generated answer to a question.

        Args:
            question: Question to answer

        Returns:
            AI-generated answer or None if unavailable

        Example:
            >>> answer = await service.quick_answer("What is LangGraph?")
        """
        response = await self.search(
            query=question,
            search_depth=SearchDepth.BASIC,
            max_results=3,
            include_answer=True,
        )
        return response.answer

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @property
    def is_configured(self) -> bool:
        """Check if Tavily API key is configured."""
        return bool(self._api_key)

    async def health_check(self) -> dict[str, Any]:
        """
        Check Tavily API health.

        Returns:
            Health status dict
        """
        if not self.is_configured:
            return {
                "status": "unconfigured",
                "message": "TAVILY_API_KEY not set",
            }

        try:
            response = await self.search(
                "test query",
                search_depth=SearchDepth.BASIC,
                max_results=1,
                include_answer=False,
                use_cache=False,
            )
            return {
                "status": "healthy",
                "response_time_ms": response.response_time_ms,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    async def clear_cache(self) -> None:
        """Clear the search cache."""
        await self._cache.clear()
        logger.info("Tavily cache cleared")


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================


def get_tavily_service() -> TavilyService:
    """
    Dependency injection for Tavily Service.

    Usage:
        @router.get("/search")
        async def search(
            query: str,
            tavily: TavilyService = Depends(get_tavily_service)
        ):
            return await tavily.search(query)
    """
    return TavilyService.get_instance()


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "SearchDepth",
    "SearchTopic",
    # Models
    "SearchResult",
    "ImageResult",
    "SearchResponse",
    "CompanyResearchResult",
    "NewsSearchResult",
    # Errors
    "TavilyError",
    "RateLimitError",
    "AuthenticationError",
    "QuotaExceededError",
    # Service
    "TavilyService",
    # DI
    "get_tavily_service",
]

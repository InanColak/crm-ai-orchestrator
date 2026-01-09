"""
Unit Tests for Tavily Search Service
====================================
Comprehensive tests for web search, news search, and company research.

Test Coverage:
- Schema validation (SearchResult, SearchResponse, etc.)
- Service initialization and configuration
- Search operations with mocked API
- Rate limiting behavior
- Caching behavior
- Error handling (auth, rate limit, quota)
- Company research aggregation
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from backend.services.tavily_service import (
    # Enums
    SearchDepth,
    SearchTopic,
    # Models
    SearchResult,
    ImageResult,
    SearchResponse,
    CompanyResearchResult,
    NewsSearchResult,
    # Errors
    TavilyError,
    RateLimitError,
    AuthenticationError,
    QuotaExceededError,
    # Service
    TavilyService,
    TokenBucketRateLimiter,
    SimpleCache,
    get_tavily_service,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_search_response():
    """Mock Tavily API search response."""
    return {
        "query": "test query",
        "results": [
            {
                "title": "Test Result 1",
                "url": "https://example.com/1",
                "content": "This is the first test result content.",
                "score": 0.95,
                "published_date": "2024-01-15",
            },
            {
                "title": "Test Result 2",
                "url": "https://example.com/2",
                "content": "This is the second test result content.",
                "score": 0.85,
            },
            {
                "title": "Test Result 3",
                "url": "https://example.com/3",
                "content": "This is the third test result content.",
                "score": 0.75,
            },
        ],
        "images": [
            {"url": "https://example.com/image1.jpg", "description": "Test image"},
        ],
        "answer": "This is an AI-generated answer summarizing the search results.",
        "follow_up_questions": [
            "What are the benefits?",
            "How does it compare to alternatives?",
        ],
    }


@pytest.fixture
def mock_company_search_response():
    """Mock Tavily API response for company search."""
    return {
        "query": "Anthropic company overview",
        "results": [
            {
                "title": "Anthropic - AI Safety Company",
                "url": "https://anthropic.com/about",
                "content": "Anthropic is an AI safety company that builds reliable AI systems.",
                "score": 0.98,
            },
        ],
        "answer": "Anthropic is an AI safety startup founded in 2021, known for creating Claude AI assistant.",
    }


@pytest.fixture
def tavily_service():
    """Create TavilyService instance with mocked settings."""
    with patch("backend.services.tavily_service.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            tavily_api_key="test-api-key",
            tavily_rate_limit_per_minute=60,
            tavily_cache_ttl_seconds=300,
        )
        # Reset singleton
        TavilyService._instance = None
        service = TavilyService.get_instance()
        return service


# =============================================================================
# SCHEMA TESTS
# =============================================================================


class TestSearchResultSchema:
    """Tests for SearchResult Pydantic model."""

    def test_valid_search_result(self):
        """Should create valid SearchResult."""
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            content="Test content",
            score=0.85,
            published_date="2024-01-15",
        )

        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.content == "Test content"
        assert result.score == 0.85
        assert result.published_date == "2024-01-15"

    def test_search_result_content_truncation(self):
        """Should truncate very long content."""
        long_content = "x" * 3000
        result = SearchResult(
            title="Test",
            url="https://example.com",
            content=long_content,
        )

        assert len(result.content) <= 2003  # 2000 + "..."
        assert result.content.endswith("...")

    def test_search_result_default_values(self):
        """Should use default values for optional fields."""
        result = SearchResult(
            title="Test",
            url="https://example.com",
            content="Content",
        )

        assert result.score == 0.0
        assert result.published_date is None

    def test_search_result_score_bounds(self):
        """Score should be between 0 and 1."""
        # Valid score
        result = SearchResult(
            title="Test",
            url="https://example.com",
            content="Content",
            score=0.5,
        )
        assert result.score == 0.5

        # Score at bounds
        result_min = SearchResult(
            title="Test",
            url="https://example.com",
            content="Content",
            score=0.0,
        )
        assert result_min.score == 0.0

        result_max = SearchResult(
            title="Test",
            url="https://example.com",
            content="Content",
            score=1.0,
        )
        assert result_max.score == 1.0


class TestSearchResponseSchema:
    """Tests for SearchResponse Pydantic model."""

    def test_valid_search_response(self):
        """Should create valid SearchResponse."""
        response = SearchResponse(
            query="test query",
            results=[
                SearchResult(title="R1", url="https://a.com", content="C1"),
                SearchResult(title="R2", url="https://b.com", content="C2"),
            ],
            answer="This is the answer.",
            response_time_ms=150,
        )

        assert response.query == "test query"
        assert len(response.results) == 2
        assert response.answer == "This is the answer."
        assert response.response_time_ms == 150

    def test_result_count_property(self):
        """result_count property should return correct count."""
        response = SearchResponse(
            query="test",
            results=[
                SearchResult(title="R1", url="https://a.com", content="C1"),
                SearchResult(title="R2", url="https://b.com", content="C2"),
                SearchResult(title="R3", url="https://c.com", content="C3"),
            ],
        )

        assert response.result_count == 3

    def test_has_answer_property(self):
        """has_answer property should detect answer presence."""
        with_answer = SearchResponse(
            query="test",
            results=[],
            answer="Some answer",
        )
        assert with_answer.has_answer is True

        without_answer = SearchResponse(
            query="test",
            results=[],
            answer=None,
        )
        assert without_answer.has_answer is False

        empty_answer = SearchResponse(
            query="test",
            results=[],
            answer="",
        )
        assert empty_answer.has_answer is False


class TestCompanyResearchResultSchema:
    """Tests for CompanyResearchResult schema."""

    def test_valid_company_result(self):
        """Should create valid CompanyResearchResult."""
        result = CompanyResearchResult(
            company_name="Anthropic",
            query="Anthropic company research",
            description="AI safety company",
            industry="Artificial Intelligence",
            key_facts=["Founded 2021", "Created Claude"],
            competitors=["OpenAI", "Google DeepMind"],
        )

        assert result.company_name == "Anthropic"
        assert result.industry == "Artificial Intelligence"
        assert len(result.key_facts) == 2
        assert len(result.competitors) == 2

    def test_company_result_has_timestamp(self):
        """Should have research timestamp."""
        result = CompanyResearchResult(
            company_name="Test Co",
            query="test",
        )

        assert result.research_timestamp is not None
        assert isinstance(result.research_timestamp, datetime)


class TestNewsSearchResultSchema:
    """Tests for NewsSearchResult schema."""

    def test_valid_news_result(self):
        """Should create valid NewsSearchResult."""
        result = NewsSearchResult(
            query="AI news",
            results=[
                SearchResult(
                    title="AI News 1",
                    url="https://news.com/1",
                    content="Breaking news about AI",
                    published_date="2024-01-15",
                )
            ],
            total_results=100,
            time_range="Last 7 days",
        )

        assert result.query == "AI news"
        assert len(result.results) == 1
        assert result.total_results == 100
        assert result.time_range == "Last 7 days"

    def test_latest_article_property(self):
        """latest_article should return first result."""
        result = NewsSearchResult(
            query="test",
            results=[
                SearchResult(title="Article 1", url="https://a.com", content="C1"),
                SearchResult(title="Article 2", url="https://b.com", content="C2"),
            ],
        )

        assert result.latest_article is not None
        assert result.latest_article.title == "Article 1"

    def test_latest_article_empty(self):
        """latest_article should return None when no results."""
        result = NewsSearchResult(query="test", results=[])
        assert result.latest_article is None


# =============================================================================
# RATE LIMITER TESTS
# =============================================================================


class TestTokenBucketRateLimiter:
    """Tests for rate limiter."""

    @pytest.mark.asyncio
    async def test_acquire_success(self):
        """Should acquire token when available."""
        limiter = TokenBucketRateLimiter(requests_per_minute=60, max_burst=5)

        # Should succeed for first requests
        assert await limiter.acquire() is True
        assert await limiter.acquire() is True

    @pytest.mark.asyncio
    async def test_acquire_exhausted(self):
        """Should fail when tokens exhausted."""
        limiter = TokenBucketRateLimiter(requests_per_minute=60, max_burst=2)

        # Exhaust tokens
        await limiter.acquire()
        await limiter.acquire()

        # Should fail
        assert await limiter.acquire() is False

    @pytest.mark.asyncio
    async def test_wait_for_token(self):
        """Should wait and acquire token."""
        limiter = TokenBucketRateLimiter(requests_per_minute=120, max_burst=1)

        # Exhaust tokens
        await limiter.acquire()

        # Wait for refill (should be quick with high rate)
        result = await limiter.wait_for_token(max_wait=2.0)
        assert result is True


# =============================================================================
# CACHE TESTS
# =============================================================================


class TestSimpleCache:
    """Tests for SimpleCache."""

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Should store and retrieve values."""
        cache = SimpleCache(ttl_seconds=60)

        await cache.set("key1", {"data": "value"})
        result = await cache.get("key1")

        assert result is not None
        assert result["data"] == "value"

    @pytest.mark.asyncio
    async def test_get_missing_key(self):
        """Should return None for missing key."""
        cache = SimpleCache(ttl_seconds=60)

        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Should clear all cached values."""
        cache = SimpleCache(ttl_seconds=60)

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        await cache.clear()

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None


# =============================================================================
# SERVICE TESTS
# =============================================================================


class TestTavilyServiceInitialization:
    """Tests for service initialization."""

    def test_singleton_pattern(self):
        """Should return same instance."""
        with patch("backend.services.tavily_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                tavily_api_key="test-key",
                tavily_rate_limit_per_minute=60,
                tavily_cache_ttl_seconds=300,
            )
            TavilyService._instance = None

            service1 = TavilyService.get_instance()
            service2 = TavilyService.get_instance()

            assert service1 is service2

    def test_is_configured_with_key(self, tavily_service):
        """is_configured should be True when API key is set."""
        assert tavily_service.is_configured is True

    def test_is_configured_without_key(self):
        """is_configured should be False when API key is missing."""
        with patch("backend.services.tavily_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                tavily_api_key=None,
                tavily_rate_limit_per_minute=60,
                tavily_cache_ttl_seconds=300,
            )
            TavilyService._instance = None
            service = TavilyService.get_instance()

            assert service.is_configured is False


class TestTavilyServiceSearch:
    """Tests for search operations."""

    @pytest.mark.asyncio
    async def test_search_success(self, tavily_service, mock_search_response):
        """Should return valid search results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_search_response

        with patch.object(
            tavily_service, "_get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await tavily_service.search(
                "test query",
                search_depth=SearchDepth.BASIC,
                max_results=5,
            )

            assert isinstance(result, SearchResponse)
            assert result.query == "test query"
            assert len(result.results) == 3
            assert result.answer is not None
            assert result.result_count == 3

    @pytest.mark.asyncio
    async def test_search_uses_cache(self, tavily_service, mock_search_response):
        """Should use cached results on second call."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_search_response

        with patch.object(
            tavily_service, "_get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            # First call
            result1 = await tavily_service.search("cache test query")

            # Second call should use cache
            result2 = await tavily_service.search("cache test query")

            # Only one API call should have been made
            assert mock_client.post.call_count == 1
            assert result1.query == result2.query

    @pytest.mark.asyncio
    async def test_search_bypass_cache(self, tavily_service, mock_search_response):
        """Should bypass cache when use_cache=False."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_search_response

        with patch.object(
            tavily_service, "_get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            # First call
            await tavily_service.search("bypass cache query", use_cache=False)

            # Second call should NOT use cache
            await tavily_service.search("bypass cache query", use_cache=False)

            # Two API calls should have been made
            assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_search_without_api_key(self):
        """Should raise error when API key not configured."""
        with patch("backend.services.tavily_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                tavily_api_key=None,
                tavily_rate_limit_per_minute=60,
                tavily_cache_ttl_seconds=300,
            )
            TavilyService._instance = None
            service = TavilyService.get_instance()

            with pytest.raises(TavilyError) as exc_info:
                await service.search("test query")

            assert "not configured" in str(exc_info.value)


class TestTavilyServiceErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_authentication_error(self, tavily_service):
        """Should raise AuthenticationError on 401."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch.object(
            tavily_service, "_get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(AuthenticationError):
                await tavily_service.search("test", use_cache=False)

    @pytest.mark.asyncio
    async def test_quota_exceeded_error(self, tavily_service):
        """Should raise QuotaExceededError on 402."""
        mock_response = MagicMock()
        mock_response.status_code = 402

        with patch.object(
            tavily_service, "_get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(QuotaExceededError):
                await tavily_service.search("test", use_cache=False)

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, tavily_service):
        """Should raise RateLimitError on 429 (after retries exhausted)."""
        import tenacity

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "30"}

        with patch.object(
            tavily_service, "_get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            # The service has retry logic, so it will exhaust retries and raise RetryError
            # We need to check that RateLimitError is the underlying cause
            with pytest.raises((RateLimitError, tenacity.RetryError)):
                await tavily_service.search("test", use_cache=False)


class TestTavilyServiceNewsSearch:
    """Tests for news search."""

    @pytest.mark.asyncio
    async def test_search_news(self, tavily_service, mock_search_response):
        """Should return NewsSearchResult."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_search_response

        with patch.object(
            tavily_service, "_get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await tavily_service.search_news("AI news", days=7)

            assert isinstance(result, NewsSearchResult)
            assert result.query == "AI news"
            assert result.time_range == "Last 7 days"


class TestTavilyServiceCompanyResearch:
    """Tests for company research."""

    @pytest.mark.asyncio
    async def test_research_company(self, tavily_service, mock_company_search_response):
        """Should return CompanyResearchResult."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_company_search_response

        with patch.object(
            tavily_service, "_get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await tavily_service.research_company("Anthropic")

            assert isinstance(result, CompanyResearchResult)
            assert result.company_name == "Anthropic"
            assert result.description is not None


class TestTavilyServiceQuickAnswer:
    """Tests for quick_answer method."""

    @pytest.mark.asyncio
    async def test_quick_answer(self, tavily_service, mock_search_response):
        """Should return AI answer."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_search_response

        with patch.object(
            tavily_service, "_get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            answer = await tavily_service.quick_answer("What is LangGraph?")

            assert answer is not None
            assert isinstance(answer, str)


class TestTavilyServiceHealthCheck:
    """Tests for health check."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, tavily_service, mock_search_response):
        """Should return healthy status."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_search_response

        with patch.object(
            tavily_service, "_get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            health = await tavily_service.health_check()

            assert health["status"] == "healthy"
            assert "response_time_ms" in health

    @pytest.mark.asyncio
    async def test_health_check_unconfigured(self):
        """Should return unconfigured status when no API key."""
        with patch("backend.services.tavily_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                tavily_api_key=None,
                tavily_rate_limit_per_minute=60,
                tavily_cache_ttl_seconds=300,
            )
            TavilyService._instance = None
            service = TavilyService.get_instance()

            health = await service.health_check()

            assert health["status"] == "unconfigured"


# =============================================================================
# DEPENDENCY INJECTION TESTS
# =============================================================================


class TestDependencyInjection:
    """Tests for DI functions."""

    def test_get_tavily_service(self):
        """get_tavily_service should return service instance."""
        with patch("backend.services.tavily_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                tavily_api_key="test-key",
                tavily_rate_limit_per_minute=60,
                tavily_cache_ttl_seconds=300,
            )
            TavilyService._instance = None

            service = get_tavily_service()

            assert isinstance(service, TavilyService)


# =============================================================================
# PHASE 4.1 VERIFICATION TESTS
# =============================================================================


class TestPhase41Verification:
    """Verification tests for Phase 4.1 completion."""

    def test_all_enums_defined(self):
        """All required enums should be defined."""
        assert SearchDepth.BASIC.value == "basic"
        assert SearchDepth.ADVANCED.value == "advanced"
        assert SearchTopic.GENERAL.value == "general"
        assert SearchTopic.NEWS.value == "news"
        assert SearchTopic.FINANCE.value == "finance"

    def test_all_errors_defined(self):
        """All error types should be defined."""
        # Base error
        err = TavilyError("test", status_code=500)
        assert err.retryable is True

        # Rate limit error
        rate_err = RateLimitError("rate limit", retry_after=30)
        assert rate_err.status_code == 429
        assert rate_err.retry_after == 30

        # Auth error
        auth_err = AuthenticationError()
        assert auth_err.status_code == 401
        assert auth_err.retryable is False

        # Quota error
        quota_err = QuotaExceededError()
        assert quota_err.status_code == 402
        assert quota_err.retryable is False

    def test_service_has_required_methods(self, tavily_service):
        """Service should have all required methods."""
        # Core methods
        assert hasattr(tavily_service, "search")
        assert hasattr(tavily_service, "search_news")
        assert hasattr(tavily_service, "research_company")
        assert hasattr(tavily_service, "quick_answer")

        # Utility methods
        assert hasattr(tavily_service, "health_check")
        assert hasattr(tavily_service, "clear_cache")
        assert hasattr(tavily_service, "close")

        # Properties
        assert hasattr(tavily_service, "is_configured")

    def test_exports_available(self):
        """All exports should be importable."""
        from backend.services.tavily_service import (
            SearchDepth,
            SearchTopic,
            SearchResult,
            ImageResult,
            SearchResponse,
            CompanyResearchResult,
            NewsSearchResult,
            TavilyError,
            RateLimitError,
            AuthenticationError,
            QuotaExceededError,
            TavilyService,
            get_tavily_service,
        )

        # All imports should work
        assert SearchDepth is not None
        assert TavilyService is not None

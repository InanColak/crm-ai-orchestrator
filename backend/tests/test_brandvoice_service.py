"""
Brandvoice Service Unit Tests (Phase 4.5)
=========================================
Tests for brandvoice RAG integration.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.documents import (
    BrandvoiceService,
    BrandvoiceDocumentType,
    BrandvoiceSearchResult,
    BrandvoiceRetrievalResult,
    get_brandvoice_service,
)
from backend.services.documents.vector_store import VectorSearchResult
from backend.app.schemas.email import BrandvoiceContext


# =============================================================================
# BRANDVOICE SERVICE TESTS
# =============================================================================


class TestBrandvoiceService:
    """Tests for BrandvoiceService."""

    def test_service_initialization(self):
        """Service should initialize with default settings."""
        service = BrandvoiceService()
        assert service._top_k == 3
        assert service._similarity_threshold == 0.65

    def test_service_with_custom_settings(self):
        """Service should accept custom settings."""
        service = BrandvoiceService(top_k=5, similarity_threshold=0.8)
        assert service._top_k == 5
        assert service._similarity_threshold == 0.8

    @pytest.mark.asyncio
    async def test_get_brandvoice_context_no_documents(self):
        """Should return None when no documents exist."""
        service = BrandvoiceService()

        # Mock vector store to return empty search results
        mock_store = MagicMock()
        mock_store.search = AsyncMock(return_value=[])
        service._vector_store = mock_store

        # Mock embedding service
        mock_embed = MagicMock()
        mock_embed.embed_single = AsyncMock(return_value=[0.1] * 1536)
        service._embedding_service = mock_embed

        result = await service.get_brandvoice_context("client-123")

        # No results means None is returned
        assert result is None

    @pytest.mark.asyncio
    async def test_get_brandvoice_context_with_documents(self):
        """Should return context when documents exist."""
        service = BrandvoiceService()

        # Mock vector store to return results for each query type
        mock_store = MagicMock()
        mock_store.search = AsyncMock(
            return_value=[
                VectorSearchResult(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    content="Our brand voice is professional yet approachable.",
                    similarity=0.9,
                    chunk_index=0,
                    filename="brandvoice_tone_guide.md",
                    metadata={},
                )
            ]
        )
        service._vector_store = mock_store

        # Mock embedding service
        mock_embed = MagicMock()
        mock_embed.embed_single = AsyncMock(return_value=[0.1] * 1536)
        service._embedding_service = mock_embed

        result = await service.get_brandvoice_context("client-123")

        assert result is not None
        assert isinstance(result, BrandvoiceContext)
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_has_brandvoice_documents_true(self):
        """Should return True when documents exist."""
        service = BrandvoiceService()

        mock_store = MagicMock()
        mock_store.count_documents = AsyncMock(return_value=3)
        service._vector_store = mock_store

        result = await service.has_brandvoice_documents("client-123")

        assert result is True

    @pytest.mark.asyncio
    async def test_has_brandvoice_documents_false(self):
        """Should return False when no documents exist."""
        service = BrandvoiceService()

        mock_store = MagicMock()
        mock_store.count_documents = AsyncMock(return_value=0)
        service._vector_store = mock_store

        result = await service.has_brandvoice_documents("client-123")

        assert result is False


# =============================================================================
# BRANDVOICE SEARCH RESULT TESTS
# =============================================================================


class TestBrandvoiceSearchResult:
    """Tests for BrandvoiceSearchResult dataclass."""

    def test_high_confidence_true(self):
        """Should identify high confidence results."""
        result = BrandvoiceSearchResult(
            content="Test content",
            similarity=0.85,
            document_type=BrandvoiceDocumentType.TONE_GUIDE,
            chunk_id="chunk-1",
            document_id="doc-1",
            filename="tone.md",
        )

        assert result.is_high_confidence is True

    def test_high_confidence_false(self):
        """Should identify low confidence results."""
        result = BrandvoiceSearchResult(
            content="Test content",
            similarity=0.7,
            document_type=BrandvoiceDocumentType.TONE_GUIDE,
            chunk_id="chunk-1",
            document_id="doc-1",
            filename="tone.md",
        )

        assert result.is_high_confidence is False


# =============================================================================
# BRANDVOICE RETRIEVAL RESULT TESTS
# =============================================================================


class TestBrandvoiceRetrievalResult:
    """Tests for BrandvoiceRetrievalResult dataclass."""

    def test_has_results_true(self):
        """Should return True when any results exist."""
        result = BrandvoiceRetrievalResult(
            tone_results=[
                BrandvoiceSearchResult(
                    content="Test",
                    similarity=0.9,
                    document_type=BrandvoiceDocumentType.TONE_GUIDE,
                    chunk_id="1",
                    document_id="1",
                    filename="test.md",
                )
            ]
        )

        assert result.has_results is True

    def test_has_results_false(self):
        """Should return False when no results exist."""
        result = BrandvoiceRetrievalResult()

        assert result.has_results is False

    def test_average_confidence_calculation(self):
        """Should calculate average confidence correctly."""
        result = BrandvoiceRetrievalResult(
            tone_results=[
                BrandvoiceSearchResult(
                    content="Test",
                    similarity=0.8,
                    document_type=BrandvoiceDocumentType.TONE_GUIDE,
                    chunk_id="1",
                    document_id="1",
                    filename="test.md",
                ),
                BrandvoiceSearchResult(
                    content="Test 2",
                    similarity=0.9,
                    document_type=BrandvoiceDocumentType.TONE_GUIDE,
                    chunk_id="2",
                    document_id="1",
                    filename="test.md",
                ),
            ]
        )

        assert abs(result.average_confidence - 0.85) < 0.0001

    def test_average_confidence_empty(self):
        """Should return 0 for empty results."""
        result = BrandvoiceRetrievalResult()

        assert result.average_confidence == 0.0


# =============================================================================
# DOCUMENT TYPE INFERENCE TESTS
# =============================================================================


class TestDocumentTypeInference:
    """Tests for document type inference."""

    def test_infer_from_filename_tone(self):
        """Should infer TONE_GUIDE from filename."""
        service = BrandvoiceService()

        doc_type = service._infer_document_type("brandvoice_tone_guide.md", {}, "tone")

        assert doc_type == BrandvoiceDocumentType.TONE_GUIDE

    def test_infer_from_filename_style(self):
        """Should infer STYLE_GUIDE from filename."""
        service = BrandvoiceService()

        doc_type = service._infer_document_type("writing_style.pdf", {}, "style")

        assert doc_type == BrandvoiceDocumentType.STYLE_GUIDE

    def test_infer_from_filename_vocab(self):
        """Should infer VOCABULARY from filename."""
        service = BrandvoiceService()

        doc_type = service._infer_document_type("vocabulary_list.docx", {}, "phrases")

        assert doc_type == BrandvoiceDocumentType.VOCABULARY

    def test_infer_from_filename_examples(self):
        """Should infer EXAMPLES from filename."""
        service = BrandvoiceService()

        doc_type = service._infer_document_type("email_examples.txt", {}, "examples")

        assert doc_type == BrandvoiceDocumentType.EXAMPLES

    def test_infer_from_metadata(self):
        """Should infer from metadata when available."""
        service = BrandvoiceService()

        doc_type = service._infer_document_type(
            "document.md",
            {"document_type": "tone_guide"},
            "general"
        )

        assert doc_type == BrandvoiceDocumentType.TONE_GUIDE

    def test_infer_fallback_to_aspect(self):
        """Should fallback to aspect-based inference."""
        service = BrandvoiceService()

        doc_type = service._infer_document_type("random.pdf", {}, "style")

        assert doc_type == BrandvoiceDocumentType.STYLE_GUIDE


# =============================================================================
# CONTENT EXTRACTION TESTS
# =============================================================================


class TestContentExtraction:
    """Tests for content extraction methods."""

    def test_summarize_content_short(self):
        """Should return full content when short."""
        service = BrandvoiceService()

        result = service._summarize_content(["Short content."], max_length=500)

        assert result == "Short content."

    def test_summarize_content_truncate(self):
        """Should truncate at sentence boundary when long."""
        service = BrandvoiceService()

        long_content = "First sentence. " * 50

        result = service._summarize_content([long_content], max_length=100)

        assert len(result) <= 105  # Some buffer for "..."
        assert result.endswith(".") or result.endswith("...")

    def test_extract_phrases_from_quoted(self):
        """Should extract quoted phrases."""
        service = BrandvoiceService()

        content = 'Use phrases like "Let\'s explore" and "I\'d love to discuss".'

        phrases = service._extract_phrases_from_content(content)

        assert "Let's explore" in phrases
        assert "I'd love to discuss" in phrases

    def test_extract_phrases_from_patterns(self):
        """Should extract phrases from patterns."""
        service = BrandvoiceService()

        content = "Prefer: collaborative approach. Use: friendly tone."

        phrases = service._extract_phrases_from_content(content)

        assert len(phrases) > 0

    def test_extract_avoid_phrases(self):
        """Should extract phrases to avoid."""
        service = BrandvoiceService()

        content = "Avoid: corporate jargon. Don't use: checking in. Never say: per my last email."

        phrases = service._extract_avoid_phrases(content)

        assert "corporate jargon" in phrases or len(phrases) > 0


# =============================================================================
# CONTEXT BUILDING TESTS
# =============================================================================


class TestContextBuilding:
    """Tests for building BrandvoiceContext from results."""

    def test_build_context_with_tone(self):
        """Should build context with tone guidelines."""
        service = BrandvoiceService()

        retrieval = BrandvoiceRetrievalResult(
            tone_results=[
                BrandvoiceSearchResult(
                    content="Our tone is professional and friendly.",
                    similarity=0.9,
                    document_type=BrandvoiceDocumentType.TONE_GUIDE,
                    chunk_id="1",
                    document_id="doc-1",
                    filename="tone.md",
                )
            ],
            successful_queries=1,
            total_queries=5,
        )

        context = service._build_context_from_results(retrieval)

        assert context.tone_guidelines is not None
        assert "professional" in context.tone_guidelines.lower()

    def test_build_context_with_examples(self):
        """Should build context with example snippets."""
        service = BrandvoiceService()

        retrieval = BrandvoiceRetrievalResult(
            example_results=[
                BrandvoiceSearchResult(
                    content="Hi Sarah, I noticed that TechCorp recently expanded...",
                    similarity=0.85,
                    document_type=BrandvoiceDocumentType.EXAMPLES,
                    chunk_id="1",
                    document_id="doc-1",
                    filename="examples.md",
                )
            ],
            successful_queries=1,
            total_queries=5,
        )

        context = service._build_context_from_results(retrieval)

        assert context.example_snippets is not None
        assert len(context.example_snippets) > 0

    def test_build_context_empty_results(self):
        """Should handle empty results gracefully."""
        service = BrandvoiceService()

        retrieval = BrandvoiceRetrievalResult()

        context = service._build_context_from_results(retrieval)

        assert context.tone_guidelines is None
        assert context.writing_style is None
        assert context.key_phrases == []
        assert context.phrases_to_avoid == []
        assert context.example_snippets == []
        assert context.source_documents == []
        assert context.confidence == 0.0


# =============================================================================
# SINGLETON TESTS
# =============================================================================


class TestSingleton:
    """Tests for singleton accessor."""

    def test_get_brandvoice_service_singleton(self):
        """Should return same instance."""
        # Reset singleton
        import backend.services.documents.brandvoice_service as module
        module._brandvoice_service_instance = None

        service1 = get_brandvoice_service()
        service2 = get_brandvoice_service()

        assert service1 is service2


# =============================================================================
# EMAIL CONTEXT BUILDER INTEGRATION TESTS
# =============================================================================


class TestEmailContextBuilderIntegration:
    """Tests for EmailContextBuilder RAG integration."""

    @pytest.mark.asyncio
    async def test_context_builder_with_rag_disabled(self):
        """Should skip RAG when disabled."""
        from backend.services.email.context_builder import EmailContextBuilder
        from backend.app.schemas.email import EmailCopilotInput, EmailRecipient, EmailType

        builder = EmailContextBuilder(rag_enabled=False)

        email_input = EmailCopilotInput(
            recipient=EmailRecipient(email="test@example.com"),
            email_type=EmailType.COLD_OUTREACH,
        )

        context = await builder.build_context(
            email_input=email_input,
            client_id="client-123",
        )

        assert context.brandvoice is None

    @pytest.mark.asyncio
    async def test_context_builder_with_rag_enabled(self):
        """Should call RAG when enabled."""
        from backend.services.email.context_builder import EmailContextBuilder
        from backend.app.schemas.email import EmailCopilotInput, EmailRecipient, EmailType

        builder = EmailContextBuilder(rag_enabled=True)

        # Mock brandvoice service
        mock_service = MagicMock()
        mock_service.has_brandvoice_documents = AsyncMock(return_value=True)
        mock_service.get_brandvoice_context = AsyncMock(
            return_value=BrandvoiceContext(
                tone_guidelines="Professional and friendly",
                confidence=0.85,
            )
        )
        builder._brandvoice_service = mock_service

        email_input = EmailCopilotInput(
            recipient=EmailRecipient(email="test@example.com"),
            email_type=EmailType.COLD_OUTREACH,
        )

        context = await builder.build_context(
            email_input=email_input,
            client_id="client-123",
        )

        assert context.brandvoice is not None
        assert context.brandvoice.tone_guidelines == "Professional and friendly"
        mock_service.has_brandvoice_documents.assert_called_once_with("client-123")

    @pytest.mark.asyncio
    async def test_context_builder_no_client_id(self):
        """Should skip RAG when no client_id."""
        from backend.services.email.context_builder import EmailContextBuilder
        from backend.app.schemas.email import EmailCopilotInput, EmailRecipient, EmailType

        builder = EmailContextBuilder(rag_enabled=True)

        email_input = EmailCopilotInput(
            recipient=EmailRecipient(email="test@example.com"),
            email_type=EmailType.COLD_OUTREACH,
        )

        context = await builder.build_context(
            email_input=email_input,
            client_id=None,  # No client_id
        )

        assert context.brandvoice is None

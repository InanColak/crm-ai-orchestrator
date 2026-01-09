"""
RAG Services Unit Tests (Phase 4.4)
===================================
Tests for document parsing, chunking, embeddings, vector store, and usage tracking.
"""

import pytest
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.documents import (
    DocumentParser,
    ParsedDocument,
    ParserError,
    UnsupportedFormatError,
    FileTooLargeError,
    TextChunker,
    TextChunk,
    EmbeddingService,
    EmbeddingResult,
    EmbeddingError,
    VectorStore,
    VectorSearchResult,
    VectorStoreError,
    UsageTracker,
    UsageType,
    UsageRecord,
    QuotaExceededError,
)


# =============================================================================
# DOCUMENT PARSER TESTS
# =============================================================================


class TestDocumentParser:
    """Tests for DocumentParser."""

    def test_parser_initialization(self):
        """Parser should initialize with default settings."""
        parser = DocumentParser()
        assert parser.allowed_types == {"pdf", "docx", "txt", "md"}

    def test_parse_txt_file(self):
        """Parser should parse TXT files."""
        parser = DocumentParser()
        content = b"Hello, this is a test document.\nWith multiple lines."

        result = parser.parse(content, "test.txt")

        assert result.filename == "test.txt"
        assert result.file_type == "txt"
        assert "Hello" in result.content
        assert result.word_count > 0
        assert result.char_count > 0

    def test_parse_md_file(self):
        """Parser should parse markdown files."""
        parser = DocumentParser()
        content = b"# Title\n\nThis is **markdown** content."

        result = parser.parse(content, "readme.md")

        assert result.file_type == "md"
        assert "Title" in result.content
        assert "markdown" in result.content

    def test_unsupported_format_error(self):
        """Parser should raise error for unsupported formats."""
        parser = DocumentParser()

        with pytest.raises(UnsupportedFormatError) as exc_info:
            parser.parse(b"data", "file.xyz")

        assert "xyz" in str(exc_info.value)

    def test_file_too_large_error(self):
        """Parser should raise error for large files."""
        parser = DocumentParser(max_file_size_mb=1)
        large_content = b"x" * (2 * 1024 * 1024)  # 2 MB

        with pytest.raises(FileTooLargeError) as exc_info:
            parser.parse(large_content, "large.txt")

        assert exc_info.value.size_mb > 1

    def test_validate_file_success(self):
        """Validate should return empty list for valid files."""
        parser = DocumentParser()

        errors = parser.validate_file("document.pdf", 1024 * 1024)

        assert errors == []

    def test_validate_file_invalid_type(self):
        """Validate should return error for invalid type."""
        parser = DocumentParser()

        errors = parser.validate_file("file.exe", 1024)

        assert len(errors) == 1
        assert "not allowed" in errors[0]

    def test_validate_file_too_large(self):
        """Validate should return error for large file."""
        parser = DocumentParser(max_file_size_mb=1)

        errors = parser.validate_file("doc.pdf", 50 * 1024 * 1024)

        assert len(errors) == 1
        assert "exceeds limit" in errors[0]


# =============================================================================
# TEXT CHUNKER TESTS
# =============================================================================


class TestTextChunker:
    """Tests for TextChunker."""

    def test_chunker_initialization(self):
        """Chunker should initialize with default settings."""
        chunker = TextChunker()
        assert chunker.chunk_size > 0
        assert chunker.chunk_overlap >= 0

    def test_chunk_short_text(self):
        """Short text should produce single chunk."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a short text."

        chunks = list(chunker.chunk(text))

        assert len(chunks) == 1
        assert chunks[0].content == "This is a short text."
        assert chunks[0].chunk_index == 0

    def test_chunk_long_text(self):
        """Long text should produce multiple chunks."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a longer text. " * 20

        chunks = list(chunker.chunk(text))

        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.token_count > 0

    def test_chunk_with_metadata(self):
        """Chunks should include provided metadata."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "Sample text for testing."
        metadata = {"source": "test", "version": 1}

        chunks = list(chunker.chunk(text, metadata=metadata))

        assert chunks[0].metadata == metadata

    def test_chunk_empty_text(self):
        """Empty text should produce no chunks."""
        chunker = TextChunker()

        chunks = list(chunker.chunk(""))
        assert len(chunks) == 0

        chunks = list(chunker.chunk("   "))
        assert len(chunks) == 0

    def test_count_tokens(self):
        """Token counting should work."""
        chunker = TextChunker()
        text = "Hello world, this is a test."

        tokens = chunker.count_tokens(text)

        assert tokens > 0
        assert tokens < len(text)  # Tokens are fewer than chars

    def test_estimate_chunk_count(self):
        """Chunk count estimation should be reasonable."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "This is sample text. " * 100

        estimate = chunker.estimate_chunk_count(text)
        actual = len(list(chunker.chunk(text)))

        # Estimate should be within 50% of actual
        assert abs(estimate - actual) <= actual * 0.5 + 1

    def test_chunk_respects_separators(self):
        """Chunking should prefer paragraph breaks."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        text = "First paragraph content here.\n\nSecond paragraph content here."

        chunks = list(chunker.chunk(text))

        # Chunks should not break mid-sentence if possible
        for chunk in chunks:
            assert chunk.content.strip()


# =============================================================================
# EMBEDDING SERVICE TESTS
# =============================================================================


class TestEmbeddingService:
    """Tests for EmbeddingService."""

    def test_service_initialization(self):
        """Service should initialize with default settings."""
        service = EmbeddingService()
        assert service.model == "text-embedding-3-small"
        assert service.dimensions == 1536

    def test_estimate_tokens(self):
        """Token estimation should work."""
        service = EmbeddingService()
        texts = ["Hello world", "This is a test"]

        tokens = service.estimate_tokens(texts)

        assert tokens > 0

    def test_estimate_cost(self):
        """Cost estimation should work."""
        service = EmbeddingService()

        cost = service.estimate_cost(1000000)  # 1M tokens

        assert cost > 0
        assert cost < 1  # Less than $1 for 1M tokens with small model

    @pytest.mark.asyncio
    async def test_embed_texts_empty(self):
        """Empty list should return empty result."""
        service = EmbeddingService()

        result = await service.embed_texts([])

        assert result.embeddings == []
        assert result.total_tokens == 0
        assert result.chunk_count == 0

    @pytest.mark.asyncio
    async def test_embed_texts_with_mock(self):
        """Embedding should work with mocked OpenAI."""
        service = EmbeddingService()

        # Mock the OpenAI client
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
        ]
        mock_response.usage = MagicMock(total_tokens=10)

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with patch.object(service, "_get_client", return_value=mock_client):
            result = await service.embed_texts(["Hello", "World"])

        assert len(result.embeddings) == 2
        assert result.total_tokens == 10
        assert result.chunk_count == 2

    @pytest.mark.asyncio
    async def test_embed_single_with_mock(self):
        """Single embedding should work."""
        service = EmbeddingService()

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.5] * 1536)]
        mock_response.usage = MagicMock(total_tokens=5)

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with patch.object(service, "_get_client", return_value=mock_client):
            embedding = await service.embed_single("Test text")

        assert len(embedding) == 1536


# =============================================================================
# VECTOR STORE TESTS
# =============================================================================


class TestVectorStore:
    """Tests for VectorStore."""

    def test_store_initialization(self):
        """Store should initialize with default settings."""
        store = VectorStore()
        assert store.DOCUMENTS_TABLE == "rag_documents"
        assert store.CHUNKS_TABLE == "rag_chunks"

    @pytest.mark.asyncio
    async def test_store_document_with_mock(self):
        """Document storage should work with mocked Supabase."""
        store = VectorStore()

        # Create mock chunks
        chunks = [
            MagicMock(content="Chunk 1", chunk_index=0, token_count=10, metadata={}),
            MagicMock(content="Chunk 2", chunk_index=1, token_count=15, metadata={}),
        ]
        embeddings = [[0.1] * 1536, [0.2] * 1536]

        # Mock Supabase client
        mock_client = MagicMock()
        mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock()

        with patch.object(store, "_get_client", return_value=mock_client):
            doc_id = await store.store_document(
                client_id="client-123",
                filename="test.pdf",
                file_type="pdf",
                chunks=chunks,
                embeddings=embeddings,
            )

        assert doc_id is not None
        assert mock_client.table.called

    @pytest.mark.asyncio
    async def test_search_with_mock(self):
        """Search should work with mocked Supabase."""
        store = VectorStore()

        # Mock response
        mock_response = MagicMock()
        mock_response.data = [
            {
                "id": "chunk-1",
                "document_id": "doc-1",
                "content": "Relevant content",
                "similarity": 0.95,
                "chunk_index": 0,
                "filename": "test.pdf",
                "metadata": {},
            }
        ]

        mock_client = MagicMock()
        mock_client.rpc.return_value.execute.return_value = mock_response

        with patch.object(store, "_get_client", return_value=mock_client):
            results = await store.search(
                client_id="client-123",
                query_embedding=[0.1] * 1536,
            )

        assert len(results) == 1
        assert results[0].similarity == 0.95

    @pytest.mark.asyncio
    async def test_chunk_count_mismatch_error(self):
        """Store should error on chunk/embedding count mismatch."""
        store = VectorStore()

        chunks = [MagicMock()]
        embeddings = [[0.1] * 1536, [0.2] * 1536]  # 2 embeddings for 1 chunk

        with pytest.raises(VectorStoreError) as exc_info:
            await store.store_document(
                client_id="client-123",
                filename="test.pdf",
                file_type="pdf",
                chunks=chunks,
                embeddings=embeddings,
            )

        assert "doesn't match" in str(exc_info.value)


# =============================================================================
# USAGE TRACKER TESTS
# =============================================================================


class TestUsageTracker:
    """Tests for UsageTracker."""

    def test_tracker_initialization(self):
        """Tracker should initialize with limits from config."""
        tracker = UsageTracker()
        assert tracker.get_limit(UsageType.EMBEDDING_TOKENS) > 0
        assert tracker.get_limit(UsageType.SEARCH_QUERIES) > 0
        assert tracker.get_limit(UsageType.DOCUMENT_COUNT) > 0

    @pytest.mark.asyncio
    async def test_get_usage_with_mock(self):
        """Get usage should work with mocked Supabase."""
        tracker = UsageTracker()

        mock_response = MagicMock()
        mock_response.data = [{"count": 100}]

        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.execute.return_value = mock_response

        with patch.object(tracker, "_get_client", return_value=mock_client):
            usage = await tracker.get_usage("client-123", UsageType.EMBEDDING_TOKENS)

        assert usage.count == 100
        assert usage.client_id == "client-123"

    @pytest.mark.asyncio
    async def test_check_quota_exceeded(self):
        """Check quota should raise error when exceeded."""
        tracker = UsageTracker()

        # Mock usage at limit
        mock_response = MagicMock()
        mock_response.data = [{"count": 500000}]  # At daily limit

        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.execute.return_value = mock_response

        with patch.object(tracker, "_get_client", return_value=mock_client):
            with pytest.raises(QuotaExceededError) as exc_info:
                await tracker.check_quota(
                    "client-123",
                    UsageType.EMBEDDING_TOKENS,
                    1000,
                )

        assert exc_info.value.client_id == "client-123"

    @pytest.mark.asyncio
    async def test_record_usage_with_mock(self):
        """Record usage should work with mocked Supabase."""
        tracker = UsageTracker()

        mock_response = MagicMock()
        mock_response.data = 150  # New count after increment

        mock_client = MagicMock()
        mock_client.rpc.return_value.execute.return_value = mock_response

        with patch.object(tracker, "_get_client", return_value=mock_client):
            record = await tracker.record_usage(
                "client-123",
                UsageType.SEARCH_QUERIES,
                1,
            )

        # The count is the value returned from upsert, not the amount added
        assert record.count == 150
        assert mock_client.rpc.called


# =============================================================================
# USAGE RECORD TESTS
# =============================================================================


class TestUsageRecord:
    """Tests for UsageRecord dataclass."""

    def test_remaining_calculation(self):
        """Remaining should be calculated correctly."""
        record = UsageRecord(
            client_id="test",
            usage_type=UsageType.EMBEDDING_TOKENS,
            date=date.today(),
            count=300000,
            limit=500000,
        )

        assert record.remaining == 200000

    def test_usage_percent_calculation(self):
        """Usage percent should be calculated correctly."""
        record = UsageRecord(
            client_id="test",
            usage_type=UsageType.EMBEDDING_TOKENS,
            date=date.today(),
            count=250000,
            limit=500000,
        )

        assert record.usage_percent == 50.0

    def test_is_exceeded(self):
        """Is exceeded should work correctly."""
        record = UsageRecord(
            client_id="test",
            usage_type=UsageType.EMBEDDING_TOKENS,
            date=date.today(),
            count=500000,
            limit=500000,
        )

        assert record.is_exceeded is True


# =============================================================================
# INTEGRATION TESTS (Mocked)
# =============================================================================


class TestRAGPipeline:
    """Integration tests for the RAG pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_mocks(self):
        """Full pipeline should work with all services mocked."""
        # Parse document
        parser = DocumentParser()
        content = b"This is a test document with enough content to chunk properly. " * 10
        parsed = parser.parse(content, "test.txt")

        assert parsed.content
        assert parsed.word_count > 0

        # Chunk document
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        chunks = list(chunker.chunk(parsed.content))

        assert len(chunks) > 0

        # Mock embedding service
        embedding_service = EmbeddingService()

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536) for _ in chunks
        ]
        mock_response.usage = MagicMock(total_tokens=len(chunks) * 10)

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with patch.object(embedding_service, "_get_client", return_value=mock_client):
            embedding_result = await embedding_service.embed_chunks(chunks)

        assert len(embedding_result.embeddings) == len(chunks)

    def test_parsed_document_counts(self):
        """Parsed document should have accurate counts."""
        parser = DocumentParser()
        content = b"Word one. Word two. Word three."

        result = parser.parse(content, "test.txt")

        assert result.word_count == 6  # "Word one Word two Word three"
        assert result.char_count == len("Word one. Word two. Word three.")


# =============================================================================
# EMBEDDING RESULT TESTS
# =============================================================================


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_average_tokens_per_chunk(self):
        """Average should be calculated correctly."""
        result = EmbeddingResult(
            embeddings=[[0.1], [0.2], [0.3]],
            total_tokens=30,
            model="test",
            dimensions=1,
            chunk_count=3,
        )

        assert result.average_tokens_per_chunk == 10.0

    def test_average_tokens_empty(self):
        """Average should be 0 for empty result."""
        result = EmbeddingResult(
            embeddings=[],
            total_tokens=0,
            model="test",
            dimensions=1,
            chunk_count=0,
        )

        assert result.average_tokens_per_chunk == 0


# =============================================================================
# VECTOR SEARCH RESULT TESTS
# =============================================================================


class TestVectorSearchResult:
    """Tests for VectorSearchResult dataclass."""

    def test_relevance_percent(self):
        """Relevance percent should be calculated correctly."""
        result = VectorSearchResult(
            chunk_id="test",
            document_id="doc",
            content="Test content",
            similarity=0.85,
            chunk_index=0,
            filename="test.pdf",
        )

        assert result.relevance_percent == 85.0

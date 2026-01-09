"""
RAG API Unit Tests (Phase 4.4)
==============================
Tests for document upload and search API endpoints.
"""

import pytest
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from io import BytesIO

from fastapi.testclient import TestClient
from fastapi import FastAPI

from backend.app.api.v1.documents import (
    router,
    get_parser,
    get_chunker,
    get_embedding_service,
    get_vector_store,
    get_usage_tracker,
)
from backend.services.documents import (
    DocumentParser,
    TextChunker,
    EmbeddingService,
    VectorStore,
    UsageTracker,
    UsageType,
    UsageRecord,
    ParsedDocument,
    TextChunk,
    EmbeddingResult,
    VectorDocument,
    VectorSearchResult,
    QuotaExceededError,
    UnsupportedFormatError,
    FileTooLargeError,
)


# =============================================================================
# TEST SETUP
# =============================================================================


@pytest.fixture
def app():
    """Create test FastAPI app."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_parser():
    """Mock document parser."""
    parser = MagicMock(spec=DocumentParser)
    parser.parse.return_value = ParsedDocument(
        content="Test content for parsing.",
        filename="test.txt",
        file_type="txt",
        page_count=None,
        word_count=4,
        char_count=25,
    )
    return parser


@pytest.fixture
def mock_chunker():
    """Mock text chunker."""
    chunker = MagicMock(spec=TextChunker)
    chunks = [
        TextChunk(
            content="Test content",
            chunk_index=0,
            start_char=0,
            end_char=12,
            token_count=3,
            metadata={},
        ),
    ]
    chunker.chunk.return_value = iter(chunks)
    return chunker


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    service = MagicMock(spec=EmbeddingService)
    service.embed_chunks = AsyncMock(
        return_value=EmbeddingResult(
            embeddings=[[0.1] * 1536],
            total_tokens=10,
            model="text-embedding-3-small",
            dimensions=1536,
            chunk_count=1,
        )
    )
    service.embed_single = AsyncMock(return_value=[0.1] * 1536)
    return service


@pytest.fixture
def mock_vector_store():
    """Mock vector store."""
    store = MagicMock(spec=VectorStore)
    store.store_document = AsyncMock(return_value="doc-123")
    store.search = AsyncMock(
        return_value=[
            VectorSearchResult(
                chunk_id="chunk-1",
                document_id="doc-123",
                content="Relevant content",
                similarity=0.95,
                chunk_index=0,
                filename="test.pdf",
                metadata={},
            )
        ]
    )
    store.list_documents = AsyncMock(
        return_value=[
            VectorDocument(
                id="doc-123",
                client_id="client-123",
                filename="test.pdf",
                file_type="pdf",
                chunk_count=5,
                total_tokens=500,
                created_at=datetime.utcnow(),
                metadata={},
            )
        ]
    )
    store.get_document = AsyncMock(
        return_value=VectorDocument(
            id="doc-123",
            client_id="client-123",
            filename="test.pdf",
            file_type="pdf",
            chunk_count=5,
            total_tokens=500,
            created_at=datetime.utcnow(),
            metadata={},
        )
    )
    store.delete_document = AsyncMock(return_value=True)
    store.count_documents = AsyncMock(return_value=1)
    return store


@pytest.fixture
def mock_usage_tracker():
    """Mock usage tracker."""
    tracker = MagicMock(spec=UsageTracker)
    tracker.check_quota = AsyncMock(return_value=True)
    tracker.record_usage = AsyncMock(
        return_value=UsageRecord(
            client_id="client-123",
            usage_type=UsageType.EMBEDDING_TOKENS,
            date=date.today(),
            count=10,
            limit=500000,
        )
    )
    tracker.get_summary = AsyncMock(
        return_value=MagicMock(
            client_id="client-123",
            date=date.today(),
            embedding_tokens=UsageRecord(
                client_id="client-123",
                usage_type=UsageType.EMBEDDING_TOKENS,
                date=date.today(),
                count=1000,
                limit=500000,
            ),
            search_queries=UsageRecord(
                client_id="client-123",
                usage_type=UsageType.SEARCH_QUERIES,
                date=date.today(),
                count=10,
                limit=1000,
            ),
            document_count=UsageRecord(
                client_id="client-123",
                usage_type=UsageType.DOCUMENT_COUNT,
                date=date.today(),
                count=5,
                limit=50,
            ),
        )
    )
    return tracker


# =============================================================================
# UPLOAD ENDPOINT TESTS
# =============================================================================


class TestUploadEndpoint:
    """Tests for document upload endpoint."""

    def test_upload_success(
        self,
        app,
        mock_parser,
        mock_chunker,
        mock_embedding_service,
        mock_vector_store,
        mock_usage_tracker,
    ):
        """Successful upload should return document info."""
        app.dependency_overrides[get_parser] = lambda: mock_parser
        app.dependency_overrides[get_chunker] = lambda: mock_chunker
        app.dependency_overrides[get_embedding_service] = lambda: mock_embedding_service
        app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
        app.dependency_overrides[get_usage_tracker] = lambda: mock_usage_tracker

        client = TestClient(app)

        response = client.post(
            "/documents/upload",
            files={"file": ("test.txt", b"Test content", "text/plain")},
            data={"client_id": "client-123"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["document_id"] == "doc-123"
        assert data["filename"] == "test.txt"
        assert data["chunk_count"] == 1

        app.dependency_overrides.clear()

    def test_upload_unsupported_format(
        self,
        app,
        mock_chunker,
        mock_embedding_service,
        mock_vector_store,
        mock_usage_tracker,
    ):
        """Unsupported format should return 415."""
        parser = MagicMock(spec=DocumentParser)
        parser.parse.side_effect = UnsupportedFormatError("xyz not supported")

        app.dependency_overrides[get_parser] = lambda: parser
        app.dependency_overrides[get_chunker] = lambda: mock_chunker
        app.dependency_overrides[get_embedding_service] = lambda: mock_embedding_service
        app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
        app.dependency_overrides[get_usage_tracker] = lambda: mock_usage_tracker

        client = TestClient(app)

        response = client.post(
            "/documents/upload",
            files={"file": ("test.xyz", b"data", "application/octet-stream")},
            data={"client_id": "client-123"},
        )

        assert response.status_code == 415

        app.dependency_overrides.clear()

    def test_upload_file_too_large(
        self,
        app,
        mock_chunker,
        mock_embedding_service,
        mock_vector_store,
        mock_usage_tracker,
    ):
        """Large file should return 413."""
        parser = MagicMock(spec=DocumentParser)
        parser.parse.side_effect = FileTooLargeError("File too large", size_mb=15)

        app.dependency_overrides[get_parser] = lambda: parser
        app.dependency_overrides[get_chunker] = lambda: mock_chunker
        app.dependency_overrides[get_embedding_service] = lambda: mock_embedding_service
        app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
        app.dependency_overrides[get_usage_tracker] = lambda: mock_usage_tracker

        client = TestClient(app)

        response = client.post(
            "/documents/upload",
            files={"file": ("large.pdf", b"x" * 1000, "application/pdf")},
            data={"client_id": "client-123"},
        )

        assert response.status_code == 413

        app.dependency_overrides.clear()

    def test_upload_quota_exceeded(
        self,
        app,
        mock_parser,
        mock_chunker,
        mock_embedding_service,
        mock_vector_store,
    ):
        """Quota exceeded should return 429."""
        tracker = MagicMock(spec=UsageTracker)
        tracker.check_quota = AsyncMock(
            side_effect=QuotaExceededError(
                "Quota exceeded",
                client_id="client-123",
                usage_type="document_count",
                current=50,
                limit=50,
            )
        )

        app.dependency_overrides[get_parser] = lambda: mock_parser
        app.dependency_overrides[get_chunker] = lambda: mock_chunker
        app.dependency_overrides[get_embedding_service] = lambda: mock_embedding_service
        app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
        app.dependency_overrides[get_usage_tracker] = lambda: tracker

        client = TestClient(app)

        response = client.post(
            "/documents/upload",
            files={"file": ("test.txt", b"content", "text/plain")},
            data={"client_id": "client-123"},
        )

        assert response.status_code == 429

        app.dependency_overrides.clear()


# =============================================================================
# SEARCH ENDPOINT TESTS
# =============================================================================


class TestSearchEndpoint:
    """Tests for document search endpoint."""

    def test_search_success(
        self,
        app,
        mock_embedding_service,
        mock_vector_store,
        mock_usage_tracker,
    ):
        """Successful search should return results."""
        app.dependency_overrides[get_embedding_service] = lambda: mock_embedding_service
        app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
        app.dependency_overrides[get_usage_tracker] = lambda: mock_usage_tracker

        client = TestClient(app)

        response = client.post(
            "/documents/search",
            params={"client_id": "client-123"},
            json={"query": "test query", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert len(data["results"]) == 1
        assert data["results"][0]["similarity"] == 0.95

        app.dependency_overrides.clear()

    def test_search_quota_exceeded(
        self,
        app,
        mock_embedding_service,
        mock_vector_store,
    ):
        """Search quota exceeded should return 429."""
        tracker = MagicMock(spec=UsageTracker)
        tracker.check_quota = AsyncMock(
            side_effect=QuotaExceededError(
                "Quota exceeded",
                client_id="client-123",
                usage_type="search_queries",
                current=1000,
                limit=1000,
            )
        )

        app.dependency_overrides[get_embedding_service] = lambda: mock_embedding_service
        app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
        app.dependency_overrides[get_usage_tracker] = lambda: tracker

        client = TestClient(app)

        response = client.post(
            "/documents/search",
            params={"client_id": "client-123"},
            json={"query": "test query"},
        )

        assert response.status_code == 429

        app.dependency_overrides.clear()


# =============================================================================
# LIST DOCUMENTS ENDPOINT TESTS
# =============================================================================


class TestListDocumentsEndpoint:
    """Tests for document list endpoint."""

    def test_list_success(self, app, mock_vector_store):
        """List should return documents."""
        app.dependency_overrides[get_vector_store] = lambda: mock_vector_store

        client = TestClient(app)

        response = client.get(
            "/documents",
            params={"client_id": "client-123"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["documents"]) == 1
        assert data["total"] == 1

        app.dependency_overrides.clear()

    def test_list_with_filter(self, app, mock_vector_store):
        """List with file type filter should work."""
        app.dependency_overrides[get_vector_store] = lambda: mock_vector_store

        client = TestClient(app)

        response = client.get(
            "/documents",
            params={"client_id": "client-123", "file_types": "pdf,docx"},
        )

        assert response.status_code == 200

        app.dependency_overrides.clear()


# =============================================================================
# GET DOCUMENT ENDPOINT TESTS
# =============================================================================


class TestGetDocumentEndpoint:
    """Tests for get document endpoint."""

    def test_get_success(self, app, mock_vector_store):
        """Get should return document."""
        app.dependency_overrides[get_vector_store] = lambda: mock_vector_store

        client = TestClient(app)

        response = client.get("/documents/doc-123")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "doc-123"
        assert data["filename"] == "test.pdf"

        app.dependency_overrides.clear()

    def test_get_not_found(self, app):
        """Get non-existent document should return 404."""
        store = MagicMock(spec=VectorStore)
        store.get_document = AsyncMock(return_value=None)

        app.dependency_overrides[get_vector_store] = lambda: store

        client = TestClient(app)

        response = client.get("/documents/non-existent")

        assert response.status_code == 404

        app.dependency_overrides.clear()


# =============================================================================
# DELETE DOCUMENT ENDPOINT TESTS
# =============================================================================


class TestDeleteDocumentEndpoint:
    """Tests for delete document endpoint."""

    def test_delete_success(self, app, mock_vector_store):
        """Delete should succeed."""
        app.dependency_overrides[get_vector_store] = lambda: mock_vector_store

        client = TestClient(app)

        response = client.delete("/documents/doc-123")

        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "doc-123"

        app.dependency_overrides.clear()


# =============================================================================
# USAGE ENDPOINT TESTS
# =============================================================================


class TestUsageEndpoint:
    """Tests for usage endpoint."""

    def test_usage_success(self, app, mock_usage_tracker):
        """Usage should return summary."""
        app.dependency_overrides[get_usage_tracker] = lambda: mock_usage_tracker

        client = TestClient(app)

        response = client.get("/documents/usage/client-123")

        assert response.status_code == 200
        data = response.json()
        assert data["client_id"] == "client-123"
        assert "embedding_tokens" in data
        assert "search_queries" in data
        assert "document_count" in data

        app.dependency_overrides.clear()

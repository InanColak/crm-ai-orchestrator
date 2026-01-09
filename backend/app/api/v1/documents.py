"""
Documents API (Phase 4.4)
=========================
REST endpoints for document upload, search, and management.

Provides:
- Document upload with automatic parsing, chunking, and embedding
- Semantic search across documents
- Document listing and deletion
- Usage tracking and quota information
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from backend.app.schemas.documents import (
    DocumentUploadResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentDeleteResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    UsageSummaryResponse,
    UsageInfo,
    DocumentErrorResponse,
)
from backend.services.documents import (
    DocumentParser,
    TextChunker,
    EmbeddingService,
    VectorStore,
    UsageTracker,
    UsageType,
    ParserError,
    UnsupportedFormatError,
    FileTooLargeError,
    EmbeddingError,
    VectorStoreError,
    QuotaExceededError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# =============================================================================
# DEPENDENCIES
# =============================================================================


def get_parser() -> DocumentParser:
    """Get document parser instance."""
    return DocumentParser()


def get_chunker() -> TextChunker:
    """Get text chunker instance."""
    return TextChunker()


def get_embedding_service() -> EmbeddingService:
    """Get embedding service instance."""
    return EmbeddingService()


def get_vector_store() -> VectorStore:
    """Get vector store instance."""
    return VectorStore()


def get_usage_tracker() -> UsageTracker:
    """Get usage tracker instance."""
    return UsageTracker()


# =============================================================================
# UPLOAD ENDPOINT
# =============================================================================


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": DocumentErrorResponse, "description": "Invalid file"},
        413: {"model": DocumentErrorResponse, "description": "File too large"},
        415: {"model": DocumentErrorResponse, "description": "Unsupported format"},
        429: {"model": DocumentErrorResponse, "description": "Quota exceeded"},
    },
)
async def upload_document(
    file: Annotated[UploadFile, File(description="Document file")],
    client_id: Annotated[str, Form(description="Client identifier")],
    parser: DocumentParser = Depends(get_parser),
    chunker: TextChunker = Depends(get_chunker),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
    usage_tracker: UsageTracker = Depends(get_usage_tracker),
) -> DocumentUploadResponse:
    """
    Upload a document for RAG processing.

    The document will be:
    1. Parsed to extract text content
    2. Split into overlapping chunks
    3. Embedded using OpenAI
    4. Stored in the vector database

    Supported formats: PDF, DOCX, TXT, MD

    Rate limits apply per client:
    - Daily embedding token limit
    - Maximum documents per client
    """
    filename = file.filename or "unknown"

    try:
        # Check document count quota
        await usage_tracker.check_quota(client_id, UsageType.DOCUMENT_COUNT, 1)

        # Read file content
        file_content = await file.read()

        # Parse document
        parsed = parser.parse(file_content, filename)

        # Chunk text
        chunks = list(chunker.chunk(parsed.content, metadata={"filename": filename}))

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document has no extractable content",
            )

        # Estimate tokens and check quota
        total_tokens = sum(chunk.token_count for chunk in chunks)
        await usage_tracker.check_quota(
            client_id, UsageType.EMBEDDING_TOKENS, total_tokens
        )

        # Generate embeddings
        embedding_result = await embedding_service.embed_chunks(chunks)

        # Store in vector database
        document_id = await vector_store.store_document(
            client_id=client_id,
            filename=filename,
            file_type=parsed.file_type,
            chunks=chunks,
            embeddings=embedding_result.embeddings,
            metadata={
                "word_count": parsed.word_count,
                "char_count": parsed.char_count,
                "page_count": parsed.page_count,
            },
        )

        # Record usage
        await usage_tracker.record_usage(
            client_id, UsageType.EMBEDDING_TOKENS, embedding_result.total_tokens
        )
        await usage_tracker.record_usage(client_id, UsageType.DOCUMENT_COUNT, 1)

        logger.info(
            f"[DocumentAPI] Uploaded '{filename}' for client {client_id}: "
            f"{len(chunks)} chunks, {total_tokens} tokens"
        )

        return DocumentUploadResponse(
            document_id=document_id,
            filename=filename,
            file_type=parsed.file_type,
            chunk_count=len(chunks),
            total_tokens=embedding_result.total_tokens,
        )

    except UnsupportedFormatError as e:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(e),
        )
    except FileTooLargeError as e:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(e),
        )
    except QuotaExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
        )
    except ParserError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except EmbeddingError as e:
        logger.exception(f"Embedding error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Embedding generation failed: {e.message}",
        )
    except VectorStoreError as e:
        logger.exception(f"Vector store error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Storage failed: {e.message}",
        )


# =============================================================================
# SEARCH ENDPOINT
# =============================================================================


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={
        429: {"model": DocumentErrorResponse, "description": "Quota exceeded"},
    },
)
async def search_documents(
    client_id: str,
    request: SearchRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
    usage_tracker: UsageTracker = Depends(get_usage_tracker),
) -> SearchResponse:
    """
    Semantic search across documents.

    Searches for chunks similar to the query using vector similarity.
    Results are ranked by cosine similarity score.
    """
    try:
        # Check search quota
        await usage_tracker.check_quota(client_id, UsageType.SEARCH_QUERIES, 1)

        # Generate query embedding
        query_embedding = await embedding_service.embed_single(request.query)

        # Search vector store
        results = await vector_store.search(
            client_id=client_id,
            query_embedding=query_embedding,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            file_types=request.file_types,
            document_ids=request.document_ids,
        )

        # Record usage
        await usage_tracker.record_usage(client_id, UsageType.SEARCH_QUERIES, 1)

        return SearchResponse(
            query=request.query,
            results=[
                SearchResultItem(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    filename=r.filename,
                    content=r.content,
                    similarity=r.similarity,
                    chunk_index=r.chunk_index,
                )
                for r in results
            ],
            total_results=len(results),
        )

    except QuotaExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
        )
    except EmbeddingError as e:
        logger.exception(f"Embedding error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Query embedding failed: {e.message}",
        )
    except VectorStoreError as e:
        logger.exception(f"Vector store error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Search failed: {e.message}",
        )


# =============================================================================
# LIST DOCUMENTS
# =============================================================================


@router.get(
    "",
    response_model=DocumentListResponse,
)
async def list_documents(
    client_id: str,
    file_types: str | None = None,
    limit: int = 50,
    offset: int = 0,
    vector_store: VectorStore = Depends(get_vector_store),
) -> DocumentListResponse:
    """
    List documents for a client.

    Supports pagination and filtering by file type.
    """
    try:
        file_type_list = file_types.split(",") if file_types else None

        documents = await vector_store.list_documents(
            client_id=client_id,
            file_types=file_type_list,
            limit=limit,
            offset=offset,
        )

        total = await vector_store.count_documents(client_id)

        return DocumentListResponse(
            documents=[
                DocumentInfo(
                    id=doc.id,
                    client_id=doc.client_id,
                    filename=doc.filename,
                    file_type=doc.file_type,
                    chunk_count=doc.chunk_count,
                    total_tokens=doc.total_tokens,
                    created_at=doc.created_at,
                    metadata=doc.metadata,
                )
                for doc in documents
            ],
            total=total,
            limit=limit,
            offset=offset,
        )

    except VectorStoreError as e:
        logger.exception(f"Vector store error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to list documents: {e.message}",
        )


# =============================================================================
# GET DOCUMENT
# =============================================================================


@router.get(
    "/{document_id}",
    response_model=DocumentInfo,
    responses={
        404: {"model": DocumentErrorResponse, "description": "Document not found"},
    },
)
async def get_document(
    document_id: str,
    vector_store: VectorStore = Depends(get_vector_store),
) -> DocumentInfo:
    """
    Get document by ID.
    """
    try:
        doc = await vector_store.get_document(document_id)

        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}",
            )

        return DocumentInfo(
            id=doc.id,
            client_id=doc.client_id,
            filename=doc.filename,
            file_type=doc.file_type,
            chunk_count=doc.chunk_count,
            total_tokens=doc.total_tokens,
            created_at=doc.created_at,
            metadata=doc.metadata,
        )

    except VectorStoreError as e:
        logger.exception(f"Vector store error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to get document: {e.message}",
        )


# =============================================================================
# DELETE DOCUMENT
# =============================================================================


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    responses={
        404: {"model": DocumentErrorResponse, "description": "Document not found"},
    },
)
async def delete_document(
    document_id: str,
    vector_store: VectorStore = Depends(get_vector_store),
) -> DocumentDeleteResponse:
    """
    Delete a document and all its chunks.
    """
    try:
        await vector_store.delete_document(document_id)

        return DocumentDeleteResponse(document_id=document_id)

    except VectorStoreError as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}",
            )
        logger.exception(f"Vector store error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to delete document: {e.message}",
        )


# =============================================================================
# USAGE ENDPOINT
# =============================================================================


@router.get(
    "/usage/{client_id}",
    response_model=UsageSummaryResponse,
)
async def get_usage(
    client_id: str,
    usage_tracker: UsageTracker = Depends(get_usage_tracker),
) -> UsageSummaryResponse:
    """
    Get usage summary for a client.

    Shows current usage and remaining quota for:
    - Embedding tokens (daily limit)
    - Search queries (daily limit)
    - Document count (total limit)
    """
    try:
        summary = await usage_tracker.get_summary(client_id)

        return UsageSummaryResponse(
            client_id=summary.client_id,
            usage_date=summary.date,
            embedding_tokens=UsageInfo(
                count=summary.embedding_tokens.count,
                limit=summary.embedding_tokens.limit,
                remaining=summary.embedding_tokens.remaining,
                usage_percent=summary.embedding_tokens.usage_percent,
            ),
            search_queries=UsageInfo(
                count=summary.search_queries.count,
                limit=summary.search_queries.limit,
                remaining=summary.search_queries.remaining,
                usage_percent=summary.search_queries.usage_percent,
            ),
            document_count=UsageInfo(
                count=summary.document_count.count,
                limit=summary.document_count.limit,
                remaining=summary.document_count.remaining,
                usage_percent=summary.document_count.usage_percent,
            ),
        )

    except Exception as e:
        logger.exception(f"Usage tracker error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to get usage: {str(e)}",
        )

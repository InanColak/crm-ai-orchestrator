"""
Document Schemas (Phase 4.4)
============================
Pydantic models for document upload and RAG operations.
"""

from __future__ import annotations

from datetime import datetime
from datetime import date as date_type
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# DOCUMENT SCHEMAS
# =============================================================================


class DocumentUploadResponse(BaseModel):
    """Response after successful document upload."""

    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type (pdf, docx, txt, md)")
    chunk_count: int = Field(..., description="Number of chunks created")
    total_tokens: int = Field(..., description="Total tokens in document")
    message: str = Field(default="Document uploaded successfully")


class DocumentInfo(BaseModel):
    """Document metadata."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Document ID")
    client_id: str = Field(..., description="Client ID")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type")
    chunk_count: int = Field(..., description="Number of chunks")
    total_tokens: int = Field(..., description="Total tokens")
    created_at: datetime = Field(..., description="Upload timestamp")
    metadata: dict = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    """Response for document list."""

    documents: list[DocumentInfo] = Field(default_factory=list)
    total: int = Field(..., description="Total document count")
    limit: int = Field(..., description="Page size")
    offset: int = Field(..., description="Page offset")


class DocumentDeleteResponse(BaseModel):
    """Response for document deletion."""

    document_id: str = Field(..., description="Deleted document ID")
    message: str = Field(default="Document deleted successfully")


# =============================================================================
# SEARCH SCHEMAS
# =============================================================================


class SearchRequest(BaseModel):
    """Request for semantic search."""

    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum similarity score"
    )
    file_types: Optional[list[str]] = Field(
        default=None, description="Filter by file types"
    )
    document_ids: Optional[list[str]] = Field(
        default=None, description="Filter by document IDs"
    )


class SearchResultItem(BaseModel):
    """Single search result."""

    chunk_id: str = Field(..., description="Chunk ID")
    document_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Source filename")
    content: str = Field(..., description="Chunk content")
    similarity: float = Field(..., description="Similarity score (0-1)")
    chunk_index: int = Field(..., description="Chunk index in document")


class SearchResponse(BaseModel):
    """Response for semantic search."""

    query: str = Field(..., description="Original query")
    results: list[SearchResultItem] = Field(default_factory=list)
    total_results: int = Field(..., description="Number of results")


# =============================================================================
# USAGE SCHEMAS
# =============================================================================


class UsageInfo(BaseModel):
    """Usage information for a specific type."""

    count: int = Field(..., description="Current usage count")
    limit: int = Field(..., description="Usage limit")
    remaining: int = Field(..., description="Remaining quota")
    usage_percent: float = Field(..., description="Usage percentage")


class UsageSummaryResponse(BaseModel):
    """Usage summary for a client."""

    client_id: str = Field(..., description="Client ID")
    usage_date: date_type = Field(..., description="Usage date", alias="date")
    embedding_tokens: UsageInfo
    search_queries: UsageInfo
    document_count: UsageInfo

    model_config = ConfigDict(populate_by_name=True)


# =============================================================================
# ERROR SCHEMAS
# =============================================================================


class DocumentErrorResponse(BaseModel):
    """Error response for document operations."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(default=None, description="Additional details")

"""
Supabase Client Service
=======================
Async Supabase client for database operations, storage, and vector queries.
Implements connection pooling, retry logic, and proper error handling.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, TypeVar, Generic
from uuid import UUID

import httpx
from pydantic import BaseModel
from supabase import create_client, Client as SupabaseClient
from supabase.lib.client_options import ClientOptions
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class DatabaseError(Exception):
    """Base exception for database operations."""

    def __init__(self, message: str, operation: str | None = None, details: dict | None = None):
        self.message = message
        self.operation = operation
        self.details = details or {}
        super().__init__(self.message)


class ConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class QueryError(DatabaseError):
    """Raised when a database query fails."""
    pass


class NotFoundError(DatabaseError):
    """Raised when a record is not found."""
    pass


class SupabaseService:
    """
    Async Supabase Service

    Provides a high-level interface for Supabase operations including:
    - Database CRUD operations with retry logic
    - Vector similarity search
    - Storage operations
    - Real-time subscriptions

    Usage:
        >>> service = SupabaseService()
        >>> await service.initialize()
        >>> result = await service.fetch_one("clients", {"id": "uuid"})
        >>> await service.close()
    """

    _instance: SupabaseService | None = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: SupabaseClient | None = None
        self._admin_client: SupabaseClient | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._initialized: bool = False

    @classmethod
    async def get_instance(cls) -> SupabaseService:
        """
        Get singleton instance of SupabaseService.

        Returns:
            SupabaseService: Initialized service instance
        """
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                await cls._instance.initialize()
            return cls._instance

    async def initialize(self) -> None:
        """Initialize Supabase clients and HTTP client."""
        if self._initialized:
            return

        try:
            # Create standard client (uses anon key)
            options = ClientOptions(
                postgrest_client_timeout=30,
                storage_client_timeout=60,
            )
            self._client = create_client(
                self._settings.supabase_url,
                self._settings.supabase_key,
                options=options,
            )

            # Create admin client if service key is available
            if self._settings.supabase_service_key:
                self._admin_client = create_client(
                    self._settings.supabase_url,
                    self._settings.supabase_service_key,
                    options=options,
                )

            # Create async HTTP client for direct REST calls
            self._http_client = httpx.AsyncClient(
                base_url=f"{self._settings.supabase_url}/rest/v1",
                headers={
                    "apikey": self._settings.supabase_key,
                    "Authorization": f"Bearer {self._settings.supabase_key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=representation",
                },
                timeout=httpx.Timeout(30.0),
            )

            self._initialized = True
            logger.info("Supabase client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise ConnectionError(
                message="Failed to initialize Supabase connection",
                operation="initialize",
                details={"error": str(e)},
            )

    async def close(self) -> None:
        """Close all connections and cleanup resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._client = None
        self._admin_client = None
        self._initialized = False
        logger.info("Supabase client connections closed")

    @property
    def client(self) -> SupabaseClient:
        """Get the standard Supabase client."""
        if not self._client:
            raise ConnectionError(
                message="Supabase client not initialized",
                operation="get_client",
            )
        return self._client

    @property
    def admin_client(self) -> SupabaseClient:
        """Get the admin Supabase client (requires service key)."""
        if not self._admin_client:
            raise ConnectionError(
                message="Admin client not available (service key not configured)",
                operation="get_admin_client",
            )
        return self._admin_client

    # =========================================================================
    # DATABASE OPERATIONS
    # =========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    )
    async def fetch_one(
        self,
        table: str,
        filters: dict[str, Any],
        columns: str = "*",
    ) -> dict[str, Any] | None:
        """
        Fetch a single record from a table.

        Args:
            table: Table name
            filters: Dictionary of column-value pairs for filtering
            columns: Columns to select (default: all)

        Returns:
            Record as dictionary or None if not found

        Raises:
            QueryError: If query execution fails
        """
        try:
            query = self.client.table(table).select(columns)

            for key, value in filters.items():
                query = query.eq(key, str(value) if isinstance(value, UUID) else value)

            response = query.limit(1).execute()

            if response.data:
                return response.data[0]
            return None

        except Exception as e:
            logger.error(f"fetch_one failed for {table}: {e}")
            raise QueryError(
                message=f"Failed to fetch record from {table}",
                operation="fetch_one",
                details={"table": table, "filters": filters, "error": str(e)},
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    )
    async def fetch_many(
        self,
        table: str,
        filters: dict[str, Any] | None = None,
        columns: str = "*",
        order_by: str | None = None,
        ascending: bool = True,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Fetch multiple records from a table.

        Args:
            table: Table name
            filters: Optional dictionary of column-value pairs for filtering
            columns: Columns to select (default: all)
            order_by: Column to order by
            ascending: Sort order (default: ascending)
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of records as dictionaries

        Raises:
            QueryError: If query execution fails
        """
        try:
            query = self.client.table(table).select(columns)

            if filters:
                for key, value in filters.items():
                    query = query.eq(key, str(value) if isinstance(value, UUID) else value)

            if order_by:
                query = query.order(order_by, desc=not ascending)

            if limit:
                query = query.limit(limit)

            if offset > 0:
                query = query.offset(offset)

            response = query.execute()
            return response.data or []

        except Exception as e:
            logger.error(f"fetch_many failed for {table}: {e}")
            raise QueryError(
                message=f"Failed to fetch records from {table}",
                operation="fetch_many",
                details={"table": table, "filters": filters, "error": str(e)},
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    )
    async def insert(
        self,
        table: str,
        data: dict[str, Any] | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Insert one or more records into a table.

        Args:
            table: Table name
            data: Single record or list of records to insert

        Returns:
            List of inserted records

        Raises:
            QueryError: If insert fails
        """
        try:
            # Ensure data is a list
            records = data if isinstance(data, list) else [data]

            # Convert UUIDs to strings and add timestamps
            now = datetime.now(timezone.utc).isoformat()
            processed_records = []
            for record in records:
                processed = {
                    k: str(v) if isinstance(v, UUID) else v
                    for k, v in record.items()
                }
                # Add created_at if not present
                if "created_at" not in processed:
                    processed["created_at"] = now
                processed_records.append(processed)

            response = self.client.table(table).insert(processed_records).execute()

            logger.info(f"Inserted {len(response.data)} record(s) into {table}")
            return response.data

        except Exception as e:
            logger.error(f"insert failed for {table}: {e}")
            raise QueryError(
                message=f"Failed to insert into {table}",
                operation="insert",
                details={"table": table, "error": str(e)},
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    )
    async def update(
        self,
        table: str,
        filters: dict[str, Any],
        data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Update records matching the filters.

        Args:
            table: Table name
            filters: Dictionary of column-value pairs for filtering
            data: Dictionary of column-value pairs to update

        Returns:
            List of updated records

        Raises:
            QueryError: If update fails
        """
        try:
            # Add updated_at timestamp
            now = datetime.now(timezone.utc).isoformat()
            update_data = {
                k: str(v) if isinstance(v, UUID) else v
                for k, v in data.items()
            }
            update_data["updated_at"] = now

            query = self.client.table(table).update(update_data)

            for key, value in filters.items():
                query = query.eq(key, str(value) if isinstance(value, UUID) else value)

            response = query.execute()

            logger.info(f"Updated {len(response.data)} record(s) in {table}")
            return response.data

        except Exception as e:
            logger.error(f"update failed for {table}: {e}")
            raise QueryError(
                message=f"Failed to update {table}",
                operation="update",
                details={"table": table, "filters": filters, "error": str(e)},
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    )
    async def upsert(
        self,
        table: str,
        data: dict[str, Any] | list[dict[str, Any]],
        on_conflict: str = "id",
    ) -> list[dict[str, Any]]:
        """
        Insert or update records (upsert).

        Args:
            table: Table name
            data: Single record or list of records
            on_conflict: Column(s) to use for conflict resolution

        Returns:
            List of upserted records

        Raises:
            QueryError: If upsert fails
        """
        try:
            records = data if isinstance(data, list) else [data]

            now = datetime.now(timezone.utc).isoformat()
            processed_records = []
            for record in records:
                processed = {
                    k: str(v) if isinstance(v, UUID) else v
                    for k, v in record.items()
                }
                processed["updated_at"] = now
                if "created_at" not in processed:
                    processed["created_at"] = now
                processed_records.append(processed)

            response = (
                self.client.table(table)
                .upsert(processed_records, on_conflict=on_conflict)
                .execute()
            )

            logger.info(f"Upserted {len(response.data)} record(s) in {table}")
            return response.data

        except Exception as e:
            logger.error(f"upsert failed for {table}: {e}")
            raise QueryError(
                message=f"Failed to upsert into {table}",
                operation="upsert",
                details={"table": table, "error": str(e)},
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    )
    async def delete(
        self,
        table: str,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Delete records matching the filters.

        Args:
            table: Table name
            filters: Dictionary of column-value pairs for filtering

        Returns:
            List of deleted records

        Raises:
            QueryError: If delete fails
        """
        try:
            query = self.client.table(table).delete()

            for key, value in filters.items():
                query = query.eq(key, str(value) if isinstance(value, UUID) else value)

            response = query.execute()

            logger.info(f"Deleted {len(response.data)} record(s) from {table}")
            return response.data

        except Exception as e:
            logger.error(f"delete failed for {table}: {e}")
            raise QueryError(
                message=f"Failed to delete from {table}",
                operation="delete",
                details={"table": table, "filters": filters, "error": str(e)},
            )

    # =========================================================================
    # VECTOR OPERATIONS
    # =========================================================================

    async def vector_search(
        self,
        table: str,
        embedding: list[float],
        match_column: str = "embedding",
        match_count: int = 10,
        match_threshold: float = 0.7,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform vector similarity search using pgvector.

        Args:
            table: Table name containing embeddings
            embedding: Query embedding vector
            match_column: Column name containing embeddings
            match_count: Maximum number of results
            match_threshold: Minimum similarity threshold (0-1)
            filters: Optional additional filters

        Returns:
            List of matching records with similarity scores

        Raises:
            QueryError: If search fails
        """
        try:
            # Use RPC call for vector search
            params = {
                "query_embedding": embedding,
                "match_count": match_count,
                "match_threshold": match_threshold,
            }

            if filters:
                params["filter_params"] = filters

            response = self.client.rpc(
                f"match_{table}",
                params,
            ).execute()

            return response.data or []

        except Exception as e:
            logger.error(f"vector_search failed for {table}: {e}")
            raise QueryError(
                message=f"Vector search failed on {table}",
                operation="vector_search",
                details={"table": table, "error": str(e)},
            )

    # =========================================================================
    # STORAGE OPERATIONS
    # =========================================================================

    async def upload_file(
        self,
        bucket: str,
        path: str,
        file_data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Upload a file to Supabase Storage.

        Args:
            bucket: Storage bucket name
            path: File path within the bucket
            file_data: File content as bytes
            content_type: MIME type of the file

        Returns:
            Public URL of the uploaded file

        Raises:
            QueryError: If upload fails
        """
        try:
            response = self.client.storage.from_(bucket).upload(
                path,
                file_data,
                {"content-type": content_type},
            )

            # Get public URL
            public_url = self.client.storage.from_(bucket).get_public_url(path)

            logger.info(f"Uploaded file to {bucket}/{path}")
            return public_url

        except Exception as e:
            logger.error(f"upload_file failed: {e}")
            raise QueryError(
                message="Failed to upload file",
                operation="upload_file",
                details={"bucket": bucket, "path": path, "error": str(e)},
            )

    async def download_file(
        self,
        bucket: str,
        path: str,
    ) -> bytes:
        """
        Download a file from Supabase Storage.

        Args:
            bucket: Storage bucket name
            path: File path within the bucket

        Returns:
            File content as bytes

        Raises:
            QueryError: If download fails
            NotFoundError: If file does not exist
        """
        try:
            response = self.client.storage.from_(bucket).download(path)
            return response

        except Exception as e:
            logger.error(f"download_file failed: {e}")
            if "not found" in str(e).lower():
                raise NotFoundError(
                    message=f"File not found: {bucket}/{path}",
                    operation="download_file",
                )
            raise QueryError(
                message="Failed to download file",
                operation="download_file",
                details={"bucket": bucket, "path": path, "error": str(e)},
            )

    async def delete_file(
        self,
        bucket: str,
        paths: list[str],
    ) -> None:
        """
        Delete files from Supabase Storage.

        Args:
            bucket: Storage bucket name
            paths: List of file paths to delete

        Raises:
            QueryError: If delete fails
        """
        try:
            self.client.storage.from_(bucket).remove(paths)
            logger.info(f"Deleted {len(paths)} file(s) from {bucket}")

        except Exception as e:
            logger.error(f"delete_file failed: {e}")
            raise QueryError(
                message="Failed to delete file(s)",
                operation="delete_file",
                details={"bucket": bucket, "paths": paths, "error": str(e)},
            )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    async def health_check(self) -> dict[str, Any]:
        """
        Check database connectivity and return status.

        Returns:
            Health status dictionary
        """
        try:
            # Simple query to check connectivity
            response = self.client.table("clients").select("id").limit(1).execute()

            return {
                "status": "healthy",
                "connected": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def count(
        self,
        table: str,
        filters: dict[str, Any] | None = None,
    ) -> int:
        """
        Count records in a table.

        Args:
            table: Table name
            filters: Optional filters

        Returns:
            Record count
        """
        try:
            query = self.client.table(table).select("*", count="exact")

            if filters:
                for key, value in filters.items():
                    query = query.eq(key, str(value) if isinstance(value, UUID) else value)

            response = query.execute()
            return response.count or 0

        except Exception as e:
            logger.error(f"count failed for {table}: {e}")
            raise QueryError(
                message=f"Failed to count records in {table}",
                operation="count",
                details={"table": table, "filters": filters, "error": str(e)},
            )


# =========================================================================
# DEPENDENCY INJECTION HELPERS
# =========================================================================

async def get_supabase() -> SupabaseService:
    """
    Dependency injection for SupabaseService.

    Usage in FastAPI:
        @router.get("/items")
        async def get_items(db: SupabaseService = Depends(get_supabase)):
            return await db.fetch_many("items")
    """
    return await SupabaseService.get_instance()


@asynccontextmanager
async def supabase_session() -> AsyncGenerator[SupabaseService, None]:
    """
    Context manager for Supabase operations.

    Usage:
        async with supabase_session() as db:
            await db.insert("items", {"name": "test"})
    """
    service = await SupabaseService.get_instance()
    try:
        yield service
    finally:
        pass  # Connection managed by singleton

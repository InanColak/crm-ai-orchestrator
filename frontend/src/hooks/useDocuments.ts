/**
 * Documents Hooks (Phase 6.1.3)
 * =============================
 * React hooks for document management operations.
 */

'use client';

import { useState, useCallback, useEffect } from 'react';
import { documentsApi } from '@/services/api';
import type {
  Document,
  DocumentUploadResponse,
  UsageSummary,
  SearchResponse,
  SearchRequest,
  FileType,
} from '@/types/document';

// Default client ID (will be replaced with auth context)
const DEFAULT_CLIENT_ID = process.env.NEXT_PUBLIC_CLIENT_ID || 'a1b2c3d4-e5f6-7890-abcd-ef1234567890';

// =============================================================================
// USE DOCUMENTS HOOK
// =============================================================================

interface UseDocumentsOptions {
  clientId?: string;
  fileTypes?: FileType[];
  pageSize?: number;
}

interface UseDocumentsReturn {
  documents: Document[];
  total: number;
  isLoading: boolean;
  error: string | null;
  page: number;
  setPage: (page: number) => void;
  refetch: () => Promise<void>;
}

export function useDocuments(options: UseDocumentsOptions = {}): UseDocumentsReturn {
  const { clientId = DEFAULT_CLIENT_ID, fileTypes, pageSize = 20 } = options;

  const [documents, setDocuments] = useState<Document[]>([]);
  const [total, setTotal] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);

  const fetchDocuments = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await documentsApi.list({
        client_id: clientId,
        file_types: fileTypes,
        limit: pageSize,
        offset: (page - 1) * pageSize,
      });

      setDocuments(response.documents);
      setTotal(response.total);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch documents';
      setError(message);
      setDocuments([]);
    } finally {
      setIsLoading(false);
    }
  }, [clientId, fileTypes, page, pageSize]);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  return {
    documents,
    total,
    isLoading,
    error,
    page,
    setPage,
    refetch: fetchDocuments,
  };
}

// =============================================================================
// USE DOCUMENT UPLOAD HOOK
// =============================================================================

interface UploadProgress {
  file: File;
  progress: number;
  status: 'pending' | 'uploading' | 'success' | 'error';
  error?: string;
  result?: DocumentUploadResponse;
}

interface UseDocumentUploadReturn {
  uploadFile: (file: File) => Promise<DocumentUploadResponse>;
  uploadFiles: (files: File[]) => Promise<DocumentUploadResponse[]>;
  uploads: UploadProgress[];
  isUploading: boolean;
  clearUploads: () => void;
}

export function useDocumentUpload(clientId: string = DEFAULT_CLIENT_ID): UseDocumentUploadReturn {
  const [uploads, setUploads] = useState<UploadProgress[]>([]);
  const [isUploading, setIsUploading] = useState(false);

  const uploadFile = useCallback(
    async (file: File): Promise<DocumentUploadResponse> => {
      // Add to upload list
      setUploads((prev) => [
        ...prev,
        { file, progress: 0, status: 'uploading' },
      ]);

      try {
        const result = await documentsApi.upload(file, clientId);

        // Update status to success
        setUploads((prev) =>
          prev.map((u) =>
            u.file === file
              ? { ...u, progress: 100, status: 'success', result }
              : u
          )
        );

        return result;
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Upload failed';

        // Update status to error
        setUploads((prev) =>
          prev.map((u) =>
            u.file === file
              ? { ...u, status: 'error', error: errorMessage }
              : u
          )
        );

        throw err;
      }
    },
    [clientId]
  );

  const uploadFiles = useCallback(
    async (files: File[]): Promise<DocumentUploadResponse[]> => {
      setIsUploading(true);
      const results: DocumentUploadResponse[] = [];

      for (const file of files) {
        try {
          const result = await uploadFile(file);
          results.push(result);
        } catch {
          // Continue with other files
        }
      }

      setIsUploading(false);
      return results;
    },
    [uploadFile]
  );

  const clearUploads = useCallback(() => {
    setUploads([]);
  }, []);

  return {
    uploadFile,
    uploadFiles,
    uploads,
    isUploading,
    clearUploads,
  };
}

// =============================================================================
// USE DOCUMENT DELETE HOOK
// =============================================================================

interface UseDocumentDeleteReturn {
  deleteDocument: (documentId: string) => Promise<void>;
  isDeleting: boolean;
  deletingId: string | null;
}

export function useDocumentDelete(): UseDocumentDeleteReturn {
  const [isDeleting, setIsDeleting] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const deleteDocument = useCallback(async (documentId: string): Promise<void> => {
    setIsDeleting(true);
    setDeletingId(documentId);

    try {
      await documentsApi.delete(documentId);
    } finally {
      setIsDeleting(false);
      setDeletingId(null);
    }
  }, []);

  return {
    deleteDocument,
    isDeleting,
    deletingId,
  };
}

// =============================================================================
// USE DOCUMENT SEARCH HOOK
// =============================================================================

interface UseDocumentSearchReturn {
  search: (query: string, options?: Partial<SearchRequest>) => Promise<SearchResponse>;
  results: SearchResponse | null;
  isSearching: boolean;
  error: string | null;
  clearResults: () => void;
}

export function useDocumentSearch(clientId: string = DEFAULT_CLIENT_ID): UseDocumentSearchReturn {
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const search = useCallback(
    async (query: string, options?: Partial<SearchRequest>): Promise<SearchResponse> => {
      setIsSearching(true);
      setError(null);

      try {
        const response = await documentsApi.search(clientId, {
          query,
          top_k: options?.top_k ?? 5,
          similarity_threshold: options?.similarity_threshold ?? 0.7,
          file_types: options?.file_types,
          document_ids: options?.document_ids,
        });

        setResults(response);
        return response;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Search failed';
        setError(message);
        throw err;
      } finally {
        setIsSearching(false);
      }
    },
    [clientId]
  );

  const clearResults = useCallback(() => {
    setResults(null);
    setError(null);
  }, []);

  return {
    search,
    results,
    isSearching,
    error,
    clearResults,
  };
}

// =============================================================================
// USE DOCUMENT USAGE HOOK
// =============================================================================

interface UseDocumentUsageReturn {
  usage: UsageSummary | null;
  isLoading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export function useDocumentUsage(clientId: string = DEFAULT_CLIENT_ID): UseDocumentUsageReturn {
  const [usage, setUsage] = useState<UsageSummary | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchUsage = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await documentsApi.getUsage(clientId);
      setUsage(response);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch usage';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [clientId]);

  useEffect(() => {
    fetchUsage();
  }, [fetchUsage]);

  return {
    usage,
    isLoading,
    error,
    refetch: fetchUsage,
  };
}

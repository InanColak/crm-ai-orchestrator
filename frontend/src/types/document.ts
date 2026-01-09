/**
 * Document Types (Phase 6.1.3)
 * ============================
 * TypeScript types for document management, matching backend schemas.
 */

// =============================================================================
// DOCUMENT TYPES
// =============================================================================

/** Supported file types for document upload */
export type FileType = 'pdf' | 'docx' | 'txt' | 'md';

/** Document metadata */
export interface Document {
  id: string;
  client_id: string;
  filename: string;
  file_type: FileType;
  chunk_count: number;
  total_tokens: number;
  created_at: string;
  metadata: {
    word_count?: number;
    char_count?: number;
    page_count?: number;
    [key: string]: unknown;
  };
}

/** Response after successful document upload */
export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  file_type: FileType;
  chunk_count: number;
  total_tokens: number;
  message: string;
}

/** Response for document list */
export interface DocumentListResponse {
  documents: Document[];
  total: number;
  limit: number;
  offset: number;
}

/** Response for document deletion */
export interface DocumentDeleteResponse {
  document_id: string;
  message: string;
}

// =============================================================================
// SEARCH TYPES
// =============================================================================

/** Request for semantic search */
export interface SearchRequest {
  query: string;
  top_k?: number;
  similarity_threshold?: number;
  file_types?: FileType[];
  document_ids?: string[];
}

/** Single search result */
export interface SearchResultItem {
  chunk_id: string;
  document_id: string;
  filename: string;
  content: string;
  similarity: number;
  chunk_index: number;
}

/** Response for semantic search */
export interface SearchResponse {
  query: string;
  results: SearchResultItem[];
  total_results: number;
}

// =============================================================================
// USAGE TYPES
// =============================================================================

/** Usage information for a specific type */
export interface UsageInfo {
  count: number;
  limit: number;
  remaining: number;
  usage_percent: number;
}

/** Usage summary for a client */
export interface UsageSummary {
  client_id: string;
  usage_date: string;
  embedding_tokens: UsageInfo;
  search_queries: UsageInfo;
  document_count: UsageInfo;
}

// =============================================================================
// UI HELPER TYPES
// =============================================================================

/** File type display information */
export interface FileTypeInfo {
  type: FileType;
  label: string;
  description: string;
  icon: string;
  color: string;
  bgColor: string;
}

/** Get file type display info */
export function getFileTypeInfo(fileType: FileType): FileTypeInfo {
  const info: Record<FileType, FileTypeInfo> = {
    pdf: {
      type: 'pdf',
      label: 'PDF',
      description: 'Adobe PDF Document',
      icon: 'FileText',
      color: 'text-red-600',
      bgColor: 'bg-red-100',
    },
    docx: {
      type: 'docx',
      label: 'DOCX',
      description: 'Microsoft Word Document',
      icon: 'FileText',
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
    },
    txt: {
      type: 'txt',
      label: 'TXT',
      description: 'Plain Text File',
      icon: 'FileText',
      color: 'text-gray-600',
      bgColor: 'bg-gray-100',
    },
    md: {
      type: 'md',
      label: 'MD',
      description: 'Markdown Document',
      icon: 'FileText',
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
    },
  };
  return info[fileType] || info.txt;
}

/** Format file size for display */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

/** Get file extension from filename */
export function getFileExtension(filename: string): FileType {
  const ext = filename.split('.').pop()?.toLowerCase();
  if (ext === 'pdf' || ext === 'docx' || ext === 'txt' || ext === 'md') {
    return ext;
  }
  return 'txt';
}

/** Check if file type is supported */
export function isSupportedFileType(filename: string): boolean {
  const ext = filename.split('.').pop()?.toLowerCase();
  return ext === 'pdf' || ext === 'docx' || ext === 'txt' || ext === 'md';
}

/** Supported MIME types */
export const SUPPORTED_MIME_TYPES: Record<FileType, string[]> = {
  pdf: ['application/pdf'],
  docx: ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
  txt: ['text/plain'],
  md: ['text/markdown', 'text/x-markdown', 'text/plain'],
};

/** Get accept string for file input */
export const ACCEPT_FILE_TYPES = '.pdf,.docx,.txt,.md';

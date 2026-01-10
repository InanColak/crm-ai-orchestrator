/**
 * API Client Service
 * ==================
 * Centralized API client for backend communication.
 */

import type {
  Workflow,
  WorkflowTriggerRequest,
  WorkflowTriggerResponse,
  WorkflowListResponse,
  WorkflowStats,
  WorkflowStatus,
  WorkflowType,
} from '@/types/workflow';
import type {
  Approval,
  ApprovalActionRequest,
  ApprovalActionResponse,
  ApprovalListResponse,
} from '@/types/approval';
import type {
  Document,
  DocumentUploadResponse,
  DocumentListResponse,
  DocumentDeleteResponse,
  SearchRequest,
  SearchResponse,
  UsageSummary,
  FileType,
} from '@/types/document';

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
const CLIENT_ID = process.env.NEXT_PUBLIC_CLIENT_ID || 'a1b2c3d4-e5f6-7890-abcd-ef1234567890';

// Error class for API errors
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

// Generic fetch wrapper with error handling
async function apiFetch<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const defaultHeaders: HeadersInit = {
    'Content-Type': 'application/json',
    'X-Client-ID': CLIENT_ID,
  };

  const response = await fetch(url, {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options.headers,
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      errorData.detail || `HTTP ${response.status}: ${response.statusText}`,
      response.status,
      errorData.code
    );
  }

  return response.json();
}

// =============================================================================
// WORKFLOW API
// =============================================================================

export const workflowApi = {
  /**
   * Trigger a new workflow
   */
  async trigger(request: WorkflowTriggerRequest): Promise<WorkflowTriggerResponse> {
    return apiFetch<WorkflowTriggerResponse>('/api/v1/workflows', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },

  /**
   * List workflows with optional filters
   */
  async list(params?: {
    status?: WorkflowStatus;
    workflow_type?: WorkflowType;
    page?: number;
    page_size?: number;
  }): Promise<WorkflowListResponse> {
    const searchParams = new URLSearchParams();
    if (params?.status) searchParams.set('status', params.status);
    if (params?.workflow_type) searchParams.set('workflow_type', params.workflow_type);
    if (params?.page) searchParams.set('page', params.page.toString());
    if (params?.page_size) searchParams.set('page_size', params.page_size.toString());

    const query = searchParams.toString();
    return apiFetch<WorkflowListResponse>(`/api/v1/workflows${query ? `?${query}` : ''}`);
  },

  /**
   * Get workflow by ID
   */
  async get(workflowId: string): Promise<Workflow> {
    return apiFetch<Workflow>(`/api/v1/workflows/${workflowId}`);
  },

  /**
   * Get workflow statistics
   */
  async getStats(): Promise<WorkflowStats> {
    return apiFetch<WorkflowStats>('/api/v1/workflows/stats');
  },

  /**
   * Resume a paused workflow
   */
  async resume(workflowId: string, additionalInput?: Record<string, unknown>): Promise<WorkflowTriggerResponse> {
    return apiFetch<WorkflowTriggerResponse>(`/api/v1/workflows/${workflowId}/resume`, {
      method: 'POST',
      body: JSON.stringify({ additional_input: additionalInput }),
    });
  },

  /**
   * Cancel a workflow
   */
  async cancel(workflowId: string, reason?: string): Promise<{ success: boolean; message: string }> {
    return apiFetch<{ success: boolean; message: string }>(`/api/v1/workflows/${workflowId}/cancel`, {
      method: 'DELETE',
      body: JSON.stringify({ reason }),
    });
  },
};

// =============================================================================
// APPROVAL API
// =============================================================================

export const approvalApi = {
  /**
   * List pending approvals
   */
  async list(params?: {
    status?: 'pending' | 'approved' | 'rejected';
    page?: number;
    page_size?: number;
  }): Promise<ApprovalListResponse> {
    const searchParams = new URLSearchParams();
    if (params?.status) searchParams.set('status', params.status);
    if (params?.page) searchParams.set('page', params.page.toString());
    if (params?.page_size) searchParams.set('page_size', params.page_size.toString());

    const query = searchParams.toString();
    return apiFetch<ApprovalListResponse>(`/api/v1/approvals${query ? `?${query}` : ''}`);
  },

  /**
   * Get approval by ID
   */
  async get(approvalId: string): Promise<Approval> {
    return apiFetch<Approval>(`/api/v1/approvals/${approvalId}`);
  },

  /**
   * Approve or reject an approval
   */
  async action(request: ApprovalActionRequest): Promise<ApprovalActionResponse> {
    return apiFetch<ApprovalActionResponse>(
      `/api/v1/approvals/${request.approval_id}/${request.action}`,
      {
        method: 'POST',
        body: JSON.stringify({
          reason: request.reason,
          modifications: request.modifications,
        }),
      }
    );
  },
};

// =============================================================================
// DOCUMENTS API
// =============================================================================

export const documentsApi = {
  /**
   * Upload a document for RAG processing
   * Note: Uses FormData for multipart upload
   */
  async upload(file: File, clientId: string): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('client_id', clientId);

    const url = `${API_BASE_URL}/api/v1/documents/upload`;
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
      // Don't set Content-Type - browser will set it with boundary
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new ApiError(
        errorData.detail || `HTTP ${response.status}: ${response.statusText}`,
        response.status,
        errorData.code
      );
    }

    return response.json();
  },

  /**
   * List documents for a client
   */
  async list(params: {
    client_id: string;
    file_types?: FileType[];
    limit?: number;
    offset?: number;
  }): Promise<DocumentListResponse> {
    const searchParams = new URLSearchParams();
    searchParams.set('client_id', params.client_id);
    if (params.file_types?.length) {
      searchParams.set('file_types', params.file_types.join(','));
    }
    if (params.limit) searchParams.set('limit', params.limit.toString());
    if (params.offset) searchParams.set('offset', params.offset.toString());

    return apiFetch<DocumentListResponse>(`/api/v1/documents?${searchParams.toString()}`);
  },

  /**
   * Get document by ID
   */
  async get(documentId: string): Promise<Document> {
    return apiFetch<Document>(`/api/v1/documents/${documentId}`);
  },

  /**
   * Delete a document
   */
  async delete(documentId: string): Promise<DocumentDeleteResponse> {
    return apiFetch<DocumentDeleteResponse>(`/api/v1/documents/${documentId}`, {
      method: 'DELETE',
    });
  },

  /**
   * Semantic search across documents
   */
  async search(clientId: string, request: SearchRequest): Promise<SearchResponse> {
    return apiFetch<SearchResponse>(`/api/v1/documents/search?client_id=${clientId}`, {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },

  /**
   * Get usage summary for a client
   */
  async getUsage(clientId: string): Promise<UsageSummary> {
    return apiFetch<UsageSummary>(`/api/v1/documents/usage/${clientId}`);
  },
};

// =============================================================================
// SETTINGS API
// =============================================================================

export interface IntegrationStatus {
  name: string;
  status: string;
  status_color: string;
  details?: string | null;
}

export interface SettingsSection {
  id: string;
  title: string;
  description: string;
  icon: string;
  items: IntegrationStatus[];
}

export interface SettingsStatusResponse {
  sections: SettingsSection[];
}

export interface SettingsHealthResponse {
  status: string;
  version: string;
  environment: string;
  llm_provider: string;
  integrations: Record<string, boolean>;
}

export const settingsApi = {
  /**
   * Get complete settings status
   */
  async getStatus(): Promise<SettingsStatusResponse> {
    return apiFetch<SettingsStatusResponse>('/api/v1/settings/status');
  },

  /**
   * Get settings health check
   */
  async getHealth(): Promise<SettingsHealthResponse> {
    return apiFetch<SettingsHealthResponse>('/api/v1/settings/health');
  },
};

// =============================================================================
// HEALTH CHECK
// =============================================================================

export const healthApi = {
  /**
   * Check backend health
   */
  async check(): Promise<{ status: string; version: string }> {
    return apiFetch<{ status: string; version: string }>('/health');
  },
};

// Export all APIs
export const api = {
  workflow: workflowApi,
  approval: approvalApi,
  documents: documentsApi,
  settings: settingsApi,
  health: healthApi,
};

export default api;

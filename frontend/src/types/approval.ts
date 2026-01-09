/**
 * Approval Types
 * ==============
 * TypeScript types for Human-in-the-Loop approvals.
 */

// Approval Types
export type ApprovalType =
  | 'crm_update'
  | 'email_send'
  | 'content_publish'
  | 'task_create'
  | 'deal_update'
  | 'contact_create';

// Approval Status
export type ApprovalStatus = 'pending' | 'approved' | 'rejected';

// Approval Interface
export interface Approval {
  id: string;
  workflow_id: string;
  client_id: string;
  approval_type: ApprovalType;
  title: string;
  description: string | null;
  data: Record<string, unknown>;
  status: ApprovalStatus;
  created_at: string;
  updated_at: string;
  resolved_at: string | null;
  resolved_by: string | null;
  // Computed fields
  time_pending_seconds?: number;
}

// Approval Action Request
export interface ApprovalActionRequest {
  approval_id: string;
  action: 'approve' | 'reject';
  reason?: string;
  modifications?: Record<string, unknown>;
}

// Approval Action Response
export interface ApprovalActionResponse {
  success: boolean;
  approval_id: string;
  new_status: ApprovalStatus;
  message: string;
}

// Approval List Response
export interface ApprovalListResponse {
  approvals: Approval[];
  total: number;
  pending_count: number;
}

// Helper functions
export function getApprovalTypeLabel(type: ApprovalType): string {
  const labels: Record<ApprovalType, string> = {
    crm_update: 'CRM Update',
    email_send: 'Send Email',
    content_publish: 'Publish Content',
    task_create: 'Create Task',
    deal_update: 'Update Deal',
    contact_create: 'Create Contact',
  };
  return labels[type] || type;
}

export function getApprovalStatusColor(status: ApprovalStatus): string {
  const colors: Record<ApprovalStatus, string> = {
    pending: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    approved: 'bg-green-100 text-green-800 border-green-200',
    rejected: 'bg-red-100 text-red-800 border-red-200',
  };
  return colors[status] || 'bg-gray-100 text-gray-800';
}

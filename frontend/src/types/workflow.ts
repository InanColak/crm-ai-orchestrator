/**
 * Workflow Types
 * ==============
 * TypeScript types matching backend Pydantic schemas.
 */

// Workflow Types (from backend/app/schemas/workflow.py)
export type WorkflowType =
  | 'lead_research'
  | 'meeting_analysis'
  | 'email_generation'
  | 'content_pipeline'
  | 'market_research'
  | 'seo_analysis';

// Workflow Status
export type WorkflowStatus =
  | 'pending'
  | 'running'
  | 'waiting_approval'
  | 'completed'
  | 'failed'
  | 'cancelled';

// Workflow Priority
export type WorkflowPriority = 'low' | 'normal' | 'high' | 'urgent';

// Base Workflow Interface
export interface Workflow {
  id: string;
  client_id: string;
  workflow_type: WorkflowType;
  status: WorkflowStatus;
  priority: WorkflowPriority;
  input_data: Record<string, unknown>;
  output_data: Record<string, unknown> | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
  // Computed fields
  duration_seconds?: number;
  steps_completed?: number;
  total_steps?: number;
}

// Workflow Trigger Request
export interface WorkflowTriggerRequest {
  workflow_type: WorkflowType;
  input_data: Record<string, unknown>;
  priority?: WorkflowPriority;
  metadata?: Record<string, unknown>;
}

// Workflow Trigger Response
export interface WorkflowTriggerResponse {
  workflow_id: string;
  status: WorkflowStatus;
  message: string;
}

// Workflow List Response
export interface WorkflowListResponse {
  workflows: Workflow[];
  total: number;
  page: number;
  page_size: number;
  has_more: boolean;
}

// Workflow Stats
export interface WorkflowStats {
  total_workflows: number;
  pending_count: number;
  running_count: number;
  completed_count: number;
  failed_count: number;
  waiting_approval_count: number;
  avg_completion_time_seconds: number;
  success_rate: number;
}

// Helper functions
export function getStatusColor(status: WorkflowStatus): string {
  const colors: Record<WorkflowStatus, string> = {
    pending: 'bg-yellow-100 text-yellow-800',
    running: 'bg-blue-100 text-blue-800',
    waiting_approval: 'bg-purple-100 text-purple-800',
    completed: 'bg-green-100 text-green-800',
    failed: 'bg-red-100 text-red-800',
    cancelled: 'bg-gray-100 text-gray-800',
  };
  return colors[status] || 'bg-gray-100 text-gray-800';
}

export function getStatusLabel(status: WorkflowStatus): string {
  const labels: Record<WorkflowStatus, string> = {
    pending: 'Pending',
    running: 'Running',
    waiting_approval: 'Awaiting Approval',
    completed: 'Completed',
    failed: 'Failed',
    cancelled: 'Cancelled',
  };
  return labels[status] || status;
}

export function getWorkflowTypeLabel(type: WorkflowType): string {
  const labels: Record<WorkflowType, string> = {
    lead_research: 'Lead Research',
    meeting_analysis: 'Meeting Analysis',
    email_generation: 'Email Generation',
    content_pipeline: 'Content Pipeline',
    market_research: 'Market Research',
    seo_analysis: 'SEO Analysis',
  };
  return labels[type] || type;
}

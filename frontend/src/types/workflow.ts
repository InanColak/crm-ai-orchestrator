/**
 * Workflow Types
 * ==============
 * TypeScript types matching backend Pydantic schemas.
 */

// Workflow Types (from backend/app/schemas/workflow.py)
export type WorkflowType =
  | 'meeting_analysis'
  | 'lead_research'
  | 'intelligence_only'
  | 'sales_ops_only'
  | 'content_only'
  | 'full_cycle';

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
    meeting_analysis: 'Meeting Analysis',
    lead_research: 'Lead Research',
    intelligence_only: 'Intelligence Research',
    sales_ops_only: 'Sales Operations',
    content_only: 'Content Pipeline',
    full_cycle: 'Full Cycle',
  };
  return labels[type] || type;
}

// Workflow type definitions with input fields
export interface WorkflowTypeDefinition {
  value: WorkflowType;
  label: string;
  description: string;
  ready: boolean;
  inputFields: {
    name: string;
    label: string;
    type: 'text' | 'textarea' | 'select';
    placeholder?: string;
    required?: boolean;
    options?: { value: string; label: string }[];
  }[];
}

export const WORKFLOW_TYPES: WorkflowTypeDefinition[] = [
  {
    value: 'meeting_analysis',
    label: 'Meeting Analysis',
    description: 'Analyze meeting transcript, extract action items, and update CRM',
    ready: true,
    inputFields: [
      {
        name: 'transcript',
        label: 'Meeting Transcript',
        type: 'textarea',
        placeholder: 'Paste meeting transcript here...',
        required: true,
      },
      {
        name: 'meeting_title',
        label: 'Meeting Title',
        type: 'text',
        placeholder: 'e.g., Sales Call with Acme Corp',
        required: false,
      },
      {
        name: 'participants',
        label: 'Participants',
        type: 'text',
        placeholder: 'e.g., John, Jane, Mike',
        required: false,
      },
    ],
  },
  {
    value: 'lead_research',
    label: 'Lead Research',
    description: 'Research leads and enrich with company data',
    ready: true,
    inputFields: [
      {
        name: 'company_name',
        label: 'Company Name',
        type: 'text',
        placeholder: 'e.g., Acme Corporation',
        required: true,
      },
      {
        name: 'domain',
        label: 'Website Domain',
        type: 'text',
        placeholder: 'e.g., acme.com',
        required: false,
      },
      {
        name: 'contact_name',
        label: 'Contact Name',
        type: 'text',
        placeholder: 'e.g., John Smith',
        required: false,
      },
    ],
  },
  {
    value: 'intelligence_only',
    label: 'Intelligence Research',
    description: 'Market research, SEO analysis, and audience building',
    ready: true,
    inputFields: [
      {
        name: 'company_name',
        label: 'Company/Topic',
        type: 'text',
        placeholder: 'e.g., Your company or research topic',
        required: true,
      },
      {
        name: 'industry',
        label: 'Industry',
        type: 'text',
        placeholder: 'e.g., SaaS, Healthcare, Finance',
        required: false,
      },
      {
        name: 'research_scope',
        label: 'Research Scope',
        type: 'select',
        options: [
          { value: 'basic', label: 'Basic' },
          { value: 'standard', label: 'Standard' },
          { value: 'comprehensive', label: 'Comprehensive' },
        ],
        required: false,
      },
    ],
  },
  {
    value: 'sales_ops_only',
    label: 'Sales Operations',
    description: 'Full sales workflow: lead research, email generation, CRM updates',
    ready: true,
    inputFields: [
      {
        name: 'company_name',
        label: 'Target Company',
        type: 'text',
        placeholder: 'e.g., TechStart Inc',
        required: true,
      },
      {
        name: 'contact_email',
        label: 'Contact Email',
        type: 'text',
        placeholder: 'e.g., john@company.com',
        required: false,
      },
      {
        name: 'objective',
        label: 'Sales Objective',
        type: 'textarea',
        placeholder: 'What are you trying to achieve?',
        required: false,
      },
    ],
  },
  {
    value: 'content_only',
    label: 'Content Pipeline',
    description: 'Content creation and publishing workflow',
    ready: false,
    inputFields: [
      {
        name: 'topic',
        label: 'Content Topic',
        type: 'text',
        placeholder: 'e.g., AI in Sales',
        required: true,
      },
    ],
  },
  {
    value: 'full_cycle',
    label: 'Full Cycle',
    description: 'Complete end-to-end workflow (all squads)',
    ready: false,
    inputFields: [
      {
        name: 'objective',
        label: 'Campaign Objective',
        type: 'textarea',
        placeholder: 'Describe your full campaign objective...',
        required: true,
      },
    ],
  },
];

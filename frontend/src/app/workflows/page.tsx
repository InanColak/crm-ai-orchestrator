'use client';

import { useState, useEffect } from 'react';
import { Header } from '@/components/layout';
import { Card, Button, StatusBadge, EmptyState, CardSkeleton, useToast } from '@/components/ui';
import { Plus, Workflow, ChevronRight, Search, RefreshCw, AlertCircle } from 'lucide-react';
import { cn, formatRelativeTime } from '@/lib/utils';
import { useWorkflows, useWorkflowRealtime } from '@/hooks';
import type { WorkflowStatus, WorkflowType, Workflow as WorkflowT } from '@/types/workflow';
import { getWorkflowTypeLabel } from '@/types/workflow';

// Mock data for fallback when API is unavailable
const mockWorkflows: WorkflowT[] = [
  {
    id: 'wf-001',
    client_id: 'client-123',
    workflow_type: 'lead_research',
    status: 'completed',
    priority: 'normal',
    input_data: { company_name: 'Acme Corporation' },
    output_data: null,
    error_message: null,
    created_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString(),
    completed_at: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'wf-002',
    client_id: 'client-123',
    workflow_type: 'email_generation',
    status: 'running',
    priority: 'high',
    input_data: { recipient: 'john@techstart.io' },
    output_data: null,
    error_message: null,
    created_at: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 1 * 60 * 1000).toISOString(),
    completed_at: null,
  },
  {
    id: 'wf-003',
    client_id: 'client-123',
    workflow_type: 'meeting_analysis',
    status: 'waiting_approval',
    priority: 'normal',
    input_data: { meeting_title: 'Q4 Planning Session' },
    output_data: null,
    error_message: null,
    created_at: new Date(Date.now() - 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
    completed_at: null,
  },
  {
    id: 'wf-004',
    client_id: 'client-123',
    workflow_type: 'lead_research',
    status: 'failed',
    priority: 'low',
    input_data: { company_name: 'Unknown LLC' },
    output_data: null,
    error_message: 'Could not find company information',
    created_at: new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString(),
    completed_at: null,
  },
  {
    id: 'wf-005',
    client_id: 'client-123',
    workflow_type: 'content_pipeline',
    status: 'pending',
    priority: 'normal',
    input_data: { topic: 'AI in Sales' },
    output_data: null,
    error_message: null,
    created_at: new Date(Date.now() - 10 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 10 * 60 * 1000).toISOString(),
    completed_at: null,
  },
];

const statusFilters: { label: string; value: WorkflowStatus | 'all' }[] = [
  { label: 'All', value: 'all' },
  { label: 'Running', value: 'running' },
  { label: 'Pending', value: 'pending' },
  { label: 'Awaiting Approval', value: 'waiting_approval' },
  { label: 'Completed', value: 'completed' },
  { label: 'Failed', value: 'failed' },
];

export default function WorkflowsPage() {
  const [selectedStatus, setSelectedStatus] = useState<WorkflowStatus | 'all'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [displayWorkflows, setDisplayWorkflows] = useState<WorkflowT[]>(mockWorkflows);
  const [useMockData, setUseMockData] = useState(false);

  const { success, error: showError, info } = useToast();

  // Fetch workflows from API
  const {
    workflows: apiWorkflows,
    isLoading,
    error: apiError,
    refetch,
  } = useWorkflows({
    status: selectedStatus === 'all' ? undefined : selectedStatus,
  });

  // Realtime updates
  useWorkflowRealtime(
    (updated) => {
      info('Workflow Updated', `Workflow ${updated.id} is now ${updated.status}`);
      refetch();
    },
    !useMockData
  );

  // Use API data or fallback to mock
  useEffect(() => {
    if (apiError) {
      setUseMockData(true);
      setDisplayWorkflows(mockWorkflows);
    } else if (apiWorkflows.length > 0) {
      setUseMockData(false);
      setDisplayWorkflows(apiWorkflows);
    }
  }, [apiWorkflows, apiError]);

  // Filter workflows based on search and status
  const filteredWorkflows = displayWorkflows.filter((wf) => {
    // Status filter (only for mock data, API handles this)
    if (useMockData && selectedStatus !== 'all' && wf.status !== selectedStatus) {
      return false;
    }

    // Search filter
    if (searchQuery) {
      const searchLower = searchQuery.toLowerCase();
      const typeLabel = getWorkflowTypeLabel(wf.workflow_type).toLowerCase();
      const inputStr = JSON.stringify(wf.input_data).toLowerCase();
      return typeLabel.includes(searchLower) || inputStr.includes(searchLower) || wf.id.toLowerCase().includes(searchLower);
    }
    return true;
  });

  const handleRefresh = async () => {
    if (useMockData) {
      info('Demo Mode', 'Using mock data - backend not connected');
    } else {
      await refetch();
      success('Refreshed', 'Workflow list updated');
    }
  };

  return (
    <div className="min-h-screen">
      <Header
        title="Workflows"
        description="Monitor and manage your AI workflow executions"
      />

      <div className="p-6 space-y-6">
        {/* API Error Banner */}
        {useMockData && (
          <div className="flex items-center gap-3 rounded-lg bg-yellow-50 border border-yellow-200 px-4 py-3">
            <AlertCircle className="h-5 w-5 text-yellow-600 shrink-0" />
            <div className="flex-1">
              <p className="text-sm font-medium text-yellow-800">Demo Mode</p>
              <p className="text-xs text-yellow-700">
                Backend API not available. Showing mock data for demonstration.
              </p>
            </div>
          </div>
        )}

        {/* Actions Bar */}
        <div className="flex flex-col sm:flex-row gap-4 justify-between">
          <div className="flex items-center gap-2">
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search workflows..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="input pl-9 w-64"
              />
            </div>
            {/* Refresh */}
            <Button
              variant="ghost"
              size="sm"
              onClick={handleRefresh}
              loading={isLoading}
              icon={<RefreshCw className="h-4 w-4" />}
            >
              Refresh
            </Button>
          </div>

          <Button icon={<Plus className="h-4 w-4" />}>
            New Workflow
          </Button>
        </div>

        {/* Status Filters */}
        <div className="flex items-center gap-2 overflow-x-auto pb-2">
          {statusFilters.map((filter) => (
            <button
              key={filter.value}
              onClick={() => setSelectedStatus(filter.value)}
              className={cn(
                'px-3 py-1.5 rounded-full text-sm font-medium whitespace-nowrap transition-colors',
                selectedStatus === filter.value
                  ? 'bg-brand-100 text-brand-700'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              )}
            >
              {filter.label}
            </button>
          ))}
        </div>

        {/* Loading State */}
        {isLoading && !useMockData && (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <CardSkeleton key={i} />
            ))}
          </div>
        )}

        {/* Workflows List */}
        {!isLoading && filteredWorkflows.length === 0 ? (
          <EmptyState
            icon={<Workflow className="h-6 w-6 text-gray-400" />}
            title="No workflows found"
            description={
              searchQuery
                ? 'Try adjusting your search or filters'
                : 'Get started by triggering a new workflow'
            }
            action={
              <Button icon={<Plus className="h-4 w-4" />}>
                New Workflow
              </Button>
            }
          />
        ) : (
          <div className="space-y-3">
            {filteredWorkflows.map((workflow) => (
              <Card
                key={workflow.id}
                hover
                className="cursor-pointer"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gray-100">
                      <Workflow className="h-5 w-5 text-gray-600" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <p className="text-sm font-medium text-gray-900">
                          {getWorkflowTypeLabel(workflow.workflow_type)}
                        </p>
                        <StatusBadge status={workflow.status} size="sm" />
                      </div>
                      <p className="text-xs text-gray-500 mt-0.5">
                        {Object.values(workflow.input_data)[0] as string} • Started {formatRelativeTime(workflow.created_at)}
                      </p>
                      {workflow.error_message && (
                        <p className="text-xs text-red-500 mt-0.5">
                          {workflow.error_message}
                        </p>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className="text-xs text-gray-400 font-mono">{workflow.id}</span>
                    <ChevronRight className="h-4 w-4 text-gray-400" />
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

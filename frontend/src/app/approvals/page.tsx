'use client';

import { useState, useEffect } from 'react';
import { Header } from '@/components/layout';
import { Card, Button, EmptyState, CardSkeleton, useToast } from '@/components/ui';
import { CheckCircle, XCircle, Clock, Eye, AlertTriangle, AlertCircle, RefreshCw } from 'lucide-react';
import { cn, formatRelativeTime } from '@/lib/utils';
import { useApprovals, useApprovalAction, useApprovalRealtime } from '@/hooks';
import type { Approval, ApprovalType, ApprovalStatus } from '@/types/approval';
import { getApprovalTypeLabel } from '@/types/approval';

// Mock pending approvals for fallback
const mockApprovals: Approval[] = [
  {
    id: 'apr-001',
    workflow_id: 'wf-003',
    client_id: 'client-123',
    approval_type: 'crm_update',
    title: 'Update Contact: John Smith',
    description: 'Add enrichment data and update lead score from research',
    data: {
      contact_id: 'contact-456',
      updates: {
        lead_score: 85,
        company_size: '50-100',
        industry: 'Technology',
      },
    },
    status: 'pending',
    created_at: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
    resolved_at: null,
    resolved_by: null,
  },
  {
    id: 'apr-002',
    workflow_id: 'wf-002',
    client_id: 'client-123',
    approval_type: 'email_send',
    title: 'Send Cold Outreach Email',
    description: 'Personalized email to Sarah at TechStart about their expansion',
    data: {
      recipient: 'sarah@techstart.io',
      subject: 'Congrats on the Series B!',
      body: 'Hi Sarah,\n\nI noticed TechStart just closed their Series B round - congratulations!\n\nCompanies in your position often face challenges scaling their sales operations. I\'ve helped three similar SaaS companies streamline their processes.\n\nWould a 15-minute call next week make sense?\n\nBest,\nJohn',
    },
    status: 'pending',
    created_at: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
    resolved_at: null,
    resolved_by: null,
  },
  {
    id: 'apr-003',
    workflow_id: 'wf-003',
    client_id: 'client-123',
    approval_type: 'task_create',
    title: 'Create Follow-up Task',
    description: 'Schedule follow-up call based on meeting notes analysis',
    data: {
      task_type: 'call',
      due_date: '2024-01-20',
      assignee: 'sales_rep_001',
      notes: 'Discuss pricing options and demo scheduling',
    },
    status: 'pending',
    created_at: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
    resolved_at: null,
    resolved_by: null,
  },
];

const typeIcons: Record<ApprovalType, React.ComponentType<{ className?: string }>> = {
  crm_update: AlertTriangle,
  email_send: CheckCircle,
  content_publish: CheckCircle,
  task_create: Clock,
  deal_update: AlertTriangle,
  contact_create: CheckCircle,
};

export default function ApprovalsPage() {
  const [displayApprovals, setDisplayApprovals] = useState<Approval[]>(mockApprovals);
  const [selectedApproval, setSelectedApproval] = useState<Approval | null>(null);
  const [useMockData, setUseMockData] = useState(false);

  const { success, error: showError, info, warning } = useToast();

  // Fetch approvals from API
  const {
    approvals: apiApprovals,
    pendingCount,
    isLoading,
    error: apiError,
    refetch,
  } = useApprovals({ status: 'pending' });

  // Approval actions
  const { approve, reject, isLoading: isActionLoading } = useApprovalAction();

  // Realtime updates for new approvals
  useApprovalRealtime(
    (newApproval) => {
      warning('New Approval', `${newApproval.title} requires your attention`);
      refetch();
    },
    !useMockData
  );

  // Use API data or fallback to mock
  useEffect(() => {
    if (apiError) {
      setUseMockData(true);
      setDisplayApprovals(mockApprovals);
    } else if (apiApprovals.length > 0) {
      setUseMockData(false);
      setDisplayApprovals(apiApprovals);
    }
  }, [apiApprovals, apiError]);

  const pendingApprovals = displayApprovals.filter((a) => a.status === 'pending');
  const displayPendingCount = useMockData ? pendingApprovals.length : pendingCount;

  const handleApprove = async (approvalId: string) => {
    if (useMockData) {
      // Mock approval
      setDisplayApprovals((prev) =>
        prev.map((a) =>
          a.id === approvalId
            ? { ...a, status: 'approved' as ApprovalStatus, resolved_at: new Date().toISOString() }
            : a
        )
      );
      setSelectedApproval(null);
      success('Approved', 'Action has been approved successfully');
      return;
    }

    try {
      await approve(approvalId);
      success('Approved', 'Action has been approved and will be executed');
      setSelectedApproval(null);
      refetch();
    } catch {
      showError('Failed', 'Could not approve the action');
    }
  };

  const handleReject = async (approvalId: string) => {
    if (useMockData) {
      // Mock rejection
      setDisplayApprovals((prev) =>
        prev.map((a) =>
          a.id === approvalId
            ? { ...a, status: 'rejected' as ApprovalStatus, resolved_at: new Date().toISOString() }
            : a
        )
      );
      setSelectedApproval(null);
      info('Rejected', 'Action has been rejected');
      return;
    }

    try {
      await reject(approvalId, 'Rejected by user');
      info('Rejected', 'Action has been rejected');
      setSelectedApproval(null);
      refetch();
    } catch {
      showError('Failed', 'Could not reject the action');
    }
  };

  const handleRefresh = async () => {
    if (useMockData) {
      info('Demo Mode', 'Using mock data - backend not connected');
    } else {
      await refetch();
      success('Refreshed', 'Approval list updated');
    }
  };

  return (
    <div className="min-h-screen">
      <Header
        title="Approvals"
        description={`${displayPendingCount} pending approval${displayPendingCount !== 1 ? 's' : ''} require your attention`}
      />

      <div className="p-6">
        {/* API Error Banner */}
        {useMockData && (
          <div className="flex items-center gap-3 rounded-lg bg-yellow-50 border border-yellow-200 px-4 py-3 mb-6">
            <AlertCircle className="h-5 w-5 text-yellow-600 shrink-0" />
            <div className="flex-1">
              <p className="text-sm font-medium text-yellow-800">Demo Mode</p>
              <p className="text-xs text-yellow-700">
                Backend API not available. Showing mock data for demonstration.
              </p>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleRefresh}
              icon={<RefreshCw className="h-4 w-4" />}
            >
              Retry
            </Button>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Approvals List */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-medium text-gray-500 uppercase tracking-wide">
                Pending Approvals
              </h2>
              {!useMockData && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleRefresh}
                  loading={isLoading}
                  icon={<RefreshCw className="h-4 w-4" />}
                >
                  Refresh
                </Button>
              )}
            </div>

            {isLoading && !useMockData ? (
              <div className="space-y-3">
                {[1, 2, 3].map((i) => (
                  <CardSkeleton key={i} />
                ))}
              </div>
            ) : pendingApprovals.length === 0 ? (
              <EmptyState
                icon={<CheckCircle className="h-6 w-6 text-green-500" />}
                title="All caught up!"
                description="No pending approvals at the moment"
              />
            ) : (
              <div className="space-y-3">
                {pendingApprovals.map((approval) => {
                  const Icon = typeIcons[approval.approval_type] || CheckCircle;
                  const isSelected = selectedApproval?.id === approval.id;

                  return (
                    <Card
                      key={approval.id}
                      hover
                      className={cn(
                        'cursor-pointer transition-all',
                        isSelected && 'ring-2 ring-brand-500 border-brand-500'
                      )}
                      onClick={() => setSelectedApproval(approval)}
                    >
                      <div className="flex items-start gap-3">
                        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-yellow-100 shrink-0">
                          <Icon className="h-4 w-4 text-yellow-600" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between gap-2">
                            <p className="text-sm font-medium text-gray-900 truncate">
                              {approval.title}
                            </p>
                            <span className="text-xs text-gray-400 shrink-0">
                              {formatRelativeTime(approval.created_at)}
                            </span>
                          </div>
                          <p className="text-xs text-gray-500 mt-0.5 line-clamp-2">
                            {approval.description}
                          </p>
                          <div className="flex items-center gap-2 mt-2">
                            <span className="inline-flex items-center rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-600">
                              {getApprovalTypeLabel(approval.approval_type)}
                            </span>
                          </div>
                        </div>
                      </div>
                    </Card>
                  );
                })}
              </div>
            )}
          </div>

          {/* Approval Detail Panel */}
          <div className="lg:sticky lg:top-24 h-fit">
            {selectedApproval ? (
              <Card padding="none">
                <div className="p-4 border-b border-gray-100">
                  <h3 className="text-lg font-semibold text-gray-900">
                    {selectedApproval.title}
                  </h3>
                  <p className="text-sm text-gray-500 mt-1">
                    {selectedApproval.description}
                  </p>
                  <div className="flex items-center gap-2 mt-2">
                    <span className="inline-flex items-center rounded-full bg-yellow-100 px-2 py-0.5 text-xs font-medium text-yellow-800">
                      {getApprovalTypeLabel(selectedApproval.approval_type)}
                    </span>
                    <span className="text-xs text-gray-400">
                      {formatRelativeTime(selectedApproval.created_at)}
                    </span>
                  </div>
                </div>

                <div className="p-4 space-y-4">
                  <div>
                    <h4 className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">
                      Action Details
                    </h4>
                    <pre className="bg-gray-50 rounded-lg p-3 text-xs text-gray-700 overflow-x-auto max-h-64">
                      {JSON.stringify(selectedApproval.data, null, 2)}
                    </pre>
                  </div>

                  <div className="flex items-center gap-2 pt-2 border-t border-gray-100">
                    <Button
                      variant="primary"
                      className="flex-1"
                      icon={<CheckCircle className="h-4 w-4" />}
                      onClick={() => handleApprove(selectedApproval.id)}
                      loading={isActionLoading}
                    >
                      Approve
                    </Button>
                    <Button
                      variant="danger"
                      className="flex-1"
                      icon={<XCircle className="h-4 w-4" />}
                      onClick={() => handleReject(selectedApproval.id)}
                      loading={isActionLoading}
                    >
                      Reject
                    </Button>
                  </div>
                </div>
              </Card>
            ) : (
              <Card className="flex items-center justify-center h-64">
                <div className="text-center">
                  <Eye className="mx-auto h-10 w-10 text-gray-300" />
                  <p className="mt-2 text-sm text-gray-500">
                    Select an approval to view details
                  </p>
                </div>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

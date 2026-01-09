'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Header } from '@/components/layout';
import { Card, CardHeader, Skeleton, useToast } from '@/components/ui';
import {
  Workflow,
  CheckCircle,
  TrendingUp,
  Mail,
  Users,
  FileText,
  AlertCircle,
} from 'lucide-react';
import { useWorkflows, useWorkflowStats, useApprovals } from '@/hooks';
import { formatRelativeTime } from '@/lib/utils';
import type { Workflow as WorkflowT } from '@/types/workflow';
import { getWorkflowTypeLabel } from '@/types/workflow';

// Mock data for fallback
const mockStats = {
  activeWorkflows: 12,
  pendingApprovals: 5,
  emailsGenerated: 48,
  leadsResearched: 156,
};

const mockRecentWorkflows: WorkflowT[] = [
  {
    id: 'wf-001',
    client_id: 'client-123',
    workflow_type: 'lead_research',
    status: 'completed',
    priority: 'normal',
    input_data: { company_name: 'Acme Corp' },
    output_data: null,
    error_message: null,
    created_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date().toISOString(),
    completed_at: new Date().toISOString(),
  },
  {
    id: 'wf-002',
    client_id: 'client-123',
    workflow_type: 'email_generation',
    status: 'running',
    priority: 'high',
    input_data: { recipient: 'TechStart Inc' },
    output_data: null,
    error_message: null,
    created_at: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
    updated_at: new Date().toISOString(),
    completed_at: null,
  },
  {
    id: 'wf-003',
    client_id: 'client-123',
    workflow_type: 'meeting_analysis',
    status: 'waiting_approval',
    priority: 'normal',
    input_data: { meeting_title: 'Sales Call - DataFlow' },
    output_data: null,
    error_message: null,
    created_at: new Date(Date.now() - 60 * 60 * 1000).toISOString(),
    updated_at: new Date().toISOString(),
    completed_at: null,
  },
  {
    id: 'wf-004',
    client_id: 'client-123',
    workflow_type: 'lead_research',
    status: 'failed',
    priority: 'low',
    input_data: { company_name: 'Unknown Company' },
    output_data: null,
    error_message: 'Could not find company',
    created_at: new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date().toISOString(),
    completed_at: null,
  },
];

const statusColors: Record<string, string> = {
  completed: 'bg-green-500/20 text-green-400 border border-green-500/30',
  running: 'bg-[#00C0F0] text-white border-none',
  waiting_approval: 'bg-purple-500/20 text-purple-400 border border-purple-500/30',
  failed: 'bg-red-500/20 text-red-400 border border-red-500/30',
  pending: 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30',
};

export default function DashboardPage() {
  const [useMockData, setUseMockData] = useState(false);

  const { info } = useToast();

  // Fetch real data
  const { stats, isLoading: statsLoading, error: statsError } = useWorkflowStats();
  const { workflows, isLoading: workflowsLoading, error: workflowsError } = useWorkflows({ pageSize: 4 });
  const { pendingCount, error: approvalsError } = useApprovals({ status: 'pending' });

  // Check if we should use mock data
  useEffect(() => {
    if (statsError || workflowsError || approvalsError) {
      setUseMockData(true);
    } else if (stats || workflows.length > 0) {
      setUseMockData(false);
    }
  }, [stats, statsError, workflows, workflowsError, approvalsError]);

  // Display data (API or mock)
  const displayStats = useMockData
    ? [
        { label: 'Active Workflows', value: mockStats.activeWorkflows.toString(), change: '+3 today', icon: Workflow, color: 'text-[#00C0F0]', bgColor: 'bg-[#00C0F0]/20' },
        { label: 'Pending Approvals', value: mockStats.pendingApprovals.toString(), change: 'Requires action', icon: CheckCircle, color: 'text-purple-400', bgColor: 'bg-purple-500/20' },
        { label: 'Emails Generated', value: mockStats.emailsGenerated.toString(), change: 'This week', icon: Mail, color: 'text-green-400', bgColor: 'bg-green-500/20' },
        { label: 'Leads Researched', value: mockStats.leadsResearched.toString(), change: 'This month', icon: Users, color: 'text-orange-400', bgColor: 'bg-orange-500/20' },
      ]
    : [
        { label: 'Active Workflows', value: (stats?.running_count ?? 0).toString(), change: `${stats?.pending_count ?? 0} pending`, icon: Workflow, color: 'text-[#00C0F0]', bgColor: 'bg-[#00C0F0]/20' },
        { label: 'Pending Approvals', value: pendingCount.toString(), change: 'Requires action', icon: CheckCircle, color: 'text-purple-400', bgColor: 'bg-purple-500/20' },
        { label: 'Completed', value: (stats?.completed_count ?? 0).toString(), change: `${((stats?.success_rate ?? 0) * 100).toFixed(0)}% success`, icon: Mail, color: 'text-green-400', bgColor: 'bg-green-500/20' },
        { label: 'Total Workflows', value: (stats?.total_workflows ?? 0).toString(), change: 'All time', icon: Users, color: 'text-orange-400', bgColor: 'bg-orange-500/20' },
      ];

  const displayWorkflows = useMockData ? mockRecentWorkflows : workflows.slice(0, 4);

  const isLoading = statsLoading || workflowsLoading;

  return (
    <div className="min-h-screen">
      <Header
        title="Dashboard"
        description="Overview of your AI-powered sales engine"
      />

      <div className="p-6 space-y-6">
        {/* Demo Mode Banner */}
        {useMockData && (
          <div className="flex items-center gap-3 rounded-lg bg-yellow-500/10 border border-yellow-500/30 px-4 py-3">
            <AlertCircle className="h-5 w-5 text-yellow-400 shrink-0" />
            <div className="flex-1">
              <p className="text-sm font-medium text-yellow-400">Demo Mode</p>
              <p className="text-xs text-yellow-500/80">
                Backend API not available. Showing mock data for demonstration.
              </p>
            </div>
          </div>
        )}

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {isLoading && !useMockData ? (
            // Loading skeletons
            Array.from({ length: 4 }).map((_, i) => (
              <Card key={i}>
                <div className="flex items-start justify-between">
                  <div className="space-y-2">
                    <Skeleton className="h-4 w-24" />
                    <Skeleton className="h-8 w-16" />
                    <Skeleton className="h-3 w-20" />
                  </div>
                  <Skeleton className="h-9 w-9 rounded-lg" />
                </div>
              </Card>
            ))
          ) : (
            displayStats.map((stat) => {
              const Icon = stat.icon;
              return (
                <Card key={stat.label} hover>
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="text-sm font-medium text-[#9D9D9D]">{stat.label}</p>
                      <p className="mt-1 text-2xl font-semibold text-white">{stat.value}</p>
                      <p className="mt-1 text-xs text-[#9D9D9D]">{stat.change}</p>
                    </div>
                    <div className={`p-2 rounded-lg ${stat.bgColor}`}>
                      <Icon className={`h-5 w-5 ${stat.color}`} />
                    </div>
                  </div>
                </Card>
              );
            })
          )}
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Recent Workflows */}
          <Card className="lg:col-span-2" padding="none">
            <div className="p-4 border-b border-[#414141]">
              <CardHeader
                title="Recent Workflows"
                description="Latest AI agent activities"
              />
            </div>
            {isLoading && !useMockData ? (
              <div className="p-4 space-y-3">
                {[1, 2, 3, 4].map((i) => (
                  <div key={i} className="flex items-center gap-3">
                    <Skeleton className="h-9 w-9 rounded-lg" />
                    <div className="flex-1 space-y-1">
                      <Skeleton className="h-4 w-32" />
                      <Skeleton className="h-3 w-48" />
                    </div>
                    <Skeleton className="h-5 w-16 rounded-full" />
                  </div>
                ))}
              </div>
            ) : (
              <>
                <div className="divide-y divide-[#414141]">
                  {displayWorkflows.map((workflow) => (
                    <div
                      key={workflow.id}
                      className="flex items-center justify-between px-4 py-3 hover:bg-[#414141]/30 transition-colors cursor-pointer"
                    >
                      <div className="flex items-center gap-3">
                        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[#414141]/50">
                          <Workflow className="h-4 w-4 text-[#D6D6D6]" />
                        </div>
                        <div>
                          <p className="text-sm font-medium text-white">
                            {getWorkflowTypeLabel(workflow.workflow_type)}
                          </p>
                          <p className="text-xs text-[#9D9D9D]">
                            {Object.values(workflow.input_data)[0] as string}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <span
                          className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${statusColors[workflow.status]}`}
                        >
                          {workflow.status === 'running' && (
                            <span className="mr-1 h-1.5 w-1.5 rounded-full bg-white animate-pulse" />
                          )}
                          {workflow.status.replace('_', ' ')}
                        </span>
                        <span className="text-xs text-[#9D9D9D]">
                          {formatRelativeTime(workflow.created_at)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="p-4 border-t border-[#414141]">
                  <Link
                    href="/workflows"
                    className="text-sm font-medium text-[#00C0F0] hover:text-[#00C0F0]/80"
                  >
                    View all workflows →
                  </Link>
                </div>
              </>
            )}
          </Card>

          {/* Quick Actions */}
          <Card padding="none">
            <div className="p-4 border-b border-[#414141]">
              <CardHeader
                title="Quick Actions"
                description="Start a new workflow"
              />
            </div>
            <div className="p-4 space-y-2">
              <button
                className="w-full flex items-center gap-3 rounded-lg border border-[#414141] p-3 text-left hover:bg-[#414141]/30 transition-colors"
                onClick={() => info('Coming Soon', 'Lead Research workflow will be available soon')}
              >
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[#00C0F0]/20">
                  <Users className="h-4 w-4 text-[#00C0F0]" />
                </div>
                <div>
                  <p className="text-sm font-medium text-white">Research Lead</p>
                  <p className="text-xs text-[#9D9D9D]">Analyze a new prospect</p>
                </div>
              </button>
              <Link href="/email-copilot">
                <button className="w-full flex items-center gap-3 rounded-lg border border-[#414141] p-3 text-left hover:bg-[#414141]/30 transition-colors">
                  <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-green-500/20">
                    <Mail className="h-4 w-4 text-green-400" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-white">Generate Email</p>
                    <p className="text-xs text-[#9D9D9D]">AI-powered outreach</p>
                  </div>
                </button>
              </Link>
              <button
                className="w-full flex items-center gap-3 rounded-lg border border-[#414141] p-3 text-left hover:bg-[#414141]/30 transition-colors"
                onClick={() => info('Coming Soon', 'Meeting Analysis workflow will be available soon')}
              >
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-purple-500/20">
                  <FileText className="h-4 w-4 text-purple-400" />
                </div>
                <div>
                  <p className="text-sm font-medium text-white">Analyze Meeting</p>
                  <p className="text-xs text-[#9D9D9D]">Extract action items</p>
                </div>
              </button>
            </div>
          </Card>
        </div>

        {/* Performance Chart Placeholder */}
        <Card>
          <CardHeader
            title="Weekly Performance"
            description="Workflow completion and success rates"
          />
          <div className="flex items-center justify-center h-64 bg-[#414141]/20 rounded-lg border-2 border-dashed border-[#414141]">
            <div className="text-center">
              <TrendingUp className="mx-auto h-10 w-10 text-[#9D9D9D]" />
              <p className="mt-2 text-sm text-[#9D9D9D]">Performance chart coming soon</p>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

'use client';

import { useState, useEffect, useCallback } from 'react';
import { workflowApi, ApiError } from '@/services/api';
import type {
  Workflow,
  WorkflowStatus,
  WorkflowType,
  WorkflowListResponse,
  WorkflowStats,
  WorkflowTriggerRequest,
} from '@/types/workflow';

interface UseWorkflowsOptions {
  status?: WorkflowStatus;
  workflowType?: WorkflowType;
  page?: number;
  pageSize?: number;
  autoFetch?: boolean;
}

interface UseWorkflowsReturn {
  workflows: Workflow[];
  total: number;
  page: number;
  hasMore: boolean;
  isLoading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  loadMore: () => Promise<void>;
}

/**
 * Hook to fetch and manage workflows list
 */
export function useWorkflows(options: UseWorkflowsOptions = {}): UseWorkflowsReturn {
  const {
    status,
    workflowType,
    page: initialPage = 1,
    pageSize = 20,
    autoFetch = true,
  } = options;

  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(initialPage);
  const [hasMore, setHasMore] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchWorkflows = useCallback(async (pageNum: number = 1, append: boolean = false) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await workflowApi.list({
        status,
        workflow_type: workflowType,
        page: pageNum,
        page_size: pageSize,
      });

      if (append) {
        setWorkflows((prev) => [...prev, ...response.workflows]);
      } else {
        setWorkflows(response.workflows);
      }
      setTotal(response.total);
      setPage(response.page);
      setHasMore(response.has_more);
    } catch (err) {
      const message = err instanceof ApiError
        ? err.message
        : 'Failed to fetch workflows';
      setError(message);
      console.error('[useWorkflows] Error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [status, workflowType, pageSize]);

  const refetch = useCallback(async () => {
    await fetchWorkflows(1, false);
  }, [fetchWorkflows]);

  const loadMore = useCallback(async () => {
    if (!hasMore || isLoading) return;
    await fetchWorkflows(page + 1, true);
  }, [hasMore, isLoading, page, fetchWorkflows]);

  useEffect(() => {
    if (autoFetch) {
      fetchWorkflows(1, false);
    }
  }, [autoFetch, fetchWorkflows]);

  return {
    workflows,
    total,
    page,
    hasMore,
    isLoading,
    error,
    refetch,
    loadMore,
  };
}

/**
 * Hook to fetch workflow statistics
 */
export function useWorkflowStats() {
  const [stats, setStats] = useState<WorkflowStats | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await workflowApi.getStats();
      setStats(response);
    } catch (err) {
      const message = err instanceof ApiError
        ? err.message
        : 'Failed to fetch workflow stats';
      setError(message);
      console.error('[useWorkflowStats] Error:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  return { stats, isLoading, error, refetch: fetchStats };
}

/**
 * Hook to fetch a single workflow
 */
export function useWorkflow(workflowId: string | null) {
  const [workflow, setWorkflow] = useState<Workflow | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchWorkflow = useCallback(async () => {
    if (!workflowId) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await workflowApi.get(workflowId);
      setWorkflow(response);
    } catch (err) {
      const message = err instanceof ApiError
        ? err.message
        : 'Failed to fetch workflow';
      setError(message);
      console.error('[useWorkflow] Error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [workflowId]);

  useEffect(() => {
    fetchWorkflow();
  }, [fetchWorkflow]);

  return { workflow, isLoading, error, refetch: fetchWorkflow };
}

/**
 * Hook to trigger a new workflow
 */
export function useTriggerWorkflow() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const trigger = useCallback(async (request: WorkflowTriggerRequest) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await workflowApi.trigger(request);
      return response;
    } catch (err) {
      const message = err instanceof ApiError
        ? err.message
        : 'Failed to trigger workflow';
      setError(message);
      console.error('[useTriggerWorkflow] Error:', err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { trigger, isLoading, error };
}

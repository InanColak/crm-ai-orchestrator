'use client';

import { useState, useEffect, useCallback } from 'react';
import { approvalApi, ApiError } from '@/services/api';
import type {
  Approval,
  ApprovalStatus,
  ApprovalListResponse,
  ApprovalActionRequest,
} from '@/types/approval';

interface UseApprovalsOptions {
  status?: ApprovalStatus;
  page?: number;
  pageSize?: number;
  autoFetch?: boolean;
}

interface UseApprovalsReturn {
  approvals: Approval[];
  total: number;
  pendingCount: number;
  isLoading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

/**
 * Hook to fetch and manage approvals list
 */
export function useApprovals(options: UseApprovalsOptions = {}): UseApprovalsReturn {
  const {
    status,
    page = 1,
    pageSize = 50,
    autoFetch = true,
  } = options;

  const [approvals, setApprovals] = useState<Approval[]>([]);
  const [total, setTotal] = useState(0);
  const [pendingCount, setPendingCount] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchApprovals = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await approvalApi.list({
        status,
        page,
        page_size: pageSize,
      });

      setApprovals(response.approvals);
      setTotal(response.total);
      setPendingCount(response.pending_count);
    } catch (err) {
      const message = err instanceof ApiError
        ? err.message
        : 'Failed to fetch approvals';
      setError(message);
      console.error('[useApprovals] Error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [status, page, pageSize]);

  useEffect(() => {
    if (autoFetch) {
      fetchApprovals();
    }
  }, [autoFetch, fetchApprovals]);

  return {
    approvals,
    total,
    pendingCount,
    isLoading,
    error,
    refetch: fetchApprovals,
  };
}

/**
 * Hook to handle approval actions (approve/reject)
 */
export function useApprovalAction() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const approve = useCallback(async (
    approvalId: string,
    modifications?: Record<string, unknown>
  ) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await approvalApi.action({
        approval_id: approvalId,
        action: 'approve',
        modifications,
      });
      return response;
    } catch (err) {
      const message = err instanceof ApiError
        ? err.message
        : 'Failed to approve';
      setError(message);
      console.error('[useApprovalAction] Approve error:', err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const reject = useCallback(async (
    approvalId: string,
    reason?: string
  ) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await approvalApi.action({
        approval_id: approvalId,
        action: 'reject',
        reason,
      });
      return response;
    } catch (err) {
      const message = err instanceof ApiError
        ? err.message
        : 'Failed to reject';
      setError(message);
      console.error('[useApprovalAction] Reject error:', err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { approve, reject, isLoading, error };
}

/**
 * Hook to get a single approval
 */
export function useApproval(approvalId: string | null) {
  const [approval, setApproval] = useState<Approval | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchApproval = useCallback(async () => {
    if (!approvalId) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await approvalApi.get(approvalId);
      setApproval(response);
    } catch (err) {
      const message = err instanceof ApiError
        ? err.message
        : 'Failed to fetch approval';
      setError(message);
      console.error('[useApproval] Error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [approvalId]);

  useEffect(() => {
    fetchApproval();
  }, [fetchApproval]);

  return { approval, isLoading, error, refetch: fetchApproval };
}

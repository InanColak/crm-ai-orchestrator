'use client';

import { useEffect, useCallback, useRef } from 'react';
import { getSupabaseClient } from '@/lib/supabase';
import type { RealtimeChannel } from '@supabase/supabase-js';

type RealtimeEvent = 'INSERT' | 'UPDATE' | 'DELETE' | '*';

interface UseRealtimeOptions {
  table: string;
  event?: RealtimeEvent;
  filter?: string;
  enabled?: boolean;
}

/**
 * Hook for Supabase Realtime subscriptions
 */
export function useRealtime<T = unknown>(
  options: UseRealtimeOptions,
  onData: (payload: { new: T; old: T; eventType: RealtimeEvent }) => void
) {
  const { table, event = '*', filter, enabled = true } = options;
  const channelRef = useRef<RealtimeChannel | null>(null);

  const subscribe = useCallback(() => {
    if (!enabled) return;

    const supabase = getSupabaseClient();
    const channelName = `${table}-${event}-${filter || 'all'}`;

    // Unsubscribe from existing channel if any
    if (channelRef.current) {
      supabase.removeChannel(channelRef.current);
    }

    // Create new channel
    const channel = supabase
      .channel(channelName)
      .on(
        'postgres_changes' as never,
        {
          event,
          schema: 'public',
          table,
          filter,
        },
        (payload: { new: T; old: T; eventType: RealtimeEvent }) => {
          onData(payload);
        }
      )
      .subscribe((status) => {
        console.log(`[useRealtime] ${channelName} status:`, status);
      });

    channelRef.current = channel;

    return () => {
      supabase.removeChannel(channel);
    };
  }, [table, event, filter, enabled, onData]);

  useEffect(() => {
    const cleanup = subscribe();
    return () => {
      cleanup?.();
    };
  }, [subscribe]);

  return { channel: channelRef.current };
}

/**
 * Hook for workflow status realtime updates
 */
export function useWorkflowRealtime(
  onUpdate: (workflow: { id: string; status: string }) => void,
  enabled: boolean = true
) {
  return useRealtime<{ id: string; status: string }>(
    {
      table: 'workflows',
      event: 'UPDATE',
      enabled,
    },
    (payload) => {
      if (payload.new) {
        onUpdate(payload.new);
      }
    }
  );
}

/**
 * Hook for new approval realtime notifications
 */
export function useApprovalRealtime(
  onNewApproval: (approval: { id: string; title: string }) => void,
  enabled: boolean = true
) {
  return useRealtime<{ id: string; title: string; status: string }>(
    {
      table: 'pending_approvals',
      event: 'INSERT',
      enabled,
    },
    (payload) => {
      if (payload.new && payload.new.status === 'pending') {
        onNewApproval(payload.new);
      }
    }
  );
}

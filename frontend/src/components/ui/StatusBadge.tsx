import { cn } from '@/lib/utils';
import type { WorkflowStatus } from '@/types/workflow';

interface StatusBadgeProps {
  status: WorkflowStatus | string;
  size?: 'sm' | 'md';
  pulse?: boolean;
}

const statusStyles: Record<string, string> = {
  pending: 'bg-yellow-100 text-yellow-800 border-yellow-200',
  running: 'bg-blue-100 text-blue-800 border-blue-200',
  waiting_approval: 'bg-purple-100 text-purple-800 border-purple-200',
  completed: 'bg-green-100 text-green-800 border-green-200',
  failed: 'bg-red-100 text-red-800 border-red-200',
  cancelled: 'bg-gray-100 text-gray-800 border-gray-200',
};

const statusLabels: Record<string, string> = {
  pending: 'Pending',
  running: 'Running',
  waiting_approval: 'Awaiting Approval',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

export function StatusBadge({ status, size = 'md', pulse = false }: StatusBadgeProps) {
  const style = statusStyles[status] || 'bg-gray-100 text-gray-800';
  const label = statusLabels[status] || status;
  const isActive = status === 'running';

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border font-medium',
        style,
        size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-2.5 py-1 text-xs',
      )}
    >
      {(pulse || isActive) && (
        <span className="relative flex h-2 w-2">
          <span className={cn(
            'absolute inline-flex h-full w-full animate-ping rounded-full opacity-75',
            isActive ? 'bg-blue-400' : 'bg-current'
          )} />
          <span className={cn(
            'relative inline-flex h-2 w-2 rounded-full',
            isActive ? 'bg-blue-500' : 'bg-current'
          )} />
        </span>
      )}
      {label}
    </span>
  );
}
